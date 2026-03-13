"""
query_service/main.py

FastAPI RAG query endpoint:
  POST /query  → embed question → search Qdrant → generate with Gemini → return answer + sources
  GET  /health → connectivity check
"""
import logging
import os
import time
import re

import google.genai as genai
from google.genai import types as genai_types
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

logging.basicConfig(level=logging.INFO, format="%(asctime)s [query] %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
GOOGLE_API_KEY    = os.environ["GOOGLE_API_KEY"]
QDRANT_HOST       = os.environ.get("QDRANT_HOST", "qdrant")
QDRANT_PORT       = int(os.environ.get("QDRANT_PORT", 6333))
COLLECTION_NAME   = os.environ.get("QDRANT_COLLECTION", "arxiv_papers")
EMBEDDING_MODEL   = os.environ.get("EMBEDDING_MODEL", "models/gemini-embedding-001")
GENERATION_MODEL  = os.environ.get("GENERATION_MODEL", "gemini-2.0-flash")
DEFAULT_TOP_K     = int(os.environ.get("TOP_K", 5))

# ── Clients ───────────────────────────────────────────────────────────────────
gemini = genai.Client(
    api_key=GOOGLE_API_KEY,
    http_options={"api_version": "v1beta"},
)
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# ── Prompt template ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
Eres un asistente experto en investigación en inteligencia artificial. \
Responde ÚNICAMENTE basándote en los fragmentos de papers científicos \
proporcionados como contexto. Si la información no es suficiente para \
responder con seguridad, indícalo explícitamente en lugar de inventar.

Para cada afirmación importante, cita el paper de origen usando el \
formato [Autor et al., año] o el arxiv_id cuando el autor no esté disponible.
"""

def build_prompt(question: str, chunks: list[dict]) -> str:
    context_parts = []
    for i, c in enumerate(chunks, 1):
        payload = c.payload
        header = f"[{i}] {payload.get('title', 'Sin título')} ({payload.get('arxiv_id', '?')})"
        text   = payload.get("chunk_text", "")
        context_parts.append(f"{header}\n{text}")

    context = "\n\n---\n\n".join(context_parts)
    return f"{SYSTEM_PROMPT}\n\nContexto:\n{context}\n\nPregunta: {question}"


# ── Helpers ───────────────────────────────────────────────────────────────────
def embed_query(question: str) -> list[float]:
    """Embed the user question with RETRIEVAL_QUERY task type."""
    for attempt in range(4):
        try:
            result = gemini.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=[question],
                config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
            )
            return result.embeddings[0].values
        except Exception as exc:
            msg = str(exc)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                match = re.search(r"retryDelay.*?(\d+)s", msg)
                wait_s = int(match.group(1)) + 2 if match else 30
                log.warning("Rate limited embedding query — sleeping %ds", wait_s)
                time.sleep(wait_s)
            else:
                raise HTTPException(status_code=502, detail=f"Embedding failed: {exc}")
    raise HTTPException(status_code=502, detail="Embedding failed after retries")


def search_qdrant(vector: list[float], top_k: int) -> list:
    """Search Qdrant for the top_k most similar active chunks."""
    return qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        query_filter=Filter(
            must=[FieldCondition(key="status", match=MatchValue(value="active"))]
        ),
        limit=top_k,
        with_payload=True,
    )


def generate_answer(prompt: str) -> str:
    """Call Gemini generation model with the built prompt."""
    response = gemini.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
    )
    return response.text


# ── Schemas ───────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int    = Field(default=DEFAULT_TOP_K, ge=1, le=20)

class SourceItem(BaseModel):
    arxiv_id:    str
    title:       str
    chunk_index: int
    score:       float

class QueryResponse(BaseModel):
    answer:  str
    sources: list[SourceItem]
    model:   str


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="arXiv RAG Query Service",
    description="Retrieval-Augmented Generation over arXiv papers (cs.AI, cs.LG, cs.CL)",
    version="1.0.0",
)


@app.get("/health")
def health():
    try:
        info = qdrant.get_collection(COLLECTION_NAME)
        vector_count = info.points_count
        status = "ok"
    except Exception as exc:
        vector_count = -1
        status = f"qdrant_error: {exc}"

    return {
        "status": status,
        "qdrant": f"{QDRANT_HOST}:{QDRANT_PORT}",
        "collection": COLLECTION_NAME,
        "vectors": vector_count,
        "embedding_model": EMBEDDING_MODEL,
        "generation_model": GENERATION_MODEL,
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    log.info("Query: %s (top_k=%d)", req.question[:80], req.top_k)

    # 1. Embed the question
    query_vector = embed_query(req.question)

    # 2. Retrieve top-k chunks from Qdrant
    hits = search_qdrant(query_vector, req.top_k)
    if not hits:
        raise HTTPException(
            status_code=404,
            detail="No relevant documents found. The collection may still be ingesting."
        )

    # 3. Build prompt with retrieved context
    prompt = build_prompt(req.question, hits)

    # 4. Generate answer
    try:
        answer = generate_answer(prompt)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Generation failed: {exc}")

    # 5. Build source list — deduplicate by (arxiv_id, chunk_index)
    seen = set()
    sources = []
    for hit in hits:
        key = (hit.payload.get("arxiv_id", ""), hit.payload.get("chunk_index", 0))
        if key not in seen:
            seen.add(key)
            sources.append(SourceItem(
                arxiv_id    = hit.payload.get("arxiv_id", ""),
                title       = hit.payload.get("title", ""),
                chunk_index = hit.payload.get("chunk_index", 0),
                score       = round(hit.score, 4),
            ))


    log.info("Answered with %d sources via %s", len(sources), GENERATION_MODEL)
    return QueryResponse(answer=answer, sources=sources, model=GENERATION_MODEL)