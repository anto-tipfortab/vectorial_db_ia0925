"""
Celery task: PDF → chunks → Gemini embeddings → Qdrant upsert → move PDF.
"""
import logging
import os
import shutil
import time
import uuid
import hashlib
from pathlib import Path

import pdfplumber
import google.genai as genai
from google.genai import types as genai_types
from celery_app import app
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO, format="%(asctime)s [worker] %(message)s")
log = logging.getLogger(__name__)

# ── Config from environment ───────────────────────────────────────────────────
GOOGLE_API_KEY   = os.environ["GOOGLE_API_KEY"]
QDRANT_HOST      = os.environ.get("QDRANT_HOST", "qdrant")
QDRANT_PORT      = int(os.environ.get("QDRANT_PORT", 6333))
COLLECTION_NAME  = os.environ.get("QDRANT_COLLECTION", "arxiv_papers")
PROCESSED_DIR    = os.environ.get("PROCESSED_DIR", "/shared/processed")
CHUNK_SIZE       = int(os.environ.get("CHUNK_SIZE", 800))
CHUNK_OVERLAP    = int(os.environ.get("CHUNK_OVERLAP", 100))
EMBEDDING_MODEL  = os.environ.get("EMBEDDING_MODEL", "models/text-embedding-004")
EMBEDDING_DIM    = 3072  # gemini-embedding-001 default output dimension
EMBED_BATCH_DELAY = float(os.environ.get("EMBED_BATCH_DELAY", 2.0))  # seconds between batches

client_genai = genai.Client(
    api_key=GOOGLE_API_KEY,
    http_options={"api_version": "v1beta"},
)

qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def ensure_collection():
    try:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        log.info("Created Qdrant collection '%s'", COLLECTION_NAME)
    except Exception:
        pass  # already exists — that's fine


def extract_text(pdf_path: str) -> str:
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
    return "\n\n".join(text_parts)


@retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=2, min=60, max=120))
def embed_texts(texts: list[str]) -> list[list[float]]:
    try:
        result = client_genai.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
            config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        return [e.values for e in result.embeddings]
    except Exception as exc:
        msg = str(exc)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            # try to extract the retryDelay the API suggests
            import re
            match = re.search(r"retryDelay.*?(\d+)s", msg)
            wait_s = int(match.group(1)) + 2 if match else 60
            log.warning("Rate limited — sleeping %ds before retry", wait_s)
            time.sleep(wait_s)
        else:
            log.error("Gemini embed_content failed (model=%s): %s", EMBEDDING_MODEL, exc)
        raise


def move_to_processed(pdf_path: str):
    Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
    dest = Path(PROCESSED_DIR) / Path(pdf_path).name
    shutil.move(pdf_path, dest)
    log.info("Moved PDF to %s", dest)


@app.task(name="tasks.process_paper", bind=True, max_retries=3, default_retry_delay=10)
def process_paper(self, message: dict):
    arxiv_id = message["arxiv_id"]
    pdf_path  = message["pdf_path"]
    log.info("Processing paper %s", arxiv_id)

    try:
        ensure_collection()

        # 1. Extract text from PDF
        raw_text = extract_text(pdf_path)
        if not raw_text.strip():
            log.warning("No text extracted from %s — skipping", arxiv_id)
            return

        # 2. Chunk
        chunks = splitter.split_text(raw_text)
        log.info("Paper %s → %d chunks", arxiv_id, len(chunks))

        # 3. Embed in batches of 20, with a delay to respect free tier (100 req/min)
        batch_size = 20
        all_embeddings = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            embeddings = embed_texts(batch)
            all_embeddings.extend(embeddings)
            if i + batch_size < len(chunks):
                time.sleep(EMBED_BATCH_DELAY)

        # 4. Build Qdrant points
        points = [
            PointStruct(
                id=str(uuid.UUID(hashlib.md5(f"{arxiv_id}_{idx}".encode()).hexdigest())),
                vector=embedding,
                payload={
                    "arxiv_id":    arxiv_id,
                    "title":       message.get("title", ""),
                    "authors":     message.get("authors", []),
                    "categories":  message.get("categories", []),
                    "published_at": message.get("published_at", ""),
                    "pdf_url":     message.get("pdf_url", ""),
                    "pdf_path":    pdf_path,
                    "chunk_index": idx,
                    "chunk_text":  chunk,
                    "status":      "active",
                },
            )
            for idx, (chunk, embedding) in enumerate(zip(chunks, all_embeddings))
        ]

        # 5. Upsert into Qdrant
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        log.info("Upserted %d points for paper %s", len(points), arxiv_id)

        # 6. Move PDF to processed folder
        move_to_processed(pdf_path)

    except Exception as exc:
        log.error("Error processing %s: %s", arxiv_id, exc)
        raise self.retry(exc=exc)