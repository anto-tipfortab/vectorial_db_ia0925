"""
query_service/streamlit_app.py

Interfaz de chat en español con:
  - Historial multi-turno
  - Sidebar con estadísticas de la colección
  - Lista de fuentes por cada respuesta
"""
import os
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
TOP_K   = int(os.environ.get("TOP_K", 5))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="arXiv RAG",
    page_icon="🔬",
    layout="wide",
)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []   # list of {role, content, sources?}


# ── Helpers ───────────────────────────────────────────────────────────────────
def fetch_health() -> dict:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.json()
    except Exception:
        return {}


def query_api(question: str) -> dict:
    r = requests.post(
        f"{API_URL}/query",
        json={"question": question, "top_k": TOP_K},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 arXiv RAG")
    st.caption("Sistema de recuperación sobre papers de cs.AI · cs.LG · cs.CL")
    st.divider()

    st.subheader("📊 Estado del sistema")
    health = fetch_health()
    if health:
        status = health.get("status", "?")
        color  = "🟢" if status == "ok" else "🔴"
        st.markdown(f"{color} **Estado:** {status}")
        st.markdown(f"🗄️ **Vectores indexados:** {health.get('vectors', '?'):,}")
        st.markdown(f"🧠 **Modelo embedding:** `{health.get('embedding_model', '?')}`")
        st.markdown(f"💬 **Modelo generación:** `{health.get('generation_model', '?')}`")
        st.markdown(f"📦 **Colección:** `{health.get('collection', '?')}`")
    else:
        st.error("No se puede conectar con la API")

    st.divider()

    st.subheader("⚙️ Configuración")
    top_k = st.slider("Chunks a recuperar (top-k)", min_value=1, max_value=10, value=TOP_K)

    st.divider()

    if st.button("🗑️ Limpiar conversación", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Datos actualizados cada 5 min desde arXiv")


# ── Main chat area ────────────────────────────────────────────────────────────
st.header("💬 Chat con papers de arXiv")

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📄 Fuentes ({len(msg['sources'])} papers)"):
                for src in msg["sources"]:
                    st.markdown(
                        f"- **{src['title']}** `{src['arxiv_id']}` "
                        f"— chunk {src['chunk_index']} "
                        f"· similitud: `{src['score']}`"
                    )

# Chat input
if prompt := st.chat_input("Pregunta sobre los últimos papers de IA..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call API and show response
    with st.chat_message("assistant"):
        with st.spinner("Buscando en Qdrant y generando respuesta..."):
            try:
                data = query_api(prompt)
                answer  = data["answer"]
                sources = data["sources"]
                model   = data["model"]

                st.markdown(answer)

                with st.expander(f"📄 Fuentes ({len(sources)} papers)"):
                    for src in sources:
                        st.markdown(
                            f"- **{src['title']}** `{src['arxiv_id']}` "
                            f"— chunk {src['chunk_index']} "
                            f"· similitud: `{src['score']}`"
                        )
                st.caption(f"Generado con `{model}` · top-k={top_k}")

                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except requests.HTTPError as e:
                detail = e.response.json().get("detail", str(e))
                st.error(f"Error de la API: {detail}")
            except Exception as e:
                st.error(f"Error inesperado: {e}")