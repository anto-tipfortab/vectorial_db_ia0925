# Sistema RAG — arXiv Papers

Sistema de Retrieval-Augmented Generation sobre papers científicos de arXiv (`cs.AI`, `cs.LG`, `cs.CL`). Ingesta continua de PDFs, embeddings con Gemini, búsqueda vectorial con Qdrant y chat en español.

---

## Requisitos

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (con Docker Compose v2)
- Una clave API de Google Gemini → [aistudio.google.com](https://aistudio.google.com)

---

## Configuración inicial

**1. Clona el repositorio y entra en el directorio:**

```bash
git clone <repo-url>
cd rag-ingestion
```

**2. Crea el fichero `.env` raíz** con tus credenciales:

```bash
cp .env.example .env   # si existe, o créalo manualmente
```

Contenido mínimo del `.env`:

```env
GOOGLE_API_KEY=tu_clave_de_gemini_aqui
RABBITMQ_USER=admin
RABBITMQ_PASS=changeme123
```

---

## Arrancar el sistema

```bash
./docker_run.sh
```

Esto ejecuta `docker compose down` + `docker system prune` + `docker compose up --build` para asegurar un arranque limpio.

Si prefieres arrancar sin limpiar el sistema completo:

```bash
docker compose up --build
```

---

## Servicios y puertos

| Servicio | URL | Descripción |
|---|---|---|
| **Streamlit** (chat UI) | http://localhost:8501 | Interfaz de chat en español |
| **FastAPI** (REST API) | http://localhost:8000 | Endpoint de consultas |
| **FastAPI Docs** | http://localhost:8000/docs | Swagger UI interactivo |
| **Qdrant Dashboard** | http://localhost:6333/dashboard | Base de datos vectorial |
| **RabbitMQ Dashboard** | http://localhost:15672 | Cola de mensajes (admin/changeme123) |

---

## Uso

### Chat web (recomendado)

Abre **http://localhost:8501** en el navegador.

- El sidebar muestra el estado del sistema y el número de vectores indexados
- Escribe tu pregunta en el chat (en español o inglés)
- Cada respuesta incluye las fuentes (arxiv_id, título, similitud)
- El historial se mantiene durante la sesión

### API REST

```bash
# Health check
curl http://localhost:8000/health

# Consulta
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Cuáles son los últimos avances en modelos de razonamiento?", "top_k": 5}'
```

Respuesta:
```json
{
  "answer": "...",
  "sources": [
    {"arxiv_id": "2603.12246v1", "title": "...", "chunk_index": 68, "score": 0.7467}
  ],
  "model": "gemini-2.5-flash"
}
```

---

## Arquitectura

```
arXiv API
    │
    ▼
data_fetcher ──(RabbitMQ)──▶ processor_consumer
    │                                │
    │ PDFs                    Celery task
    ▼                                ▼
/shared/pdfs             processor_worker
                          │  · pdfplumber
                          │  · chunking
                          │  · Gemini embedding
                          ▼
                        Qdrant ◀──── query_service
                                      │
                                  FastAPI + Streamlit
                                      │
                                    Usuario
```

### Servicios Docker

| Contenedor | Rol |
|---|---|
| `rabbitmq` | Cola de mensajes entre fetcher y processor |
| `qdrant` | Base de datos vectorial (HNSW, 3072 dims) |
| `data_fetcher` | Consulta arXiv cada 5 min, descarga PDFs |
| `processor_consumer` | Lee la cola RabbitMQ, despacha tareas Celery |
| `processor_worker` | Embeds + upsert a Qdrant (Celery worker) |
| `query_service` | FastAPI + Streamlit (RAG query layer) |

---

## Configuración avanzada

### `data_fetcher/.env`

```env
INTERVAL_MINUTES=5       # frecuencia de consulta a arXiv
PAPERS_PER_RUN=50        # máx. papers por ciclo
```

### `processor/.env`

```env
EMBED_BATCH_DELAY=3.0    # segundos entre lotes (free tier)
                         # reducir a 0.0 con tier de pago
CHUNK_SIZE=800
CHUNK_OVERLAP=100
```

### `query_service/.env`

```env
GENERATION_MODEL=gemini-2.5-flash   # o gemini-2.5-pro para mejor calidad
TOP_K=5                             # chunks recuperados por consulta
```

---

## Parar el sistema

```bash
# Parar sin borrar datos
docker compose down

# Parar y borrar todos los volúmenes (resetea Qdrant y RabbitMQ)
docker compose down -v
```