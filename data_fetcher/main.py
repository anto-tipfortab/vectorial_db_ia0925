import os
import json
import logging
import time
from pathlib import Path

import arxiv
import pika
from apscheduler.schedulers.blocking import BlockingScheduler
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [fetcher] %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
RABBITMQ_HOST    = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT    = int(os.getenv("RABBITMQ_PORT", 5672))
RABBITMQ_USER    = os.getenv("RABBITMQ_USER", "admin")
RABBITMQ_PASS    = os.getenv("RABBITMQ_PASS", "changeme123")
RABBITMQ_QUEUE   = os.getenv("RABBITMQ_QUEUE", "papers")
CATEGORIES       = os.getenv("ARXIV_CATEGORIES", "cs.AI,cs.LG,cs.CL").split(",")
PAPERS_PER_RUN   = int(os.getenv("PAPERS_PER_RUN", 50))
INTERVAL_MINUTES = int(os.getenv("INTERVAL_MINUTES", 30))
PDF_OUTPUT_DIR   = Path(os.getenv("PDF_OUTPUT_DIR", "/shared/pdfs"))
SEEN_IDS_FILE    = Path(os.getenv("SEEN_IDS_FILE", "/shared/seen_ids.json"))


# ── Persistence helpers ────────────────────────────────────────────────────────

def load_seen_ids() -> set:
    """Load already-processed arXiv IDs from disk (survives container restarts)."""
    if SEEN_IDS_FILE.exists():
        with open(SEEN_IDS_FILE) as f:
            return set(json.load(f))
    return set()


def save_seen_ids(seen: set) -> None:
    SEEN_IDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SEEN_IDS_FILE, "w") as f:
        json.dump(list(seen), f)


# ── RabbitMQ ───────────────────────────────────────────────────────────────────

def get_rabbitmq_channel():
    """Open a fresh connection + channel and declare the queue."""
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
    params = pika.ConnectionParameters(
        host=RABBITMQ_HOST,
        port=RABBITMQ_PORT,
        credentials=credentials,
        heartbeat=600,
        blocked_connection_timeout=300,
    )
    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    channel.queue_declare(queue=RABBITMQ_QUEUE, durable=True)
    return connection, channel


# ── PDF download ───────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=4, max=10))
def download_pdf(paper: arxiv.Result, output_dir: Path) -> Path:
    """Download PDF to shared volume. Skips if already present. Retries 3x."""
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_id  = paper.get_short_id().replace("/", "_")
    pdf_path = output_dir / f"{safe_id}.pdf"

    if pdf_path.exists():
        log.info("Already on disk, skipping: %s", safe_id)
        return pdf_path

    paper.download_pdf(dirpath=str(output_dir), filename=pdf_path.name)
    log.info("Downloaded: %s", safe_id)
    return pdf_path


# ── Main job ───────────────────────────────────────────────────────────────────

def fetch_and_publish() -> None:
    log.info("── Fetch job started ──────────────────────────────────────────")
    seen_ids  = load_seen_ids()
    new_count = 0

    # e.g. "cat:cs.AI OR cat:cs.LG OR cat:cs.CL"
    query = " OR ".join(f"cat:{c}" for c in CATEGORIES)
    log.info("Query: %s  (max %d)", query, PAPERS_PER_RUN)

    search = arxiv.Search(
        query=query,
        max_results=PAPERS_PER_RUN,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    try:
        connection, channel = get_rabbitmq_channel()
    except Exception as e:
        log.error("RabbitMQ unavailable: %s — aborting this run", e)
        return

    client = arxiv.Client(delay_seconds=3)  # polite to arXiv servers

    for paper in client.results(search):
        arxiv_id = paper.get_short_id()

        if arxiv_id in seen_ids:
            log.debug("Skip (seen): %s", arxiv_id)
            continue

        try:
            pdf_path = download_pdf(paper, PDF_OUTPUT_DIR)
        except Exception as e:
            log.error("Download failed %s: %s", arxiv_id, e)
            continue

        message = {
            "arxiv_id":     arxiv_id,
            "title":        paper.title,
            "abstract":     paper.summary,
            "authors":      [a.name for a in paper.authors],
            "categories":   paper.categories,
            "published_at": paper.published.isoformat(),
            "pdf_url":      paper.pdf_url,
            "pdf_path":     str(pdf_path),
        }

        try:
            channel.basic_publish(
                exchange="",
                routing_key=RABBITMQ_QUEUE,
                body=json.dumps(message),
                properties=pika.BasicProperties(
                    delivery_mode=2,          # message survives RabbitMQ restart
                    content_type="application/json",
                ),
            )
            seen_ids.add(arxiv_id)
            new_count += 1
            log.info("Published [%d]: %s — %.60s", new_count, arxiv_id, paper.title)
        except Exception as e:
            log.error("Publish failed %s: %s", arxiv_id, e)

        time.sleep(0.1)

    connection.close()
    save_seen_ids(seen_ids)
    log.info("── Done: %d new papers published ─────────────────────────────", new_count)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Fetcher starting — categories=%s interval=%dmin", CATEGORIES, INTERVAL_MINUTES)

    # Run immediately on startup, then on schedule
    fetch_and_publish()

    scheduler = BlockingScheduler()
    scheduler.add_job(
        fetch_and_publish,
        trigger="interval",
        minutes=INTERVAL_MINUTES,
        id="fetch_job",
    )
    log.info("Scheduler active — next run in %d minutes", INTERVAL_MINUTES)
    scheduler.start()