"""
Pika consumer: reads JSON messages from the 'papers' RabbitMQ queue
and dispatches a Celery task for each one.
Keeps the data_fetcher message format untouched.
"""
import json
import logging
import os
import time

import pika
from tasks import process_paper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [consumer] %(message)s")
log = logging.getLogger(__name__)

RABBITMQ_USER = os.environ["RABBITMQ_USER"]
RABBITMQ_PASS = os.environ["RABBITMQ_PASS"]
RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_PORT = int(os.environ.get("RABBITMQ_PORT", 5672))
RABBITMQ_QUEUE = os.environ.get("RABBITMQ_QUEUE", "papers")


def get_connection():
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
    params = pika.ConnectionParameters(
        host=RABBITMQ_HOST,
        port=RABBITMQ_PORT,
        credentials=credentials,
        heartbeat=60,
        blocked_connection_timeout=300,
    )
    return pika.BlockingConnection(params)


def on_message(channel, method, _properties, body):
    try:
        message = json.loads(body)
        arxiv_id = message.get("arxiv_id", "unknown")
        log.info("Received paper %s — dispatching Celery task", arxiv_id)
        process_paper.delay(message)          # hand off to Celery worker
        channel.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as exc:
        log.error("Failed to dispatch task: %s", exc)
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)


def main():
    while True:
        try:
            log.info("Connecting to RabbitMQ...")
            connection = get_connection()
            channel = connection.channel()
            channel.queue_declare(queue=RABBITMQ_QUEUE, durable=True)
            channel.basic_qos(prefetch_count=1)
            channel.basic_consume(queue=RABBITMQ_QUEUE, on_message_callback=on_message)
            log.info("Waiting for messages on queue '%s'", RABBITMQ_QUEUE)
            channel.start_consuming()
        except pika.exceptions.AMQPConnectionError as exc:
            log.warning("RabbitMQ connection lost (%s) — retrying in 5s", exc)
            time.sleep(5)
        except KeyboardInterrupt:
            log.info("Shutting down consumer")
            break


if __name__ == "__main__":
    main()