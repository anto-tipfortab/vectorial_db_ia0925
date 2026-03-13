#!/bin/bash
set -e

echo "🛑 Stopping containers..."
docker compose down

echo "🧹 Pruning system (images, containers, networks, volumes)..."
docker system prune -af --volumes

echo "🚀 Building and starting all services..."
docker compose up --build