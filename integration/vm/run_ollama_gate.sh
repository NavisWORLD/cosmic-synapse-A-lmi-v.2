#!/usr/bin/env bash
set -euo pipefail

container="cosmic-ollama-ci"
volume="cosmic-ollama-ci"
model="${OLLAMA_VM_MODEL:-qwen2.5:0.5b}"

cleanup() {
  docker rm -f "$container" >/dev/null 2>&1 || true
}
trap cleanup EXIT
cleanup

docker volume create "$volume" >/dev/null
docker run -d --name "$container" \
  -p 11434:11434 \
  -v "$volume:/root/.ollama" \
  ollama/ollama >/dev/null

for _ in $(seq 1 90); do
  if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
curl -fsS http://127.0.0.1:11434/api/tags >/dev/null

docker exec "$container" ollama pull "$model"
python integration/vm/live_services.py ollama-live --model "$model"

docker exec "$container" ollama ps
