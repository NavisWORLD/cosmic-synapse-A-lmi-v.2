#!/usr/bin/env bash
set -euo pipefail

export A_LMI_MINIO_ACCESS_KEY="ciuser${RANDOM}"
export A_LMI_MINIO_SECRET_KEY="$(openssl rand -hex 24)"
export MILVUS_MINIO_ACCESS_KEY="cimilvus${RANDOM}"
export MILVUS_MINIO_SECRET_KEY="$(openssl rand -hex 24)"
export A_LMI_NEO4J_USERNAME="neo4j"
export A_LMI_NEO4J_PASSWORD="$(openssl rand -hex 24)"

topic="cosmic-vm-${GITHUB_RUN_ID:-local}"
compose=(docker compose -f infrastructure/docker-compose.yml)

cleanup() {
  "${compose[@]}" down -v --remove-orphans >/dev/null 2>&1 || true
}
trap cleanup EXIT
cleanup

"${compose[@]}" up -d zookeeper kafka

for _ in $(seq 1 90); do
  if nc -z 127.0.0.1 9092; then
    if python - <<'PY'
from kafka.admin import KafkaAdminClient
admin = KafkaAdminClient(bootstrap_servers="127.0.0.1:9092", request_timeout_ms=3000)
admin.list_topics()
admin.close()
PY
    then
      break
    fi
  fi
  sleep 2
done

python integration/vm/live_services.py kafka-write --topic "$topic"

"${compose[@]}" restart kafka
for _ in $(seq 1 90); do
  if nc -z 127.0.0.1 9092; then
    if python - <<'PY'
from kafka.admin import KafkaAdminClient
admin = KafkaAdminClient(bootstrap_servers="127.0.0.1:9092", request_timeout_ms=3000)
admin.list_topics()
admin.close()
PY
    then
      break
    fi
  fi
  sleep 2
done

python integration/vm/live_services.py kafka-read --topic "$topic"
