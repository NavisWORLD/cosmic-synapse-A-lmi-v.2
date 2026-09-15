#!/usr/bin/env bash
set -euo pipefail

export A_LMI_MINIO_ACCESS_KEY="ciuser${RANDOM}"
export A_LMI_MINIO_SECRET_KEY="$(openssl rand -hex 24)"
export MILVUS_MINIO_ACCESS_KEY="cimilvus${RANDOM}"
export MILVUS_MINIO_SECRET_KEY="$(openssl rand -hex 24)"
export A_LMI_NEO4J_USERNAME="neo4j"
export A_LMI_NEO4J_PASSWORD="$(openssl rand -hex 24)"
marker="${GITHUB_RUN_ID:-local}"
compose=(docker compose -f infrastructure/docker-compose.yml)

cleanup() {
  "${compose[@]}" down -v --remove-orphans >/dev/null 2>&1 || true
}
trap cleanup EXIT
cleanup

"${compose[@]}" up -d etcd minio-storage milvus

wait_ready() {
  for _ in $(seq 1 120); do
    if curl -fsS http://127.0.0.1:9091/healthz >/dev/null 2>&1; then
      if python - <<'PY'
from pymilvus import connections, utility
connections.connect(alias="default", host="127.0.0.1", port="19530")
utility.list_collections()
connections.disconnect("default")
PY
      then
        return 0
      fi
    fi
    sleep 2
  done
  "${compose[@]}" logs milvus etcd minio-storage
  return 1
}

wait_ready
python integration/vm/live_services.py milvus-write --marker "$marker"

"${compose[@]}" restart milvus
wait_ready
python integration/vm/live_services.py milvus-read --marker "$marker"
