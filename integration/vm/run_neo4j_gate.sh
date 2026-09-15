#!/usr/bin/env bash
set -euo pipefail

export A_LMI_NEO4J_USERNAME="neo4j"
export A_LMI_NEO4J_PASSWORD="$(openssl rand -hex 24)"
container="cosmic-neo4j-ci"
volume="cosmic-neo4j-ci"
marker="${GITHUB_RUN_ID:-local}"

cleanup() {
  docker rm -f "$container" >/dev/null 2>&1 || true
}
trap cleanup EXIT
cleanup

docker volume create "$volume" >/dev/null

docker run -d --name "$container" \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH="neo4j/$A_LMI_NEO4J_PASSWORD" \
  -e NEO4J_server_memory_heap_initial__size=256m \
  -e NEO4J_server_memory_heap_max__size=512m \
  -e NEO4J_server_memory_pagecache_size=256m \
  -v "$volume:/data" \
  neo4j:5.14.0 >/dev/null

wait_ready() {
  for _ in $(seq 1 90); do
    if curl -fsS http://127.0.0.1:7474 >/dev/null 2>&1; then
      if python - <<'PY'
import os
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    "bolt://127.0.0.1:7687",
    auth=("neo4j", os.environ["A_LMI_NEO4J_PASSWORD"]),
)
try:
    driver.verify_connectivity()
finally:
    driver.close()
PY
      then
        return 0
      fi
    fi
    sleep 2
  done
  docker logs "$container"
  return 1
}

wait_ready
python integration/vm/live_services.py neo4j-write --marker "$marker"

docker restart "$container" >/dev/null
wait_ready
python integration/vm/live_services.py neo4j-read --marker "$marker"
