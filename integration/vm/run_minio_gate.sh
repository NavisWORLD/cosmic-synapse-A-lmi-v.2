#!/usr/bin/env bash
set -euo pipefail

export A_LMI_MINIO_ACCESS_KEY="ciuser${RANDOM}"
export A_LMI_MINIO_SECRET_KEY="$(openssl rand -hex 24)"
container="cosmic-minio-ci"
volume="cosmic-minio-ci"

cleanup() {
  docker rm -f "$container" >/dev/null 2>&1 || true
}
trap cleanup EXIT

cleanup
docker volume create "$volume" >/dev/null

docker run -d --name "$container" -p 9000:9000 \
  -e MINIO_ROOT_USER="$A_LMI_MINIO_ACCESS_KEY" \
  -e MINIO_ROOT_PASSWORD="$A_LMI_MINIO_SECRET_KEY" \
  -v "$volume:/data" \
  quay.io/minio/minio:latest server /data >/dev/null

for _ in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:9000/minio/health/live >/dev/null; then
    break
  fi
  sleep 2
done
curl -fsS http://127.0.0.1:9000/minio/health/live >/dev/null

python integration/vm/live_services.py minio-write

docker restart "$container" >/dev/null
for _ in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:9000/minio/health/live >/dev/null; then
    break
  fi
  sleep 2
done
curl -fsS http://127.0.0.1:9000/minio/health/live >/dev/null

python integration/vm/live_services.py minio-read
