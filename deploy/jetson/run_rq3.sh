#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.jetson.yml"
ENV_FILE="${SCRIPT_DIR}/.env.jetson"
OUTPUT_DIR="${BUNDLE_ROOT}/output/jetson_rq3"
OUTPUT_JSON="${OUTPUT_DIR}/edge_profile.json"
OUTPUT_MD="${OUTPUT_DIR}/edge_profile_summary.md"
STACK_STARTED=0

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[rq3] missing required command: $1" >&2
    exit 1
  fi
}

require_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "${path}" ]]; then
    echo "[rq3] missing ${label}: ${path}" >&2
    exit 1
  fi
}

wait_http() {
  local url="$1"
  local label="$2"
  local attempts="${3:-60}"
  local sleep_s="${4:-2}"
  for ((i=1; i<=attempts; i++)); do
    if curl -fsS "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep "${sleep_s}"
  done
  echo "[rq3] ${label} health check failed: ${url}" >&2
  return 1
}

wait_tcp() {
  local host="$1"
  local port="$2"
  local label="$3"
  local attempts="${4:-60}"
  local sleep_s="${5:-2}"
  python3 - "$host" "$port" "$label" "$attempts" "$sleep_s" <<'PY'
import socket
import sys
import time

host, port, label, attempts, sleep_s = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), float(sys.argv[5])
for _ in range(attempts):
    try:
        with socket.create_connection((host, port), timeout=2.0):
            raise SystemExit(0)
    except OSError:
        time.sleep(sleep_s)
print(f"[rq3] {label} TCP health check failed: {host}:{port}", file=sys.stderr)
raise SystemExit(1)
PY
}

collect_results() {
  if [[ "${STACK_STARTED}" == "1" ]]; then
    bash "${SCRIPT_DIR}/collect_results.sh" || true
  fi
}

trap collect_results EXIT

require_cmd docker
require_cmd curl
require_cmd python3

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "[rq3] missing env file: ${ENV_FILE}" >&2
  echo "[rq3] copy deploy/jetson/.env.jetson.example to deploy/jetson/.env.jetson first" >&2
  exit 1
fi

set -a
# shellcheck source=/dev/null
source "${ENV_FILE}"
set +a

require_path "${BUNDLE_ROOT}/smartfarm-llm-inference/models/Qwen3-4B-Q4_K_M.gguf" "GGUF model"
require_path "${BUNDLE_ROOT}/smartfarm-search/models/embeddings/BAAI__bge-m3" "embedding model directory"
require_path "${BUNDLE_ROOT}/data/index/qdrant" "Qdrant index"
require_path "${BUNDLE_ROOT}/data/index/falkordb" "FalkorDB index"
require_path "${BUNDLE_ROOT}/data/artifacts/fusion_weights.runtime.json" "fusion weights artifact"
require_path "${BUNDLE_ROOT}/data/artifacts/fusion_profile_meta.runtime.json" "fusion profile metadata"
require_path "${SCRIPT_DIR}/rq3_queries.txt" "RQ3 query set"

mkdir -p "${OUTPUT_DIR}"

echo "[rq3] starting Jetson runtime stack"
docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" up -d --build qdrant falkordb llama api
STACK_STARTED=1

wait_http "http://localhost:${QDRANT_HOST_PORT:-6333}/collections" "qdrant"
wait_tcp "127.0.0.1" "${FALKORDB_HOST_PORT:-6379}" "falkordb"
wait_http "http://localhost:${LLAMA_HOST_PORT:-45857}/health" "llama"
wait_http "http://localhost:${API_HOST_PORT:-41177}/health/ready" "api"

echo "[rq3] running edge profile"
docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" run --rm --no-deps benchmarking \
  python -m benchmarking.experiments.edge_profile \
  --base-url http://api:41177 \
  --farm-id "${RQ3_FARM_ID:-farm_profile}" \
  --query-file "${RQ3_QUERY_FILE:-/workspace/deploy/jetson/rq3_queries.txt}" \
  --methods "${RQ3_METHODS:-rrf,ours_structural}" \
  --rounds "${RQ3_ROUNDS:-50}" \
  --timeout-s "${RQ3_TIMEOUT_S:-120}" \
  --collect-docker \
  --measure-dat \
  --out-json /workspace/output/jetson_rq3/edge_profile.json \
  --out-md /workspace/output/jetson_rq3/edge_profile_summary.md

python3 - "${OUTPUT_JSON}" "${RQ3_METHODS:-rrf,ours_structural}" <<'PY'
from pathlib import Path
import json
import sys

result_path = Path(sys.argv[1])
methods = [m.strip() for m in sys.argv[2].split(",") if m.strip()]
if not result_path.exists():
    raise SystemExit(f"[rq3] missing result file: {result_path}")
data = json.loads(result_path.read_text(encoding="utf-8"))
for method in methods:
    success_rounds = int((data.get(method) or {}).get("success_rounds") or 0)
    if success_rounds <= 0:
        raise SystemExit(f"[rq3] {method} produced success_rounds=0")
print("[rq3] success_rounds verified for:", ", ".join(methods))
PY

python3 - "${OUTPUT_JSON}" "${RQ3_METHODS:-rrf,ours_structural}" <<'PY'
from pathlib import Path
import json
import sys

result_path = Path(sys.argv[1])
methods = [m.strip() for m in sys.argv[2].split(",") if m.strip()]
data = json.loads(result_path.read_text(encoding="utf-8"))
print("[rq3] output:", result_path)
for method in methods:
    row = data.get(method) or {}
    print(
        f"[rq3] {method}: retrieval_p50={float(row.get('p50_ms') or 0):.2f} ms, "
        f"retrieval_p95={float(row.get('p95_ms') or 0):.2f} ms, "
        f"retrieval_p99={float(row.get('p99_ms') or 0):.2f} ms, "
        f"ttft_p50={float(row.get('ttft_p50_ms') or 0):.2f} ms, "
        f"rss_peak={float(row.get('rss_peak_mb') or 0):.2f} MB, "
        f"qps={float(row.get('qps_success') or 0):.2f}"
    )
docker_mem = data.get("docker_memory") or {}
if docker_mem:
    print(
        f"[rq3] docker total memory: {float(docker_mem.get('total_mb') or 0):.2f} MB "
        f"/ target {float(docker_mem.get('total_budget_mb') or 0):.2f} MB"
    )
PY

echo "[rq3] summary markdown: ${OUTPUT_MD}"
echo "[rq3] done"
