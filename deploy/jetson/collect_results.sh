#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.jetson.yml"
ENV_FILE="${SCRIPT_DIR}/.env.jetson"
OUTPUT_DIR="${BUNDLE_ROOT}/output/jetson_rq3"
LOG_DIR="${OUTPUT_DIR}/logs"

mkdir -p "${LOG_DIR}"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${ENV_FILE}"
  set +a
fi

docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" ps > "${OUTPUT_DIR}/compose_ps.txt" 2>&1 || true
docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" config > "${OUTPUT_DIR}/compose_resolved.yml" 2>&1 || true

for svc in qdrant falkordb llama api; do
  docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" logs --no-color "${svc}" > "${LOG_DIR}/${svc}.log" 2>&1 || true
done

if [[ -f "${ENV_FILE}" ]]; then
  python3 - "${ENV_FILE}" "${OUTPUT_DIR}/env_snapshot.txt" <<'PY'
from pathlib import Path
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
redacted_tokens = ("KEY", "TOKEN", "SECRET", "PASSWORD")
lines = []
for raw in src.read_text(encoding="utf-8").splitlines():
    if "=" not in raw or raw.lstrip().startswith("#"):
        lines.append(raw)
        continue
    key, value = raw.split("=", 1)
    if any(tok in key.upper() for tok in redacted_tokens):
        lines.append(f"{key}=<redacted>")
    else:
        lines.append(f"{key}={value}")
dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
fi

if [[ -f "${SCRIPT_DIR}/rq3_queries.txt" ]]; then
  cp "${SCRIPT_DIR}/rq3_queries.txt" "${OUTPUT_DIR}/rq3_queries.txt"
fi

echo "[collect-results] wrote artifacts under ${OUTPUT_DIR}"
