#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EDGE_COMPOSE="$ROOT_DIR/infra/compose/compose.edge.yml"
AUDIT_SCRIPT="$ROOT_DIR/smartfarm-search/scripts/tools/sovereignty_audit.py"

if [[ ! -f "$EDGE_COMPOSE" ]]; then
  echo "[FAIL] missing compose file: $EDGE_COMPOSE"
  exit 2
fi

if ! grep -q "internal: true" "$EDGE_COMPOSE"; then
  echo "[FAIL] private network is not internal in compose.edge.yml"
  exit 3
fi

if ! grep -q "ALLOW_DEV_REMOTE_PRIVATE=.*false" "$EDGE_COMPOSE"; then
  echo "[FAIL] compose.edge.yml must default ALLOW_DEV_REMOTE_PRIVATE=false"
  exit 4
fi

if ! grep -q "PRIVATE_LLM_POLICY=.*local_only" "$EDGE_COMPOSE"; then
  echo "[FAIL] compose.edge.yml must default PRIVATE_LLM_POLICY=local_only"
  exit 5
fi

if ! grep -q "PRIVATE_EGRESS_BLOCK=.*true" "$EDGE_COMPOSE"; then
  echo "[FAIL] compose.edge.yml must default PRIVATE_EGRESS_BLOCK=true"
  exit 6
fi

if ! grep -q "LLM_BACKEND=.*llama_cpp" "$EDGE_COMPOSE"; then
  echo "[FAIL] compose.edge.yml must default LLM_BACKEND=llama_cpp"
  exit 7
fi

if [[ -x "$ROOT_DIR/smartfarm-search/.venv/bin/python" ]]; then
  "$ROOT_DIR/smartfarm-search/.venv/bin/python" "$AUDIT_SCRIPT"
else
  PYTHONPATH="$ROOT_DIR/smartfarm-search:${PYTHONPATH:-}" python3 "$AUDIT_SCRIPT"
fi

echo "[OK] sovereignty gate passed"
