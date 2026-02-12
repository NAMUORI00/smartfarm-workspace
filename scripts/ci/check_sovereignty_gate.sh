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

if ! grep -q "EMBED_BACKEND=llama_cpp" "$EDGE_COMPOSE"; then
  echo "[FAIL] compose.edge.yml must force EMBED_BACKEND=llama_cpp"
  exit 8
fi

SCAN_CMD="rg -n"
if ! command -v rg >/dev/null 2>&1; then
  SCAN_CMD="grep -nE"
fi
if $SCAN_CMD "Deterministic hash embedding fallback|_dummy_vector\\(|sha256\\(\\(text or \"\"\\)\\.encode\\(\"utf-8\"\\)\\)\\.digest\\(" \
  "$ROOT_DIR/smartfarm-search/core/retrieval/qdrant_client.py" \
  "$ROOT_DIR/smartfarm-ingest/pipeline/vector_writer.py" \
  "$ROOT_DIR/smartfarm-benchmarking/benchmarking/experiments/paper_eval.py" >/dev/null; then
  echo "[FAIL] dummy/hash embedding code detected in runtime embedding path"
  exit 9
fi

if [[ -x "$ROOT_DIR/smartfarm-search/.venv/bin/python" ]]; then
  "$ROOT_DIR/smartfarm-search/.venv/bin/python" "$AUDIT_SCRIPT"
else
  PYTHONPATH="$ROOT_DIR/smartfarm-search:${PYTHONPATH:-}" python3 "$AUDIT_SCRIPT"
fi

echo "[OK] sovereignty gate passed"
