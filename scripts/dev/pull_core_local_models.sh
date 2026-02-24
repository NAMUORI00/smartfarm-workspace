#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

LLM_REPO="${CORE_LLM_REPO:-Qwen/Qwen3-4B-GGUF}"
LLM_FILE="${CORE_LLM_FILE:-Qwen3-4B-Q4_K_M.gguf}"
LLM_DIR="${CORE_LLM_DIR:-${ROOT_DIR}/smartfarm-llm-inference/models}"

EMBED_REPO="${CORE_EMBED_REPO:-BAAI/bge-m3}"
EMBED_DIR="${CORE_EMBED_DIR:-${ROOT_DIR}/smartfarm-search/models/embeddings/BAAI__bge-m3}"

mkdir -p "${LLM_DIR}" "${EMBED_DIR}"

echo "[core-models] python=${PYTHON_BIN}"
echo "[core-models] llm=${LLM_REPO}/${LLM_FILE} -> ${LLM_DIR}"
echo "[core-models] embed=${EMBED_REPO} -> ${EMBED_DIR}"

"${PYTHON_BIN}" - <<PY
from pathlib import Path
import sys

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except Exception as exc:
    print("[core-models] missing dependency: huggingface_hub", file=sys.stderr)
    print(f"[core-models] install with: {sys.executable} -m pip install huggingface_hub", file=sys.stderr)
    raise SystemExit(1) from exc

llm_repo = "${LLM_REPO}"
llm_file = "${LLM_FILE}"
llm_dir = Path("${LLM_DIR}")
embed_repo = "${EMBED_REPO}"
embed_dir = Path("${EMBED_DIR}")

llm_path = llm_dir / llm_file
if llm_path.exists():
    print(f"[core-models] skip llm (exists): {llm_path}")
else:
    print(f"[core-models] downloading llm: {llm_repo}/{llm_file}")
    out = hf_hub_download(repo_id=llm_repo, filename=llm_file, local_dir=str(llm_dir))
    print(f"[core-models] llm ready: {out}")

if any(p.name != ".gitkeep" for p in embed_dir.iterdir()):
    print(f"[core-models] embed dir already populated: {embed_dir}")
else:
    print(f"[core-models] downloading embed snapshot: {embed_repo}")
    out = snapshot_download(repo_id=embed_repo, local_dir=str(embed_dir))
    print(f"[core-models] embed ready: {out}")
PY

echo "[core-models] done"
