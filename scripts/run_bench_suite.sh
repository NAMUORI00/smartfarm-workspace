#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_BIN_IN="${PY_BIN:-python3}"
if [[ "$PY_BIN_IN" != /* ]]; then
  PY_BIN_IN="$ROOT_DIR/$PY_BIN_IN"
fi

# Load workspace env (if present) so HF token and API settings are available.
if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

# HuggingFace token alias harmonization.
if [[ -n "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi
if [[ -n "${HUGGINGFACE_HUB_TOKEN:-}" && -z "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN="$HUGGINGFACE_HUB_TOKEN"
fi

cd "$ROOT_DIR/smartfarm-benchmarking"

export PY_BIN="$PY_BIN_IN"
bash benchmarking/runners/run_suite.sh
