#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_BIN_IN="${PY_BIN:-python3}"
if [[ "$PY_BIN_IN" != /* ]]; then
  PY_BIN_IN="$ROOT_DIR/$PY_BIN_IN"
fi
cd "$ROOT_DIR/smartfarm-benchmarking"

export PY_BIN="$PY_BIN_IN"
bash benchmarking/runners/run_suite.sh
