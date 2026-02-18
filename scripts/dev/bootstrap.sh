#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"

echo "[bootstrap] root=${ROOT_DIR}"
echo "[bootstrap] python=${PYTHON_BIN}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[bootstrap] missing python runtime: ${PYTHON_BIN}" >&2
  exit 1
fi

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python3 -m pip install --upgrade pip setuptools wheel

python3 -m pip install -r "${ROOT_DIR}/smartfarm-search/requirements.txt"
python3 -m pip install -r "${ROOT_DIR}/smartfarm-ingest/requirements.txt"
python3 -m pip install -r "${ROOT_DIR}/smartfarm-benchmarking/requirements.txt"

# Expose ingest console scripts: public-ingest / artifact-export / artifact-import
python3 -m pip install -e "${ROOT_DIR}/smartfarm-ingest"

(
  cd "${ROOT_DIR}/smartfarm-search"
  python3 -m uvicorn core.main:app --help >/dev/null
)
(
  cd "${ROOT_DIR}/smartfarm-ingest"
  python3 -m pipeline.public_ingest_runner --help >/dev/null
  python3 -m pipeline.artifact_export --help >/dev/null
  python3 -m pipeline.artifact_import --help >/dev/null
)
(
  cd "${ROOT_DIR}/smartfarm-benchmarking"
  python3 -m benchmarking.experiments.paper_eval --help >/dev/null
)

echo "[bootstrap] success"
echo "[bootstrap] activate with: source ${VENV_DIR}/bin/activate"
