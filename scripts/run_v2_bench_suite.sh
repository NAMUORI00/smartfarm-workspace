#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR/smartfarm-benchmarking"

bash benchmarking/runners/run_v2_suite.sh
