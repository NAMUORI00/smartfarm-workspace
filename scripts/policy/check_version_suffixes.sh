#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

TMP_FILES="$(mktemp)"
trap 'rm -f "$TMP_FILES"' EXIT

collect_tracked_files() {
  local repo_root="$1"
  local prefix="$2"
  if [[ ! -d "$repo_root/.git" && ! -f "$repo_root/.git" ]]; then
    return
  fi
  git -C "$repo_root" ls-files | sed "s|^|$prefix/|"
}

{
  collect_tracked_files "$ROOT_DIR" "."
  collect_tracked_files "$ROOT_DIR/smartfarm-search" "smartfarm-search"
  collect_tracked_files "$ROOT_DIR/smartfarm-ingest" "smartfarm-ingest"
  collect_tracked_files "$ROOT_DIR/smartfarm-benchmarking" "smartfarm-benchmarking"
} | sed 's|^\./||' | sort -u > "$TMP_FILES"

# Internal migration/version suffixes that must not remain in final naming.
BANNED_RE='(/v2/|query_v2|paper_eval_v2|edge_profile_v2|run_v2_suite|run_v2_bench_suite|run_v2_local_e2e|test_v2_|smartfarm_chunks_v2|smartfarm_v2|query_v2\.log|graphrag_effectiveness_v2|build_trigraph_index_v2)'

# Explicit allowlist for external versioned assets/protocols.
ALLOW_RE='(/v1($|/)|agxqa_v1|MiniLM-L6-v2|MiniLM-L12-v2|mpnet-base-v2|bge-base-en-v1\.5|moonshot-v1-8k|SSPL v1|qdrant/qdrant:v1|api\.openai\.com/v1)'

violations=0
while IFS= read -r rel_path; do
  [[ -n "$rel_path" ]] || continue
  if [[ "$rel_path" == "scripts/policy/check_version_suffixes.sh" ]]; then
    continue
  fi
  abs_path="$ROOT_DIR/$rel_path"
  [[ -f "$abs_path" ]] || continue

  # Skip binary files.
  if ! grep -Iq . "$abs_path"; then
    continue
  fi

  matches="$(grep -nE "$BANNED_RE" "$abs_path" || true)"
  [[ -n "$matches" ]] || continue

  while IFS= read -r line; do
    [[ -n "$line" ]] || continue
    if [[ "$line" =~ $ALLOW_RE ]]; then
      continue
    fi
    echo "[FAIL] forbidden version suffix: ${rel_path}:${line}"
    violations=1
  done <<< "$matches"
done < "$TMP_FILES"

if [[ "$violations" -ne 0 ]]; then
  echo "[FAIL] version suffix policy check failed."
  exit 11
fi

echo "[OK] version suffix policy check passed"
