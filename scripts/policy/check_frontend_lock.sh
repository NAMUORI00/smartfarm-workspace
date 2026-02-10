#!/usr/bin/env bash
set -euo pipefail

if [[ "${ALLOW_FRONTEND_UNLOCK:-0}" == "1" ]]; then
  exit 0
fi

BLOCKED_REGEX='^smartfarm-frontend(/|$)'

usage() {
  cat <<'EOF'
Usage:
  check_frontend_lock.sh --staged
  check_frontend_lock.sh --worktree
  check_frontend_lock.sh --range <from_sha> <to_sha>
EOF
}

fail_if_blocked() {
  local changed_paths="$1"
  local blocked
  blocked="$(printf '%s\n' "$changed_paths" | grep -E "${BLOCKED_REGEX}" || true)"
  if [[ -n "${blocked}" ]]; then
    echo "[lock] frontend path is locked by policy: smartfarm-frontend/**" >&2
    echo "[lock] Set ALLOW_FRONTEND_UNLOCK=1 to override intentionally." >&2
    echo "[lock] blocked paths:" >&2
    printf '%s\n' "${blocked}" >&2
    return 1
  fi
  return 0
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

case "$1" in
  --staged)
    changed="$(git diff --cached --name-only --diff-filter=ACMRD)"
    fail_if_blocked "${changed}"
    ;;
  --worktree)
    changed="$(git status --porcelain --ignore-submodules=none | awk '{print $2}')"
    fail_if_blocked "${changed}"
    ;;
  --range)
    if [[ $# -ne 3 ]]; then
      usage >&2
      exit 2
    fi
    from_sha="$2"
    to_sha="$3"
    changed="$(git diff --name-only "${from_sha}..${to_sha}")"
    fail_if_blocked "${changed}"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
