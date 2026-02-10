#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

chmod +x .githooks/pre-commit .githooks/pre-push scripts/policy/check_frontend_lock.sh
git config core.hooksPath .githooks

echo "[hooks] installed: core.hooksPath=.githooks"
echo "[hooks] frontend lock is active (override: ALLOW_FRONTEND_UNLOCK=1)."

if git -C smartfarm-frontend rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  frontend_hooks_dir="$(git -C smartfarm-frontend rev-parse --git-path hooks)"
  mkdir -p "${frontend_hooks_dir}"

  cat >"${frontend_hooks_dir}/pre-commit" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${ALLOW_FRONTEND_UNLOCK:-0}" == "1" ]]; then
  exit 0
fi
echo "[lock] smartfarm-frontend commit is blocked by policy." >&2
echo "[lock] Set ALLOW_FRONTEND_UNLOCK=1 to override intentionally." >&2
exit 1
EOF

  cat >"${frontend_hooks_dir}/pre-push" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${ALLOW_FRONTEND_UNLOCK:-0}" == "1" ]]; then
  exit 0
fi
echo "[lock] smartfarm-frontend push is blocked by policy." >&2
echo "[lock] Set ALLOW_FRONTEND_UNLOCK=1 to override intentionally." >&2
exit 1
EOF

  chmod +x "${frontend_hooks_dir}/pre-commit" "${frontend_hooks_dir}/pre-push"
  echo "[hooks] installed frontend submodule hooks: ${frontend_hooks_dir}"
fi
