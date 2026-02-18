#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


ALLOWED_ENV_KEYS = {
    "LLM_BACKEND",
    "OPENAI_COMPAT_BASE_URL",
    "OPENAI_COMPAT_API_KEY",
    "OPENAI_COMPAT_MODEL",
    "JUDGE_RUNTIME",
    "RAGAS_BASE_URL",
    "RAGAS_API_KEY",
    "HF_TOKEN",
}

PROJECT_PREFIXES = (
    "LLM_",
    "OPENAI_",
    "RAGAS_",
    "JUDGE_",
    "HF_",
    "HUGGING_",
    "EXTRACTOR_",
    "EMBED_",
    "LLMLITE_",
    "QDRANT_",
    "FALKORDB_",
    "PRIVATE_",
    "RETRIEVAL_",
    "SOVEREIGNTY_",
    "UNSTRUCTURED_",
    "CHUNK_",
    "DOCLING_",
    "SENSOR_",
)

_LINE_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")


def _parse_keys(env_path: Path) -> list[str]:
    if not env_path.exists():
        return []
    keys: list[str] = []
    for line in env_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        m = _LINE_RE.match(raw)
        if not m:
            continue
        keys.append(m.group(1))
    return keys


def _is_project_key(key: str) -> bool:
    if key == "API_KEY":
        return True
    return key.startswith(PROJECT_PREFIXES)


def main() -> int:
    parser = argparse.ArgumentParser(description="Lint .env keys against strict SmartFarm allowlist")
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args()

    env_file = Path(args.env_file).resolve()
    keys = _parse_keys(env_file)
    unknown = sorted({k for k in keys if _is_project_key(k) and k not in ALLOWED_ENV_KEYS})

    if unknown:
        print(f"[env-lint] disallowed project env keys in {env_file}:")
        for key in unknown:
            print(f"  - {key}")
        print("[env-lint] allowed keys:")
        for key in sorted(ALLOWED_ENV_KEYS):
            print(f"  - {key}")
        return 1

    print(f"[env-lint] ok: {env_file} uses only allowed project env keys")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
