#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Dict

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None  # type: ignore[assignment]


def _env(name: str, default: str = "") -> str:
    return str(os.getenv(name, default) or default).strip()


def run() -> int:
    base_url = _env("RAGAS_BASE_URL") or _env("OPENAI_COMPAT_BASE_URL") or _env("OPENAI_BASE_URL")
    api_key = _env("RAGAS_API_KEY") or _env("OPENAI_COMPAT_API_KEY") or _env("OPENAI_API_KEY")
    model = _env("RAGAS_MODEL", "openai/gpt-oss-120b")

    result: Dict[str, Any] = {
        "base_url": base_url,
        "model": model,
        "status": "unknown",
    }

    if not base_url:
        result["status"] = "skipped"
        result["reason"] = "base_url_missing"
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if not api_key:
        result["status"] = "skipped"
        result["reason"] = "api_key_missing"
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a strict evaluator."},
            {"role": "user", "content": "Respond with a single word: ok"},
        ],
        "temperature": 0.0,
        "max_tokens": 8,
        "reasoning_effort": "none",
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        if httpx is not None:
            with httpx.Client(timeout=httpx.Timeout(20.0, connect=5.0)) as client:
                r = client.post(f"{base_url.rstrip('/')}/chat/completions", json=payload, headers=headers)
                r.raise_for_status()
                body = r.json() or {}
        else:
            req = urllib.request.Request(
                url=f"{base_url.rstrip('/')}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={**headers, "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=20.0) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            body = json.loads(raw) if raw else {}
        msg = ((body.get("choices") or [{}])[0] or {}).get("message") or {}
        content = str(msg.get("content") or "").strip()
        result["status"] = "ok"
        result["response_preview"] = content[:80]
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    sys.exit(run())
