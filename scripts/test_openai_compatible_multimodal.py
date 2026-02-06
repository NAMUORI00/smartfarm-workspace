#!/usr/bin/env python3
from __future__ import annotations

"""
OpenAI-compatible endpoint smoke test: discover a multimodal-capable model.

This script:
  1) Lists available models via GET /v1/models
  2) Probes candidate models with a tiny inlined PNG image (data URL)
  3) Reports which model(s) accept multimodal input (image + text)

It intentionally reads secrets from environment variables only:
  - OPENAI_BASE_URL (e.g., http://host:port/v1)
  - API_KEY (or OPENAI_API_KEY fallback)

Usage:
  python scripts/test_openai_compatible_multimodal.py

  OPENAI_BASE_URL=http://.../v1 API_KEY=... \
    python scripts/test_openai_compatible_multimodal.py --max-probe 20
"""

import argparse
import base64
import json
import os
import struct
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional convenience dependency
    load_dotenv = None


def _ensure_v1(base_url: str) -> str:
    raw = (base_url or "").strip()
    if not raw:
        return ""
    raw = raw.rstrip("/")
    if raw.endswith("/v1"):
        return raw
    return raw + "/v1"


def _png_data_url_rgb(r: int, g: int, b: int, *, w: int = 32, h: int = 32) -> str:
    """Create a tiny RGB PNG as a data URL (no external image hosting needed)."""
    w = max(1, int(w))
    h = max(1, int(h))
    r = max(0, min(255, int(r)))
    g = max(0, min(255, int(g)))
    b = max(0, min(255, int(b)))

    def _chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8-bit RGB

    # Each scanline: filter byte 0 + RGB pixels.
    row = bytes([0] + [r, g, b] * w)
    raw = row * h
    compressed = zlib.compress(raw, level=9)

    png = signature + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", compressed) + _chunk(b"IEND", b"")
    b64 = base64.b64encode(png).decode("ascii")
    return "data:image/png;base64," + b64


def _safe_err_text(obj: Any) -> str:
    """Extract a short provider error message without leaking request data."""
    if isinstance(obj, dict):
        err = obj.get("error")
        if isinstance(err, dict):
            msg = err.get("message") or err.get("error") or err.get("type")
            if msg:
                return str(msg)[:240]
        if "message" in obj:
            return str(obj.get("message"))[:240]
    return str(obj)[:240]


@dataclass(frozen=True)
class ProbeResult:
    model: str
    ok: bool
    status_code: int
    latency_ms: float
    answer: str
    error: str


def _score_model_id(model_id: str) -> int:
    mid = (model_id or "").lower()
    score = 0
    # Prefer obvious multimodal families first.
    for pat, w in [
        ("gpt-4o", 120),
        ("gpt-4.1", 110),
        ("vision", 105),
        ("vl", 100),
        ("multimodal", 95),
        ("gemini", 90),
        ("claude-3", 80),
        ("llava", 70),
        ("qwen", 60),
    ]:
        if pat in mid:
            score += int(w)
    # Prefer smaller/cheaper variants for iterative probing if present.
    for pat, w in [("mini", 12), ("flash", 10), ("lite", 8), ("haiku", 6)]:
        if pat in mid:
            score += int(w)
    # Slight penalty for clearly text-only hints.
    for pat, w in [("text", -10), ("instruct", -2)]:
        if pat in mid:
            score += int(w)
    return int(score)


def _list_models(client: httpx.Client, *, base_url_v1: str) -> List[str]:
    r = client.get(f"{base_url_v1}/models")
    r.raise_for_status()
    data = r.json()
    items = data.get("data") if isinstance(data, dict) else None
    out: List[str] = []
    if isinstance(items, list):
        for it in items:
            if isinstance(it, dict):
                mid = str(it.get("id") or "").strip()
                if mid:
                    out.append(mid)
    return sorted(set(out))


def _probe_multimodal(
    client: httpx.Client,
    *,
    base_url_v1: str,
    model: str,
    image_url: str,
    timeout_s: float,
) -> ProbeResult:
    payload: Dict[str, Any] = {
        "model": str(model),
        "temperature": 0,
        "max_tokens": 16,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look at the image. What color is the square? Reply with one word."},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
    }

    t0 = time.perf_counter()
    try:
        r = client.post(f"{base_url_v1}/chat/completions", json=payload, timeout=float(timeout_s))
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if r.status_code != 200:
            try:
                obj = r.json()
            except Exception:
                obj = {"message": (r.text or "").strip()}
            return ProbeResult(
                model=str(model),
                ok=False,
                status_code=int(r.status_code),
                latency_ms=float(dt_ms),
                answer="",
                error=_safe_err_text(obj),
            )
        obj = r.json()
        answer = ""
        try:
            choices = obj.get("choices") if isinstance(obj, dict) else None
            if isinstance(choices, list) and choices:
                msg = choices[0].get("message") or {}
                answer = str(msg.get("content") or "").strip()
        except Exception:
            answer = ""
        return ProbeResult(
            model=str(model),
            ok=bool(answer),
            status_code=int(r.status_code),
            latency_ms=float(dt_ms),
            answer=answer[:120],
            error="",
        )
    except Exception as e:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return ProbeResult(
            model=str(model),
            ok=False,
            status_code=0,
            latency_ms=float(dt_ms),
            answer="",
            error=f"{type(e).__name__}: {e}",
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="OpenAI-compatible multimodal model probe (safe; env-secrets only)")
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""), help="OpenAI-compatible base URL (e.g., http://host:port/v1)")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--max-probe", type=int, default=25, help="Max number of models to probe with image input")
    ap.add_argument("--list-only", action="store_true", help="Only list models; do not probe")
    args = ap.parse_args()

    # Convenience: load workspace-root .env/.env.local if python-dotenv is available.
    if load_dotenv is not None:
        try:
            repo_root = Path(__file__).resolve().parents[1]
            load_dotenv(repo_root / ".env", override=False)
            load_dotenv(repo_root / ".env.local", override=False)
        except Exception:
            pass

    base_url_v1 = _ensure_v1(str(args.base_url))
    if not base_url_v1:
        raise SystemExit("OPENAI_BASE_URL (or --base-url) is required.")

    api_key = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    if not api_key.strip():
        raise SystemExit("API_KEY (or OPENAI_API_KEY) env var is required (do not inline secrets into commands).")

    headers = {"Authorization": f"Bearer {api_key.strip()}"}
    client = httpx.Client(headers=headers)

    models = _list_models(client, base_url_v1=base_url_v1)
    print(f"[models] count={len(models)} base_url={base_url_v1}")
    if not models:
        return 1

    # Print a short sample list (keep output readable).
    preview = models[:20]
    if preview:
        print("[models] sample:")
        for m in preview:
            print(f"  - {m}")
        if len(models) > len(preview):
            print(f"  ... (+{len(models) - len(preview)} more)")

    if bool(args.list_only):
        return 0

    # Probe: rank models by heuristic score and probe top-N.
    ranked = sorted(models, key=lambda m: (-_score_model_id(m), m))
    to_probe = ranked[: max(1, int(args.max_probe))]

    img = _png_data_url_rgb(255, 0, 0, w=32, h=32)

    results: List[ProbeResult] = []
    for m in to_probe:
        res = _probe_multimodal(
            client,
            base_url_v1=base_url_v1,
            model=m,
            image_url=img,
            timeout_s=float(args.timeout),
        )
        results.append(res)
        status = "OK" if res.ok else "FAIL"
        extra = f" ans={res.answer!r}" if res.ok else f" err={res.error!r}"
        print(f"[probe] {status} model={m} http={res.status_code} latency_ms={res.latency_ms:.1f}{extra}")

    oks = [r for r in results if r.ok]
    print(f"[summary] probed={len(results)} ok={len(oks)}")
    if oks:
        best = sorted(oks, key=lambda r: (r.latency_ms, r.model))[0]
        print(f"[recommend] model={best.model} (first working by latency among probed)")
        print(f"[recommend] export OPENAI_MODEL={best.model}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
