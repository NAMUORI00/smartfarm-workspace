#!/usr/bin/env python3
"""End-to-end smoke test for SmartFarm stack (llama.cpp + RAG API).

Runs inside Docker Compose network and validates:
  - API health endpoint is reachable
  - (optional) llama.cpp health endpoint is reachable
  - /query returns a response (optionally requiring LLM + Tri-Graph retrieval)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, List, Optional


def _wait_http_ok(client, url: str, timeout_s: float, interval_s: float = 1.0) -> bool:
    deadline = time.time() + timeout_s
    last_err: Optional[str] = None
    while time.time() < deadline:
        try:
            r = client.get(url)
            if r.status_code == 200:
                return True
            last_err = f"HTTP {r.status_code}"
        except Exception as e:  # pragma: no cover - network dependent
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(interval_s)
    print(f"[smoke][FAIL] timeout waiting for {url} ({last_err})", file=sys.stderr)
    return False


def _require_trigraph(sources: List[Dict[str, Any]]) -> bool:
    for s in sources:
        meta = (s or {}).get("metadata") or {}
        mode = str(meta.get("retrieval_mode") or "")
        if mode.startswith("trigraph_") or "trigraph" in mode:
            return True
    return False


def _require_overlay(sources: List[Dict[str, Any]]) -> bool:
    for s in sources:
        meta = (s or {}).get("metadata") or {}
        if meta.get("overlay_version"):
            return True
    return False


def _wait_job_done(client, url: str, timeout_s: float, interval_s: float = 1.0) -> Optional[Dict[str, Any]]:
    deadline = time.time() + timeout_s
    last: Optional[Dict[str, Any]] = None
    while time.time() < deadline:
        try:
            r = client.get(url)
            if r.status_code != 200:
                time.sleep(interval_s)
                continue
            last = r.json()
            status = str(last.get("status") or "")
            if status == "done":
                return last
            if status == "error":
                print(f"[smoke][FAIL] ingest job error: {last.get('error')}", file=sys.stderr)
                return None
        except Exception:  # pragma: no cover - network dependent
            pass
        time.sleep(interval_s)
    print(f"[smoke][FAIL] timeout waiting for ingest job: {url}", file=sys.stderr)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="SmartFarm Docker Compose smoke test")
    parser.add_argument("--api-url", default="http://api:41177", help="API base URL (in Compose network)")
    parser.add_argument("--llm-url", default="http://llama:8080", help="llama.cpp base URL (in Compose network)")
    parser.add_argument("--timeout", type=float, default=120.0, help="Max seconds to wait for services")
    parser.add_argument("--require-llm", action="store_true", help="Fail if LLM generation did not succeed")
    parser.add_argument(
        "--require-trigraph",
        action="store_true",
        help="Fail if retrieval did not use Tri-Graph mode (retrieval_mode contains trigraph)",
    )
    parser.add_argument(
        "--require-overlay",
        action="store_true",
        help="Fail if retrieval sources did not include overlay_version (uploads overlay not applied)",
    )
    parser.add_argument(
        "--exercise-ingest",
        action="store_true",
        help="Enqueue a small upload ingest job before running /query",
    )
    parser.add_argument(
        "--require-edge-lightrag",
        action="store_true",
        help="(deprecated) Alias for --require-trigraph (kept for older compose files)",
    )
    parser.add_argument("--skip-llm-health", action="store_true", help="Skip llama.cpp /health check")
    args = parser.parse_args()

    try:
        import httpx
    except Exception as e:  # pragma: no cover
        print(f"[smoke][FAIL] httpx not available: {e}", file=sys.stderr)
        return 2

    api_health = f"{args.api_url.rstrip('/')}/health"
    api_query = f"{args.api_url.rstrip('/')}/query"
    api_ingest_file = f"{args.api_url.rstrip('/')}/ingest_file"
    api_job = f"{args.api_url.rstrip('/')}/ingest_jobs"
    llm_health = f"{args.llm_url.rstrip('/')}/health"

    # /query can be slow on CPU (edge) especially on cold start; keep read timeout aligned with --timeout.
    timeout = httpx.Timeout(connect=5.0, read=max(10.0, float(args.timeout)), write=5.0, pool=5.0)
    with httpx.Client(timeout=timeout) as client:
        if not _wait_http_ok(client, api_health, timeout_s=args.timeout):
            return 1

        if not args.skip_llm_health:
            if not _wait_http_ok(client, llm_health, timeout_s=args.timeout):
                return 1

        question = f"와사비 생육 적정 온도는? (smoke {int(time.time())})"

        if args.exercise_ingest:
            token = f"SMOKE_TOKEN_{int(time.time())}"
            filename = f"smoke_{token}.txt"
            content = f"{token}\n이 문서는 스모크 테스트를 위한 업로드 문서입니다.\n"
            r = client.post(api_ingest_file, files={"file": (filename, content.encode("utf-8"), "text/plain")})
            if r.status_code != 200:
                print(f"[smoke][FAIL] /ingest_file HTTP {r.status_code}: {r.text[:500]}", file=sys.stderr)
                return 1
            try:
                job_id = r.json().get("job_id")
            except Exception:
                job_id = None
            if not job_id:
                print(f"[smoke][FAIL] /ingest_file missing job_id: {r.text[:500]}", file=sys.stderr)
                return 1

            job_data = _wait_job_done(client, f"{api_job}/{job_id}", timeout_s=args.timeout)
            if job_data is None:
                return 1
            question = f"{token} 관련 내용을 찾아줘. (smoke {int(time.time())})"

        # Avoid exact cache hits so we can verify the current retrieval path (e.g., Tri-Graph / overlay)
        payload = {"question": question, "top_k": 3, "ranker": "none"}
        r = client.post(api_query, json=payload)
        if r.status_code != 200:
            print(f"[smoke][FAIL] /query HTTP {r.status_code}: {r.text[:500]}", file=sys.stderr)
            return 1

        try:
            data = r.json()
        except Exception:
            print(f"[smoke][FAIL] /query returned non-JSON: {r.text[:500]}", file=sys.stderr)
            return 1

        answer = str(data.get("answer") or "")
        fallback_mode = data.get("fallback_mode")
        from_cache = bool(data.get("from_cache", False))
        sources = data.get("sources") or []

        if not answer.strip():
            print("[smoke][FAIL] empty answer", file=sys.stderr)
            return 1

        if args.require_llm and fallback_mode is not None:
            print(f"[smoke][FAIL] expected LLM success but fallback_mode={fallback_mode!r}", file=sys.stderr)
            return 1

        require_trigraph = bool(args.require_trigraph or args.require_edge_lightrag)
        if require_trigraph and not _require_trigraph(sources):
            print("[smoke][FAIL] expected Tri-Graph retrieval but no sources had retrieval_mode containing trigraph", file=sys.stderr)
            print("[smoke] sources metadata sample:", file=sys.stderr)
            for s in list(sources)[:3]:
                meta = (s or {}).get("metadata") or {}
                print(json.dumps(meta, ensure_ascii=False)[:500], file=sys.stderr)
            return 1

        if args.require_overlay and not _require_overlay(sources):
            print("[smoke][FAIL] expected uploads overlay but no sources had overlay_version", file=sys.stderr)
            print("[smoke] sources metadata sample:", file=sys.stderr)
            for s in list(sources)[:3]:
                meta = (s or {}).get("metadata") or {}
                print(json.dumps(meta, ensure_ascii=False)[:500], file=sys.stderr)
            return 1

        print("[smoke][OK]")
        print(f"  answer_len={len(answer)} fallback_mode={fallback_mode!r} from_cache={from_cache} sources={len(sources)}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
