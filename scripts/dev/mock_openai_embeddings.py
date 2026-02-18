#!/usr/bin/env python3
from __future__ import annotations

"""Local OpenAI-compatible embeddings mock.

Reference interface:
- OpenAI embeddings API schema (`/v1/embeddings`) compatible payload/response fields.

License note:
- This file is project-original utility code (no third-party source copy).
"""

import argparse
import hashlib
import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List


def _hash_to_unit(seed: str, index: int) -> float:
    h = hashlib.sha256(f"{seed}:{index}".encode("utf-8")).digest()
    # map first 8 bytes to [-1, 1]
    n = int.from_bytes(h[:8], "big") / float(2**64 - 1)
    return (n * 2.0) - 1.0


def embed_text(text: str, *, dim: int) -> List[float]:
    base = text or ""
    return [_hash_to_unit(base, i) for i in range(dim)]


class _ServerConfig:
    dim: int = 512
    default_model: str = "Qwen/Qwen3-VL-Embedding-2B"
    artificial_delay_ms: float = 0.0


_CONFIG = _ServerConfig()
_PRINT_LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    server_version = "MockOpenAIEmbed/0.1"

    def _send_json(self, code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw.decode("utf-8"))
            return body if isinstance(body, dict) else {}
        except Exception:
            return {}

    def _embeddings(self) -> None:
        body = self._read_json()
        inputs = body.get("input")
        model = str(body.get("model") or _CONFIG.default_model)

        if isinstance(inputs, str):
            texts = [inputs]
        elif isinstance(inputs, list):
            texts = [str(x or "") for x in inputs]
        else:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {
                    "error": {
                        "message": "input must be string or list",
                        "type": "invalid_request_error",
                    }
                },
            )
            return

        if _CONFIG.artificial_delay_ms > 0:
            time.sleep(_CONFIG.artificial_delay_ms / 1000.0)

        data = []
        for i, txt in enumerate(texts):
            data.append(
                {
                    "object": "embedding",
                    "index": i,
                    "embedding": embed_text(txt, dim=int(_CONFIG.dim)),
                }
            )

        self._send_json(
            HTTPStatus.OK,
            {
                "object": "list",
                "model": model,
                "data": data,
                "usage": {
                    "prompt_tokens": 0,
                    "total_tokens": 0,
                },
            },
        )

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.rstrip("/")
        if path in {"/embeddings", "/v1/embeddings"}:
            return self._embeddings()
        self._send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.rstrip("/")
        if path in {"", "/health", "/v1/health"}:
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "dim": int(_CONFIG.dim),
                    "model": _CONFIG.default_model,
                },
            )
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})

    def log_message(self, fmt: str, *args: Any) -> None:
        with _PRINT_LOCK:
            print(f"{self.address_string()} - {fmt % args}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal OpenAI-compatible embeddings mock server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=48080)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--delay-ms", type=float, default=0.0)
    args = parser.parse_args()

    _CONFIG.dim = max(1, int(args.dim))
    _CONFIG.default_model = str(args.model)
    _CONFIG.artificial_delay_ms = max(0.0, float(args.delay_ms))

    server = ThreadingHTTPServer((str(args.host), int(args.port)), Handler)
    print(
        json.dumps(
            {
                "event": "mock_openai_embeddings_server_started",
                "host": args.host,
                "port": int(args.port),
                "dim": int(_CONFIG.dim),
                "model": _CONFIG.default_model,
                "delay_ms": float(_CONFIG.artificial_delay_ms),
            },
            ensure_ascii=False,
        )
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
