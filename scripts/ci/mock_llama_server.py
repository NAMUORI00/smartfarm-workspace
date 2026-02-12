#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import math
from typing import Any, Dict

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="mock-llama")


class CompletionRequest(BaseModel):
    prompt: str
    stream: bool = False
    temperature: float = 0.0
    n_predict: int = 128
    seed: int | None = None
    model: str | None = None


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str | None = None
    encoding_format: str | None = None


def _embed_text(text: str, dim: int = 512) -> list[float]:
    h = hashlib.sha256((text or "").encode("utf-8")).digest()
    vals = [((h[i % len(h)] / 255.0) * 2.0) - 1.0 for i in range(dim)]
    norm = math.sqrt(sum(v * v for v in vals)) or 1.0
    return [v / norm for v in vals]


def _embedding_response(req: EmbeddingRequest) -> Dict[str, Any]:
    if isinstance(req.input, list):
        texts = [str(x or "") for x in req.input]
    else:
        texts = [str(req.input or "")]
    data = []
    for i, text in enumerate(texts):
        data.append({"object": "embedding", "index": i, "embedding": _embed_text(text)})
    return {"object": "list", "data": data, "model": req.model or "mock-embed"}


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.post("/completion")
def completion(req: CompletionRequest) -> Dict[str, Any]:
    text = str(req.prompt or "").strip().splitlines()
    # Return deterministic small output for CI/local E2E gates.
    hint = text[-1][:64] if text else "mock-local-answer"
    return {
        "content": f"mock-local-answer: {hint}",
        "response": f"mock-local-answer: {hint}",
    }


@app.post("/v1/embeddings")
def embeddings_v1(req: EmbeddingRequest) -> Dict[str, Any]:
    return _embedding_response(req)


@app.post("/embeddings")
def embeddings(req: EmbeddingRequest) -> Dict[str, Any]:
    return _embedding_response(req)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run mock llama.cpp /completion server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=45857)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host=str(args.host), port=int(args.port), log_config=None, server_header=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
