#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

