#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import httpx


def _wait_ready(url: str, timeout_s: float = 30.0) -> bool:
    t0 = time.time()
    while (time.time() - t0) < float(timeout_s):
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.3)
    return False


def _run(cmd: list[str], *, cwd: Path, env: Dict[str, str]) -> subprocess.Popen[str]:
    return subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _terminate(p: subprocess.Popen[str] | None) -> None:
    if p is None:
        return
    if p.poll() is not None:
        return
    try:
        p.send_signal(signal.SIGTERM)
        p.wait(timeout=4)
    except Exception:
        try:
            p.kill()
        except Exception:
            pass


def _post_json(base: str, path: str, payload: Dict[str, Any], timeout: float = 20.0) -> Dict[str, Any]:
    r = httpx.post(f"{base.rstrip('/')}{path}", json=payload, timeout=timeout)
    r.raise_for_status()
    out = r.json()
    if not isinstance(out, dict):
        return {}
    return out


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def _collect_tail(p: subprocess.Popen[str] | None, max_lines: int = 80) -> str:
    if p is None or p.stdout is None:
        return ""
    try:
        lines = p.stdout.read().splitlines()
    except Exception:
        return ""
    return "\n".join(lines[-max_lines:])


def run_local_e2e(
    *,
    workspace: Path,
    py_bin: str,
    api_port: int,
    llama_port: int,
    run_dataset_bench: bool,
) -> Tuple[Dict[str, Any], int]:
    search_dir = workspace / "smartfarm-search"
    bench_dir = workspace / "smartfarm-benchmarking"
    out_dir = workspace / "output" / "e2e_local"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_url = f"http://127.0.0.1:{api_port}"
    llama_url = f"http://127.0.0.1:{llama_port}"

    llama_proc: subprocess.Popen[str] | None = None
    api_proc: subprocess.Popen[str] | None = None

    summary: Dict[str, Any] = {
        "base_url": base_url,
        "llama_url": llama_url,
        "py_bin": py_bin,
        "dataset_bench": bool(run_dataset_bench),
        "checks": {},
    }

    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(search_dir),
            "HOST": "127.0.0.1",
            "PORT": str(api_port),
            "LLM_BACKEND": "llama_cpp",
            "LLMLITE_HOST": llama_url,
            "PRIVATE_LLM_POLICY": "local_only",
            "SOVEREIGNTY_ENFORCED": "true",
            "PRIVATE_EGRESS_BLOCK": "true",
            "ALLOW_DEV_REMOTE_PRIVATE": "false",
            "PRIVATE_STORE_DB_PATH": str(out_dir / "private_overlay.sqlite"),
            "LOG_PATH": str(out_dir / "query.log"),
            "QDRANT_HOST": "127.0.0.1",
            "QDRANT_PORT": "6333",
            "FALKORDB_HOST": "127.0.0.1",
            "FALKORDB_PORT": "6379",
        }
    )

    try:
        subprocess.check_call([py_bin, "-c", "import fastapi,uvicorn,httpx"], cwd=str(workspace), env=env)
        summary["checks"]["python_runtime_ok"] = True

        llama_proc = _run(
            [py_bin, str(workspace / "scripts" / "ci" / "mock_llama_server.py"), "--host", "127.0.0.1", "--port", str(llama_port)],
            cwd=workspace,
            env=env,
        )
        _assert(_wait_ready(f"{llama_url}/health", timeout_s=20.0), "mock llama server did not start in time")
        summary["checks"]["mock_llama_ready"] = True

        api_proc = _run(
            [py_bin, "-m", "uvicorn", "core.main:app", "--host", "127.0.0.1", "--port", str(api_port)],
            cwd=search_dir,
            env=env,
        )
        _assert(_wait_ready(f"{base_url}/health/live", timeout_s=30.0), "search API did not become healthy")
        summary["checks"]["api_live"] = True

        memo = _post_json(
            base_url,
            "/private/memo",
            {"farm_id": "farm-e2e", "text": "하우스 습도 88 기록", "source_type": "memo"},
        )
        _assert(str(memo.get("status")) == "ok", "private memo ingest failed")
        _assert(int(memo.get("extracted_entities") or 0) > 0, "private extraction entities must be created")
        summary["checks"]["ingest_private_memo"] = True

        q1 = _post_json(
            base_url,
            "/query",
            {"question": "습도 88 기록 요약해줘", "farm_id": "farm-e2e", "top_k": 5, "debug": True},
        )
        routing1 = q1.get("routing") or {}
        _assert(str(routing1.get("generation_provider")) == "local", "private query must route to local provider")
        _assert(str(routing1.get("embedding_provider")) == "llama_cpp", "edge embedding provider must be llama_cpp")
        _assert(bool(routing1.get("private_present")) is True, "private query should include private evidence")
        _assert(bool(routing1.get("private_eligible")) is True, "private eligible should be true for sensor/memo-style query")
        summary["checks"]["private_query_routing"] = True

        purge = _post_json(base_url, "/admin/purge_private", {"farm_id": "farm-e2e"})
        _assert(int(purge.get("deleted_overlay_rows") or 0) >= 1, "private purge did not delete overlay rows")
        summary["checks"]["purge_private"] = True

        q2 = _post_json(
            base_url,
            "/query",
            {"question": "습도 88 기록 요약해줘", "farm_id": "farm-e2e", "top_k": 5, "debug": True},
        )
        routing2 = q2.get("routing") or {}
        _assert(bool(routing2.get("private_present")) is False, "private must not appear after purge")
        summary["checks"]["post_purge_private_absent"] = True

        edge_out = out_dir / "edge_profile_local.json"
        edge_cmd = [
            py_bin,
            "-m",
            "benchmarking.experiments.edge_profile",
            "--base-url",
            base_url,
            "--farm-id",
            "farm-e2e",
            "--query",
            "습도 88 기록 기반으로 병해 리스크 알려줘",
            "--rounds",
            "10",
            "--out",
            str(edge_out),
        ]
        subprocess.check_call(edge_cmd, cwd=str(bench_dir), env=env)
        edge_data = json.loads(edge_out.read_text(encoding="utf-8"))
        _assert(int(edge_data.get("success_rounds") or 0) > 0, "edge profile must include successful rounds")
        summary["checks"]["edge_profile_success"] = True
        summary["edge_profile"] = edge_data

        if run_dataset_bench:
            bench_cmd = [
                "bash",
                str(workspace / "scripts" / "run_bench_suite.sh"),
            ]
            bench_env = dict(env)
            bench_env.update(
                {
                    "PY_BIN": py_bin,
                    "TRACK_A_QUERIES": "1",
                    "TRACK_B_QUERIES": "1",
                    "ABLATION_QUERIES": "1",
                    "RETRIEVAL_ONLY_FLAG": "--retrieval-only",
                }
            )
            subprocess.check_call(bench_cmd, cwd=str(workspace), env=bench_env)
            summary["checks"]["dataset_bench_smoke"] = True

        summary["status"] = "ok"
        code = 0
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = str(exc)
        summary["api_log_tail"] = _collect_tail(api_proc)
        summary["llama_log_tail"] = _collect_tail(llama_proc)
        code = 1
    finally:
        _terminate(api_proc)
        _terminate(llama_proc)

    out_file = out_dir / "local_e2e_summary.json"
    out_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary, code


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local E2E gate with mock llama server")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--py-bin", default=str(Path("smartfarm-search/.venv/bin/python")))
    parser.add_argument("--api-port", type=int, default=41177)
    parser.add_argument("--llama-port", type=int, default=45857)
    parser.add_argument("--with-dataset-bench", action="store_true")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    py_bin = str(Path(args.py_bin))
    if not Path(py_bin).is_absolute():
        py_bin = str((workspace / py_bin).absolute())

    _, rc = run_local_e2e(
        workspace=workspace,
        py_bin=py_bin,
        api_port=int(args.api_port),
        llama_port=int(args.llama_port),
        run_dataset_bench=bool(args.with_dataset_bench),
    )
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
