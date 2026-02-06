#!/usr/bin/env python3
from __future__ import annotations

"""Compose-first end-to-end test runner for SmartFarm EdgeKG stack.

This script is designed to run inside Docker Compose network (recommended) but
also supports host execution by overriding --api-url/--llm-url and path envs.

It exercises:
  - API + llama.cpp health
  - Base retrieval (Tri-Graph enabled)
  - Overlay update (manual ingest + file ingest) + local llama KB extraction (gleaning)
  - Sensor ingest -> rollup -> overlay update
  - OCR image ingest (multimodal via OCR)
  - Base update apply (inbox -> base.sqlite -> base bundle rebuild -> hot reload)
  - Delete -> rebuild regression
"""

import argparse
import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


def _utc_ts() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _now_ms() -> int:
    return int(time.time() * 1000)


def _resolve_path(p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


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
    print(f"[e2e][FAIL] timeout waiting for {url} ({last_err})", file=sys.stderr)
    return False


def _wait_job_done(client, url: str, timeout_s: float, interval_s: float = 1.0) -> Optional[dict]:
    deadline = time.time() + timeout_s
    last: Optional[dict] = None
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
                return last
        except Exception:  # pragma: no cover - network dependent
            pass
        time.sleep(interval_s)
    return last


def _query(client, api_url: str, *, question: str, top_k: int = 4, owner_id: str = "default") -> dict:
    payload = {"question": question, "top_k": int(top_k), "ranker": "none", "owner_id": str(owner_id or "default")}
    r = client.post(f"{api_url.rstrip('/')}/query", json=payload)
    r.raise_for_status()
    return r.json()


def _create_base_sqlite(sqlite_path: Path, *, token: str) -> None:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    if sqlite_path.exists():
        sqlite_path.unlink()

    conn = sqlite3.connect(str(sqlite_path))
    try:
        # Keep the file self-contained (avoid WAL sidecars in the inbox).
        conn.execute("PRAGMA journal_mode=DELETE;")
        conn.execute("PRAGMA synchronous=FULL;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT,
                source TEXT,
                created_at TEXT,
                text TEXT,
                metadata_json TEXT,
                content_sha256 TEXT,
                sensitivity TEXT DEFAULT 'public',
                owner_id TEXT,
                deleted INTEGER DEFAULT 0
            );
            """
        )
        doc_id = f"base_e2e_doc_{_utc_ts()}"
        chunk_id = f"{doc_id}#c0"
        text = f"{token}\nThis is an E2E base chunk.\n"
        conn.execute(
            """
            INSERT INTO chunks(chunk_id, doc_id, source, created_at, text, metadata_json, content_sha256, sensitivity, owner_id, deleted)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, 0);
            """,
            (
                chunk_id,
                doc_id,
                "base_e2e",
                _utc_ts(),
                text,
                json.dumps({"source": "base_e2e"}, ensure_ascii=False),
                "",
                "public",
                None,
            ),
        )
        conn.commit()
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _wait_base_applied(marker_path: Path, *, expected_version: str, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        raw = _read_json(marker_path)
        if isinstance(raw, dict) and str(raw.get("version") or "") == str(expected_version):
            return True
        time.sleep(1.0)
    return False


def _open_sqlite_ro(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn


def _assert_extraction_ok(db_path: Path, *, chunk_id: str) -> dict:
    conn = _open_sqlite_ro(db_path)
    try:
        row = conn.execute(
            "SELECT status, gleaning_passes_used, extracted_json FROM extractions WHERE chunk_id = ? LIMIT 1;",
            (str(chunk_id),),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"missing extraction row for chunk_id={chunk_id}")
        status = str(row["status"] or "")
        if status != "ok":
            raise RuntimeError(f"extraction status != ok (chunk_id={chunk_id} status={status!r})")
        return {"status": status, "gleaning_passes_used": int(row["gleaning_passes_used"] or 0)}
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _generate_ocr_image_png(*, token: str) -> bytes:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Pillow not available: {e}")

    img = Image.new("RGB", (1400, 420), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Prefer DejaVu fonts if present (common in Debian images). Fallback to default bitmap font.
    font = None
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            font = ImageFont.truetype(p, size=96)
            break
        except Exception:
            font = None
    if font is None:
        font = ImageFont.load_default()

    text = f"OCRTEST {token}"
    draw.text((40, 140), text, fill=(0, 0, 0), font=font)

    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


@dataclass
class StepResult:
    name: str
    ok: bool
    ms: int
    details: Dict[str, Any] = field(default_factory=dict)


def main() -> int:
    ap = argparse.ArgumentParser(description="SmartFarm EdgeKG v3.2 Compose-first E2E runner")
    ap.add_argument("--api-url", default=os.getenv("E2E_API_URL", "http://api:41177"))
    ap.add_argument("--llm-url", default=os.getenv("E2E_LLM_URL", "http://llama:8080"))
    ap.add_argument("--timeout", type=float, default=float(os.getenv("E2E_TIMEOUT", "300")))
    ap.add_argument("--run-id", default=os.getenv("E2E_RUN_ID", _utc_ts()))
    ap.add_argument("--report-dir", default=os.getenv("E2E_REPORT_DIR", "output/e2e"))
    ap.add_argument("--skip-ocr", action="store_true", help="Skip OCR ingest step (for quick debugging)")
    args = ap.parse_args()

    try:
        import httpx
    except Exception as e:  # pragma: no cover
        print(f"[e2e][FAIL] httpx not available: {e}", file=sys.stderr)
        return 2

    run_id = str(args.run_id or _utc_ts())
    results: list[StepResult] = []

    def _record(name: str, ok: bool, started_ms: int, **details: Any) -> None:
        results.append(StepResult(name=name, ok=bool(ok), ms=_now_ms() - started_ms, details=dict(details)))

    api_url = str(args.api_url).rstrip("/")
    llm_url = str(args.llm_url).rstrip("/")
    timeout_s = float(args.timeout)

    # Align with slow edge/CPU inference.
    timeout = httpx.Timeout(connect=5.0, read=max(10.0, timeout_s), write=5.0, pool=5.0)

    overlay_db_path = _resolve_path(os.getenv("INGEST_DB_PATH", "data/kb/overlay.sqlite"))
    kb_inbox_dir = _resolve_path(os.getenv("KB_INBOX_DIR", "data/inbox/base_updates"))
    base_kb_path = _resolve_path(os.getenv("BASE_KB_PATH", "data/kb/base.sqlite"))
    base_applied_marker = base_kb_path.parent / "base_applied.json"

    report_dir = _resolve_path(str(args.report_dir)) / run_id
    report_path = report_dir / "report.json"

    def _write_report(exit_code: int) -> None:
        payload = {
            "run_id": run_id,
            "api_url": api_url,
            "llm_url": llm_url,
            "timestamp_utc": _utc_ts(),
            "exit_code": int(exit_code),
            "steps": [
                {"name": s.name, "ok": s.ok, "ms": s.ms, "details": s.details}
                for s in results
            ],
        }
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[e2e] report: {report_path}")

    try:
        with httpx.Client(timeout=timeout) as client:
            # -----------------------------------------------------------------
            # 1) Health
            # -----------------------------------------------------------------
            started = _now_ms()
            api_ok = _wait_http_ok(client, f"{api_url}/health", timeout_s=timeout_s)
            llm_ok = _wait_http_ok(client, f"{llm_url}/health", timeout_s=timeout_s)
            _record("health", api_ok and llm_ok, started, api_health=api_ok, llm_health=llm_ok)
            if not (api_ok and llm_ok):
                _write_report(1)
                return 1

            # -----------------------------------------------------------------
            # 1.5) Clear response cache (avoid stale retrieval_mode / from_cache)
            # -----------------------------------------------------------------
            started = _now_ms()
            cache_cleared = False
            try:
                rr = client.post(f"{api_url}/cache/clear")
                cache_cleared = rr.status_code == 200
            except Exception:
                cache_cleared = False
            _record("cache_clear", True, started, cleared=cache_cleared)

            # -----------------------------------------------------------------
            # 2) Tier-0 query smoke (requires LLM success + Tri-Graph enabled)
            # -----------------------------------------------------------------
            started = _now_ms()
            q0 = f"와사비 생육 적정 온도는? (e2e {run_id} t={int(time.time())})"
            data0 = _query(client, api_url, question=q0, top_k=3)
            answer0 = str(data0.get("answer") or "")
            sources0 = data0.get("sources") or []
            mode0 = ""
            try:
                if sources0:
                    mode0 = str((sources0[0].get("metadata") or {}).get("retrieval_mode") or "")
            except Exception:
                mode0 = ""
            ok0 = bool(answer0.strip()) and (data0.get("fallback_mode") is None) and ("trigraph" in mode0)
            _record(
                "query_smoke",
                ok0,
                started,
                fallback_mode=data0.get("fallback_mode"),
                from_cache=bool(data0.get("from_cache", False)),
                retrieval_mode=mode0,
                sources=len(sources0),
            )
            if not ok0:
                _write_report(1)
                return 1

            # -----------------------------------------------------------------
            # 3) Overlay manual ingest -> extraction -> overlay publish
            # -----------------------------------------------------------------
            started = _now_ms()
            memo_doc_id = f"e2e_memo_{run_id}"
            memo_token = f"MEMO_E2E_{run_id}"
            r = client.post(
                f"{api_url}/ingest",
                json={
                    "id": memo_doc_id,
                    "text": f"{memo_token}\n이 문서는 E2E 테스트용 메모입니다.\n원인/해결 키워드 포함.",
                    "metadata": {"source": "e2e_manual"},
                    "owner_id": "default",
                    "sensitivity": "private",
                },
            )
            r.raise_for_status()
            job_id = str((r.json() or {}).get("job_id") or "")
            if not job_id:
                _record("overlay_manual_ingest", False, started, error="missing job_id")
                _write_report(1)
                return 1

            job = _wait_job_done(client, f"{api_url}/ingest_jobs/{job_id}", timeout_s=timeout_s)
            if not isinstance(job, dict) or str(job.get("status") or "") != "done":
                _record("overlay_manual_ingest", False, started, job=job)
                _write_report(1)
                return 1

            # Query for the memo token and ensure it appears in sources.
            q1 = f"{memo_token} 내용을 찾아줘. (e2e {run_id} t={int(time.time())})"
            data1 = _query(client, api_url, question=q1, top_k=4)
            sources1 = data1.get("sources") or []
            found1 = any(memo_token in str(s.get("text") or "") for s in sources1)
            overlay_version = None
            try:
                for s in sources1:
                    meta = (s.get("metadata") or {}) if isinstance(s, dict) else {}
                    if meta.get("overlay_version"):
                        overlay_version = str(meta.get("overlay_version"))
                        break
            except Exception:
                overlay_version = None

            # Extraction status check (gleaning included)
            chunk_id = f"{memo_doc_id}#c0"
            extraction_info = _assert_extraction_ok(overlay_db_path, chunk_id=chunk_id)

            ok1 = bool(found1 and overlay_version)
            _record(
                "overlay_manual_ingest",
                ok1,
                started,
                job_id=job_id,
                overlay_version=overlay_version,
                extraction=extraction_info,
            )
            if not ok1:
                _write_report(1)
                return 1

            # -----------------------------------------------------------------
            # 4) Sensor ingest -> rollup job -> query
            # -----------------------------------------------------------------
            started = _now_ms()
            farm_id = f"farm_e2e_{run_id}"
            zone_id = f"zone_e2e_{run_id}"
            sr = client.post(
                f"{api_url}/sensor/ingest",
                json={
                    "owner_id": "default",
                    "sensitivity": "private",
                    "rollup_window_minutes": 60,
                    "readings": [
                        {
                            "farm_id": farm_id,
                            "zone_id": zone_id,
                            "sensor_id": "s1",
                            "metric": "temperature",
                            "value": 18.5,
                            "unit": "℃",
                        },
                        {
                            "farm_id": farm_id,
                            "zone_id": zone_id,
                            "sensor_id": "s1",
                            "metric": "humidity",
                            "value": 75.0,
                            "unit": "%",
                        },
                    ],
                },
            )
            sr.raise_for_status()
            s_job_id = str((sr.json() or {}).get("job_id") or "")
            if not s_job_id:
                _record("sensor_ingest", False, started, error="missing job_id")
                _write_report(1)
                return 1

            s_job = _wait_job_done(client, f"{api_url}/ingest_jobs/{s_job_id}", timeout_s=timeout_s)
            if not isinstance(s_job, dict) or str(s_job.get("status") or "") != "done":
                _record("sensor_ingest", False, started, job=s_job)
                _write_report(1)
                return 1

            q2 = f"farm={farm_id} zone={zone_id} sensor summary를 찾아줘. (e2e {run_id} t={int(time.time())})"
            data2 = _query(client, api_url, question=q2, top_k=4)
            sources2 = data2.get("sources") or []
            found2 = any((farm_id in str(s.get("text") or "")) and ("Sensor Summary" in str(s.get("text") or "")) for s in sources2)
            _record("sensor_ingest", bool(found2), started, job_id=s_job_id, sources=len(sources2))
            if not found2:
                _write_report(1)
                return 1

            # -----------------------------------------------------------------
            # 5) OCR image ingest (multimodal via OCR)
            # -----------------------------------------------------------------
            if args.skip_ocr:
                _record("ocr_ingest", True, _now_ms(), skipped=True)
            else:
                started = _now_ms()
                ocr_token = f"{run_id[-10:]}".replace("_", "")
                png = _generate_ocr_image_png(token=ocr_token)
                filename = f"ocr_e2e_{run_id}.png"
                fr = client.post(
                    f"{api_url}/ingest_file",
                    files={"file": (filename, png, "image/png")},
                )
                fr.raise_for_status()
                f_job_id = str((fr.json() or {}).get("job_id") or "")
                if not f_job_id:
                    _record("ocr_ingest", False, started, error="missing job_id")
                    _write_report(1)
                    return 1

                f_job = _wait_job_done(client, f"{api_url}/ingest_jobs/{f_job_id}", timeout_s=timeout_s)
                if not isinstance(f_job, dict) or str(f_job.get("status") or "") != "done":
                    _record("ocr_ingest", False, started, job=f_job)
                    _write_report(1)
                    return 1

                # Query a stable token ("OCRTEST") rather than the run_id suffix.
                q3 = f"OCRTEST {ocr_token} (e2e t={int(time.time())})"
                data3 = _query(client, api_url, question=q3, top_k=4)
                sources3 = data3.get("sources") or []
                found3 = any("OCRTEST" in str(s.get("text") or "").upper() for s in sources3)
                _record("ocr_ingest", bool(found3), started, job_id=f_job_id, sources=len(sources3))
                if not found3:
                    _write_report(1)
                    return 1

            # -----------------------------------------------------------------
            # 6) Base update apply (inbox -> base.sqlite -> base bundle rebuild -> hot reload)
            # -----------------------------------------------------------------
            started = _now_ms()
            base_token = f"BASE_E2E_{run_id}"
            base_version = f"e2e_{run_id}"
            inbox_version_dir = kb_inbox_dir / base_version
            inbox_sqlite = inbox_version_dir / "base.sqlite"
            inbox_latest = kb_inbox_dir / "LATEST"
            inbox_version_dir.mkdir(parents=True, exist_ok=True)
            _create_base_sqlite(inbox_sqlite, token=base_token)
            _write_text(inbox_latest, base_version)

            # Wait until worker applies the inbox version
            applied_ok = _wait_base_applied(base_applied_marker, expected_version=base_version, timeout_s=timeout_s)

            # Query until the base token appears in sources (hot reload)
            found_base = False
            base_sources = 0
            base_deadline = time.time() + timeout_s
            while time.time() < base_deadline:
                data4 = _query(client, api_url, question=f"{base_token} (e2e base t={int(time.time())})", top_k=4)
                srcs = data4.get("sources") or []
                base_sources = len(srcs)
                if any(base_token in str(s.get("text") or "") for s in srcs):
                    found_base = True
                    break
                time.sleep(1.0)

            ok_base = bool(applied_ok and found_base)
            _record(
                "base_update_apply",
                ok_base,
                started,
                version=base_version,
                marker=str(base_applied_marker),
                sources=base_sources,
            )
            if not ok_base:
                _write_report(1)
                return 1

            # -----------------------------------------------------------------
            # 7) Delete -> rebuild regression (overlay memo)
            # -----------------------------------------------------------------
            started = _now_ms()
            dr = client.delete(f"{api_url}/documents/{memo_doc_id}")
            dr.raise_for_status()
            rebuild_job_id = str((dr.json() or {}).get("rebuild_job_id") or "")
            if not rebuild_job_id:
                _record("delete_rebuild", False, started, error="missing rebuild_job_id")
                _write_report(1)
                return 1

            rjob = _wait_job_done(client, f"{api_url}/ingest_jobs/{rebuild_job_id}", timeout_s=timeout_s)
            if not isinstance(rjob, dict) or str(rjob.get("status") or "") != "done":
                _record("delete_rebuild", False, started, job=rjob)
                _write_report(1)
                return 1

            data5 = _query(client, api_url, question=f"{memo_token} (e2e delete t={int(time.time())})", top_k=4)
            sources5 = data5.get("sources") or []
            still_there = any(memo_token in str(s.get("text") or "") for s in sources5)
            _record("delete_rebuild", not still_there, started, job_id=rebuild_job_id, sources=len(sources5))
            if still_there:
                _write_report(1)
                return 1

    except Exception as e:  # pragma: no cover - unexpected
        started = _now_ms()
        _record("unexpected_exception", False, started, error=f"{type(e).__name__}: {e}")
        _write_report(1)
        return 1

    _write_report(0)
    print("[e2e][OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
