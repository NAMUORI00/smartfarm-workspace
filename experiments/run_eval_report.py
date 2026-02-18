#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in [REPO_ROOT / "smartfarm-search", REPO_ROOT / "smartfarm-benchmarking"]:
    p = str(_path)
    if p not in sys.path:
        sys.path.insert(0, p)

from benchmarking.metrics.qa_metrics import exact_match, f1_score
from benchmarking.metrics.ragas_adapter import RagasRecord, evaluate_ragas
from benchmarking.metrics.retrieval_metrics import recall_at_k
from benchmarking.utils.canonical_id_mapper import (
    map_gold_ids_to_canonical_doc_ids,
    map_retrieved_hits_to_canonical_doc_ids,
)
from core.Config.Settings import settings

ALLOWED_MODES = ("no_rag", "no_rag_ragprompt", "naive_rag", "ours")
DEFAULT_MODES = ("no_rag", "naive_rag", "ours")
DEFAULT_INDEX_PATHS = ("data/index", "smartfarm-search/data/cache")
_TOKEN_RE = re.compile(r"\s+")
_PROCESS = psutil.Process(os.getpid()) if psutil is not None else None
_RUNTIME_IMPORTED = False
ProviderRouter = None  # type: ignore[assignment]
RetrievalService = None  # type: ignore[assignment]
build_context = None  # type: ignore[assignment]
build_rag_prompt = None  # type: ignore[assignment]


def _ensure_runtime_imports() -> None:
    global _RUNTIME_IMPORTED
    global ProviderRouter
    global RetrievalService
    global build_context
    global build_rag_prompt

    if _RUNTIME_IMPORTED:
        return

    try:
        from core.llm_gateway.router import ProviderRouter as _ProviderRouter
        from core.retrieval.context_builder import build_context as _build_context
        from core.retrieval.context_builder import build_rag_prompt as _build_rag_prompt
        from core.retrieval.service import RetrievalService as _RetrievalService
    except Exception as exc:
        raise RuntimeError(
            "failed to import smartfarm-search runtime modules. "
            "Install runtime deps (e.g. httpx) in the execution environment."
        ) from exc

    ProviderRouter = _ProviderRouter
    RetrievalService = _RetrievalService
    build_context = _build_context
    build_rag_prompt = _build_rag_prompt
    _RUNTIME_IMPORTED = True


@dataclass
class EvalSample:
    qid: str
    question: str
    gold_answers: List[str] = field(default_factory=list)
    gold_doc_ids: List[str] = field(default_factory=list)
    gold_canonical_doc_ids: List[str] = field(default_factory=list)
    farm_id: str = "farm_eval"
    top_k: Optional[int] = None
    modalities: Optional[List[str]] = None
    include_private: Optional[bool] = None


@dataclass
class EvalRow:
    mode: str
    qid: str
    question: str
    answer: str
    gold_answers: List[str]
    gold_doc_ids: List[str]
    gold_canonical_doc_ids: List[str]
    retrieved_doc_ids: List[str]
    retrieved_canonical_doc_ids: List[str]
    retrieved_scores: List[float]
    exact_match: Optional[float]
    f1: Optional[float]
    task_accuracy: Optional[float]
    recall_at_k: Optional[float]
    recall_at_k_doc: Optional[float]
    recall_at_k_chunk: Optional[float]
    k_used: int
    ttft_ms: float
    ttft_source: str
    total_latency_ms: float
    retrieve_ms: float
    generate_ms: float
    peak_rss_mb: float
    cold_start_ms: float
    index_size_mb: float
    error: str
    contexts: List[str] = field(default_factory=list, repr=False)

    def to_csv_row(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "qid": self.qid,
            "question": self.question,
            "answer": self.answer,
            "gold_answers": json.dumps(self.gold_answers, ensure_ascii=False),
            "gold_doc_ids": json.dumps(self.gold_doc_ids, ensure_ascii=False),
            "gold_canonical_doc_ids": json.dumps(self.gold_canonical_doc_ids, ensure_ascii=False),
            "retrieved_doc_ids": json.dumps(self.retrieved_doc_ids, ensure_ascii=False),
            "retrieved_canonical_doc_ids": json.dumps(self.retrieved_canonical_doc_ids, ensure_ascii=False),
            "retrieved_scores": json.dumps(self.retrieved_scores, ensure_ascii=False),
            "exact_match": self.exact_match,
            "f1": self.f1,
            "task_accuracy": self.task_accuracy,
            "recall_at_k": self.recall_at_k,
            "recall_at_k_doc": self.recall_at_k_doc,
            "recall_at_k_chunk": self.recall_at_k_chunk,
            "k_used": self.k_used,
            "ttft_ms": self.ttft_ms,
            "ttft_source": self.ttft_source,
            "total_latency_ms": self.total_latency_ms,
            "retrieve_ms": self.retrieve_ms,
            "generate_ms": self.generate_ms,
            "peak_rss_mb": self.peak_rss_mb,
            "cold_start_ms": self.cold_start_ms,
            "index_size_mb": self.index_size_mb,
            "error": self.error,
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _elapsed_ms(start: float) -> float:
    return max(0.0, (time.perf_counter() - start) * 1000.0)


def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(float(v) for v in values)
    idx = int(round((float(p) / 100.0) * (len(s) - 1)))
    idx = max(0, min(len(s) - 1, idx))
    return float(s[idx])


def _mean(values: Sequence[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _mean_or_none(values: Sequence[float]) -> Optional[float]:
    return float(statistics.mean(values)) if values else None


def _compact_whitespace(text: str) -> str:
    return _TOKEN_RE.sub(" ", str(text or "")).strip()


def _as_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, tuple):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, set):
        return [str(x).strip() for x in sorted(raw) if str(x).strip()]
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                obj = json.loads(s)
                if isinstance(obj, list):
                    return [str(x).strip() for x in obj if str(x).strip()]
            except Exception:
                pass
        return [x.strip() for x in re.split(r"[,\t;|]", s) if x.strip()]
    return [str(raw).strip()] if str(raw).strip() else []


def _dedupe_keep_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        key = str(item).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _parse_bool(raw: Any) -> Optional[bool]:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    s = str(raw).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    return None


def _resolve_path(path_like: str) -> Path:
    p = Path(str(path_like))
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return int(path.stat().st_size)
        except Exception:
            return 0
    total = 0
    for f in path.rglob("*"):
        if not f.is_file():
            continue
        try:
            total += int(f.stat().st_size)
        except Exception:
            continue
    return total


def _index_size_mb(paths: Sequence[Path]) -> float:
    return float(sum(_dir_size_bytes(p) for p in paths)) / (1024.0 * 1024.0)


def _current_rss_mb() -> float:
    if _PROCESS is not None:
        try:
            return float(_PROCESS.memory_info().rss) / (1024.0 * 1024.0)
        except Exception:
            pass
    try:
        import resource

        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":
            return rss / (1024.0 * 1024.0)
        return rss / 1024.0
    except Exception:
        return 0.0


def _device_spec() -> Dict[str, Any]:
    mem_mb = None
    if psutil is not None:
        try:
            mem_mb = float(psutil.virtual_memory().total) / (1024.0 * 1024.0)
        except Exception:
            mem_mb = None
    if mem_mb is None:
        try:
            pages = int(os.sysconf("SC_PHYS_PAGES"))
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            mem_mb = (pages * page_size) / (1024.0 * 1024.0)
        except Exception:
            mem_mb = None

    cpu_model = ""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        try:
            for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.lower().startswith("model name"):
                    cpu_model = line.split(":", 1)[-1].strip()
                    break
        except Exception:
            cpu_model = ""

    return {
        "platform": platform.platform(),
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "cpu_logical": int(os.cpu_count() or 0),
        "cpu_model": cpu_model,
        "ram_total_mb": mem_mb,
        "python": sys.version.split()[0],
    }


def _parse_qrels_tsv(path: Optional[Path]) -> Dict[str, List[str]]:
    if path is None or not path.exists():
        return {}

    out: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        header_checked = False
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if not header_checked:
                header_checked = True
                if len(parts) >= 2 and parts[0].lower() in {"query-id", "qid"}:
                    continue
            if len(parts) < 2:
                continue
            qid = str(parts[0]).strip()
            doc_id = ""
            score = 1.0
            # Support both 3-col and 4-col qrels TSV.
            # 3-col: qid, doc_id, score
            # 4-col: qid, iter, doc_id, score
            if len(parts) >= 4:
                doc_id = str(parts[2]).strip()
                try:
                    score = float(parts[3])
                except Exception:
                    score = 1.0
            else:
                doc_id = str(parts[1]).strip()
                if len(parts) >= 3:
                    try:
                        score = float(parts[2])
                    except Exception:
                        score = 1.0
            if not qid or not doc_id or score <= 0:
                continue
            out.setdefault(qid, []).append(doc_id)

    for qid, doc_ids in list(out.items()):
        out[qid] = _dedupe_keep_order(doc_ids)
    return out


def _load_samples(
    *,
    questions_path: Path,
    qrels: Dict[str, List[str]],
    default_farm_id: str,
    default_modalities: Optional[List[str]],
    max_questions: int,
) -> List[EvalSample]:
    if not questions_path.exists():
        raise FileNotFoundError(f"questions file not found: {questions_path}")

    samples: List[EvalSample] = []
    with questions_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                raise ValueError(f"invalid jsonl at line {i}: {exc}") from exc
            if not isinstance(row, dict):
                continue

            qid = str(
                row.get("qid")
                or row.get("id")
                or row.get("question_id")
                or row.get("_id")
                or f"q{i:04d}"
            ).strip()
            question = str(row.get("question") or row.get("query") or row.get("text") or "").strip()
            if not question:
                continue

            gold_answers = _as_list(row.get("gold_answers"))
            if not gold_answers:
                gold_answers = _as_list(
                    row.get("gold_answer")
                    or row.get("reference_answer")
                    or row.get("ground_truth")
                    or row.get("answer")
                )
            gold_answers = [x for x in (_compact_whitespace(x) for x in gold_answers) if x]

            gold_doc_ids = _as_list(row.get("gold_doc_ids") or row.get("relevant_doc_ids") or row.get("doc_ids"))
            if not gold_doc_ids:
                gold_doc_ids = list(qrels.get(qid) or [])
            gold_doc_ids = _dedupe_keep_order(gold_doc_ids)
            gold_canonical_doc_ids = _as_list(
                row.get("gold_canonical_doc_ids")
                or row.get("canonical_gold_doc_ids")
            )
            if not gold_canonical_doc_ids:
                gold_canonical_doc_ids = map_gold_ids_to_canonical_doc_ids(gold_doc_ids, dedupe=True)

            farm_id = str(row.get("farm_id") or default_farm_id).strip() or default_farm_id
            top_k = None
            if row.get("top_k") is not None:
                try:
                    top_k = max(1, int(row.get("top_k")))
                except Exception:
                    top_k = None
            if top_k is None and row.get("k") is not None:
                try:
                    top_k = max(1, int(row.get("k")))
                except Exception:
                    top_k = None

            modalities = None
            if row.get("modalities") is not None:
                m = _as_list(row.get("modalities"))
                modalities = [x.lower() for x in m if x.lower() in {"text", "table", "image"}] or None
            if modalities is None and default_modalities:
                modalities = list(default_modalities)

            include_private = _parse_bool(row.get("include_private"))

            samples.append(
                EvalSample(
                    qid=qid,
                    question=question,
                    gold_answers=gold_answers,
                    gold_doc_ids=gold_doc_ids,
                    gold_canonical_doc_ids=gold_canonical_doc_ids,
                    farm_id=farm_id,
                    top_k=top_k,
                    modalities=modalities,
                    include_private=include_private,
                )
            )
            if max_questions > 0 and len(samples) >= max_questions:
                break
    return samples


def _best_qa(prediction: str, references: Sequence[str]) -> tuple[Optional[float], Optional[float], Optional[float]]:
    refs = [r for r in references if str(r).strip()]
    if not refs:
        return None, None, None
    em = max(float(exact_match(prediction, ref)) for ref in refs)
    f1 = max(float(f1_score(prediction, ref)) for ref in refs)
    return em, f1, em


def _resolve_modes(raw: str) -> List[str]:
    out: List[str] = []
    for part in str(raw or "").split(","):
        m = part.strip().lower()
        if not m:
            continue
        if m not in ALLOWED_MODES:
            raise ValueError(f"unsupported mode: {m} (allowed: {', '.join(ALLOWED_MODES)})")
        if m not in out:
            out.append(m)
    return out or list(DEFAULT_MODES)


def _summarize_mode(mode: str, rows: Sequence[EvalRow], ragas_result: Dict[str, Any]) -> Dict[str, Any]:
    success_rows = [r for r in rows if not r.error]
    lat = [float(r.total_latency_ms) for r in success_rows]
    warm_lat = [float(r.total_latency_ms) for r in rows[1:] if not r.error]
    ttft = [float(r.ttft_ms) for r in success_rows if float(r.ttft_ms) > 0.0]
    retrieve = [float(r.retrieve_ms) for r in success_rows]
    generate = [float(r.generate_ms) for r in success_rows]
    peak_rss = max([float(r.peak_rss_mb) for r in rows], default=0.0)
    cold_start = float(rows[0].cold_start_ms) if rows else 0.0
    index_size = float(rows[0].index_size_mb) if rows else 0.0

    em_vals = [float(r.exact_match) for r in rows if r.exact_match is not None]
    f1_vals = [float(r.f1) for r in rows if r.f1 is not None]
    acc_vals = [float(r.task_accuracy) for r in rows if r.task_accuracy is not None]
    recall_vals = [float(r.recall_at_k) for r in rows if r.recall_at_k is not None]
    recall_doc_vals = [float(r.recall_at_k_doc) for r in rows if r.recall_at_k_doc is not None]
    recall_chunk_vals = [float(r.recall_at_k_chunk) for r in rows if r.recall_at_k_chunk is not None]

    return {
        "mode": mode,
        "counts": {
            "total": len(rows),
            "success": len(success_rows),
            "errors": len(rows) - len(success_rows),
            "with_gold_answer": len(em_vals),
            "with_gold_doc_ids": len(recall_vals),
        },
        "quality": {
            "em_mean": _mean_or_none(em_vals),
            "f1_mean": _mean_or_none(f1_vals),
            "task_accuracy_mean": _mean_or_none(acc_vals),
        },
        "retrieval": {
            "recall_at_k_mean": _mean_or_none(recall_vals),
            "recall_at_k_doc_mean": _mean_or_none(recall_doc_vals),
            "recall_at_k_chunk_mean": _mean_or_none(recall_chunk_vals),
        },
        "system": {
            "ttft_ms": {
                "mean": _mean(ttft),
                "p50": _percentile(ttft, 50),
                "p95": _percentile(ttft, 95),
            },
            "total_latency_ms": {
                "mean": _mean(lat),
                "p50": _percentile(lat, 50),
                "p95": _percentile(lat, 95),
            },
            "retrieve_ms_mean": _mean(retrieve),
            "generate_ms_mean": _mean(generate),
            "peak_rss_mb": peak_rss,
            "index_size_mb": index_size,
            "cold_start_ms": cold_start,
            "warm_p50_ms": _percentile(warm_lat, 50),
        },
        "ragas": ragas_result,
    }


def _metric_from_summary(summary: Dict[str, Any], key: str) -> Optional[float]:
    cur: Any = summary
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    try:
        return float(cur)
    except Exception:
        return None


def _build_delta(ours: Dict[str, Any], baseline: Dict[str, Any], baseline_mode: str) -> Dict[str, Any]:
    metric_keys = [
        "quality.em_mean",
        "quality.f1_mean",
        "quality.task_accuracy_mean",
        "retrieval.recall_at_k_mean",
        "system.ttft_ms.p50",
        "system.ttft_ms.p95",
        "system.total_latency_ms.p50",
        "system.total_latency_ms.p95",
        "system.peak_rss_mb",
        "system.cold_start_ms",
        "system.warm_p50_ms",
    ]
    metrics: Dict[str, Any] = {}
    for mk in metric_keys:
        ours_v = _metric_from_summary(ours, mk)
        base_v = _metric_from_summary(baseline, mk)
        if ours_v is None or base_v is None:
            continue
        delta = ours_v - base_v
        rel = None
        if abs(base_v) > 1e-12:
            rel = delta / base_v
        metrics[mk] = {
            "ours": ours_v,
            "baseline": base_v,
            "delta_ours_minus_baseline": delta,
            "relative_delta": rel,
        }
    return {
        "baseline_mode": baseline_mode,
        "metrics": metrics,
    }


def _failure_type(row: EvalRow, latency_p95_by_mode: Dict[str, float]) -> Optional[str]:
    mode = row.mode
    if row.error:
        return f"generation_error/{mode}"
    if not _compact_whitespace(row.answer):
        return f"empty_answer/{mode}"
    has_gold_for_recall = False
    if row.recall_at_k_doc is not None and row.gold_canonical_doc_ids:
        has_gold_for_recall = True
    if row.recall_at_k_chunk is not None and row.gold_doc_ids:
        has_gold_for_recall = True
    if has_gold_for_recall and row.recall_at_k is not None and float(row.recall_at_k) <= 0.0:
        return f"retrieval_miss/{mode}"
    if row.exact_match is not None and float(row.exact_match) < 1.0:
        return f"answer_mismatch/{mode}"
    p95 = float(latency_p95_by_mode.get(mode) or 0.0)
    if p95 > 0.0 and float(row.total_latency_ms) > p95 * 1.2:
        return f"latency_outlier/{mode}"
    return None


def _write_top_failures(path: Path, rows: Sequence[EvalRow], summary_by_mode: Dict[str, Dict[str, Any]]) -> None:
    latency_p95 = {
        mode: float(((summary.get("system") or {}).get("total_latency_ms") or {}).get("p95") or 0.0)
        for mode, summary in summary_by_mode.items()
    }
    grouped: Dict[str, List[EvalRow]] = {}
    for row in rows:
        kind = _failure_type(row, latency_p95)
        if not kind:
            continue
        grouped.setdefault(kind, []).append(row)

    sorted_groups = sorted(grouped.items(), key=lambda kv: len(kv[1]), reverse=True)
    top_groups = sorted_groups[:5]

    desc = {
        "generation_error": "LLM 호출/파이프라인에서 예외 발생",
        "empty_answer": "응답 텍스트가 비어 있음",
        "retrieval_miss": "gold_doc_ids가 있지만 recall@k=0",
        "answer_mismatch": "gold_answer 대비 exact_match < 1.0",
        "latency_outlier": "모드 내 latency p95 대비 20% 이상 높은 지연",
    }

    lines: List[str] = []
    lines.append("# Top Failures")
    lines.append("")
    lines.append(f"- generated_at: {_now_iso()}")
    lines.append(f"- total_failure_rows: {sum(len(v) for v in grouped.values())}")

    if not top_groups:
        lines.append("")
        lines.append("현재 휴리스틱 기준의 실패 row가 없습니다.")
        path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return

    for idx, (kind, group_rows) in enumerate(top_groups, start=1):
        base = kind.split("/", 1)[0]
        lines.append("")
        lines.append(f"## {idx}. {kind} ({len(group_rows)} cases)")
        lines.append(f"- rule: {desc.get(base, '휴리스틱 분류')}")
        for ex in group_rows[:3]:
            q_preview = _compact_whitespace(ex.question)
            if len(q_preview) > 120:
                q_preview = q_preview[:117] + "..."
            lines.append(
                f"- example: qid={ex.qid}, mode={ex.mode}, recall@k={ex.recall_at_k}, "
                f"latency_ms={round(ex.total_latency_ms, 2)}, question={q_preview}"
            )

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _selected_provider_name(private_present: bool) -> str:
    if private_present:
        if settings.PRIVATE_LLM_POLICY == "dev_allow_remote" and bool(settings.ALLOW_DEV_REMOTE_PRIVATE):
            return "openai_compatible"
        return "local"
    if settings.LLM_BACKEND == "openai_compatible":
        return "openai_compatible"
    return "local"


def _extract_openai_stream_text(event: Dict[str, Any]) -> str:
    choices = event.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    choice = choices[0] or {}
    if not isinstance(choice, dict):
        return ""
    delta = choice.get("delta") or {}
    if not isinstance(delta, dict):
        return ""
    content = delta.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _stream_local_completion(
    *,
    prompt: str,
    max_tokens: int,
    temperature: float,
    seed: int,
) -> tuple[Optional[str], Optional[float]]:
    import httpx

    timeout = httpx.Timeout(
        connect=5.0,
        read=float(settings.LLMLITE_TIMEOUT),
        write=5.0,
        pool=5.0,
    )
    payload: Dict[str, Any] = {
        "prompt": prompt,
        "stream": True,
        "temperature": float(temperature),
        "n_predict": int(max_tokens),
    }
    if int(seed) >= 0:
        payload["seed"] = int(seed)
    if str(settings.LLMLITE_MODEL or "").strip():
        payload["model"] = str(settings.LLMLITE_MODEL)

    started = time.perf_counter()
    first_token_ms: Optional[float] = None
    chunks: List[str] = []
    url = f"{str(settings.LLMLITE_HOST).rstrip('/')}/completion"
    with httpx.stream("POST", url, json=payload, timeout=timeout) as r:
        r.raise_for_status()
        for raw_line in r.iter_lines():
            line = str(raw_line or "").strip()
            if not line:
                continue
            if line.startswith("data:"):
                line = line[5:].strip()
            if not line or line == "[DONE]":
                continue
            try:
                event = json.loads(line)
            except Exception:
                continue
            text = str(event.get("content") or event.get("response") or "")
            if not text:
                continue
            if first_token_ms is None:
                first_token_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
            chunks.append(text)

    answer = "".join(chunks).strip()
    if not answer:
        return None, None
    return answer, first_token_ms


def _stream_openai_completion(
    *,
    prompt: str,
    max_tokens: int,
    temperature: float,
    seed: int,
) -> tuple[Optional[str], Optional[float]]:
    import httpx

    base_url = str(settings.OPENAI_COMPAT_BASE_URL or "").strip()
    if not base_url:
        return None, None

    timeout = httpx.Timeout(
        connect=5.0,
        read=float(settings.OPENAI_COMPAT_TIMEOUT),
        write=5.0,
        pool=5.0,
    )
    payload: Dict[str, Any] = {
        "model": str(settings.OPENAI_COMPAT_MODEL or settings.LLMLITE_MODEL or ""),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "reasoning_effort": "none",
        "stream": True,
    }
    if int(seed) >= 0:
        payload["seed"] = int(seed)
    headers: Dict[str, str] = {}
    api_key = str(settings.OPENAI_COMPAT_API_KEY or "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    started = time.perf_counter()
    first_token_ms: Optional[float] = None
    chunks: List[str] = []
    url = f"{base_url.rstrip('/')}/chat/completions"
    with httpx.stream("POST", url, json=payload, headers=headers, timeout=timeout) as r:
        r.raise_for_status()
        for raw_line in r.iter_lines():
            line = str(raw_line or "").strip()
            if not line:
                continue
            if line.startswith("data:"):
                line = line[5:].strip()
            if not line:
                continue
            if line == "[DONE]":
                break
            try:
                event = json.loads(line)
            except Exception:
                continue
            text = _extract_openai_stream_text(event)
            if not text:
                continue
            if first_token_ms is None:
                first_token_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
            chunks.append(text)

    answer = "".join(chunks).strip()
    if not answer:
        return None, None
    return answer, first_token_ms


def _try_stream_generate(
    *,
    prompt: str,
    private_present: bool,
    max_tokens: int,
    temperature: float,
    seed: int,
) -> tuple[Optional[str], Optional[float]]:
    provider_name = _selected_provider_name(private_present)
    try:
        if provider_name == "openai_compatible":
            return _stream_openai_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
            )
        return _stream_local_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
        )
    except Exception:
        return None, None


class ModeRunner:
    def __init__(
        self,
        mode: str,
        *,
        naive_top_k: int,
        ours_top_k: int,
        metric_unit: str,
        max_tokens: int,
        temperature: float,
        seed: int,
    ) -> None:
        _ensure_runtime_imports()
        self.mode = mode
        self.naive_top_k = max(1, int(naive_top_k))
        self.ours_top_k = max(1, int(ours_top_k))
        self.metric_unit = str(metric_unit).strip().lower() or "doc"
        self.max_tokens = max(1, int(max_tokens))
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.router = ProviderRouter()
        self.retrieval: Optional[RetrievalService] = None
        if mode in {"naive_rag", "ours"}:
            self.retrieval = RetrievalService()
        self.peak_rss_mb = 0.0

    def _resolve_include_private(self, sample: EvalSample) -> bool:
        if sample.include_private is not None:
            return bool(sample.include_private)
        if self.retrieval is None:
            return False
        try:
            return bool(self.retrieval.overlay.has_private_records(farm_id=sample.farm_id))
        except Exception:
            return False

    def run_one(self, sample: EvalSample) -> EvalRow:
        started = time.perf_counter()
        answer = ""
        error = ""
        ttft_ms = 0.0
        ttft_source = "none"
        retrieve_ms = 0.0
        generate_ms = 0.0
        hits: List[Any] = []
        contexts: List[str] = []
        k_used = 0

        try:
            if self.mode == "no_rag":
                prompt = sample.question
                private_present = False
            elif self.mode == "no_rag_ragprompt":
                prompt = build_rag_prompt(sample.question, "")
                private_present = False
            elif self.mode == "naive_rag":
                if self.retrieval is None:
                    raise RuntimeError("naive_rag requires retrieval service")
                include_private = self._resolve_include_private(sample)
                k_used = int(self.naive_top_k)
                t0 = time.perf_counter()
                raw_hits = self.retrieval.qdrant.search(
                    query=sample.question,
                    farm_id=sample.farm_id,
                    limit=k_used,
                    include_private=include_private,
                    modalities=sample.modalities,
                    enable_dense=True,
                    enable_sparse=False,
                )
                hits = sorted(raw_hits, key=lambda x: float(getattr(x, "score", 0.0)), reverse=True)[:k_used]
                retrieve_ms = _elapsed_ms(t0)
                context, _, private_present = build_context(hits)
                contexts = [str(getattr(h, "text", "") or "") for h in hits if str(getattr(h, "text", "") or "").strip()]
                prompt = build_rag_prompt(sample.question, context)
            elif self.mode == "ours":
                if self.retrieval is None:
                    raise RuntimeError("ours mode requires retrieval service")
                include_private = sample.include_private
                k_used = int(sample.top_k or self.ours_top_k)
                t0 = time.perf_counter()
                hits, _diag = self.retrieval.retrieve_with_diagnostics(
                    query=sample.question,
                    farm_id=sample.farm_id,
                    top_k=k_used,
                    modalities=sample.modalities,
                    include_private=include_private,
                )
                retrieve_ms = _elapsed_ms(t0)
                context, _, private_present = build_context(hits)
                contexts = [str(getattr(h, "text", "") or "") for h in hits if str(getattr(h, "text", "") or "").strip()]
                prompt = build_rag_prompt(sample.question, context)
            else:
                raise ValueError(f"unsupported mode: {self.mode}")

            t1 = time.perf_counter()
            streamed_answer, streamed_ttft = _try_stream_generate(
                prompt=prompt,
                private_present=bool(private_present),
                max_tokens=int(self.max_tokens),
                temperature=float(self.temperature),
                seed=int(self.seed),
            )
            if streamed_answer is not None:
                answer = str(streamed_answer or "").strip()
                generate_ms = _elapsed_ms(t1)
                if streamed_ttft is not None:
                    ttft_ms = float(max(0.0, streamed_ttft))
                    ttft_source = "stream_first_token"
                else:
                    ttft_ms = float(generate_ms)
                    ttft_source = "fallback"
            else:
                routed = self.router.generate(
                    prompt=prompt,
                    private_present=bool(private_present),
                    max_tokens=int(self.max_tokens),
                    temperature=float(self.temperature),
                )
                generate_ms = _elapsed_ms(t1)
                ttft_ms = float(generate_ms)
                ttft_source = "fallback"
                answer = str(getattr(routed, "answer", "") or "").strip()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        total_latency = _elapsed_ms(started)
        rss_now = _current_rss_mb()
        self.peak_rss_mb = max(self.peak_rss_mb, rss_now)

        retrieved_doc_ids = [str(getattr(h, "id", "")) for h in hits if str(getattr(h, "id", "")).strip()]
        retrieved_canonical_doc_ids = map_retrieved_hits_to_canonical_doc_ids(hits, dedupe=False)
        retrieved_scores = [float(getattr(h, "score", 0.0) or 0.0) for h in hits]
        em, f1, acc = _best_qa(answer, sample.gold_answers)

        recall_k = None
        recall_doc = None
        recall_chunk = None
        if self.mode not in {"no_rag", "no_rag_ragprompt"}:
            k_metric = max(1, int(k_used or len(retrieved_doc_ids) or 1))
            if self.metric_unit in {"doc", "both"} and sample.gold_canonical_doc_ids:
                recall_doc = float(
                    recall_at_k(
                        retrieved_canonical_doc_ids,
                        set(sample.gold_canonical_doc_ids),
                        k_metric,
                        dedupe_retrieved=True,
                    )
                )
            if self.metric_unit in {"chunk", "both"} and sample.gold_doc_ids:
                recall_chunk = float(
                    recall_at_k(
                        retrieved_doc_ids,
                        set(sample.gold_doc_ids),
                        k_metric,
                        dedupe_retrieved=False,
                    )
                )
        if self.metric_unit == "chunk":
            recall_k = recall_chunk
        else:
            recall_k = recall_doc if recall_doc is not None else recall_chunk

        return EvalRow(
            mode=self.mode,
            qid=sample.qid,
            question=sample.question,
            answer=answer,
            gold_answers=list(sample.gold_answers),
            gold_doc_ids=list(sample.gold_doc_ids),
            gold_canonical_doc_ids=list(sample.gold_canonical_doc_ids),
            retrieved_doc_ids=retrieved_doc_ids,
            retrieved_canonical_doc_ids=retrieved_canonical_doc_ids,
            retrieved_scores=retrieved_scores,
            exact_match=em,
            f1=f1,
            task_accuracy=acc,
            recall_at_k=recall_k,
            recall_at_k_doc=recall_doc,
            recall_at_k_chunk=recall_chunk,
            k_used=int(k_used),
            ttft_ms=float(ttft_ms),
            ttft_source=ttft_source,
            total_latency_ms=float(total_latency),
            retrieve_ms=float(retrieve_ms),
            generate_ms=float(generate_ms),
            peak_rss_mb=float(self.peak_rss_mb),
            cold_start_ms=0.0,
            index_size_mb=0.0,
            error=error,
            contexts=contexts,
        )


def _run_mode(
    *,
    mode: str,
    samples: Sequence[EvalSample],
    naive_top_k: int,
    ours_top_k: int,
    metric_unit: str,
    max_tokens: int,
    temperature: float,
    seed: int,
    index_size_mb: float,
    with_ragas: bool,
    ragas_model: str,
    ragas_base_url: str,
    ragas_api_key: str,
) -> tuple[List[EvalRow], Dict[str, Any]]:
    runner = ModeRunner(
        mode,
        naive_top_k=naive_top_k,
        ours_top_k=ours_top_k,
        metric_unit=metric_unit,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
    )
    rows: List[EvalRow] = []
    cold_start = 0.0
    seen_first = False
    ragas_records: List[RagasRecord] = []

    for sample in samples:
        row = runner.run_one(sample)
        if not seen_first:
            cold_start = float(row.total_latency_ms)
            seen_first = True
        row.cold_start_ms = cold_start
        row.index_size_mb = float(index_size_mb)
        rows.append(row)

        if with_ragas and (not sample.gold_answers) and (not row.error) and row.contexts:
            ragas_records.append(
                RagasRecord(
                    question=sample.question,
                    answer=row.answer,
                    contexts=list(row.contexts),
                    ground_truth="",
                )
            )

    ragas_result: Dict[str, Any] = {"enabled": bool(with_ragas), "status": "disabled"}
    if with_ragas:
        if not str(ragas_base_url or "").strip():
            ragas_result = {
                "enabled": True,
                "status": "skipped",
                "reason": "ragas_base_url_missing",
            }
        elif not ragas_records:
            ragas_result = {
                "enabled": True,
                "status": "skipped",
                "reason": "no_records_without_gold_answer_or_context",
            }
        else:
            ragas_result = evaluate_ragas(
                records=ragas_records,
                model=str(ragas_model),
                base_url=str(ragas_base_url),
                api_key=str(ragas_api_key or ""),
            )

    summary = _summarize_mode(mode, rows, ragas_result)
    return rows, summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Edge LLM + RAG evaluation and generate report artifacts."
    )
    p.add_argument("--questions", default="datasets/questions.jsonl", help="JSONL question set path")
    p.add_argument("--qrels", default="", help="Optional qrels TSV path (query-id, corpus-id, score)")
    p.add_argument("--results-dir", default="results", help="Output directory for report artifacts")
    p.add_argument("--modes", default="no_rag,naive_rag,ours", help="Comma-separated modes")
    p.add_argument("--farm-id", default="farm_eval", help="Default farm_id when absent in questions")
    p.add_argument("--modalities", default="", help="Default modalities (comma-separated: text,table,image)")
    p.add_argument("--naive-top-k", type=int, default=5, help="top_k for naive_rag")
    p.add_argument(
        "--ours-top-k",
        type=int,
        default=int(settings.QUERY_DEFAULT_TOP_K),
        help="top_k for ours (unless per-sample top_k exists)",
    )
    p.add_argument(
        "--metric-unit",
        choices=("doc", "chunk", "both"),
        default="doc",
        help="Retrieval metric unit: canonical doc ids (doc), raw ids (chunk), or both",
    )
    p.add_argument("--seed", type=int, default=42, help="Decoding seed (fixed default for reproducibility)")
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Decoding temperature (forced to 0.0 by default unless overridden)",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=int(settings.LLMLITE_MAX_TOKENS),
        help="Decoding max tokens",
    )
    p.add_argument("--max-questions", type=int, default=0, help="Limit samples (0 = all)")
    p.add_argument(
        "--index-paths",
        default=",".join(DEFAULT_INDEX_PATHS),
        help="Comma-separated index paths for index_size_mb",
    )
    p.add_argument("--with-ragas", action="store_true", help="Enable RAGAS for rows without gold_answer")
    p.add_argument(
        "--ragas-model",
        default=str(os.getenv("OPENAI_COMPAT_MODEL", "openai/gpt-oss-120b")),
        help="Judge model for RAGAS",
    )
    p.add_argument(
        "--ragas-base-url",
        default=str(os.getenv("RAGAS_BASE_URL", "")),
        help="RAGAS judge base URL",
    )
    p.add_argument(
        "--ragas-api-key",
        default=str(os.getenv("RAGAS_API_KEY", "")),
        help="RAGAS judge API key",
    )
    return p.parse_args()


def _resolve_default_dataset_paths(questions: Path, qrels: Optional[Path]) -> tuple[Path, Optional[Path]]:
    resolved_questions = questions
    resolved_qrels = qrels
    used_crop_fallback = False

    if not resolved_questions.exists() and resolved_questions == (REPO_ROOT / "datasets/questions.jsonl"):
        fallback_questions = REPO_ROOT / "smartfarm-search/data/crop/queries.jsonl"
        if fallback_questions.exists():
            resolved_questions = fallback_questions
            used_crop_fallback = True
            print(f"[info] fallback questions dataset: {resolved_questions}")

    if resolved_qrels is None:
        sibling_qrels = resolved_questions.with_name("qrels.tsv")
        if sibling_qrels.exists():
            resolved_qrels = sibling_qrels
        elif used_crop_fallback or resolved_questions == (REPO_ROOT / "smartfarm-search/data/crop/queries.jsonl"):
            crop_qrels = REPO_ROOT / "smartfarm-search/data/crop/qrels/test.tsv"
            if crop_qrels.exists():
                resolved_qrels = crop_qrels
                print(f"[info] fallback qrels: {resolved_qrels}")

    return resolved_questions, resolved_qrels


def main() -> int:
    args = _parse_args()
    modes = _resolve_modes(str(args.modes))
    default_modalities = [m.strip().lower() for m in str(args.modalities or "").split(",") if m.strip()]
    default_modalities = [m for m in default_modalities if m in {"text", "table", "image"}]
    seed = int(args.seed)
    temperature = float(args.temperature)
    max_tokens = max(1, int(args.max_tokens))
    metric_unit = str(args.metric_unit).strip().lower()
    settings.LLMLITE_SEED = int(seed)

    questions_path = _resolve_path(str(args.questions))
    qrels_path = _resolve_path(str(args.qrels)) if str(args.qrels).strip() else None
    questions_path, qrels_path = _resolve_default_dataset_paths(questions_path, qrels_path)
    qrels = _parse_qrels_tsv(qrels_path)
    samples = _load_samples(
        questions_path=questions_path,
        qrels=qrels,
        default_farm_id=str(args.farm_id),
        default_modalities=default_modalities or None,
        max_questions=max(0, int(args.max_questions)),
    )
    if not samples:
        raise RuntimeError("no samples loaded; check questions dataset format/path")

    index_paths = [_resolve_path(p.strip()) for p in str(args.index_paths or "").split(",") if p.strip()]
    index_size_mb = _index_size_mb(index_paths)

    results_dir = _resolve_path(str(args.results_dir))
    results_dir.mkdir(parents=True, exist_ok=True)
    rows_path = results_dir / "eval_rows.csv"
    summary_path = results_dir / "eval_summary.json"
    failures_path = results_dir / "top_failures.md"

    all_rows: List[EvalRow] = []
    mode_summary: Dict[str, Dict[str, Any]] = {}
    started = time.perf_counter()

    for mode in modes:
        mode_rows, summary = _run_mode(
            mode=mode,
            samples=samples,
            naive_top_k=max(1, int(args.naive_top_k)),
            ours_top_k=max(1, int(args.ours_top_k)),
            metric_unit=metric_unit,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            index_size_mb=float(index_size_mb),
            with_ragas=bool(args.with_ragas),
            ragas_model=str(args.ragas_model),
            ragas_base_url=str(args.ragas_base_url),
            ragas_api_key=str(args.ragas_api_key),
        )
        all_rows.extend(mode_rows)
        mode_summary[mode] = summary

    with rows_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(
            EvalRow(
                mode="",
                qid="",
                question="",
                answer="",
                gold_answers=[],
                gold_doc_ids=[],
                gold_canonical_doc_ids=[],
                retrieved_doc_ids=[],
                retrieved_canonical_doc_ids=[],
                retrieved_scores=[],
                exact_match=None,
                f1=None,
                task_accuracy=None,
                recall_at_k=None,
                recall_at_k_doc=None,
                recall_at_k_chunk=None,
                k_used=0,
                ttft_ms=0.0,
                ttft_source="",
                total_latency_ms=0.0,
                retrieve_ms=0.0,
                generate_ms=0.0,
                peak_rss_mb=0.0,
                cold_start_ms=0.0,
                index_size_mb=0.0,
                error="",
            ).to_csv_row().keys()
        )
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row.to_csv_row())

    summary_doc: Dict[str, Any] = {
        "generated_at": _now_iso(),
        "run_elapsed_sec": float(time.perf_counter() - started),
        "input": {
            "questions_path": str(questions_path),
            "qrels_path": str(qrels_path) if qrels_path else "",
            "sample_count": len(samples),
            "modes": modes,
            "naive_top_k": int(args.naive_top_k),
            "ours_top_k": int(args.ours_top_k),
            "index_paths": [str(p) for p in index_paths],
        },
        "runtime_config": {
            "metric_unit": metric_unit,
            "top_k": {
                "naive_rag": int(args.naive_top_k),
                "ours_default": int(args.ours_top_k),
            },
            "seed": seed,
            "backend": {
                "llm_provider": str(getattr(settings, "LLM_PROVIDER", "") or ""),
                "embed_model": str(getattr(settings, "EMBED_MODEL", "") or ""),
                "ragas_model": str(args.ragas_model),
            },
            "decoding": {
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        },
        "device_spec": _device_spec(),
        "modes": mode_summary,
        "delta": {},
    }

    if "ours" in mode_summary and "no_rag" in mode_summary:
        summary_doc["delta"]["ours_minus_no_rag"] = _build_delta(
            mode_summary["ours"], mode_summary["no_rag"], "no_rag"
        )
    if "ours" in mode_summary and "naive_rag" in mode_summary:
        summary_doc["delta"]["ours_minus_naive_rag"] = _build_delta(
            mode_summary["ours"], mode_summary["naive_rag"], "naive_rag"
        )

    summary_path.write_text(json.dumps(summary_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_top_failures(failures_path, all_rows, mode_summary)

    print(f"[ok] rows: {rows_path}")
    print(f"[ok] summary: {summary_path}")
    print(f"[ok] failures: {failures_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
