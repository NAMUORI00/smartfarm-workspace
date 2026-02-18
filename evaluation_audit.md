# Evaluation Audit (EdgeRAG/MobileRAG 관점)

- Audit date: 2026-02-18
- Scope:
  - RAG 온라인 엔트리포인트: `smartfarm-search/core/Api/query.py:22` (`query`)
  - Retrieval 핵심 함수: `smartfarm-search/core/retrieval/service.py:307` (`retrieve_with_diagnostics`)
  - 평가 리포트 엔트리포인트: `experiments/run_eval_report.py:883` (`main`)

## Checklist (PASS / PARTIAL / FAIL)

| 항목 | Status | 근거 코드 (파일/함수) | 판단 요약 |
|---|---|---|---|
| A. Baselines: `no_rag / naive_rag / ours` 비교 존재 | **PASS** | `experiments/run_eval_report.py:34` (`ALLOWED_MODES`), `experiments/run_eval_report.py:649` (`ModeRunner.run_one`), `experiments/run_eval_report.py:955` (`_build_delta` 호출) | 3모드 실행 및 `ours-baseline` 델타 계산이 구현됨. |
| B. Retrieval metrics: recall@k / MRR / nDCG 중 ≥1 + 단위/중복/gold 매핑 명확성 | **PARTIAL** | 지표 구현: `smartfarm-benchmarking/benchmarking/metrics/retrieval_metrics.py:36` (`recall_at_k`), `smartfarm-benchmarking/benchmarking/metrics/retrieval_metrics.py:54` (`mrr`), `smartfarm-benchmarking/benchmarking/metrics/retrieval_metrics.py:87` (`ndcg_at_k`), 사용: `experiments/run_eval_report.py:724`; 중복 처리: `smartfarm-search/core/retrieval/service.py:355`, `smartfarm-search/core/retrieval/service.py:668`; gold 매핑: `experiments/run_eval_report.py:302`, `experiments/run_eval_report.py:393` | 지표 함수는 존재하고 recall@k는 실제 사용됨. 다만 평가 단위가 `hit.id` 기반이며(doc/chunk), ingest ID 스키마와 gold(qrels) 스키마 일치성이 코드에서 강제되지 않아 단위 명확성이 부족함. |
| C. Groundedness: RAGAS 또는 grounding annotation 평가 | **PASS** | `smartfarm-benchmarking/benchmarking/metrics/ragas_adapter.py:22` (`evaluate_ragas`), `experiments/run_eval_report.py:780` (RAGAS record 구성), `experiments/run_eval_report.py:791` (`--with-ragas` 경로) | RAGAS 기반 groundedness 평가 경로가 존재함(옵션 플래그 기반). |
| D. E2E 시스템 지표: TTFT, total latency, p50/p95, peak memory, index size, cold start 기록 | **PARTIAL** | 지표 기록 필드: `experiments/run_eval_report.py:101`, `experiments/run_eval_report.py:103`, `experiments/run_eval_report.py:107`, `experiments/run_eval_report.py:108`; 집계: `experiments/run_eval_report.py:492`, `experiments/run_eval_report.py:497`; 산출: `experiments/run_eval_report.py:908`, `experiments/run_eval_report.py:964`; TTFT fallback: `experiments/run_eval_report.py:708`; 메모리 측정: `experiments/run_eval_report.py:247` | 요구 지표는 모두 기록됨. 그러나 TTFT가 실제 first-token이 아니라 `generate_ms` fallback이며, memory는 evaluator 프로세스 RSS여서 “논문급 E2E 계측 정확도”는 부분 충족. |
| E. 재현성: seed/config 고정 + csv/json 저장 + 1-command 재현 | **PARTIAL** | 결과 저장: `experiments/run_eval_report.py:908`, `experiments/run_eval_report.py:909`, `experiments/run_eval_report.py:964`; 1-command 엔트리포인트: `experiments/run_eval_report.py:883`; seed 반영 조건: `smartfarm-search/core/llm_gateway/local_provider.py:25`, `smartfarm-search/core/llm_gateway/openai_compat_provider.py:50`; 기본 seed: `smartfarm-search/core/Config/Settings.py:163` | 1커맨드 실행과 CSV/JSON 저장은 충족. 다만 seed 기본값이 `-1`로 고정 재현이 보장되지 않고, 평가 실행 시 seed를 강제/기록하는 계약이 부족함. |

## Fix Plan (FAIL/PARTIAL 대상)

### P0
- 평가 단위(doc/chunk) 정규화 계층 추가
  - 작업: retrieved id와 gold id를 직접 비교하지 말고, `canonical_doc_id`로 매핑 후 metric 계산.
  - 근거 위치: `experiments/run_eval_report.py:724`, `experiments/run_eval_report.py:393`, `smartfarm-ingest/pipeline/vector_writer.py:108`, `smartfarm-search/core/retrieval/qdrant_client.py:222`.
  - 산출물: `canonical_id_mapper` 유틸 + `eval_rows.csv`에 `retrieved_canonical_ids` 컬럼.

- 재현성 계약 강화
  - 작업: `--seed` CLI 추가, 실행 시 LLM/embedding 관련 핵심 설정을 summary에 고정 기록.
  - 근거 위치: `experiments/run_eval_report.py:816`, `smartfarm-search/core/Config/Settings.py:163`.
  - 산출물: `eval_summary.json`에 `runtime_config` 블록(모델, seed, top_k, backend).

### P1
- TTFT/메모리 계측 정확도 개선
  - 작업: streaming generation path를 도입해 실제 first-token 시점 측정, 서버 프로세스 RSS/VRAM 계측 분리.
  - 근거 위치: `experiments/run_eval_report.py:708`, `experiments/run_eval_report.py:247`.
  - 산출물: `ttft_source=stream_first_token`, `server_peak_rss_mb` 필드.

- cold start 정의 엄밀화
  - 작업: 모드별 첫 질의 전 프로세스/캐시 초기화 또는 warm/cold를 명시 분리 측정.
  - 근거 위치: `experiments/run_eval_report.py:774`.
  - 산출물: `cold_start_ms`, `warm_p50_ms` 동시 보고.

### P2
- Groundedness 확장
  - 작업: RAGAS 외 annotation 기반 grounding(예: citation support rate) 추가.
  - 근거 위치: `smartfarm-benchmarking/benchmarking/metrics/ragas_adapter.py:22`, `experiments/run_eval_report.py:791`.
  - 산출물: `grounding_support_rate`, `unsupported_claim_rate`.

## 고위험 버그/평가 함정 (최소 3개)

1. **doc/chunk ID 혼용으로 recall 왜곡 위험**
- 위치: `experiments/run_eval_report.py:724`, `experiments/run_eval_report.py:393`, `smartfarm-search/core/retrieval/qdrant_client.py:222`, `smartfarm-ingest/pipeline/vector_writer.py:108`.
- 설명: 평가는 `hit.id`를 그대로 recall@k에 사용하지만, ingest는 `chunk_id`를 point ID로 저장할 수 있음. gold가 doc-level이면 recall이 체계적으로 과소평가될 수 있음.

2. **TTFT가 실제 first-token latency가 아님**
- 위치: `experiments/run_eval_report.py:708`.
- 설명: 현재 TTFT는 `generate_ms`를 fallback으로 기록. 논문 비교 시 “실제 TTFT”보다 과대/과소가 발생할 수 있고, 모델/서빙 방식 간 공정 비교가 어려움.

3. **peak memory가 시스템 E2E 메모리가 아님**
- 위치: `experiments/run_eval_report.py:247`, `experiments/run_eval_report.py:716`.
- 설명: 측정 대상이 evaluator 프로세스 RSS로 제한됨. 실제 병목인 search API/LLM server 메모리 사용량이 누락되어 edge fit 판단을 왜곡할 수 있음.

4. **cold start 측정 오염 가능성**
- 위치: `experiments/run_eval_report.py:774`.
- 설명: mode 첫 요청 latency를 그대로 cold start로 사용. 프로세스 재시작/캐시 플러시 없이 측정되어 진짜 cold start가 아닐 수 있음.

5. **baseline 공정성 위험(no_rag 프롬프트 비대칭)**
- 위치: `experiments/run_eval_report.py:662`, `experiments/run_eval_report.py:684`, `smartfarm-search/core/retrieval/context_builder.py:41`.
- 설명: `no_rag`는 질문 원문만 전달, `naive/ours`는 RAG 전용 지시문(prompt template) 사용. 성능 차이가 retrieval 효과가 아니라 prompt 차이에서 일부 발생할 수 있음.

## 종합 판단

- 현재 구현은 **비교 실험 자동화(3모드 + CSV/JSON/실패리포트)** 관점에서 실무적으로 유용하고, RAGAS/IR metric 기반 확장성도 갖춤.
- 다만 EdgeRAG/MobileRAG 논문 수준의 엄밀성 기준에서는 **평가 단위 정합성(doc/chunk), TTFT·메모리 계측 정의, 재현성(seed 계약)** 이 보강되어야 함.
