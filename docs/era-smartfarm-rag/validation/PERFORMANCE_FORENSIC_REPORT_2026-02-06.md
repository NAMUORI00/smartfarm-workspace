# Performance Forensic Report (2026-02-06, Closeout)

## 1) 요약

이번 조치의 목적은 성능 튜닝 이전에 **계측 신뢰성 체인 복구**를 완료하는 것이다.

주요 조치:

1. benchmark 엔트리포인트를 tracked `.py` 소스로 복구
2. client/server latency 시간원을 `perf_counter`로 통일
3. `transport`를 raw/normalized로 분리
4. `negative_latency_transport`를 `hard`/`jitter`로 분리 집계
5. e2e 측정 시 run 단위 API 재기동으로 메모리 캐시 오염 제거

## 2) 코드 변경 내역

### 2.1 smartfarm-search

- `smartfarm-search/core/Api/routes_query.py`
  - `time.perf_counter()` 기반 계측
  - `debug_timing=true` 시 `diagnostics` 반환
- `smartfarm-search/core/Models/Schemas/BaseSchemas.py`
  - `QueryRequest.debug_timing: bool = False`
  - `QueryResponse.diagnostics: Optional[Dict[str, Any]]`
- `smartfarm-search/tests/test_query_timing_diagnostics.py`
  - debug on/off 스키마 검증

### 2.2 smartfarm-benchmarking

- `smartfarm-benchmarking/benchmarking/utils/experiment_utils.py`
  - `latency_transport_s_raw` 추가
  - `latency_transport_s`는 0 이상 정규화
  - `TRANSPORT_JITTER_EPSILON_MS=25` 적용
- `smartfarm-benchmarking/benchmarking/experiments/batch_eval_rag.py`
  - quality flag를 `negative_latency_transport_hard/jitter`로 분리
  - `negative_latency_hard_count`, `negative_latency_jitter_count` 집계
- `smartfarm-benchmarking/.dockerignore`
  - `__pycache__/`, `.pytest_cache/`, `.cache/`, `output/` 차단
- 테스트
  - `benchmarking/tests/test_experiment_utils_timing.py`
  - `benchmarking/tests/test_batch_eval_rag_entrypoint.py`

## 3) 검증 결과

### 3.1 단위/회귀 테스트

성공:

```bash
/tmp/codex-venv/bin/python -m pytest -s -q \
  tests/test_query_timing_diagnostics.py \
  tests/test_offline_fallback.py \
  tests/test_chunking.py
# 19 passed
```

```bash
/tmp/codex-venv/bin/python -m pytest -s -q \
  benchmarking/tests/test_experiment_utils_timing.py \
  benchmarking/tests/test_batch_eval_rag_entrypoint.py \
  benchmarking/tests/test_chunking_sweep_k_consistency.py
# 7 passed
```

### 3.2 정적 검증

성공:

- `py_compile` (변경 파일 대상)
- `python -m benchmarking.experiments.batch_eval_rag --help`
- `docker compose -f docker-compose.yml -f docker-compose.research.yml config`
- `docker compose -f docker-compose.yml -f docker-compose.research.yml -f docker-compose.e2e.yml config`

### 3.3 성능 실측 (Closeout 공식값)

공식 집계 경로:

- `output/perf_analysis/2026-02-06-closeout/summary.json`

공식 run:

- Retrieval-only: `retrieval_only_run1/2/3.json`
- End-to-End: `e2e_strict_run1/2/3.json`

보조/진단 run:

- `e2e_fresh_run*`, `e2e_run*`, `e2e_nocache_run*` 등은 캐시/프로토콜 진단용이며 공식 판정에서 제외

결과:

| Scenario | Mean p50 (s) | Mean p95 (s) | Success | Cache Ratio | Hard Negatives | Jitter Ratio | 판정 |
|---|---:|---:|---:|---:|---:|---:|---|
| Retrieval-only | 0.0210 | 0.1534 | 1.000 | 0.000 | 0 | 0.000 | 통과 |
| End-to-End | 7.5364 | 15.6204 | 1.000 | 0.000 | 0 | 0.010 | latency 미통과 |

해석:

- 계측 무결성 관점: `negative_latency_hard_count == 0` 달성
- End-to-End 성능 관점: 목표(`p50<=2.5s`, `p95<=5.0s`) 미달

## 4) 재현 커맨드

```bash
# benchmark 엔트리포인트 확인
cd smartfarm-benchmarking
/tmp/codex-venv/bin/python -m benchmarking.experiments.batch_eval_rag --help

# Retrieval-only 예시
OFFLINE_MODE=true /tmp/codex-venv/bin/python -m benchmarking.experiments.batch_eval_rag \
  --host http://127.0.0.1:41177 \
  --input ../smartfarm-ingest/output/wasabi_qa_dataset.jsonl \
  --ranker none --top_k 4 --limit 100 --debug-timing \
  --scenario retrieval_only \
  --output ../output/perf_analysis/2026-02-06-closeout/retrieval_only_run1.json

# End-to-End strict 예시 (run마다 API 재기동 후 실행)
OFFLINE_MODE=false /tmp/codex-venv/bin/python -m benchmarking.experiments.batch_eval_rag \
  --host http://127.0.0.1:41177 \
  --input ../smartfarm-ingest/output/wasabi_qa_dataset.jsonl \
  --ranker none --top_k 4 --limit 100 --sleep-s 0.0 --debug-timing \
  --scenario e2e \
  --output ../output/perf_analysis/2026-02-06-closeout/e2e_strict_run1.json
```

## 5) 결론

1. 계측 체인 복구 목표는 달성(hard negative 0, source/실행 경로 정합성 확보).
2. 운영 목표 관점에서 End-to-End latency는 추가 튜닝이 필요.
3. 다음 단계는 성능 튜닝(생성 경로 병목 축소, 프롬프트/토큰/모델 설정 최적화)이다.

## 6) 참고문헌

- [29] BEIR (https://arxiv.org/abs/2104.08663)
- [30] Lost in the Middle (https://arxiv.org/abs/2307.03172)
