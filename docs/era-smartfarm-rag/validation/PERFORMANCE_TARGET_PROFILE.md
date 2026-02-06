# 성능 Target 프로파일 (Retrieval-only vs End-to-End)

기준일: 2026-02-06  
적용 범위: `smartfarm-search` + `smartfarm-benchmarking` 배치 평가 체인

## 1) 목적

단일 latency 숫자(예: 150~180ms)만으로는 실제 서비스 상태를 해석하기 어렵다.  
따라서 공식 목표를 아래 두 시나리오로 분리한다.

- Scenario A: Retrieval-only (`OFFLINE_MODE=true`)
- Scenario B: End-to-End (`OFFLINE_MODE=false`, LLM 생성 포함)

## 2) 계측 규약(고정)

- 시간원: client/server 모두 `time.perf_counter()` (monotonic)
- 출력 필드:
  - `latency_client_s`
  - `latency_server_s`
  - `latency_transport_s_raw` (raw 차이: `client - server`)
  - `latency_transport_s` (정규화 값, 0 이상)
  - `timing_source` (`perf_counter`)
- transport 정규화 기준:
  - `TRANSPORT_JITTER_EPSILON_MS=25` (기본값)
  - `abs(raw) <= 25ms`의 음수는 jitter로 분류하고 `latency_transport_s=0`으로 정규화
  - `raw < -25ms`는 hard 음수로 분류
- 품질 플래그:
  - `negative_latency_transport_hard`
  - `negative_latency_transport_jitter`
  - `negative_latency_client/server`
  - `missing_latency_server`, `invalid_latency_*`

관련 코드:

- `smartfarm-search/core/Api/routes_query.py`
- `smartfarm-search/core/Models/Schemas/BaseSchemas.py`
- `smartfarm-benchmarking/benchmarking/utils/experiment_utils.py`
- `smartfarm-benchmarking/benchmarking/experiments/batch_eval_rag.py`

## 3) 공식 목표 (운영값)

### Scenario A: Retrieval-only

- `success_rate >= 0.99`
- `latency_client_p50 <= 0.20s`
- `latency_client_p95 <= 0.50s`
- `latency_client_p99 <= 1.00s`
- `negative_latency_hard_count == 0`
- `negative_latency_jitter_ratio <= 0.01`

### Scenario B: End-to-End

- `success_rate >= 0.99`
- `latency_client_p50 <= 2.50s`
- `latency_client_p95 <= 5.00s`
- `fallback_mode 비율 <= 0.15`
- `negative_latency_hard_count == 0`
- `negative_latency_jitter_ratio <= 0.01`

## 4) 실행 프로토콜

공통:

- 질의셋: `wasabi_qa_dataset.jsonl` (100)
- `top_k=4`, 동일 하드웨어/모델
- warmup 10 후 측정
- 시나리오별 3회 반복
- 공식 결과는 `output/perf_analysis/2026-02-06-closeout/summary.json`의 `official_runs`에 명시된 파일만 사용

Scenario B(e2e) 추가 규칙:

- run 단위로 API를 재기동해 메모리 캐시 오염 제거
- `from_cache` 비율은 공식 판정에서 0.0이어야 함

## 5) 판정 규칙

- 무효 조건: `negative_latency_hard_count > 0`
- jitter는 무효가 아니라 모니터링 지표로 관리 (`<=1%` 권고)
- 목표 판정은 Scenario A/B 각각 독립적으로 수행
- README latency 문구는 반드시 시나리오 태그를 병기

## 6) 2026-02-06 Closeout 스냅샷

공식 판정 run:

- Retrieval-only: `retrieval_only_run1/2/3.json`
- End-to-End: `e2e_strict_run1/2/3.json`
- 통합 요약: `output/perf_analysis/2026-02-06-closeout/summary.json`

결과 요약:

- Retrieval-only 평균: `p50=0.0210s`, `p95=0.1534s`, `success_rate=1.0`, `hard=0`, `jitter=0` (통과)
- End-to-End 평균: `p50=7.5364s`, `p95=15.6204s`, `success_rate=1.0`, `hard=0`, `jitter_ratio=0.01` (latency 목표 미통과)

판정:

- Retrieval-only: 통과
- End-to-End: `p50/p95` 미통과 (계측 무결성 자체는 hard 0으로 충족)

## 7) 참고문헌

- [29] BEIR benchmark (https://arxiv.org/abs/2104.08663)
- [30] Lost in the Middle (https://arxiv.org/abs/2307.03172)
- [31] tiktoken (https://github.com/openai/tiktoken)
