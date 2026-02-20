# ERA-SmartFarm Env Contract (Strict Minimal + DAT)

본 워크스페이스는 단일 `.env` 기준으로 아래 키를 프로젝트 계약 키로 관리합니다.

## 1) Allowed Keys
- `LLM_BACKEND` (`openai_compatible|llama_cpp`)
- `OPENAI_COMPAT_BASE_URL` (`LLM_BACKEND=openai_compatible`일 때 필수)
- `OPENAI_COMPAT_API_KEY` (`LLM_BACKEND=openai_compatible`일 때 필수)
- `OPENAI_COMPAT_MODEL` (선택)
- `JUDGE_RUNTIME` (`api|self_host`)
- `RAGAS_BASE_URL` (선택, Judge endpoint override)
- `RAGAS_API_KEY` (선택, Judge auth override)
- `HF_TOKEN` (선택, Hugging Face 데이터셋/임베딩 API 사용 시 필요)
- `DAT_MODE` (`static|hybrid`, 기본 `hybrid`)
- `FUSION_PROFILE_PATH` (DAT weights 경로)
- `FUSION_PROFILE_META_PATH` (DAT meta 경로)
- `DAT_MIN_WEIGHT_PER_CHANNEL` (기본 `0.10`)
- `DAT_MAX_WEIGHT_PER_CHANNEL` (기본 `0.80`)
- `DAT_MAX_DELTA_PER_UPDATE` (기본 `0.15`)
- `DAT_MIN_PROFILE_QUERIES` (기본 `300`)
- `DAT_PROFILE_TTL_HOURS` (기본 `168`)

## 2) Removed/Deprecated Keys
- `OPENAI_COMPAT_GRAPH_MODEL`
- `HUGGING_FACE_HUB_TOKEN`
- 프로젝트 내부 fallback 체인용 키:
  - `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `API_KEY`
  - `EXTRACTOR_*`, `EMBED_*`
  - `RAGAS_SELF_HOST_*`, `JUDGE_SELF_HOST_*`
  - `UNSTRUCTURED_*`, `CHUNK_*`, `DOCLING_*`, `SENSOR_*`

## 3) Runtime Validation
- `LLM_BACKEND=openai_compatible`인데 `OPENAI_COMPAT_BASE_URL` 또는 `OPENAI_COMPAT_API_KEY`가 없으면 즉시 실패.
- `JUDGE_RUNTIME=api`에서 Judge endpoint/key 해석 우선순위:
  - 1순위 `RAGAS_BASE_URL`, `RAGAS_API_KEY` (override)
  - 2순위 `OPENAI_COMPAT_BASE_URL`, `OPENAI_COMPAT_API_KEY` (fallback)
- 위 우선순위 적용 후 endpoint/key가 최종 해석되지 않으면 즉시 실패.
- 허용 목록 밖의 프로젝트 키는 무시하지 않고 경고한다.

## 4) DAT Runtime Rules
- 런타임 DAT는 `schema_version=dat` 아티팩트만 허용.
- 아티팩트 구조: `default`, `segments`, `guardrails`, `quality`.
- 품질 게이트(`DAT_MIN_PROFILE_QUERIES`, `DAT_PROFILE_TTL_HOURS`) 미충족 시 `default_fallback`.
- `DAT_MODE=static`이면 DAT 프로파일이 존재해도 `default_fallback`.

## 5) Defaults Fixed in Code/Compose
- Qdrant/FalkorDB host/port/collection
- llama host/port/model path
- timeout/retry/batch/perf tuning values

위 항목들은 필요 시 코드 수정으로 변경하며, env 관리 대상에서 제외한다.

## 6) Judge Runtime Tuning (CLI/Protocol, not ENV)
RAGAS judge 안정화 파라미터는 env 키가 아니라 실행 인자/프로토콜로 관리한다.

- `judge_timeout_sec` (default: `120`)
- `judge_max_retries` (default: `3`)
- `judge_retry_backoff_sec` (default: `2.0`)
- `judge_cooldown_sec` (default: `1.0`)
- `judge_fail_fast_threshold` (default: `8`)
- `judge_max_workers` (default: `1`)

N/A 정책:
- `no_rag`는 `ir.*`가 `not_applicable_no_retrieval`.
- `no_rag`의 context 기반 RAGAS 지표는 `not_applicable_no_context`.
- `N/A`는 0으로 집계하지 않고 랭킹 계산에서 제외한다.
