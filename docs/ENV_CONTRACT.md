# ERA-SmartFarm Env Contract (Strict Minimal)

본 워크스페이스는 단일 `.env` 기준으로 아래 8개 키만 프로젝트 계약 키로 관리합니다.

## 1) Allowed Keys (8)
- `LLM_BACKEND` (`openai_compatible|llama_cpp`)
- `OPENAI_COMPAT_BASE_URL` (`LLM_BACKEND=openai_compatible`일 때 필수)
- `OPENAI_COMPAT_API_KEY` (`LLM_BACKEND=openai_compatible`일 때 필수)
- `OPENAI_COMPAT_MODEL` (선택)
- `JUDGE_RUNTIME` (`api|self_host`)
- `RAGAS_BASE_URL` (`JUDGE_RUNTIME=api`일 때 필수)
- `RAGAS_API_KEY` (`JUDGE_RUNTIME=api`일 때 필수)
- `HF_TOKEN` (선택)

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
- `JUDGE_RUNTIME=api`인데 `RAGAS_BASE_URL` 또는 `RAGAS_API_KEY`가 없으면 즉시 실패.
- 허용 목록 밖의 프로젝트 키는 무시하지 않고 경고한다.

## 4) Defaults Fixed in Code/Compose
- Qdrant/FalkorDB host/port/collection
- llama host/port/model path
- timeout/retry/batch/perf tuning values

위 항목들은 필요 시 코드 수정으로 변경하며, env 관리 대상에서 제외한다.
