# SmartFarm Project Rules

## LLM Runtime Contract — 절대 규칙

프로젝트의 LLM 런타임 구성은 다음 원칙을 **반드시** 준수합니다.
`.env` 또는 코드를 수정할 때 이 규칙을 위반하지 마세요.

| # | 역할 | 공급자 | 모델 | 환경변수 |
|:--|:-----|:------|:-----|:---------|
| 1 | **답변 생성** (Answer Generation) | Featherless AI | `Qwen/Qwen3-4B` (4-bit 양자화) | `ANSWER_OPENAI_COMPAT_*` |
| 2 | **RAGAS 저지** (Judge/Evaluator) | Vertex AI | `openai/gpt-oss-120b-maas` | `RAGAS_BASE_URL`, `RAGAS_API_KEY` |
| 3 | **그래프 엔티티 추출** (Extraction) | Vertex AI | `openai/gpt-oss-120b-maas` | `OPENAI_COMPAT_*` |
| 4 | **임베딩** (Embedding) | 로컬 | `BAAI/bge-m3` (sentence-transformers, 1024d) | `EMBED_BACKEND=sentence_transformers_local` |

### ⚠️ 금지 사항
- `ANSWER_OPENAI_COMPAT_*` 값을 Vertex AI 엔드포인트로 변경 금지
- 임베딩을 원격 API로 변경 금지
- **과거 사고**: RAGAS 429 에러 해결 과정에서 `ANSWER_OPENAI_COMPAT_*`를 Vertex AI로 변경하여 답변 생성 모델이 `gpt-oss-120b-maas`로 바뀌는 사고 발생. 429 에러 시에도 **ANSWER_* 변수는 Featherless를 유지**하고, Judge만 Vertex AI를 사용하는 분리 구조를 유지할 것.

### `resolve_answer_runtime()` 우선순위
```
Priority 1: ANSWER_OPENAI_COMPAT_*  (명시적 오버라이드)
Priority 2: OPENAI_COMPAT_*         (하위 호환 fallback — 현재 Vertex AI이므로 주의)
Priority 3: default = Qwen/Qwen3-4B + local llama.cpp
```
