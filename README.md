# SmartFarm Workspace

엣지 환경용 도메인 특화 RAG 시스템 개발을 위한 통합 워크스페이스

## 프로젝트 개요

본 프로젝트는 **저사양 엣지 디바이스(Jetson, 8GB RAM)**에서 동작하는 스마트팜 도메인 특화 RAG(Retrieval-Augmented Generation) 시스템을 개발합니다.

### 핵심 기술

| 기술 | 설명 |
|------|------|
| **LightRAG** | Dual-Level 그래프 검색 (Entity + Community) |
| **Adaptive Hybrid** | Query Specificity 기반 적응형 라우팅 (Dense/BM25/RRF) |
| **Qwen3 + llama.cpp** | GGUF 양자화 모델로 엣지 환경 추론 |

---

## EdgeKG v3.2 아키텍처(의도 요약)

본 워크스페이스의 최근 방향은 **Asymmetric RAG**(Base + Overlay)로 정리됩니다.

- **Base KB (`base.sqlite`)**: 공개 문서를 대상으로 SOTA LLM로 엔티티/인과관계까지 “컴파일”한 결과를 SQLite(SoT)로 번들링해 현장에 전달합니다.  
  - Base 컴파일의 청킹/추출 안정화 기본값은 **LightRAG 스타일**(token chunking 1200/overlap 100, gleaning 1)을 참조합니다.
- **Overlay KB (`overlay.sqlite`)**: 센서/메모/업로드 등 온프레미스 민감 입력은 로컬 llama.cpp로 구조화 추출 후 Overlay에만 저장·갱신합니다.
- **검색(Retrieval)**: Base+Overlay를 함께 사용하며 Dense/Sparse/TriGraph 3채널을 weighted RRF로 융합합니다.
- **인덱스는 캐시**: FAISS/CSR/mmap 기반 인덱스·그래프 스냅샷은 파생물(캐시)로 취급하고, `base.sqlite + overlay.sqlite`만으로 재생성 가능하게 유지합니다.

---

## 연구 근거 기반 기본값

현재 기본 튜닝 baseline은 아래와 같이 고정합니다.

- `CHUNK_METHOD=token`
- `CHUNK_TOKEN_SIZE=1200`
- `CHUNK_TOKEN_OVERLAP=100`
- `KBUPDATE_MAX_GLEANINGS=1`

선정 근거:

- [4] LightRAG 기본 설정과 실무 운용 사례를 baseline으로 채택 (https://arxiv.org/abs/2410.05779)
- [3] GraphRAG의 chunking/graph extraction 설정 가능성 및 튜닝 필요성 (https://arxiv.org/abs/2404.16130)
- [30] long-context 위치 편향(Lost in the Middle)으로 인해 “무조건 큰 chunk” 전략을 지양 (https://arxiv.org/abs/2307.03172)
- [31] `tiktoken` 기반 토큰 길이 정규화, 미설치 시 문자 기반 fallback 사용 (https://github.com/openai/tiktoken)

## LLM 라우팅 기본 정책 (RAG vs Judge 분리)

- **RAG 답변 모델**: `Qwen/Qwen3-4B` (OpenAI-compatible, Featherless)
- **Graph 인게스트 추출 모델**: `moonshotai/Kimi-K2.5` (OpenAI-compatible, Featherless)
- **Judge 모델(RAGAS)**: `openai/gpt-oss-120b` 고정
- **권장 env**:
  - `LLM_BACKEND=openai_compatible`
  - `OPENAI_COMPAT_BASE_URL=https://api.featherless.ai/v1`
  - `OPENAI_COMPAT_MODEL=Qwen/Qwen3-4B`
  - `OPENAI_COMPAT_GRAPH_MODEL=moonshotai/Kimi-K2.5`
  - `OPENAI_COMPAT_GRAPH_MAX_RETRIES=2`
  - `OPENAI_BASE_URL=https://api.featherless.ai/v1`
  - `RAGAS_MODEL=openai/gpt-oss-120b`

엣지 배포 시 전환(로컬 llama.cpp):
- `LLM_BACKEND=llama_cpp`
- `LLMLITE_HOST=http://localhost:45857`
- (선택) `LLMLITE_MODEL=` (빈 값이면 로컬 로드 GGUF 사용)

---

## 의사결정 추적

- 현재 워크스페이스 문서는 **논문 초안 중심**으로 유지합니다.
- 섹션 초안: `docs/era-smartfarm-rag/paper/00_abstract.md` ~ `docs/era-smartfarm-rag/paper/06_conclusion.md`
- 참고문헌 원본(번호 기준): `docs/era-smartfarm-rag/paper/references.md`
- `03/04/05`는 의도적으로 `추후 작성` 상태입니다.

### Graph-KB 검증 실행 원칙 (현재)

- 비교군: `dense_only`, `bm25_only`, `rrf`, `trigraph_only`, `ours_structural`
- 평가지표 계산은 RAGAS 공식 라이브러리 기준으로 유지
- 구조 수정은 검색/그래프 경로에 한정하고, 평가 코드 자체는 변경하지 않음
- 데이터셋: `agxqa(test)`, `2wiki(validation)`, `scifact(BEIR-style)`

---

## 주요 성과

### 시스템 성능 (엣지 환경)

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Index Build | 233s | 24s | **10x faster** |
| Query Latency (Retrieval-only p50) | 4-7s | 21ms | **190-330x faster** |
| Memory Usage | 3.4GB | 855MB | **4x reduction** |

주의: 위 latency 수치는 retrieval-only 기준입니다. 최신 수치는 실행 시점 산출물(`output/`)을 기준으로 해석합니다.

### 데이터셋 통계

- **Corpus**: 402개 청크 (bilingual EN-KO)
- **QA Pairs**: 220개 (RAGEval 방법론)
- **Quality Score**: 0.67 (LLM-as-a-Judge)
- **Diversity**: 0.93 (ROUGE-L)

---

## 프로젝트 구성

| 폴더 | 설명 | 저장소 |
|------|------|--------|
| `smartfarm-search/` | 검색/RAG API 서버 (엣지 환경용) | [smartfarm-search](https://github.com/NAMUORI00/smartfarm-search) |
| `smartfarm-ingest/` | 데이터 인게스트/오프라인 인덱싱 + Dataset pipeline | [smartfarm-ingest](https://github.com/NAMUORI00/smartfarm-ingest) |
| `smartfarm-benchmarking/` | 벤치마크/실험 코드 | [smartfarm-benchmarking](https://github.com/NAMUORI00/smartfarm-benchmarking) |
| `smartfarm-llm-inference/` | LLM 추론(gguf/llama.cpp) 서비스 | [smartfarm-llm-inference](https://github.com/NAMUORI00/smartfarm-llm-inference) |
| `smartfarm-frontend/` | Streamlit 프론트엔드 UI (정책 잠금: 기본 비활성) | [smartfarm-frontend](https://github.com/NAMUORI00/smartfarm-frontend) |
| `docs/` | 통합 문서 (논문, 검증 보고서, 데이터셋 카드) | - |

### Frontend 잠금 정책

- 기본값: `FRONTEND_LOCKED=1` (frontend 실행/호출 차단)
- 예외 해제: `FRONTEND_LOCKED=0` 또는 `ALLOW_FRONTEND_UNLOCK=1`
- Compose 기본 실행은 `frontend` 서비스 프로파일(`ui`)을 포함하지 않습니다.
- 커밋/푸시 차단 훅 설치: `./scripts/policy/install_hooks.sh`

---

## Quick Start

### 클론

```bash
git clone --recurse-submodules https://github.com/NAMUORI00/smartfarm-workspace.git
cd smartfarm-workspace
```

기존 클론 후 서브모듈 가져오기:
```bash
git submodule update --init --recursive
```

### RAG 서버 실행

```bash
cd smartfarm-search
python setup.py --mode local    # 의존성 설치 + 모델 준비
uvicorn core.main:app --port 41177 --reload
```

### Dataset Pipeline 실행

```bash
cd smartfarm-ingest
pip install -r requirements.txt
pip install -e ".[dev]"
python tests/test_pipeline.py   # Smoke test
```

---

## 문서 안내

모든 문서는 워크스페이스 루트의 `docs/` 디렉토리에서 관리됩니다.

| 카테고리 | 경로 | 주요 문서 |
|----------|------|-----------|
| **논문** | `docs/era-smartfarm-rag/paper/` | 섹션별 초안 (00~06), 테이블, 그림 |
| **검증 보고서** | `docs/era-smartfarm-rag/validation/` | ERA_RAG_VALIDATION_REPORT.md |
| **데이터셋** | `docs/dataset-pipeline/dataset/` | DATASET_CARD.md, LIMITATIONS.md |
| **배포 가이드** | `docs/era-smartfarm-rag/deployment/` | PIPELINE.md |

문서 인덱스 파일(`docs/README.md`)은 현재 유지하지 않으며, `docs/era-smartfarm-rag/paper/`를 기준으로 관리합니다.

---

## 워크플로우

### 서브모듈 작업 후 동기화

```bash
# 1. 각 서브모듈에서 작업 및 커밋
cd smartfarm-search
git add . && git commit -m "Update" && git push

# 2. 워크스페이스에서 서브모듈 참조 업데이트
cd ..
git add smartfarm-search
git commit -m "chore: bump smartfarm-search submodule"
git push
```

### 서브모듈 최신 버전으로 업데이트

```bash
git submodule update --remote --merge
```

---

## 개별 프로젝트 사용

워크스페이스 없이 각 프로젝트를 독립적으로 사용할 수 있습니다:

```bash
# RAG 서버만
git clone https://github.com/NAMUORI00/smartfarm-search.git

# Ingest/Dataset 파이프라인만
git clone https://github.com/NAMUORI00/smartfarm-ingest.git

# LLM 추론만
git clone https://github.com/NAMUORI00/smartfarm-llm-inference.git

# Frontend만
git clone https://github.com/NAMUORI00/smartfarm-frontend.git

# Benchmarking만
git clone https://github.com/NAMUORI00/smartfarm-benchmarking.git
```

---

## 최근 진행 상황

### 논문 작업
- IEEE Access 형식 LaTeX 템플릿 적용
- 실험 섹션: BEIR validation, Adaptive Hybrid vs LightRAG 비교

### 주요 실험
- **BEIR Benchmark**: 외부 데이터셋에서 Adaptive Hybrid 라우팅 검증
- **LightRAG 평가**: Dual-Level 그래프 검색 성능 측정
- **RAGAS 평가**: 데이터셋 품질 자동 평가
