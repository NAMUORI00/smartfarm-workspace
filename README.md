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
- **검색(Retrieval)**: Base+Overlay를 함께 사용하며 Dense/Sparse/TriGraph에 더해 TagHash/CausalGraph 채널을 추가하고 weighted RRF로 융합합니다.
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

---

## 의사결정 추적

- 튜닝 프로토콜/실행 커맨드/선택 규칙:
  `docs/era-smartfarm-rag/validation/CHUNKING_GLEANING_TUNING.md`
- 성능 계측 포렌식/목표 프로파일:
  `docs/era-smartfarm-rag/validation/PERFORMANCE_FORENSIC_REPORT_2026-02-06.md`,
  `docs/era-smartfarm-rag/validation/PERFORMANCE_TARGET_PROFILE.md`
- 참고문헌 원본(번호 기준):
  `docs/era-smartfarm-rag/paper/references.md`

---

## 주요 성과

### 시스템 성능 (엣지 환경)

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Index Build | 233s | 24s | **10x faster** |
| Query Latency (Retrieval-only p50) | 4-7s | 21ms | **190-330x faster** |
| Memory Usage | 3.4GB | 855MB | **4x reduction** |

주의: 위 latency 수치는 retrieval-only 기준입니다. End-to-End(LLM 포함) 성능 목표/해석 기준은 `docs/era-smartfarm-rag/validation/PERFORMANCE_TARGET_PROFILE.md`를 따릅니다.

2026-02-06 closeout 실측 스냅샷(non-cache, 100문항×3회):
- Retrieval-only 평균: `p50=21.0ms`, `p95=153.4ms` (목표 통과)
- End-to-End 평균: `p50=7.536s`, `p95=15.620s` (latency 목표 미통과, 계측 hard 오류 0)
- 상세 리포트: `docs/era-smartfarm-rag/validation/PERFORMANCE_FORENSIC_REPORT_2026-02-06.md`

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
| `smartfarm-frontend/` | Streamlit 프론트엔드 UI | [smartfarm-frontend](https://github.com/NAMUORI00/smartfarm-frontend) |
| `docs/` | 통합 문서 (논문, 검증 보고서, 데이터셋 카드) | - |

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
python setup.py --mode local    # 의존성 설치 + GGUF 모델 다운로드
make build && make up           # Docker 빌드 및 실행
```

개발 모드 (Docker 없이):
```bash
cd smartfarm-search
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

자세한 문서 구조는 [docs/README.md](docs/README.md) 참조.

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
