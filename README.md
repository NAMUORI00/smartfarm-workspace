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

## 주요 성과

### 시스템 성능 (엣지 환경)

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Index Build | 233s | 24s | **10x faster** |
| Query Latency | 4-7s | 150-180ms | **25-40x faster** |
| Memory Usage | 3.4GB | 855MB | **4x reduction** |

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
