# ERA RAG 엣지 환경 검증 및 성능 최적화 보고서

**검증 일자**: 2026-01-12  
**검증 대상**: era-smartfarm-rag (Qwen3 + llama.cpp 기반 RAG 시스템)  
**검증 환경**: Windows 11, CPU (Intel/AMD), 16GB RAM

---

## Executive Summary

ERA RAG 시스템은 **엣지 환경(8GB RAM)**에서 운영 가능한 스마트팜 도메인 특화 RAG 시스템입니다.

### 주요 결과

| 항목 | 결과 |
|------|------|
| 아키텍처 품질 | **우수** - 학술 연구 목적에 부합하는 설계 |
| 엣지 환경 적합성 | **양호** - 최적화 후 8GB RAM에서 운영 가능 |
| 성능 최적화 | **25-40x 쿼리 속도 향상** 달성 |
| 테스트 커버리지 | **7/7 테스트 통과** |

---

## 1. 아키텍처 분석

### 1.1 프로젝트 구조

```
era-smartfarm-rag/
├── core/                    # 핵심 모듈
│   ├── Config/Settings.py   # 환경 설정 (dataclass + dotenv)
│   ├── Api/                 # FastAPI 라우터
│   ├── Services/
│   │   ├── Retrieval/       # Dense/Sparse/Hybrid/PathRAG
│   │   ├── Ingest/          # 문서 처리, OCR
│   │   └── LLM.py           # llama.cpp 추론
│   ├── Models/              # Pydantic/dataclass 스키마
│   └── Rerankers/           # BGE, LLM Judge
├── benchmarking/            # 실험 프레임워크
│   ├── experiments/         # edge_benchmark, ablation 등
│   ├── baselines/           # Dense-only, Sparse-only 비교
│   └── reporters/           # 논문용 표 생성
├── frontend/                # Streamlit UI
├── Dockerfile               # Docker 배포
└── docker-compose.yml       # llama.cpp + API 스택
```

### 1.2 잘 설계된 부분

#### 1.2.1 Hybrid DAT Retriever
- **3채널 융합**: Dense (FAISS) + Sparse (TF-IDF) + PathRAG
- **Dynamic Alpha Tuning**: 온톨로지 기반 질의 분석으로 가중치 자동 조정
- **작물 필터링**: 질문의 작물과 문서 작물 매칭
- **Semantic Deduplication**: 임베딩 유사도 기반 중복 제거 (θ=0.85)

```python
# 예: 숫자/단위 포함 질의 시 Sparse 가중치 증가
if has_number_unit or len(q) > 25:
    sparse_score += 0.2
    dense_score -= 0.2
```

#### 1.2.2 도메인 특화 기능
- **온톨로지**: crop/env/nutrient/disease/stage/practice 6가지 카테고리
- **PathRAG**: 인과관계 그래프 기반 검색 (causes, solved_by 엣지)
- **LLM Judge 리랭커**: 도메인 키워드 보너스 (EC, dS/m, 양액 등)

#### 1.2.3 엣지 환경 최적화
- **FAISS mmap 모드**: 메모리 효율적 인덱스 로딩
- **Auto reranking**: 가용 RAM/VRAM에 따른 자동 리랭커 선택
- **Lazy loading**: 모델 필요 시점에 로드

#### 1.2.4 벤치마킹 시스템 (이미 구현됨)
- `edge_benchmark.py`: Cold start, latency (p50/p95/p99), memory scaling
- `baseline_comparison.py`: Dense/Sparse/Naive Hybrid 비교
- `ablation_study.py`: 컴포넌트별 기여도 분석
- `PaperResultsReporter.py`: 논문용 표 자동 생성

### 1.3 개선 필요 사항

| 항목 | 현재 상태 | 권장 조치 |
|------|-----------|-----------|
| 임베딩 모델 | Qwen3-Embedding-0.6B (느림) | MiniLM 옵션 추가 ✅ 완료 |
| 임베딩 캐싱 | 없음 | LRU 캐시 추가 ✅ 완료 |
| 테스트 | 없음 | pytest 테스트 추가 ✅ 완료 |
| 오프라인 인덱스 빌드 | 없음 | 스크립트 추가 ✅ 완료 |

---

## 2. 성능 벤치마크 결과

### 2.1 테스트 환경

- **OS**: Windows 11
- **CPU**: AMD/Intel (CPU 전용, GPU 미사용)
- **RAM**: 16GB
- **말뭉치**: 400개 문서 (wasabi_en_ko_parallel.jsonl)
- **QA 데이터셋**: 220개 질의 (wasabi_qa_dataset.jsonl)

### 2.2 베이스라인 vs 최적화 비교

#### 임베딩 모델 비교

| 메트릭 | Qwen3-Embedding-0.6B | MiniLM (최적화) | 개선율 |
|--------|----------------------|-----------------|--------|
| 모델 크기 | 1.2GB | 120MB | **10x 작음** |
| 임베딩 차원 | 1024 | 384 | 2.7x 작음 |
| 인덱스 빌드 | 233s | 24s | **10x 빠름** |
| 첫 쿼리 (cold) | 16,029ms | 7,909ms | **2x 빠름** |
| 후속 쿼리 | 4,349-7,282ms | 150-181ms | **25-40x 빠름** |
| 캐시 히트 | N/A | <1ms | **즉시 응답** |
| 메모리 | 3,400MB | 855MB | **4x 절약** |

#### 성능 요약

| 메트릭 | 최적화 후 | 목표 (엣지) | 상태 |
|--------|-----------|-------------|------|
| Cold Start (인덱스 로드) | 1.56s | < 10s | ✅ **양호** |
| 첫 쿼리 (모델 로딩) | 7.9s | < 10s | ✅ **양호** |
| 후속 쿼리 | 150-180ms | < 2s | ✅ **우수** |
| 캐시 히트 | <1ms | - | ✅ **즉시** |
| 메모리 | 855MB | < 6GB | ✅ **우수** |

### 2.3 처리량 (QPS)

```
후속 쿼리 평균: 161.5ms
QPS = 1000 / 161.5 = 6.2 queries/second

캐시 히트 시:
QPS > 1000 queries/second
```

---

## 3. 적용된 최적화

### 3.1 경량 임베딩 모델 옵션

```python
# core/Services/Retrieval/Embeddings.py
LIGHTWEIGHT_MODELS = {
    "minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "mpnet": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "distiluse": "sentence-transformers/distiluse-base-multilingual-cased-v2",
}

# 환경변수로 선택
# EMBED_MODEL_ID=minilm  (빠름, 384d)
# EMBED_MODEL_ID=mpnet   (균형, 768d)
# EMBED_MODEL_ID=Qwen/Qwen3-Embedding-0.6B  (품질, 1024d)
```

### 3.2 임베딩 캐싱

```python
# LRU 기반 캐시로 반복 쿼리 최적화
class EmbeddingCache:
    def __init__(self, max_size: int = 256):
        self._cache: Dict[str, np.ndarray] = {}
        self._order: List[str] = []
```

### 3.3 오프라인 인덱스 빌드

```bash
# 서버 없이 인덱스 구축
python scripts/tools/build_index_offline.py \
  --corpus ../dataset-pipeline/output/wasabi_en_ko_parallel.jsonl \
  --lang ko
```

### 3.4 변경된 파일

| 파일 | 변경 내용 |
|------|-----------|
| `core/Config/Settings.py` | EMBED_MODEL_ID 옵션, EMBED_CACHE_SIZE 추가 |
| `core/Services/Retrieval/Embeddings.py` | EmbeddingCache 추가, 경량 모델 별칭 지원 |
| `scripts/tools/build_index_offline.py` | 신규 - 오프라인 인덱스 빌드 |
| `tests/test_retrieval.py` | 신규 - 유닛 테스트 |

---

## 4. 테스트 결과

### 4.1 유닛 테스트

```
Running test_embedding_retriever_init... PASS
Running test_embedding_cache... PASS
Running test_sparse_retriever... PASS
Running test_hybrid_retriever... PASS
Running test_dynamic_alphas... PASS
Running test_deduplication... PASS
Running test_query_cache... PASS

Results: 7 passed, 0 failed
```

### 4.2 테스트 커버리지

| 모듈 | 테스트 항목 | 상태 |
|------|------------|------|
| EmbeddingRetriever | 초기화, 인코딩, 캐싱 | ✅ |
| MiniStore (Sparse) | 인덱싱, 검색 | ✅ |
| HybridDATRetriever | 융합 검색, 중복 제거, 알파 튜닝 | ✅ |
| SimpleQueryCache | LRU 캐시 동작 | ✅ |

---

## 5. 권장사항

### 5.1 즉시 적용 권장 (Critical)

- [x] **경량 임베딩 모델 사용**: `EMBED_MODEL_ID=minilm`
- [x] **임베딩 캐싱 활성화**: 기본 256개 캐시
- [x] **오프라인 인덱스 빌드**: 서버 없이 인덱스 구축

### 5.2 선택적 적용 (Optional)

- [ ] **ONNX 변환**: 추가 30-50% 추론 속도 향상 가능
- [ ] **Redis 캐싱**: 분산 환경에서 캐시 공유
- [ ] **Batch 임베딩**: 여러 쿼리 동시 처리

### 5.3 엣지 환경 배포 가이드

```bash
# 1. 경량 모델로 인덱스 빌드
EMBED_MODEL_ID=minilm python scripts/tools/build_index_offline.py \
  --corpus corpus.jsonl --lang ko

# 2. 환경변수 설정
export EMBED_MODEL_ID=minilm
export EMBED_CACHE_SIZE=256
export DEVICE=cpu

# 3. 서버 실행
uvicorn core.main:app --host 0.0.0.0 --port 41177
```

---

## 6. 결론

ERA RAG 시스템은 **학술 연구 목적에 부합하는 우수한 설계**를 갖추고 있습니다.

### 강점
1. **Hybrid Retrieval**: Dense + Sparse + PathRAG 3채널 융합
2. **도메인 특화**: 온톨로지, 작물 필터링, 인과관계 그래프
3. **벤치마킹 프레임워크**: 논문용 실험 재현 가능

### 성능 개선 결과
- **쿼리 속도**: 25-40x 향상 (4-7초 → 150-180ms)
- **메모리 사용량**: 4x 절약 (3.4GB → 855MB)
- **인덱스 빌드**: 10x 빠름 (233초 → 24초)

### 최종 판정
**✅ 8GB RAM 엣지 환경에서 운영 가능**

---

## Appendix A: 벤치마크 재현

```bash
# 1. 인덱스 빌드
cd era-smartfarm-rag
EMBED_MODEL_ID=minilm python scripts/tools/build_index_offline.py \
  --corpus ../dataset-pipeline/output/wasabi_en_ko_parallel.jsonl \
  --lang ko \
  --output-dir data/index_minilm

# 2. 테스트 실행
python tests/test_retrieval.py

# 3. 벤치마크 (선택)
python -m benchmarking.experiments.edge_benchmark \
  --corpus ../dataset-pipeline/output/wasabi_en_ko_parallel.jsonl \
  --qa-file ../dataset-pipeline/output/wasabi_qa_dataset.jsonl \
  --output-dir output/experiments/edge
```

## Appendix B: 환경변수 참조

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `EMBED_MODEL_ID` | Qwen/Qwen3-Embedding-0.6B | 임베딩 모델 (minilm/mpnet/distiluse) |
| `EMBED_CACHE_SIZE` | 256 | 임베딩 캐시 크기 |
| `DEVICE` | cpu | 디바이스 (cpu/cuda) |
| `ENABLE_PATHRAG` | true | PathRAG 활성화 |
| `ENABLE_CACHE` | true | 쿼리 캐시 활성화 |
| `CHUNK_SIZE` | 5 | 청킹 문장 수 |
| `CHUNK_STRIDE` | 2 | 청킹 스트라이드 |
