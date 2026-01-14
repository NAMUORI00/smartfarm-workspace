# ERA-SmartFarm-RAG 아키텍처 상세 기술 사양

> **목적**: 논문 본문에서 다루기 어려운 상세 기술 사양, 구현 세부사항, 최신 연구 동향을 정리

---

## 1. 코드 ↔ Figure 매핑

| Figure 요소 | 소스 코드 위치 |
|-------------|---------------|
| Layer 0 Runtime | `core/Config/Settings.py` |
| Layer 1 Knowledge Store | `core/Api/deps.py` (인덱스 초기화) |
| Layer 2 HybridRetriever | `core/Services/Retrieval/Hybrid.py` |
| Layer 2 PathRAG | `core/Services/Retrieval/PathRAG.py` |
| Layer 3 Context Shaping | `Hybrid.py` (`_deduplicate`, `_apply_crop_filter`) |
| Layer 4 Generation | `core/Services/LLM.py`, `PromptTemplates.py` |
| Layer 5 API | `core/Api/routes_query.py` |
| Fallback | `ResponseCache.py`, `TemplateResponder.py` |
| Graph Building | `core/Services/Ingest/GraphBuilder.py` |
| Ontology | `core/Services/Ontology.py` |

---

## 2. HybridDATRetriever 구현 상세

### 2.1 주요 파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `DEDUP_THRESHOLD` | 0.85 | 시맨틱 중복 판단 임계값 |
| `CROP_MATCH_BONUS` | 0.5 | 작물 일치 시 스코어 보너스 |
| `CROP_MISMATCH_PENALTY` | 0.85 | 작물 불일치 시 패널티 계수 |

### 2.2 Dynamic Alpha 계산 (`dynamic_alphas`)

- 온톨로지 매칭 결과와 수치/단위 패턴을 분석하여 3채널 가중치 반환
- LLM 호출 없이 규칙 기반으로 동작하여 엣지 환경 최적화

---

## 3. PathRAG-lite 구현 상세

### 3.1 그래프 구조

```
                    ┌─────────────────────────────────────────────┐
                    │          SmartFarm Knowledge Graph          │
                    └─────────────────────────────────────────────┘
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        │                                │                                │
        ▼                                ▼                                ▼
┌───────────────┐                ┌───────────────┐                ┌───────────────┐
│  Concept Node │                │ Practice Node │                │  Edge Types   │
│               │                │  (Document)   │                │               │
│ crop:와사비   │  recommended   │ chunk_001     │   causes       │ recommended   │
│ env:온도      │◄──────────────►│ chunk_002     │◄──────────────►│ associated    │
│ disease:연부병│  for           │ chunk_003     │   solved_by    │ mentions      │
│ nutrient:양액 │                │               │                │ causes        │
│ stage:생육    │                │ metadata:     │                │ solved_by     │
│ practice:차광 │  associated    │  causal_role: │                │               │
│               │◄──────────────►│  cause/effect │                │               │
└───────────────┘  with          │  /solution    │                └───────────────┘
                                 └───────────────┘
```

### 3.2 탐색 전략

- 시작점: 쿼리에서 매칭된 온톨로지 개념 노드
- 최대 깊이: 2-hop (기본값)
- 인과관계 엣지(`causes`, `solved_by`) 우선 탐색

---

## 4. 메모리 적응형 리랭킹 상세

### 4.1 설정 파라미터

- `AUTO_RERANK_MIN_RAM_GB`: 0.8
- `AUTO_BGE_MIN_RAM_GB`: 1.5
- `AUTO_BGE_MIN_VRAM_GB`: 1.5 (GPU 사용 시)

### 4.2 Reranker 선택 로직

```
┌─────────────────────────────────────────────────────────────┐
│              Runtime Memory Check                            │
│                                                              │
│   RAM = _available_ram_gb()                                  │
│   VRAM = _available_vram_gb()                                │
└──────────────────────┬──────────────────────────────────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
         ▼             ▼             ▼
   ┌──────────┐  ┌──────────┐  ┌──────────┐
   │ RAM < 0.8│  │0.8 ≤ RAM │  │RAM ≥ 1.5 │
   │   GB     │  │  < 1.5GB │  │   GB     │
   └────┬─────┘  └────┬─────┘  └────┬─────┘
        │             │             │
        ▼             ▼             ▼
   ┌──────────┐  ┌──────────┐  ┌──────────┐
   │   none   │  │ LLM-lite │  │   BGE    │
   │          │  │llama.cpp │  │BAAI/bge- │
   │ (skip)   │  │ rerank   │  │reranker  │
   │          │  │ (~0MB)   │  │ (~500MB) │
   └──────────┘  └──────────┘  └──────────┘
```

---

## 5. 인덱스 영속화

오프라인 환경 지원을 위해 인덱스를 파일로 저장/로드:

| 파일 | 내용 | 형식 |
|------|------|------|
| `dense.faiss` | 문서 임베딩 벡터 인덱스 | Binary (mmap 가능) |
| `dense_docs.jsonl` | 문서 텍스트 및 메타데이터 | JSON Lines |
| `sparse.pkl` | TF-IDF 키워드 빈도 행렬 | Pickle |

---

## 6. 최신 Edge RAG 아키텍처 패턴 (2024-2025)

### 6.1 MobileRAG 아키텍처 (Park et al., 2025)

**핵심 구성: EcoVector + SCR (Selective Content Reduction)**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    EcoVector 2-Layer Indexing                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Level 1: Centroids Graph (RAM-resident, 50-200MB)                   │
│  ├─ k-means clustering of entire vector corpus                       │
│  ├─ HNSW graph built ONLY on cluster centroids                       │
│  └─ Always in memory for query routing                               │
│                                                                      │
│  Level 2: Inverted Lists Graph (Disk-based)                          │
│  ├─ Independent HNSW graph per cluster                               │
│  ├─ Stored on flash storage                                          │
│  └─ Loaded on-demand → Immediately unloaded after search             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

Search Flow:
Query → Embed → Centroids Graph (RAM) → Identify n_P centroids
                      ↓
    Load n_P cluster graphs from disk → Search → Unload → Return top-k
```

**SCR (토큰 절감) 알고리즘**:
- Retrieved Docs → Sentence Split → Sliding Window (3-5 sentences)
- Query Similarity Scoring → Select Top Windows → Merge
- Token Reduction: ~42% average

### 6.2 EdgeRAG 아키텍처 (Seemakhupt et al., 2024)

**핵심: 온라인 인덱싱 + 적응형 캐싱**

```
┌─────────────────────────────────────────────────────────────────────┐
│                   EdgeRAG: Compute vs Data Tradeoff                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Level 1: IVF Centroids (Always in RAM)                              │
│  └─ Query routing to cluster                                         │
│                                                                      │
│  Level 2: Cluster Embeddings (Selective)                             │
│  ├─ Heavy clusters (gen_time > SLO): Pre-computed & stored           │
│  ├─ Light clusters (gen_time < SLO): Discard, generate on-demand     │
│  └─ Adaptive cache: Cost-aware LFU replacement                       │
│                                                                      │
│  Cache Policy:                                                       │
│  evict_score = generation_latency × access_count                     │
│  → Keep expensive-to-regenerate, frequently-accessed clusters        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

Results (Jetson Orin Nano 8GB):
- TTFT: 1.8× faster vs baseline IVF
- Datasets >8GB fit without thrashing
```

### 6.3 Edge RAG 아키텍처 패턴 비교

| 패턴 | RAM 요구 | 인덱스 전략 | 토큰 절감 | 런타임 |
|------|---------|------------|----------|--------|
| **Pattern A: MobileRAG** | 4-7GB | EcoVector (클러스터 계층) | SCR (42%↓) | AI Edge/MLX |
| **Pattern B: EdgeRAG** | 6-9GB | IVF + 적응형 캐시 | 없음 | llama.cpp |
| **Pattern C: Minimal** | 2-3GB | Binary PQ + mmap | KV 양자화 | NanoLLM |
| **ERA-SmartFarm-RAG** | 0.8-1.5GB | FAISS mmap | Semantic Dedup + Crop Filter | llama.cpp |

---

## 7. 양자화 전략 상세

### 7.1 llama.cpp GGUF 양자화 수준

| 양자화 | 모델 크기 (7B) | 품질 | 속도 | 권장 환경 |
|--------|---------------|------|------|----------|
| Q8_0 | ~7.5GB | Near FP16 | 1.0× | 서버 |
| **Q4_K_M** | ~4.5GB | Good | 1.5× | **8GB RAM 엣지 (권장)** |
| IQ3_M | ~3.5GB | Moderate | 2.0× | 4GB RAM 엣지 |
| IQ2_M | ~2.5GB | Degraded | 2.5× | 극저사양 |

### 7.2 KV 캐시 메모리 최적화

```
7B Q4_K_M 메모리 분해:
├─ 모델 가중치: ~4.5GB
├─ KV 캐시 (ctx=8192): ~2.8GB  ← 주요 병목!
├─ 활성화: ~1.0GB
└─ 런타임 오버헤드: ~0.3GB
────────────────────────────────
총계: ~8.6GB

최적화: ctx=4096으로 줄이면 → ~6.0GB (KV 캐시 절반)
```

### 7.3 최신 KV 캐시 양자화 기법

- **AQUA-KV** (arXiv:2501.19392): 2-2.5비트에서 near-lossless
- **KeyDiff** (arXiv:2504.15364): 키 유사도 기반 KV eviction

---

## 8. FAISS 메모리 최적화 기법

### 8.1 Product Quantization (PQ) - 고압축

```
원본 벡터 (768 dims, FP16): 1536 bytes
       ↓
PQ 압축 (m=12, code_size=8): 12 bytes
       ↓
압축률: 128×

메모리 추정 (1M 벡터, 768 dims):
~12MB (벡터) + ~98MB (코드북) + ~8MB (ID) = ~118MB
```

### 8.2 FastScan (4-bit PQ)

- HNSW 대비 2.7× 메모리 절감 (그래프 오버헤드 없음)
- 64 bytes/vector 목표
- SIMD 가속 CPU 커널

### 8.3 ERA-SmartFarm-RAG 적용

- 현재: FAISS mmap + IndexFlatIP (정확도 우선)
- 대안: PQ 적용 시 메모리 100× 절감 가능 (Recall trade-off)

---

## 9. 전체 시스템 컴포넌트 맵

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Frontend                                   │
│   Streamlit App (frontend/streamlit/)                                │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                                 │
│   ┌───────────────────────────────────────────────────────────────┐ │
│   │                      API Routes                                │ │
│   │  routes_query.py  routes_ingest.py  routes_monitoring.py      │ │
│   └───────────────────────────────────────────────────────────────┘ │
│                                                                      │
│   deps.py (전역 리트리버 초기화)                                     │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Core Services                                  │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    Retrieval/                                │   │
│   │  Hybrid.py  Embeddings.py  Sparse.py  PathRAG.py            │   │
│   └─────────────────────────────────────────────────────────────┘   │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                     Ingest/                                  │   │
│   │  GraphBuilder.py  Chunking.py  OCREngine.py                 │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│   LLM.py  Ontology.py  ResponseCache.py  TemplateResponder.py       │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Data Storage                                  │
│   data/index/        data/cache/         data/ontology/             │
│   dense.faiss        responses.jsonl     wasabi_ontology.json       │
│   sparse.pkl                                                         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 10. 참고 문헌 (최신 연구)

### 핵심 논문
- **MobileRAG**: Park, T., Lee, G., Kim, M.-S. (2025). "MobileRAG: A Fast, Memory-Efficient, and Energy-Efficient Method for On-Device RAG." [arXiv:2507.01079](https://arxiv.org/abs/2507.01079)
- **EdgeRAG**: Seemakhupt, K., Liu, S., Khan, S. (2024). "EdgeRAG: Online-Indexed RAG for Edge Devices." [arXiv:2412.21023](https://arxiv.org/abs/2412.21023)
- **PathRAG**: Chen, B., et al. (2025). "PathRAG: Pruning Graph-based RAG with Relational Paths." [arXiv:2502.14902](https://arxiv.org/abs/2502.14902)

### 시스템/런타임
- **llama.cpp**: Gerganov, G. (2024). [GitHub](https://github.com/ggml-org/llama.cpp)
- **FAISS**: Johnson, J., et al. (2019). "Billion-scale similarity search with GPUs." [arXiv:1702.08734](https://arxiv.org/abs/1702.08734)

### 양자화/최적화
- **AQUA-KV**: Shutova, A., et al. (2025). "Cache Me If You Must: Adaptive Key-Value Quantization." [arXiv:2501.19392](https://arxiv.org/abs/2501.19392)
- **KeyDiff**: Park, J., et al. (2025). "KeyDiff: Key Similarity-Based KV Cache Eviction." [arXiv:2504.15364](https://arxiv.org/abs/2504.15364)
- **UniQL**: Wang, Y., et al. (2025). "UniQL: Unified Quantization and Low-Rank Compression." [arXiv:2512.03383](https://arxiv.org/abs/2512.03383)
- **FlexQuant**: Chen, X., et al. (2025). "FlexQuant: Elastic Quantization for Edge Devices." [arXiv:2501.07139](https://arxiv.org/abs/2501.07139)
