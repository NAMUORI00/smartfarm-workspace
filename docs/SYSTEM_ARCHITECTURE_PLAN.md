# ERA-SmartFarm-RAG: Multimodal Knowledge-Graph-Augmented RAG for Edge-Deployable Precision Agriculture

> **System Architecture & Research Plan**  
> **Target Venue**: IEEE Access (Open Access, IF ~3.4)  
> **작성일**: 2026-02-11  
> **핵심 키워드**: Multimodal RAG, Knowledge Graph, Edge Computing, Precision Agriculture, LLM Evaluation

---

## 1. 논문 컨셉 및 학술적 기여점 (Novel Contributions)

### 1.1 제안 논문 제목 (Working Title)

> **"ERA-SmartFarm: A Multimodal Knowledge-Graph-Augmented RAG Framework  
> with Dual-Level Retrieval for Edge-Deployable Precision Agriculture Advisory Systems"**

### 1.2 연구 배경 및 Gap 분석

**기존 연구의 한계점**:

| 기존 연구 | 한계점 | 우리의 해결 |
|---|---|---|
| 전통 RAG (Dense+Sparse 검색) | 플랫 청크 검색 → 문서 간 관계 소실 | **지식그래프 기반 Dual-Level 검색** |
| GraphRAG (Microsoft, 2024) | 그래프 구축에 대규모 LLM 필요, 엣지 배포 불가 | **오프라인 사전 인제스트 + 경량 LLM 런타임** |
| LightRAG (HKUDS, 2024) | 텍스트 전용, 멀티모달 미지원 | **Unstructured 기반 멀티모달 파싱** |
| RAG-Anything (HKUDS, 2025) | 그래프 DB 미사용 (인메모리), 엣지 배포 미고려 | **FalkorDB + Qdrant 영속 저장소, 엣지 최적화** |
| RAG-MMF-SF (2025, Smart Farming) | 범용 프레임워크 아닌 특정 진단 | **도메인 불문 농업 전반 지식 QA** |
| IEEE Access "Precision Farming" (2024) | RAG/KG 미적용, 전통 ML 중심 | **LLM + KG + RAG 통합 프레임워크** |

### 1.3 핵심 학술적 기여점 (Contributions)

IEEE Access 등재를 위한 **4가지 명확한 기여점**:

#### C1: 도메인 특화 Multimodal Knowledge Graph Construction Pipeline
- **Unstructured** 기반 멀티모달 파싱 (PDF 테이블/이미지/수식)
- **OpenAI-compatible LLM** (설정 가능; 실험: Kimi-K2.5)을 활용한 농업 도메인 엔티티/관계 추출
- **FalkorDB** 기반 영속적 계층구조 지식그래프 (Nested/Hierarchical)
- 차별점: RAG-Anything은 인메모리 그래프만 지원 → 우리는 영속 DB + Cypher 쿼리

#### C2: Dual-Level Hybrid Retrieval with Vector-Graph Fusion
- **Qdrant** (Dense+Sparse 하이브리드) + **FalkorDB** (Graph Traversal) 3채널 퓨전
- Low-Level: 개체/속성 직접 검색 (세부 질의)
- High-Level: Multi-hop 서브그래프 탐색 (인과관계/추론 질의)
- 차별점: LightRAG의 Dual-Level을 실제 그래프 DB로 구현

#### C3: Edge-Optimized Deployment Architecture
- 인제스트: OpenAI-compatible LLM API (온라인 사전처리) → 지식그래프/벡터 인덱스 생성
- 런타임: **경량 LLM** (설정 가능; 실험: Qwen3-4B Q4_K_M, ~2.3GB VRAM) llama.cpp 기반 로컬 추론
- 엣지 디바이스 (8GB RAM) 단독 운영 가능한 오프라인 QA 시스템
- 차별점: 기존 GraphRAG는 GPT-4급 필요 → 우리는 4B급 경량 모델로 답변 생성

#### C4: Comprehensive RAGAS-based Evaluation with OSS Judge LLM
- **RAGAS** 프레임워크 기반 자동 평가 (설정 가능한 메트릭; 실험: Faithfulness, Context Precision/Recall, Answer Relevancy)
- **OSS Judge LLM** (설정 가능; 실험: Qwen3-235B-A22B 또는 동급 MoE) 으로 상용 API 의존 제거
- 농업 도메인 벤치마크 데이터셋 활용 (실험: AgXQA 등 공개 농업 QA)
- 차별점: 기존 연구는 GPT-4 Judge 의존 → 완전 오픈소스 평가 파이프라인

---

## 2. 시스템 아키텍처

### 2.1 전체 구조도

```
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 1: OFFLINE INGEST (Online Server)           │
│                                                                     │
│  ┌─────────────┐   ┌──────────────────┐   ┌───────────────────┐    │
│  │  Raw Data    │   │   Unstructured   │   │   Kimi-K2.5 API   │    │
│  │  ─────────   │──▸│   Partitioner    │──▸│   Vision+NLU      │    │
│  │  PDF/LOG/    │   │   ─────────────  │   │   ──────────────  │    │
│  │  Images/     │   │  • Text Chunks   │   │  • Entity Extr.  │    │
│  │  Sensor CSV  │   │  • Tables→HTML   │   │  • Relation Extr.│    │
│  └─────────────┘   │  • Images→Base64 │   │  • Deduplication │    │
│                     │  • Equations     │   │  • Normalization │    │
│                     └──────────────────┘   └────────┬──────────┘    │
│                                                     │               │
│                     ┌───────────────────────────────┴──────────┐    │
│                     │        Knowledge Artifacts               │    │
│                     │  ┌──────────────┐ ┌───────────────────┐  │    │
│                     │  │  FalkorDB    │ │   Qdrant          │  │    │
│                     │  │  ──────────  │ │   ─────────────── │  │    │
│                     │  │  Nested KG   │ │   Dense Vectors   │  │    │
│                     │  │  (Cypher)    │ │   + Sparse (BM25) │  │    │
│                     │  │  Entities    │ │   + Metadata      │  │    │
│                     │  │  Relations   │ │   (HNSW+Payload)  │  │    │
│                     │  │  Hierarchy   │ │                   │  │    │
│                     │  └──────────────┘ └───────────────────┘  │    │
│                     └──────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                                  │ Sync artifacts to edge
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 2: ONLINE RETRIEVAL (Edge Device)           │
│                                                                     │
│  User Query ──▸ Query Analysis                                      │
│                     │                                               │
│        ┌────────────┼────────────────┐                              │
│        ▼            ▼                ▼                              │
│  ┌──────────┐ ┌──────────┐  ┌─────────────────┐                    │
│  │ Qdrant   │ │ Qdrant   │  │ FalkorDB        │                    │
│  │ Dense    │ │ Sparse   │  │ Graph Retrieval  │                    │
│  │ (HNSW)   │ │ (BM25)   │  │ ─────────────── │                    │
│  │          │ │          │  │ Low: Entity+1hop │                    │
│  │          │ │          │  │ High: Subgraph   │                    │
│  └────┬─────┘ └────┬─────┘  └────────┬────────┘                    │
│       └─────────────┼────────────────┘                              │
│                     ▼                                               │
│            Weighted RRF Fusion                                      │
│                     │                                               │
│                     ▼                                               │
│        ┌──────────────────────┐                                     │
│        │  Context Shaping     │                                     │
│        │  + Graph Path Info   │                                     │
│        └──────────┬───────────┘                                     │
│                   ▼                                                 │
│        ┌──────────────────────┐                                     │
│        │  Qwen3-4B Q4         │                                     │
│        │  (llama.cpp / vLLM)  │                                     │
│        │  Answer Generation   │                                     │
│        └──────────┬───────────┘                                     │
│                   ▼                                                 │
│              Final Answer + Sources                                 │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 3: EVALUATION (Batch)                      │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │                    RAGAS Framework                        │       │
│  │                                                          │       │
│  │  ┌────────────┐ ┌────────────────┐ ┌─────────────────┐  │       │
│  │  │Faithfulness│ │Context         │ │Answer            │  │       │
│  │  │            │ │Precision/Recall│ │Relevancy/Correct.│  │       │
│  │  └────────────┘ └────────────────┘ └─────────────────┘  │       │
│  │                                                          │       │
│  │  Judge LLM: OSS 120B (DeepSeek-R1 / Qwen3-235B-A22B)    │       │
│  │  Test Set: 농업 도메인 QA 300+ pairs                      │       │
│  └──────────────────────────────────────────────────────────┘       │
│                                                                     │
│  Ablation Studies:                                                  │
│  • Dense-only vs Sparse-only vs Graph-only vs Hybrid                │
│  • With/Without KG augmentation                                     │
│  • Qwen3-4B vs 8B vs API-based generation quality                   │
│  • Edge latency profiling (p50, p95, p99)                           │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 기술 스택 상세

> **모델 유연성 원칙**: 아래 표의 LLM/임베딩 모델은 모두 OpenAI-compatible API를 통해 **설정 가능(configurable)**합니다. 특정 모델에 의존하지 않으며, 논문 실험에서는 아래 기본 모델을 사용하여 검증합니다.

| 계층 | 기술 | 버전/스펙 | 역할 | 학술적 근거 |
|---|---|---|---|---|
| **전처리** | Unstructured | OSS | 멀티모달 문서 파싱 (PDF/Table/Image) | ETL for RAG (Unstructured.io, 2024) |
| **추출 LLM** | OpenAI-compatible (설정 가능) | 실험: Kimi-K2.5 | 엔티티/관계 추출, 다중 모델 폴백 지원 | MoonshotAI (2025) 등 |
| **임베딩 모델** | OpenAI-compatible (설정 가능) | 실험: Qwen3-VL-Embedding-2B (512d) | Dense+Image 벡터 생성 | 환경변수로 모델/차원 변경 가능 |
| **벡터 DB** | Qdrant | v1.10+ | Dense(HNSW) + Sparse(BM25) 하이브리드 | Filterable HNSW, Named Vectors |
| **그래프 DB** | FalkorDB | v4+ | 지식그래프 저장 + Cypher 탐색 | GraphBLAS sparse matrix, <1ms 지연 |
| **답변 생성 LLM** | llama.cpp / OpenAI-compatible (설정 가능) | 실험: Qwen3-4B-Q4_K_M | 엣지 디바이스 답변 생성 | 4-bit 양자화 GGUF |
| **평가 프레임워크** | RAGAS | v0.2+ | 자동 평가 (메트릭 설정 가능) | Es et al. (ACL 2024) |
| **평가 Judge LLM** | OpenAI-compatible (설정 가능) | 실험: OSS 120B급 MoE | RAGAS 메트릭 판정 | LLM-as-Judge (2024) |
| **참조 프레임워크** | RAG-Anything | — | Multimodal KG 아키텍처 참조 | Guo et al. (arXiv:2510.12323) |

---

## 3. 각 컴포넌트 상세 설계

### 3.1 Stage 1: Offline Ingest Pipeline

#### 3.1.1 Unstructured 기반 멀티모달 파싱

```
Input Documents (PDF, DOCX, HTML, Images, CSV)
    │
    ▼
Unstructured Partition
    │
    ├── Text Elements ────────▸ Semantic Chunking (512 tokens, stride 128)
    │                              └── chunk_id, text, metadata, source_doc_id
    │
    ├── Table Elements ───────▸ HTML 테이블 보존 + 텍스트 요약 생성
    │                              └── table_id, html, summary, source_doc_id
    │
    ├── Image Elements ───────▸ Base64 인코딩 + Kimi-K2.5 캡셔닝
    │                              └── image_id, base64, caption, source_doc_id
    │
    └── Formula Elements ─────▸ LaTeX 보존 + 의미 설명 생성
                                   └── formula_id, latex, explanation, source_doc_id
```

**RAG-Anything 대비 차별점**:
- RAG-Anything은 MinerU 파서를 사용하지만, 우리는 **Unstructured**를 사용하여 더 넓은 포맷 지원
- Unstructured는 LangChain/LlamaIndex와 네이티브 통합, 산업 표준 ETL

**구현 핵심**:
```python
# Unstructured 파티셔닝 예시
from unstructured.partition.auto import partition

elements = partition(
    filename="농업매뉴얼.pdf",
    strategy="hi_res",           # 고해상도 레이아웃 분석
    extract_images_in_pdf=True,  # 이미지 추출
    infer_table_structure=True,  # 테이블 구조 추론
    languages=["kor", "eng"],    # 한국어+영어
)

# Element 타입별 라우팅
text_chunks = [e for e in elements if e.category == "NarrativeText"]
tables      = [e for e in elements if e.category == "Table"]
images      = [e for e in elements if e.category == "Image"]
```

#### 3.1.2 OpenAI-compatible LLM 기반 지식 추출

**LLM Extraction Pipeline** (RAG-Anything의 Multimodal Analysis Engine 참조):

> 추출 인터페이스는 OpenAI-compatible 고정이며,
> 모델 선택은 `OPENAI_COMPAT_MODEL` 1개 키와 코드 내 안전 폴백 체인으로 운영합니다.

```
                    ┌──────────────────────────┐
                    │  LLM Extractor           │
                    │  (OpenAI-compatible API)  │
                    │  실험: Kimi-K2.5           │
                    │                          │
  Text Chunks ─────▸│  Entity Extraction       │──▸ Entities + Relations
                    │  ・작물/병해/환경/재배법   │
                    │  ・인과관계 추출          │
                    │  ・Confidence Score       │
                    │                          │
  Table HTML ──────▸│  Structured Extraction   │──▸ Table Entities
                    │  ・수치 데이터 해석       │     (metrics, thresholds)
                    │  ・조건-결과 관계         │
                    │                          │
  Image Base64 ────▸│  Visual Analysis         │──▸ Image Descriptions
                    │  ・병징 식별              │     + Visual Entities
                    │  ・생육 상태 분석         │
                    └──────────────────────────┘
```

**추출 프롬프트 구조** (RAG-Anything Section 3의 Multimodal Entity Extraction 참조):

```json
{
  "entities": [
    {
      "text": "토마토 잎곰팡이병",
      "type": "DISEASE",
      "canonical_id": "disease_tomato_leaf_mold",
      "confidence": 0.92,
      "source_modality": "text"
    },
    {
      "text": "습도 85% 이상",
      "type": "CONDITION",
      "canonical_id": "env_humidity_above_85",
      "confidence": 0.88,
      "source_modality": "table"
    }
  ],
  "relations": [
    {
      "source": "env_humidity_above_85",
      "target": "disease_tomato_leaf_mold",
      "type": "CAUSES",
      "confidence": 0.85,
      "evidence": "습도가 85% 이상이면 잎곰팡이병 발생 위험 증가"
    }
  ]
}
```

#### 3.1.3 FalkorDB Nested Knowledge Graph

**그래프 스키마 설계** (RAG-Anything의 Hierarchical Structure Preservation + FalkorDB Cypher):

```
Node Types:
  (:Crop       {name, scientific_name, canonical_id, aliases[]})
  (:Disease    {name, symptoms, canonical_id, severity})
  (:Pest       {name, canonical_id, lifecycle})
  (:Environment{metric, unit, optimal_range, canonical_id})
  (:Practice   {name, description, canonical_id, category})
  (:Condition  {expression, threshold, unit, canonical_id})
  (:Chunk      {chunk_id, text, source_doc, page, modality})
  (:Document   {doc_id, filename, doc_type, ingested_at})

Edge Types:
  (Condition)-[:CAUSES {confidence, evidence}]->(Disease)
  (Disease)-[:TREATED_BY {confidence}]->(Practice)
  (Practice)-[:REQUIRES {confidence}]->(Condition)
  (Crop)-[:SUSCEPTIBLE_TO {confidence}]->(Disease)
  (Environment)-[:AFFECTS {direction, magnitude}]->(Crop)
  (Chunk)-[:MENTIONS {relevance}]->(Entity)
  (Chunk)-[:BELONGS_TO]->(Document)
  (Entity)-[:PART_OF]->(Entity)  // Hierarchical nesting
```

**계층(Nested) 구조의 핵심**:
```cypher
// 계층 구조 예시: 토마토 → 시설재배 → 병해충 → 잎곰팡이병
CREATE (tomato:Crop {name: '토마토', canonical_id: 'crop_tomato'})
CREATE (greenhouse:Practice {name: '시설재배', canonical_id: 'practice_greenhouse'})
CREATE (diseases:Category {name: '병해충관리', canonical_id: 'cat_disease_mgmt'})
CREATE (leaf_mold:Disease {name: '잎곰팡이병', canonical_id: 'disease_leaf_mold'})

CREATE (greenhouse)-[:APPLIED_TO]->(tomato)
CREATE (diseases)-[:PART_OF]->(greenhouse)
CREATE (leaf_mold)-[:PART_OF]->(diseases)

// Multi-hop 쿼리: 토마토 시설재배의 모든 병해와 대처법
MATCH (c:Crop {name:'토마토'})<-[:APPLIED_TO]-(p:Practice)
      <-[:PART_OF*1..3]-(d:Disease)-[:TREATED_BY]->(t:Practice)
RETURN d.name, t.name, t.description
```

**FalkorDB 선택 근거**:
- GraphBLAS sparse matrix → <1ms 그래프 탐색 지연
- OpenCypher 지원 → 표준 쿼리 언어
- Redis 기반 → 엣지 디바이스에서도 경량 운영 가능
- GraphRAG-SDK → 자동 온톨로지 탐지 + LLM 기반 Cypher 생성

#### 3.1.4 Qdrant 하이브리드 벡터 인덱스

**컬렉션 설계**:

```python
from qdrant_client import QdrantClient, models

client = QdrantClient(host="localhost", port=6333)

# Multi-vector 컬렉션: Dense (text + image) + Sparse를 하나의 point에 저장
# 임베딩 모델/차원은 코드 기본값으로 고정 (Qwen3-VL-Embedding-2B, 512d)
client.create_collection(
    collection_name="smartfarm_chunks",
    vectors_config={
        "dense_text": models.VectorParams(
            size=512,
            distance=models.Distance.COSINE,
        ),
        "dense_image": models.VectorParams(
            size=512,
            distance=models.Distance.COSINE,
        ),
    },
    sparse_vectors_config={
        "sparse": models.SparseVectorParams(
            modifier=models.Modifier.IDF,  # BM25 스타일 IDF
        ),
    },
)

# Named Vectors로 Dense(text/image) + Sparse 동시 저장
client.upsert(
    collection_name="smartfarm_chunks",
    points=[
        models.PointStruct(
            id=chunk_id,
            vector={
                "dense_text": text_embedding,   # [0.1, 0.3, ...]
                "dense_image": image_embedding,  # 이미지 모달리티용 (없으면 zero)
                "sparse": sparse_vector,         # {indices: [1, 5, 100], values: [0.8, 0.2, 0.5]}
            },
            payload={
                "text": chunk_text,
                "source_doc": doc_id,
                "modality": "text",         # text | table | image
            },
        )
    ],
)
```

**하이브리드 검색 (RRF 퓨전)**:

```python
# Qdrant 네이티브 Hybrid Search (v1.10+)
results = client.query_points(
    collection_name="smartfarm_chunks",
    prefetch=[
        models.Prefetch(
            query=dense_query_vector,
            using="dense",
            limit=20,
        ),
        models.Prefetch(
            query=sparse_query_vector,
            using="sparse",
            limit=20,
        ),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),  # Reciprocal Rank Fusion
    limit=10,
)
```

**Qdrant 선택 근거**:
- Named Vectors → Dense+Sparse를 단일 컬렉션에서 관리
- 네이티브 RRF 퓨전 → 별도 퓨전 코드 불필요
- Filterable HNSW → 메타데이터 기반 필터링 (문서 타입, 모달리티)
- Qdrant Edge → 엣지 디바이스 배포 지원 (2025)

### 3.2 Stage 2: Online Retrieval Pipeline

#### 3.2.1 3-Channel Retrieval Architecture

```python
class DualLevelGraphRetriever:
    """FalkorDB 기반 Dual-Level 그래프 검색.
    
    RAG-Anything의 Modality-Aware Retrieval +
    LightRAG의 Dual-Level 아이디어를 FalkorDB Cypher로 구현.
    """
    
    def search_low_level(self, query_entities: List[str], k: int = 5):
        """Low-Level: 개체 직접 매칭 + 1-hop 이웃.
        
        세부 질의: "토마토 잎곰팡이병의 증상은?"
        """
        cypher = """
        MATCH (e {canonical_id: $entity_id})
        OPTIONAL MATCH (e)-[r]-(neighbor)
        OPTIONAL MATCH (neighbor)-[:MENTIONS]-(c:Chunk)
        RETURN e, r, neighbor, c
        ORDER BY r.confidence DESC
        LIMIT $k
        """
        return self._execute(cypher, entity_id=query_entities[0], k=k)
    
    def search_high_level(self, query_entities: List[str], k: int = 5):
        """High-Level: Multi-hop 서브그래프 탐색.
        
        인과관계 질의: "습도가 높으면 어떤 병이 생기고 어떻게 대처하나?"
        """
        cypher = """
        MATCH path = (start {canonical_id: $entity_id})
                     -[*1..3]-(target:Practice)
        WHERE ALL(r IN relationships(path) WHERE r.confidence > 0.5)
        WITH target, 
             reduce(score = 1.0, r IN relationships(path) | 
                    score * r.confidence) AS path_score,
             length(path) AS hops
        ORDER BY path_score DESC, hops ASC
        LIMIT $k
        MATCH (target)-[:MENTIONS]-(c:Chunk)
        RETURN target, c, path_score
        """
        return self._execute(cypher, entity_id=query_entities[0], k=k)
```

#### 3.2.2 3-Channel Weighted RRF Fusion

```python
class TriChannelFusion:
    """Dense + Sparse + Graph 3채널 퓨전.
    
    LightRAG의 Dual-Level + Qdrant RRF를 결합.
    """
    
    def __init__(self):
        self.weights = {
            "dense":  0.35,  # 의미적 유사도
            "sparse": 0.30,  # 키워드 정확도
            "graph":  0.35,  # 구조적 관계
        }
    
    def search(self, query: str, k: int = 10):
        # 1. Qdrant Hybrid (Dense + Sparse with native RRF)
        qdrant_results = self.qdrant.query_points(
            prefetch=[dense_prefetch, sparse_prefetch],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=k * 2,
        )
        
        # 2. FalkorDB Graph Retrieval (Low + High level)
        graph_entities = self._extract_query_entities(query)
        graph_results = self.graph_retriever.search(graph_entities, k=k)
        
        # 3. Cross-channel RRF Fusion
        return self._weighted_rrf(
            qdrant_results, graph_results,
            w_vector=0.65,  # Dense+Sparse combined
            w_graph=0.35,
        )
```

#### 3.2.3 엣지 답변 생성 (Configurable LLM)

> 답변 생성 LLM은 최소 환경변수(`LLM_BACKEND`, `OPENAI_COMPAT_*`)로 설정합니다.
> 인터페이스 표준은 `llama_cpp` 또는 `openai_compatible` 두 가지이며,
> 나머지 네트워크/튜닝 파라미터는 코드 기본값으로 고정합니다.

**엣지 배포 스펙** (실험 기준):

| 항목 | 스펙 |
|---|---|
| 모델 (실험) | Qwen3-4B-Instruct |
| 양자화 | Q4_K_M (GGUF) |
| 추론 엔진 | llama.cpp (GGUF) 또는 OpenAI-compatible API |
| VRAM 사용량 | ~2.3 GB (실험 모델 기준) |
| 생성 속도 (예상) | 10~15 tok/s (CPU), 40~60 tok/s (GPU) |
| 컨텍스트 윈도우 | 설정 가능 (`LLAMA_CTX_SIZE`, 기본 8192) |
| thinking 모드 | 비활성 (non-thinking, 빠른 응답) |

**RAG 프롬프트 구조**:
```
<|system|>
You are an agricultural advisor. Answer questions using ONLY the provided context.
Include specific data (temperatures, humidity, etc.) when available.
Cite sources using [Source N] format.

<|context|>
[Source 1] (graph_path: Humidity>85% --CAUSES--> Leaf Mold --TREATED_BY--> Ventilation)
시설 내 습도가 85% 이상으로 유지되면 잎곰팡이병(Fulvia fulva) 발생 위험이 높아진다.

[Source 2] (table, doc: 토마토재배매뉴얼.pdf, p.45)
| 환기 조건 | 습도 변화 | 병 발생률 |
|-----------|-----------|-----------|
| 1일 2회   | 85→65%   | -40%      |
| 1일 4회   | 85→55%   | -70%      |

<|question|>
습도가 높을 때 토마토에 생기는 병과 대처법은?

<|answer|>
```

---

### 3.3 Stage 3: RAGAS Evaluation Framework

#### 3.3.1 평가 메트릭

| 메트릭 | 측정 대상 | RAGAS 함수 | 의미 |
|---|---|---|---|
| **Faithfulness** | 생성 품질 | `faithfulness` | 답변이 검색된 컨텍스트에 기반하는 정도 |
| **Context Precision** | 검색 품질 | `context_precision` | 검색된 문서 중 관련 문서 비율 |
| **Context Recall** | 검색 커버리지 | `context_recall` | 필요한 정보가 모두 검색되었는지 |
| **Answer Relevancy** | 답변 적절성 | `answer_relevancy` | 답변이 질문에 적절한 정도 |
| **Latency (p50/p95)** | 응답 속도 | 커스텀 | 엣지 디바이스 실용성 |

> 추가 메트릭(answer_correctness 등)은 RAGAS 설정으로 필요 시 확장 가능합니다.

#### 3.3.2 Judge LLM 구성

> Judge LLM은 OpenAI-compatible API를 통해 **설정 가능**합니다.
> 환경변수는 `JUDGE_RUNTIME`, `RAGAS_BASE_URL`, `RAGAS_API_KEY`만 사용합니다.

**OSS Judge 모델 후보** (실험에서 사용 가능한 모델):

| 모델 | 파라미터 | 특징 | 호스팅 |
|---|---|---|---|
| **DeepSeek-R1** | 671B (MoE, ~37B active) | 추론 특화, LLM-as-Judge 학습에 사용됨 | vLLM / API |
| **Qwen3-235B-A22B** | 235B (MoE, ~22B active) | GPT-4급 정렬, 한국어 강점 | vLLM |
| **gpt-oss-120b** | ~117B (MoE) | 오픈 웨이트, RAG 최적화 | vLLM |
| **Llama-3.1-405B** | 405B | 범용 성능, 높은 기준선 | 8xA100 / API |

**권장**: Qwen3-235B-A22B (MoE이므로 실제 활성 파라미터 22B, 실용적 GPU 요구량)

**RAGAS 설정**:
```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    context_precision,
    context_recall,
    answer_relevancy,
)
from ragas.llms import LangchainLLMWrapper

# Judge LLM (strict env: JUDGE_RUNTIME, RAGAS_BASE_URL, RAGAS_API_KEY)
judge_llm = LangchainLLMWrapper(
    ChatOpenAI(
        model="openai/gpt-oss-120b",
        base_url=os.getenv("RAGAS_BASE_URL", "http://judge-server:8000/v1"),
        api_key=os.getenv("RAGAS_API_KEY", "not-needed"),
    )
)

results = evaluate(
    dataset=eval_dataset,
    metrics=[
        faithfulness,
        context_precision,
        context_recall,
        answer_relevancy,
    ],
    llm=judge_llm,
)
```

#### 3.3.3 농업 도메인 벤치마크 데이터셋

> 데이터셋은 환경변수(`--dataset`)로 설정 가능합니다.
> 실험에서는 AgXQA (HuggingFace 공개 농업 QA) 등을 사용하며,
> HotPotQA, MuSiQue, SciFact, 2WikiMultiHopQA 등 추가 벤치마크도 지원합니다.

**질의 유형별 벤치마크 구성**:

| 카테고리 | 예시 질의 | 질의 유형 | 목표 |
|---|---|---|---|
| **작물 재배** | "토마토 정식 후 적정 야간 온도는?" | Low-Level (사실 기반) | Context Precision |
| **병해 진단** | "잎에 흰 가루가 생긴 원인과 대처법" | Multi-hop (인과 추론) | Graph 채널 효과 |
| **환경 제어** | "EC 3.5일 때 어떤 작물에 적합한가?" | Table 기반 | 멀티모달 검색 |
| **재배 비교** | "수경재배와 토경재배의 병해 차이" | High-Level (비교 분석) | Subgraph 탐색 |
| **센서 해석** | "습도 90%, 온도 28°C 데이터 해석" | 수치 추론 | Context + 테이블 |
| **종합 조언** | "여름철 딸기 시설재배 관리 요점" | 종합 (다구간 답변) | Answer Completeness |

---

## 4. 실험 설계 (Ablation Studies)

### 4.1 Baseline 비교 실험

IEEE Access 등재를 위해 **5가지 baseline과 체계적 비교** 필요:

| # | 시스템 | 검색 방식 | 그래프 | LLM |
|---|---|---|---|---|
| B1 | **Dense-only** | Qdrant HNSW만 | ✗ | Qwen3-4B |
| B2 | **Sparse-only** | Qdrant BM25만 | ✗ | Qwen3-4B |
| B3 | **Hybrid (Dense+Sparse)** | Qdrant RRF | ✗ | Qwen3-4B |
| B4 | **GraphRAG-only** | FalkorDB만 | ✓ | Qwen3-4B |
| B5 | **LightRAG** (원본) | LightRAG default | ✓ (인메모리) | Qwen3-4B |
| **Ours** | **ERA-SmartFarm** | Qdrant+FalkorDB 3채널 | ✓ (FalkorDB) | Qwen3-4B |

### 4.2 Ablation Study Matrix

| 실험 | 변형 | 측정 목적 |
|---|---|---|
| A1 | Graph 채널 제거 | 지식그래프 기여도 |
| A2 | Sparse 채널 제거 | 키워드 매칭 기여도 |
| A3 | Dense 채널 제거 | 시맨틱 검색 기여도 |
| A4 | Dual-Level → Low-Level only | High-Level 서브그래프 기여도 |
| A5 | Kimi-K2.5 → 규칙 기반 추출 | LLM 추출 vs 규칙 추출 |
| A6 | Qwen3-4B → Qwen3-8B | 생성 모델 크기 영향 |
| A7 | 멀티모달 off (텍스트만) | 테이블/이미지 추출 기여도 |

### 4.3 Edge 성능 프로파일링

| 측정항목 | 측정 방법 | 목표 |
|---|---|---|
| **E2E Latency** | p50, p95, p99 | <5s (엣지 실용성) |
| **Retrieval Latency** | 3채널 각각 | 병목 식별 |
| **Generation Latency** | 첫 토큰 시간 (TTFT) | <1s |
| **Memory Footprint** | RSS peak | <6GB 총합 |
| **Throughput** | QPS (queries/sec) | >1 QPS |

### 4.4 결과 리포트 구조 (IEEE Access Table 형식)

```
Table 3: Comparison of retrieval-augmented generation methods
on SmartFarm Agriculture QA Benchmark (300 questions)

| Method          | Faith. | Ctx.P | Ctx.R | Ans.R | Ans.C | Lat.(s) |
|-----------------|--------|-------|-------|-------|-------|---------|
| Dense-only      | 0.72   | 0.65  | 0.60  | 0.70  | 0.55  | 2.1     |
| Sparse-only     | 0.68   | 0.70  | 0.55  | 0.65  | 0.50  | 1.8     |
| Hybrid (D+S)    | 0.78   | 0.75  | 0.68  | 0.76  | 0.62  | 2.3     |
| GraphRAG-only   | 0.75   | 0.60  | 0.72  | 0.73  | 0.60  | 3.5     |
| LightRAG        | 0.80   | 0.72  | 0.74  | 0.78  | 0.65  | 3.0     |
| ERA-SmartFarm   | 0.85   | 0.80  | 0.82  | 0.83  | 0.72  | 3.2     |
                                    (예상 목표값, 실험으로 검증)
```

---

## 5. Docker 컴포지션 및 배포 구조

### 5.1 개발/인게스트 환경 (Online Server)

```yaml
# docker-compose.ingest.yml
version: "3.8"

services:
  # 벡터 DB
  qdrant:
    image: qdrant/qdrant:v1.13.2
    ports: ["6333:6333", "6334:6334"]
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      QDRANT__SERVICE__GRPC_PORT: 6334

  # 그래프 DB
  falkordb:
    image: falkordb/falkordb:latest
    ports: ["6379:6379"]
    volumes:
      - falkordb_data:/data
    environment:
      FALKORDB_ARGS: "--maxmemory 2gb"

  # 인게스트 워커
  ingest-worker:
    build: ./smartfarm-ingest
    depends_on: [qdrant, falkordb]
    environment:
      LLM_BACKEND: ${LLM_BACKEND:-openai_compatible}
      OPENAI_COMPAT_BASE_URL: ${OPENAI_COMPAT_BASE_URL}
      OPENAI_COMPAT_API_KEY: ${OPENAI_COMPAT_API_KEY}
      OPENAI_COMPAT_MODEL: ${OPENAI_COMPAT_MODEL:-Qwen/Qwen3-4B}
    volumes:
      - ./data/raw:/app/data/raw
      - ./data/index:/app/data/index

  # RAGAS 평가 (배치)
  evaluator:
    build: ./smartfarm-benchmarking
    depends_on: [qdrant, falkordb]
    environment:
      JUDGE_RUNTIME: ${JUDGE_RUNTIME:-api}
      RAGAS_BASE_URL: ${RAGAS_BASE_URL}
      RAGAS_API_KEY: ${RAGAS_API_KEY}

volumes:
  qdrant_data:
  falkordb_data:
```

### 5.2 엣지 배포 환경 (Edge Device, 8GB RAM)

```yaml
# docker-compose.edge.yml
version: "3.8"

services:
  # 벡터 DB (경량 모드)
  qdrant:
    image: qdrant/qdrant:v1.13.2
    ports: ["6333:6333"]
    volumes:
      - ./data/index/qdrant:/qdrant/storage
    deploy:
      resources:
        limits:
          memory: 1G

  # 그래프 DB (경량 모드)
  falkordb:
    image: falkordb/falkordb:latest
    ports: ["6379:6379"]
    volumes:
      - ./data/index/falkordb:/data
    environment:
      FALKORDB_ARGS: "--maxmemory 512mb"
    deploy:
      resources:
        limits:
          memory: 768M

  # LLM 추론 서버
  llm-server:
    build: ./smartfarm-llm-inference
    ports: ["8080:8080"]
    volumes:
      - ./models:/app/models
    environment:
      MODEL_PATH: "/app/models/qwen3-4b-instruct-q4_k_m.gguf"
      CONTEXT_SIZE: 8192
      GPU_LAYERS: 99  # 가용 GPU에 맞춰 조정
    deploy:
      resources:
        limits:
          memory: 3G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # API 서버
  api:
    build: ./smartfarm-search
    ports: ["8000:8000"]
    depends_on: [qdrant, falkordb, llm-server]
    environment:
      LLM_BACKEND: "llama_cpp"
      OPENAI_COMPAT_BASE_URL: ${OPENAI_COMPAT_BASE_URL}
      OPENAI_COMPAT_API_KEY: ${OPENAI_COMPAT_API_KEY}
      OPENAI_COMPAT_MODEL: ${OPENAI_COMPAT_MODEL:-Qwen/Qwen3-4B}
    deploy:
      resources:
        limits:
          memory: 2G

  # 프론트엔드
  frontend:
    build: ./smartfarm-frontend
    ports: ["3000:3000"]
    deploy:
      resources:
        limits:
          memory: 256M
```

**엣지 메모리 분배**:

| 컴포넌트 | 메모리 | 비고 |
|---|---|---|
| Qdrant | 1.0 GB | 벡터 인덱스 + HNSW |
| FalkorDB | 0.75 GB | 그래프 + Redis |
| Qwen3-4B Q4 (llama.cpp) | 3.0 GB | 모델 웨이트 + KV 캐시 |
| API Server | 2.0 GB | Python + 검색 로직 |
| Frontend | 0.25 GB | Static assets |
| OS/기타 | 1.0 GB | |
| **합계** | **8.0 GB** | ✅ 8GB 디바이스 적합 |

---

## 6. 구현 로드맵

### Phase 1: Foundation (Week 1~2)

| 작업 | 상세 | 산출물 |
|---|---|---|
| 1-1 | Docker Compose 구성 (Qdrant + FalkorDB + API) | `docker-compose.yml` |
| 1-2 | FalkorDB 스키마 정의 + Cypher 테스트 | `schema.cypher` |
| 1-3 | Qdrant 컬렉션 생성 + 테스트 데이터 upsert | 컬렉션 설정 코드 |
| 1-4 | Unstructured 파티셔닝 파이프라인 구현 | `DocumentParser.py` |
| 1-5 | 기존 코드 정리 (Ontology, rule-based 제거) | 코드 간소화 |

### Phase 2: Ingest Pipeline (Week 3~4)

| 작업 | 상세 | 산출물 |
|---|---|---|
| 2-1 | Kimi-K2.5 API 클라이언트 구현 | `KimiClient.py` |
| 2-2 | 엔티티/관계 추출 프롬프트 설계 + 테스트 | 추출 프롬프트 |
| 2-3 | FalkorDB 그래프 빌더 (엔티티→노드, 관계→엣지) | `GraphIngestor.py` |
| 2-4 | Qdrant 벡터 인덱서 (Dense+Sparse) | `VectorIngestor.py` |
| 2-5 | E2E 인게스트 워커 통합 | `ingest_worker.py` 리팩토링 |
| 2-6 | 테스트 코퍼스 인게스트 (농업 PDF 50건+) | 인덱스 아티팩트 |

### Phase 3: Retrieval Pipeline (Week 5~6)

| 작업 | 상세 | 산출물 |
|---|---|---|
| 3-1 | FalkorDB Dual-Level Retriever 구현 | `GraphRetriever.py` |
| 3-2 | Qdrant Hybrid Search 구현 | `VectorRetriever.py` |
| 3-3 | 3채널 RRF Fusion 구현 | `TriChannelFusion.py` |
| 3-4 | Context Shaping (그래프 경로 정보 포함) | `ContextBuilder.py` |
| 3-5 | Qwen3-4B-Q4 llama.cpp 서버 설정 | LLM 서버 구성 |
| 3-6 | E2E Query Pipeline 통합 테스트 | API 엔드포인트 |

### Phase 4: Evaluation (Week 7~8)

| 작업 | 상세 | 산출물 |
|---|---|---|
| 4-1 | 농업 QA 벤치마크 데이터셋 구축 (300+ pairs) | `eval_dataset.json` |
| 4-2 | OSS Judge LLM 서버 구성 | vLLM 서버 |
| 4-3 | RAGAS 평가 파이프라인 구현 | `run_evaluation.py` |
| 4-4 | Baseline 시스템 5개 구현 + 평가 | 비교 결과 |
| 4-5 | Ablation Study 실행 (7개 변형) | ablation 테이블 |
| 4-6 | Edge 성능 프로파일링 | 레이턴시 리포트 |

### Phase 5: Paper Writing (Week 9~10)

| 작업 | 상세 | 산출물 |
|---|---|---|
| 5-1 | IEEE Access 템플릿 기반 논문 초안 | 논문 초안 |
| 5-2 | 실험 결과 테이블/그래프 작성 | Figures/Tables |
| 5-3 | 관련 연구 정리 (30+ references) | Related Work 섹션 |
| 5-4 | 논문 리뷰 및 수정 | 최종본 |

---

## 7. IEEE Access 논문 구조 (Proposed)

```
1. Introduction
   - Smart agriculture의 정보 접근성 문제
   - 기존 RAG/GraphRAG의 한계 (엣지 배포, 멀티모달, 도메인 특화)
   - 우리의 기여점 4가지 (C1~C4)

2. Related Work
   2.1 Retrieval-Augmented Generation
   2.2 Knowledge Graph for Agriculture
   2.3 Graph-Augmented RAG (LightRAG, RAG-Anything, GraphRAG)
   2.4 Edge AI for Precision Farming

3. System Architecture
   3.1 Overview (3-stage pipeline)
   3.2 Multimodal Document Parsing (Unstructured)
   3.3 LLM-based Knowledge Extraction (Kimi-K2.5)
   3.4 Dual-Store Indexing (Qdrant + FalkorDB)
   3.5 Tri-Channel Hybrid Retrieval
   3.6 Edge-Optimized Generation (Qwen3-4B-Q4)

4. Experimental Setup
   4.1 Dataset (SmartFarm Agriculture QA Benchmark)
   4.2 Baselines (5 systems)
   4.3 Evaluation Metrics (RAGAS + Latency)
   4.4 Implementation Details

5. Results and Analysis
   5.1 Main Results (Table 3)
   5.2 Ablation Studies (Table 4)
   5.3 Edge Performance Analysis (Table 5)
   5.4 Case Studies (질의별 검색 경로 분석)

6. Discussion
   6.1 KG Contribution Analysis
   6.2 Edge Deployment Feasibility
   6.3 Limitations and Future Work

7. Conclusion
```

---

## 8. 현재 코드베이스 전환 매핑

### 8.1 유지/재활용 모듈

| 현재 모듈 | → 전환 후 | 변경 사항 |
|---|---|---|
| `CausalExtractor.py` | → `KimiExtractor.py` | LLM을 Kimi-K2.5 API로 교체 |
| `CausalSchema.py` | → `AgriSchema.py` | FalkorDB Cypher 스키마 추가 |
| `Chunking.py` | → Unstructured 파티셔닝으로 대체 | 전면 교체 |
| `LLMGraphBuilder.py` | → `FalkorDBIngestor.py` | FalkorDB Cypher MERGE로 교체 |
| `Fusion.py` | → `TriChannelFusion.py` | 5채널→3채널, Qdrant RRF 활용 |
| `Embeddings.py` | → `QdrantRetriever.py` | FAISS→Qdrant 교체 |
| `Sparse.py` | → Qdrant sparse vector로 통합 | 별도 모듈 제거 |
| `PathScoring.py` | → `GraphTraversal.py` | flow-scoring → Cypher 경로 점수 |
| `ragas_agriculture_benchmark.py` | → 유지 + Judge LLM 교체 | OSS 120B Judge 설정 |

### 8.2 제거 모듈

| 모듈 | 제거 이유 |
|---|---|
| `Ontology.py` | Kimi-K2.5 LLM 추출이 완전 대체 |
| `GraphBuilder.py` (rule-based) | FalkorDB + LLM 추출이 대체 |
| `TriGraph/` (Builder, Retriever, Index) | FalkorDB Graph Retrieval이 대체 |
| `TagHash.py` | Qdrant sparse search가 대체 |
| `CausalGraph.py` | FalkorDB 통합 그래프에 흡수 |
| `OverlayIndex/Manager/Fusion.py` | Qdrant + FalkorDB 증분 업데이트가 대체 |
| `TemplateResponder.py` | Qwen3-4B가 항상 생성 가능 |
| `PathRAG.py` | FalkorDB Cypher 탐색이 대체 |

---

## 9. 참고 문헌

### 핵심 참조

1. **RAG-Anything**: Guo, Z., et al. "RAG-Anything: All-in-One RAG Framework." arXiv:2510.12323 (2025).
   - https://github.com/HKUDS/RAG-Anything

2. **LightRAG**: Guo, Z., et al. "LightRAG: Simple and Fast Retrieval-Augmented Generation." arXiv:2410.05779 (2024).
   - https://github.com/HKUDS/LightRAG

3. **RAGAS**: Es, S., et al. "RAGAS: Automated Evaluation of Retrieval Augmented Generation." ACL 2024.
   - https://github.com/vibrantlabsai/ragas

4. **FalkorDB**: "A Super Fast Graph Database Using GraphBLAS." 
   - https://github.com/FalkorDB/FalkorDB
   - GraphRAG-SDK: https://github.com/FalkorDB/GraphRAG-SDK

5. **Qdrant**: "High-Performance Vector Search Engine."
   - https://github.com/qdrant/qdrant

6. **Unstructured**: "ETL for LLMs: Parse, chunk, and embed any document."
   - https://github.com/Unstructured-IO/unstructured

7. **Kimi-K2.5**: MoonshotAI. "Scaling Reinforcement Learning with LLMs." (2025)
   - https://huggingface.co/moonshotai/Kimi-K2.5

8. **Qwen3**: Alibaba. "Qwen3 Technical Report." (2025)
   - https://qwenlm.github.io/blog/qwen3/

### 관련 연구

9. **GraphRAG**: Edge, D., et al. "From Local to Global: A Graph RAG Approach." Microsoft Research (2024).

10. **PathRAG**: BUPT-GAMMA. "Pruning Graph-based Retrieval Augmented Generation with Relational Paths."
    - https://github.com/BUPT-GAMMA/PathRAG

11. **LinearRAG**: DEEP-PolyU. "Tri-Graph + Semantic Bridging + PPR-style Retrieval." arXiv:2510.10114.

12. **IEEE Access Smart Farming**: "RAG-Driven Memory Architectures in Conversational LLMs - A Literature Review with Insights into Emerging Agriculture." IEEE Access (2025).

13. **RAG-MMF-SF**: "RAG-Enhanced Smart Farming: A Methodology for Multimodal Fusion and AI-driven Crop Health Diagnosis." (2025).

14. **RRF**: Cormack, G.V., et al. "Reciprocal Rank Fusion outperforms Condorcet and Individual Rank Learning Methods." SIGIR 2009.

15. **BEIR**: Thakur, N., et al. "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of IR Models." NeurIPS 2021.

16. **LLM-as-Judge**: Zheng, L., et al. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." NeurIPS 2023.

---

## 부록 A. 2026-02-18 갱신: 휴리스틱 배제형 DAT 적용

### A.1 정책 반영
- Stage 2 퓨전에서 쿼리 규칙 기반(휴리스틱) 가중치를 사용하지 않는다.
- `dense/sparse/graph` 채널 가중치는 데이터 기반 튜닝(DAT: Data-driven Adaptive Tuning)으로 결정한다.
- 인터페이스는 기존 원칙대로 `openai_compatible`, `llama_cpp`만 유지한다.

### A.2 구현 반영 위치
- `smartfarm-benchmarking/benchmarking/experiments/paper_eval.py`
- 반영 내용:
1. `ours_structural`에 대한 채널 가중치 자동 탐색(grid search, nDCG@k 목적함수)
2. 결과 JSON에 `adaptive_fusion` 메타데이터 기록
3. A1~A7 실험 경로 재사용(회귀 없이 동작)

### A.3 근거 연구(다국어/경량)
- mDAPT: https://arxiv.org/abs/2503.17488
- Retrofitting Small Multilingual Models for Dense Retrieval: https://arxiv.org/abs/2507.02705
- MAD-X: https://aclanthology.org/2020.emnlp-main.617/

### A.4 실험 게이트 반영
- 성능 향상 확인 시에만 계획서 고정 항목으로 채택한다.
- 최신 측정 결과(40 queries x 3 datasets, retrieval-only)에서 `ours_structural`의 Macro 평균:
  - `nDCG@10`: `0.4485 -> 0.7547` (+0.3062)
  - `MRR`: `0.4236 -> 0.8277` (+0.4041)
- 상세 리포트: `docs/DAT_PERFORMANCE_REPORT_2026-02-18.md`
