# 스마트팜 도메인 특화 하이브리드 RAG 시스템

## 초록 (Abstract)

본 연구에서는 스마트팜 도메인에 특화된 하이브리드 RAG(Retrieval-Augmented Generation) 시스템을 제안한다. 제안 시스템은 Dense-Sparse-Graph 3채널 검색 융합, 온톨로지 기반 개념 매칭, 시맨틱 중복 제거, 인과관계 그래프 탐색을 통합하여 농업 전문 지식 검색의 정확도와 다양성을 향상시킨다. 특히 엣지 디바이스 환경을 고려한 경량화 설계와 작물별 필터링 메커니즘을 통해 실용적인 스마트팜 질의응답 시스템을 구현하였다.

**키워드**: RAG, 하이브리드 검색, 지식 그래프, 온톨로지, 스마트팜, 인과관계 추출

---

## 1. 서론 (Introduction)

### 1.1 연구 배경

대규모 언어 모델(LLM)의 발전으로 자연어 기반 질의응답 시스템이 다양한 도메인에 적용되고 있다. 그러나 LLM 단독 사용 시 도메인 특화 지식의 부재, 환각(hallucination) 현상, 최신 정보 반영 한계 등의 문제가 발생한다[1]. 이를 해결하기 위해 RAG(Retrieval-Augmented Generation) 접근법이 제안되었으며, 외부 지식 베이스에서 관련 문서를 검색하여 LLM의 생성 품질을 향상시킨다[2].

스마트팜 도메인은 작물 재배, 환경 제어, 병해충 관리 등 전문적인 농업 지식을 요구한다. 기존의 범용 RAG 시스템은 다음과 같은 스마트팜 특수성을 충분히 반영하지 못한다:

1. **작물별 맥락 의존성**: 동일한 환경 조건이라도 작물에 따라 적정 범위와 대응 방법이 상이함
2. **인과관계의 중요성**: 문제 원인과 해결책 간의 연결이 핵심적인 정보 구조를 형성
3. **수치/단위 정보의 정확성**: EC, pH, 온도 등 정량적 정보의 정확한 매칭 필요
4. **온톨로지 기반 개념 체계**: 작물-환경-병해-영양소 간의 구조화된 관계 존재

### 1.2 연구 목적

본 연구의 목적은 스마트팜 도메인의 특수성을 반영한 하이브리드 RAG 시스템을 설계하고 구현하는 것이다. 구체적인 연구 목표는 다음과 같다:

1. Dense-Sparse-Graph 3채널 검색 융합을 통한 검색 성능 향상
2. 온톨로지 기반 개념 매칭으로 도메인 특화 검색 지원
3. 인과관계 그래프를 활용한 문맥적 연관 문서 검색
4. 시맨틱 중복 제거를 통한 검색 결과 다양성 확보
5. 엣지 디바이스 환경을 고려한 경량화 설계

---

## 2. 관련 연구 (Related Work)

### 2.1 RAG (Retrieval-Augmented Generation)

RAG는 Lewis et al.(2020)이 제안한 접근법으로, 질의에 대해 외부 지식 베이스에서 관련 문서를 검색하고 이를 LLM의 컨텍스트로 제공하여 생성 품질을 향상시킨다[2]. 기본 RAG 파이프라인은 다음과 같다:

```
Query → Retriever → Top-k Documents → LLM + Context → Answer
```

초기 RAG 연구는 Dense Passage Retrieval(DPR)을 주로 활용하였으나, 최근에는 다양한 검색 전략의 융합이 연구되고 있다.

### 2.2 Hybrid Retrieval

Dense retrieval은 의미적 유사성을 잘 포착하지만 키워드 매칭에 취약하고, Sparse retrieval(BM25)은 키워드 매칭에 강하지만 의미적 유사성을 놓칠 수 있다. Hybrid retrieval은 두 방식의 장점을 결합한다[3].

**DAT(Decomposed-Attention Transformer)**[4] 스타일의 융합은 질의 특성에 따라 Dense와 Sparse의 가중치를 동적으로 조정한다. 본 연구에서는 이를 확장하여 Graph 채널을 추가한 3채널 융합을 제안한다.

가중치 결정 요소:
- 숫자/단위 패턴 포함 여부 → Sparse 가중치 증가
- 추상적 개념 질의 → Dense 가중치 증가
- 구조적 관계 질의 → Graph 가중치 증가

### 2.3 Graph-based RAG

**GraphRAG**[5]는 지식 그래프를 활용하여 엔티티 간 관계를 명시적으로 모델링하고 검색에 활용한다. 노드는 개념이나 문서를, 엣지는 관계를 표현한다.

**PathRAG**[6]는 질의와 관련된 시작 노드에서 경로 탐색을 수행하여 연관 문서를 수집한다. 본 연구에서는 PathRAG-lite 변형을 구현하여 다음을 수행한다:

1. 온톨로지 개념 매칭으로 시작 노드 결정
2. BFS(너비 우선 탐색)로 practice 노드까지 경로 탐색
3. 인과관계 엣지(causes, solved_by)를 따라 관련 문서 수집

### 2.4 Semantic Deduplication

검색 결과의 다양성 확보를 위해 중복/유사 문서 제거가 필요하다. **MMR(Maximal Marginal Relevance)**[7]은 관련성과 다양성의 균형을 맞추는 대표적인 방법이다:

$$MMR = \arg\max_{D_i \in R \setminus S} [\lambda \cdot Sim(D_i, Q) - (1-\lambda) \cdot \max_{D_j \in S} Sim(D_i, D_j)]$$

본 연구에서는 임베딩 코사인 유사도 기반의 greedy deduplication을 적용한다. 유사도가 임계값(θ=0.85) 이상인 문서 쌍에서 후순위 문서를 제거한다.

### 2.5 Causal Information Extraction

인과관계 추출은 텍스트에서 원인-결과 관계를 식별하는 태스크이다[8]. 본 연구에서는 규칙 기반 패턴 매칭을 활용하여 문서의 인과관계 역할(cause/effect/solution)을 분류하고, 공통 키워드를 통해 문서 간 인과관계 엣지를 자동 생성한다.

---

## 3. 제안 방법론 (Proposed Methodology)

### 3.1 시스템 아키텍처

제안 시스템은 다음 구성요소로 이루어진다:

```
┌─────────────────────────────────────────────────────────────┐
│                     Query Processing                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Crop Extract │  │ Ontology    │  │ Dynamic Alpha       │  │
│  │             │  │ Matching    │  │ Estimation          │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   3-Channel Retrieval                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Dense     │  │   Sparse    │  │      PathRAG        │  │
│  │  (FAISS)    │  │  (BM25)     │  │   (Graph BFS)       │  │
│  │   α_d       │  │   α_s       │  │      α_p            │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Post-Processing                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Score Fusion │  │ Crop Filter │  │ Semantic Dedup      │  │
│  │             │  │             │  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      LLM Generation                          │
│            Context + Query → Answer                          │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 온톨로지 기반 개념 매칭

스마트팜 도메인 온톨로지는 다음 개념 유형을 정의한다:

| 유형 | 설명 | 예시 |
|------|------|------|
| crop | 재배 작물 | 토마토, 파프리카, 딸기, 상추, 와사비 |
| env | 환경 요소 | 온도, 습도, EC, pH, CO2, 광량 |
| nutrient | 영양소 | 질소, 인산, 칼륨, 칼슘, 마그네슘 |
| disease | 병해충 | 흰가루병, 뿌리썩음병, 잿빛곰팡이병 |
| stage | 생육 단계 | 육묘기, 정식기, 개화기, 착과기, 수확기 |
| practice | 재배 실천 | 관수, 시비, 환기, 적엽, 적심 |

**OntologyMatcher** 클래스는 질의 텍스트에서 온톨로지 개념을 추출한다:

```python
def match(self, text: str) -> Dict[str, Set[str]]:
    """텍스트에서 온톨로지 개념 추출"""
    hits = {typ: set() for typ in self.ontology}
    for typ, concepts in self.ontology.items():
        for concept in concepts:
            if concept in text:
                hits[typ].add(concept)
    return hits
```

### 3.3 Dynamic Alpha Estimation

3채널 가중치 (α_d, α_s, α_p)는 질의 특성에 따라 동적으로 결정된다:

```python
def dynamic_alphas(self, q: str) -> Tuple[float, float, float]:
    # 기본 가중치
    dense_score, sparse_score = 0.5, 0.5

    # 숫자/단위 패턴 → Sparse 강화
    if re.search(r"\d+\.?\d*\s*(ds/m|℃|%|ppm|ec|ph)", q.lower()):
        sparse_score += 0.2
        dense_score -= 0.2

    # 온톨로지 매칭 → 채널별 조정
    hits = ontology_matcher.match(q)
    if hits.get("env") or hits.get("nutrient"):
        sparse_score += 0.1

    # PathRAG 활성화 조건
    path_score = 0.0
    if hits.get("disease") or hits.get("practice"):
        path_score = 0.3

    # 정규화
    total = dense_score + sparse_score + path_score
    return dense_score/total, sparse_score/total, path_score/total
```

### 3.4 인과관계 그래프 구축

#### 3.4.1 인과관계 역할 탐지

문서 텍스트에서 인과관계 역할을 패턴 매칭으로 분류한다:

| 역할 | 패턴 예시 |
|------|----------|
| cause | "원인", "때문", "~하면", "~로 인해" |
| effect | "결과", "증상", "문제", "~가 발생" |
| solution | "해결", "방법", "관리", "~해야" |

```python
CAUSE_PATTERNS = [
    r"(원인|이유|때문|로 인해|하면|높으면|낮으면)",
    r"(발생\s*원인|주요\s*원인)",
]
EFFECT_PATTERNS = [
    r"(결과|영향|증상|문제|장애|저하)",
    r"(발생|나타|생기|떨어지)",
]
SOLUTION_PATTERNS = [
    r"(해결|대응|방법|조치|관리|예방)",
    r"(해야|필요|권장|추천)",
]
```

#### 3.4.2 인과관계 엣지 생성

공통 키워드를 기반으로 문서 간 인과관계 엣지를 자동 생성한다:

```
cause_doc ──causes──▶ effect_doc ──solved_by──▶ solution_doc
```

키워드 추출 대상:
- 환경 요소: 온도, 습도, EC, pH
- 작물명: 토마토, 파프리카 등
- 병해: 흰가루병, 뿌리썩음 등
- 상태: 고온, 저온, 과습, 건조

### 3.5 시맨틱 중복 제거

검색된 문서의 다양성 확보를 위해 임베딩 유사도 기반 중복 제거를 수행한다:

```python
def _deduplicate(self, docs: List[SourceDoc], threshold: float = 0.85):
    embeddings = self.dense.encode([d.text for d in docs])
    faiss.normalize_L2(embeddings)
    sim_matrix = embeddings @ embeddings.T

    keep_indices = []
    for i in range(len(docs)):
        is_duplicate = False
        for kept_idx in keep_indices:
            if sim_matrix[i, kept_idx] >= threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            keep_indices.append(i)

    return [docs[i] for i in keep_indices]
```

**알고리즘 특성**:
- Greedy 방식으로 순차 처리
- 앞 순위(고득점) 문서 우선 유지
- O(n²) 시간 복잡도 (n: 문서 수)

### 3.6 작물 필터링

질의에서 추출된 작물과 문서의 작물 메타데이터를 비교하여 스코어를 조정한다:

| 조건 | 스코어 조정 |
|------|------------|
| 작물 일치 | +0.5 (보너스) |
| 작물 불일치 | ×0.15 (패널티) |
| 작물 정보 없음 | 유지 |

```python
def _apply_crop_filter(self, scores: dict, query_crop: str):
    for idx, score in scores.items():
        doc_crop = get_doc_crop(self.dense.docs[idx])

        if doc_crop == query_crop:
            scores[idx] = score + CROP_MATCH_BONUS
        elif doc_crop and doc_crop != query_crop:
            scores[idx] = score * (1 - CROP_MISMATCH_PENALTY)

    return scores
```

---

## 4. 시스템 구현 (Implementation)

### 4.1 기술 스택

| 구성요소 | 기술 |
|----------|------|
| Dense Retrieval | FAISS + Qwen3-Embedding-0.6B |
| Sparse Retrieval | BM25 (scikit-learn TfidfVectorizer) |
| LLM | Qwen3-4B-Q4_K_M (llama.cpp) |
| API Server | FastAPI |
| Containerization | Docker + CUDA support |

### 4.2 인덱스 구조

```
indices/
├── faiss_index.bin      # Dense vector index
├── sparse_index.pkl     # BM25 TF-IDF matrix
├── docs.pkl             # Document store
└── graph.json           # Knowledge graph (nodes + edges)
```

### 4.3 API 엔드포인트

| Endpoint | Method | 설명 |
|----------|--------|------|
| /ingest | POST | 문서 업로드 및 인덱싱 |
| /query | POST | 질의응답 |
| /prompts | GET/PUT | 프롬프트 템플릿 관리 |

---

## 5. 실험 및 평가 (Experiments)

### 5.1 평가 데이터셋

스마트팜 전문가가 작성한 질의-응답 쌍 100개를 평가에 활용하였다:

- 환경 제어 관련 질의: 35개
- 병해충 관리 질의: 30개
- 영양 관리 질의: 20개
- 재배 기술 질의: 15개

### 5.2 평가 지표

| 지표 | 설명 |
|------|------|
| Recall@k | 상위 k개 문서 내 정답 포함 비율 |
| Precision@k | 상위 k개 문서 중 관련 문서 비율 |
| Diversity | 검색 결과의 주제 다양성 |

### 5.3 Ablation Study (예정)

| 구성 | 설명 |
|------|------|
| Base | Dense only |
| +Sparse | Dense + Sparse hybrid |
| +PathRAG | 3-channel fusion |
| +Dedup | + Semantic deduplication |
| +CropFilter | + Crop-aware filtering |
| Full | 전체 시스템 |

---

## 6. 결론 및 향후 연구 (Conclusion)

### 6.1 결론

본 연구에서는 스마트팜 도메인 특화 하이브리드 RAG 시스템을 제안하였다. 주요 기여는 다음과 같다:

1. **3채널 검색 융합**: Dense-Sparse-Graph 채널의 동적 가중치 조정
2. **인과관계 그래프**: 자동 인과관계 추출 및 그래프 탐색
3. **도메인 특화 필터링**: 작물별 문서 우선순위 조정
4. **시맨틱 중복 제거**: 검색 결과 다양성 확보

### 6.2 향후 연구

1. **OCR 백엔드 벤치마킹**: 도메인 특화 문서(농업 매뉴얼, 재배 가이드) 대상 OCR 정확도 비교 및 파인튜닝
2. **온톨로지 자동 확장**: LLM 기반 온톨로지 개념 자동 추출
3. **다국어 지원**: 영어/일본어 농업 문헌 통합 검색
4. **실시간 센서 데이터 연동**: IoT 센서 데이터와 RAG 연계

---

## 참고문헌 (References)

[1] Ji, Z., et al. (2023). "Survey of Hallucination in Natural Language Generation." ACM Computing Surveys.

[2] Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS 2020.

[3] Karpukhin, V., et al. (2020). "Dense Passage Retrieval for Open-Domain Question Answering." EMNLP 2020.

[4] Luan, Y., et al. (2021). "Sparse, Dense, and Attentional Representations for Text Retrieval." TACL 2021.

[5] Edge, D., et al. (2024). "From Local to Global: A Graph RAG Approach to Query-Focused Summarization." arXiv:2404.16130.

[6] Chen, B., et al. (2025). "PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths." arXiv:2502.14902.

[7] Carbonell, J., & Goldstein, J. (1998). "The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries." SIGIR 1998.

[8] Feder, A., et al. (2021). "Causal Inference in Natural Language Processing: Estimation, Prediction, Interpretation and Beyond." arXiv:2109.00725.
