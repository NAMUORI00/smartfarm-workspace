# 스마트팜 도메인 특화 하이브리드 RAG 시스템

## 초록 (Abstract)

(작성 예정)

---

## 1. 서론 (Introduction)

(작성 예정)

---

## 2. 관련 연구 (Related Work)

### 2.1 Retrieval-Augmented Generation (RAG)

RAG는 대규모 언어 모델(LLM)의 생성 과정에 외부 지식 베이스 검색을 결합하여 환각(hallucination)을 줄이고 도메인 특화 지식을 제공하는 접근법이다[1]. Gao et al.(2024)의 서베이에 따르면 RAG 패러다임은 Naive RAG에서 Advanced RAG, Modular RAG로 진화하였으며, 검색(retrieval), 증강(augmentation), 생성(generation)의 세 축을 중심으로 발전하고 있다[2].

최근 RAG 연구는 검색 품질 향상, 멀티모달 확장, 도메인 특화 등으로 확장되고 있다. 특히 2025년 기준 GraphRAG, PathRAG 등 그래프 기반 RAG가 주목받고 있으며[3], 농업 분야에서도 지식 그래프와 LLM의 결합이 활발히 연구되고 있다[4].

### 2.2 Hybrid Retrieval

Dense retrieval과 Sparse retrieval은 상호 보완적 특성을 가진다. Dense 방식은 의미적 유사성을 포착하지만 희소 키워드에 취약하고, Sparse 방식(BM25, TF-IDF)은 정확한 키워드 매칭에 강하지만 의미적 유사성을 놓칠 수 있다[5].

Hybrid retrieval은 두 방식을 융합하여 검색 성능을 향상시킨다. SIGIR 2024에서 발표된 클러스터 기반 부분 Dense 검색 연구는 Sparse 검색 결과를 가이드로 활용하여 융합 최적화를 달성하였다[6]. 또한 RRF(Reciprocal Rank Fusion)를 통한 결과 융합이 널리 사용되며, BEIR 벤치마크에서 하이브리드 방식이 단일 방식 대비 nDCG@10을 향상시킨 결과가 보고되었다.

### 2.3 Graph-based RAG

Microsoft Research의 GraphRAG(2024)는 문서를 지식 그래프로 구조화하여 전역적 질의에 대응한다[7]. 그러나 기존 그래프 기반 RAG는 검색된 정보의 **중복성(redundancy)**이 문제로 지적되었다.

**PathRAG**(Chen et al., 2025)는 관계 경로(relational path) 기반 검색으로 이를 해결한다[8]. 실험 결과 PathRAG는 농업(Agriculture), 법률, 역사 등 6개 도메인에서 GraphRAG, LightRAG 대비 우수한 성능을 보였다.

### 2.4 농업 도메인 온톨로지 및 지식 그래프

농업 분야에서 온톨로지 기반 지식 표현이 활발히 연구되고 있다. Bhuyan et al.(2021)은 스마트 농업을 위한 온톨로지 기반 지식 표현 프레임워크를 제안하였으며, 시공간 농업 데이터에 대한 추론을 지원하는 래티스(lattice) 구조를 활용하였다[9].

2024년 연구에서 Springer의 *International Journal of Information Technology*에 게재된 논문은 NLP를 활용한 농업 온톨로지 개발 방법론을 제시하였으며, 텍스트에서 관계 추출을 통해 의미적 이해를 향상시켰다[10]. 또한 Cornei et al.(2024)은 스마트 농업의 시공간 역학을 포착하는 온톨로지 기반 솔루션을 RCIS 2024에서 발표하였다[11].

작물 병해충 분야에서 **CropDP-KG**(2025)는 중국의 작물 병해충 지식 그래프로, NLP 기술(NER, RE)을 활용하여 13,840개 엔티티와 21,961개 관계를 자동 구축하였다[12]. 토마토 잎 병해충 지식 그래프(2024)는 Stanford 온톨로지 구축 방법론에 따라 도메인 범위 결정, 주요 요소 나열, 용어 및 관계 정의, 클래스 계층 정의의 6단계 프로세스를 적용하였다[13].

### 2.5 인과관계 추출 (Causal Relation Extraction)

인과관계 추출은 텍스트에서 원인-결과 관계를 식별하는 NLP 태스크이다. Khoo et al.의 서베이에 따르면, 지식 기반, 통계적 기계학습, 딥러닝 기반 접근법이 사용된다[14]. 농업 도메인에서는 WordNet과 식물 병해 정보를 활용한 동사 쌍 규칙 기반 연구가 3,000개 농업 문장에서 정밀도 86.0%, 재현율 70.0%를 달성하였다[15].

2024년 ICIC에서 발표된 CaEXR은 단어 쌍 네트워크 기반 인과관계 공동 추출 프레임워크를 제안하였다[16]. 농업 분야의 인과적 기계학습 연구(2024)는 작물 생육 모델의 CO2, 온도, 수분 등 요인에 대한 민감도 분석에 인과 추론을 적용하였다[17].

### 2.6 검색 다양성 (Diversity in Retrieval)

Carbonell & Goldstein(1998)의 **MMR(Maximal Marginal Relevance)**는 관련성과 다양성의 균형을 조절하는 대표적 방법이다[18]. VRSD(2024) 연구는 MMR 대비 90% 이상의 승률을 보이는 새로운 다양성 알고리즘을 제안하였으며[19], SIGIR 2025에서는 샘플링 기반 MMR(SMMR)이 발표되었다[20].

---

## 3. 제안 방법론 (Proposed Methodology)

### 3.1 시스템 개요

본 시스템은 스마트팜 도메인에 특화된 하이브리드 RAG로, 세 가지 핵심 컴포넌트로 구성된다:

1. **3채널 검색 융합**: Dense + Sparse + PathRAG
2. **도메인 온톨로지**: 작물-환경-병해-영양소 개념 체계
3. **품질 향상 후처리**: 작물 필터링 + 시맨틱 중복 제거

```
┌────────────────────────────────────────────────────┐
│                  Query Processing                   │
│   작물 추출 → 온톨로지 매칭 → 채널 가중치 결정      │
└────────────────────────────────────────────────────┘
                         ↓
┌────────────────────────────────────────────────────┐
│               3-Channel Retrieval                   │
│                                                     │
│   Dense (α_d)    Sparse (α_s)    PathRAG (α_p)     │
│   의미 유사도     키워드 매칭     인과관계 탐색      │
└────────────────────────────────────────────────────┘
                         ↓
┌────────────────────────────────────────────────────┐
│                 Post-Processing                     │
│   스코어 정규화 → 작물 필터링 → 중복 제거 → Top-k  │
└────────────────────────────────────────────────────┘
                         ↓
┌────────────────────────────────────────────────────┐
│                  LLM Generation                     │
│              Context + Query → Answer               │
└────────────────────────────────────────────────────┘
```

### 3.2 스마트팜 온톨로지

#### 3.2.1 설계 배경

온톨로지 설계는 Stanford 온톨로지 구축 방법론[13]과 기존 농업 온톨로지 연구[9,10,11]를 참조하여 스마트팜 도메인에 적합한 6개 개념 유형을 정의하였다. CropDP-KG[12]의 엔티티 구조(작물명, 증상, 발생조건, 영향부위 등)와 AgriKG[21]의 농업 엔티티 분류를 참고하여 한국 스마트팜 환경에 맞게 구성하였다.

#### 3.2.2 개념 유형 정의

| 유형 | 설명 | 근거 |
|------|------|------|
| **crop** | 재배 작물 (토마토, 파프리카, 딸기, 와사비, 상추) | CropDP-KG의 Crops Name 엔티티[12] |
| **env** | 환경 요소 (온도, 습도, EC, pH, CO2, 광량) | 스마트팜 IoT 센서 데이터 표준[11] |
| **nutrient** | 영양소 (양액, 비료, 관수) | 농업 지식 베이스[9] |
| **disease** | 병해충 (흰가루병, 뿌리썩음병, 연부병) | CropDP-KG의 Disease/Pest 분류[12,13] |
| **stage** | 생육 단계 (육묘, 정식, 생육, 수확) | 작물 생육 모델[17] |
| **practice** | 재배 실천 (차광, 환기, 난방, 살균) | 농업 실천 온톨로지[9,10] |

각 개념은 동의어/유의어 목록(alias)을 포함한다. 이는 NLP 기반 농업 온톨로지 연구[10]에서 제안된 의미적 확장 기법을 적용한 것이다.

### 3.3 동적 채널 가중치 (Dynamic Alpha)

#### 3.3.1 설계 근거

Hybrid retrieval의 가중치 동적 조정은 DAT 연구[6]와 질의 특성 기반 적응적 융합 연구를 참조하였다. 농업 도메인에서 수치/단위 정보(EC, pH, 온도 등)의 정확한 매칭이 중요하다는 점[4]을 반영하여 Sparse 가중치를 조정한다.

#### 3.3.2 가중치 규칙

| 질의 특성 | Dense | Sparse | PathRAG | 근거 |
|----------|-------|--------|---------|------|
| 일반 질의 | 0.5 | 0.5 | 0.0 | 기본 균형[6] |
| 수치/단위 포함 | 0.3 | 0.7 | 0.0 | 수치 매칭 중요성[4] |
| 병해/재배 관련 | 0.35 | 0.35 | 0.3 | 인과관계 활성화[8] |

### 3.4 인과관계 그래프 (Causal Graph)

#### 3.4.1 설계 배경

농업 도메인에서 "고온 → 착과율 저하 → 야간 온도 관리" 같은 인과 체인이 핵심 정보 구조를 형성한다[17]. Khoo et al.[14]의 인과관계 추출 서베이와 농업 도메인 인과 추출 연구[15,16]를 참조하여 규칙 기반 패턴 매칭 방식을 적용하였다.

#### 3.4.2 인과관계 역할 분류

텍스트 패턴 매칭으로 문서의 역할을 분류한다. 패턴 설계는 한국어 인과 표현 연구와 농업 문서의 언어적 특성을 반영하였다[15].

| 역할 | 판별 패턴 | 예시 문장 |
|------|----------|----------|
| **Cause** | "원인", "때문", "~하면" | "고온 환경에서는 화분 활력이 저하된다" |
| **Effect** | "결과", "증상", "문제" | "착과율이 떨어지는 문제가 발생한다" |
| **Solution** | "관리", "해야", "방법" | "야간 온도를 18℃ 이하로 관리해야 한다" |

#### 3.4.3 인과관계 엣지 생성

CropDP-KG[12]의 관계 추출 방식을 참조하여 공통 키워드(작물, 환경요소, 병해, 상태) 기반으로 문서 간 인과관계 엣지를 자동 생성한다:

```
[Cause 문서] ──causes──▶ [Effect 문서] ──solved_by──▶ [Solution 문서]
```

PathRAG[8]의 경로 탐색 시 이 인과관계 엣지를 따라 관련 문서를 수집한다.

### 3.5 후처리 (Post-Processing)

#### 3.5.1 작물 필터링 (Crop-aware Filtering)

농업 지식 그래프 연구[4,12]에서 작물별 맥락 의존성이 강조되었다. 이를 반영하여 질의의 작물과 문서의 작물 메타데이터를 비교하여 스코어를 조정한다.

| 조건 | 스코어 조정 | 효과 |
|------|------------|------|
| 작물 일치 | +0.5 | 관련 문서 우선 |
| 작물 불일치 | ×0.15 | 무관한 작물 정보 억제 |
| 작물 정보 없음 | 유지 | 일반 정보 보존 |

#### 3.5.2 시맨틱 중복 제거 (Semantic Deduplication)

MMR[18]과 VRSD[19]를 참조하여 검색 결과의 다양성을 확보한다. 임베딩 코사인 유사도가 임계값(θ=0.85) 이상인 문서 쌍에서 후순위 문서를 제거한다.

---

## 4. 시스템 구현 (Implementation)

### 4.1 기술 스택

| 구성요소 | 기술 |
|----------|------|
| Dense Retrieval | FAISS + 임베딩 모델 |
| Sparse Retrieval | TF-IDF (scikit-learn) |
| 지식 그래프 | 커스텀 그래프 구조 (JSON) |
| LLM | 경량 LLM (llama.cpp) |
| API | FastAPI + Docker |

### 4.2 그래프 스키마

CropDP-KG[12]와 AgriKG[21]의 스키마 설계를 참조하여 구성하였다.

**노드 타입**: practice(문서), crop, env, disease, nutrient, stage

**엣지 타입**:
| 타입 | 의미 | 참조 |
|------|------|------|
| recommended_for | 작물 → 실천 | AgriKG[21] |
| associated_with | 병해 → 실천 | CropDP-KG[12] |
| mentions | 실천 → 개념 | 농업 온톨로지[10] |
| **causes** | 실천 → 실천 | 인과 추출[14,15] |
| **solved_by** | 실천 → 실천 | 인과 추출[14,15] |

---

## 5. 실험 및 평가 (Experiments)

(작성 예정)

---

## 6. 결론 (Conclusion)

(작성 예정)

---

## 참고문헌 (References)

[1] Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS 2020*.

[2] Gao, Y., et al. (2024). "Retrieval-Augmented Generation for Large Language Models: A Survey." *arXiv:2312.10997*.

[3] Peng, B., et al. (2024). "Graph Retrieval-Augmented Generation: A Survey." *arXiv:2408.08921*.

[4] Wang, X., et al. (2025). "The Application Progress and Research Trends of Knowledge Graphs and Large Language Models in Agriculture." *Computers and Electronics in Agriculture*.

[5] Karpukhin, V., et al. (2020). "Dense Passage Retrieval for Open-Domain Question Answering." *EMNLP 2020*.

[6] SIGIR (2024). "Cluster-based Partial Dense Retrieval Fused with Sparse Text Retrieval." *Proceedings of SIGIR 2024*.

[7] Edge, D., et al. (2024). "From Local to Global: A Graph RAG Approach to Query-Focused Summarization." *arXiv:2404.16130*.

[8] Chen, B., et al. (2025). "PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths." *arXiv:2502.14902*.

[9] Bhuyan, M., et al. (2021). "An Ontological Knowledge Representation for Smart Agriculture." *IEEE ICAC3N 2021*.

[10] Springer (2024). "Developing an Agriculture Ontology for Extracting Relationships from Texts Using NLP." *International Journal of Information Technology*.

[11] Cornei, L., et al. (2024). "An Ontology-Driven Solution for Capturing Spatial and Temporal Dynamics in Smart Agriculture." *RCIS 2024, Lecture Notes in Business Information Processing, vol 513*.

[12] Nature Scientific Data (2025). "A Knowledge Graph for Crop Diseases and Pests in China (CropDP-KG)." *Scientific Data*.

[13] Frontiers in Plant Science (2024). "Research on the Construction of a Knowledge Graph for Tomato Leaf Pests and Diseases Based on NER Model."

[14] Khoo, C., et al. (2022). "A Survey on Extraction of Causal Relations from Natural Language Text." *Knowledge and Information Systems*.

[15] Semi-supervised Relation Extraction in Agriculture Documents (2023). *ResearchGate*.

[16] CaEXR (2024). "A Joint Extraction Framework for Causal Relationships Based on Word-Pair Network." *ICIC 2024*.

[17] Causal Machine Learning for Sustainable Agroecosystems (2024). *arXiv:2408.13155*.

[18] Carbonell, J., & Goldstein, J. (1998). "The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries." *SIGIR 1998*.

[19] Zhang, L., et al. (2024). "VRSD: Rethinking Similarity and Diversity for Retrieval in Large Language Models." *arXiv:2407.04573*.

[20] SIGIR (2025). "SMMR: Sampling-Based MMR Reranking for Faster, More Diverse, and Balanced Recommendations." *Proceedings of SIGIR 2025*.

[21] AgriKG (2019). "An Agricultural Knowledge Graph and Its Applications." *DASFAA 2019*.
