# 관련 연구 상세 정리

본 문서는 스마트팜 RAG 시스템 관련 연구를 표 형식으로 정리한다.

---

## 1. RAG 연구 발전 과정

RAG(Retrieval-Augmented Generation)는 검색과 생성을 결합하여 LLM의 환각을 줄이는 접근법이다. 본 섹션에서는 RAG의 발전 과정을 시간순으로 정리하고, 그래프 기반 RAG와 농업 도메인 적용 연구를 포함한다.

### 1.1 초기 RAG 연구 (2020~2022)

| 연구 | 연도 | 주요 기여 | 한계점 |
|------|------|----------|--------|
| RAG (Lewis et al.) [1] | 2020 | 검색+생성 결합 → LLM이 모르는 정보도 외부 문서에서 찾아 답변, 환각 감소 | 클라우드 GPU 필수 → 농가 현장 배포 불가 |
| DPR (Karpukhin) [2] | 2020 | Dense retrieval(BERT 기반 의미 검색) → 정확한 단어 없어도 비슷한 뜻의 문서 찾음, 정확도 +15% | "EC 2.5 dS/m" 같은 수치 검색 실패 → 농업 수치 질의 부정확 |
| BEIR Benchmark [3] | 2021 | 18개 데이터셋 평가 → Hybrid(키워드+의미 결합)가 Dense만보다 우수 입증 | 농업 데이터셋 미포함 → 도메인 성능 검증 안됨 |

### 1.2 개선된 RAG 기법 (2023~2024)

| 연구 | 연도 | 주요 기여 | 한계점 |
|------|------|----------|--------|
| RAG Survey (Gao et al.) [4] | 2024 | RAG 발전 단계 분류 (Naive→Advanced→Modular) → 설계 시 참조할 체계적 가이드 제공 | 엣지/오프라인 미언급 → 네트워크 불안정 현장 적용 어려움 |
| Graph RAG Survey [5] | 2024 | 문서 간 관계를 그래프로 연결한 RAG 50편 분석 → 구조화된 지식 활용법 정리 | 농업 적용 사례 미포함 (2024년 기준) |
| GraphRAG (MS) [6] | 2024 | 문서들을 지식 그래프로 구조화 → "전체 문서 요약해줘" 같은 전역 질의(global query) 가능 | GPT-4 수천 회 호출 필요 → 문서 1000개당 $100+ 비용 |
| LightRAG [7] | 2024 | GraphRAG 경량화 → 동일 품질에 비용 10배↓ | 여전히 LLM으로 NER(Named Entity Recognition) → 엣지 실행 불가 |
| Cluster-based Fusion [8] | 2024 | Sparse(키워드 매칭)로 후보 추린 뒤 Dense 적용 → 검색 속도 2배 향상 | 2가지 방식만 결합, 가중치 고정 → 질의 유형별 최적화 불가 |
| EdgeRAG [9] | 2024 | 계층적 인덱싱(hierarchical indexing) → 필요한 임베딩만 로드하여 메모리 제약 해결 | Dense 검색만 지원 → 수치 검색 약함, 도메인 지식 활용 안함 |

### 1.3 최신 농업 도메인 RAG 연구 (2025)

| 연구 | 연도 | 주요 기여 | 한계점 |
|------|------|----------|--------|
| KG+LLM in Agriculture [10] | 2025 | 농업 지식그래프+LLM 결합 동향 최초 정리 → 농업 AI 연구 방향 제시 | 실제 배포 방안 미제시 → 연구-현장 격차 해소 안됨 |
| PathRAG (Chen) [11] | 2025 | 관계 경로(relational path) 기반 검색 → 비슷한 문서 중복 감소, 다양한 정보 제공 | 범용 관계 유형만 사용 → "원인→증상→해결책" 인과 추론 안됨 |
| Crop GraphRAG (Wu) [12] | 2025 | 병해충 KG+RAG 융합 → 환각 억제, 농업 도메인 QA 정확도 향상 | 엣지/오프라인 환경 미언급 |
| AHR-RAG (Yang) [13] | 2025 | 적응형 하이브리드 검색(단일홉/다중홉) → 91만 트리플릿 KB, 복잡 질의 대응 | 91만 트리플릿 규모 KB 전제 |
| ReG-RAG (Li) [14] | 2025 | 쿼리 재작성 + 지식그래프 강화 → 질의 의도 명확화, 검색 정확도 향상 | 클라우드 LLM 의존 → 엣지 배포 불가 |
| AgroMetLLM (Ray) [15] | 2025 | Raspberry Pi 4B에서 양자화 LLM → 오프라인 1-2초 응답 | 증발산(ET) 예측 특화 → 범용 Q&A 불가 |
| Agri-LLM (Jiang) [16] | 2025 | 정밀 지식 검색 + 협업 생성 → 농업 LLM 품질 향상 | 엣지 배포 미고려 |

**본 연구 대응:**
- **LightRAG[7] 기반 그래프 검색 프레임워크 채택** → 검증된 Dual-Level 검색(엔티티+커뮤니티) 활용
- **농업 도메인 적응**: 스마트팜 특화 엔티티 타입 6종(crop, disease, environment, practice, nutrient, stage), 농업 온톨로지 기반 엔티티 추출 가이드
- **엣지 환경 최적화**: Q4_K_M 양자화 → 8GB RAM 엣지 디바이스에서 전체 RAG 파이프라인 배포
- 작물별 검색 필터링 → "와사비 질문에 상추 문서" 문제 해결
- FAISS mmap 기반 인덱스 로딩 → 대용량 인덱스도 저메모리에서 사용 가능
- 메모리 적응형 리랭커 선택 → 가용 RAM에 맞춰 최선의 품질 제공

---

## 2. 하이브리드 검색 기법

Dense retrieval과 Sparse retrieval을 결합하여 검색 품질을 향상시키는 기법이다. Section 1에서 다룬 기초 연구(DPR[2], Cluster-based Fusion[8])를 기반으로, 본 섹션에서는 검색 결과 융합 방법론을 정리한다.

| 연구 | 연도 | 주요 기여 | 한계점 |
|------|------|----------|--------|
| RRF [17] | 2009 | 여러 검색 결과의 순위를 역수 합산(Reciprocal Rank Fusion) → 간단하게 여러 방식 통합 | 질의 특성 무시 → 수치/개념 질의 동일 처리 |
| BEIR Benchmark [3] | 2021 | 18개 데이터셋 평가 → Hybrid(키워드+의미 결합)가 Dense만보다 우수 입증 | 농업 데이터셋 미포함 → 도메인 성능 검증 안됨 |

**본 연구 대응:**
- **LightRAG의 Dual-Level 검색 활용** → 엔티티 수준(local) + 커뮤니티 수준(global) 검색 통합
- LightRAG 그래프 구조 내 벡터 검색 → Dense-Sparse 통합을 그래프 기반으로 확장
- 농업 도메인 수치 정보(온도, EC, pH 등) 매칭 최적화 필요 → 향후 연구 과제

---

## 3. 농업 도메인 지식 기반의 AI 접근 방법

농업 도메인에서 AI를 효과적으로 적용하기 위해서는 도메인 지식의 체계적 표현과 활용이 필수적이다. 본 섹션에서는 농업 지식을 구조화하는 온톨로지/지식그래프 연구와, 이를 활용한 인과관계 추출 연구를 함께 정리한다.

### 3.1 농업 온톨로지 및 지식 그래프

| 연구 | 연도 | 주요 기여 | 한계점 |
|------|------|----------|--------|
| Bhuyan et al. [18] | 2021 | 농업 개념을 래티스(lattice, 계층 격자) 구조로 정리 → 작물-환경-병해 관계 체계화 | 지식 표현만 → 실제 검색할 때 활용 방법 없음 |
| Ahmadzai et al. [19] | 2024 | 텍스트에서 RE(Relation Extraction, 엔티티 간 관계 추출)로 자동 구축 → 수작업 없이 지식 구조 생성 | RAG와 연결 안됨 → 검색 품질 향상에 기여 못함 |
| Cornei et al. [20] | 2024 | 센서 데이터를 시간순으로 표현 → 환경 변화 흐름 모델링 | 구조가 복잡함 → 실시간 검색에 부적합 |
| CropDP-KG (Yan) [21] | 2025 | NER(개체명 인식)+RE(관계 추출)로 병해충 지식 13,840개 항목 자동 구축 → 대규모 KG 구축법 제시 | 학습 데이터 수만 건 필요 → 소규모 프로젝트 적용 어려움 |
| Tomato KG (Wang) [22] | 2024 | Stanford 온톨로지 방법론 6단계 적용 → 체계적 지식그래프 구축 검증 | 토마토만 다룸 → 여러 작물 재배 농가 활용 불가 |
| AgriKG (Chen) [23] | 2019 | 농업 지식그래프 활용 사례 → 추천/진단 등 응용 가능성 제시 | 검색 시스템과 연결 안됨 → 지식 있어도 찾기 어려움 |

### 3.2 인과관계 추출 및 지식 연결

| 연구 | 연도 | 주요 기여 | 한계점 |
|------|------|----------|--------|
| Yang et al. [24] | 2022 | 인과관계 추출(Causal RE) 방법 분류 (규칙/ML/DL) → 방법 선택 가이드 제공 | RAG 통합 미고려 → 추출해도 검색에 활용 안됨 |
| Semi-sup. Agri RE [25] | 2023 | 규칙 기반 인과관계 추출 → GPU 없이 정밀도 86% 달성 | 한 문장 안에서만 추출 → 서로 다른 문서 간 연결 불가 |
| CaEXR (Liu) [26] | 2024 | 단어 쌍 네트워크로 인과관계 공동 추출 → 복잡한 관계도 F1 82% | GPU 필수 → 8GB RAM 엣지에서 실행 불가 |
| Sitokonstantinou et al. [27] | 2024 | CO2/온도가 수확량에 미치는 영향 수치화 → 환경 요인별 기여도(sensitivity) 분석 | 수치 데이터만 분석 → 텍스트 기반 "원인-해결" 추론 못함 |

**본 연구 대응:**
- **LightRAG 자동 그래프 구축 활용** → 대규모 학습 데이터 없이 LLM 기반 엔티티/관계 자동 추출
- **농업 온톨로지 기반 엔티티 추출 가이드** → 6개 유형(crop/disease/environment/practice/nutrient/stage) 도메인 특화
- 검색 단계에서 온톨로지 직접 매칭 → 쿼리의 작물/환경/병해 개념을 즉시 인식
- llama.cpp 통합으로 로컬 LLM 사용 → GPU 없이 CPU만으로 실행 가능

---

## 4. 검색 다양성 및 RAG 후처리

RAG 시스템에서 검색된 문서들을 그대로 LLM에 전달하면 중복된 정보로 인해 답변 품질이 저하된다. 검색 다양성(diversity) 기법은 RAG 후처리 단계에서 관련성과 다양성의 균형을 맞춰 최종 컨텍스트를 선별한다.

| 연구 | 연도 | 주요 기여 | 한계점 |
|------|------|----------|--------|
| MMR [28] | 1998 | MMR(Maximal Marginal Relevance) → 관련성 높은 문서 중 서로 다른 내용만 선택, 중복 없이 다양한 정보 제공 | 다양성 조절 파라미터(λ) 고정 → 작물별 중복 제거 같은 세부 요구 반영 불가 |
| VRSD [29] | 2024 | 새로운 다양성 선택 알고리즘 → MMR 대비 우수 | 일반 데이터로만 테스트 → 농업에서 효과 미검증 |
| SMMR [30] | 2025 | 샘플링 기반 MMR → 동일 품질에 로그 속도 향상(logarithmic speedup) | 작물 구분 없음 → "와사비 질문에 상추 문서" 섞여 나옴 |

**본 연구 대응:**
- **LightRAG의 ego-network 기반 효율적 그래프 탐색** → 중복 감소, 다양한 정보 제공
- 작물별 검색 필터링: 질문 작물과 일치 시 보너스, 불일치 시 패널티 → "와사비 질문에 상추 문서" 문제 해결
- 메모리 적응형 리랭커 선택 → 가용 RAM에 따라 최적 품질 제공

---

## 5. 온디바이스 엣지 환경

농업 현장에서 RAG 시스템을 실용화하려면 클라우드 의존 없이 저사양 엣지 디바이스에서 실행 가능해야 한다. 본 섹션에서는 LLM 경량화, 엣지 RAG, 스마트팜 엣지 컴퓨팅 연구를 통합하여 정리한다.

### 5.1 LLM 경량화 및 배포

| 연구 | 연도 | 주요 기여 | 한계점 |
|------|------|----------|--------|
| Edge LLM Survey [31] | 2025 | 모델 압축 기법 정리 (양자화/지식증류/프루닝) → 엣지 배포 전략 가이드 | LLM만 다룸 → 검색+답변 전체 RAG 시스템 통합 방법 없음 |
| llama.cpp [32] | 2024 | GGUF 양자화(Q4_K_M 등) → 4B 모델 기준 8GB→2.5GB (약 70%↓), 일반 CPU에서도 LLM 추론 가능 | 생성만 지원 → 문서 검색 기능은 별도 구축 필요 |
| Model2Vec [33] | 2024 | 지식 증류(knowledge distillation, 큰 모델→작은 모델 지식 전달) → 256차원 정적 벡터, 속도 100-400배↑로 실시간 임베딩 가능 | 품질 5-10%↓ → 정밀 검색 필요한 농업 질의에 부정확 |
| EmbeddingGemma [34] | 2025 | 308M 파라미터(약 600MB)로 1B급 성능 달성 → 적은 자원으로 고품질 벡터 검색 | 농업 데이터 벤치마크 없음 → 도메인 성능 보장 안됨 |

### 5.2 엣지 RAG 시스템

| 연구 | 연도 | 주요 기여 | 한계점 |
|------|------|----------|--------|
| EdgeRAG [9] | 2024 | 계층적 인덱싱(hierarchical indexing) → 필요한 임베딩만 로드하여 메모리 제약 해결 | Dense 검색만 지원 → 수치 검색 약함, 도메인 지식 활용 안함 |

### 5.3 스마트팜 엣지 컴퓨팅

| 연구 | 연도 | 주요 기여 | 한계점 |
|------|------|----------|--------|
| FarmBeats (MS) [35] | 2017 | 농업 IoT 아키텍처 설계 → 센서 데이터 수집/전송 체계화 | 센서 데이터만 처리 → 텍스트 지식 검색/LLM 기능 없음 |
| Smart Farming Review [36] | 2020 | 농업 데이터 관리 전략 종합 리뷰 → Agriculture 5.0 로드맵 제시 | 데이터 관리 중심 → 실시간 추론/RAG 미다룸 |
| Farm-LightSeek [37] | 2025 | 경량 LLM 기반 멀티모달 IoT 분석 프레임워크 → 엣지에서 크로스모달 추론 | 이미지+센서 중심 → 텍스트 문서 기반 RAG 미포함 |

**본 연구 대응:**
- **llama.cpp Q4_K_M 양자화** → 8GB RAM 환경에서 전체 RAG 파이프라인 배포
- **FAISS mmap 기반 인덱스 로딩** → 필요한 부분만 로드하여 대용량 인덱스도 저메모리에서 사용 가능
- **메모리 적응형 리랭커 선택** → 가용 RAM 감지하여 리랭커 자동 선택, 자원에 맞춰 최선의 품질 제공
- LightRAG + llama.cpp 통합 → 로컬 LLM으로 그래프 구축 및 검색 수행
- 텍스트 질문-답변 RAG 지원 → 센서/이미지만 처리하던 기존 스마트팜 엣지에 자연어 질의 기능 추가

---

## 6. 연구 공백 및 기여 요약

| 영역 | 기존 연구 한계 | 본 연구 대응 |
|------|---------------|-------------|
| **Graph RAG** | LightRAG: 범용 엔티티 타입, 도메인 특화 없음 | 농업 엔티티 타입 6종(crop, disease, environment, practice, nutrient, stage), 온톨로지 통합 |
| **Edge Deployment** | EdgeRAG: 범용 최적화, AgroMetLLM: 특정 태스크 한정 | llama.cpp Q4_K_M + FAISS mmap + 메모리 적응형 리랭커 → 8GB RAM 타겟 |
| **Agricultural KG** | CropDP-KG, AHR-RAG: 대규모 학습 데이터 필요 | LightRAG 자동 구축 + 경량 온톨로지 |
| **Evaluation** | IR metrics: Ground Truth 의존, 고비용 어노테이션 | RAGAS reference-free + 로컬 LLM → 평가 비용 최소화 |
| **검색 다양성** | 작물 구분 없음 → 다른 작물 문서 섞임 | 작물별 검색 필터링 → 와사비 질문엔 와사비 문서만 |

---

## 7. RAG 평가 및 벤치마크

RAG 시스템의 성능 평가는 검색 품질(Retrieval)과 생성 품질(Generation) 두 측면에서 이루어진다. 전통적인 평가는 수동 레이블링된 Ground Truth에 의존하지만, 최근 LLM-as-Judge 기반 Reference-free 평가 방법이 주목받고 있다.

### 7.1 전통적 평가 메트릭

| 단계 | 메트릭 | 설명 |
|------|--------|------|
| **검색 (IR)** | Precision@K, Recall@K | 상위 K개 결과의 정밀도/재현율 |
| | MRR (Mean Reciprocal Rank) | 첫 번째 관련 문서의 역순위 평균 |
| | NDCG (Normalized DCG) | 순위 품질 평가 (순위별 가중치) |
| | MAP (Mean Average Precision) | 평균 정밀도 |
| **생성 (QA)** | Exact Match (EM) | 정답과 정확히 일치 여부 |
| | F1 Score | 토큰 수준 정밀도/재현율 조화평균 |
| | ROUGE | n-gram 기반 재현율 (요약 평가용) |
| | BLEU | n-gram 기반 정밀도 (번역 평가용) |
| **의미 유사도** | BERTScore | 임베딩 기반 의미 유사도 (n-gram 한계 보완) |

### 7.2 LLM-as-Judge 기반 Reference-free 평가

| 연구 | 연도 | 학회 | 주요 기여 | 한계점 |
|------|------|------|----------|--------|
| **RAGAS** [38] | 2024 | EACL | Ground Truth 없이 Faithfulness, Answer Relevancy, Context Precision 측정 → RAG 평가 de facto 표준 | LLM judge 품질에 의존 |
| **ARES** [39] | 2024 | NAACL | Synthetic QA 자동 생성 + Fine-tuned LLM Judges | LLM API 비용 |

### 7.3 최신 RAG 벤치마크 (2024-2025)

| 벤치마크 | 연도 | 규모 | 특징 | 한계점 |
|----------|------|------|------|--------|
| **RAGBench** [40] | 2024 | 69K 예제 | 산업별 RAG 평가 지원 | 농업 미포함 |
| **CRAG** [41] | 2024 | 4,409 QA | KDD Cup 2024, 웹 검색 기반 현실적 시나리오 | 일반 도메인 |
| **GraphRAG-Bench** [42] | 2025 | - | NeurIPS 2025, Graph 구조 활용 효과 정량화 | 최신 (적용 사례 제한적) |
| **AgXQA** [43] | - | - | 농업 기술 Q&A 데이터셋 | 영어 중심, 한국어 미지원 |

### 7.4 RAGAS 메트릭 상세

| 메트릭 | Ground Truth | 평가 대상 | 설명 |
|--------|-------------|----------|------|
| **Faithfulness** | 불필요 | Generation | 답변이 검색된 context에 충실한가 (환각 여부) |
| **Answer Relevancy** | 불필요 | Generation | 답변이 질문에 관련 있는가 |
| **Context Precision** | 불필요 | Retrieval | 검색된 문서 중 관련 문서 비율 |
| **Context Recall** | 선택적 | Retrieval | 답변 생성에 필요한 정보가 context에 있는가 |
| **Answer Correctness** | 필요 | Generation | 답변이 정답과 일치하는가 |

**본 연구 대응:**
- **RAGAS 기반 Reference-free 평가** → Ground Truth 의존성 해소, 도메인 특화 데이터셋 부재 문제 우회
- **로컬 LLM 사용** → LLM API 비용 최소화
- **IR 메트릭 + RAGAS 조합**: Recall@K, MRR, NDCG (검색) + Faithfulness, Answer Relevancy (생성) → 재현 가능한 평가 체계
- 한국어 스마트팜 도메인 특화 벤치마크 부재 → 자체 평가 데이터셋 구축 및 RAGAS 활용

---

## 참고문헌

[1] Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS 2020*. ([링크](https://arxiv.org/abs/2005.11401))

[2] Karpukhin, V., et al. (2020). "Dense Passage Retrieval for Open-Domain Question Answering." *EMNLP 2020*. ([링크](https://arxiv.org/abs/2004.04906))

[3] Thakur, N., et al. (2021). "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models." *NeurIPS Datasets and Benchmarks 2021*. ([링크](https://arxiv.org/abs/2104.08663))

[4] Gao, Y., et al. (2024). "Retrieval-Augmented Generation for Large Language Models: A Survey." *arXiv:2312.10997*. ([링크](https://arxiv.org/abs/2312.10997))

[5] Peng, B., et al. (2024). "Graph Retrieval-Augmented Generation: A Survey." *arXiv:2408.08921*. ([링크](https://arxiv.org/abs/2408.08921))

[6] Edge, D., et al. (2024). "From Local to Global: A Graph RAG Approach to Query-Focused Summarization." *arXiv:2404.16130*. ([링크](https://arxiv.org/abs/2404.16130))

[7] Guo, Z., et al. (2024). "LightRAG: Simple and Fast Retrieval-Augmented Generation." *arXiv:2410.05779*. ([링크](https://arxiv.org/abs/2410.05779))

[8] Yang, Y., et al. (2024). "Cluster-based Partial Dense Retrieval Fused with Sparse Text Retrieval." *SIGIR 2024*. ([링크](https://dl.acm.org/doi/10.1145/3626772.3657972))

[9] Seemakhupt, K., et al. (2024). "EdgeRAG: Online-Indexed RAG for Edge Devices." *arXiv:2412.21023*. ([링크](https://arxiv.org/abs/2412.21023))

[10] Gong, R., & Li, X. (2025). "The Application Progress and Research Trends of Knowledge Graphs and Large Language Models in Agriculture." *Computers and Electronics in Agriculture*, 235, 110396. ([링크](https://www.sciencedirect.com/science/article/abs/pii/S0168169925005022))

[11] Chen, B., et al. (2025). "PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths." *arXiv:2502.14902*. ([링크](https://arxiv.org/abs/2502.14902))

[12] Wu, H., Xie, N., Wang, X., Fan, J., Li, Y., & Zhibo, M. (2025). "Crop GraphRAG: Pest and Disease Knowledge Base Q&A System for Sustainable Crop Protection." *Frontiers in Plant Science*. ([링크](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2025.1696872))

[13] Yang, J., Yang, W., Yang, S., He, L., & Zhang, D. (2025). "Intelligent Q&A Method for Crop Pests and Diseases Using LLM Augmented by Adaptive Hybrid Retrieval." *Smart Agriculture*. ([링크](https://www.smartag.net.cn/EN/10.12133/j.smartag.SA202506026))

[14] Li, X., Zhang, J., Zhang, H., & Nie, X. (2025). "ReG-RAG: A Large Language Model-based Question Answering Framework with Query Rewriting and Knowledge Graph Enhancement." *Smart Agriculture*. ([링크](https://www.smartag.net.cn/EN/10.12133/j.smartag.SA202507011))

[15] Ray, P. P., & Pradhan, M. P. (2025). "AgroMetLLM: An Evapotranspiration and Agro-advisory System Using Localized Large Language Models in Resource-constrained Edge." *Journal of Agrometeorology*, 27(3), 320-326. ([링크](https://doi.org/10.54386/jam.v27i3.3081))

[16] Jiang, J., Yan, L., & Liu, J. (2025). "Agricultural Large Language Model Based on Precise Knowledge Retrieval and Knowledge Collaborative Generation." *Smart Agriculture*, 7(1), 20-32. ([링크](https://doi.org/10.12133/j.smartag.SA202410025))

[17] Cormack, G. V., Clarke, C. L. A., & Büttcher, S. (2009). "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods." *SIGIR 2009*. ([링크](https://dl.acm.org/doi/10.1145/1571941.1572114))

[18] Bhuyan, M., et al. (2021). "An Ontological Knowledge Representation for Smart Agriculture." *IEEE BigData 2021*. ([링크](https://arxiv.org/abs/2112.12768))

[19] Ahmadzai, H., et al. (2024). "Innovative Agricultural Ontology Construction Using NLP." *Engineering Science and Technology, an International Journal*, 53, 101699. ([링크](https://www.sciencedirect.com/science/article/pii/S2215098624000612))

[20] Cornei, L., Cornei, D., & Foșalău, C. (2024). "An Ontology-Driven Solution for Capturing Spatial and Temporal Dynamics in Smart Agriculture." *RCIS 2024, LNBIP vol 513*. ([링크](https://link.springer.com/book/10.1007/978-3-031-59465-6))

[21] Yan, R., et al. (2025). "A Knowledge Graph for Crop Diseases and Pests in China (CropDP-KG)." *Scientific Data*. ([링크](https://www.nature.com/articles/s41597-025-04492-0))

[22] Wang, K., et al. (2024). "Research on the Construction of a Knowledge Graph for Tomato Leaf Pests and Diseases Based on NER Model." *Frontiers in Plant Science*. ([링크](https://www.frontiersin.org/articles/10.3389/fpls.2024.1482275/full))

[23] Chen, Y., Kuang, J., Cheng, D., Zheng, J., Gao, M., & Zhou, A. (2019). "AgriKG: An Agricultural Knowledge Graph and Its Applications." *DASFAA 2019, LNCS vol 11448*. ([링크](https://link.springer.com/chapter/10.1007/978-3-030-18590-9_81))

[24] Yang, J., et al. (2022). "A Survey on Extraction of Causal Relations from Natural Language Text." *Knowledge and Information Systems*. ([링크](https://arxiv.org/abs/2101.06426))

[25] IEEE (2023). "Semi-Supervised Approach for Relation Extraction in Agriculture Documents." *IEEE Conference*. ([링크](https://ieeexplore.ieee.org/document/10053800/))

[26] Liu, C., et al. (2024). "CaEXR: A Joint Extraction Framework for Causal Relationships Based on Word-Pair Network." *ICIC 2024, LNCS vol 14878*. ([링크](https://link.springer.com/chapter/10.1007/978-981-97-5672-8_38))

[27] Sitokonstantinou, V., et al. (2024). "Causal Machine Learning for Sustainable Agroecosystems." *arXiv:2408.13155*. ([링크](https://arxiv.org/abs/2408.13155))

[28] Carbonell, J., & Goldstein, J. (1998). "The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries." *SIGIR 1998*. ([링크](https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf))

[29] Gao, H., & Zhang, Y. (2024). "VRSD: Rethinking Similarity and Diversity for Retrieval in Large Language Models." *arXiv:2407.04573*. ([링크](https://arxiv.org/abs/2407.04573))

[30] Liakhnovich, K., et al. (2025). "SMMR: Sampling-Based MMR Reranking for Faster, More Diverse, and Balanced Recommendations and Retrieval." *SIGIR 2025*. ([링크](https://dl.acm.org/doi/10.1145/3726302.3730250))

[31] Xu, Z., et al. (2025). "Sustainable LLM Inference for Edge AI: Evaluating Quantized LLMs." *arXiv:2504.03360*. ([링크](https://arxiv.org/abs/2504.03360))

[32] Gerganov, G. (2024). "llama.cpp: LLM Inference in C/C++." *GitHub*. ([링크](https://github.com/ggml-org/llama.cpp))

[33] Tulkens, S., & van Dongen, T. (2024). "Model2Vec: Fast State-of-the-Art Static Embeddings." *GitHub*. ([링크](https://github.com/MinishLab/model2vec))

[34] Google Research (2025). "EmbeddingGemma: Best-in-Class Open Model for On-Device Embeddings." *Google AI*. ([링크](https://ai.google.dev/gemma/docs/embeddinggemma/model_card))

[35] Vasisht, D., et al. (2017). "FarmBeats: An IoT Platform for Data-Driven Agriculture." *NSDI 2017*. ([링크](https://www.usenix.org/conference/nsdi17/technical-sessions/presentation/vasisht))

[36] Saiz-Rubio, V., & Rovira-Más, F. (2020). "From Smart Farming Towards Agriculture 5.0: A Review on Crop Data Management." *Agronomy*, 10(2), 207. ([링크](https://www.mdpi.com/2073-4395/10/2/207))

[37] Jiang, D., et al. (2025). "Farm-LightSeek: An Edge-centric Multimodal Agricultural IoT Data Analytics Framework with Lightweight LLMs." *arXiv:2506.03168*. ([링크](https://arxiv.org/abs/2506.03168))

[38] Es, S., et al. (2024). "RAGAS: Automated Evaluation of Retrieval Augmented Generation." *EACL 2024*. ([링크](https://arxiv.org/abs/2309.15217))

[39] Saad-Falcon, J., et al. (2024). "ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems." *NAACL 2024*. ([링크](https://arxiv.org/abs/2311.09476))

[40] Fröbe, M., et al. (2024). "RAGBench: Explainable Benchmark for Retrieval-Augmented Generation Systems." *arXiv:2407.11005*. ([링크](https://arxiv.org/abs/2407.11005))

[41] Yang, X., et al. (2024). "CRAG - Comprehensive RAG Benchmark." *KDD Cup 2024*. ([링크](https://www.aicrowd.com/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024))

[42] Wu, Y., et al. (2025). "GraphRAG-Bench: Benchmarking Graph-based Retrieval Augmented Generation." *NeurIPS 2025*. ([링크](https://openreview.net/forum?id=graphrag-bench))

[43] AgXQA: Agricultural Expert Question Answering Dataset. ([링크](https://huggingface.co/datasets/agxqa))
