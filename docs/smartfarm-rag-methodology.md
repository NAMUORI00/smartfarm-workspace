# 스마트팜 도메인 특화 하이브리드 RAG 시스템

## 초록 (Abstract)

본 연구는 스마트팜 도메인에 특화된 온디바이스 하이브리드 RAG(Retrieval-Augmented Generation) 시스템을 제안한다. 기존 RAG 시스템의 클라우드 의존성과 도메인 특화 부족 문제를 해결하기 위해, (1) Dense-Sparse-PathRAG 3채널 검색 융합, (2) 농업 도메인 온톨로지 기반 개념 매칭, (3) 인과관계 그래프를 통한 컨텍스트 확장, (4) 양자화 기반 엣지 배포 최적화를 적용하였다. 작물-환경-병해-영양소-생육단계-재배실천의 6개 개념 유형으로 구성된 온톨로지와 규칙 기반 인과관계 추출을 통해 도메인 특화 검색 품질을 향상시켰다. llama.cpp GGUF 양자화(Q4_K_M)를 적용하여 8GB RAM 환경의 엣지 디바이스에서 실시간 추론이 가능하며, 오프라인 폴백 모드를 통해 네트워크 단절 환경에서도 기본적인 검색 기능을 제공한다. 제안 시스템은 스마트팜 현장의 자원 제약 환경에서 실시간 농업 의사결정 지원을 가능하게 한다.

---

## 1. 서론 (Introduction)

(작성 예정)

---

## 2. 관련 연구 (Related Work)

### 2.1 Retrieval-Augmented Generation (RAG)

RAG는 대규모 언어 모델(LLM)의 생성 과정에 외부 지식 베이스 검색을 결합하여 환각(hallucination, LLM이 사실이 아닌 내용을 마치 사실인 것처럼 생성하는 현상)을 줄이고 도메인 특화 지식을 제공하는 접근법이다[1]. Gao et al.(2024)의 서베이에 따르면 RAG 패러다임은 Naive RAG(단순 검색-생성)에서 Advanced RAG(쿼리 변환, 리랭킹 등 추가), Modular RAG(컴포넌트 조합 가능)로 진화하였으며, 검색(retrieval), 증강(augmentation), 생성(generation)의 세 축을 중심으로 발전하고 있다[2]. 최근에는 GraphRAG, PathRAG 등 문서들의 관계를 그래프로 구조화하여 검색하는 그래프 기반 RAG가 주목받고 있으며[3], 농업 분야에서도 지식 그래프와 LLM의 결합이 연구되고 있다[4].

2025년 농업 도메인 RAG 연구가 본격화되었다. **Crop GraphRAG**[33]는 병해충 지식 그래프와 RAG를 결합하여 환각을 억제하였고, **AHR-RAG**[34]는 91만 트리플릿 KB와 적응형 하이브리드 검색(단일홉/다중홉)으로 복잡 질의에 대응하였다. **ReG-RAG**[35]는 쿼리 재작성과 지식 그래프 강화를 결합하였으며, **AgroMetLLM**[36]은 Raspberry Pi에서 양자화 LLM으로 오프라인 농업 자문을 구현하였다.

**한계점:** Crop GraphRAG[33]와 AHR-RAG[34]는 엣지/오프라인 환경을 다루지 않았으며, AHR-RAG[34]는 91만 트리플릿 규모 KB를 전제한다. AgroMetLLM[36]은 Raspberry Pi에서 엣지 배포를 달성했으나 증발산(ET) 예측 및 관개 자문에 특화되어 있다.

**본 연구의 대응:** 양자화 기반 8GB RAM 엣지 배포와 오프라인 폴백 모드로 자원 제약 환경을 지원하고, 대규모 KB 없이 규칙 기반 경량 온톨로지로 농업 도메인 특화 검색을 구현한다.

### 2.2 Hybrid Retrieval

Dense retrieval(BERT 등 딥러닝 모델로 텍스트를 벡터로 변환하여 의미적 유사성으로 검색)은 "수분 스트레스"와 "물 부족"처럼 다른 표현이지만 같은 의미인 문서를 찾을 수 있으나, "EC 2.5 dS/m" 같은 수치 정보 매칭에 취약하다. 반면 Sparse retrieval(BM25, TF-IDF 등 키워드 빈도 기반 검색)은 정확한 단어가 일치해야 하므로 의미적 유사성을 놓칠 수 있지만, 수치와 단위의 정확한 매칭에 강하다[5]. Hybrid retrieval은 두 방식을 융합하여 각각의 약점을 보완한다. SIGIR 2024의 클러스터 기반 부분 Dense 검색 연구는 Sparse 검색 결과를 가이드로 활용하여 융합 최적화를 달성하였다[6].

**한계점:** 기존 Hybrid retrieval은 Dense와 Sparse 2채널 융합에 머물며, 그래프 기반 검색과의 결합이 부족하다. 또한 고정 가중치를 사용하여 질의 특성에 따른 적응적 조정이 이루어지지 않는다.

**본 연구의 대응:** Dense-Sparse-PathRAG 3채널 융합을 도입하고, 질의의 수치/단위 포함 여부와 온톨로지 매칭 결과에 따라 채널 가중치를 동적으로 조정한다.

### 2.3 Graph-based RAG

Microsoft Research의 GraphRAG(2024)는 문서들 사이의 관계를 지식 그래프(노드=개념, 엣지=관계)로 구조화하여 "전체 문서를 요약해줘"와 같은 전역적 질의(global query)에 대응한다[7]. **PathRAG**(Chen et al., 2025)는 그래프 내 관계 경로(relational path, 노드와 노드를 연결하는 엣지들의 연쇄)를 따라 검색하여 "비슷한 내용의 문서만 계속 나오는" 중복성 문제를 해결하며, 농업(Agriculture) 포함 6개 도메인에서 GraphRAG, LightRAG[32] 대비 우수한 성능을 보였다[8].

**한계점:** GraphRAG는 LLM으로 문서에서 개체(entity)와 관계를 추출해야 하므로 구축 비용이 높다(문서 1000개당 GPT-4 수천 회 호출 → $100+). PathRAG는 범용 관계 유형만 사용하여 농업 도메인의 "고수온 → 연부병 발생 → 수온 관리"와 같은 인과관계(원인-결과-해결책) 구조를 명시적으로 모델링하지 않는다. 두 방식 모두 엣지 환경에서의 경량화 고려가 부족하다.

**본 연구의 대응:** 규칙 기반 패턴 매칭으로 **인과관계 그래프** 구축 시 LLM/NER 모델 없이(비용 $0) 구축하고(단, Dense 인덱스는 임베딩 모델 사용), PathRAG[8]의 경로 탐색 개념을 차용한 경량 구현(PathRAG-lite, BFS 기반 2-hop 탐색)을 적용한다. `causes`(원인→결과), `solved_by`(문제→해결책) 엣지 타입으로 농업 도메인의 인과 구조를 명시적으로 표현한다.

### 2.4 농업 도메인 온톨로지 및 지식 그래프

농업 분야에서 온톨로지(ontology, 특정 도메인의 개념들과 그 관계를 정의한 지식 체계) 기반 지식 표현이 활발히 연구되고 있다. Bhuyan et al.(2021)은 시공간 농업 데이터에 대한 추론을 지원하는 래티스(lattice, 개념들을 계층적 격자 구조로 정리) 구조를 제안하였다[9]. NLP를 활용한 농업 온톨로지 개발 방법론[10], 스마트 농업의 시공간 역학을 포착하는 온톨로지 솔루션[11] 등이 발표되었다.

**CropDP-KG**(2025)는 NER(Named Entity Recognition, 텍스트에서 작물명/병해명/환경요소 등 주요 개체 추출)과 RE(Relation Extraction, 추출된 개체들 사이의 관계 식별)를 활용하여 13,840개 엔티티와 21,961개 관계를 자동 구축하였다[12]. 토마토 잎 병해충 지식 그래프(2024)는 Stanford 온톨로지 구축 방법론의 6단계 프로세스를 적용하였다[13].

**한계점:** 기존 농업 온톨로지는 주로 지식 표현(정리·저장)에 초점을 맞추며, RAG 시스템의 검색 단계에서 직접 활용되지 않아 지식을 정리해도 검색 품질 향상에 기여하지 못한다. 또한 NER/RE 기반 대규모 지식 그래프 구축에는 학습 데이터 수만 건과 상당한 레이블링 비용이 소요되어 소규모 프로젝트 적용이 어렵다.

**본 연구의 대응:** 경량 온톨로지(6개 개념 유형: crop, env, nutrient, disease, stage, practice)를 설계하고, 규칙 기반 패턴 매칭으로 검색 단계에서 직접 활용한다. 동의어/유의어 목록으로 커버리지를 확보하면서 구축 비용을 최소화한다.

### 2.5 인과관계 추출 (Causal Relation Extraction)

인과관계 추출(Causal Relation Extraction)은 텍스트에서 "A 때문에 B가 발생했다"와 같은 원인-결과 관계를 식별하는 NLP 태스크이다[14]. 농업 도메인에서는 WordNet과 식물 병해 정보를 활용한 동사 쌍 규칙 기반 연구가 정밀도 86.0%, 재현율 70.0%를 GPU 없이 달성하였다[15]. CaEXR(2024)은 단어 쌍 네트워크 기반 인과관계 공동 추출 프레임워크를 제안하였다[16].

**한계점:** 기존 인과관계 추출 연구는 한 문장 안에서의 관계 식별에 집중하여, "문서 A의 원인"과 "문서 B의 해결책"처럼 서로 다른 문서 간 인과관계 연결이 어렵다. 딥러닝 기반 방식(CaEXR 등)은 GPU가 필수라 8GB RAM 엣지 환경에서 실행이 불가능하다.

**본 연구의 대응:** 문서 단위로 인과관계 역할(cause/effect/solution)을 분류하고, 공통 키워드(작물, 환경요소, 병해)를 기반으로 문서 간 인과관계 그래프를 구축한다. 규칙 기반 패턴 매칭("원인", "때문", "관리", "해야" 등)으로 CPU만으로 실행 가능하다.

### 2.6 검색 다양성 (Diversity in Retrieval)

Carbonell & Goldstein(1998)의 **MMR(Maximal Marginal Relevance, 관련성 높은 문서 중 서로 내용이 다른 것만 선택하여 중복 없이 다양한 정보 제공)**는 관련성과 다양성의 균형을 조절하는 대표적 방법이다[18]. VRSD(2024)는 MMR 대비 90% 이상의 승률을 보이는 새로운 다양성 알고리즘을 제안하였다[19].

**한계점:** 기존 다양성 연구는 범용 검색에 초점을 맞추며, 농업 도메인에서 "와사비 질문에 상추 문서가 섞여 나오는" 문제나 동일 작물에 대한 비슷한 내용 중복 문제를 다루지 않는다. MMR의 다양성 조절 파라미터(λ)도 고정값을 사용하여 작물별 필터링 같은 도메인 특화 요구를 반영하지 못한다.

**본 연구의 대응:** 시맨틱 중복 제거(임베딩 코사인 유사도 85% 이상이면 같은 내용으로 판단하여 제거)와 작물 필터링(질문 작물과 일치하면 +0.5, 불일치하면 ×0.15)을 결합하여 도메인 특화 다양성을 확보한다.

### 2.7 엣지 AI 및 온디바이스 추론 (Edge AI and On-Device Inference)

#### 2.7.1 엣지 LLM 배포

엣지 디바이스에서의 LLM 추론을 위한 모델 압축 기법들이 체계적으로 연구되고 있다[22]:
- **양자화(quantization)**: 모델 가중치의 정밀도를 낮춤(FP32→INT4) → 용량 75% 감소
- **지식 증류(knowledge distillation)**: 큰 모델(teacher)의 출력을 작은 모델(student)이 학습 → 적은 파라미터로 유사 성능
- **프루닝(pruning)**: 영향력 적은 가중치 제거 → 연산량 감소

**llama.cpp**는 GGUF 양자화 포맷을 활용하여 일반 CPU 및 저사양 GPU에서 LLM 추론을 가능하게 하며[23], Q4_K_M 양자화(4비트 정밀도)는 4B 파라미터(40억 개 가중치) 모델 기준 원본 대비 메모리 약 70% 절감을 달성한다(FP16 8GB → Q4 2.5GB).

#### 2.7.2 EdgeRAG

**EdgeRAG**(2024)는 계층적 인덱싱(hierarchical indexing, 문서를 중요도/접근 빈도별로 계층화하여 필요한 것만 메모리에 로드)과 선택적 문서 로딩 전략으로 메모리 사용량을 50% 이상 감소시키면서 검색 품질을 유지하였다[24].

**한계점:** EdgeRAG는 범용 메모리 최적화에 초점을 맞추며, 도메인 특화 지식 구조를 활용하지 않는다. 단일 Dense 검색만 지원하여 "EC 2.5 dS/m" 같은 수치/단위 정보의 정확한 매칭이 어렵다.

#### 2.7.3 경량 임베딩 모델

**Model2Vec**(2024)은 지식 증류(큰 임베딩 모델의 지식을 작은 모델로 전달)를 통해 256차원 정적 벡터로 100-400배 빠른 추론 속도와 15배 작은 모델 크기를 달성하나, 품질이 5-10% 하락하여 정밀 검색이 필요한 농업 질의에는 부정확할 수 있다[25]. **EmbeddingGemma**(2024)는 308M 파라미터(약 600MB)로 1B(10억) 파라미터 이상 모델과 유사한 검색 성능을 보인다[26].

#### 2.7.4 스마트팜 엣지 컴퓨팅

Microsoft **FarmBeats**는 농업 IoT 센서 데이터 수집/전송 아키텍처를 제안하였고[27], 엣지 기반 스마트 농업 프레임워크(IoT 스트림 실시간 처리)[28], **Farm-LightSeek**(2024)의 경량 CNN 병해충 이미지 분류 프레임워크[29] 등이 연구되었다.

**한계점:** 기존 스마트팜 엣지 연구는 센서 수치 데이터 처리나 이미지 기반 병해충 탐지에 집중하며, "잎이 노랗게 변했는데 원인이 뭔가요?"와 같은 텍스트 질문-답변을 위한 RAG 기반 지식 검색과 LLM 추론의 엣지 배포는 다루지 않는다.

**본 연구의 대응:** llama.cpp Q4_K_M 양자화(메모리 75%↓), FAISS mmap(메모리 맵, 인덱스 전체를 RAM에 올리지 않고 필요한 부분만 로드), 메모리 적응형 리랭킹(가용 메모리에 따라 리랭커 자동 선택: 0.8GB 미만→비활성화, 0.8~1.5GB→경량, 1.5GB 이상→고품질)을 통해 RAG 시스템 전체의 엣지 배포를 지원한다.

### 2.8 연구 공백 및 본 연구의 기여

| 연구 영역 | 기존 연구의 한계 | 본 연구의 기여 |
|----------|-----------------|---------------|
| RAG | 클라우드 의존, 도메인 범용 | 엣지 배포, 농업 도메인 특화 |
| Hybrid Retrieval | 2채널, 고정 가중치 | 3채널 융합, 동적 가중치 |
| Graph RAG | LLM 기반 구축, 범용 관계 | 규칙 기반, 인과관계 명시화 |
| 농업 온톨로지 | 지식 표현 중심 | 검색 단계 직접 활용 |
| 인과관계 추출 | 문장 수준, 딥러닝 의존 | 문서 수준, 규칙 기반 |
| 엣지 RAG | 범용 메모리 최적화 | 도메인 특화 + 품질 향상 |

---

## 3. 제안 방법론 (Proposed Methodology)

### 3.1 시스템 개요

본 시스템은 스마트팜 도메인에 특화된 하이브리드 RAG로, 세 가지 핵심 컴포넌트로 구성된다:

1. **3채널 검색 융합**: Dense(의미 유사도 검색) + Sparse(키워드 매칭) + PathRAG-lite(인과관계 경로 탐색, BFS 기반 경량 구현)
2. **도메인 온톨로지**: 작물-환경-병해-영양소-생육단계-재배실천 6개 개념 체계
3. **품질 향상 후처리**: 작물 필터링(와사비 질문엔 와사비 문서 우선) + 시맨틱 중복 제거(비슷한 내용 중복 방지)

```
┌────────────────────────────────────────────────────┐
│                  Query Processing                   │
│   작물 추출 → 온톨로지 매칭 → 채널 가중치 결정      │
└────────────────────────────────────────────────────┘
                         ↓
┌────────────────────────────────────────────────────┐
│               3-Channel Retrieval                   │
│                                                     │
│   Dense (α_d)    Sparse (α_s)    PathRAG-lite (α_p) │
│   의미 유사도     키워드 매칭     인과관계 탐색       │
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
| **crop** | 재배 작물 (와사비, 토마토, 파프리카, 딸기, 상추) | CropDP-KG의 Crops Name 엔티티[12] |
| **env** | 환경 요소 (온도, 습도, EC, pH, CO2, 광량) | 스마트팜 IoT 센서 데이터 표준[11] |
| **nutrient** | 영양소 (양액, 비료, 관수) | 농업 지식 베이스[9] |
| **disease** | 병해충 (흰가루병, 뿌리썩음병, 연부병) | CropDP-KG의 Disease/Pest 분류[12,13] |
| **stage** | 생육 단계 (육묘, 정식, 생육, 수확) | 작물 생육 모델[17] |
| **practice** | 재배 실천 (차광, 환기, 난방, 살균) | 농업 실천 온톨로지[9,10] |

각 개념은 동의어/유의어 목록(alias)을 포함한다. 예를 들어 "와사비"의 alias에는 "산와사비", "본와사비"가 포함되어 사용자가 어떤 표현을 쓰더라도 동일 개념으로 인식한다. 이는 NLP 기반 농업 온톨로지 연구[10]에서 제안된 의미적 확장 기법을 적용한 것이다.

### 3.3 동적 채널 가중치 (Dynamic Alpha)

#### 3.3.1 설계 근거

세 검색 채널의 결과를 통합할 때 각 채널에 가중치(α)를 부여한다. 예를 들어 α_d=0.5, α_s=0.5이면 Dense와 Sparse 결과를 동등하게 반영한다. Hybrid retrieval의 가중치 동적 조정은 DAT 연구[6]와 질의 특성 기반 적응적 융합 연구를 참조하였다. 농업 도메인에서 수치/단위 정보(EC, pH, 온도 등)의 정확한 매칭이 중요하다는 점[4]을 반영하여 Sparse 가중치를 조정한다.

#### 3.3.2 가중치 규칙

질의 내용을 분석하여 가중치를 자동 결정한다:

| 질의 특성 | Dense | Sparse | PathRAG | 왜 이 가중치인가 |
|----------|-------|--------|---------|-----------------|
| 일반 질의 | 0.5 | 0.5 | 0.0 | 의미 검색과 키워드 매칭을 균형있게 활용 |
| 수치/단위 포함 ("EC 2.5", "25℃") | 0.3 | 0.7 | 0.0 | 수치는 정확히 일치해야 하므로 키워드 매칭 강화 |
| 병해/재배 관련 ("흰가루병 원인") | 0.35 | 0.35 | 0.3 | 원인→증상→해결책 연결을 위해 인과관계 검색 활성화 |

### 3.4 인과관계 그래프 (Causal Graph)

#### 3.4.1 설계 배경

농업 도메인에서 "고수온 → 연부병 발생 → 수온 관리" 같은 인과 체인이 핵심 정보 구조를 형성한다[17]. Yang et al.[14]의 인과관계 추출 서베이와 농업 도메인 인과 추출 연구[15,16]를 참조하여 규칙 기반 패턴 매칭 방식을 적용하였다.

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

MMR[18]과 VRSD[19]를 참조하여 검색 결과의 다양성을 확보한다. 두 문서의 임베딩 벡터 간 코사인 유사도(cosine similarity, 두 벡터가 같은 방향을 가리키는 정도로 1에 가까울수록 내용이 비슷함)가 임계값(θ=0.85, 85% 이상 유사하면 같은 내용으로 간주) 이상인 문서 쌍에서 후순위 문서를 제거한다.

### 3.6 엣지 배포 최적화 (Edge Deployment Optimization)

#### 3.6.1 설계 배경

스마트팜 현장은 네트워크 연결이 불안정하거나 부재한 환경이 많다. 온실, 비닐하우스, 야외 재배지에서의 실시간 의사결정 지원을 위해 클라우드 의존성을 최소화하는 엣지 배포가 필수적이다[27,28]. EdgeRAG[24]와 경량화 연구[22,25]를 참조하여 자원 제약 환경에 최적화된 배포 전략을 설계하였다.

#### 3.6.2 LLM 양자화 전략

양자화(quantization)는 모델 가중치의 숫자 정밀도를 낮추는 기법이다. FP16(16비트 소수점)을 INT4(4비트 정수)로 줄이면 용량이 1/4로 감소한다:

| 양자화 수준 | 메모리 (4B 모델) | 품질 손실 | 적용 환경 |
|------------|-----------------|----------|----------|
| FP16 (원본, 16비트) | ~8GB | 없음 | 서버 환경 (GPU 필수) |
| INT8 (8비트) | ~4GB | 최소 | 고사양 엣지 (8GB RAM) |
| **Q4_K_M** (4비트 혼합) | ~2.5GB | 낮음 | 일반 엣지 디바이스 (4GB RAM) |
| Q2_K (2비트) | ~1.5GB | 중간 | 극저사양 환경 (2GB RAM) |

본 시스템은 llama.cpp의 GGUF 포맷[23]을 활용하여 Q4_K_M 양자화를 기본으로 적용한다. Q4_K_M은 중요한 레이어는 5비트, 나머지는 4비트로 혼합 양자화하여 품질 대비 메모리 효율의 최적점으로 평가된다.

#### 3.6.3 경량 임베딩 선택

Dense retrieval의 임베딩 모델(텍스트를 숫자 벡터로 변환하는 모델) 선택은 엣지 성능에 직접적 영향을 미친다. 세 가지 전략을 고려하였다:

| 전략 | 모델 예시 | 추론 속도 | 품질 | 적용 환경 |
|------|----------|----------|------|----------|
| Static Embedding (고정 벡터 조합) | Model2Vec[25] | 매우 빠름 (CPU만) | 중간 | 극저사양 (2GB RAM) |
| Small Transformer (경량 신경망) | MiniLM-L6 | 빠름 | 양호 | 일반 엣지 (4GB RAM) |
| Distilled Model (지식 증류 모델) | EmbeddingGemma[26] | 중간 | 우수 | 고사양 엣지 (8GB RAM) |

오프라인 환경에서는 Static Embedding과 TF-IDF Sparse 검색의 조합으로 LLM 없이도 기본적인 문서 검색이 가능하도록 설계하였다(검색만 가능, 답변 생성은 불가).

#### 3.6.4 계층적 배포 아키텍처

EdgeRAG[24]의 계층적 접근을 참조하여 삼중 배포 구조를 설계하였다:

```
┌─────────────────────────────────────────────────────────────┐
│                   Cloud (Optional)                          │
│   Full RAG + Large LLM + 전체 문서 인덱스                   │
└─────────────────────────────────────────────────────────────┘
                              ↑ 동기화 (네트워크 가용 시)
┌─────────────────────────────────────────────────────────────┐
│                  Edge Gateway                               │
│   Hybrid Retrieval + Quantized LLM (Q4_K_M)                │
│   작물별 서브셋 인덱스 + 인과관계 그래프                    │
└─────────────────────────────────────────────────────────────┘
                              ↑ 로컬 네트워크
┌─────────────────────────────────────────────────────────────┐
│                  IoT Sensor Node                            │
│   센서 데이터 수집 + 이상 탐지 (규칙 기반)                  │
└─────────────────────────────────────────────────────────────┘
```

#### 3.6.5 오프라인 폴백 모드

네트워크 단절 또는 메모리 부족 시 다음과 같은 폴백(fallback, 대체 동작) 전략을 적용한다:

| 폴백 단계 | 동작 | 언제 사용 |
|----------|------|----------|
| **검색 전용 모드** | LLM 없이 Sparse 검색 결과(관련 문서 목록)만 반환 | LLM 로드 불가 시 |
| **캐시 응답** | 동일/유사 질의에 대한 이전 응답 재활용 | 반복 질의 시 |
| **규칙 기반 응답** | 온톨로지 매칭 결과로 정형 응답 생성 (예: "와사비 고수온 피해 → 수온 관리 권장") | 간단한 조회 시 |

이는 Farm-LightSeek[29]의 필드 환경 적응 전략을 참고하였다.

---

## 4. 시스템 구현 (Implementation)

### 4.1 기술 스택

| 구성요소 | 서버 환경 | 엣지 환경 | 참조 |
|----------|----------|----------|------|
| Dense Retrieval | FAISS + Qwen3-Embedding-0.6B (6억 파라미터, ~1.2GB) | FAISS + MiniLM-L6 (2,200만 파라미터, ~90MB) | [25,26] |
| Sparse Retrieval | TF-IDF (scikit-learn) | TF-IDF (동일) | - |
| 지식 그래프 | 커스텀 그래프 (JSON) | 서브셋 그래프 | - |
| LLM | llama.cpp (FP16/INT8) | llama.cpp (Q4_K_M) | [23] |
| API | FastAPI + Docker | FastAPI (경량) | - |
| 오프라인 폴백 | - | 캐시 + 규칙 기반 | [24,29] |

### 4.2 핵심 모듈 구현

#### 4.2.1 HybridDATRetriever

3채널 검색 융합과 후처리를 담당하는 핵심 리트리버이다.

**주요 파라미터:**
| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `DEDUP_THRESHOLD` | 0.85 | 시맨틱 중복 판단 임계값 |
| `CROP_MATCH_BONUS` | 0.5 | 작물 일치 시 스코어 보너스 |
| `CROP_MISMATCH_PENALTY` | 0.85 | 작물 불일치 시 패널티 계수 |

**동적 가중치 계산 (`dynamic_alphas`):**
- 온톨로지 매칭 결과와 수치/단위 패턴을 분석하여 3채널 가중치 반환
- LLM 호출 없이 규칙 기반으로 동작하여 엣지 환경 최적화

#### 4.2.2 PathRAGRetriever (PathRAG-lite)

PathRAG[8]의 경로 탐색 개념을 차용한 경량 구현이다. 원본 PathRAG의 relational path pruning 대신 BFS(Breadth-First Search, 너비 우선 탐색: 가까운 노드부터 순서대로 방문) 기반 단순화된 탐색을 수행한다.

**탐색 전략:**
- 시작점: 쿼리에서 매칭된 온톨로지 개념 노드 (예: "와사비 고수온" → crop:와사비, env:고수온)
- 최대 깊이: 2-hop (2번까지 엣지를 따라 이동, 기본값)
- 인과관계 엣지(`causes`, `solved_by`) 우선 탐색하여 원인→결과→해결책 문서 수집

#### 4.2.3 GraphBuilder

문서 인제스트 시 인과관계 그래프를 자동 구축한다.

**인과관계 패턴:**
```
CAUSE_PATTERNS: "원인", "이유", "때문", "~하면", "높으면", "낮으면"
EFFECT_PATTERNS: "결과", "영향", "증상", "문제", "장애", "저하"
SOLUTION_PATTERNS: "해결", "대응", "방법", "조치", "관리", "예방"
```

**엣지 생성 로직:**
1. 각 문서의 인과관계 역할 탐지 (cause/effect/solution)
2. 공통 키워드(작물, 환경요소, 병해, 상태) 추출
3. 키워드 교집합이 존재하는 문서 쌍에 엣지 생성

#### 4.2.4 EmbeddingRetriever

FAISS(Facebook AI Similarity Search, 벡터 유사도 검색 라이브러리) 기반 Dense 검색을 담당한다.

**특징:**
- Lazy loading(지연 로딩): 시작 시가 아닌 첫 쿼리 시점에 모델 로드 → 초기 메모리 절약
- L2 정규화된 임베딩으로 코사인 유사도 검색 (벡터 길이 1로 맞춰 방향만 비교)
- mmap(memory-mapped file, 파일을 메모리에 통째로 올리지 않고 필요한 부분만 로드) 지원으로 대용량 인덱스도 저메모리에서 사용 가능

### 4.3 메모리 적응형 리랭킹

리랭킹(reranking)은 검색된 문서들을 다시 정렬하여 가장 관련 높은 문서를 상위로 올리는 과정이다. 런타임 가용 메모리에 따라 리랭커를 동적으로 선택한다:

| 가용 RAM | 리랭커 | 설명 |
|----------|--------|------|
| < 0.8GB | none | 리랭킹 비활성화 (검색 결과 그대로 사용) |
| 0.8GB ~ 1.5GB | LLM-lite | llama.cpp 기반 경량 리랭킹 (쿼리-문서 관련성 재평가) |
| ≥ 1.5GB | BGE | BGE-reranker-v2-m3 (BERT 기반 고품질 리랭킹) |

**설정 파라미터:**
- `AUTO_RERANK_MIN_RAM_GB`: 0.8
- `AUTO_BGE_MIN_RAM_GB`: 1.5
- `AUTO_BGE_MIN_VRAM_GB`: 1.5 (GPU 사용 시)

### 4.4 인덱스 영속화

오프라인 환경 지원을 위해 인덱스(검색용 데이터 구조)를 파일로 저장/로드한다. 시스템 재시작 시 문서를 다시 처리하지 않고 저장된 인덱스를 바로 로드한다:

| 파일 | 내용 | 형식 |
|------|------|------|
| `dense.faiss` | 문서 임베딩 벡터 인덱스 | Binary (mmap 가능, 부분 로드) |
| `dense_docs.jsonl` | 문서 텍스트 및 메타데이터 | JSON Lines (한 줄에 한 문서) |
| `sparse.pkl` | TF-IDF 키워드 빈도 행렬 | Pickle (Python 직렬화) |

### 4.5 그래프 스키마

CropDP-KG[12]와 AgriKG[21]의 스키마 설계를 참조하여 구성하였다.

**노드 타입**: practice(문서), crop, env, disease, nutrient, stage

**엣지 타입:**
| 타입 | 의미 | 참조 |
|------|------|------|
| recommended_for | 작물 → 실천 | AgriKG[21] |
| associated_with | 병해 → 실천 | CropDP-KG[12] |
| mentions | 실천 → 개념 | 농업 온톨로지[10] |
| **causes** | 실천 → 실천 | 인과 추출[14,15] |
| **solved_by** | 실천 → 실천 | 인과 추출[14,15] |

### 4.6 엣지 배포 사양

| 환경 | 최소 사양 | 권장 사양 | 지원 기능 |
|------|----------|----------|----------|
| **서버** | 32GB RAM, GPU | 64GB RAM, RTX 4090 | 전체 기능 |
| **엣지 게이트웨이** | 8GB RAM, CPU | 16GB RAM, CPU/NPU | RAG + Q4 LLM |
| **저사양 엣지** | 4GB RAM | 8GB RAM | 검색 전용 |
| **IoT 노드** | 512MB RAM | 1GB RAM | 센서 + 규칙 |

### 4.7 EdgeRAG와의 구현 비교

| 구분 | EdgeRAG[24] | 본 시스템 |
|------|-------------|----------|
| **최적화 초점** | 범용 메모리 최적화 | 도메인 특화 + 엣지 배포 |
| **인덱싱 전략** | 온라인 계층적 인덱싱 | 오프라인 사전 인덱싱 + FAISS mmap |
| **검색 채널** | 단일 Dense | Dense + Sparse + PathRAG 3채널 |
| **그래프 활용** | 없음 | 인과관계 그래프 (causes, solved_by) |
| **도메인 지식** | 범용 | 농업 온톨로지 6개 유형 |
| **메모리 절감** | 계층적 로딩으로 50%↓ | 양자화(Q4_K_M)로 75%↓ + mmap |
| **오프라인 지원** | 제한적 | Sparse 검색 + 캐시 폴백 |
| **품질 향상** | 메모리 효율 우선 | 작물 필터링, 중복 제거 |

**핵심 차별점:**
1. **도메인 특화**: EdgeRAG가 범용 메모리 최적화에 집중하는 반면, 본 시스템은 농업 온톨로지와 인과관계 그래프를 활용하여 검색 품질 향상
2. **멀티 채널**: 수치/단위 정보(EC, pH)의 정확한 매칭을 위한 Sparse 채널 유지
3. **메모리 적응형**: 런타임 가용 메모리에 따른 동적 리랭커 선택

---

## 5. 실험 및 평가 (Experiments)

(작성 예정)

## 6. 결론 (Conclusion)

(작성 예정)

---

## 참고문헌 (References)

[1] Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS 2020*. ([링크](https://arxiv.org/abs/2005.11401))

[2] Gao, Y., et al. (2024). "Retrieval-Augmented Generation for Large Language Models: A Survey." *arXiv:2312.10997*. ([링크](https://arxiv.org/abs/2312.10997))

[3] Peng, B., et al. (2024). "Graph Retrieval-Augmented Generation: A Survey." *arXiv:2408.08921*. ([링크](https://arxiv.org/abs/2408.08921))

[4] Gong, R., & Li, X. (2025). "The Application Progress and Research Trends of Knowledge Graphs and Large Language Models in Agriculture." *Computers and Electronics in Agriculture*, 235, 110396. ([링크](https://www.sciencedirect.com/science/article/abs/pii/S0168169925005022))

[5] Karpukhin, V., et al. (2020). "Dense Passage Retrieval for Open-Domain Question Answering." *EMNLP 2020*. ([링크](https://arxiv.org/abs/2004.04906))

[6] Yang, Y., et al. (2024). "Cluster-based Partial Dense Retrieval Fused with Sparse Text Retrieval." *SIGIR 2024*. ([링크](https://dl.acm.org/doi/10.1145/3626772.3657972))

[7] Edge, D., et al. (2024). "From Local to Global: A Graph RAG Approach to Query-Focused Summarization." *arXiv:2404.16130*. ([링크](https://arxiv.org/abs/2404.16130))

[8] Chen, B., et al. (2025). "PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths." *arXiv:2502.14902*. ([링크](https://arxiv.org/abs/2502.14902))

[9] Bhuyan, M., et al. (2021). "An Ontological Knowledge Representation for Smart Agriculture." *IEEE BigData 2021*. ([링크](https://arxiv.org/abs/2112.12768))

[10] Ahmadzai, H., et al. (2024). "Innovative Agricultural Ontology Construction Using NLP." *Engineering Science and Technology, an International Journal*, 53, 101699. ([링크](https://www.sciencedirect.com/science/article/pii/S2215098624000612))

[11] Cornei, L., Cornei, D., & Foșalău, C. (2024). "An Ontology-Driven Solution for Capturing Spatial and Temporal Dynamics in Smart Agriculture." *RCIS 2024, LNBIP vol 513*. ([링크](https://link.springer.com/book/10.1007/978-3-031-59465-6))

[12] Yan, R., et al. (2025). "A Knowledge Graph for Crop Diseases and Pests in China (CropDP-KG)." *Scientific Data*. ([링크](https://www.nature.com/articles/s41597-025-04492-0))

[13] Wang, K., et al. (2024). "Research on the Construction of a Knowledge Graph for Tomato Leaf Pests and Diseases Based on NER Model." *Frontiers in Plant Science*. ([링크](https://www.frontiersin.org/articles/10.3389/fpls.2024.1482275/full))

[14] Yang, J., et al. (2022). "A Survey on Extraction of Causal Relations from Natural Language Text." *Knowledge and Information Systems*. ([링크](https://arxiv.org/abs/2101.06426))

[15] IEEE (2023). "Semi-Supervised Approach for Relation Extraction in Agriculture Documents." *IEEE Conference*. ([링크](https://ieeexplore.ieee.org/document/10053800/))

[16] Liu, C., et al. (2024). "CaEXR: A Joint Extraction Framework for Causal Relationships Based on Word-Pair Network." *ICIC 2024, LNCS vol 14878*. ([링크](https://link.springer.com/chapter/10.1007/978-981-97-5672-8_38))

[17] Sitokonstantinou, V., et al. (2024). "Causal Machine Learning for Sustainable Agroecosystems." *arXiv:2408.13155*. ([링크](https://arxiv.org/abs/2408.13155))

[18] Carbonell, J., & Goldstein, J. (1998). "The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries." *SIGIR 1998*. ([링크](https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf))

[19] Gao, H., & Zhang, Y. (2024). "VRSD: Rethinking Similarity and Diversity for Retrieval in Large Language Models." *arXiv:2407.04573*. ([링크](https://arxiv.org/abs/2407.04573))

[20] Ananieva, E., et al. (2025). "SMMR: Sampling-Based MMR Reranking for Faster, More Diverse, and Balanced Recommendations." *SIGIR 2025*. ([링크](https://dl.acm.org/doi/10.1145/3726302.3730250))

[21] Chen, Y., Kuang, J., Cheng, D., Zheng, J., Gao, M., & Zhou, A. (2019). "AgriKG: An Agricultural Knowledge Graph and Its Applications." *DASFAA 2019, LNCS vol 11448*. ([링크](https://link.springer.com/chapter/10.1007/978-3-030-18590-9_81))

[22] Xu, Z., et al. (2025). "Sustainable LLM Inference for Edge AI: Evaluating Quantized LLMs." *arXiv:2504.03360*. ([링크](https://arxiv.org/abs/2504.03360))

[23] Gerganov, G. (2024). "llama.cpp: LLM Inference in C/C++." *GitHub*. ([링크](https://github.com/ggml-org/llama.cpp))

[24] Seemakhupt, K., et al. (2024). "EdgeRAG: Online-Indexed RAG for Edge Devices." *arXiv:2412.21023*. ([링크](https://arxiv.org/abs/2412.21023))

[25] Tulkens, S., & van Dongen, T. (2024). "Model2Vec: Fast State-of-the-Art Static Embeddings." *GitHub*. ([링크](https://github.com/MinishLab/model2vec))

[26] Google Research (2025). "EmbeddingGemma: Powerful and Lightweight Text Representations." *Google AI*. ([링크](https://ai.google.dev/gemma/docs/embeddinggemma/model_card))

[27] Vasisht, D., et al. (2017). "FarmBeats: An IoT Platform for Data-Driven Agriculture." *NSDI 2017*. ([링크](https://www.usenix.org/conference/nsdi17/technical-sessions/presentation/vasisht))

[28] Saiz-Rubio, V., & Rovira-Más, F. (2020). "From Smart Farming Towards Agriculture 5.0: A Review on Crop Data Management." *Agronomy*, 10(2), 207. ([링크](https://www.mdpi.com/2073-4395/10/2/207))

[29] Jiang, D., et al. (2025). "Farm-LightSeek: An Edge-centric Multimodal Agricultural IoT Data Analytics Framework with Lightweight LLMs." *arXiv:2506.03168*. ([링크](https://arxiv.org/abs/2506.03168))

[30] Cormack, G. V., Clarke, C. L. A., & Büttcher, S. (2009). "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods." *SIGIR 2009*. ([링크](https://dl.acm.org/doi/10.1145/1571941.1572114))

[31] Thakur, N., et al. (2021). "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models." *NeurIPS Datasets and Benchmarks 2021*. ([링크](https://arxiv.org/abs/2104.08663))

[32] Guo, Z., et al. (2024). "LightRAG: Simple and Fast Retrieval-Augmented Generation." *arXiv:2410.05779*. ([링크](https://arxiv.org/abs/2410.05779))

[33] Wu, H., Xie, N., Wang, X., Fan, J., Li, Y., & Zhibo, M. (2025). "Crop GraphRAG: Pest and Disease Knowledge Base Q&A System for Sustainable Crop Protection." *Frontiers in Plant Science*. ([링크](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2025.1696872))

[34] Yang, J., Yang, W., Yang, S., He, L., & Zhang, D. (2025). "Intelligent Q&A Method for Crop Pests and Diseases Using LLM Augmented by Adaptive Hybrid Retrieval." *Smart Agriculture*. ([링크](https://www.smartag.net.cn/EN/10.12133/j.smartag.SA202506026))

[35] Li, X., Zhang, J., Zhang, H., & Nie, X. (2025). "ReG-RAG: A Large Language Model-based Question Answering Framework with Query Rewriting and Knowledge Graph Enhancement." *Smart Agriculture*. ([링크](https://www.smartag.net.cn/EN/10.12133/j.smartag.SA202507011))

[36] Ray, P. P., & Pradhan, M. P. (2025). "AgroMetLLM: An Evapotranspiration and Agro-advisory System Using Localized Large Language Models in Resource-constrained Edge." *Journal of Agrometeorology*, 27(3), 320-326. ([링크](https://doi.org/10.54386/jam.v27i3.3081))

[37] Jiang, J., Yan, L., & Liu, J. (2025). "Agricultural Large Language Model Based on Precise Knowledge Retrieval and Knowledge Collaborative Generation." *Smart Agriculture*, 7(1), 20-32. ([링크](https://doi.org/10.12133/j.smartag.SA202410025))
