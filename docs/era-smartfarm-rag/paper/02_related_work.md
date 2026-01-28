# 2. Related Work

본 연구는 RAG 시스템, 하이브리드 검색, 그래프 기반 지식 표현, 엣지 배포의 교차점에 위치한다. 본 장에서는 각 영역의 최근 발전과 한계를 검토하고, 연구 공백을 식별한다.

## 2.1 RAG and Graph-based Retrieval

Retrieval-Augmented Generation(RAG)은 외부 지식 검색을 LLM 생성과 결합하여 환각을 줄이는 패러다임이다 [1]. Gao et al.의 서베이 [2]는 RAG 발전을 Naive RAG(단순 검색-생성), Advanced RAG(쿼리 변환, 리랭킹), Modular RAG(컴포넌트 조합)로 분류하였다.

최근 그래프 기반 RAG가 주목받고 있다. **GraphRAG** [7]는 커뮤니티 탐지로 문서를 클러스터링하여 전역 질의에 대응하나, LLM 기반 엔티티 추출에 높은 비용이 소요된다(1K 문서당 GPT-4 수천 회 호출, $100+). **LightRAG** [32]는 Dual-Level 검색(엔티티 수준 + 커뮤니티 수준)과 ego-network 기반 효율적 그래프 탐색으로 GraphRAG 대비 검색 효율성을 크게 개선하였으며, 로컬 LLM과의 통합이 용이하다. 그러나 범용 엔티티 타입(person, organization, location 등)을 사용하여 농업 등 특정 도메인에 대한 최적화가 필요하다. **PathRAG** [8]는 관계 경로 기반 검색으로 중복을 줄이며 농업 포함 6개 도메인에서 우수한 성능을 보였다.

2025년 농업 도메인 RAG 연구가 본격화되었다. **Crop GraphRAG** [33]는 병해충 지식 그래프와 RAG를 결합하였고, **AHR-RAG** [34]는 91만 트리플릿 KB로 복잡 질의에 대응하였다. **AgroMetLLM** [36]은 Raspberry Pi에서 양자화 LLM 기반 오프라인 농업 자문을 구현하였으나 증발산 예측에 특화되어 있다. 이들 연구는 엣지 환경에서의 범용 농업 Q&A를 다루지 않는다.

## 2.2 Hybrid Retrieval

Dense retrieval은 의미적 유사성 매칭에 강하나 수치/단위 정보 매칭에 취약하고 [5], Sparse retrieval(BM25)은 그 반대 특성을 보인다. Hybrid retrieval은 두 방식을 융합하여 상호 보완한다 [6].

융합 전략으로 **Reciprocal Rank Fusion(RRF)** [37]과 **Convex Combination** [6]이 널리 사용된다. Bruch et al. [6]은 RRF가 파라미터에 민감하며, Convex Combination이 in-domain/out-of-domain 모두에서 우수함을 보였다. SIGIR 2024의 클러스터 기반 연구는 Sparse 결과를 가이드로 활용한 융합 최적화를 제안하였다.

기존 하이브리드 검색은 Dense-Sparse 2채널 융합에 머물며, 그래프 기반 검색과의 통합은 제한적이다. LightRAG는 그래프 구조 내에서 벡터 검색을 수행하여 이러한 통합을 시도하나, 농업 도메인의 수치 정보(온도, EC, pH 등) 매칭에 대한 최적화는 부족하다.

## 2.3 Agricultural Knowledge Systems

농업 온톨로지 연구는 지식 표현에 초점을 맞춰왔다. Bhuyan et al. [9]은 시공간 농업 데이터 추론을 위한 래티스 구조를 제안하였고, 스마트 농업 온톨로지 [11]와 NLP 기반 개발 방법론 [10]이 발표되었다. **CropDP-KG** [12]는 NER/RE로 13,840 엔티티와 21,961 관계를 구축하였으나, 수만 건의 학습 데이터와 레이블링 비용이 필요하다.

인과관계 추출 연구 [14-16]는 문장 수준 관계 식별에 집중하며, 문서 간 인과관계 연결("문서 A의 원인 → 문서 B의 해결책")을 다루지 않는다. 딥러닝 기반 방식 [16]은 GPU가 필수라 엣지 환경에서 실행이 불가능하다.

기존 농업 온톨로지는 검색 단계에서 직접 활용되지 않아 지식 정리가 검색 품질 향상에 기여하지 못한다.

## 2.4 Edge Deployment for RAG

엣지 LLM 배포를 위한 압축 기법으로 양자화, 지식 증류, 프루닝이 연구되고 있다 [22]. **llama.cpp** [23]는 GGUF 양자화로 CPU/저사양 GPU에서 LLM 추론을 가능하게 하며, Q4_K_M 양자화는 메모리를 약 70% 절감한다.

**EdgeRAG** [24]는 계층적 인덱싱과 선택적 로딩으로 메모리 50%+ 감소를 달성하였으나, 도메인 특화 지식을 활용하지 않으며 단일 Dense 검색만 지원한다. **Model2Vec** [25]는 100-400배 빠른 추론을 달성하나 품질이 5-10% 하락한다.

스마트팜 엣지 연구로 **FarmBeats** [27]의 IoT 아키텍처, **Farm-LightSeek** [29]의 경량 CNN 병해충 분류가 있으나, 텍스트 Q&A를 위한 RAG 기반 지식 검색의 엣지 배포는 다루지 않는다.

## 2.5 RAG Evaluation

전통적 RAG 평가는 검색 단계에서 Precision@K, Recall@K, MRR, NDCG, MAP 등 IR 메트릭을, 생성 단계에서 Exact Match(EM), F1 Score, ROUGE, BLEU 등을 사용한다. BERTScore는 임베딩 기반 의미 유사도를 측정하여 n-gram 한계를 보완한다. 그러나 이들 메트릭은 Ground Truth 구축에 높은 비용이 소요되며, 도메인 특화 데이터셋 부재 시 적용이 어렵다.

**RAGAS** [38]는 LLM-as-Judge 기반 reference-free 평가 프레임워크로 Faithfulness, Answer Relevancy, Context Precision을 측정하며, EACL 2024에서 발표되어 RAG 평가의 de facto 표준으로 자리잡았다. **ARES** [39]는 Synthetic QA 생성과 Fine-tuned LLM Judge를 결합하며 NAACL 2024에서 발표되었다.

2024-2025년 RAG 벤치마크가 다양화되었다. **RAGBench** [40]는 69K 예제로 산업별 RAG 평가를 지원하고, **CRAG** [41]은 웹 검색 기반 4,409개 QA 쌍으로 현실적 시나리오를 평가한다. Graph RAG 특화 평가로 **GraphRAG-Bench** [42]가 NeurIPS 2025에서 발표되어 그래프 구조 활용 효과를 정량화한다.

농업 도메인 벤치마크는 제한적이다. **AgXQA** [43]는 농업 기술 Q&A 데이터셋이나 영어 중심이며, 한국어 스마트팜 도메인 특화 벤치마크는 부재하다. 이에 본 연구는 RAGAS 기반 reference-free 평가와 표준 IR 메트릭을 조합하여 재현 가능한 평가 체계를 구축한다.

## 2.6 Research Gap and Our Contributions

Table 1은 기존 연구와 본 연구의 차별점을 요약한다.

**Table 1. Comparison with Existing Approaches**

| Aspect | Prior Work | Gap | Our Approach |
|--------|-----------|-----|--------------|
| **Graph RAG** | LightRAG [32], PathRAG [8] | 범용 엔티티 타입, 도메인 특화 없음 | HybridDAT: Dense+Sparse+PathRAG 3채널 융합, 농업 온톨로지 통합 |
| **Edge Deployment** | EdgeRAG [24], AgroMetLLM [36] | 범용 최적화, 특정 태스크 한정 | llama.cpp Q4_K_M + FAISS mmap, 8GB RAM 타겟 |
| **Agricultural KG** | CropDP-KG [12], AHR-RAG [34] | 대규모 학습 데이터 필요 | HybridGraphBuilder (Rule + LLM 기반 인과관계 추출) |
| **Evaluation** | IR metrics with Ground Truth | 고비용 어노테이션 | RAGAS reference-free + 로컬 LLM |

본 연구는 기존 Graph RAG 연구(LightRAG [32], PathRAG [8])의 개념을 참고하되, **농업 도메인 특화 하이브리드 검색 시스템(HybridDAT)**을 독자적으로 설계한다:

1. **Dense-Sparse-Graph 3채널 융합 (HybridDATRetriever)**: RRF 기반 점수 융합과 DAT(Dynamic Alpha Tuning)를 통한 질의 적응형 가중치 조정. 농업 온톨로지 매칭으로 도메인 관련성 강화
2. **하이브리드 그래프 구축 (HybridGraphBuilder)**: 규칙 기반 패턴 매칭과 LLM 기반 인과관계 추출(CausalExtractor)을 결합하여 농업 지식 그래프 자동 구축. 농업 엔티티 6종(crop, disease, environment, practice, nutrient, stage) 지원
3. **엣지 환경 최적화**: llama.cpp Q4_K_M 양자화로 8GB RAM 환경 지원, FAISS mmap 기반 인덱스 로딩, 메모리 적응형 리랭커 선택
4. **Reference-free 평가**: RAGAS 기반 평가 파이프라인으로 Ground Truth 의존성 해소, 로컬 LLM으로 평가 비용 최소화

Baseline 비교로 LightRAG [32]와 직접 성능 비교를 수행하여 제안 시스템의 도메인 특화 효과를 검증한다. 초기에 규칙 기반 동적 가중치(Crop Filter, Semantic Dedup)를 탐색하였으나, 휴리스틱 설계의 일반화 어려움과 성능 저하로 제거하고 현재의 4-컴포넌트 구조(RRF, DAT, Ontology, PathRAG)로 정착하였다.
