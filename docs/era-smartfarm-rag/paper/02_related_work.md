# 2. Related Work

본 연구는 RAG 시스템, 하이브리드 검색, 그래프 기반 지식 표현, 엣지 배포의 교차점에 위치한다. 본 장에서는 각 영역의 최근 발전과 한계를 검토하고, 연구 공백을 식별한다.

## 2.1 RAG and Graph-based Retrieval

Retrieval-Augmented Generation(RAG)은 외부 지식 검색을 LLM 생성과 결합하여 환각을 줄이는 패러다임이다 [1]. Gao et al.의 서베이 [2]는 RAG 발전을 Naive RAG(단순 검색-생성), Advanced RAG(쿼리 변환, 리랭킹), Modular RAG(컴포넌트 조합)로 분류하였다.

최근 그래프 기반 RAG가 주목받고 있다. **GraphRAG** [3]는 커뮤니티 탐지로 문서를 클러스터링하여 전역 질의에 대응하나, 그래프 구축 과정에서 LLM 기반 엔티티/관계 추출 및 요약이 병목이 될 수 있다. **LightRAG** [4]는 Dual-Level 검색(엔티티 수준 + 커뮤니티 수준)과 ego-network 기반 효율적 그래프 탐색으로 GraphRAG 대비 검색 효율성을 크게 개선하였으며, 로컬 LLM과의 통합이 용이하다. 그러나 범용 엔티티 타입(person, organization, location 등)을 사용하여 농업 등 특정 도메인에 대한 최적화가 필요하다. **PathRAG** [5]는 관계 경로 기반 검색으로 중복을 줄이며 농업 포함 6개 도메인에서 우수한 성능을 보였다.

한편, 엣지 배포를 고려할 때 그래프 구축(인덱싱)에서의 “하드웨어/토큰 세금”을 낮추려는 흐름도 나타난다. **LinearRAG** [6]는 관계 추출(LLM 호출)에 의존하지 않고, 텍스트의 포함 관계를 기반으로 한 Tri-Graph(엔티티–문장–구절) 구조와 의미 전파(semantic bridging), 그리고 PPR 기반 전역 집계를 통해 다중 홉 검색을 수행하는 방식을 제안한다. 이러한 접근은 온프레미스/엣지 환경에서 “원타임 인덱싱 + 런타임 경량 검색”이라는 운영 형태에 자연스럽게 부합한다.

2025년 농업 도메인 RAG 연구가 본격화되었다. **Crop GraphRAG** [7]는 병해충 지식 그래프와 RAG를 결합하였고, **AHR-RAG** [8]는 91만 트리플릿 KB로 복잡 질의에 대응하였다. **AgroMetLLM** [9]은 Raspberry Pi에서 양자화 LLM 기반 오프라인 농업 자문을 구현하였으나 증발산 예측에 특화되어 있다. 이들 연구는 엣지 환경에서의 범용 농업 Q&A를 다루지 않는다.

## 2.2 Hybrid Retrieval

Dense retrieval(예: DPR [10])은 의미적 유사성 매칭에 강하나 수치/단위 정보 매칭에 취약하고, Sparse retrieval(BM25)은 그 반대 특성을 보인다. Hybrid retrieval은 두 방식을 융합하여 상호 보완한다 [11].

융합 전략으로 **Reciprocal Rank Fusion(RRF)** [12]과 score-level 결합(가중 합/정규화 등)이 널리 사용된다. 특히 Sparse 신호(BM25)가 강한 도메인에서는 Dense와의 단순 결합만으로도 안정적인 개선을 얻을 수 있으며, 최근 연구는 Sparse 결과를 가이드로 활용해 Dense 측 표현/순위를 보정하는 융합을 제안한다 [11].

기존 하이브리드 검색은 Dense-Sparse 2채널 융합에 머물며, 그래프 기반 검색과의 통합은 제한적이다. LightRAG는 그래프 구조 내에서 벡터 검색을 수행하여 이러한 통합을 시도하나, 농업 도메인의 수치 정보(온도, EC, pH 등) 매칭에 대한 최적화는 부족하다.

## 2.3 Agricultural Knowledge Systems

농업 온톨로지 연구는 지식 표현에 초점을 맞춰왔다. Bhuyan et al. [13]은 시공간 농업 데이터 추론을 위한 래티스 구조를 제안하였고, 스마트 농업 온톨로지 [15]와 NLP 기반 개발 방법론 [14]이 발표되었다. **CropDP-KG** [16]는 NER/RE로 13,840 엔티티와 21,961 관계를 구축하였으나, 수만 건의 학습 데이터와 레이블링 비용이 필요하다.

인과관계 추출 연구 [17-18]는 문장 수준 관계 식별에 집중하며, 문서 간 인과관계 연결("문서 A의 원인 → 문서 B의 해결책")을 다루지 않는다. 딥러닝 기반 방식 [18]은 학습/추론 비용이 커 엣지 환경에서 상시 동작시키기 어렵다.

기존 농업 온톨로지는 검색 단계에서 직접 활용되지 않아 지식 정리가 검색 품질 향상에 기여하지 못한다.

## 2.4 Edge Deployment for RAG

엣지 LLM 배포를 위한 압축 기법으로 양자화, 지식 증류, 프루닝이 연구되고 있다 [19]. **llama.cpp** [20]는 GGUF 양자화로 CPU/저사양 GPU에서 LLM 추론을 가능하게 하며, Q4_K_M 양자화는 메모리를 약 70% 절감한다.

**EdgeRAG** [21]는 온라인 인덱싱과 선택적 임베딩 로딩을 통해 메모리 제약 하에서 검색 지연을 낮추는 방향을 제안한다.

## 2.5 RAG Evaluation

전통적 RAG 평가는 검색 단계에서 Precision@K, Recall@K, MRR, NDCG, MAP 등 IR 메트릭을, 생성 단계에서 Exact Match(EM), F1 Score, ROUGE, BLEU 등을 사용한다. BERTScore는 임베딩 기반 의미 유사도를 측정하여 n-gram 한계를 보완한다. 그러나 이들 메트릭은 Ground Truth 구축에 높은 비용이 소요되며, 도메인 특화 데이터셋 부재 시 적용이 어렵다.

**RAGAS** [22]는 LLM-as-Judge 기반 reference-free 평가 프레임워크로 Faithfulness, Answer Relevancy, Context Precision을 측정하며, EACL 2024에서 발표되어 RAG 평가의 de facto 표준으로 자리잡았다. **ARES** [23]는 Synthetic QA 생성과 fine-tuned LLM Judge를 결합해 자동 평가를 구성한다.

진단/벤치마크 측면에서, **RAGChecker** [24]는 claim-level 등 미세 단위의 오류/근거 문제를 진단하는 프레임워크를 제공한다. 또한 **CRAG** [25]은 다양한 RAG 설정을 포괄하는 벤치마크로 현실적 시나리오 평가를 지원한다.

농업 도메인 벤치마크는 제한적이며, 한국어 스마트팜 도메인 특화 벤치마크는 부재하다. 이에 본 연구는 IR 메트릭과 RAGAS 기반 reference-free 평가를 조합하여 재현 가능한 평가 체계를 구축한다.

## 2.6 Research Gap and Our Contributions

Table 1은 기존 연구와 본 연구의 차별점을 요약한다.

**Table 1. Comparison with Existing Approaches**

| Aspect | Prior Work | Gap | Our Approach |
|--------|-----------|-----|--------------|
| **Graph RAG** | LightRAG [4], PathRAG [5], LinearRAG [6] | 인덱싱(구축) 비용/자원 소모, 엣지 배포 난점 | Tri-Graph 기반 multi-hop 채널 + Dense/Sparse 결합 (Fusion) |
| **Hybrid Retrieval** | Dense+Sparse 융합 [11], RRF [12] | 2채널 중심, 그래프 신호 통합 제한 | Dense+Sparse+Tri-Graph 3채널을 weighted RRF로 통합 |
| **Edge Deployment** | EdgeRAG [21], AgroMetLLM [9] | 특정 태스크 한정 또는 단일 검색 채널 | llama.cpp [20] + mmap-friendly 인덱스 로딩 + Compose 기반 원타임 인덱싱 |
| **On-Prem Update & Privacy** | LightRAG [4] (증분 업데이트), GraphRAG [3] | 민감 데이터의 온프레미스 업데이트/검색을 “외부 유출 0”로 보장하는 운영 경계 부재 | 외부지식 Ingress 분리 + Private Overlay(Egress 0) + 로컬 LLM 기반 규격화 업데이트 |
| **Evaluation** | IR metrics with Ground Truth | 고비용 어노테이션/데이터셋 부재 | IR 메트릭 + RAGAS [22] 기반 reference-free 평가 |

본 연구는 기존 Graph RAG 및 하이브리드 검색 연구(LightRAG [4], PathRAG [5], LinearRAG [6], RRF [12])의 개념을 참고하되, **온프레미스 엣지 배포를 전제로 한 “LLM-free 인덱싱 + 다중 홉 검색” 중심의 Tri-Graph RAG 구조**를 독자적으로 구현한다:

1. **LLM-free 원타임 인덱싱 파이프라인**: 오프라인에서 Dense(FAISS)·Sparse(BM25)·Tri-Graph 아티팩트를 한 번에 생성하고, 런타임은 로딩/검색만 수행하도록 분리한다.
2. **Tri-Graph multi-hop 검색 채널**: 엔티티–문장–구절 구조에서 semantic bridging과 (선택적) PPR 기반 전역 집계를 통해 다중 홉 컨텍스트를 구성한다 [6].
3. **3채널 융합(Fusion)**: Dense+Sparse+Tri-Graph 결과를 weighted RRF로 통합하고, 질의 유형(수치/단위 vs 원인/절차)에 따라 가중치를 조정해 엣지 환경에서의 견고성을 확보한다 [12].
4. **온프레미스 프라이버시 경계(Private Egress 0) + 업데이트**: 외부지식 유입(Ingress)과 민감지식 오버레이(Overlay)를 분리하고, 로컬 LLM로 민감 입력을 규격화하여 오버레이를 업데이트함으로써 “내부지식 유출 0” 운영 요구를 아키텍처로 고정한다.
5. **재현 가능한 엣지 운영/검증**: llama.cpp 기반 로컬 추론 [20]과 Docker Compose 기반 스택(서비스/연구 분리)으로 “원타임 인덱싱 → 검색/생성/업데이트” 워크플로우를 재현 가능하게 구성한다.

Baseline 비교는 그래프 기반 접근(LightRAG [4]) 및 하이브리드 검색 조합을 중심으로 수행하며, 특히 “인덱싱 비용/운영 복잡도”와 “엣지 런타임 성능(지연/메모리)”의 trade-off 관점에서 제안 구조의 실용성을 검증한다.
