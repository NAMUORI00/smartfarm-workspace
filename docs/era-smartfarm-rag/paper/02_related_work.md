# 2. Related Work

본 연구는 RAG 시스템, 하이브리드 검색, 그래프 기반 지식 표현, 엣지 배포, 프라이버시 보존의 교차점에 위치한다. 본 장에서는 각 영역의 최근 발전과 한계를 검토하고, 연구 공백을 식별한다.

## 2.1 RAG and Graph-based Retrieval

Retrieval-Augmented Generation(RAG)은 외부 지식 검색을 LLM 생성과 결합하여 환각을 줄이는 패러다임이다 [1]. Gao et al.의 서베이 [2]는 RAG 발전을 Naive RAG(단순 검색-생성), Advanced RAG(쿼리 변환, 리랭킹), Modular RAG(컴포넌트 조합)로 분류하였다.

최근 그래프 기반 RAG가 주목받고 있다. **GraphRAG** [3]는 커뮤니티 탐지로 문서를 클러스터링하여 전역 질의에 대응하나, 그래프 구축 과정에서 LLM 기반 엔티티/관계 추출 및 요약이 병목이 될 수 있다. **LightRAG** [4]는 Dual-Level 검색(엔티티 수준 + 커뮤니티 수준)과 ego-network 기반 효율적 그래프 탐색으로 GraphRAG 대비 검색 효율성을 크게 개선하였으며, 로컬 LLM과의 통합이 용이하다. 그러나 범용 엔티티 타입(person, organization, location 등)을 사용하여 농업 등 특정 도메인에 대한 최적화가 필요하다.

**RAG-Anything** [38]은 LightRAG를 기반으로 멀티모달 확장을 시도한 최신 연구로, MinerU 파서를 활용하여 텍스트·이미지·테이블·수식을 통합 처리하고, Dual-Graph 구조(cross-modal 관계 + 텍스트 의미)를 구축한다. 그러나 그래프를 인메모리에만 유지하여 영속성이 보장되지 않으며, 엣지 배포를 고려하지 않는다.

한편, 엣지 배포를 고려할 때 그래프 구축(인덱싱)에서의 "하드웨어/토큰 세금"을 낮추려는 흐름도 나타난다. **LinearRAG** [6]는 관계 추출(LLM 호출)에 의존하지 않고, 텍스트의 포함 관계를 기반으로 한 Tri-Graph(엔티티–문장–구절) 구조와 의미 전파(semantic bridging), 그리고 PPR 기반 전역 집계를 통해 다중 홉 검색을 수행하는 방식을 제안한다. 이러한 접근은 온프레미스/엣지 환경에서 "원타임 인덱싱 + 런타임 경량 검색"이라는 운영 형태에 자연스럽게 부합한다.

2025년 농업 도메인 RAG 연구가 본격화되었다. **Crop GraphRAG** [7]는 병해충 지식 그래프와 RAG를 결합하였고, **AHR-RAG** [8]는 91만 트리플릿 KB로 복잡 질의에 대응하였다. **AgroMetLLM** [9]은 Raspberry Pi에서 양자화 LLM 기반 오프라인 농업 자문을 구현하였으나 증발산 예측에 특화되어 있다. 이들 연구는 엣지 환경에서의 범용 농업 Q&A를 다루지 않는다.

## 2.2 Hybrid Retrieval

Dense retrieval(예: DPR [10])은 의미적 유사성 매칭에 강하나 수치/단위 정보 매칭에 취약하고, Sparse retrieval(BM25)은 그 반대 특성을 보인다. Hybrid retrieval은 두 방식을 융합하여 상호 보완한다 [11].

융합 전략으로 **Reciprocal Rank Fusion(RRF)** [12]과 score-level 결합(가중 합/정규화 등)이 널리 사용된다. 특히 Sparse 신호(BM25)가 강한 도메인에서는 Dense와의 단순 결합만으로도 안정적인 개선을 얻을 수 있으며, 최근 연구는 Sparse 결과를 가이드로 활용해 Dense 측 표현/순위를 보정하는 융합을 제안한다 [11].

**Qdrant** [40]는 Dense 벡터(HNSW)와 Sparse 벡터(BM25)를 단일 컬렉션 내에서 네이티브 RRF로 융합하는 기능을 제공하며, payload 기반 필터링이 가능하여 메타데이터 조건부 검색에 적합하다. 

최근 하이브리드 검색 시스템의 Reranking 단계에서는 BERT 등 언어모델에 기반한 무거운 **딥러닝(Deep Learning) 리랭커**(Neural Reranker, 예: Cross-encoder [12])를 도입하는 것이 주류를 이룬다. 하지만 딥러닝(DL) 기반 추론 모델은 다중 모달리티(특히 Graph 구조)를 융합할 때 그래프 특유의 입체적인 인과관계(Topology)를 단순 1차원 자연어 시퀀스로 강제 평탄화(Flattening)해야 하는 본질적인 한계에 직면한다 [47]. 이러한 구조적 왜곡은 텍스트 변환 과정에서 원래의 위상 신호가 훼손됨을 의미하며, 결과적으로 검색 성능이 하향 평준화되는 **'끌어내림(Pull-down)'** 현상을 초래한다. 더욱이 딥러닝 모델의 복잡한 신경망 레이어 연산은 8GB RAM 수준의 엣지 디바이스 환경에서 감당할 수 없는 메모리 점유 및 치명적인 추론 지연(Latency)을 야기한다 [48].

이러한 한계를 근본적으로 극복하기 위해, 본 연구는 다중 채널 융합 단계에서 신경망 기반의 딥러닝(DL) 모델 개입을 완전히 배제하고, 그 대안으로 **그리드 탐색(Grid Search) 기반의 통계적 머신러닝(Statistical Machine Learning) 최적화 기법**을 채택하였다. 이를 구체화한 본 연구의 **동적 가중치 튜닝(DAT: Data-driven Adaptive Tuning)** 파이프라인은 신경망 역전파(Backpropagation) 대신, 과거 로그 데이터($\mathcal{Q}_{train}$)의 nDCG 목적함수를 극대화하는 최적의 3채널 이산 가중치($w_d, w_s, w_g$) 조합을 머신러닝으로 오프라인 사전 학습(Learning)한다 [12][49]. 런타임에는 $O(1)$ 복잡도의 단순 스칼라 매핑 연산만 수행하므로, Qdrant [40]의 Vector 검색과 **FalkorDB** [39]의 Graph 경로 탐색의 수학적 Native 점수를 사실상 0-Latency 비용으로 엣지 환경에서 무손실 융합할 수 있다.
## 2.3 Agricultural Knowledge Systems

농업 도메인 지식표현 연구는 구조화된 개념 체계에 초점을 맞춰왔다. Bhuyan et al. [13]은 시공간 농업 데이터 추론을 위한 래티스 구조를 제안하였고, 스마트 농업 지식모델 [15]과 NLP 기반 개발 방법론 [14]이 발표되었다. **CropDP-KG** [16]는 NER/RE로 13,840 엔티티와 21,961 관계를 구축하였으나, 수만 건의 학습 데이터와 레이블링 비용이 필요하다.

인과관계 추출 연구 [17-18]는 문장 수준 관계 식별에 집중하며, 문서 간 인과관계 연결("문서 A의 원인 → 문서 B의 해결책")을 다루지 않는다. 딥러닝 기반 방식 [18]은 학습/추론 비용이 커 엣지 환경에서 상시 동작시키기 어렵다.

멀티모달 문서 파싱 측면에서, **MinerU** [41]는 LayoutLMv3 기반 레이아웃 분석, YOLOv8 수식 감지, TableMaster 테이블 인식, PaddleOCR 기반 109개 언어 OCR을 통합한 오픈소스 문서 파서로, OmniDocBench에서 90.67%의 종합 정확도를 달성하였다. 특히 테이블(88.22%)과 수식(88.46%) 추출에서 GPT-4o를 상회하는 성능을 보이며, 농업 문서의 재배조건표·양액 배합표 등 정밀 추출에 유리하다.

기존 농업 지식모델은 검색 단계에서 직접 활용되지 않아 지식 정리가 검색 품질 향상에 기여하지 못한다.

## 2.4 Edge Deployment for RAG

엣지 LLM 배포를 위한 압축 기법으로 양자화, 지식 증류, 프루닝이 연구되고 있다 [19]. **llama.cpp** [20]는 GGUF 양자화로 CPU/저사양 GPU에서 LLM 추론을 가능하게 하며, Q4_K_M 양자화는 메모리를 약 70% 절감한다.

**Qwen3** [42]는 Alibaba가 공개한 LLM 시리즈로, 4B 파라미터 모델(Qwen3-4B-Instruct)은 4-bit AWQ 양자화 시 약 2.3GB VRAM으로 32K 토큰 컨텍스트를 지원하며, RAG 벤치마크에서 83%의 점수를 달성하여 엣지 배포에 적합하다.

**EdgeRAG** [21]는 온라인 인덱싱과 선택적 임베딩 로딩을 통해 메모리 제약 하에서 검색 지연을 낮추는 방향을 제안한다.

**FalkorDB** [39]는 GraphBLAS 기반 희소 행렬 연산을 활용한 그래프 데이터베이스로, Redis 호환 프로토콜을 지원하면서 sub-millisecond 지연의 Cypher 쿼리를 제공한다. 인메모리 그래프에 비해 영속성을 보장하면서도 경량 자원(0.75GB)으로 엣지 환경에 적합하며, MERGE 기반 증분 업데이트를 네이티브로 지원한다.

## 2.5 Privacy-Preserving RAG

RAG 시스템에서 프라이버시 보존은 아직 초기 연구 단계에 있다. 기존 접근은 주로 모델 학습 단계의 프라이버시에 집중한다.

**Federated RAG** (FedE4RAG) [43]는 분산 학습을 통해 여러 노드의 데이터를 중앙으로 모으지 않고 RAG를 학습하지만, 지식 저장소 자체의 프라이버시 분리는 다루지 않는다. **DP-FedLoRA** [44]는 차분 프라이버시(Differential Privacy)를 LoRA fine-tuning에 적용하여 온디바이스 LLM 적응을 가능하게 하였으나, 지식 검색 계층에서의 데이터 주권 보장은 범위 밖이다.

스마트농업 현장에서는 센서 원시값, 작업일지, 개인 메모, 재배 설정 등 **운영·개인 정보가 결합된 민감 지식**이 지속적으로 발생한다. 이 데이터는 규정·보안상 외부 전송이 어려우며, 지식베이스 업데이트 시점에서도 온프레미스 내부에서 처리되어야 한다. 그러나 기존 RAG 연구는 지식 저장소(Knowledge Graph + Vector DB)를 Public/Private로 아키텍처적으로 분리하고, 엣지 LLM이 민감 데이터를 자율적으로 구조화·증분 반영하는 접근을 제시하지 않는다.

**Graphiti** [45]는 FalkorDB 기반 실시간 지식그래프 증분 업데이트 프레임워크로, 본 연구의 Private Tier 증분 업데이트 설계에 참고한다.

## 2.6 RAG Evaluation

전통적 RAG 평가는 검색 단계에서 Precision@K, Recall@K, MRR, NDCG, MAP 등 IR 메트릭을, 생성 단계에서 Exact Match(EM), F1 Score, ROUGE, BLEU 등을 사용한다. BERTScore는 임베딩 기반 의미 유사도를 측정하여 n-gram 한계를 보완한다. 그러나 이들 메트릭은 Ground Truth 구축에 높은 비용이 소요되며, 도메인 특화 데이터셋 부재 시 적용이 어렵다.

**RAGAS** [22]는 LLM-as-Judge 기반 reference-free 평가 프레임워크로 Faithfulness, Answer Relevancy, Context Precision을 측정하며, EACL 2024에서 발표되어 RAG 평가의 de facto 표준으로 자리잡았다. **ARES** [23]는 Synthetic QA 생성과 fine-tuned LLM Judge를 결합해 자동 평가를 구성한다.

진단/벤치마크 측면에서, **RAGChecker** [24]는 claim-level 등 미세 단위의 오류/근거 문제를 진단하는 프레임워크를 제공한다. 또한 **CRAG** [25]은 다양한 RAG 설정을 포괄하는 벤치마크로 현실적 시나리오 평가를 지원한다.

농업 도메인 벤치마크는 제한적이며, 한국어 스마트팜 도메인 특화 벤치마크는 부재하다. 이에 본 연구는 IR 메트릭과 RAGAS 기반 reference-free 평가를 조합하고, 오픈소스 120B급 LLM(Qwen3-235B-A22B)을 Judge로 활용하여 상용 API 의존 없이 재현 가능한 평가 체계를 구축한다.

또한 실제 현장 운영 관점에서는 "빠른 첫 토큰"보다 "근거 기반 신뢰성"과 "반복 실행 시 일관성"이 우선될 수 있다. 이에 최근 RAG 평가 실무는 평균 점수뿐 아니라 unsupported claim 비율, citation support율, run-to-run 분산 등 안정성 지표를 함께 보고하는 방향으로 확장되고 있다. 본 연구 역시 동일한 관점에서 속도 지표를 보조 SLA로 취급하고, 신뢰성/안정성 지표를 1차 판단 축으로 둔다.

## 2.7 Research Gap and Our Contributions

Table 1은 기존 연구와 본 연구의 차별점을 요약한다.

**Table 1. Comparison with Existing Approaches**

| Aspect | Prior Work | Gap | Our Approach |
|--------|-----------|-----|--------------| 
| **Multimodal KG** | RAG-Anything [38] (인메모리 그래프) | 그래프 영속성 없음, 엣지 배포 미고려 | MinerU [41] 파싱 + Kimi-K2.5 추출 + FalkorDB [39] 영속 KG (C3) |
| **Hybrid Retrieval** | Dense+Sparse 2채널 [11], C-Encoder [12] | 그래프 위상(Topology) 평탄화 소실, 구조 훼손 및 Pull-down 현상 | 딥러닝 배제형 무손실 3채널 융합 (Zero-latency DAT 알고리즘 도입) (C1) |
| **Edge Deployment** | EdgeRAG [21], AgroMetLLM [9] | 특정 태스크 한정, 범용 농업 QA 미지원 | Qwen3-4B Q4 [42] + llama.cpp [20], 8GB RAM 로컬 디바이스 (C4) |
| **Evaluation** | IR metrics with Ground Truth | 고비용 어노테이션, 상용 API 의존 | RAGAS [22] + OSS 120B Judge LLM, reference-free (C5) |
| **On-Prem Privacy** | FedE4RAG [43], GraphRAG [3] | 혼재된 지식 저장소에서의 데이터 오염 및 완벽한 격리(Isolation) 분리 부재 | Sovereign Architecture: 단일 DB 내 복합키(`canonical_id + tier + farm_id`)를 활용한 100% 논리적 격리 환경 구현 (C2) |

본 연구는 기존 Graph RAG 및 하이브리드 검색 연구(LightRAG [4], RAG-Anything [38], RRF [12])의 개념을 참고하되, **질의 적응적 검색, 프라이버시 보존, 엣지 배포를 함께 다루는 통합 프레임워크**를 제안한다:

1. **Query-Adaptive Tri-Channel Fusion (C1)**: Qdrant [40]의 Dense+Sparse 네이티브 RRF와 FalkorDB [39]의 Dual-Level 그래프 검색을 무거운 신경망 없이 수학적 원본 등수(Rank)와 오프라인 그리드 검증치만으로 조합하는 **DAT(Data-driven Adaptive Tuning)** 구조로 융합한다. 타 모델들의 Topology 변환에 따른 '강제 끌어내림(Pull-down)' 현상을 우회하며, 런타임 지연(Latency) 제로 수준의 극적인 Trade-off 우위를 달성하였다.
2. **Edge-Local Private Store (C2)**: 외부지식(공개 문헌) 유입(Ingress)과 민감지식(센서, 메모, 대화) Private Store를 아키텍처적으로 철저히 격리한다. 엣지 LLM이 민감 데이터를 자율 구조화하여 FalkorDB/Qdrant에 반영할 때, 반드시 **소버린 복합키(`canonical_id + tier + farm_id`)** 생태계를 사용하여 "내부지식 유출 0(Egress 0)"이자 오염 확률 0%의 운영을 보장한다.
3. **End-to-End Edge-Deployable SmartFarm RAG System (C3)**: MinerU [41] 기반 멀티모달 파싱, 비전 LLM [46] 기반 지식 추출, FalkorDB+Qdrant 영속 저장, Qwen3-4B Q4 [42] + llama.cpp [20] 엣지 추론, RAGAS [22] + IR 메트릭 2-Track 평가를 하나의 재현 가능한 파이프라인으로 통합하여 8GB RAM 엣지 디바이스에서 동작하는 농업 QA 시스템을 구현한다.

Baseline 비교는 5개 직접 재현 시스템(Dense, Sparse, Hybrid, Graph-only, LightRAG [4])과 문헌 수치 비교 3개(PathRAG, GraphRAG [3], HippoRAG/RAPTOR)를 포함하며, "검색 품질 + 근거 신뢰성 + 엣지 런타임 안정성"의 trade-off 관점에서 제안 구조의 실용성을 검증한다. 추가로 Global-only vs Global+Private 검색의 품질 차이를 Ablation으로 분석하여 C2의 기여도를 정량적으로 검증한다.

