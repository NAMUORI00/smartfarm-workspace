# 4. 구현 및 시스템 아키텍처 (Implementation and System Architecture)

본 장에서는 제안한 방법론(DAT, Sovereign DB)을 8GB RAM 수준의 제한된 엣지(Edge) 환경에 실제로 배포하고 구동하기 위한 엔드투엔드(End-to-End) 시스템 엔지니어링 및 최적화 구현 전략을 상술한다. 단순한 개념 증명(Proof of Concept)을 넘어, 하드웨어 제약 극복, 데이터 무결성 보장, 그리고 재현 가능한 평가 파이프라인(Evaluation Pipeline) 구축에 초점을 맞춘다.

<br/>

<div align="center">
  <img src="figures/fig1_overall_architecture.png" alt="Overall Architecture" width="80%">
  <br/>
  <p><strong>Figure 1.</strong> Overall system architecture of ERA-SmartFarm-RAG. Stage 1 (Offline Ingest) constructs the multimodal knowledge base on a cloud server; Stage 2 (Edge Runtime) performs 3-channel DAT fusion with pull-down prevention on an 8GB device; Stage 3 (Evaluation) runs strict 2-track RAGAS+IR benchmarking. The Sovereign DB layer enforces Egress-0 data isolation across all stages.</p>
</div>

<br/>

## 4.1 하드웨어 제약 및 엣지 추론 최적화
농업 현장의 열악한 인프라를 고려하여 본 시스템은 보급형 미니 PC 또는 라즈베리 파이(Raspberry Pi 5) 수준의 8GB RAM 엣지 디바이스 환경을 타겟으로 설계되었다.
- **LLM 추론 엔진 최적화:** 무거운 딥러닝 프레임워크(PyTorch 등) 대신 C/C++ 기반의 `llama.cpp` 엔진을 채택하였다. 메인 언어모델인 Qwen3-4B-Instruct를 GGUF 규격의 4-bit(Q4_K_M)로 양자화(Quantization)하여, 4B 파라미터 모델을 2.3GB 수준의 VRAM/RAM 점유율로 구동하는 데 성공하였다.
- **메모리 할당 전략:** RAG 추론 시 필요한 컨텍스트 윈도우(KV Cache)용 1GB, Qdrant(Vector DB) 및 FalkorDB(Graph DB)의 인메모리 상주용 2.5GB를 제외하고, 전체 시스템 운영체제 오버헤드까지 합산하여 최대 7GB 선에서 시스템 병목 없이 즉각적인 생성이 가능하도록 아키텍처를 설계하였다.

## 4.2 멀티모달 지식 구축 파이프라인 (Knowledge Ingestion)
자연어뿐만 아니라 표, 수식, 레이아웃 등 다양한 형태로 존재하는 농업 매뉴얼과 정형 데이터를 효과적으로 처리하기 위해 오프라인 지식 구축 파이프라인(`smartfarm-ingest`)을 구축하였다.
- **컴퓨터 비전 기반 파싱:** 정형화되지 않은 PDF 문서가 입력되면 오픈소스 멀티모달 파서인 MinerU(도클링/VLM 파이프라인)를 통해 문서를 파싱하고 청크(Chunk) 단위로 분할한다.
- **이중 영속화(Dual-Persistence):** 각 청크는 `canonical_doc_id`와 `modality` 식별 메타데이터를 부여받은 뒤, 의미 공간 검색을 위해 **Qdrant** 벡터 인덱스로 적재된다. 동시에 LLM(가령 Kimi-K2.5 등)을 통해 추출된 엔티티(Entity) 및 관계 쌍(Relations)은 **FalkorDB**에 지식 그래프 노드와 엣지(Edge)로 기록되어 영속성(Persistence)을 확보한다.

## 4.3 소버린 격리 환경의 물리적 구현 (Sovereign DB Isolation)
3장(Methodology)에서 설계한 'Egress-0' 및 데이터 주권(Sovereign) 보장 논리를 물리적 데이터베이스 계층에 반영하기 위해 멀티테넌시(Multi-tenancy) 라우팅을 구현하였다.
- **복합키 기반 그래프 병합(MERGE):** 농가의 민감 정보(센서, 메모, 작업일지)가 시스템 내부에서 자율적으로 증분 업데이트(Incremental Update) 될 때, FalkorDB의 Cypher 쿼리문에 반드시 `canonical_id + tier + farm_id`의 복합 식별자(Composite Unique Key)를 강제하여 `MERGE` 혹은 `MATCH` 오퍼레이션을 수행한다.
- **동적 필터 주입:** 사용자 질의 시 Qdrant 검색 객체의 Payload 필터에 `{"must": [{"key": "tier", "match": {"value": "public"}}]}` 조건과 `{"key": "farm_id", "match": {"value": local_farm_id}}` 조건을 런타임에 동적으로 주입한다. 이를 통해 논리적으로 단일화된 DB 풀(Pool) 내에서도 타 농장의 데이터가 유출되거나 혼재되는 물리적 오염 현상을 원천 차단한다.

## 4.4 제로-레이턴시 엣지 라우팅 엔진 (Zero-Latency Edge Routing)
Qdrant(Dense, Sparse)와 FalkorDB(Graph)에서 기원한 이기종 점수들을 무거운 신경망 딥러닝 리랭커 없이 결합하기 위해 $O(1)$ 스칼라 연산 라우터를 설계하였다(`smartfarm-search`).
- **가중치 매핑:** 정규식 기반의 가벼운 4D 질의 특성 추출기가 입력 질의를 분석하면, 메모리에 캐싱(Caching)된 DAT 프로파일 테이블에 O(1) 해시 테이블 조회를 수행하여 $w_d, w_s, w_g$의 최적 스칼라 가중치를 즉시 반환받는다.
- **비동기 IO 병렬화:** 병렬 구조(비동기 커루틴)를 도입하여 3개의 데이터베이스 쿼리를 동시에 실행한 후, 반환된 Native 점수를 앞서 획득한 스칼라 가중치로 가중합(Weighted RRF)함으로써 치명적인 런타임 지연(Latency) 없이 3채널의 순위 융합을 완료한다.

## 4.5 환각 방지 및 응답 거부(Rejection) 메커니즘
엣지 LLM의 내부 파라미터 지식에 의존한 환각(Hallucination) 발현을 원천 차단하기 위해 파이프라인 상에 2중 안전장치(Guardrails)를 구현하였다.
- **검색 임계값 차단 (Retrieval Thresholding):** 3채널 융합 모듈이 반환한 Top-1 문서의 최종 점수가 사전 정의된 허들 임계값($\tau_{reject}$)에 미달할 경우, 검색된 근거가 빈약하다고 판단하여 연산 비용이 큰 LLM 추론 단계를 즉시 생략(Short-circuit)하고 응답 거부(Rejection)를 반환한다.
- **프롬프트 통제 (Strict Prompting):** 임계값을 통과한 컨텍스트가 LLM에 주입될 때, 시스템 프롬프트(System Prompt) 최상단에 `주어진 Context 내에서만 답변할 것. Context에 질문에 대한 답이 없다면 상상하지 말고 반드시 '관련 정보가 없어 답변할 수 없습니다'라고 출력할 것.`이라는 강제 지시어를 하드코딩한다. 이는 후속 평가 레이어에서 진실성(Faithfulness) 지표를 극대화하는 핵심 안전판이 된다.

## 4.6 엄격한 2-Track 평가 파이프라인 (Evaluation Pipeline)
본 구현체는 단순한 추론(Inference)을 넘어 논문의 재현성(Reproducibility)을 담보하기 위한 결정론적 평가 스크립트(`smartfarm-benchmarking`)를 내장하고 있다.
- **데이터 분할 프로토콜 강제화:** 데이터 누수(Data Leakage)를 기술적으로 차단하기 위해, 실행 시 입력 데이터셋(AgXQA 등)을 `20%(DAT 학습용) / 80%(평가용)` 로 외부 분할하고, 런타임에 QID(Query ID) 겹침을 내부적으로 유효성(Validation) 검사한다.
- **RAGAS 파이프라인 결합:** 메인 평가 레이어는 IR 벤치마킹 메트릭(nDCG, MRR) 산출과 동시에 RAGAS 기반의 생성 평가(Faithfulness, Answer Relevancy)를 수행하며, 신뢰성 있는 채점을 위해 상용 API 대신 로컬에 오픈소스 다국어 모델(Qwen Judge Server)을 연결하여 Reference-free 평가 결과를 최종 JSON과 LaTeX 테이블 코드 형태로 자동 산출(`generate_ieee_tables`)한다.
