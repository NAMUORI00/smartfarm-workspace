# 3. 제안 방법론 (Proposed Methodology)

## 3.1 시스템 아키텍처 개요

본 시스템은 스마트팜 도메인에 특화된 온디바이스 하이브리드 RAG로, 6계층 스택 아키텍처로 구성된다.

### 3.1.1 리소스 제약 및 설계 목표

엣지 환경의 리소스 제약을 명확히 정의하고, 이를 기반으로 각 컴포넌트를 설계하였다.

| 리소스 항목 | 최소 사양 | 권장 사양 | 설계 근거 |
|------------|----------|----------|----------|
| **RAM** | 8GB | 16GB | Jetson Orin Nano 타겟 |
| **저장공간** | 10GB | 20GB | GGUF 모델 + FAISS 인덱스 |
| **목표 지연** | p95 < 500ms | p95 < 300ms | 실시간 현장 응답 |
| **LLM 메모리** | ~2.5GB | ~4GB | Q4_K_M 양자화 기준 |
| **처리량** | 3 QPS | 8 QPS | CPU 단독 환경 |

### 3.1.2 6계층 아키텍처 (Figure 1)

```mermaid
flowchart TB
    subgraph L5["<b>Layer 5: Application & Policy</b>"]
        direction LR
        UI["Streamlit UI<br/>시각화 대시보드"]
        API["FastAPI REST<br/>/query /ingest /health"]
        Policy["Offline Fallback Policy<br/>Cache → Template → Search"]
    end

    subgraph L4["<b>Layer 4: Generation & Grounding</b>"]
        direction LR
        Prompt["Prompt Template<br/>Jinja2 기반"]
        Template["TemplateResponder<br/>온톨로지 기반 폴백"]
    end

    subgraph L3["<b>Layer 3: Context Shaping</b><br/>논문 핵심 기여"]
        direction LR
        Crop["Crop Filter<br/>+0.5 / ×0.15"]
        Dedup["Semantic Dedup<br/>θ=0.85"]
        Rerank["Memory-aware Reranking<br/>BGE / LLM-lite / none"]
    end

    subgraph L2["<b>Layer 2: Retrieval Core</b><br/>3채널 융합 검색"]
        direction LR
        Dense["Dense<br/>FAISS"]
        Sparse["Sparse<br/>TF-IDF"]
        PathRAG["PathRAG-lite<br/>BFS 2-hop"]
    end

    subgraph L1["<b>Layer 1: On-device Knowledge Store</b>"]
        direction LR
        DenseIdx["dense.faiss<br/>(mmap)"]
        SparseIdx["sparse.pkl"]
        Graph["Causal Graph<br/>causes / solved_by"]
        Onto["Ontology<br/>6 types"]
    end

    subgraph L0["<b>Layer 0: Device & Runtime</b><br/>8GB RAM / Q4_K_M"]
        direction LR
        LLM["llama.cpp<br/>Qwen3-0.6B"]
        Embed["Embedding<br/>MiniLM 90MB"]
        FAISS["FAISS<br/>mmap enabled"]
    end

    L5 --> L4
    L4 --> L3
    L3 --> L2
    L2 --> L1
    L1 --> L0
```

**계층별 핵심 역할**:

| 계층 | 역할 | 핵심 컴포넌트 |
|------|------|--------------|
| **L5** | 사용자 인터페이스 및 정책 | FastAPI, Streamlit, 폴백 정책 |
| **L4** | 응답 생성 및 그라운딩 | 프롬프트 템플릿, 템플릿 응답기 |
| **L3** | 컨텍스트 압축 (논문 핵심) | 작물 필터, 중복 제거, 리랭킹 |
| **L2** | 3채널 하이브리드 검색 | Dense, Sparse, PathRAG 융합 |
| **L1** | 온디바이스 지식 저장소 | FAISS 인덱스, 인과관계 그래프, 온톨로지 |
| **L0** | 디바이스 런타임 | llama.cpp, 임베딩 모델, FAISS |

---

## 3.2 스마트팜 온톨로지

### 3.2.1 설계 배경

온톨로지 설계는 Stanford 온톨로지 구축 방법론[13]과 기존 농업 온톨로지 연구[9,10,11]를 참조하여 스마트팜 도메인에 적합한 6개 개념 유형을 정의하였다. CropDP-KG[12]의 엔티티 구조와 AgriKG[21]의 농업 엔티티 분류를 참고하여 한국 스마트팜 환경에 맞게 구성하였다.

### 3.2.2 개념 유형 정의 (Figure 2)

```mermaid
flowchart TD
    subgraph Ontology["<b>SmartFarm Domain Ontology</b>"]
        direction TB
        
        subgraph Crop["<b>crop</b> 작물"]
            C1["와사비"]
            C2["토마토"]
            C3["파프리카"]
            C4["딸기"]
            C5["상추"]
        end
        
        subgraph Env["<b>env</b> 환경"]
            E1["온도<br/><i>기온, 수온, ℃</i>"]
            E2["습도<br/><i>RH, %</i>"]
            E3["EC<br/><i>전기전도도, dS/m</i>"]
            E4["pH<br/><i>산도</i>"]
            E5["CO2<br/><i>이산화탄소</i>"]
        end
        
        subgraph Nutrient["<b>nutrient</b> 영양"]
            N1["양액"]
            N2["비료"]
            N3["관수"]
        end
        
        subgraph Disease["<b>disease</b> 병해"]
            D1["흰가루병"]
            D2["뿌리썩음병"]
            D3["연부병"]
        end
        
        subgraph Stage["<b>stage</b> 생육단계"]
            S1["육묘"]
            S2["정식"]
            S3["생육"]
            S4["수확"]
        end
        
        subgraph Practice["<b>practice</b> 재배실천"]
            P1["차광"]
            P2["환기"]
            P3["난방"]
            P4["냉각"]
            P5["살균"]
        end
    end
    
    Query["Query: 와사비 고온 관리"] --> Match["OntologyMatcher.match()"]
    Match --> Result["hits = {<br/>crop: [와사비],<br/>env: [온도]<br/>}"]
```

| 유형 | 설명 | 예시 | 근거 |
|------|------|------|------|
| **crop** | 재배 작물 | 와사비, 토마토, 딸기 | CropDP-KG의 Crops Name 엔티티[12] |
| **env** | 환경 요소 | 온도, 습도, EC, pH, CO2 | 스마트팜 IoT 센서 데이터 표준[11] |
| **nutrient** | 영양소 | 양액, 비료, 관수 | 농업 지식 베이스[9] |
| **disease** | 병해충 | 흰가루병, 뿌리썩음병, 연부병 | CropDP-KG의 Disease/Pest 분류[12,13] |
| **stage** | 생육 단계 | 육묘, 정식, 생육, 수확 | 작물 생육 모델[17] |
| **practice** | 재배 실천 | 차광, 환기, 난방, 살균 | 농업 실천 온톨로지[9,10] |

각 개념은 동의어/유의어 목록(alias)을 포함한다. 예를 들어 "와사비"의 alias에는 "산와사비", "본와사비"가 포함되어 사용자가 어떤 표현을 쓰더라도 동일 개념으로 인식한다.

---

## 3.3 3채널 하이브리드 검색 (HybridDAT)

### 3.3.1 설계 근거

Dense retrieval은 의미적 유사성 검색에 강하지만 "EC 2.5 dS/m" 같은 수치 정보 매칭에 취약하다. Sparse retrieval은 정확한 키워드 매칭에 강하지만 의미적 유사성을 놓칠 수 있다[5]. 본 시스템은 Dense-Sparse-PathRAG 3채널 융합과 질의 특성에 따른 동적 가중치 조정(Dynamic Alpha Tuning)을 적용한다.

### 3.3.2 HybridDATRetriever 플로우 (Figure 3)

```mermaid
flowchart TD
    Query["Query: 와사비 적정 온도는?"]
    
    subgraph OntologyMatch["1. 온톨로지 매칭"]
        OM["OntologyMatcher.match()"]
        Hits["hits = {crop: 와사비, env: 온도}"]
        OM --> Hits
    end
    
    subgraph AlphaCalc["2. Dynamic Alpha 계산"]
        Check1{"수치/단위<br/>포함?"}
        Check2{"병해/재배<br/>관련?"}
        
        A1["α_d=0.5, α_s=0.5, α_p=0.0<br/><i>일반 질의</i>"]
        A2["α_d=0.3, α_s=0.7, α_p=0.0<br/><i>수치 질의</i>"]
        A3["α_d=0.35, α_s=0.35, α_p=0.3<br/><i>병해/재배</i>"]
        
        Check1 -->|Yes| A2
        Check1 -->|No| Check2
        Check2 -->|Yes| A3
        Check2 -->|No| A1
    end
    
    subgraph Channels["3. 3채널 병렬 검색"]
        Dense["<b>Dense Channel</b><br/>───────────<br/>EmbeddingRetriever<br/>FAISS IndexFlatIP<br/>cosine similarity"]
        
        Sparse["<b>Sparse Channel</b><br/>───────────<br/>MiniStore<br/>TfidfVectorizer<br/>keyword matching"]
        
        Path["<b>PathRAG Channel</b><br/>───────────<br/>SmartFarmGraph<br/>BFS 2-hop traversal<br/>causal edges"]
    end
    
    subgraph Fusion["4. Score Fusion"]
        Norm["Min-Max 정규화"]
        Combine["final = α_d×dense + α_s×sparse + α_p×path"]
        Norm --> Combine
    end
    
    Query --> OntologyMatch
    OntologyMatch --> AlphaCalc
    AlphaCalc --> Channels
    Channels --> Fusion
    Fusion --> Output["Top-k × 2 후보"]
```

### 3.3.3 동적 가중치 규칙 (Dynamic Alpha)

질의 내용을 분석하여 가중치를 자동 결정한다:

| 질의 특성 | Dense (α_d) | Sparse (α_s) | PathRAG (α_p) | 설계 근거 |
|----------|-------------|--------------|---------------|----------|
| 일반 질의 | 0.5 | 0.5 | 0.0 | 의미 검색과 키워드 매칭 균형 |
| 수치/단위 포함 ("EC 2.5", "25℃") | 0.3 | 0.7 | 0.0 | 수치는 정확히 일치해야 함[4] |
| 병해/재배 관련 ("흰가루병 원인") | 0.35 | 0.35 | 0.3 | 인과관계 탐색 활성화 |

---

## 3.4 인과관계 그래프 (PathRAG-lite)

### 3.4.1 설계 배경

농업 도메인에서 "고수온 → 연부병 발생 → 수온 관리" 같은 인과 체인이 핵심 정보 구조를 형성한다[17]. GraphRAG[7]는 LLM으로 개체와 관계를 추출하므로 구축 비용이 높다(문서 1000개당 GPT-4 $100+). 본 시스템은 규칙 기반 패턴 매칭으로 인과관계 그래프를 구축하여 비용을 $0으로 절감한다.

### 3.4.2 인과관계 역할 분류

텍스트 패턴 매칭으로 문서의 역할을 분류한다:

| 역할 | 판별 패턴 | 예시 문장 |
|------|----------|----------|
| **Cause** | "원인", "때문", "~하면", "높으면", "낮으면" | "고온 환경에서는 화분 활력이 저하된다" |
| **Effect** | "결과", "증상", "문제", "장애", "저하" | "착과율이 떨어지는 문제가 발생한다" |
| **Solution** | "관리", "해야", "방법", "조치", "예방" | "야간 온도를 18℃ 이하로 관리해야 한다" |

### 3.4.3 PathRAG-lite BFS 탐색 (Figure 4)

PathRAG[8]의 경로 탐색 개념을 차용한 경량 구현이다. BFS(너비 우선 탐색) 기반 2-hop 탐색으로 원인→결과→해결책 문서를 수집한다.

```mermaid
flowchart TD
    subgraph Query["Query 분석"]
        Q["와사비 고온 피해 해결 방법"]
        Match["온톨로지 매칭:<br/>crop:와사비, env:고온"]
    end
    
    subgraph Graph["SmartFarm Knowledge Graph"]
        direction TB
        
        subgraph Concepts["Concept Nodes"]
            C1["crop:와사비"]
            C2["env:온도"]
            C3["disease:연부병"]
        end
        
        subgraph Practices["Practice Nodes (Documents)"]
            P1["chunk_001<br/><i>고온 시 잎 손상...</i><br/>role: cause"]
            P2["chunk_002<br/><i>생육 저하 발생...</i><br/>role: effect"]
            P3["chunk_003<br/><i>차광망 설치 필요...</i><br/>role: solution"]
            P4["chunk_004<br/><i>수온 18℃ 관리...</i><br/>role: solution"]
        end
        
        C1 -->|recommended_for| P1
        C2 -->|mentions| P1
        C2 -->|mentions| P2
        P1 -->|causes| P2
        P2 -->|solved_by| P3
        P2 -->|solved_by| P4
        C3 -->|associated_with| P2
    end
    
    subgraph BFS["BFS 2-hop 탐색"]
        H0["Hop 0: 시작점<br/>crop:와사비, env:온도"]
        H1["Hop 1: 연결 문서<br/>chunk_001, chunk_002"]
        H2["Hop 2: 인과관계 따라<br/>chunk_003, chunk_004"]
        
        H0 --> H1 --> H2
    end
    
    Query --> Graph
    Graph --> BFS
    BFS --> Result["검색 결과:<br/>P3, P4 (solutions)"]
```

### 3.4.4 그래프 스키마

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

## 3.5 Context Shaping (컨텍스트 압축)

엣지 LLM은 토큰이 곧 지연/전력 비용이므로, 검색 결과를 그대로 전달하지 않고 압축/필터링하는 것이 핵심이다.

### 3.5.1 Context Shaping 파이프라인 (Figure 5)

```mermaid
flowchart LR
    subgraph Input["입력"]
        In["검색 결과<br/>16 docs"]
    end
    
    subgraph CropFilter["Crop Filter"]
        direction TB
        CF1["질의 작물 추출<br/>'와사비'"]
        CF2{"문서 작물<br/>비교"}
        CF3["일치: score + 0.5"]
        CF4["불일치: score × 0.15"]
        CF5["없음: 유지"]
        
        CF1 --> CF2
        CF2 -->|일치| CF3
        CF2 -->|불일치| CF4
        CF2 -->|정보없음| CF5
    end
    
    subgraph SemanticDedup["Semantic Deduplication"]
        direction TB
        SD1["임베딩 계산"]
        SD2["유사도 행렬<br/>sim = emb @ emb.T"]
        SD3{"sim ≥ 0.85?"}
        SD4["후순위 문서 제거"]
        SD5["유지"]
        
        SD1 --> SD2 --> SD3
        SD3 -->|Yes| SD4
        SD3 -->|No| SD5
    end
    
    subgraph Reranking["Memory-aware Reranking"]
        direction TB
        RAM{"가용 RAM<br/>체크"}
        R1["< 0.8GB<br/>→ none (skip)"]
        R2["0.8-1.5GB<br/>→ LLM-lite"]
        R3["≥ 1.5GB<br/>→ BGE Reranker"]
        
        RAM --> R1
        RAM --> R2
        RAM --> R3
    end
    
    subgraph Output["출력"]
        Out["최종 Top-k<br/>4 docs"]
    end
    
    Input --> CropFilter
    CropFilter -->|"~12 docs"| SemanticDedup
    SemanticDedup -->|"~8 docs"| Reranking
    Reranking --> Output
```

### 3.5.2 작물 필터링 (Crop-aware Filtering)

농업 지식 그래프 연구[4,12]에서 작물별 맥락 의존성이 강조되었다. 질의의 작물과 문서의 작물 메타데이터를 비교하여 스코어를 조정한다.

| 조건 | 스코어 조정 | 효과 |
|------|------------|------|
| 작물 일치 | +0.5 | 관련 문서 우선 |
| 작물 불일치 | ×0.15 | 무관한 작물 정보 억제 |
| 작물 정보 없음 | 유지 | 일반 정보 보존 |

### 3.5.3 시맨틱 중복 제거 (Semantic Deduplication)

MMR[18]과 VRSD[19]를 참조하여 검색 결과의 다양성을 확보한다. 두 문서의 임베딩 벡터 간 코사인 유사도가 임계값(θ=0.85) 이상인 문서 쌍에서 후순위 문서를 제거한다.

### 3.5.4 메모리 적응형 리랭킹

런타임 가용 메모리에 따라 리랭커를 동적으로 선택한다:

| 가용 RAM | 리랭커 | 추가 메모리 | 설명 |
|----------|--------|------------|------|
| < 0.8GB | none | 0MB | 리랭킹 비활성화 |
| 0.8GB ~ 1.5GB | LLM-lite | ~0MB | llama.cpp 재사용 |
| ≥ 1.5GB | BGE | ~500MB | BGE-reranker-v2-m3 |

---

## 3.6 엣지 배포 최적화

### 3.6.1 메모리 계층 구조 (RAM vs Flash)

엣지 환경에서 "벡터 인덱스가 RAM에 다 못 올라간다"는 병목을 해결하기 위해 계층적 메모리 구조를 설계하였다.

```mermaid
flowchart TB
    subgraph RAM["<b>RAM (Hot Data)</b><br/>항상 메모리에 상주"]
        direction LR
        
        QC["Query Cache<br/>LRU 128<br/>검색 결과"]
        EC["Embedding Cache<br/>LRU 256<br/>쿼리 임베딩"]
        MP["FAISS mmap<br/>Active Pages<br/>자주 접근하는 벡터"]
        Model["LLM Weights<br/>~2.5GB<br/>Q4_K_M"]
    end
    
    subgraph Flash["<b>Flash/SSD (Cold Data)</b><br/>필요시 로드"]
        direction LR
        
        DenseFile["dense.faiss<br/>전체 인덱스<br/>(mmap)"]
        SparseFile["sparse.pkl<br/>TF-IDF 행렬"]
        CacheFile["responses.jsonl<br/>응답 캐시"]
        GraphFile["graph.json<br/>지식 그래프"]
    end
    
    subgraph MemoryBudget["<b>메모리 예산 (8GB RAM)</b>"]
        direction LR
        
        B1["LLM: ~2.5GB"]
        B2["Embedding: ~90MB"]
        B3["FAISS Active: ~200MB"]
        B4["Caches: ~50MB"]
        B5["Runtime: ~500MB"]
        B6["<b>여유: ~4.6GB</b>"]
    end
    
    RAM <-->|mmap I/O| Flash
```

### 3.6.2 LLM 양자화 전략

llama.cpp의 GGUF 포맷[23]을 활용하여 Q4_K_M 양자화를 기본으로 적용한다.

| 양자화 수준 | 메모리 (4B 모델) | 품질 손실 | 적용 환경 |
|------------|-----------------|----------|----------|
| FP16 (원본) | ~8GB | 없음 | 서버 환경 (GPU 필수) |
| INT8 | ~4GB | 최소 | 고사양 엣지 (8GB RAM) |
| **Q4_K_M** | ~2.5GB | 낮음 | **일반 엣지 (권장)** |
| Q2_K | ~1.5GB | 중간 | 극저사양 환경 |

Q4_K_M은 중요한 레이어는 5비트, 나머지는 4비트로 혼합 양자화하여 품질 대비 메모리 효율의 최적점으로 평가된다.

### 3.6.3 오프라인 폴백 모드 (Figure 6)

네트워크 단절 또는 LLM 장애 시 다음과 같은 폴백 전략을 적용한다:

```mermaid
flowchart TD
    Start["LLM 생성 실패"]
    
    subgraph Fallback["Fallback Chain"]
        direction TB
        
        F1["<b>1. Similar Cache</b><br/>────────────────<br/>ResponseCache.get_similar()<br/>임베딩 유사도 ≥ 0.9<br/>이전 유사 질의 응답 재활용"]
        
        F2["<b>2. Template Response</b><br/>────────────────<br/>TemplateResponder.generate()<br/>온톨로지 매칭 기반<br/>정형화된 응답 생성"]
        
        F3["<b>3. Search Only</b><br/>────────────────<br/>검색 결과만 반환<br/>LLM 없이 문서 목록 표시"]
        
        F1 -->|miss| F2
        F2 -->|"매칭 실패"| F3
    end
    
    subgraph Templates["Template Types"]
        direction LR
        T1["crop_env<br/>'와사비 온도 관련...'"]
        T2["crop_disease<br/>'토마토 흰가루병...'"]
        T3["disease_solution<br/>'연부병 해결...'"]
    end
    
    Start --> Fallback
    F2 -.-> Templates
    Fallback --> End["응답 반환<br/>fallback_mode 표시"]
```

| 폴백 단계 | 동작 | 언제 사용 |
|----------|------|----------|
| **Similar Cache** | 이전 유사 질의 응답 재활용 | 반복/유사 질의 시 |
| **Template Response** | 온톨로지 기반 정형 응답 생성 | 간단한 조회 시 |
| **Search Only** | LLM 없이 검색 결과만 반환 | LLM 완전 불가 시 |

---

## 3.7 관련 연구와의 비교

### 3.7.1 EdgeRAG vs ERA-SmartFarm-RAG

| 구분 | EdgeRAG[24] | ERA-SmartFarm-RAG |
|------|-------------|-------------------|
| **최적화 초점** | 범용 메모리 최적화 | 도메인 특화 + 엣지 배포 |
| **인덱싱 전략** | 온라인 계층적 인덱싱 | 오프라인 사전 인덱싱 + mmap |
| **검색 채널** | 단일 Dense | **Dense + Sparse + PathRAG** |
| **그래프 활용** | 없음 | **인과관계 그래프** |
| **도메인 지식** | 범용 | **농업 온톨로지 6개 유형** |
| **메모리 절감** | 계층적 로딩 50%↓ | **양자화 75%↓ + mmap** |
| **오프라인 지원** | 제한적 | **폴백 체인 (Cache→Template→Search)** |

### 3.7.2 MobileRAG 패턴 비교

| 구분 | MobileRAG (EcoVector+SCR) | ERA-SmartFarm-RAG |
|------|---------------------------|-------------------|
| **인덱스 파티셔닝** | k-means 클러스터 계층 | FAISS mmap (전체 인덱스) |
| **부분 로딩** | 클러스터별 on-demand | mmap lazy load (OS 페이지 캐시) |
| **토큰 절감** | SCR (Selective Content Reduction) | **Semantic Dedup + Crop Filter** |
| **런타임** | AI Edge / MLX | **llama.cpp GGUF** |

### 3.7.3 핵심 차별점

1. **도메인 특화**: 범용 메모리 최적화 대신 농업 온톨로지와 인과관계 그래프 활용
2. **3채널 검색**: 수치/단위 정보(EC, pH)의 정확한 매칭을 위한 Sparse 채널 유지
3. **경량 Context Shaping**: SCR 대신 Semantic Dedup + Crop Filter (구현 단순화)
4. **완전 오프라인**: Template Responder로 LLM 없이도 기본 응답 가능

---

## 참조

> 상세 Mermaid 다이어그램: [figures/ARCHITECTURE_MERMAID.md](figures/ARCHITECTURE_MERMAID.md)
> 
> 최신 연구 동향 및 기술 사양: [supplementary/architecture_details.md](supplementary/architecture_details.md)
