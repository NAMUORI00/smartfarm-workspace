# ERA-SmartFarm-RAG 6-Layer 아키텍처 Mermaid 다이어그램

> **목적**: 디자이너에게 전달하기 위한 시각화 가능한 Mermaid 다이어그램 모음

---

## 1. 전체 시스템 아키텍처 (6-Layer Stack)

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#e1f5fe', 'primaryTextColor': '#01579b', 'primaryBorderColor': '#0288d1', 'lineColor': '#0288d1', 'secondaryColor': '#fff3e0', 'tertiaryColor': '#f3e5f5'}}}%%

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

    subgraph L3["<b>Layer 3: Context Shaping</b><br/>🎯 논문 핵심 기여"]
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

    subgraph L0["<b>Layer 0: Device & Runtime</b><br/>⚡ 8GB RAM / Q4_K_M"]
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

    style L3 fill:#fff9c4,stroke:#f9a825,stroke-width:3px
    style L2 fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style L0 fill:#efebe9,stroke:#5d4037,stroke-width:2px
```

---

## 2. Query 처리 플로우 (메인 파이프라인)

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#e8f5e9', 'primaryTextColor': '#1b5e20'}}}%%

flowchart TD
    Start([/"사용자 질의"/]) --> Cache{"ResponseCache<br/>조회"}
    
    Cache -->|Hit| CacheReturn["캐시 응답 반환<br/>⚡ <10ms"]
    Cache -->|Miss| Retrieval
    
    subgraph Retrieval["Hybrid Retrieval"]
        direction TB
        Onto["온톨로지 매칭<br/>crop/env/disease..."]
        Alpha["Dynamic Alpha 계산<br/>α_d, α_s, α_p"]
        
        subgraph ThreeChannel["3채널 병렬 검색"]
            D["Dense<br/>FAISS ANN"]
            S["Sparse<br/>TF-IDF cosine"]
            P["PathRAG<br/>Graph BFS"]
        end
        
        Fusion["Score Fusion<br/>α_d×D + α_s×S + α_p×P"]
        
        Onto --> Alpha
        Alpha --> ThreeChannel
        ThreeChannel --> Fusion
    end
    
    Retrieval --> Shaping
    
    subgraph Shaping["Context Shaping"]
        direction TB
        CF["Crop Filter<br/>작물 일치 +0.5"]
        SD["Semantic Dedup<br/>유사도 ≥0.85 제거"]
        RR["Reranking<br/>(메모리 적응형)"]
        TopK["Top-k 선택<br/>k=4"]
        
        CF --> SD --> RR --> TopK
    end
    
    Shaping --> LLM{"LLM 생성<br/>llama.cpp"}
    
    LLM -->|성공| Success["정상 응답<br/>+ 캐시 저장"]
    LLM -->|실패| Fallback
    
    subgraph Fallback["Fallback Chain"]
        direction TB
        Similar["1. Similar Cache<br/>임베딩 유사 검색"]
        Tmpl["2. Template Response<br/>온톨로지 기반"]
        Raw["3. Search Only<br/>검색 결과만"]
        
        Similar --> Tmpl --> Raw
    end
    
    CacheReturn --> End([/"응답 반환"/])
    Success --> End
    Fallback --> End

    style Retrieval fill:#e3f2fd,stroke:#1976d2
    style Shaping fill:#fff9c4,stroke:#f9a825
    style Fallback fill:#ffebee,stroke:#c62828
```

---

## 3. HybridDATRetriever 상세 (3채널 융합)

```mermaid
%%{init: {'theme': 'base'}}%%

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

    style Dense fill:#bbdefb,stroke:#1976d2
    style Sparse fill:#c8e6c9,stroke:#388e3c
    style Path fill:#f8bbd9,stroke:#c2185b
```

---

## 4. PathRAG-lite 인과관계 그래프 탐색

```mermaid
%%{init: {'theme': 'base'}}%%

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

    style P1 fill:#ffcdd2,stroke:#c62828
    style P2 fill:#fff9c4,stroke:#f9a825
    style P3 fill:#c8e6c9,stroke:#388e3c
    style P4 fill:#c8e6c9,stroke:#388e3c
```

---

## 5. Context Shaping 상세

```mermaid
%%{init: {'theme': 'base'}}%%

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

    style CropFilter fill:#e8f5e9,stroke:#2e7d32
    style SemanticDedup fill:#fff3e0,stroke:#ef6c00
    style Reranking fill:#e3f2fd,stroke:#1565c0
```

---

## 6. 오프라인 폴백 전략

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#ffebee'}}}%%

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
        T3["crop_nutrient<br/>'딸기 양액 관리...'"]
        T4["disease_solution<br/>'연부병 해결...'"]
        T5["general<br/>'검색 결과 N개...'"]
    end
    
    Start --> Fallback
    F2 -.-> Templates
    Fallback --> End["응답 반환<br/>fallback_mode 표시"]

    style F1 fill:#e3f2fd,stroke:#1565c0
    style F2 fill:#fff9c4,stroke:#f9a825
    style F3 fill:#ffcdd2,stroke:#c62828
```

---

## 7. 메모리 계층 구조 (RAM vs Flash)

```mermaid
%%{init: {'theme': 'base'}}%%

flowchart TB
    subgraph RAM["<b>RAM (Hot Data)</b><br/>항상 메모리에 상주"]
        direction LR
        
        QC["Query Cache<br/>LRU 128<br/>검색 결과"]
        EC["Embedding Cache<br/>LRU 256<br/>쿼리 임베딩"]
        MP["FAISS mmap<br/>Active Pages<br/>자주 접근하는 벡터"]
        Model["LLM Weights<br/>~400MB<br/>Q4_K_M"]
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
        B2["Embedding Model: ~90MB"]
        B3["FAISS Active: ~200MB"]
        B4["Caches: ~50MB"]
        B5["Runtime: ~500MB"]
        B6["<b>여유: ~4.6GB</b>"]
    end
    
    RAM <-->|mmap I/O| Flash
    
    style RAM fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style Flash fill:#efebe9,stroke:#5d4037,stroke-width:2px
    style MemoryBudget fill:#f3e5f5,stroke:#7b1fa2
```

---

## 8. 온톨로지 구조 (6개 Concept Types)

```mermaid
%%{init: {'theme': 'base'}}%%

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

    style Crop fill:#c8e6c9,stroke:#2e7d32
    style Env fill:#bbdefb,stroke:#1565c0
    style Nutrient fill:#fff9c4,stroke:#f9a825
    style Disease fill:#ffcdd2,stroke:#c62828
    style Stage fill:#e1bee7,stroke:#7b1fa2
    style Practice fill:#ffe0b2,stroke:#ef6c00
```

---

## 9. Reranker 선택 로직

```mermaid
%%{init: {'theme': 'base'}}%%

flowchart TD
    Start["리랭킹 요청"]
    
    AutoCheck{"settings.AUTO_RERANK<br/>활성화?"}
    
    AutoCheck -->|No| ModeCheck{"req.ranker<br/>명시?"}
    ModeCheck -->|none| Skip["리랭킹 스킵"]
    ModeCheck -->|llm| LLMJudge
    ModeCheck -->|bge| BGE
    ModeCheck -->|llm-lite| LLMLite
    
    AutoCheck -->|Yes| RAMCheck
    
    subgraph RAMCheck["메모리 체크"]
        direction TB
        GetRAM["RAM = _available_ram_gb()"]
        GetVRAM["VRAM = _available_vram_gb()"]
    end
    
    RAMCheck --> Decision
    
    subgraph Decision["자동 선택"]
        D1{"VRAM ≥ 1.5GB<br/>또는<br/>RAM ≥ 1.5GB?"}
        D2{"RAM ≥ 0.8GB?"}
        
        D1 -->|Yes| BGE
        D1 -->|No| D2
        D2 -->|Yes| LLMLite
        D2 -->|No| Skip
    end
    
    subgraph Rerankers["Reranker 구현"]
        BGE["<b>BGEReranker</b><br/>────────────<br/>BAAI/bge-reranker-v2-m3<br/>~500MB 메모리<br/>고품질"]
        
        LLMJudge["<b>LLMJudgeReranker</b><br/>────────────<br/>외부 LLM API<br/>score 파싱<br/>도메인 보너스"]
        
        LLMLite["<b>LLMLiteReranker</b><br/>────────────<br/>llama.cpp 재사용<br/>~0MB 추가<br/>경량"]
    end
    
    Skip --> End["검색 결과 그대로 반환"]
    BGE --> End
    LLMJudge --> End
    LLMLite --> End

    style BGE fill:#c8e6c9,stroke:#2e7d32
    style LLMLite fill:#fff9c4,stroke:#f9a825
    style LLMJudge fill:#e3f2fd,stroke:#1565c0
```

---

## 10. 전체 시스템 컴포넌트 맵

```mermaid
%%{init: {'theme': 'base'}}%%

flowchart TB
    subgraph Frontend["Frontend"]
        Streamlit["Streamlit App<br/>frontend/streamlit/"]
    end
    
    subgraph API["FastAPI Backend"]
        direction TB
        Main["core/main.py<br/>FastAPI App"]
        
        subgraph Routes["API Routes"]
            RQ["routes_query.py<br/>/query"]
            RI["routes_ingest.py<br/>/ingest"]
            RP["routes_prompts.py<br/>/prompts"]
            RM["routes_monitoring.py<br/>/health"]
        end
        
        Deps["deps.py<br/>전역 리트리버 초기화"]
    end
    
    subgraph Services["Core Services"]
        direction TB
        
        subgraph Retrieval["Retrieval/"]
            Hybrid["Hybrid.py<br/>HybridDATRetriever"]
            Emb["Embeddings.py<br/>EmbeddingRetriever"]
            Spar["Sparse.py<br/>MiniStore"]
            PR["PathRAG.py<br/>PathRAGRetriever"]
        end
        
        subgraph Ingest["Ingest/"]
            GB["GraphBuilder.py"]
            Chunk["Chunking.py"]
            OCR["OCREngine.py"]
        end
        
        LLM["LLM.py<br/>llama.cpp 클라이언트"]
        Ont["Ontology.py"]
        Cache["ResponseCache.py"]
        Tmpl["TemplateResponder.py"]
    end
    
    subgraph Rerankers["Rerankers/"]
        BGE["BGEReranker"]
        Judge["LLMJudgeReranker"]
        Lite["LLMLiteReranker"]
    end
    
    subgraph Models["Models/"]
        Schema["Schemas.py<br/>SourceDoc, QueryRequest"]
        Graph["Graph.py<br/>SmartFarmGraph"]
    end
    
    subgraph Config["Config/"]
        Settings["Settings.py<br/>환경변수 관리"]
    end
    
    subgraph Data["Data Storage"]
        Index["data/index/<br/>dense.faiss, sparse.pkl"]
        CacheFile["data/cache/<br/>responses.jsonl"]
        OntoFile["data/ontology/<br/>wasabi_ontology.json"]
    end
    
    Streamlit --> API
    API --> Services
    Services --> Rerankers
    Services --> Models
    Services --> Config
    Services --> Data

    style Retrieval fill:#e3f2fd,stroke:#1565c0
    style Ingest fill:#fff9c4,stroke:#f9a825
    style Rerankers fill:#e8f5e9,stroke:#2e7d32
```

---

## 디자이너 전달 가이드

### 색상 팔레트 (권장)

| Layer | 색상 | HEX | 의미 |
|-------|------|-----|------|
| Layer 0 (Device) | 갈색 | `#efebe9` | 하드웨어/제약 |
| Layer 1 (Storage) | 보라 | `#f3e5f5` | 데이터 저장 |
| Layer 2 (Retrieval) | 파랑 | `#e3f2fd` | 검색 엔진 |
| Layer 3 (Shaping) | 노랑 | `#fff9c4` | **핵심 기여** |
| Layer 4 (Generation) | 초록 | `#e8f5e9` | 생성/응답 |
| Layer 5 (Application) | 회색 | `#fafafa` | UI/API |

### 강조 포인트

1. **Layer 3 (Context Shaping)** - 논문 핵심 기여, 굵은 테두리 또는 하이라이트
2. **Layer 2 (3채널 융합)** - Dense/Sparse/PathRAG 세 갈래 화살표
3. **Layer 0 (리소스 제약)** - "8GB RAM", "Q4_K_M" 뱃지 표시
4. **RAM ↔ Flash 경계선** - Layer 1-2 사이 점선

### Figure 우선순위

| 순위 | 다이어그램 | 용도 |
|------|----------|------|
| 1 | 전체 6-Layer Stack (섹션 1) | 논문 Figure 1 |
| 2 | Query 처리 플로우 (섹션 2) | 논문 Figure 2 |
| 3 | 3채널 융합 상세 (섹션 3) | 보충 자료 |
| 4 | 오프라인 폴백 (섹션 6) | 논문 Figure 3 |
