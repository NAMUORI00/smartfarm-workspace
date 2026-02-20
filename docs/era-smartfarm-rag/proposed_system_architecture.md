# Proposed System Architecture — Figure Description

> **파일**: `proposed_system_architecture.png`
> **생성일**: 2026-02-19

---

## 생성 프롬프트

```text
Create a high-resolution system architecture diagram. NO title. Clean white background.
Modern infographic with flat icons. Colored boundary regions. Sharp crisp text.
Non-hierarchical hub-and-spoke layout with Knowledge Base at center.

=== TOP region (Soft Blue background) === Label: "Offline Knowledge Construction"

[📄 Public Agricultural Data (Papers, Manuals)] → [🔍 Multimodal Parser]

From Multimodal Parser, TWO separate paths split:

Path 1 (direct): Arrow going DOWN-RIGHT from Parser directly into Vector Index box
inside KB, labeled "chunk embeddings (direct)"

Path 2 (via LLM): Arrow going RIGHT from Parser → [🧠 LLM Extraction (Entity + Relation)]
→ Arrow going DOWN-LEFT into Knowledge Graph box inside KB, labeled "entities, relations"

=== CENTER (White background, dark border, PROMINENT) === "Unified Knowledge Base"
Show TWO distinct sub-components side by side inside one large bordered box:
LEFT: [Vector Index] with a database/cylinder icon — receives direct embeddings
RIGHT: [Knowledge Graph] with a network/graph icon — receives extracted entities and relations
Tag below: "public + private tier merged"

Arrow from Private Enrichment going UP into BOTH sub-components labeled
"node addition, relation update"
Arrow going DOWN-RIGHT from BOTH sub-components into retrieval channels labeled
"read from merged KB"

=== BOTTOM (Soft Green background, large) === Label: "Edge Runtime"

Left sub-section "Private Knowledge Enrichment":
[🌡️📝💬 Farm Private Data (Sensor, Memo, Conversation)] → [⚡ SLM (Local Extraction)]
Arrow UP into KB labeled "private update (farm_id 🔒)"

Right sub-section "3-Channel Retrieval & Generation":
[👨‍🌾 User Query] → three parallel channels:
[📐 Dense Retrieval] ← reads from Vector Index
[🔤 Sparse Retrieval] ← reads from Vector Index
[🕸️ Graph Traversal (multi-hop)] ← reads from Knowledge Graph
Three merge → [⚖️ Weighted RRF Fusion] → [💬 SLM (Answer Generation)] → [✅ Answer / Advice]

=== VERY BOTTOM (Light Grey, separated) === Label: "Evaluation & Analysis"
NO connections to other regions. Three independent parallel boxes:
[📋 QA Benchmarks] [📈 RAGAS Metrics] [⏱️ System Metrics]

CRITICAL:
- Show the TWO distinct paths from Parser clearly:
  direct to Vector Index vs via LLM to Knowledge Graph
- NO title text
- Very high resolution, sharp text
- Clean layout
```

---

## 시각 요소 설명

### 경계 영역 (Boundary Regions)

| 영역 | 색상 | 설명 |
|------|------|------|
| Offline Knowledge Construction | Soft Blue | 서버사이드 오프라인 지식 구축 프로세스 |
| Unified Knowledge Base | White (bordered) | 중앙 허브 — Vector Index + Knowledge Graph |
| Edge Runtime | Soft Green | 엣지 디바이스 — Private 업데이트 + 3채널 검색·생성 |
| Evaluation & Analysis | Light Grey | 완전 독립 평가, 다른 영역과 연결 없음 |

### 아이콘 및 구성 요소

| 아이콘 | 구성 요소 | 위치 | 역할 |
|--------|-----------|------|------|
| 📄 문서 스택 | Public Agricultural Data | Offline | 논문·매뉴얼·가이드라인 원본 데이터 |
| 🔍 돋보기 | Multimodal Parser | Offline | 텍스트·테이블·이미지 멀티모달 파싱 |
| 🧠 뇌 | LLM Extraction | Offline | 대형 LLM으로 엔티티·관계 추출 (Knowledge Graph용) |
| 💾 DB 실린더 | Vector Index | KB 중앙 | Dense+Sparse 임베딩 저장 (direct from parser) |
| 🕸️ 네트워크 | Knowledge Graph | KB 중앙 | 엔티티·관계 구조화 저장 (via LLM extraction) |
| 🌡️📝💬 | Farm Private Data | Edge Runtime | 센서·메모·대화 — 농장주 개인 데이터 |
| ⚡ 칩 | SLM (Local Extraction) | Edge Runtime | 경량 SLM으로 private 엔티티·관계 로컬 추출 |
| 🔒 자물쇠 | farm_id isolated | Edge Runtime | 소버린 — private 데이터 외부 유출 차단 |
| 👨‍🌾 농부 | User Query | Edge Runtime | 농장주/관리자의 질의 |
| 📐 벡터 | Dense Retrieval | Edge Runtime | 임베딩 기반 시맨틱 유사도 검색 (← Vector Index) |
| 🔤 키워드 | Sparse Retrieval | Edge Runtime | 단어 일치(BM25) 기반 검색 (← Vector Index) |
| 🕸️ 그래프 | Graph Traversal | Edge Runtime | 지식그래프 멀티홉 추론 검색 (← Knowledge Graph) |
| ⚖️ 저울 | Weighted RRF Fusion | Edge Runtime | 3채널 가중치 랭크 융합 |
| 💬 말풍선 | SLM (Answer Generation) | Edge Runtime | 경량 SLM으로 답변/조언 생성 |
| ✅ 체크 | Answer / Advice | Edge Runtime | 최종 출력 |
| 📋 체크리스트 | QA Benchmarks | Evaluation | AgXQA, HotpotQA, 2WikiMultiHop (독립) |
| 📈 차트 | RAGAS Metrics | Evaluation | Faithfulness, Answer Relevance (독립) |
| ⏱️ 스톱워치 | System Metrics | Evaluation | Latency, Memory 사용량 (독립) |

---

## 데이터 흐름

### Offline → KB (두 갈래 경로)

| 경로 | 흐름 | KB 대상 |
|------|------|---------|
| **직접 (Path 1)** | Parser → chunk embeddings → **직접 저장** | Vector Index |
| **LLM 경유 (Path 2)** | Parser → LLM Extraction → entities, relations | Knowledge Graph |

### Edge → KB (Private 업데이트)

| 흐름 | 라벨 |
|------|------|
| Farm Private Data → SLM (Local Extraction) → KB | `node addition, relation update (private, farm_id 🔒)` |

### KB → Edge Retrieval (읽기)

| 소스 | 채널 |
|------|------|
| Vector Index | Dense Retrieval, Sparse Retrieval |
| Knowledge Graph | Graph Traversal (multi-hop) |

### 융합 → 답변

```
Dense + Sparse + Graph → Weighted RRF Fusion → SLM (Answer Generation) → Answer / Advice
```

### Evaluation (독립)

- QA Benchmarks, RAGAS Metrics, System Metrics 3개 트랙이 **병렬 독립 실행**
- 다른 영역과 **연결 없음**

---

## 설계 원칙

| 원칙 | 표현 |
|------|------|
| **비계층적 (Non-hierarchical)** | Layer 번호 없음, 경계(Boundary) 이름만 사용 |
| **KB 중심 허브 (Hub-and-spoke)** | 통합 KB가 중앙에서 public + private 머지 |
| **LLM vs SLM 구분** | 서버사이드 고품질 = **LLM**, 엣지 경량 = **SLM** |
| **두 갈래 인제스트** | Parser → Vector Index (직접) / Parser → LLM → Knowledge Graph |
| **소버린 (Sovereign)** | 🔒 farm_id 격리, SLM 로컬 전용 |
| **3채널 융합** | Dense + Sparse + Graph → Weighted RRF |
| **평가 독립** | 다른 영역과 연결 없음, 내부도 병렬 독립 |
