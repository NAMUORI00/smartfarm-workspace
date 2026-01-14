# 4. 시스템 구현 (Implementation)

## 4.1 기술 스택

| 구성요소 | 서버 환경 | 엣지 환경 | 참조 |
|----------|----------|----------|------|
| Dense Retrieval | FAISS + Qwen3-Embedding-0.6B (6억 파라미터, ~1.2GB) | FAISS + MiniLM-L6 (2,200만 파라미터, ~90MB) | [25,26] |
| Sparse Retrieval | TF-IDF (scikit-learn) | TF-IDF (동일) | - |
| 지식 그래프 | 커스텀 그래프 (JSON) | 서브셋 그래프 | - |
| LLM | llama.cpp (FP16/INT8) | llama.cpp (Q4_K_M) | [23] |
| API | FastAPI + Docker | FastAPI (경량) | - |
| 오프라인 폴백 | - | 캐시 + 규칙 기반 | [24,29] |

## 4.2 핵심 모듈 구현

### 4.2.1 HybridDATRetriever

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

### 4.2.2 PathRAGRetriever (PathRAG-lite)

PathRAG[8]의 경로 탐색 개념을 차용한 경량 구현이다. 원본 PathRAG의 relational path pruning 대신 BFS(Breadth-First Search, 너비 우선 탐색: 가까운 노드부터 순서대로 방문) 기반 단순화된 탐색을 수행한다.

**탐색 전략:**
- 시작점: 쿼리에서 매칭된 온톨로지 개념 노드 (예: "와사비 고수온" → crop:와사비, env:고수온)
- 최대 깊이: 2-hop (2번까지 엣지를 따라 이동, 기본값)
- 인과관계 엣지(`causes`, `solved_by`) 우선 탐색하여 원인→결과→해결책 문서 수집

### 4.2.3 GraphBuilder

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

### 4.2.4 EmbeddingRetriever

FAISS(Facebook AI Similarity Search, 벡터 유사도 검색 라이브러리) 기반 Dense 검색을 담당한다.

**특징:**
- Lazy loading(지연 로딩): 시작 시가 아닌 첫 쿼리 시점에 모델 로드 → 초기 메모리 절약
- L2 정규화된 임베딩으로 코사인 유사도 검색 (벡터 길이 1로 맞춰 방향만 비교)
- mmap(memory-mapped file, 파일을 메모리에 통째로 올리지 않고 필요한 부분만 로드) 지원으로 대용량 인덱스도 저메모리에서 사용 가능

## 4.3 메모리 적응형 리랭킹

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

## 4.4 인덱스 영속화

오프라인 환경 지원을 위해 인덱스(검색용 데이터 구조)를 파일로 저장/로드한다. 시스템 재시작 시 문서를 다시 처리하지 않고 저장된 인덱스를 바로 로드한다:

| 파일 | 내용 | 형식 |
|------|------|------|
| `dense.faiss` | 문서 임베딩 벡터 인덱스 | Binary (mmap 가능, 부분 로드) |
| `dense_docs.jsonl` | 문서 텍스트 및 메타데이터 | JSON Lines (한 줄에 한 문서) |
| `sparse.pkl` | TF-IDF 키워드 빈도 행렬 | Pickle (Python 직렬화) |

## 4.5 그래프 스키마

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

## 4.6 엣지 배포 사양

| 환경 | 최소 사양 | 권장 사양 | 지원 기능 |
|------|----------|----------|----------|
| **서버** | 32GB RAM, GPU | 64GB RAM, RTX 4090 | 전체 기능 |
| **엣지 게이트웨이** | 8GB RAM, CPU | 16GB RAM, CPU/NPU | RAG + Q4 LLM |
| **저사양 엣지** | 4GB RAM | 8GB RAM | 검색 전용 |
| **IoT 노드** | 512MB RAM | 1GB RAM | 센서 + 규칙 |

## 4.7 EdgeRAG와의 구현 비교

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
