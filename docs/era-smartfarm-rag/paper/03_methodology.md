# 3. 제안 방법론 (Proposed Methodology)

본 장에서는 온프레미스 엣지 환경에서 안정적으로 동작하는 RAG 시스템을 구축하기 위해, (1) 지식 구조화(인덱싱) 단계의 자원 소모를 최소화하고, (2) 런타임은 경량 검색·생성만 수행하도록 분리한 **edge-first RAG 방법론**을 제안한다. 핵심 아이디어는 관계 추출(LLM 호출)에 의존하지 않는 **Tri-Graph(Entity–Sentence–Chunk)** 기반의 multi-hop 검색 채널을 구성하고 [6], 이를 Dense(FAISS)·Sparse(BM25) 채널과 **weighted RRF**로 융합하여 [12] 엣지에서도 견고한 검색 품질을 확보하는 것이다.

## 3.1 설계 목표 및 문제 정의

본 연구의 목표는 다음과 같다.

1. **LLM-free 원타임 인덱싱**: 인덱싱 과정에서 유료 LLM API 호출 없이(Zero-token indexing), 검색/생성에 필요한 모든 아티팩트를 오프라인에서 한 번에 생성한다.
2. **엣지 런타임 경량화**: 런타임은 (a) 인덱스 로드, (b) 검색(Fusion), (c) 로컬 LLM 생성만 수행하도록 단순화한다.
3. **다중 홉 질의 대응**: 단순 키워드/유사도 검색으로 해결되지 않는 원인–대응형 질의(예: “왜 생육이 떨어지나?”, “어떻게 예방하나?”)에 대해 multi-hop 컨텍스트를 구성한다.
4. **재현 가능성**: 입력 코퍼스(JSONL)만 주어지면 동일한 인덱싱·검색·생성 파이프라인을 재현할 수 있도록 아티팩트, 설정값, 측정 프로토콜을 명시한다.

우리는 RAG를 (i) 검색기 $R(q)$가 질의 $q$에 대한 상위 $K$개의 문서 조각(청크) $C=\{c_1,\dots,c_K\}$를 반환하고, (ii) 생성기 $G(q, C)$가 컨텍스트에 근거한 답변을 생성하는 문제로 정의한다 [1]. 본 연구의 핵심은 엣지 환경에서 $R$의 구성(인덱싱·검색) 비용을 최소화하면서도 $G$의 환각을 억제할 수 있는 근거 컨텍스트를 제공하는 것이다.

## 3.2 시스템 구성(4-컴포넌트 분해)

현재 구현은 다음 4개 컴포넌트로 분해된다.

1. **Offline / One-time Ingest & Indexing**: 코퍼스(JSONL)로부터 Dense/Sparse/Tri-Graph 아티팩트를 생성하여 디스크에 저장한다.
2. **검색(Retrieval)**: 런타임에서 Dense(FAISS), Sparse(BM25), Tri-Graph 채널을 수행하고 weighted RRF로 융합하여 상위 $K$개의 근거 청크를 반환한다 [12].
3. **LLM 추론(Generation)**: llama.cpp 기반 로컬 LLM으로 컨텍스트 기반 답변을 생성한다 [20].
4. **프론트엔드(UI)**: 사용자는 UI를 통해 질의를 입력하고, 답변과 근거 청크를 확인한다(본 연구에서는 UI는 시스템 검증용 인터페이스로 취급).

운영 흐름은 다음과 같다.

```
[Offline]  corpus.jsonl
    └─(LLM-free indexing)→  Dense/Sparse/TriGraph artifacts
                              ├─ dense.faiss + dense_docs.jsonl
                              ├─ sparse_bm25.pkl
                              └─ trigraph_edge/ (mmap-friendly .npy + CSR .npz)

[Runtime]  /query(q)
    └─ Retrieval: Dense + Sparse + TriGraph → Fusion(RRF)
          └─ contexts(top-K chunks)
                └─ Generation: llama.cpp → answer
```

## 3.3 LLM-free 원타임 인덱싱(Offline Indexing)

본 연구는 “인덱싱=오프라인, 검색/생성=런타임”의 분리를 강제한다. 인덱싱 스크립트는 입력 코퍼스를 읽어 다음 3종 아티팩트를 **한 번에** 생성한다.

- Dense 인덱스: FAISS 기반 chunk-level 벡터 검색
- Sparse 인덱스: BM25 기반 키워드 검색 [27]
- Tri-Graph 아티팩트: Entity–Sentence–Chunk 그래프(LLM-free multi-hop 채널) [6]

### 3.3.1 입력 코퍼스 및 청크 ID

입력은 JSONL 형식이며, 각 레코드는 최소 `id`와 텍스트 필드를 포함한다. 구현은 두 가지 입력 스타일을 지원한다.

- **(A) Raw 문서 입력**: `id`가 `#cN` 접미사를 갖지 않으면, 문장을 분리한 뒤 window chunking을 수행하여 내부적으로 `id#c{n}` 형식의 청크 ID를 생성한다.
- **(B) 이미 청킹된 코퍼스**: `id`가 `...#cN` 형식이면, 이를 **그대로 청크 ID로 유지**하여(재청킹 없음) 추적성과 재현성을 확보한다.

본 연구의 실험 코퍼스는 (B) 스타일을 사용하며, 예시는 `smartfarm-ingest/output/wasabi_en_ko_parallel.jsonl`와 같다.

### 3.3.2 문장 분할 및 window chunking

문서는 문장 단위로 분할한 뒤, $W$문장 윈도우를 stride $S$로 이동시키며 청크를 만든다. 구현 기본값은 $W=5$, $S=2$이며 이는 (i) 청크당 컨텍스트 길이를 제한해 엣지 LLM의 입력 부담을 줄이고, (ii) stride로 인한 중복을 통해 문맥 단절을 완화하는 절충이다.

### 3.3.3 Dense 인덱스 (FAISS, cosine/IP)

Dense 채널은 질의/청크 텍스트를 SentenceTransformer 임베딩으로 변환한 뒤, L2 정규화된 벡터에 대해 내적 기반 유사도 검색을 수행한다(내적=코사인 유사도). 구현은 FAISS `IndexFlatIP`(정확 검색)를 사용하며 [28], 인덱스/문서 메타데이터를 `dense.faiss`, `dense_docs.jsonl`로 저장한다. 런타임은 FAISS 인덱스를 memory-mapping으로 로드할 수 있어(옵션) 엣지에서 초기 로딩 시간을 줄인다.

### 3.3.4 Sparse 인덱스 (BM25)

Sparse 채널은 BM25 점수로 키워드 기반 매칭을 수행한다 [27]. 본 구현은 토크나이저를 단순화하여(기본: `\\w+` 정규식 기반) 엣지에서의 처리 부담을 낮추고, 상태를 `sparse_bm25.pkl`에 저장하여 런타임 로딩 시간을 줄인다.

### 3.3.5 Tri-Graph 아티팩트 생성 (Entity–Sentence–Chunk)

Tri-Graph는 관계(엔티티–엔티티) 추출을 수행하지 않고, **텍스트의 포함 관계**만으로 그래프를 구성한다 [6].

- 노드: Entity, Sentence, Chunk
- 엣지: Entity–Sentence(문장에 엔티티가 등장), Sentence–Chunk(문장이 속한 청크)
- 파생 엣지: Entity–Chunk(엔티티가 등장하는 문장들이 속한 청크)

엔티티는 도메인 온톨로지/사전 규칙에 의존하지 않고, 문장 텍스트에서 후보 토큰을 추출한 뒤 **문장 DF 기반 필터링**으로 선택한다(너무 희귀/너무 흔한 토큰 제거). 또한 엔티티 임베딩과 문장 임베딩을 float16으로 저장하고, 인접 관계는 CSR 희소 행렬(`.npz`)로 저장하여 디스크/메모리 사용을 최소화한다.

**Table 2. Offline indexing artifacts (runtime load targets)**

| Artifact | Path (default) | Purpose | Load mode |
|---|---|---|---|
| Dense index | `smartfarm-search/data/index/dense.faiss` | Chunk-level dense retrieval | FAISS mmap 가능 |
| Dense docs | `smartfarm-search/data/index/dense_docs.jsonl` | Chunk text + metadata | JSONL |
| Sparse(BM25) | `smartfarm-search/data/index/sparse_bm25.pkl` | Keyword retrieval state | Pickle |
| Tri-Graph meta | `smartfarm-search/data/index/trigraph_edge/meta.json` | Corpus hash, dims, stats | JSON |
| Tri-Graph embeddings | `.../entity_embeddings.npy`, `.../sentence_embeddings.npy` | Entity/Sentence vectors | `.npy` mmap |
| Tri-Graph mapping | `.../sentence_chunk_idx.npy`, `.../chunk_ids.json` | Sentence→Chunk, Chunk ids | `.npy`/JSON |
| Tri-Graph adjacency | `.../s2e.npz`, `.../e2s.npz`, `.../e2c.npz` | CSR adjacency | NPZ |

## 3.4 Tri-Graph Multi-hop 검색(LLM-free)

Tri-Graph 검색은 “질의 임베딩으로 시드 엔티티를 찾고 → 의미 전파(semantic bridging)로 관련 엔티티/문장을 확장하고 → 청크 후보를 점수화”하는 흐름으로 구성된다 [6].

### 3.4.1 Seed entity selection

질의 $q$를 임베딩 벡터 $\\mathbf{q}$로 변환하고, 엔티티 임베딩 $\\mathbf{E}\\in\\mathbb{R}^{|V_e|\\times d}$와의 유사도를 계산하여 상위 $k_e$개의 시드 엔티티를 선택한다. 구현 기본값은 $k_e=10$, 임계값 $\\tau_e=0.35$이다.

### 3.4.2 Semantic bridging (Entity→Sentence→Entity)

시드 엔티티 $e$로부터 연결된 문장 집합 $S(e)$를 가져온 뒤, 각 문장 임베딩과 $\\mathbf{q}$의 유사도로 상위 $k_s$개의 문장을 선택한다. 선택된 문장의 엔티티들을 다시 활성화시키며, 이를 최대 $T$회 반복해 multi-hop 확장을 수행한다(기본: $k_s=3$, $T=3$). 활성 엔티티 수는 `max_active_entities`로 상한을 두어 엣지 런타임을 제한한다.

### 3.4.3 Chunk scoring & candidate pruning

활성화된 엔티티들의 점수를 엔티티–청크 연결(e2c)에 누적하여 청크 점수를 계산한다. 이후 후보 청크 수를 `max_candidate_chunks`로 제한하여(기본: 256) 후속 단계(PPR 및 융합)의 계산량을 통제한다.

### 3.4.4 Optional global aggregation via PPR

선택적으로, 상위 엔티티–청크로 유도된 부분 그래프에서 개인화 PageRank(PPR) 스타일의 전역 집계를 수행해 청크를 재정렬한다 [26]. 본 구현은 전체 그래프가 아닌 **후보 subgraph**에서만 PPR을 수행해 엣지 계산량을 제한한다(기본: damping=0.85, iters=16).

**Algorithm 1. Tri-Graph retrieval (semantic bridging + optional PPR)**

```text
Input: query q, top_k K
1: q_vec ← Embed(q)
2: seeds ← TopEntities(q_vec; k_e, τ_e)
3: entity_scores ← ExpandEntities(q_vec, seeds; T, k_s, τ_iter, max_active)
4: chunk_scores ← Sum_{e}( entity_scores[e] → e2c[e] )
5: if use_ppr: chunk_scores ← PPR_Rerank(entity_scores, chunk_scores; damping, iters)
6: return TopChunks(chunk_scores, K)
```

## 3.5 3채널 융합: Weighted RRF + 질의 적응 가중치

Dense/Sparse/Tri-Graph 각 채널이 반환한 순위 리스트를 weighted RRF로 융합한다 [12]. 문서(청크) $d$에 대한 최종 점수는 다음과 같이 계산한다.

$$
\\mathrm{score}(d)=\\sum_{c\\in\\{dense,sparse,trigraph\\}} \\alpha_c \\cdot \\frac{1}{k + \\mathrm{rank}_c(d) + 1}
$$

여기서 $k$는 RRF 하이퍼파라미터(기본 60), $\\alpha_c$는 채널 가중치이며, 구현에서는 $\\mathrm{rank}_c(d)$를 0부터 시작하는 순위 인덱스로 취급해 분모에 +1을 더한다. 본 구현은 간단한 질의 적응 규칙을 사용한다.

- **수치/단위 포함 질의**(예: ℃, %, EC, pH 등): Sparse 가중치 증가
- **원인/해결/방법 질의**: Tri-Graph 가중치 증가

기본 가중치는 (Dense, Sparse, Tri-Graph) = (0.45, 0.35, 0.20)이며, Tri-Graph 로드 실패 시 해당 채널은 자동으로 제외된다.

## 3.6 엣지 LLM 생성(Answer Generation)

생성기는 검색된 상위 $K$개 청크를 컨텍스트로 받아 답변을 생성한다 [1]. 본 연구의 구현은 llama.cpp 서버를 호출하여 JSON 스키마 기반으로 답변 필드만을 강제 출력한다 [20]. 이는 엣지 환경에서 불필요한 장문 출력(자기평가/중복 설명)을 줄이고, 파싱 실패로 인한 오류 전파를 완화하기 위한 설계다.

### 3.6.1 컨텍스트 트리밍(엣지 친화)

엣지 환경에서 생성 지연을 줄이기 위해, 컨텍스트는 최대 문서 수 및 문자 수로 트리밍한다(기본: 2개 문서, 문서당 600자). 이때 검색 근거는 유지하되, 프롬프트 길이를 상한으로 두어 타임아웃 위험을 줄인다.

### 3.6.2 실패 인지 및 폴백

LLM 호출 실패(타임아웃/서버 장애 등) 시, 시스템은 (i) 캐시 재사용, (ii) 템플릿 기반 응답, (iii) 검색 결과만 반환 등의 폴백 모드로 전환하여 서비스 연속성을 유지한다. 이는 엣지 환경에서의 불안정성을 고려한 운영 설계이며, 본 논문에서는 “정상 생성”뿐 아니라 “폴백 빈도”를 신뢰성 지표로 함께 측정한다(Section 3.7).

## 3.7 평가 프로토콜(방법)

방법론 단계에서의 평가는 “검색 성능”, “생성 품질”, “엣지 실용성(지연/메모리)”의 세 축으로 구성한다.

1. **Retrieval metrics**: Precision@K, Recall@K, MRR, NDCG@K 등을 사용해 채널별/융합별 검색 품질을 비교한다.
2. **Reference-free Generation metrics**: RAGAS 기반 faithfulness, answer relevancy, context precision/recall 등을 측정한다 [22].
3. **Edge practicality**: (a) 오프라인 인덱싱 시간, (b) 런타임 검색 지연(p50/p95), (c) 생성 지연 및 end-to-end 지연, (d) 메모리 사용량(상주/피크)을 측정한다.

특히 본 구현의 기본 질의 파라미터는 `top_k=4`이며, 실험에서도 $K=4$를 기본값으로 사용한다.

## 3.8 재현 가능한 실행 구성(Docker/스크립트)

재현을 위해, (i) 오프라인 인덱싱은 단일 스크립트로 수행하고, (ii) 런타임은 Docker Compose로 서비스 스택을 기동한다.

1. **Offline indexing**
   - `python smartfarm-ingest/scripts/indexing/build_trigraph_index_v2.py --input-jsonl <corpus> --lang ko --embed-model-id minilm`
2. **Runtime**
   - API는 인덱스 디렉토리(`smartfarm-search/data/index/`)를 로드하고, Tri-Graph 채널은 `TRIGRAPH_INDEX_DIR`로 지정된 아티팩트를 로드한다.
   - llama.cpp는 별도 컨테이너(또는 로컬 서버)로 실행되며, API는 `LLMLITE_HOST`로 접근한다.
