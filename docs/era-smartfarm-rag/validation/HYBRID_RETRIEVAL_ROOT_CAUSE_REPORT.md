# Hybrid Retrieval 성능 저하 원인 및 개선 보고서

작성일: 2026-01-21  
대상: ERA SmartFarm RAG (Jetson 64GB 기준, 도메인 추가학습 없이)

---

## 1. 목적

현재 제안된 하이브리드 구조가 여러 벤치마크에서 TF‑IDF 대비 낮은 성능을 보이는 원인을 규명하고, 엣지 제약(특히 Jetson 64GB)에서 재현 가능한 개선 방안을 정리한다.

---

## 2. 관측된 성능 저하 (내부 벤치)

내부 baseline 결과에서 **sparse(TF‑IDF)**가 가장 높고, **proposed(full)**가 가장 낮다.

- 결과 파일: `era-smartfarm-rag/output/baseline_rrf_v2_baseline/baseline_summary.json`

| 방법 | MRR | NDCG |
|------|-----|------|
| sparse_only (TF‑IDF) | 0.567 | 0.607 |
| rrf_hybrid | 0.344 | 0.387 |
| proposed (full) | 0.038 | 0.048 |

또한 ablation 결과에서 **rrf+dat+onto**가 가장 높고, **crop/dedup/pathrag**가 오히려 저하 및 지연을 유발한다.

- 결과 파일: `era-smartfarm-rag/output/ablation_rrf/ablation_summary.json`

| 구성 | MRR | NDCG | 비고 |
|------|-----|------|------|
| rrf+dat+onto | 0.527 | 0.563 | 최고 성능 |
| rrf+dat+onto+crop | 0.437 | 0.460 | 성능 하락 |
| rrf+dat+onto+dedup | 0.492 | 0.514 | **지연 폭증** (mean 284ms) |
| rrf_full_proposed | 0.375 | 0.397 | 성능 저하 + 지연 |

---

## 3. 원인 분석

### 3.1 점수 스케일 불일치 + Min-Max 정규화
- `HybridDATRetriever`는 dense/sparse/pathrag 점수를 **min-max 정규화 후 가중합**한다.
- 이 방식은 분포가 다른 점수 간 스케일을 강제로 맞추며, 상위 몇 개 결과만 급격히 부각되는 구조를 만든다.
- 근거: `era-smartfarm-rag/core/Services/Retrieval/Hybrid.py`

### 3.2 규칙 기반 가중치(DAT)와 휴리스틱 필터의 부정적 상호작용
- `dynamic_alphas()`는 ontology 매칭/길이/단위 패턴으로 가중치를 결정한다.
- 잘못된 활성화가 sparse/dense 균형을 깨트리고, 특히 **crop filter**는 실제 정답을 과도하게 패널티 처리한다.
- 근거: `Hybrid.py`의 `dynamic_alphas`, `_apply_crop_filter`

### 3.3 Semantic Dedup으로 인한 Recall 손실 + 지연 증가
- 임베딩 유사도 기반 중복 제거는 **정답 문서를 누락**시키는 경우가 많고, CPU 환경에서 지연이 급증한다.
- ablation에서 dedup 활성화 시 평균 284ms로 폭증.
- 근거: `Hybrid.py`의 `_deduplicate`

### 3.4 PathRAG 그래프 품질/노이즈 문제
- PathRAG는 키워드 기반 매칭 + 2hop 탐색으로 문서를 연결한다.
- 그래프는 **규칙 기반 인과 패턴**으로 구성되어 노이즈가 많고, 범용 벤치에서는 매칭 실패/부정확 연결이 증가한다.
- 근거: `PathRAG.py`, `GraphBuilder.py`

### 3.5 Sparse 기준점의 약화 (TF‑IDF 한계)
- TF‑IDF는 길이 정규화나 문서 길이 가중을 충분히 반영하지 못한다.
- BEIR 등 외부 벤치에서 BM25가 더 견고한 기준선으로 알려져 있음.

---

## 4. 해결/개선 조치 (적용 및 권고)

### 4.1 적용 완료: BM25 기반 Sparse Store 추가
- TF‑IDF 대신 BM25 기반 sparse retriever를 추가해 **기본 성능 상향**
- 파일: `era-smartfarm-rag/core/Services/Retrieval/Sparse.py` (`BM25Store`)

### 4.2 적용 완료: BEIR 표준 벤치마크 추가
- 도메인 추가학습 없이도 일반 벤치에서 성능 확인 가능
- 파일: `era-smartfarm-rag/benchmarking/experiments/beir_benchmark.py`
- 결과: `era-smartfarm-rag/output/beir/beir_summary.json`

### 4.3 운영 권고: 하이브리드 구성의 최소화
**단기 권고**: `rrf+dat+onto` 구성만 유지  
**비활성화 권고**: crop filter, semantic dedup, pathrag (기본 off)

근거: ablation에서 성능 하락/지연 폭증이 확인됨.

### 4.4 적용된 기본 설정 (서버 기본값)
- `SPARSE_METHOD=bm25`
- `HYBRID_USE_RRF=true`
- `HYBRID_USE_DAT=true`
- `HYBRID_USE_ONTOLOGY=true`
- `HYBRID_USE_PATHRAG=false`
- `HYBRID_USE_CROP_FILTER=false`
- `HYBRID_USE_DEDUP=false`

---

## 5. 개선 후 외부 벤치 성능 (BEIR)

BM25 + Dense + RRF 조합은 다수 벤치에서 sparse 대비 개선됨.

- 결과 파일: `era-smartfarm-rag/output/beir/beir_summary.json`

| Dataset | sparse MRR | rrf MRR | 비고 |
|---------|------------|---------|------|
| scifact | 0.703 | **0.770** | 개선 |
| nfcorpus | 0.559 | **0.645** | 개선 |
| arguana | 0.237 | **0.281** | 개선 |
| scidocs | 0.247 | **0.355** | 개선 (dense와 근접) |

---

## 6. 엣지( Jetson 64GB ) 제약 대응

- **도메인 추가학습 없이** pretrained embedding 사용 (예: `BAAI/bge-base-en-v1.5`)
- 연산량 높은 모듈(dedup/pathrag)은 **옵션화**하고, 기본 비활성 유지
- RRF는 랭크 기반이라 점수 스케일 문제에 강하고, 구현이 단순해 엣지 친화적

---

## 7. 재현 방법 (벤치 실행)

```bash
cd era-smartfarm-rag
.venv/bin/python benchmarking/experiments/beir_benchmark.py \
  --datasets scifact nfcorpus arguana scidocs \
  --embed-model BAAI/bge-base-en-v1.5 \
  --sparse bm25 \
  --retrieval-k 100
```

내부 벤치/ablation 결과 확인:

```bash
cat output/baseline_rrf_v2_baseline/baseline_summary.json
cat output/ablation_rrf/ablation_summary.json
```

---

## 8. 외부 참고자료 (MCP)

- Cormack et al., 2009, Reciprocal Rank Fusion  
  https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf
- Thakur et al., 2021, BEIR Benchmark  
  https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/65b9eea6e1cc6bb9f0cd2a47751a186f-Paper-round2.pdf
- Robertson & Zaragoza, 2009, BM25 Foundations  
  https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf

---

## 9. 결론

성능 저하의 핵심 원인은 **점수 융합 방식(min‑max + heuristic weights)**,  
**휴리스틱 필터(crop/dedup/pathrag)**로 확인되며,  
**BM25 + RRF 기반의 단순 하이브리드**로의 전환이 가장 안정적이다.  
Jetson 64GB 환경에서는 이 경량 구조가 성능/지연/안정성 측면에서 우선순위가 높다.
