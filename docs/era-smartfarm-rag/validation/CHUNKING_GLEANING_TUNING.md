# Chunking + Gleaning 튜닝 리포트 (EdgeKG v3.2)

본 문서는 EdgeKG v3.2에서 **Chunking(청킹)** 과 **Gleaning(구조화 추출 보완 패스)** 의 기본값을 “찍는” 방식이 아니라, **재현 가능한 실험 절차(sweep)** 로 선택하기 위한 실행 가이드를 제공한다.

## 1) 배경/목표

- Base KB(`base.sqlite`)와 Overlay KB(`overlay.sqlite`)가 동일한 형태의 검색/인덱싱 채널(Dense/Sparse/TriGraph/Tags/Graph)을 공유하려면 **청킹 단위의 일관성**이 중요하다.
- Overlay 업데이트는 로컬 LLM(llama.cpp)이 `CausalSchema`에 맞춰 facts(entities/relations)를 생성한다. 이때 **한 번 더 보완 추출(gleaning)** 을 허용하면 coverage가 늘 수 있으나, 비용/시간이 늘어날 수 있으므로 **효익/비용** 기준으로 선택해야 한다.

## 2) 튜닝 대상 파라미터(정의)

### 2.1 Chunking

- `CHUNK_METHOD`
  - `sentence_window`: 문장 분리 후 슬라이딩 윈도우(레거시)
  - `token`: 토큰 기반 고정 길이 청킹(“LightRAG 스타일” 정렬 목적)
- `CHUNK_TOKEN_SIZE`: 토큰 청킹 시 목표 토큰 수
- `CHUNK_TOKEN_OVERLAP`: 인접 청크 간 토큰 오버랩
- `CHUNK_TOKENIZER_MODEL`: `tiktoken` 모델명(미설치/미지원 시 fallback)

> 구현 위치: `smartfarm-search/core/Services/Ingest/Chunking.py`

### 2.2 Gleaning (KBUpdateExtractor)

- `KBUPDATE_MAX_GLEANINGS` ∈ {0,1,2}
  - 0: 1-pass 추출만 수행
  - 1/2: 기존 추출 결과를 프롬프트에 넣고 “누락분만 추가”를 요청하는 보완 패스를 수행
- 조기 종료(early stop): pass에서 신규 엔티티/관계 증가율이 5% 미만이면 중단
- 기록: `extractions.gleaning_passes_used` (overlay ingest DB)

> 구현 위치:  
> - 추출: `smartfarm-search/core/Services/Ingest/KBUpdateExtractor.py`  
> - 저장: `smartfarm-search/core/Services/Ingest/JobStore.py`

## 3) Baseline(실험 전 고정)

- **Baseline A (LightRAG-compatible)**  
  - `CHUNK_METHOD=token`, `CHUNK_TOKEN_SIZE=1200`, `CHUNK_TOKEN_OVERLAP=100`, `KBUPDATE_MAX_GLEANINGS=1`
- **Baseline B (small-chunk sanity)**  
  - `CHUNK_METHOD=token`, `CHUNK_TOKEN_SIZE=512`, `CHUNK_TOKEN_OVERLAP=64`, `KBUPDATE_MAX_GLEANINGS=1`

이 두 baseline은 sweep 결과가 “왜 바뀌었는지/안 바뀌었는지”를 논문에서 설명하기 위한 기준점으로 항상 포함한다.

## 4) Chunking sweep (retrieval-only)

### 4.1 목적

- `CHUNK_TOKEN_SIZE/OVERLAP`이 Dense/Sparse/TriGraph 융합 검색의 품질/지연/인덱스 크기에 미치는 영향 측정
- Generation을 제외하고 retrieval-only로 평가하여, LLM 생성/프롬프트/서빙 변수를 배제

### 4.2 그리드(권장 기본)

- `chunk_token_size ∈ {256, 512, 768, 1024, 1200, 1536}`
- `overlap_ratio ∈ {0%, 8%, 12%}` → overlap = round(size * ratio)

### 4.3 실행 방법

```bash
cd smartfarm-benchmarking

# 빠른 실행을 위해 embed 모델을 경량으로 교체 가능(예: minilm)
python -m benchmarking.experiments.chunking_sweep \
  --out output/chunking_sweep.json \
  --max-queries 300 \
  --embed-model-id minilm
```

### 4.4 선택 규칙(스크립트 내 고정)

1) `mean_recall@K(agxqa) + mean_recall@K(2wiki)` 최대 (`K`는 `--k`, 기본값 4)  
2) 동률이면 p95 latency(합) 최소  
3) 동률이면 index size(합) 최소  

> 참고: best가 256-token처럼 “너무 작은 값”으로 선택되면, 스크립트가 bootstrap CI(Recall@K 차이)를 추가로 계산하도록 설계되어 있다.

## 5) Gleaning sweep (구조화 추출)

### 5.1 목적

- `max_gleanings`가 facts coverage(entities/relations) 및 downstream(TagHash/CausalGraph)의 효익에 미치는 영향 측정

### 5.2 실행 방법(로컬 llama.cpp 필요)

```bash
cd smartfarm-benchmarking

# 예시 입력: 공개 코퍼스 JSONL (id/_id + text/content ...)
python -m benchmarking.experiments.gleaning_sweep \
  --input-jsonl ../smartfarm-search/data/agriqa/corpus.jsonl \
  --base-index-dir ../smartfarm-search/data/index \
  --out output/gleaning_sweep.json \
  --gleanings 0,1,2 \
  --chunk-method token \
  --chunk-token-size 1200 \
  --chunk-token-overlap 100
```

- `--with-downstream` 옵션을 켜면 TagHash/CausalGraph 히트율(hit@K)을 함께 계산한다. (시간/리소스 증가)

## 6) 운영 적용 방법(선택된 기본값 반영)

- 런타임/워커 공통 환경변수(예시):

```bash
export CHUNK_METHOD=token
export CHUNK_TOKEN_SIZE=1200
export CHUNK_TOKEN_OVERLAP=100
export CHUNK_TOKENIZER_MODEL=gpt-4o-mini
export KBUPDATE_MAX_GLEANINGS=1
```

## 7) 주의사항/재현성 메모

- `token` 청킹은 `tiktoken`이 설치되어 있으면 이를 사용하고, 없으면 문자 기반 heuristic로 fallback한다.  
  논문/보고서 재현성을 위해서는 실험 환경에서 `tiktoken` 사용을 권장한다.
- sweep는 “값 선택 절차”를 고정하기 위한 목적이며, 실제 최종 default는 **데이터셋/현장 질의 분포**에 따라 달라질 수 있다.

## 8) Code Traceability

아래 표는 “파라미터/선택 규칙”이 코드 어디에 반영되는지 추적하기 위한 기준이다.

| 항목 | 코드 위치 | 설명 | 참고 |
|---|---|---|---|
| token chunking 구현 | `smartfarm-search/core/Services/Ingest/Chunking.py` | overlap 클램프 + step>0 보장(무한루프 방지) | [4](https://arxiv.org/abs/2410.05779), [30](https://arxiv.org/abs/2307.03172), [31](https://github.com/openai/tiktoken) |
| ingest chunk 생성 | `smartfarm-search/core/Services/Ingest/ingest_worker.py::_chunk_rows` | 런타임 `CHUNK_METHOD`에 따라 sentence/token 경로 분기 | [4](https://arxiv.org/abs/2410.05779) |
| overlay trigraph 메타 정합성 | `smartfarm-search/core/Services/Ingest/ingest_worker.py::_effective_trigraph_chunk_params` | token 모드에서 meta `chunk_size/stride`를 토큰 기준으로 기록 | [6](https://arxiv.org/abs/2510.10114) |
| 설정 검증 | `smartfarm-search/core/Config/Settings.py::__post_init__` | chunk/gleaning 값 범위 제한 및 안전 기본값 | [4](https://arxiv.org/abs/2410.05779) |
| chunking sweep 선택 규칙 | `smartfarm-benchmarking/benchmarking/experiments/chunking_sweep.py` | `--k` 기준 Recall@K 우선 + 비용 tie-break | [29](https://arxiv.org/abs/2104.08663), [25](https://arxiv.org/abs/2406.04744) |
| gleaning sweep | `smartfarm-benchmarking/benchmarking/experiments/gleaning_sweep.py` | {0,1,2} pass에서 coverage/비용 비교 | [4](https://arxiv.org/abs/2410.05779), [17](https://arxiv.org/abs/2101.06426) |

## 9) Decision Log (템플릿)

최종 기본값을 결정할 때 아래 포맷으로 기록한다.

| Date | Decision | Evidence | Rejected Alternatives | Owner |
|---|---|---|---|---|
| YYYY-MM-DD | `CHUNK_TOKEN_SIZE=<n>`, `CHUNK_TOKEN_OVERLAP=<m>`, `KBUPDATE_MAX_GLEANINGS=<g>` 채택 | `chunking_sweep.json`, `gleaning_sweep.json`의 score/latency/drop-rate | 후보 A/B/C, 반려 사유(품질 미세 이득 대비 지연/비용 증가) | <name> |

기록 규칙:

- 품질 우선: `sum(mean_recall@K)` 최대값을 먼저 본다.
- 동률 처리: p95 latency → index size 순으로 tie-break 한다.
- gleaning은 facts coverage 증가율이 낮거나 drop-rate가 급증하면 더 작은 pass를 채택한다.

## 10) Reproducibility Checklist

아래 항목을 채운 실행 로그가 있어야 “재현 완료”로 간주한다.

- [ ] 코드 리비전 기록: workspace commit / submodule commit hash
- [ ] 실행 시각(UTC) 및 러너 환경(OS, Python, CUDA 유무)
- [ ] `seed` 값 (`PAPER_EVAL_SEED` / `--seed`)
- [ ] `k` 값 (`--k`, primary metric = `mean_recall@K`)
- [ ] embedding model (`EMBED_MODEL_ID`)
- [ ] chunking 파라미터 (`CHUNK_METHOD`, `CHUNK_TOKEN_SIZE`, `CHUNK_TOKEN_OVERLAP`, `CHUNK_TOKENIZER_MODEL`)
- [ ] gleaning 파라미터 (`KBUPDATE_MAX_GLEANINGS`)
- [ ] 실행 커맨드 원문 및 출력 산출물 경로
- [ ] references 번호 확인 (`docs/era-smartfarm-rag/paper/references.md`)

권장 실행 스냅샷:

```bash
# Chunking sweep (retrieval-only)
python -m benchmarking.experiments.chunking_sweep \
  --out output/chunking_sweep.json \
  --datasets agxqa,2wiki \
  --method ours_full \
  --k 4 \
  --max-queries 300 \
  --seed 42 \
  --embed-model-id minilm

# Gleaning sweep
python -m benchmarking.experiments.gleaning_sweep \
  --input-jsonl ../smartfarm-search/data/agriqa/corpus.jsonl \
  --base-index-dir ../smartfarm-search/data/index \
  --out output/gleaning_sweep.json \
  --gleanings 0,1,2 \
  --chunk-method token \
  --chunk-token-size 1200 \
  --chunk-token-overlap 100
```
