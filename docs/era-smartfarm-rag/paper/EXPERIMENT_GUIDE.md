# 실험 실행 가이드

이 문서는 논문에 사용되는 실험을 직접 실행하고 결과를 수집하는 방법을 설명합니다.

## 사전 준비

### 1. 데이터셋 확인

```bash
# 코퍼스 파일 (400개 문서)
ls -lh dataset-pipeline/output/wasabi_en_ko_parallel.jsonl

# QA 데이터셋 (220개 질문)
ls -lh dataset-pipeline/output/wasabi_qa_dataset.jsonl
```

### 2. 인덱스 빌드

MiniLM 임베딩 모델로 인덱스를 빌드합니다 (약 2분 소요):

```bash
cd era-smartfarm-rag

# 환경변수 설정
export EMBED_MODEL_ID=minilm

# 인덱스 빌드
python scripts/tools/build_index_offline.py \
  --corpus ../dataset-pipeline/output/wasabi_en_ko_parallel.jsonl \
  --output data/index_minilm
```

### 3. 결과 디렉토리 생성

```bash
mkdir -p output/experiments/{baseline,ablation,edge}
```

---

## 실험 1: Baseline Comparison

### 목적
Dense-only, Sparse-only, Naive Hybrid 베이스라인과 제안 방법(HybridDAT) 비교

### 실행 명령

```bash
cd era-smartfarm-rag

python -m benchmarking.experiments.baseline_comparison \
  --corpus ../dataset-pipeline/output/wasabi_en_ko_parallel.jsonl \
  --qa-file ../dataset-pipeline/output/wasabi_qa_dataset.jsonl \
  --output-dir output/experiments/baseline \
  --n-runs 3
```

### 예상 소요 시간
- 약 10-15분 (3회 반복 실행)

### 출력 파일
- `output/experiments/baseline/results.json`

### 측정 메트릭
| 메트릭 | 설명 |
|--------|------|
| MRR@4 | Mean Reciprocal Rank (Top-4) |
| Recall@4 | 재현율 (Top-4) |
| nDCG@4 | Normalized Discounted Cumulative Gain |
| Latency (p95) | 95% 쿼리 레이턴시 (ms) |

---

## 실험 2: Ablation Study

### 목적
각 컴포넌트(온톨로지 매칭, 작물 필터링, 중복 제거, PathRAG, Dynamic Alpha) 제거 시 성능 변화 측정

### 실행 명령

```bash
cd era-smartfarm-rag

python -m benchmarking.experiments.ablation_study \
  --corpus ../dataset-pipeline/output/wasabi_en_ko_parallel.jsonl \
  --qa-file ../dataset-pipeline/output/wasabi_qa_dataset.jsonl \
  --output-dir output/experiments/ablation \
  --n-runs 3
```

### 예상 소요 시간
- 약 20-30분 (6개 설정 × 3회 반복)

### 출력 파일
- `output/experiments/ablation/results.json`

### 설정 조합
| 설정 | 설명 |
|------|------|
| Full | 모든 컴포넌트 활성화 |
| w/o Ontology | 온톨로지 매칭 비활성화 |
| w/o Crop Filter | 작물 필터링 비활성화 |
| w/o Dedup | 시맨틱 중복 제거 비활성화 |
| w/o PathRAG | 인과관계 그래프 검색 비활성화 |
| w/o Dynamic Alpha | 고정 가중치 사용 (0.5, 0.5, 0.0) |

---

## 실험 3: Edge Benchmark

### 목적
8GB RAM 엣지 환경에서의 실시간 성능 측정

### 실행 명령

```bash
cd era-smartfarm-rag

# 경량 임베딩 모델 사용
export EMBED_MODEL_ID=minilm

python -m benchmarking.experiments.edge_benchmark \
  --corpus ../dataset-pipeline/output/wasabi_en_ko_parallel.jsonl \
  --qa-file ../dataset-pipeline/output/wasabi_qa_dataset.jsonl \
  --output-dir output/experiments/edge \
  --n-runs 5 \
  --measure-memory
```

### 예상 소요 시간
- 약 15-20분 (5회 반복 + 메모리 측정)

### 출력 파일
- `output/experiments/edge/results.json`

### 측정 메트릭
| 메트릭 | 설명 |
|--------|------|
| Cold Start Time | 초기 모델 로드 시간 (초) |
| Index Build Time | 인덱스 빌드 시간 (초) |
| Latency p50/p95/p99 | 쿼리 레이턴시 분포 (ms) |
| QPS | 초당 처리 쿼리 수 |
| Peak RAM | 최대 메모리 사용량 (GB) |
| Avg RAM | 평균 메모리 사용량 (GB) |

---

## 결과 수집 및 테이블 생성

### LaTeX 테이블 생성

실험 결과 JSON 파일에서 논문용 LaTeX 테이블을 자동 생성합니다:

```bash
cd era-smartfarm-rag

python scripts/analysis/collect_paper_results.py \
  --results-dir output/experiments \
  --output-dir ../docs/paper/tables
```

### 출력 파일
- `docs/paper/tables/table1_baseline.tex` - Baseline Comparison
- `docs/paper/tables/table2_ablation.tex` - Ablation Study
- `docs/paper/tables/table3_edge.tex` - Edge Performance

### LaTeX 문서에서 사용

```latex
% main.tex
\input{tables/table1_baseline}
\input{tables/table2_ablation}
\input{tables/table3_edge}
```

---

## 스모크 테스트 (빠른 검증)

전체 실험 대신 빠른 검증이 필요할 때:

```bash
cd era-smartfarm-rag

# 벤치마크 스모크 테스트 (약 1분)
python tests/test_benchmark_smoke.py
```

---

## 문제 해결

### 메모리 부족 오류

```bash
# 경량 임베딩 모델 사용
export EMBED_MODEL_ID=minilm

# 또는 배치 크기 줄이기
python -m benchmarking.experiments.baseline_comparison --batch-size 8 ...
```

### 인덱스 없음 오류

```bash
# 인덱스 빌드 먼저 실행
python scripts/tools/build_index_offline.py --corpus ... --output data/index_minilm
```

### 데이터셋 없음 오류

```bash
# dataset-pipeline에서 데이터셋 생성
cd ../dataset-pipeline
python src/pipeline.py --config config/settings.yaml
```

---

## 결과 예시 (기대값)

실험 완료 후 예상되는 결과 범위:

### Table 1: Baseline Comparison
| Method | MRR@4 | Recall@4 |
|--------|-------|----------|
| Dense-only | 0.45-0.55 | 0.50-0.60 |
| Sparse-only | 0.35-0.45 | 0.40-0.50 |
| Naive Hybrid | 0.50-0.60 | 0.55-0.65 |
| **HybridDAT** | **0.60-0.70** | **0.65-0.75** |

### Table 3: Edge Performance
| Metric | Expected Range |
|--------|---------------|
| Query Latency p95 | 150-300 ms |
| Peak RAM | 0.8-1.2 GB |
| QPS | 3-8 QPS |

---

## 참고 사항

1. **실험 재현성**: 동일 환경에서 3-5회 반복 실행하여 평균/표준편차 계산
2. **환경 정보**: 결과 JSON에 CPU, RAM, GPU 정보가 자동 기록됨
3. **시드 고정**: 재현성을 위해 랜덤 시드가 고정됨 (기본값: 42)
