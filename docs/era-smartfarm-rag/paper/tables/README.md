# Paper Tables

논문 실험 결과 테이블 모음

## 파일 목록

| 파일 | 내용 | 관련 실험 |
|------|------|----------|
| `table1_baseline.md` | 베이스라인 비교 | `baseline_comparison` |
| `table2_ablation.md` | 제거 실험 (컴포넌트 기여도) | `ablation_study` |
| `table3_edge.md` | 엣지 배포 성능 | `edge_benchmark` |

---

## 사용법

### 1. 실험 실행

```bash
# 베이스라인 비교
python -m benchmarking.experiments.baseline_comparison \
    --corpus data/corpus \
    --qa-file data/qa.json

# 제거 실험
python -m benchmarking.experiments.ablation_study \
    --corpus data/corpus \
    --qa-file data/qa.json

# 엣지 벤치마크
python -m benchmarking.experiments.edge_benchmark \
    --corpus data/corpus \
    --qa-file data/qa.json \
    --measure-memory
```

### 2. 테이블 자동 생성

실험 결과 JSON → 마크다운 테이블 자동 생성:

```bash
python scripts/analysis/collect_paper_results.py \
    --results-dir output/experiments \
    --output-dir docs/paper/tables
```

### 3. 결과 확인

VSCode에서 마크다운 미리보기로 테이블 확인:
- `Ctrl+Shift+V` (Windows/Linux)
- `Cmd+Shift+V` (macOS)

---

## 결과 파일 구조

```
output/experiments/
├── baseline/
│   └── results.json      # Table 1 데이터
├── ablation/
│   └── results.json      # Table 2 데이터
└── edge/
    └── results.json      # Table 3 데이터
```

---

## TBD 값 설명

테이블의 `TBD` 값은 실험 미실행 상태를 의미합니다.

실험 실행 후 `collect_paper_results.py`를 재실행하면 실제 값으로 대체됩니다.
