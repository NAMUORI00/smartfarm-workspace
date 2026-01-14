# Table 1: Baseline Comparison

**설명**: Dense-only, Sparse-only, Naive Hybrid 베이스라인과 제안 방법(HybridDAT) 비교

**실험 실행**:
```bash
python -m benchmarking.experiments.baseline_comparison --corpus <corpus_path> --qa-file <qa_path>
```

**결과 파일**: `output/experiments/baseline/results.json`

---

## Retrieval Performance Comparison with Baselines

| Method | MRR@4 | Recall@4 | nDCG@4 | Latency (p95) |
|--------|-------|----------|--------|---------------|
| Dense-only | TBD | TBD | TBD | TBD ms |
| Sparse-only | TBD | TBD | TBD | TBD ms |
| Naive Hybrid | TBD | TBD | TBD | TBD ms |
| **HybridDAT (Ours)** | **TBD** | **TBD** | **TBD** | TBD ms |

---

**실험 환경**: TBD

---

## 메트릭 설명

- **MRR@4**: Mean Reciprocal Rank (상위 4개 결과 기준)
- **Recall@4**: 상위 4개 결과에서 정답 포함 비율
- **nDCG@4**: Normalized Discounted Cumulative Gain
- **Latency (p95)**: 95 백분위 응답 시간
