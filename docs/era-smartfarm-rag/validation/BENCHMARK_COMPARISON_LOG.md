# Benchmark Comparison Log (Wasabi QA v2)

**Date**: 2026-01-19  
**Corpus**: `dataset-pipeline/output/wasabi_en_ko_parallel.jsonl` (402 docs)  
**QA**: v2_baseline vs v2_improved (220 queries each)  
**Embedding**: `EMBED_MODEL_ID=minilm`

## Commands

```bash
# (권장) era-smartfarm-rag 루트에서 실행
cd era-smartfarm-rag

EMBED_MODEL_ID=minilm SPARSE_METHOD=bm25 python -m benchmarking.experiments.baseline_comparison \
  --corpus ../dataset-pipeline/output/wasabi_en_ko_parallel.jsonl \
  --qa-file ../dataset-pipeline/output/wasabi_qa_dataset_v2_baseline.jsonl \
  --output-dir output/baseline_rrf_v2_baseline

EMBED_MODEL_ID=minilm SPARSE_METHOD=bm25 python -m benchmarking.experiments.baseline_comparison \
  --corpus ../dataset-pipeline/output/wasabi_en_ko_parallel.jsonl \
  --qa-file ../dataset-pipeline/output/wasabi_qa_dataset_v2_improved.jsonl \
  --output-dir output/baseline_rrf_v2_improved
```

## Results (MRR)

| Method | v2_baseline | v2_improved | Δ (improved - baseline) |
|---|---:|---:|---:|
| dense_only | 0.2000 | 0.3106 | +0.1106 |
| sparse_only | 0.5674 | 0.5235 | -0.0439 |
| naive_hybrid | 0.1777 | 0.3330 | +0.1553 |
| rrf_hybrid | 0.3443 | 0.4515 | +0.1072 |
| proposed | 0.0383 | 0.1602 | +0.1220 |

## Notes

- 개선본은 Dense/Hybrid 계열을 유의하게 끌어올렸지만 Sparse-only는 소폭 하락.
- Proposed(HybridDAT)는 개선됐지만 여전히 Sparse 대비 낮음.
- 상세 결과는 아래 JSON을 참조.
  - `era-smartfarm-rag/output/baseline_rrf_v2_baseline/baseline_summary.json`
  - `era-smartfarm-rag/output/baseline_rrf_v2_improved/baseline_summary.json`
