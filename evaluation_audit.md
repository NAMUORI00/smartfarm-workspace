# Evaluation Audit (Post-Legacy-Cut)

- Audit date: 2026-02-20
- Scope:
  - Retrieval/Generation endpoint: `smartfarm-search/core/Api/query.py`
  - DAT runtime policy: `smartfarm-search/core/retrieval/fusion_profile.py`
  - Main evaluation runner: `smartfarm-benchmarking/benchmarking/experiments/run_main_eval.py`
  - Core eval runner: `smartfarm-benchmarking/benchmarking/experiments/paper_eval.py`

## 1) Legacy Removal Status

| Item | Status | Note |
|---|---|---|
| `experiments/run_eval_report.py` | Removed | Non-official duplicate path removed |
| `track_a.py`, `track_b.py` | Removed | Unified into `paper_eval`/`run_main_eval` |
| `run_suite.sh` | Removed | Multi-entry orchestration removed |
| Legacy proxy ablations (A5~A7) | Replaced | Converted to DAT-focused ablations |

## 2) Current Official Evaluation Path

1. `python3 -m benchmarking.experiments.run_main_eval`
2. `python3 -m benchmarking.experiments.ablation`
3. `python3 -m benchmarking.experiments.edge_profile`
4. `python3 -m benchmarking.experiments.generate_ieee_tables`

No backward compatibility wrapper is provided.

## 3) Methodological Soundness Checks

### A. Split protocol and leakage guard
- Outer split: 20% DAT-tune / 80% final-eval
- Inner split: 60/20/20 (train/val/dat_test) inside DAT-tune
- Runtime guard: overlap check across split groups by QID

Status: **PASS**

### B. DAT contribution isolation (A5~A7)
- A5 `fixed_weight_no_dat`
- A6 `dat_global_only_no_segment`
- A7 `no_evidence_adjustment`

Status: **PASS** (proxy naming removed)

### C. Runtime vs benchmark consistency
- Runtime uses DAT guardrails + quality gate + evidence-aware channel behavior
- Benchmark structural path supports DAT variants and evidence-adjustment ablation

Status: **PASS (with known simplification)**
- Benchmark graph channel is a lightweight simulation, not full runtime DB traversal

### D. Edge metrics readiness
- Metrics path exists for p50/p95/p99, TTFT, RSS, QPS
- If service is unavailable, result remains failure (`success_rounds=0`)

Status: **PARTIAL**
- Harness is ready; real edge-device execution is deferred by project policy

## 4) Remaining Risks

1. Real edge hardware numbers are pending, so system-level claims must be scoped as "execution-ready" not "field-validated".
2. Dataset-specific DAT sensitivity (especially HotpotQA-like multi-hop settings) requires further tuning.
3. Wasabi domain should be reported as domain-specific supplemental track, separate from public benchmark headline.

## 5) Sign-off Checklist

- [ ] Benchmarking unit tests green
- [ ] Main eval artifact regenerated after legacy cut
- [ ] Ablation artifact regenerated with new A5~A7 IDs
- [ ] IEEE table regeneration path verified
- [ ] Paper sections 4/5 synchronized with code paths
