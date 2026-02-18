# ERA-SmartFarm v2 Conformance Report

## Scope
- Included: `smartfarm-search`, `smartfarm-ingest`, `smartfarm-benchmarking`, `smartfarm-llm-inference`
- Excluded: `smartfarm-frontend`

## Stage Mapping
| Stage | Plan Requirement | Evidence |
|---|---|---|
| Stage 1 | Offline ingest + artifact export/import | `smartfarm-ingest/pipeline/public_ingest_runner.py`, `smartfarm-ingest/pipeline/artifact_export.py`, `smartfarm-ingest/pipeline/artifact_import.py` |
| Stage 2 | Online retrieval + overlay + kill-switch | `smartfarm-search/core/retrieval/service.py`, `smartfarm-search/core/retrieval/qdrant_client.py`, `smartfarm-search/core/Api/query.py` |
| Stage 3 | Benchmarking + ragas + ablation + edge metrics | `smartfarm-benchmarking/benchmarking/experiments/paper_eval.py`, `smartfarm-benchmarking/benchmarking/experiments/ablation.py`, `smartfarm-benchmarking/benchmarking/experiments/edge_profile.py` |

## Dataset Policy
- Registry: `smartfarm-benchmarking/benchmarking/datasets/registry.py`
- Commercial-only default: enabled
- HF token handling: runtime env only (`HF_TOKEN`)

## Remaining Risks
- Full E2E runtime validation depends on dependency installation and service availability.
- Judge self-host quality parity with API judge must be validated in target hardware.

## Sign-off Checklist
- [ ] Unit tests green
- [ ] Integration tests green
- [ ] Compose ingest/edge/eval smoke pass
- [ ] Benchmark artifacts reproducible
- [ ] Rollback procedure verified
