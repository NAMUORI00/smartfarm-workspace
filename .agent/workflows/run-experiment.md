---
description: Full E2E experiment workflow — Ingest → Benchmark → Profile Export
---

# Full Experiment Workflow

> **Pre-requisite**: Run `python preflight_check.py` from workspace root to verify all modules load cleanly.

## Phase 0: Infrastructure

// turbo
1. Start Docker services (Qdrant + FalkorDB only):
```pwsh
docker compose up -d qdrant falkordb
```

// turbo
2. Verify services are healthy:
```pwsh
docker compose ps
```

## Phase 1: Offline Ingest (KB Build + Ontology Auto-Generation)

3. Run public ingest pipeline (builds Qdrant vectors + FalkorDB graph + domain_ontology.json):
```pwsh
cd smartfarm-ingest
python -m pipeline.public_ingest_runner --input-dir ../data/sample_docs --qdrant-host localhost --qdrant-port 6333 --falkor-host localhost --falkor-port 6379
```

Expected output:
- `[ontology-build] generated domain_ontology.json (entities=N relations=N keywords_ko=N keywords_en=N)`
- `[public-ingest] done chunks=N entities=N relations=N`

## Phase 2: Artifact Export

// turbo
4. Export artifact manifest for edge sync:
```pwsh
cd smartfarm-ingest
python -m pipeline.artifact_export --qdrant-dir ../data/index/qdrant --falkordb-dir ../data/index/falkordb --output-dir ../data/index/export
```

## Phase 3: Benchmark — Main Evaluation

### Option A: Fast (개발용, ~15분)
5a. Run fast-profile benchmark:
```pwsh
cd smartfarm-benchmarking
python -m benchmarking.experiments.run_main_eval --fast-profile --allow-noncommercial
```

### Option B: Nightly (논문급, ~2-4시간)
5b. Run nightly-profile benchmark:
```pwsh
cd smartfarm-benchmarking
python -m benchmarking.experiments.run_main_eval --nightly-profile --allow-noncommercial --with-ragas
```

Output files:
- `output/paper_eval_main.json` — Full results
- `output/paper_eval_main.csv` — Table-ready CSV
- `output/comparison_report.json` — Statistical test report

## Phase 4: DAT Runtime Profile Export

// turbo
6. Export fusion profile for runtime:
```pwsh
cd smartfarm-benchmarking
python -m benchmarking.experiments.export_fusion_profile --paper-main output/paper_eval_main.json --meta-out ../data/artifacts/fusion_profile_meta.runtime.json
```

## Phase 5: Runtime Verification

// turbo
7. Start full stack (API + LLM + DBs):
```pwsh
docker compose up -d
```

8. Test query:
```pwsh
curl http://localhost:41177/query -X POST -H "Content-Type: application/json" -d '{"query": "토마토 잿빛곰팡이병 원인"}'
```

## Phase 6: Teardown

// turbo
9. Stop all services:
```pwsh
docker compose down
```
