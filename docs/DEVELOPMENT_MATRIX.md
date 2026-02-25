# Development Execution Matrix

## Module Commands

### Workspace Bootstrap
- `bash scripts/dev/bootstrap.sh`

### smartfarm-search
- Install: `pip install -r requirements.txt`
- Run: `python3 -m uvicorn core.main:app --host 0.0.0.0 --port 41177 --reload`
- Health: `curl -sS http://localhost:41177/health`

### smartfarm-ingest
- Install: `pip install -r requirements.txt`
- Public ingest: `python3 -m pipeline.public_ingest_runner --input-dir ./data/public_docs --qdrant-host localhost --qdrant-port 6333 --falkor-host localhost --falkor-port 6379`
- Artifact export: `python3 -m pipeline.artifact_export --qdrant-dir ../data/index/qdrant --falkordb-dir ../data/index/falkordb --output-dir ../data/index/export`
- Artifact import: `python3 -m pipeline.artifact_import --manifest ../data/index/export/artifact_manifest.json --qdrant-dir ../data/index/qdrant --falkordb-dir ../data/index/falkordb`

### smartfarm-benchmarking
- Install: `pip install -r requirements.txt`
- Paper eval: `python3 -m benchmarking.experiments.paper_eval --dataset agxqa,2wiki,hotpotqa --method no_rag,bm25_only,dense_only,rrf,graph_only,lightrag,ours_structural --out output/paper_eval.json`
- Main comparison: `python3 -m benchmarking.experiments.run_main_eval --dataset agxqa,2wiki,hotpotqa --method no_rag,bm25_only,dense_only,rrf,graph_only,lightrag,ours_structural --seeds 42,52,62 --max-queries 40 --with-ragas --answer-mode llm_generated --out-json output/paper_eval_main.json --out-csv output/paper_eval_main.csv --comparison-out output/comparison_report.json`
- DAT export: `python3 -m benchmarking.experiments.export_fusion_profile --paper-main output/paper_eval_main.json --comparison-report output/comparison_report.json --weights-out ../data/artifacts/fusion_weights.runtime.json --meta-out ../data/artifacts/fusion_profile_meta.runtime.json`
- Repro protocol: `python3 -m benchmarking.experiments.ragas_eval --protocol benchmarking/configs/paper_protocol_v2.json`
- RAGAS protocol sample: `python3 -m benchmarking.experiments.ragas_eval --protocol benchmarking/configs/paper_protocol_v2.json`
- RAGAS smoke: `python3 -m benchmarking.experiments.ragas_eval --stage smoke --dataset agxqa --method no_rag,bm25_only,dense_only,rrf,ours_structural --seeds 42 --max-queries 8 --ragas-max-queries 8`
- RAGAS mini: `python3 -m benchmarking.experiments.ragas_eval --dataset agxqa,2wiki,hotpotqa --method no_rag,dense_only,rrf,lightrag,ours_structural --seeds 42,52 --max-queries 20 --ragas-max-queries 20`
- Ablation: `python3 -m benchmarking.experiments.ablation --dataset agxqa --max-queries 200 --out output/ablation_results.json --summary-out output/ablation_summary.md` (A0~A9)
- Edge profile: `python3 -m benchmarking.experiments.edge_profile --base-url http://localhost:41177 --rounds 50 --out-json output/edge_profile.json --out-md output/edge_profile_summary.md`
- Query-type adaptive analysis: `python3 -m benchmarking.experiments.query_type_analysis --dataset agxqa --max-queries 200 --seed 42 --method ours_structural,no_evidence_adjustment,dat_no_guardrails,dat_no_quality_gate --out-json output/query_type_analysis.json --out-csv output/query_type_analysis.csv --out-md output/query_type_analysis.md`

### smartfarm-llm-inference
- Run: `docker compose up -d`
- Health: `curl -sS http://localhost:45857/health`

## Compose Profiles
- `docker compose -f docker-compose.ingest.yml up -d`
- `docker compose -f docker-compose.yml up -d`
- `docker compose -f docker-compose.eval.yml up -d`

## Phase Gates
- Gate 1: Qdrant/FalkorDB bootstrap + `/health` ready
- Gate 2: public ingest 50+ docs + artifact manifest generated
- Gate 3: `/query` + private ingest/purge API E2E pass + retrieval kill-switch 검증
- Gate 4: paper_eval + ablation(A1~A7) + edge profile(TTFT/RSS/QPS 포함) outputs generated
