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
- Paper eval: `python3 -m benchmarking.experiments.paper_eval --dataset agxqa,2wiki,hotpotqa --method ours_structural --out output/paper_eval.json`
- Ablation: `python3 -m benchmarking.experiments.ablation --dataset agxqa --out output/ablation_results.json` (A1~A7)
- Edge profile: `python3 -m benchmarking.experiments.edge_profile --base-url http://localhost:41177`

### smartfarm-llm-inference
- Run: `docker compose up -d`
- Health: `curl -sS http://localhost:45857/health`

## Compose Profiles
- `docker compose -f docker-compose.ingest.yml up -d`
- `docker compose -f docker-compose.edge.yml up -d`
- `docker compose -f docker-compose.eval.yml up -d`

## Phase Gates
- Gate 1: Qdrant/FalkorDB bootstrap + `/health` ready
- Gate 2: public ingest 50+ docs + artifact manifest generated
- Gate 3: `/query` + private ingest/purge API E2E pass + retrieval kill-switch 검증
- Gate 4: paper_eval + ablation(A1~A7) + edge profile(TTFT/RSS/QPS 포함) outputs generated
