# SmartFarm Workspace

ERA-SmartFarm-RAG 통합 워크스페이스입니다.

## Rebuilt Modules
- `smartfarm-search`: Online retrieval API + private overlay
- `smartfarm-ingest`: Offline ingest (Unstructured + OpenAI-compatible extractor + Qdrant/FalkorDB writers)
- `smartfarm-benchmarking`: paper_eval / ablation / edge_profile / ragas_eval
- `smartfarm-llm-inference`: llama.cpp OpenAI-compatible inference
- `smartfarm-frontend`: 이번 범위 제외(UI 제외 정책)

## Core Docs
- Architecture plan: `docs/SYSTEM_ARCHITECTURE_PLAN.md`
- Env contract: `docs/ENV_CONTRACT.md`
- Graph schema seed: `schema.cypher`
- Release runbook: `docs/RELEASE_RUNBOOK.md`
- Conformance report: `docs/CONFORMANCE_REPORT.md`

## Compose Profiles
- Ingest: `docker-compose.ingest.yml`
- Edge runtime: `docker-compose.edge.yml`
- Eval batch: `docker-compose.eval.yml`
- Default alias: `docker-compose.yml` (edge)

## Bootstrap
```bash
bash scripts/dev/bootstrap.sh
python3 scripts/bootstrap_qdrant.py --host localhost --port 6333 --collection smartfarm_chunks
python3 scripts/bootstrap_falkordb_schema.py --host localhost --port 6379 --graph smartfarm --schema schema.cypher
```

## Env Contract
```bash
cp .env.example .env
python3 scripts/dev/lint_env.py --env-file .env
```

## Quick Start (Edge)
```bash
docker compose -f docker-compose.edge.yml up -d
curl -sS http://localhost:41177/health
```

## Quick Start (Offline Ingest)
```bash
docker compose -f docker-compose.ingest.yml up -d
```

## Dataset Policy (Benchmarking)
- 기본은 Hugging Face 공개 데이터셋 + 상업적 사용 가능 라이선스만 허용
- HF 인증은 런타임 환경변수(`HF_TOKEN`)만 사용

## Key API Endpoints
- `POST /query`
- `POST /private/ingest/memo`
- `POST /private/ingest/sensor`
- `POST /private/ingest/conversation`
- `POST /admin/purge/private`
- `POST /admin/purge/private-expired`
- `GET /health`

## Notes
- 본 워크스페이스는 API/ingest/benchmarking 우선이며 UI는 비포함입니다.
