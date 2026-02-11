# SmartFarm v2 Compose Profiles

- Edge runtime: `docker compose -f infra/compose/compose.edge.yml up -d`
- Offline ingest/eval: `docker compose -f infra/compose/compose.ingest.yml up -d`
- Sovereignty gate: `bash scripts/ci/check_sovereignty_gate.sh`
- Edge defaults: `LLM_BACKEND=llama_cpp`, `PRIVATE_LLM_POLICY=local_only`, `ALLOW_DEV_REMOTE_PRIVATE=false`

Legacy root `docker-compose*.yml` files were removed in the v2 rebuild.
