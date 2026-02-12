# SmartFarm Compose Profiles

- Edge runtime: `docker compose -f infra/compose/compose.edge.yml up -d`
- Offline ingest/eval: `docker compose -f infra/compose/compose.ingest.yml up -d`
- Sovereignty gate: `bash scripts/ci/check_sovereignty_gate.sh`
- Local E2E gate (mock llama + API): `smartfarm-search/.venv/bin/python scripts/ci/run_local_e2e.py --workspace . --py-bin smartfarm-search/.venv/bin/python`
- Judge preflight (OpenAI-compatible): `python scripts/ci/validate_openai_compat_judge.py`
- Edge defaults: `LLM_BACKEND=llama_cpp`, `PRIVATE_LLM_POLICY=local_only`, `ALLOW_DEV_REMOTE_PRIVATE=false`

Legacy root `docker-compose*.yml` files were removed in the rebuild.
