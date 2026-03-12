# Jetson One-Command RQ3 Bundle

This bundle is intended for a Jetson Orin-class Linux host. The Jetson path is
local-only by default: prebuilt indices and fusion profiles are shipped from
the development environment, and the Jetson host is used only for service
startup, health checks, and RQ3 measurement.

## Bundle Contents
- `deploy/jetson/docker-compose.jetson.yml`
- `deploy/jetson/.env.jetson.example`
- `deploy/jetson/run_rq3.sh`
- `deploy/jetson/collect_results.sh`
- `deploy/jetson/rq3_queries.txt`
- `smartfarm-search/`
- `smartfarm-benchmarking/`
- `smartfarm-llm-inference/models/Qwen3-4B-Q4_K_M.gguf`
- `data/index/qdrant/`
- `data/index/falkordb/`
- `data/artifacts/fusion_weights.runtime.json`
- `data/artifacts/fusion_profile_meta.runtime.json`

## Build the Bundle on the Development Machine
```bash
python3 deploy/jetson/build_bundle.py --out-dir output/jetson_bundle
```

## Upload to Jetson
Copy the generated `output/jetson_bundle/` directory to the Jetson host. The
bundle layout is preserved, so the Jetson host only needs the unpacked bundle.

## Prepare the Environment File
```bash
cp deploy/jetson/.env.jetson.example deploy/jetson/.env.jetson
```

Edit only the values that differ on the target Jetson host.

## Run RQ3
```bash
bash deploy/jetson/run_rq3.sh
```

The script performs:
1. artifact existence checks,
2. `docker compose` startup for `qdrant`, `falkordb`, `llama`, and `api`,
3. host-side health checks,
4. `benchmarking.experiments.edge_profile` execution,
5. result collection and log export.

## Outputs
- `output/jetson_rq3/edge_profile.json`
- `output/jetson_rq3/edge_profile_summary.md`
- `output/jetson_rq3/compose_ps.txt`
- `output/jetson_rq3/compose_resolved.yml`
- `output/jetson_rq3/logs/*.log`

## Notes
- The bundle is designed for **measurement**, not for on-device index creation.
- The query set is fixed by `deploy/jetson/rq3_queries.txt` unless
  `RQ3_QUERY_FILE` is overridden.
- Memory values in the compose file are recorded as **budget targets** for
  reporting. They are not treated as hard enforcement guarantees.
