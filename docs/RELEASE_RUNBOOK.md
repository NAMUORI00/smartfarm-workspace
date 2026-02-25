# ERA-SmartFarm v2 Release Runbook

## 1. Preconditions
- Python runtime: `python3.12`
- Docker/Compose available
- `.env` values prepared (no secrets committed)
- Optional HF auth: `HF_TOKEN`

## 2. Bootstrap
```bash
bash scripts/dev/bootstrap.sh
```

## 3. Storage Bootstrap
```bash
python3 scripts/bootstrap_qdrant.py --host localhost --port 6333 --collection smartfarm_chunks
python3 scripts/bootstrap_falkordb_schema.py --host localhost --port 6379 --graph smartfarm --schema schema.cypher
```

## 4. Runtime Profiles
### Edge
```bash
docker compose -f docker-compose.yml up -d
curl -sS http://localhost:41177/health
```

### Ingest
```bash
docker compose -f docker-compose.ingest.yml up -d
```

### Eval
```bash
docker compose -f docker-compose.eval.yml up -d
```

## 5. Gate Checklist
- Gate 1: Qdrant/FalkorDB bootstrap success
- Gate 2: public ingest + artifact export/import success
- Gate 3: `/query` + private ingest/purge + retrieval kill-switch regression pass
- Gate 4: paper_eval + ablation(A1~A7) + edge_profile(TTFT/RSS/QPS) artifacts generated

## 6. Rollback
- Restore Qdrant snapshot
- Restore FalkorDB dump
- Revert env/model version bundle to last passing tag (`v2-phaseN-passed`)

## 7. Incident Switches
- `JUDGE_RUNTIME=api|self_host`

## 8. Artifact Policy
- `smartfarm-benchmarking/output/` 산출물은 재현 실행으로 생성하며 저장소에는 커밋하지 않는다.
