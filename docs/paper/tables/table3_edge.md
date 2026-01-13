# Table 3: Edge Deployment Performance

**설명**: 8GB RAM 엣지 환경에서의 실시간 성능 측정

**실험 실행**:
```bash
python -m benchmarking.experiments.edge_benchmark --corpus <corpus_path> --qa-file <qa_path> --measure-memory
```

**결과 파일**: `output/experiments/edge/results.json`

---

## Edge Deployment Performance (8GB RAM)

### Startup

| Metric | Value |
|--------|-------|
| Cold Start Time | TBD s |
| Index Build Time | TBD s |

### Query Latency

| Percentile | Value |
|------------|-------|
| p50 | TBD ms |
| p95 | TBD ms |
| p99 | TBD ms |

### Throughput

| Metric | Value |
|--------|-------|
| Queries Per Second | TBD QPS |

### Memory Usage

| Metric | Value |
|--------|-------|
| Peak RAM | TBD GB |
| Average RAM | TBD GB |

---

**실험 환경**: TBD

---

## 측정 항목 설명

- **Cold Start Time**: 서버 시작부터 첫 질의 응답 가능까지 시간
- **Index Build Time**: 전체 코퍼스 인덱싱 소요 시간
- **p50/p95/p99**: 레이턴시 백분위수 (50%, 95%, 99%)
- **QPS**: 초당 처리 가능한 질의 수
- **Peak/Average RAM**: 최대/평균 메모리 사용량
