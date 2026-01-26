# Table 3: Edge Deployment Performance

**설명**: 8GB RAM 엣지 환경에서의 실시간 성능 측정

**실험 실행**:
```bash
# Retrieval only
python -m benchmarking.experiments.edge_benchmark --corpus <corpus_path> --qa-file <qa_path> --measure-memory

# End-to-End (Retrieval + LLM Generation)
python -m benchmarking.experiments.ete_latency_benchmark --corpus <corpus_path> --qa-file <qa_path> --n-samples 30
```

**결과 파일**:
- `output/experiments/edge/results.json`
- `output/experiments/ete/ete_latency_results.json`

---

## Edge Deployment Performance (8GB RAM)

### Startup

| Metric | Value |
|--------|-------|
| Cold Start Time | TBD s |
| Index Build Time | TBD s |

### Retrieval-only Latency

| Percentile | Value |
|------------|-------|
| p50 | TBD ms |
| p95 | TBD ms |
| p99 | TBD ms |

### End-to-End Latency (Retrieval + LLM Generation)

| Metric | Retrieval | Generation | **EtE Total** | Target | Status |
|--------|-----------|------------|---------------|--------|--------|
| p50 | 3,423 ms | 2,485 ms | **6,359 ms** | < 10s | ✅ |
| p95 | 6,591 ms | 4,310 ms | **10,095 ms** | < 15s | ✅ |
| p99 | 6,784 ms | 4,647 ms | **10,499 ms** | < 20s | ✅ |

> **Note**: CPU 환경(Qwen3-Embedding-0.6B, Qwen3-0.6B) 기준 측정. GPU 환경에서 2-5x 성능 향상 예상.

### Throughput

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Queries Per Second (EtE) | 0.16 QPS | > 0.1 | ✅ |

### Memory Usage

| Metric | Value |
|--------|-------|
| Peak RAM | TBD GB |
| Average RAM | TBD GB |

---

**실험 환경**:
- CPU: Intel/AMD (CPU-only mode)
- Embedding Model: Qwen/Qwen3-Embedding-0.6B (1024d)
- LLM: Qwen3-0.6B via llama.cpp
- Corpus: 402 documents
- QA Samples: 30 queries

---

## 측정 항목 설명

- **Cold Start Time**: 서버 시작부터 첫 질의 응답 가능까지 시간
- **Index Build Time**: 전체 코퍼스 인덱싱 소요 시간
- **Retrieval Latency**: 검색(임베딩 + FAISS + BM25 hybrid) 레이턴시
- **Generation Latency**: LLM 추론 레이턴시 (llama.cpp)
- **EtE Latency**: End-to-End 레이턴시 (검색 + LLM 생성)
- **p50/p95/p99**: 레이턴시 백분위수 (50%, 95%, 99%)
- **QPS**: 초당 처리 가능한 질의 수
- **Peak/Average RAM**: 최대/평균 메모리 사용량
