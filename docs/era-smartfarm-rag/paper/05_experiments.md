# 5. 실험 및 평가 (Experiments)

> **Note**: 이 문서는 논문의 실험 섹션입니다. 실제 실험 결과가 생성되면 수치를 업데이트해야 합니다.
> 
> **AI 사용 명시**: 이 문서의 초안 작성에 AI 어시스턴트(Claude)가 사용되었습니다.

---

## 5.1 Experimental Setup

### 5.1.1 Dataset

본 연구에서는 와사비(Wasabi, *Eutrema japonicum*) 도메인에 특화된 한국어 RAG 데이터셋을 구축하였다.

**말뭉치 (Corpus)**:
- 총 402개 문서 청크
- 영어 원문: Wikipedia, Britannica, 학술 자료, 상업 재배 가이드
- 한국어 번역: LLM 기반 병렬 번역 (품질 검증 완료)
- 평균 청크 길이: ~500자

**평가 데이터셋 (QA Dataset)**:
- 220개 질의-응답 쌍 (목표)
- 카테고리: 재배기술, 환경관리, 병해충, 영양관리, 가공유통, 품종
- 복잡도 레벨: basic, intermediate, advanced
- LLM-as-a-Judge 방식으로 생성

### 5.1.2 Baselines

네 가지 검색 베이스라인을 비교 평가하였다:

1. **Dense-only**: FAISS 기반 임베딩 유사도 검색만 사용
2. **Sparse-only**: TF-IDF 기반 키워드 매칭만 사용
3. **Naive Hybrid**: Dense + Sparse 고정 가중치 결합 (α=0.5)
4. **Proposed (HybridDAT)**: 제안하는 동적 가중치 하이브리드 시스템
   - Dynamic Alpha Tuning (DAT)
   - 온톨로지 기반 매칭
   - 작물 필터링
   - 의미적 중복 제거

### 5.1.3 Evaluation Metrics

**검색 성능 (Retrieval)**:
- Precision@K (P@K): 상위 K개 결과의 정밀도
- Recall@K (R@K): 상위 K개 결과의 재현율
- MRR (Mean Reciprocal Rank): 첫 번째 정답 순위의 역수 평균
- NDCG@K: Normalized Discounted Cumulative Gain

**엣지 성능 (Edge Performance)**:
- Cold Start Time: 인덱스 로드 시간
- Query Latency: 쿼리 응답 시간 (p50, p95, p99)
- Memory Usage: 메모리 사용량
- QPS (Queries Per Second): 처리량

### 5.1.4 Implementation Details

- **임베딩 모델**: Qwen3-Embedding-0.6B
- **인덱스**: FAISS IndexFlatIP (내적 유사도)
- **하드웨어**: 8GB RAM 타겟 (Jetson/RTX 4060 Ti)
- **소프트웨어**: Python 3.10+, FastAPI, llama.cpp

---

## 5.2 Baseline Comparison (RQ1)

**연구 질문 1**: 제안하는 HybridDAT 시스템이 기존 검색 방법 대비 얼마나 성능이 향상되는가?

### 5.2.1 Overall Results

Table 1은 네 가지 베이스라인의 검색 성능을 비교한다.

| Method | P@4 | R@4 | MRR | NDCG@4 |
|--------|-----|-----|-----|--------|
| Dense-only | [TBD] | [TBD] | [TBD] | [TBD] |
| Sparse-only | [TBD] | [TBD] | [TBD] | [TBD] |
| Naive Hybrid | [TBD] | [TBD] | [TBD] | [TBD] |
| **Proposed** | **[TBD]** | **[TBD]** | **[TBD]** | **[TBD]** |

*Table 1: Retrieval performance comparison across baseline methods*

### 5.2.2 Analysis

[실험 결과 생성 후 작성]

- Dense-only vs Sparse-only 비교 분석
- Naive Hybrid의 한계점
- 제안 시스템의 개선 요인

---

## 5.3 Ablation Study (RQ2)

**연구 질문 2**: 제안 시스템의 각 컴포넌트가 성능 향상에 얼마나 기여하는가?

### 5.3.1 Component Analysis

Table 2는 각 컴포넌트를 순차적으로 추가했을 때의 성능 변화를 보여준다.

| Configuration | DAT | Crop | Dedup | MRR | ΔMRR |
|---------------|-----|------|-------|-----|------|
| Base (Naive) | - | - | - | [TBD] | -- |
| +DAT | ✓ | - | - | [TBD] | +[TBD] |
| +Ontology | ✓ | - | - | [TBD] | +[TBD] |
| +Crop Filter | ✓ | ✓ | - | [TBD] | +[TBD] |
| +Dedup | ✓ | - | ✓ | [TBD] | +[TBD] |
| **Full** | ✓ | ✓ | ✓ | **[TBD]** | **+[TBD]** |

*Table 2: Ablation study showing contribution of each component*

### 5.3.2 Key Findings

[실험 결과 생성 후 작성]

1. **Dynamic Alpha Tuning**: 질의 특성에 따른 동적 가중치 조정 효과
2. **Ontology Matching**: 도메인 개념 매칭의 기여도
3. **Crop Filtering**: 작물 특화 필터링 효과
4. **Semantic Deduplication**: 중복 제거를 통한 다양성 확보 효과

---

## 5.4 Domain-Specific Analysis (RQ3)

**연구 질문 3**: 도메인 특화 기능들이 실제로 농업 도메인 질의에 효과적인가?

### 5.4.1 Category-wise Performance

Table 3은 질의 카테고리별 성능을 비교한다.

| Category | N | MRR | NDCG@4 |
|----------|---|-----|--------|
| 재배기술 | [TBD] | [TBD] | [TBD] |
| 환경관리 | [TBD] | [TBD] | [TBD] |
| 병해충 | [TBD] | [TBD] | [TBD] |
| 영양관리 | [TBD] | [TBD] | [TBD] |

*Table 3: Performance by query category*

### 5.4.2 Complexity Analysis

| Complexity | N | MRR |
|------------|---|-----|
| Basic | [TBD] | [TBD] |
| Intermediate | [TBD] | [TBD] |
| Advanced | [TBD] | [TBD] |

*Table 4: Performance by query complexity level*

### 5.4.3 Ontology Effect

온톨로지 매칭이 활성화된 질의와 그렇지 않은 질의의 성능 비교:

- With ontology: MRR = [TBD] (N = [TBD])
- Without ontology: MRR = [TBD] (N = [TBD])
- Improvement: [TBD]%

---

## 5.5 Edge Performance (RQ4)

**연구 질문 4**: 제안 시스템이 엣지 환경(8GB RAM)에서 실용적인 성능을 보이는가?

### 5.5.1 Latency Analysis

Table 5는 엣지 환경에서의 성능 지표를 보여준다.

| Metric | Value |
|--------|-------|
| Cold Start Time | [TBD] s |
| Index Memory | [TBD] MB |
| Query Latency (p50) | [TBD] ms |
| Query Latency (p95) | [TBD] ms |
| Query Latency (p99) | [TBD] ms |
| Throughput | [TBD] QPS |

*Table 5: Edge device performance metrics*

### 5.5.2 Memory Scaling

Figure X는 문서 수에 따른 메모리 사용량 스케일링을 보여준다.

[그래프 데이터: figure_data.json의 memory_scaling 참조]

### 5.5.3 Practical Considerations

[실험 결과 생성 후 작성]

- 8GB RAM 환경에서의 실용성 분석
- 오프라인 운영 가능성
- 응답 시간 사용자 경험 분석

---

## 5.6 Discussion

### 5.6.1 Limitations

1. **데이터셋 규모**: 402개 청크, 220개 QA 쌍은 대규모 벤치마크 대비 소규모
2. **단일 도메인**: 와사비 도메인에 특화되어 일반화 검증 필요
3. **자동 평가**: LLM 생성 QA의 품질 한계

### 5.6.2 Future Work

1. 다양한 농작물 도메인으로 확장
2. PathRAG 그래프 검색 통합
3. 실제 농가 현장 배포 및 사용자 연구

---

## Appendix: Reproducibility

### A.1 실험 재현 방법

```bash
# 1. 전체 실험 실행
cd era-smartfarm-rag
python -m benchmarking.experiments.run_all_experiments \
    --corpus ../dataset-pipeline/output/wasabi_en_ko_parallel.jsonl \
    --qa-file ../dataset-pipeline/output/wasabi_qa_dataset.jsonl \
    --output-dir output/experiments

# 2. 논문용 결과 생성
python -m benchmarking.reporters.PaperResultsReporter \
    --experiments-dir output/experiments \
    --output-dir output/paper

# 3. 도메인 분석
python -m benchmarking.experiments.domain_analysis \
    --output-dir output/experiments/domain
```

### A.2 생성되는 파일

```
output/
├── experiments/
│   ├── baselines/baseline_summary.json
│   ├── ablation/ablation_summary.json
│   ├── edge/edge_benchmark_summary.json
│   └── domain/domain_analysis_summary.json
└── paper/
    ├── table1_baseline.tex
    ├── table2_ablation.tex
    ├── table3_edge.tex
    └── figure_data.json
```

---

## AI Usage Statement

This paper was prepared with assistance from AI language models (Claude by Anthropic) for:
- Code implementation and debugging
- Documentation drafting
- Experimental design suggestions

All scientific claims, experimental results, and conclusions were verified and validated by the human authors.
