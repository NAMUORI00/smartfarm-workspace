# 5. 실험 및 평가 (Experiments)

> **Note**: 이 문서는 논문의 실험 섹션입니다. 실제 실험 결과가 생성되면 수치를 업데이트해야 합니다.

---

## 5.1 Experimental Setup

### 5.1.1 Dataset and Scope

본 연구는 엣지 환경에서의 도메인 특화 RAG 시스템 설계를 검증하는 **파일럿 스터디(pilot study)**이다. 소규모 데이터셋의 통계적 한계를 인지하고, 시스템 아키텍처의 타당성 검증과 엣지 배포 가능성 확인에 초점을 맞춘다.

**실험 도메인:**

| 도메인 | 역할 | 데이터 소스 | 비고 |
|--------|------|-------------|------|
| **와사비 (Main)** | 주요 검증 | 시즈오카현 가이드라인 PDF, 농업 논문, 위키 | 서론 1.6절 근거 |
| **SeedBench (Aux)** | 일반화 검증 | 벼 육종 QA 2,264개 | ACL 2025 벤치마크 |

**QA 데이터셋 생성 (RAGEval 방법론 적용):**

RAGEval (Zhu et al., ACL 2025)의 시나리오 기반 QA 생성 방법론을 적용하여 평가 데이터셋을 구축하였다:

1. **핵심 구절 추출**: 코퍼스에서 도메인 관련 Key Points 자동 추출
2. **QA 쌍 생성**: LLM을 활용한 질문-답변 쌍 생성
3. **복잡도 분류**: Basic(단일 사실) / Intermediate(추론) / Advanced(다단계 참조)
4. **질의 유형 분류** (Know Your RAG, Cuconasu et al., COLING 2025):
   - Factoid: 단순 사실 확인 질의
   - Reasoning: 인과관계 추론 필요 질의
   - Multi-hop: 다중 문서 참조 필요 질의

**와사비 데이터셋 구성:**

| 항목 | 규모 | 비고 |
|------|------|------|
| 말뭉치 | 402개 청크 | PDF/위키 텍스트 추출 |
| QA 데이터셋 | 220개 쌍 | RAGEval 방법론 적용 |
| 카테고리 | 4개 | 재배기술, 환경관리, 병해충, 영양관리 |
| 복잡도 | 3단계 | Basic, Intermediate, Advanced |
| 질의 유형 | 3종 | Factoid, Reasoning, Multi-hop |

**통계적 한계:**

본 데이터셋은 BEIR (수천~수백만 문서)나 LegalBench-RAG (6,889 QA)에 비해 소규모이다. Card et al. (2020)의 분석에 따르면, N=220에서 80% 검정력으로 검출 가능한 최소 효과 크기(MDE)는 약 **4-5% MRR 차이**이다. 이보다 작은 개선은 통계적으로 유의하지 않을 수 있다. 비율 지표(Hit Rate, Precision)에는 Wilson score interval을, 연속 지표(MRR, NDCG)에는 표준편차를 함께 보고하여 소표본 불확실성을 명시한다.

### 5.1.2 Baselines

네 가지 검색 베이스라인을 비교 평가한다:

| 베이스라인 | 설명 | 특징 |
|------------|------|------|
| **Dense-only** | FAISS 임베딩 유사도 검색 | 의미적 유사성 |
| **BM25** | Sparse 키워드 검색 | 정확한 용어 매칭 |
| **RRF** | Reciprocal Rank Fusion (Dense+BM25) | 하이브리드 융합 |
| **LightRAG** | Dual-Level 그래프 검색 (Entity + Community) | 지식 그래프 기반 |

**LightRAG (Guo et al., EMNLP 2025) 제안 시스템 특징:**
- **Entity-Level**: 개별 엔티티 노드 기반 검색
- **Community-Level**: Leiden 알고리즘으로 클러스터링된 커뮤니티 요약 활용
- **Ego-Network Traversal**: 관련 엔티티의 이웃 노드까지 확장 탐색
- **도메인 적응**: 농업 도메인 온톨로지 연계 (작물명, 병해충, 환경요인 등)

> **구현 참고**: 베이스라인 수식 및 LightRAG 상세 알고리즘은 Section 3.3 참조. 모든 베이스라인은 공정한 비교를 위해 동일 임베딩 모델(MiniLM-L12-v2)로 자체 구현하였다.

### 5.1.3 Metrics and K Selection

**검색 성능 메트릭:**
- Precision@K, Recall@K: 상위 K개 결과의 정밀도/재현율
- MRR (Mean Reciprocal Rank): 첫 번째 정답 순위의 역수 평균
- NDCG@K: Normalized Discounted Cumulative Gain
- Hit Rate@K: 상위 K개 중 최소 1개 정답 포함 여부

**K=4 선택 근거:**

본 연구에서는 K=4를 주요 평가 기준으로 사용한다:

1. **프롬프트 제한**: 8GB RAM에서 최대 4개 문서만 프롬프트에 포함 가능
2. **응답 시간**: 문서 추가 시 생성 시간 증가, 실시간 응답 위해 제한
3. **품질 균형**: "lost in the middle" 문제 방지 (Liu et al., 2024)

> 표준 벤치마크 K=1, 5, 10 결과는 Appendix B 참조.

**엣지 성능 메트릭:**
- Cold Start Time, Query Latency (p50/p95/p99), Memory Usage, QPS

> **구현 상세**: 하이퍼파라미터(θ=0.85, 작물 보너스 등)는 Section 4.2 참조.

---

## 5.2 Results

### 5.2.1 Baseline Comparison (RQ1)

**RQ1**: 제안하는 LightRAG 기반 시스템이 기존 검색 방법 대비 얼마나 성능이 향상되는가?

**Table 1: Baseline Performance Comparison (N=220)**

| Method | P@4 | R@4 | MRR | NDCG@4 | Hit@4 |
|--------|-----|-----|-----|--------|-------|
| Dense-only | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| BM25 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| RRF | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| **LightRAG** | **[TBD]** | **[TBD]** | **[TBD]** | **[TBD]** | **[TBD]** |

*각 값은 mean ± std 형식. MDE ≈ 4-5%이므로 이보다 작은 차이는 통계적으로 유의하지 않을 수 있음.*

**분석:**

[실험 결과 생성 후 작성]

- Dense vs BM25: 의미적 vs 키워드 매칭 특성 비교
- RRF 한계: 단순 랭킹 융합으로 관계 정보 미활용
- LightRAG 개선: 그래프 기반 엔티티/커뮤니티 검색으로 맥락 풍부화

### 5.2.2 Ablation Study (RQ2)

**RQ2**: LightRAG의 각 컴포넌트가 성능 향상에 얼마나 기여하는가?

**Table 2: Ablation Results (N=220)**

| Configuration | Entity | Community | Graph Traverse | Domain Ontology | MRR | ΔMRR |
|---------------|--------|-----------|----------------|-----------------|-----|------|
| Dense-only (Base) | - | - | - | - | [TBD] | -- |
| +Entity | ✓ | - | - | - | [TBD] | +[TBD] |
| +Community | - | ✓ | - | - | [TBD] | +[TBD] |
| +Entity+Community | ✓ | ✓ | - | - | [TBD] | +[TBD] |
| +Full Graph | ✓ | ✓ | ✓ | - | [TBD] | +[TBD] |
| **LightRAG (Full)** | ✓ | ✓ | ✓ | ✓ | **[TBD]** | **+[TBD]** |

*Δ는 Base 대비 누적 개선. 컴포넌트 간 상호작용으로 개별 기여도 합이 전체와 불일치할 수 있음.*

**Key Findings:**

1. **Entity-Level 검색**: 명시적 엔티티(작물명, 병해충, 환경요인) 매칭으로 정확도 향상
2. **Community-Level 검색**: Leiden 클러스터링 기반 요약으로 광범위 질의에 효과적
3. **Graph Traverse**: Ego-network 탐색으로 관련 엔티티 확장 → 맥락 풍부화
4. **Domain Ontology**: 농업 도메인 온톨로지 연계로 동의어/상위개념 처리

**질의 유형별 컴포넌트 효과:**

| 질의 유형 | 효과적 컴포넌트 | 이유 |
|-----------|-----------------|------|
| Factoid | Entity-Level | 단일 엔티티 정확 매칭 |
| Reasoning | Community-Level | 개념 요약 활용 |
| Multi-hop | Graph Traverse | 다단계 관계 탐색 |

### 5.2.3 Domain Analysis (RQ3)

**RQ3**: 도메인 특화 기능들이 농업 도메인 질의에 효과적인가?

**Table 3: Performance by Category and Complexity**

| 분석 기준 | 구분 | N | MRR | NDCG@4 |
|-----------|------|---|-----|--------|
| **카테고리** | 재배기술 | [TBD] | [TBD] | [TBD] |
| | 환경관리 | [TBD] | [TBD] | [TBD] |
| | 병해충 | [TBD] | [TBD] | [TBD] |
| | 영양관리 | [TBD] | [TBD] | [TBD] |
| **복잡도** | Basic | [TBD] | [TBD] | [TBD] |
| | Intermediate | [TBD] | [TBD] | [TBD] |
| | Advanced | [TBD] | [TBD] | [TBD] |

**온톨로지 효과:**
- With ontology matching: MRR = [TBD] (N = [TBD])
- Without ontology matching: MRR = [TBD] (N = [TBD])
- Improvement: [TBD]%

### 5.2.4 Edge Performance (RQ4)

**RQ4**: 제안 시스템이 엣지 환경(8GB RAM)에서 실용적인 성능을 보이는가?

**Table 4: Edge Performance Metrics**

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Cold Start Time | [TBD] s | < 10s | [TBD] |
| Index Memory | [TBD] MB | < 1GB | [TBD] |
| Retrieval Latency (p50) | 3,423 ms | < 5s | ✅ |
| Retrieval Latency (p95) | 6,591 ms | < 10s | ✅ |
| Generation Latency (p50) | 2,485 ms | < 5s | ✅ |
| Generation Latency (p95) | 4,310 ms | < 8s | ✅ |
| **EtE Latency (p50)** | **6,359 ms** | < 10s | ✅ |
| **EtE Latency (p95)** | **10,095 ms** | < 15s | ✅ |
| **EtE Latency (p99)** | **10,499 ms** | < 20s | ✅ |
| Throughput (EtE) | 0.16 QPS | > 0.1 | ✅ |

*CPU 환경(Qwen3-Embedding-0.6B, Qwen3-0.6B) 기준. GPU 환경에서 2-5x 성능 향상 예상.*

**Memory Scaling:**

문서 수 증가에 따른 메모리 사용량은 선형적으로 증가하며, 400개 문서 기준 약 [TBD] KB/doc의 메모리 효율을 보인다.

### 5.2.5 RAG Quality Evaluation (RQ5)

**RQ5**: 제안 시스템의 생성 품질(Generation Quality)이 베이스라인 대비 우수한가?

전통적인 IR 메트릭(MRR, NDCG 등)은 검색 품질만 측정하며, 최종 답변의 품질은 평가하지 못한다. 본 연구에서는 RAGAS (Es et al., EACL 2024) 프레임워크를 활용하여 **Reference-free** 방식으로 생성 품질을 평가한다.

**Table 5: RAGAS Evaluation Results (N=220)**

| Method | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|--------|-------------|------------------|-------------------|----------------|
| Dense-only | [TBD] | [TBD] | [TBD] | [TBD] |
| BM25 | [TBD] | [TBD] | [TBD] | [TBD] |
| RRF | [TBD] | [TBD] | [TBD] | [TBD] |
| **LightRAG** | **[TBD]** | **[TBD]** | **[TBD]** | **[TBD]** |

*평가 LLM: Qwen3-0.6B (로컬), 임베딩: MiniLM-L12-v2. 각 메트릭은 0-1 범위, 높을수록 좋음.*

**RAGAS 메트릭 설명:**
- **Faithfulness**: 답변이 검색된 context에 근거하는가 (환각 억제 정도)
- **Answer Relevancy**: 답변이 질문에 적절히 대응하는가
- **Context Precision**: 검색된 문서들이 답변 생성에 유용한가
- **Context Recall**: 정답 생성에 필요한 정보가 context에 포함되었는가

**분석:**

[실험 결과 생성 후 작성]

- **Faithfulness**: LightRAG의 Entity-Level 검색이 정확한 근거 문서 제공 → 환각 감소 기대
- **Context Precision**: Community-Level 요약으로 관련 정보 집중 → 정밀도 향상 기대
- **Answer Relevancy**: Graph Traverse로 질문 의도에 맞는 관계 탐색 → 적합성 향상 기대

> **실행 방법**: `python -m benchmarking.experiments.ragas_eval --qa-file QA_PATH --output OUTPUT_PATH`

---

## 5.3 Discussion

### 5.3.1 Key Findings

1. **LightRAG 효과성**: Dual-Level(Entity+Community) 그래프 검색이 단순 하이브리드(RRF) 대비 [TBD]% MRR 개선
2. **도메인 특화**: 농업 온톨로지 연계로 환경/병해충 관련 질의에서 효과적
3. **엣지 실용성**: 경량 그래프 구조로 8GB RAM 환경에서 실시간 응답 가능

### 5.3.2 Limitations

본 연구는 다음 한계를 명시적으로 인정한다:

| 한계 | 설명 | 영향 |
|------|------|------|
| **L1. 소규모 데이터셋** | N=220, MDE ~4-5% | 미세 차이 검출 불가, 통계적 검정력 제한 |
| **L2. 단일 도메인** | 와사비 단일 작물 특화 | 다른 작물/도메인으로 일반화 검증 필요 |
| **L3. 합성 평가 데이터** | LLM 생성 QA, 전문가 검증 없음 | 실제 농가 질의 패턴과 차이 가능 |

> 베이스라인 자체 구현 한계 및 향후 연구 방향은 Section 6.2 참조.

### 5.3.3 Threats to Validity

| 유형 | 위협 | 완화 조치 |
|------|------|-----------|
| Internal | 베이스라인 구현 편향 | 동일 인프라/모델 사용, 코드 공개 |
| External | 단일 도메인 | "pilot study"로 범위 명시 |
| Construct | K=4 비표준 | 근거 명시, Appendix B에 K=1,5,10 제공 |
| Statistical | 소표본 | MDE 명시, 비율 지표에 Wilson CI, 연속 지표에 std 보고 |

---

## Appendix A: Reproducibility

### A.1 실험 재현 방법

```bash
cd era-smartfarm-rag

# 1. 전체 실험 실행
python -m benchmarking.experiments.run_all_experiments \
    --corpus ../dataset-pipeline/output/wasabi_en_ko_parallel.jsonl \
    --qa-file ../dataset-pipeline/output/wasabi_qa_dataset.jsonl \
    --output-dir output/experiments

# 2. 논문용 결과 생성
python -m benchmarking.reporters.PaperResultsReporter \
    --experiments-dir output/experiments \
    --output-dir output/paper
```

### A.2 출력 파일

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
    └── figure_data.json
```

---

## Appendix B: Additional K Values

표준 벤치마크와의 비교를 위한 추가 K 값 결과.

**Table B1: Precision/Recall at Various K**

| Method | P@1 | P@5 | P@10 | R@5 | R@10 |
|--------|-----|-----|------|-----|------|
| Dense-only | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| BM25 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| RRF | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| **LightRAG** | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

**Table B2: NDCG at Various K**

| Method | NDCG@1 | NDCG@5 | NDCG@10 |
|--------|--------|--------|---------|
| Dense-only | [TBD] | [TBD] | [TBD] |
| BM25 | [TBD] | [TBD] | [TBD] |
| RRF | [TBD] | [TBD] | [TBD] |
| **LightRAG** | [TBD] | [TBD] | [TBD] |

---

## References (Experiment Design)

**핵심 참조 논문 (동료평가 출판물):**

- **Es, S., et al. (2024).** RAGAS: Automated Evaluation of Retrieval Augmented Generation. *EACL 2024*.
- **Zhu, K., et al. (2025).** RAGEval: Scenario Specific RAG Evaluation Dataset Generation Framework. *ACL 2025*.
- **Cuconasu, F., et al. (2025).** Know Your RAG: Dataset Taxonomy and Generation Strategies for Evaluating RAG Systems. *COLING 2025*.
- **Ying, J., et al. (2025).** SeedBench: A Multi-task Benchmark for Evaluating Large Language Models in Seed Science. *ACL 2025*.
- **Guo, Z., et al. (2025).** LightRAG: Simple and Fast Retrieval-Augmented Generation. *EMNLP 2025 Findings*.

**통계 및 평가 방법론:**

- Card, D., Henderson, P., Khandelwal, U., & Jurafsky, D. (2020). With little power comes great responsibility. *EMNLP 2020*.
- Liu, N. F., et al. (2024). Lost in the middle: How language models use long contexts. *TACL*.


