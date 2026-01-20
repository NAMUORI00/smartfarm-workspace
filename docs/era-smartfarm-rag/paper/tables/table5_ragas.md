# Table 5: RAGAS Evaluation Results

> **연구 질문 RQ5**: 제안 시스템의 생성 품질(Generation Quality)이 베이스라인 대비 우수한가?

## 메트릭 설명

| 메트릭 | Ground Truth | 평가 대상 | 설명 |
|--------|-------------|----------|------|
| **Faithfulness** | 불필요 | Generation | 답변이 검색된 context에 근거하는가 (환각 억제 정도) |
| **Answer Relevancy** | 불필요 | Generation | 답변이 질문에 적절히 대응하는가 |
| **Context Precision** | 불필요 | Retrieval | 검색된 문서들이 답변 생성에 유용한가 |
| **Context Recall** | 선택적 | Retrieval | 정답 생성에 필요한 정보가 context에 포함되었는가 |

## 평가 결과 (N=220)

| Method | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|--------|-------------|------------------|-------------------|----------------|
| Dense-only | [TBD] | [TBD] | [TBD] | [TBD] |
| Sparse-only | [TBD] | [TBD] | [TBD] | [TBD] |
| Naive Hybrid | [TBD] | [TBD] | [TBD] | [TBD] |
| **Proposed** | **[TBD]** | **[TBD]** | **[TBD]** | **[TBD]** |

*평가 LLM: Qwen3-0.6B (로컬), 임베딩: MiniLM-L12-v2. 각 메트릭은 0-1 범위, 높을수록 좋음.*

## 실행 방법

```bash
cd era-smartfarm-rag

# RAGAS 평가 실행
python -m benchmarking.experiments.ragas_eval \
    --qa-file ../dataset-pipeline/output/wasabi_qa_dataset.jsonl \
    --output output/ragas_result.json \
    --llm-model qwen3-0.6b \
    --emb-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# 옵션
# --metric faithfulness --metric answer_relevancy  # 특정 메트릭만 평가
# --limit 50  # 샘플 수 제한
# --no-progress  # 진행률 표시 비활성화
```

## 참고문헌

- [38] Es, S., et al. (2024). "RAGAS: Automated Evaluation of Retrieval Augmented Generation." *EACL 2024*.
- [39] Saad-Falcon, J., et al. (2024). "ARES: An Automated Evaluation Framework for RAG Systems." *NAACL 2024*.

## 분석 예정

[실험 결과 생성 후 작성]

- **Faithfulness**: 제안 시스템의 온톨로지 매칭이 관련 문서 검색 → 환각 감소 기대
- **Context Precision**: 작물 필터링으로 무관 문서 배제 → 정밀도 향상 기대
- **Answer Relevancy**: PathRAG-lite의 인과관계 탐색 → 질문 의도에 맞는 답변 기대

---

*이 테이블은 RAGAS 프레임워크(EACL 2024)를 사용하여 Reference-free 방식으로 생성됨.*
