# Table 2: Ablation Study

**설명**: 각 컴포넌트 제거 시 성능 변화 측정 (컴포넌트별 기여도 분석)

**실험 실행**:
```bash
python -m benchmarking.experiments.RagExperimentRunner --config ablation
```

**결과 파일**: `output/experiments/ablation/results.json`

---

## Component Contribution Analysis

| Configuration | RRF | DAT | Ontology | PathRAG | MRR | ΔMRR |
|---------------|-----|-----|----------|---------|-----|------|
| Dense-only (Base) | - | - | - | - | [TBD] | -- |
| +RRF | ✓ | - | - | - | [TBD] | +[TBD] |
| +DAT | - | ✓ | - | - | [TBD] | +[TBD] |
| +RRF+DAT | ✓ | ✓ | - | - | [TBD] | +[TBD] |
| +Ontology | ✓ | ✓ | ✓ | - | [TBD] | +[TBD] |
| **HybridDAT (Full)** | ✓ | ✓ | ✓ | ✓ | **[TBD]** | **+[TBD]** |

---

**참고**: ΔMRR은 Base 대비 성능 변화. Crop Filter와 Dedup은 성능 저하로 제외됨.

---

## 컴포넌트 설명

- **RRF (Reciprocal Rank Fusion)**: Dense + Sparse 결과를 랭킹 기반 융합
- **DAT (Dynamic Alpha Tuning)**: 질의 특성에 따른 Dense/Sparse 가중치 동적 조정
- **Ontology Matching**: 농업 온톨로지 개념 매칭으로 도메인 관련성 부스팅
- **PathRAG**: 인과관계 그래프 기반 경로 탐색으로 Multi-hop 질의 대응

## Deprecated Components (성능 저하로 제외)

- ~~Crop Filter~~: 작물명 기반 필터링 - 오탐/미탐으로 인한 정확도 저하
- ~~Semantic Dedup~~: 임베딩 유사도 중복 제거 - 과도한 다양성 손실
