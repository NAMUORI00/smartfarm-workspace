# Table 2: Ablation Study

**설명**: 각 컴포넌트 제거 시 성능 변화 측정 (컴포넌트별 기여도 분석)

**실험 실행**:
```bash
python -m benchmarking.experiments.ablation_study --corpus <corpus_path> --qa-file <qa_path>
```

**결과 파일**: `output/experiments/ablation/results.json`

---

## Component Contribution Analysis

| Configuration | MRR@4 | ΔMRR | Latency (ms) |
|---------------|-------|------|--------------|
| Full (All components) | TBD | -- | TBD |
| w/o Ontology Matching | TBD | TBD | TBD |
| w/o Crop Filtering | TBD | TBD | TBD |
| w/o Semantic Dedup | TBD | TBD | TBD |
| w/o PathRAG | TBD | TBD | TBD |
| w/o Dynamic Alpha | TBD | TBD | TBD |

---

**참고**: ΔMRR은 Full 대비 성능 변화. 음수는 성능 하락을 의미.

---

## 컴포넌트 설명

- **Ontology Matching**: 작물/환경/병해/영양소 온톨로지 기반 검색 부스팅
- **Crop Filtering**: 질의 내 작물명 기반 문서 필터링
- **Semantic Dedup**: 임베딩 유사도 기반 중복 제거 (θ=0.85)
- **PathRAG**: 인과관계 그래프 기반 검색
- **Dynamic Alpha**: 질의 특성에 따른 Dense/Sparse 가중치 동적 조정
