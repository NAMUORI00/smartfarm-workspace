# SmartFarm Workspace Documentation

스마트팜 프로젝트 통합 문서 저장소입니다. 각 서브프로젝트별로 문서가 분류되어 있습니다.

## Directory Structure

```
docs/
├── README.md                          # This file
├── era-smartfarm-rag/                 # ERA RAG 시스템 문서
│   ├── methodology/                   # 연구 방법론
│   │   ├── smartfarm-rag-methodology.md
│   │   └── related-work-detail.md
│   ├── validation/                    # 시스템 검증 보고서
│   │   └── ERA_RAG_VALIDATION_REPORT.md
│   ├── deployment/                    # 배포 가이드
│   │   └── EDGE_DEPLOYMENT.md
│   └── paper/                         # 논문 초안
│       ├── paper_sections_draft.md
│       ├── experiments_section_draft.md
│       ├── EXPERIMENT_GUIDE.md
│       └── tables/
└── dataset-pipeline/                  # Dataset Pipeline 문서
    ├── dataset/                       # 데이터셋 문서
    │   ├── DATASET_CARD.md
    │   └── LIMITATIONS.md
    ├── validation/                    # 데이터셋 검증 보고서
    │   └── DATASET_VALIDATION_REPORT.md
    └── paper/                         # 논문 초안 (예정)
```

---

## era-smartfarm-rag

Edge 환경용 스마트팜 도메인 특화 RAG 시스템

### Methodology (`era-smartfarm-rag/methodology/`)

| Document | Description |
|----------|-------------|
| `smartfarm-rag-methodology.md` | 시스템 설계 방법론, 아키텍처 결정 근거 |
| `related-work-detail.md` | 관련 연구 상세 정리 (RAG, Hybrid Retrieval, Graph RAG 등) |

### Validation (`era-smartfarm-rag/validation/`)

| Document | Description |
|----------|-------------|
| `ERA_RAG_VALIDATION_REPORT.md` | 엣지 환경 검증 보고서 (25-40x 속도 향상 달성) |

### Deployment (`era-smartfarm-rag/deployment/`)

| Document | Description |
|----------|-------------|
| `EDGE_DEPLOYMENT.md` | 8GB RAM 엣지 디바이스 배포 가이드 |

### Paper (`era-smartfarm-rag/paper/`)

| Document | Description |
|----------|-------------|
| `paper_sections_draft.md` | 논문 Abstract, Introduction, Conclusion 초안 |
| `experiments_section_draft.md` | 실험 섹션 초안 (Baseline, Ablation, Edge) |
| `EXPERIMENT_GUIDE.md` | 실험 실행 및 재현 가이드 |
| `tables/` | 논문용 테이블 (Baseline, Ablation, Edge Performance) |

---

## dataset-pipeline

LLM-as-a-Judge 기반 데이터셋 생성 파이프라인

### Dataset (`dataset-pipeline/dataset/`)

| Document | Description |
|----------|-------------|
| `DATASET_CARD.md` | Hugging Face 형식 데이터셋 카드 |
| `LIMITATIONS.md` | 알려진 한계점 및 윤리적 고려사항 |

### Validation (`dataset-pipeline/validation/`)

| Document | Description |
|----------|-------------|
| `DATASET_VALIDATION_REPORT.md` | 데이터셋 품질 검증 보고서 (ROUGE-L 0.93 다양성) |

### Paper (`dataset-pipeline/paper/`)

향후 데이터셋 논문 초안 작성 예정

---

## Key Results Summary

### ERA RAG Performance

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Index Build | 233s | 24s | **10x faster** |
| Query Latency | 4-7s | 150-180ms | **25-40x faster** |
| Memory Usage | 3.4GB | 855MB | **4x reduction** |

### Dataset Statistics

- **Corpus**: 400 documents (bilingual EN-KO)
- **QA Pairs**: 220 questions
- **Quality Score**: 0.67 (LLM-as-a-Judge)
- **Diversity**: 0.93 (ROUGE-L)

---

## Contributing

새 문서 추가 시:

1. 해당 프로젝트 폴더 하위에 적절한 카테고리 선택
2. Markdown 형식으로 작성
3. 이 README에 문서 목록 업데이트
