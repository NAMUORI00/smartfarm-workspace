# SmartFarm Workspace Documentation

스마트팜 프로젝트 통합 문서 저장소입니다. 각 서브프로젝트별로 문서가 분류되어 있습니다.

## Directory Structure

```
docs/
├── README.md                          # This file
├── era-smartfarm-rag/                 # ERA RAG 시스템 문서
│   ├── paper/                         # 논문 섹션별 파일
│   │   ├── 00_abstract.md             # 초록
│   │   ├── 01_introduction.md         # 서론
│   │   ├── 02_related_work.md         # 관련 연구
│   │   ├── 03_methodology.md          # 제안 방법론
│   │   ├── 04_implementation.md       # 시스템 구현
│   │   ├── 05_experiments.md          # 실험 및 평가
│   │   ├── 06_conclusion.md           # 결론
│   │   ├── references.md              # 참고문헌
│   │   ├── supplementary/             # 보충 자료
│   │   │   └── related_work_tables.md # 관련 연구 상세 테이블
│   │   ├── tables/                    # 논문용 테이블
│   │   └── figures/                   # 논문용 그림
│   ├── guides/                        # 실행 가이드
│   │   └── EXPERIMENT_GUIDE.md        # 실험 실행 및 재현 가이드
│   ├── validation/                    # 시스템 검증 보고서
│   │   └── ERA_RAG_VALIDATION_REPORT.md
│   └── deployment/                    # 배포 가이드
│       └── EDGE_DEPLOYMENT.md
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

### Paper (`era-smartfarm-rag/paper/`)

논문을 섹션별로 분리하여 관리합니다.

| Section | File | Description |
|---------|------|-------------|
| 초록 | `00_abstract.md` | 연구 요약 |
| 1. 서론 | `01_introduction.md` | 연구 배경, 목표, 논문 구성 |
| 2. 관련 연구 | `02_related_work.md` | RAG, Hybrid Retrieval, Graph RAG, 농업 온톨로지 등 |
| 3. 방법론 | `03_methodology.md` | 시스템 개요, 온톨로지, 동적 가중치, 인과관계 그래프 |
| 4. 구현 | `04_implementation.md` | 기술 스택, 핵심 모듈, 엣지 배포 사양 |
| 5. 실험 | `05_experiments.md` | 실험 설계, 베이스라인 비교, Ablation, 엣지 성능 |
| 6. 결론 | `06_conclusion.md` | 연구 요약, 향후 연구 |
| 참고문헌 | `references.md` | 37개 참고문헌 |

### Supplementary Materials

| Document | Description |
|----------|-------------|
| `supplementary/related_work_tables.md` | 관련 연구 상세 비교 테이블 |
| `tables/` | 논문용 테이블 (Baseline, Ablation, Edge Performance) |
| `figures/` | 시스템 아키텍처 다이어그램 |

### Guides (`era-smartfarm-rag/guides/`)

| Document | Description |
|----------|-------------|
| `EXPERIMENT_GUIDE.md` | 실험 실행 및 재현 가이드 |

### Validation (`era-smartfarm-rag/validation/`)

| Document | Description |
|----------|-------------|
| `ERA_RAG_VALIDATION_REPORT.md` | 엣지 환경 검증 보고서 (25-40x 속도 향상 달성) |
| `HYBRID_RETRIEVAL_ROOT_CAUSE_REPORT.md` | 하이브리드 성능 저하 원인 및 개선 보고서 |

### Deployment (`era-smartfarm-rag/deployment/`)

| Document | Description |
|----------|-------------|
| `EDGE_DEPLOYMENT.md` | 8GB RAM 엣지 디바이스 배포 가이드 |

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
