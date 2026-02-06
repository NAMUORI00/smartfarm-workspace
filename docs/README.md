# SmartFarm Workspace Documentation

스마트팜 프로젝트 통합 문서 저장소입니다. (문서는 **workspace root `docs/`** 아래에만 저장)

## 현재 문서 구조 (2026-02-06 기준)

```
docs/
├── README.md
└── era-smartfarm-rag/
    ├── validation/
    │   ├── CHUNKING_GLEANING_TUNING.md
    │   ├── PERFORMANCE_TARGET_PROFILE.md
    │   └── PERFORMANCE_FORENSIC_REPORT_2026-02-06.md
    └── paper/
        ├── 00_abstract.md
        ├── 01_introduction.md
        ├── 02_related_work.md
        ├── 03_methodology.md
        ├── 04_implementation.md
        ├── 05_experiments.md
        ├── 05_experiments_beir.md
        ├── 06_conclusion.md
        └── references.md
```

> 참고: 과거 문서 정리 과정에서 `validation/`은 제거되었으나, 2026-02부터 “튜닝/벤치 리포트” 용도로 **최소한만** 재도입했습니다.

## 논문 파일 (era-smartfarm-rag/paper/)

- 섹션별 Markdown 초안 관리
- 엔드투엔드 스택/구현 변경 사항은 논문 초안에 반영 (Tri-Graph RAG clean-room 구현 및 compose 기반 검증 흐름 등)

## 검증 문서 (era-smartfarm-rag/validation/)

- `CHUNKING_GLEANING_TUNING.md`: 청킹/글리닝 기본값 선정 근거 및 재현 절차
- `PERFORMANCE_TARGET_PROFILE.md`: Retrieval-only vs End-to-End 목표 분리 프로파일
- `PERFORMANCE_FORENSIC_REPORT_2026-02-06.md`: latency 계측 무결성 점검 및 재현 커맨드

closeout 산출물(ignored runtime output):

- `output/perf_analysis/2026-02-06-closeout/summary.json`
  - 공식 판정 run 목록(`official_runs`)
  - hard/jitter 분리 집계
  - 시나리오별 목표 판정 결과
