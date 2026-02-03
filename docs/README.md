# SmartFarm Workspace Documentation

스마트팜 프로젝트 통합 문서 저장소입니다. (문서는 **workspace root `docs/`** 아래에만 저장)

## 현재 문서 구조 (2026-02-03 기준)

```
docs/
├── README.md
└── era-smartfarm-rag/
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

> 참고: 과거에 존재하던 `validation/`, `deployment/`, `dataset-pipeline/`, `paper-tex/`, 이미지 초안(.image) 등은 문서 정리 과정에서 제거되었습니다.

## 논문 파일 (era-smartfarm-rag/paper/)

- 섹션별 Markdown 초안 관리
- 엔드투엔드 스택/구현 변경 사항은 논문 초안에 반영 (Tri-Graph RAG clean-room 구현 및 compose 기반 검증 흐름 등)
