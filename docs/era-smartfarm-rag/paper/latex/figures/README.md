# Figures Directory

LaTeX 논문용 그림 파일을 섹션별로 관리합니다.

## Directory Structure

```
figures/
├── architecture/     # 시스템 아키텍처 다이어그램
│   └── 6-layer-stack.pdf       # fig:architecture
├── methodology/      # 방법론 관련 그림
│   ├── hybriddat-flow.pdf      # fig:hybriddat
│   └── context-shaping.pdf     # fig:context_shaping
└── experiments/      # 실험 결과 그래프
    ├── baseline-comparison.pdf # (optional)
    └── edge-performance.pdf    # (optional)
```

## Current Placeholders (TODO)

논문에서 현재 placeholder로 표시된 그림 목록:

| Figure | Label | Location | Description |
|--------|-------|----------|-------------|
| Figure 1 | `fig:architecture` | methodology.tex | 6-Layer Stack Architecture |
| Figure 2 | `fig:hybriddat` | methodology.tex | HybridDAT Retrieval Flow |
| Figure 3 | `fig:context_shaping` | methodology.tex | Context Shaping Pipeline |

## Usage in LaTeX

`main.tex`에서 `\graphicspath`가 설정되어 있으므로, 파일명만으로 참조 가능:

```latex
% 현재 placeholder 코드:
\fbox{\parbox{...}{...}}

% 실제 이미지로 교체 시:
\includegraphics[width=\textwidth]{6-layer-stack}
```

## Supported Formats

- **PDF** (권장): 벡터 그래픽, 최고 품질
- **PNG**: 래스터 이미지, 300 DPI 이상 권장
- **JPG**: 사진류 이미지

## Design Specifications

그림 디자인 스펙은 `docs/era-smartfarm-rag/paper/figures/` 참조:
- `ARCHITECTURE_FIGURE_DESIGN.md`
- `6_LAYER_STACK_DESIGN_SPEC.md`
- `ARCHITECTURE_MERMAID.md`
