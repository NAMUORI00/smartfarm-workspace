# Tables Directory

LaTeX 논문용 테이블을 섹션별로 관리합니다.

## Directory Structure

```
tables/
├── methodology/          # Section 3: 방법론
│   ├── resource-constraints.tex
│   ├── layers.tex
│   ├── ontology.tex
│   ├── dynamic-alpha.tex
│   ├── causal-patterns.tex
│   ├── crop-filter.tex
│   ├── reranking.tex
│   ├── quantization.tex
│   └── comparison.tex
├── implementation/       # Section 4: 구현
│   ├── tech-stack.tex
│   ├── reranking-impl.tex
│   ├── index-files.tex
│   ├── edge-types.tex
│   ├── deployment-spec.tex
│   └── edgerag-comparison.tex
└── experiments/          # Section 5: 실험
    ├── dataset.tex
    ├── baselines.tex
    ├── baseline-results.tex
    ├── ablation-results.tex
    ├── domain-analysis.tex
    ├── edge-performance.tex
    ├── limitations.tex
    └── threats.tex
```

## Usage in LaTeX

섹션 파일에서 `\input` 명령으로 테이블 포함:

```latex
% sections/methodology.tex
\input{tables/methodology/resource-constraints}
\input{tables/methodology/layers}
```

## Table Labels

| File | Label | Caption |
|------|-------|---------|
| resource-constraints.tex | `tab:resource_constraints` | 엣지 환경 리소스 제약 |
| layers.tex | `tab:layers` | 계층별 역할 및 컴포넌트 |
| ontology.tex | `tab:ontology` | 온톨로지 개념 유형 |
| ... | ... | ... |

## Status

- ✅ 완료: 데이터가 채워진 테이블
- 🔶 TBD: 실험 결과 대기 중
