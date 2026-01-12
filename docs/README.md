# SmartFarm RAG Documentation

This directory contains all documentation for the SmartFarm RAG project, including methodology, validation reports, dataset cards, and paper drafts.

## Directory Structure

```
docs/
├── README.md                      # This file
├── methodology/                   # Research methodology and related work
│   ├── smartfarm-rag-methodology.md
│   └── related-work-detail.md
├── validation/                    # System validation reports
│   ├── ERA_RAG_VALIDATION_REPORT.md   # RAG system validation
│   └── DATASET_VALIDATION_REPORT.md   # Dataset quality validation
├── dataset/                       # Dataset documentation
│   ├── DATASET_CARD.md
│   └── LIMITATIONS.md
└── paper/                         # Academic paper drafts
    ├── paper_sections_draft.md
    ├── experiments_section_draft.md
    └── figures/                   # Figures and tables for paper
```

## Document Overview

### Methodology (`methodology/`)

| Document | Description |
|----------|-------------|
| `smartfarm-rag-methodology.md` | RAG system design methodology and architecture decisions |
| `related-work-detail.md` | Detailed analysis of related research (RAG, Hybrid Retrieval, Graph RAG, etc.) |

### Validation Reports (`validation/`)

| Document | Description |
|----------|-------------|
| `ERA_RAG_VALIDATION_REPORT.md` | Edge environment validation with performance benchmarks (25-40x speedup achieved) |
| `DATASET_VALIDATION_REPORT.md` | QA dataset quality validation using LLM-as-a-Judge methodology |

### Dataset Documentation (`dataset/`)

| Document | Description |
|----------|-------------|
| `DATASET_CARD.md` | Dataset card following Hugging Face format |
| `LIMITATIONS.md` | Known limitations and ethical considerations |

### Paper Drafts (`paper/`)

| Document | Description |
|----------|-------------|
| `paper_sections_draft.md` | Draft of main paper sections |
| `experiments_section_draft.md` | Experiments section with placeholder tables |

## Key Results Summary

### ERA RAG Performance (from validation report)

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Index Build | 233s | 24s | **10x faster** |
| Query Latency | 4-7s | 150-180ms | **25-40x faster** |
| Memory Usage | 3.4GB | 855MB | **4x reduction** |
| Cache Hit | N/A | <1ms | **Instant** |

### Dataset Statistics

- **Corpus**: 400 documents (bilingual EN-KO)
- **QA Pairs**: 220 questions
- **Categories**: cultivation, environment, pest, nutrition, processing, variety
- **Quality Score**: 0.67 (LLM-as-a-Judge)

## Contributing

When adding new documentation:

1. Place methodology/design docs in `methodology/`
2. Place validation/benchmark reports in `validation/`
3. Place dataset-related docs in `dataset/`
4. Place paper drafts and figures in `paper/`

All documents should use Markdown format with clear headings and tables.
