<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-27 | Updated: 2026-01-27 -->

# dataset-pipeline/

Documentation for the dataset-pipeline project - an LLM-as-a-Judge based synthetic dataset generation pipeline for smart farming QA systems.

## Key Files

| File | Description |
|------|-------------|
| None at this level | See subdirectories for specific documentation |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `dataset/` | Dataset documentation including Hugging Face-format dataset card and limitations (see `dataset/AGENTS.md`) |
| `validation/` | Dataset quality validation reports and statistical analysis (see `validation/AGENTS.md`) |

## Key Results

- **Corpus**: 400 documents (bilingual EN-KO)
- **QA Pairs**: 220 questions
- **Quality Score**: 0.67 (LLM-as-a-Judge)
- **Diversity**: 0.93 (ROUGE-L)

## For AI Agents

- This directory contains comprehensive documentation for the dataset-pipeline project
- Core codebase is in the `dataset-pipeline/` submodule (NOT in this directory)
- Documentation includes: dataset cards, limitations, validation reports
- NEVER modify these files without explicit user request
- Reference dataset documentation when understanding training data characteristics
- Check validation reports for dataset quality metrics and statistical properties
