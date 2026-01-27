<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-27 | Updated: 2026-01-27 -->

# dataset-pipeline/dataset/

Dataset documentation including Hugging Face dataset card format and known limitations.

## Key Files

| File | Description |
|------|-------------|
| `DATASET_CARD.md` | Hugging Face-format dataset card with schema, statistics, and usage examples |
| `LIMITATIONS.md` | Known limitations, ethical considerations, and usage guidelines |

## Dataset Statistics

- **Size**: 220 QA pairs
- **Corpus**: 400 documents (bilingual English-Korean)
- **Domain**: Wasabi cultivation, SmartFarm, Hydroponics
- **Language**: Korean (한국어)
- **Quality Score**: 0.67 (LLM-as-a-Judge evaluation)
- **Diversity**: 0.93 (ROUGE-L based diversity metric)

## For AI Agents

- These files document the Wasabi SmartFarm QA dataset
- Dataset card follows Hugging Face conventions
- NEVER modify dataset documentation without explicit user request
- Reference when understanding dataset characteristics, schema, or usage
- Check limitations before using dataset in new experiments or models
- Dataset is synthetic (LLM-generated) - review ethical considerations before deployment
