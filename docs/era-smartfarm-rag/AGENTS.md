<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-27 | Updated: 2026-01-27 -->

# era-smartfarm-rag/

Documentation for the ERA-SmartFarm-RAG project - an edge-optimized, domain-specialized Retrieval-Augmented Generation (RAG) system for smart farming operations. Features hybrid retrieval (Dense-Sparse-PathRAG), agricultural domain ontology, causal graphs, and GGUF quantization for 8GB RAM edge devices.

## Key Files

| File | Description |
|------|-------------|
| None at this level | See subdirectories for specific documentation |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `paper/` | Research paper sections and supplementary materials (see `paper/AGENTS.md`) |
| `validation/` | System validation reports and benchmark comparisons (see `validation/AGENTS.md`) |
| `deployment/` | Deployment guides and operational documentation (see `deployment/AGENTS.md`) |

## Key Results

- **Query Latency**: 25-40x faster (4-7s → 150-180ms)
- **Index Build**: 10x faster (233s → 24s)
- **Memory**: 4x reduction (3.4GB → 855MB)
- **Edge Target**: 8GB RAM (Jetson/4060 Ti) with GGUF Q4_K_M quantization

## For AI Agents

- This directory contains comprehensive documentation for the ERA-SmartFarm-RAG research project
- Core codebase is in the `era-smartfarm-rag/` submodule (NOT in this directory)
- Documentation includes: paper sections, experiments, validation reports, deployment guides
- NEVER modify these files without explicit user request
- Reference paper sections when understanding system architecture and design decisions
- Check validation reports for performance benchmarks and optimization results
