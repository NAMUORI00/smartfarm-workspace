# Citation Key Mapping

This file maps markdown reference numbers `[N]` from the original document to BibTeX citation keys for LaTeX.

## Usage in LaTeX

Replace markdown citations with `\cite{key}` or `\citep{key}`:

- Markdown: `[1]` → LaTeX: `\cite{lewis2020rag}`
- Multiple: `[1, 2]` → LaTeX: `\cite{lewis2020rag,gao2024survey}`

## Mapping Table

| Markdown [N] | BibTeX Key | Short Description |
|--------------|------------|-------------------|
| [1] | lewis2020rag | RAG original paper (NeurIPS 2020) |
| [2] | gao2024survey | RAG survey for LLMs |
| [3] | peng2024graph | Graph RAG survey |
| [4] | gong2025application | KG and LLM in agriculture trends |
| [5] | karpukhin2020dense | Dense Passage Retrieval (DPR) |
| [6] | yang2024cluster | Cluster-based partial dense retrieval |
| [7] | edge2024graphrag | Microsoft GraphRAG (local to global) |
| [8] | chen2025pathrag | PathRAG (relational path pruning) |
| [9] | bhuyan2021ontology | Smart agriculture ontology |
| [10] | ahmadzai2024innovative | Agricultural ontology using NLP |
| [11] | cornei2024ontology | Spatio-temporal ontology for smart agriculture |
| [12] | yan2025cropdpkg | CropDP-KG (China crop diseases/pests) |
| [13] | wang2024tomatokg | Tomato leaf pests/diseases KG |
| [14] | yang2022causal | Causal relation extraction survey |
| [15] | ieee2023semisupervised | Semi-supervised relation extraction in agriculture |
| [16] | liu2024caexr | CaEXR (causal extraction framework) |
| [17] | sitokonstantinou2024causal | Causal ML for sustainable agroecosystems |
| [18] | carbonell1998mmr | MMR (Maximal Marginal Relevance) |
| [19] | gao2024vrsd | VRSD (rethinking similarity/diversity) |
| [20] | ananieva2025smmr | SMMR (sampling-based MMR) |
| [21] | chen2019agrikg | AgriKG (Chinese agricultural KG) |
| [22] | xu2025sustainable | Quantized LLMs for edge AI |
| [23] | gerganov2024llamacpp | llama.cpp (LLM inference in C++) |
| [24] | seemakhupt2024edgerag | EdgeRAG (online-indexed RAG for edge) |
| [25] | tulkens2024model2vec | Model2Vec (fast static embeddings) |
| [26] | google2025embeddinggemma | EmbeddingGemma (lightweight embeddings) |
| [27] | vasisht2017farmbeats | FarmBeats (IoT platform for agriculture) |
| [28] | saizrubio2020smartfarming | Smart farming to Agriculture 5.0 review |
| [29] | jiang2025farmlightseek | Farm-LightSeek (edge-centric multimodal IoT) |
| [30] | cormack2009rrf | Reciprocal Rank Fusion (RRF) |
| [31] | thakur2021beir | BEIR (IR model evaluation benchmark) |
| [32] | guo2024lightrag | LightRAG (simple fast RAG) |
| [33] | wu2025cropgraphrag | Crop GraphRAG (pest/disease Q&A) |
| [34] | yang2025intelligentqa | Intelligent Q&A with adaptive hybrid retrieval |
| [35] | li2025regrag | ReG-RAG (query rewriting + KG enhancement) |
| [36] | ray2025agrometllm | AgroMetLLM (edge evapotranspiration system) |
| [37] | jiang2025agricultural | Agricultural LLM with precise retrieval |
| [38] | es2024ragas | RAGAS (automated RAG evaluation) |
| [39] | saadfalcon2024ares | ARES (RAG evaluation framework) |
| [40] | niu2024ragchecker | RAGChecker (fine-grained RAG diagnostics) |

## BibTeX Entry Types Used

- **@inproceedings**: Conference papers (SIGIR, NeurIPS, EMNLP, etc.)
- **@article**: Journal articles (Nature, Frontiers, Smart Agriculture, etc.)
- **@misc**: arXiv preprints and technical reports
- **@software**: Software packages and tools (llama.cpp, Model2Vec)

## Citation Key Format

Citation keys follow the pattern: `{firstAuthorLastname}{year}{keyword}`

Examples:
- `lewis2020rag` - Lewis et al., 2020, RAG paper
- `gao2024survey` - Gao et al., 2024, Survey paper
- `edge2024graphrag` - Edge et al., 2024, GraphRAG paper

## Notes

- Author lists with "et al." are expanded to `and others` in BibTeX
- All entries include `url` field for accessibility
- arXiv papers include `eprint`, `archiveprefix`, and `primaryclass` fields
- DOI included where available (e.g., ray2025agrometllm, jiang2025agricultural)
