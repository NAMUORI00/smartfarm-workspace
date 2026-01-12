# AGENTS.md - Agentic Coding Guidelines

## Repository Structure

Git submodule-based monorepo with two Python 3.10+ projects:

| Directory | Description |
|-----------|-------------|
| `era-smartfarm-rag/` | RAG server (Edge/Jetson, Qwen3 + llama.cpp) |
| `dataset-pipeline/` | LLM-as-a-Judge dataset generation pipeline |

---

## Build/Lint/Test Commands

### era-smartfarm-rag

```bash
cd era-smartfarm-rag
python setup.py --mode local    # Install deps + download GGUF models
make build && make up           # Docker build & run
make logs                       # View logs
uvicorn core.main:app --port 41177 --reload  # Dev server (no Docker)
```

### dataset-pipeline

```bash
cd dataset-pipeline
pip install -r requirements.txt
pip install -e ".[dev]"         # pytest, black, ruff

# Run tests
python tests/test_pipeline.py   # Smoke test (no LLM)
pytest tests/test_pipeline.py -v  # Single test with pytest

# Lint
ruff check src/ && black --check src/
```

### Environment Variables

```bash
export API_KEY="your-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export HF_TOKEN="your-hf-token"  # Optional, for gated datasets
```

---

## Code Style

### Imports

**era-smartfarm-rag** - Absolute imports from `core`:
```python
from core.Models.Schemas import QueryRequest
from core.Services.Retrieval.Base import BaseRetriever
```

**dataset-pipeline** - Relative imports within package:
```python
from .llm_connector import LLMConnector
```

**Order**: stdlib > third-party > local (blank line between each)

**Always**: `from __future__ import annotations`

### Naming

| Element | Convention | Example |
|---------|------------|---------|
| Classes | PascalCase | `DatasetPipeline` |
| Functions | snake_case | `load_documents` |
| Constants | UPPER_SNAKE | `CHUNK_SIZE` |
| Files | snake_case | `llm_connector.py` |
| Dirs (RAG) | PascalCase | `Api/`, `Services/` |

### Type Hints (Required)

```python
from typing import List, Dict, Optional, Any

def search(self, q: str, k: int = 4) -> List[SourceDoc]: ...

@dataclass
class LLMConfig:
    base_url: str
    temperature: float = 0.7

class QueryRequest(BaseModel):  # Pydantic for API
    question: str = Field(...)
```

### Error Handling

```python
# Optional dependency
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

# Abstract method
@abstractmethod
def search(self, q: str, k: int = 4) -> List[SourceDoc]:
    raise NotImplementedError
```

### Docstrings

```python
def build_dataset(self, doc_path: str) -> str:
    """
    Build QA dataset from documents.
    
    Args:
        doc_path: Path to document file or directory
    
    Returns:
        Output file path
    """
```

---

## Architecture

### Config Management

- **RAG**: `dataclass` + `dotenv` in `core/Config/Settings.py`
- **Pipeline**: YAML (`config/settings.yaml`) with `${ENV_VAR}` substitution

### Interfaces

```python
from abc import ABC, abstractmethod

class BaseRetriever(ABC):
    @abstractmethod
    def search(self, q: str, k: int = 4) -> List[SourceDoc]:
        raise NotImplementedError
```

### Data Structures

- `dataclass` for internal state
- `pydantic.BaseModel` for API request/response

---

## Key Dependencies

| era-smartfarm-rag | dataset-pipeline |
|-------------------|------------------|
| fastapi, uvicorn | openai |
| llama-cpp-python | langchain |
| faiss-cpu | sentence-transformers |
| sentence-transformers | pytest, black, ruff |
| easyocr | |

---

## Documentation

All documentation MUST be stored in the **workspace root `docs/` directory**, not in submodules.

### Directory Structure

```
docs/
├── README.md                      # Document index
├── methodology/                   # Research methodology, related work
├── validation/                    # Validation reports, benchmarks
├── dataset/                       # Dataset cards, limitations
└── paper/                         # Paper drafts, figures
    └── figures/
```

### Rules

| Document Type | Location | Examples |
|---------------|----------|----------|
| Validation reports | `docs/validation/` | `ERA_RAG_VALIDATION_REPORT.md` |
| Paper drafts | `docs/paper/` | `experiments_section_draft.md` |
| Dataset documentation | `docs/dataset/` | `DATASET_CARD.md`, `LIMITATIONS.md` |
| Methodology/Design | `docs/methodology/` | `smartfarm-rag-methodology.md` |

**NEVER** create documentation in submodule directories (`era-smartfarm-rag/docs/`, `dataset-pipeline/docs/`).

---

## Important Notes

1. **Submodules**: Commit in each subdir first, then update workspace refs
2. **Edge Target**: 8GB RAM (Jetson/4060 Ti), use GGUF Q4_K_M
3. **Korean**: Comments/docstrings may be Korean - maintain consistency
4. **No CI/CD**: Run tests manually before commits
5. **LLM Tests**: Most require `API_KEY` env var
6. **Documentation**: All docs in workspace root `docs/` (see Documentation section above)
