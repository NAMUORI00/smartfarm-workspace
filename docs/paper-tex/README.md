# ERA-SmartFarm-RAG Paper: LaTeX Source

A comprehensive research paper on hybrid retrieval-augmented generation (RAG) systems specialized for smart farm domains, with on-device deployment optimization for edge environments.

**Title:** 스마트팜 도메인 특화 하이브리드 RAG 시스템 - 엣지 환경을 위한 Dense-Sparse-PathRAG 융합

**Title (English):** Hybrid Retrieval-Augmented Generation System for Smart Farm Domain: Dense-Sparse-PathRAG Fusion for Edge Deployment

## Project Overview

This paper presents an on-device hybrid RAG system optimized for smart farm operations. The system addresses:

- **Cloud dependency elimination** through edge deployment
- **Domain-specific knowledge integration** using agricultural ontology
- **Multi-channel retrieval fusion** (Dense, Sparse, and PathRAG)
- **Real-time inference** on resource-constrained devices (8GB RAM)
- **Offline fallback capability** for network-disconnected environments

### Research Focus

The paper evaluates the proposed system on **wasabi cultivation scenarios**, combining:
- Sensor-based structured data (temperature, humidity, CO₂, nutrients)
- Unstructured information (cultivation records, work logs, observations)
- Knowledge graphs with causal relationships
- Domain-specific ontology (6 concept types: crop, environment, disease, nutrients, growth stage, cultivation practice)

## Prerequisites

### System Requirements

- **Operating System:** Windows, macOS, or Linux
- **RAM:** Minimum 4GB (8GB recommended for compilation)
- **Disk Space:** 2GB for TeX Live/MiKTeX + dependencies

### LaTeX Distribution

Choose one of the following:

#### Windows
- **MiKTeX** (recommended)
  - Download: https://miktex.org/download
  - Includes automatic package installation
  - Handles Korean support automatically

- **TeX Live** (alternative)
  - Download: https://tug.org/texlive/acquire-netinstall.html
  - Install full scheme for complete package coverage

#### macOS
- **MacTeX**
  - Download: https://tug.org/mactex/
  - Includes all necessary packages
  - Install via Homebrew: `brew install --cask mactex`

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get install texlive-full texlive-xetex texlive-fonts-recommended
```

#### Linux (Fedora)
```bash
sudo dnf install texlive-full texlive-xetex
```

### XeLaTeX

XeLaTeX is the primary PDF engine and must be installed as part of your TeX distribution. Verify installation:

```bash
xelatex --version
```

### Korean Font Support

The document uses XeLaTeX with KoTeX for Korean typography. Fonts are configured in `preamble.tex`.

#### Install Required Fonts

**Windows/macOS/Linux:**
- **Noto CJK Fonts** (recommended)
  - Google Fonts: https://fonts.google.com/noto/specimen/Noto+Serif+CJK+KR
  - Families: Noto Serif CJK KR, Noto Sans CJK KR, Noto Sans Mono CJK KR
  - Installation:
    - Download `.otf` files
    - Install system-wide (fonts folder)
    - No configuration needed; XeLaTeX auto-detects

**Font Configuration (already done in preamble.tex):**
```latex
\setmainfont{Noto Serif CJK KR}
\setsansfont{Noto Sans CJK KR}
\setmonofont{Noto Sans Mono CJK KR}
```

### Bibliography Tools

**Biber** (for bibliography processing with XeLaTeX):

```bash
# Windows (included with MiKTeX/TeX Live)
biber --version

# macOS
brew install biber

# Linux (Ubuntu/Debian)
sudo apt-get install biber

# Linux (Fedora)
sudo dnf install biber
```

### Build Tools

**latexmk** (automatic build orchestration):

```bash
# Usually included with TeX Live/MiKTeX/MacTeX
latexmk --version

# If missing (macOS)
brew install latexmk

# If missing (Linux)
sudo apt-get install latexmk  # Ubuntu/Debian
sudo dnf install latexmk      # Fedora
```

**Make** (for command shortcuts):

```bash
# Windows (with Git Bash or MSYS2)
make --version

# macOS/Linux (pre-installed)
make --version
```

## Installation

### 1. Verify TeX Installation

```bash
xelatex --version
biber --version
latexmk --version
```

### 2. Install Korean Fonts (if needed)

- Download Noto CJK fonts from https://fonts.google.com/noto
- Install system-wide and verify with XeLaTeX

### 3. Clone or Download Project

```bash
git clone <repository-url>
cd docs/paper-tex
```

### 4. Test Build

```bash
make
```

Expected output: `output/main.pdf` is generated

## Build Instructions

### Quick Build

Generate PDF from LaTeX source:

```bash
make
# or
make all
```

Output: `output/main.pdf`

### Live Preview / Watch Mode

Continuous compilation with automatic PDF preview updates (requires PDF viewer with auto-reload):

```bash
make watch
```

- Monitors all source files for changes
- Automatically recompiles on file save
- Launches default PDF viewer
- Press `Ctrl+C` to stop

Supported on: Windows, macOS, Linux

### Clean Build Artifacts (Keep PDF)

Remove intermediate files (`.aux`, `.log`, `.toc`, `.bbl`, etc.) while preserving the PDF:

```bash
make clean
```

### Full Clean (Remove Everything)

Remove all generated files including the PDF:

```bash
make distclean
```

### Show Available Commands

```bash
make help
```

## Directory Structure

```
docs/paper-tex/
├── README.md                      # This file
├── Makefile                       # Build automation
├── .latexmkrc                     # Latexmk configuration (XeLaTeX, Biber, output dir)
├── main.tex                       # Main document entry point
├── preamble.tex                   # Package imports and styling
├── macros.tex                     # Custom LaTeX commands and utilities
│
├── frontmatter/
│   └── abstract.tex               # Paper abstract (Korean)
│
├── sections/
│   ├── 01-introduction.tex        # Introduction & motivation
│   ├── 02-related-work.tex        # Literature review
│   ├── 03-methodology.tex         # System design & architecture (TODO)
│   ├── 04-implementation.tex      # Implementation details (TODO)
│   ├── 05-experiments.tex         # Main experiment results (TODO)
│   ├── 05-experiments-beir.tex    # BEIR benchmark supplement (TODO)
│   └── 06-conclusion.tex          # Conclusion & future work (TODO)
│
├── appendices/
│   ├── appendix-architecture.tex  # Detailed architecture diagrams (TODO)
│   └── appendix-related-work.tex  # Extended literature review (TODO)
│
├── bibliography/
│   ├── references.bib             # BibTeX reference database
│   └── citation_keys.md           # Reference key documentation
│
├── figures/                       # TikZ diagrams & graphics (TODO)
│   └── [diagram files]
│
└── output/
    └── main.pdf                   # Generated PDF (created after first build)
```

## File Descriptions

### Core Files

| File | Purpose |
|------|---------|
| `main.tex` | Main document that loads all sections and components |
| `preamble.tex` | LaTeX package imports, Korean support, styling, theorem environments |
| `macros.tex` | Custom commands, math operators, convenience functions |

### Document Structure

| Section | File | Status | Content |
|---------|------|--------|---------|
| Abstract | `frontmatter/abstract.tex` | Complete | Korean abstract with research overview |
| Introduction | `sections/01-introduction.tex` | Complete | Problem statement, motivation, research objectives |
| Related Work | `sections/02-related-work.tex` | TBD | Literature review on RAG, edge LLM, agriculture ML |
| Methodology | `sections/03-methodology.tex` | TBD | System design, ontology, retrieval fusion strategy |
| Implementation | `sections/04-implementation.tex` | TBD | Technical details, code examples, deployment setup |
| Experiments | `sections/05-experiments.tex` | TBD | Wasabi cultivation domain evaluation, metrics, results |
| BEIR Benchmark | `sections/05-experiments-beir.tex` | TBD | Standard benchmark results for comparison |
| Conclusion | `sections/06-conclusion.tex` | TBD | Findings, contributions, limitations, future work |
| Architecture | `appendices/appendix-architecture.tex` | TBD | Detailed system architecture diagrams |
| Extended Refs | `appendices/appendix-related-work.tex` | TBD | Additional references and background |

### Supporting Files

| File | Purpose |
|------|---------|
| `bibliography/references.bib` | BibTeX database of research papers and sources |
| `bibliography/citation_keys.md` | Documentation of citation keys for easy reference |

## Editing Workflow

### Adding New Content

1. **Create a new section file** in `sections/`:
   ```latex
   \section{Your Section Title}
   \label{sec:section-label}

   Your content here...
   ```

2. **Include in main.tex**:
   ```latex
   \input{sections/your-new-section}
   ```

3. **Rebuild**:
   ```bash
   make
   ```

### Adding Figures

#### Using TikZ Diagrams (Recommended for diagrams)

```latex
\begin{figure}[h]
  \centering
  \begin{tikzpicture}
    % TikZ code here
  \end{tikzpicture}
  \caption{Your caption here}
  \label{fig:your-label}
\end{figure}
```

#### Using Imported Graphics

```latex
\begin{figure}[h]
  \centering
  \includegraphics[width=0.8\textwidth]{figures/your-image.pdf}
  \caption{Your caption here}
  \label{fig:your-label}
\end{figure}
```

### Adding Code Listings

Python code example:
```latex
\begin{lstlisting}[language=Python, caption=Your code caption, label=lst:your-label]
# Python code here
def example():
    pass
\end{lstlisting}
```

JSON configuration example:
```latex
\begin{lstlisting}[style=jsonstyle, caption=Your config, label=lst:your-label]
{
  "key": "value"
}
\end{lstlisting}
```

### Adding References

1. **Add to `bibliography/references.bib`**:
   ```bibtex
   @inproceedings{Smith2024,
     author = {Smith, John},
     title = {Your Paper Title},
     booktitle = {Proceedings of Conference},
     year = {2024}
   }
   ```

2. **Cite in text**:
   ```latex
   As shown by \cite{Smith2024}, ...
   or \citet{Smith2024} argue that ...
   ```

3. **Rebuild** to regenerate bibliography:
   ```bash
   make
   ```

## Known Placeholders and TODOs

### Author Information
- **Status:** TODO
- **Location:** `main.tex` (line 18-21)
- **Action:** Add author names, affiliations, and email addresses

### Section Content

The following sections are structured but require content:

| Section | File | Notes |
|---------|------|-------|
| Related Work | `sections/02-related-work.tex` | Review RAG systems, edge LLM deployment, agricultural AI |
| Methodology | `sections/03-methodology.tex` | Explain Dense-Sparse-PathRAG fusion, ontology design, causal graph extraction |
| Implementation | `sections/04-implementation.tex` | Detail technical stack, quantization approach (Q4_K_M), llama.cpp integration |
| Experiments | `sections/05-experiments.tex` | Present wasabi domain evaluation, compare retrieval approaches |
| BEIR Benchmark | `sections/05-experiments-beir.tex` | Show standard benchmark comparisons |
| Conclusion | `sections/06-conclusion.tex` | Summarize contributions and future directions |

### Figure TODOs

Key diagrams to add:

- **System architecture diagram** (methodology section)
- **Ontology structure visualization** (methodology section)
- **Dense-Sparse-PathRAG pipeline** (implementation section)
- **Quantization comparison charts** (implementation section)
- **Experiment results plots** (experiments section)
- **Performance comparison tables** (experiments section)

Use TikZ in `figures/` directory or include as PNG/PDF.

### Bibliography

- **Current status:** Framework in place, keys documented
- **Action:** Populate `bibliography/references.bib` with research papers
- **See:** `bibliography/citation_keys.md` for organized key structure

## Compilation Troubleshooting

### Common Issues

#### "XeLaTeX not found"
```
Error: xelatex command not found
```
**Solution:** Install TeX Live, MiKTeX, or MacTeX (see Prerequisites section)

#### "Korean fonts not recognized"
```
Error: Font "Noto Serif CJK KR" cannot be found
```
**Solution:**
1. Verify font installation: Check System Fonts folder
2. Update font cache: `fc-cache -fv` (Linux)
3. Use alternative Korean font in `preamble.tex` (e.g., Noto Sans CJK KR)
4. If still failing, fonts can be commented out; XeLaTeX will use fallback

#### "Biber not found"
```
Error: biber command not found
```
**Solution:**
1. Verify installation: `biber --version`
2. Reinstall: `brew install biber` (macOS) or `apt-get install biber` (Linux)
3. For Windows MiKTeX: Run MiKTeX Console → Settings → Packages → Install biber

#### "Bibliography entries not appearing"
**Solution:**
1. Ensure `references.bib` exists and has valid entries
2. Run: `make distclean && make` (full rebuild)
3. Check `.log` file for biber errors: `output/main.log`

#### "PDF viewer fails to open in watch mode"
**Solution:**
1. Set `$pdf_previewer` in `.latexmkrc` to your preferred viewer:
   - Windows: `start %O %S`, `mupdf %O %S`
   - macOS: `open -a Preview %O %S`
   - Linux: `evince %O %S`, `okular %O %S`
2. Manually open `output/main.pdf` after build

## Editing with IDE/Editors

### Recommended Tools

- **VS Code** + LaTeX Workshop extension
- **TeXStudio** (cross-platform, LaTeX-focused)
- **Overleaf** (online, requires export for local edits)
- **Vim/Neovim** + vim-latex plugin

### VS Code Setup

1. Install **LaTeX Workshop** extension by James Yu
2. Settings (`.vscode/settings.json`):
   ```json
   {
     "latex-workshop.latex.tools": [
       {
         "name": "xelatex",
         "command": "xelatex",
         "args": ["-interaction=nonstopmode", "-synctex=1", "%DOC%"]
       }
     ],
     "latex-workshop.latex.recipes": [
       {
         "name": "xelatex + biber",
         "tools": ["xelatex", "biber", "xelatex", "xelatex"]
       }
     ]
   }
   ```

## References & Resources

### LaTeX Documentation
- **Comprehensive TeX Archive Network (CTAN):** https://ctan.org
- **LaTeX Project:** https://www.latex-project.org
- **The Not So Short Introduction to LaTeX:** https://ctan.org/pkg/lshort

### Korean Typography
- **KoTeX Documentation:** https://www.ktug.org
- **XeLaTeX with Korean:** https://en.wikibooks.org/wiki/LaTeX/Fonts#Chinese_Japanese_Korean

### RAG & Edge LLM
- **Langchain RAG Guide:** https://python.langchain.com/docs/use_cases/qa_structured_data/sql
- **llama.cpp Documentation:** https://github.com/ggerganov/llama.cpp
- **Hugging Face Model Quantization:** https://huggingface.co/docs/transformers/quantization

### Smart Farm / Agriculture
- **Controlled Environment Agriculture (CEA) Research:** IEEE, ACM proceedings
- **Plant Disease Detection:** Computer vision datasets and benchmarks
- **Wasabi Cultivation Research:** Agricultural extension publications

## Build Automation Details

### Makefile Targets

| Target | Command | Effect |
|--------|---------|--------|
| `all` (default) | `make` or `make all` | Build PDF once |
| `watch` | `make watch` | Continuous build with preview |
| `clean` | `make clean` | Remove build artifacts (keep PDF) |
| `distclean` | `make distclean` | Remove all generated files |
| `help` | `make help` | Show available commands |

### Latexmk Configuration

The `.latexmkrc` file configures:

- **PDF Engine:** XeLaTeX (`$pdf_mode = 5`)
- **XeLaTeX Options:** Shell escape, synctex, nonstop mode
- **Output Directory:** `output/`
- **Bibliography Processor:** Biber with input directory configuration
- **Clean Extensions:** Build artifacts (`.aux`, `.log`, `.bbl`, etc.)
- **Preview Mode:** Auto-opens PDF viewer on first build and changes

## Performance Notes

### Typical Build Times

- **First build:** 30-60 seconds (includes Biber bibliography processing)
- **Incremental build:** 5-15 seconds (small changes)
- **Full rebuild:** 30-60 seconds
- **Watch mode:** Instant file monitoring, 5-15s per change

### Memory Usage

- **TeX Live full installation:** ~3-5GB disk, ~500MB RAM active
- **MiKTeX with on-demand package installation:** ~2-3GB, lazy loading
- **Compilation:** Typically <200MB RAM peak

### Output Size

- **Typical PDF (50 pages):** 1-2MB
- **With high-resolution figures:** 5-10MB+

## Project Status

### Completion Progress

- **Introduction:** 100% (complete, includes abstract)
- **Related Work:** 0% (outline only)
- **Methodology:** 0% (outline only)
- **Implementation:** 0% (outline only)
- **Experiments:** 0% (outline only)
- **BEIR Benchmark:** 0% (outline only)
- **Conclusion:** 0% (outline only)
- **Appendices:** 0% (placeholders)

### Next Steps

1. Complete author information in `main.tex`
2. Populate Related Work section with literature review
3. Detail methodology (retrieval fusion, ontology)
4. Document implementation specifics
5. Add experiment results and figures
6. Create TikZ diagrams for system architecture
7. Generate and embed benchmark comparison tables

## Contact & Author

**Authors:** [TODO: Add author names and contact information]

**Affiliation:** [TODO: Add institutional affiliation]

**Email:** [TODO: Add contact email]

**Repository:** [TODO: Add repository URL]

**Last Updated:** 2026-01-29

## License

[TODO: Specify license - e.g., Creative Commons, MIT, etc.]

---

**Note:** This LaTeX project is part of the ERA-SmartFarm-RAG research initiative. For implementation code, datasets, and supplementary materials, see the main project repository.
