# Makefile for Korean LaTeX document compilation
# Uses XeLaTeX via latexmk for proper Korean font support

.PHONY: all watch clean distclean help

LATEXMK = latexmk
MAIN_TEX = main.tex
OUTPUT_DIR = output
PDF_OUTPUT = $(OUTPUT_DIR)/main.pdf

all: $(PDF_OUTPUT)

$(PDF_OUTPUT): $(MAIN_TEX)
	@echo "Building PDF with XeLaTeX..."
	$(LATEXMK) $(MAIN_TEX)

watch:
	@echo "Starting continuous compilation with preview..."
	@echo "Press Ctrl+C to stop"
	$(LATEXMK) -pvc $(MAIN_TEX)

clean:
	@echo "Cleaning build artifacts..."
	$(LATEXMK) -c $(MAIN_TEX)
	@if [ -d $(OUTPUT_DIR) ]; then \
		find $(OUTPUT_DIR) -type f ! -name '*.pdf' -delete; \
	fi

distclean:
	@echo "Removing all generated files including PDF..."
	$(LATEXMK) -C $(MAIN_TEX)
	@if [ -d $(OUTPUT_DIR) ]; then \
		rm -rf $(OUTPUT_DIR); \
	fi

help:
	@echo "Available targets:"
	@echo "  all        - Build PDF (default)"
	@echo "  watch      - Continuous compilation with preview"
	@echo "  clean      - Remove build artifacts (keep PDF)"
	@echo "  distclean  - Remove everything including PDF"
	@echo "  help       - Show this help message"
