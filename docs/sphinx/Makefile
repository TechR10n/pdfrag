# Minimal makefile for Sphinx documentation

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
.PHONY: help Makefile clean markdown pdf pdftex rinohtex alldocs

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Build HTML documentation
html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Clean build directory
clean:
	rm -rf $(BUILDDIR)/*

# Generate Markdown output
markdown:
	@$(SPHINXBUILD) -M markdown "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Check if rinohtype is available
check-rinoh:
	@python -c "import rinoh" >/dev/null 2>&1 || python -c "import rinohtype" >/dev/null 2>&1 || (echo "rinohtype not available. Install with: pip install rinohtype==0.5.4"; exit 1)

# Build PDF documentation with rinohtype (cross-platform, no LaTeX required)
rinohpdf: check-rinoh
	@$(SPHINXBUILD) -b rinoh "$(SOURCEDIR)" "$(BUILDDIR)/rinoh" $(SPHINXOPTS) $(O) || echo "Failed to build with rinohtype. Check if rinohtype is properly installed."
	@echo
	@echo "Build finished. The PDF file is in $(BUILDDIR)/rinoh."

# Build PDF documentation with LaTeX (requires macOS TeX installation)
# This is the preferred method on macOS for higher quality output
latexpdf:
	@echo "Building PDF with LaTeX (macOS texmf)..."
	@$(SPHINXBUILD) -M latex "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "Running LaTeX files through pdflatex..."
	@cd $(BUILDDIR)/latex && make PDFLATEX=pdflatex
	@echo "PDF output is in $(BUILDDIR)/latex."

# Default PDF generation - uses LaTeX if available, falls back to rinohtype
pdf:
	@if command -v pdflatex >/dev/null 2>&1; then \
		$(MAKE) latexpdf; \
	else \
		echo "pdflatex not found, trying rinohtype..."; \
		if python -c "import rinoh" >/dev/null 2>&1 || python -c "import rinohtype" >/dev/null 2>&1; then \
			$(MAKE) rinohpdf; \
		else \
			echo "Neither pdflatex nor rinohtype is available. PDF generation is disabled."; \
			echo "Install rinohtype with: pip install rinohtype==0.5.4"; \
		fi; \
	fi

# Build all documentation formats
alldocs: html
	@echo
	@echo "HTML documentation has been built."
	@echo "HTML: $(BUILDDIR)/html/"
	@if command -v pdflatex >/dev/null 2>&1; then \
		$(MAKE) latexpdf && \
		echo "PDF (LaTeX): $(BUILDDIR)/latex/pdfragsystem.pdf"; \
	elif python -c "import rinoh" >/dev/null 2>&1 || python -c "import rinohtype" >/dev/null 2>&1; then \
		$(MAKE) rinohpdf && \
		echo "PDF (rinohtype): $(BUILDDIR)/rinoh/pdfragsystem.pdf"; \
	else \
		echo "PDF generation skipped (neither pdflatex nor rinohtype is available)"; \
	fi

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
