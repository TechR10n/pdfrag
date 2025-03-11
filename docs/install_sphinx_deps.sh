#!/bin/bash
# Script to install Sphinx dependencies directly

echo "Installing Sphinx and related dependencies..."

# Install core Sphinx packages
pip install sphinx==7.2.6 sphinx-rtd-theme==2.0.0 sphinx-markdown-tables==0.0.17 myst-parser==2.0.0 sphinx-copybutton==0.5.2 sphinxcontrib-bibtex==2.5.0

# Install rinohtype and its dependencies
echo "Installing rinohtype and its dependencies..."
pip install rinoh-typeface-texgyrecursor==0.1.1 rinoh-typeface-texgyreheros==0.1.1 rinoh-typeface-texgyrepagella==0.1.1
pip install rinohtype==0.5.4

echo "Installation complete!"
echo "You can now build documentation with:"
echo "cd sphinx && python build_pdf.py" 