#!/bin/bash

# Script to generate LaTeX files, fix issues, and compile with local pdflatex
# Usage: ./build_pdf_fixed.sh

# Set the current directory to the script's directory
cd "$(dirname "$0")"

echo "Step 1: Generating LaTeX files only..."
make latexonly

if [ $? -ne 0 ]; then
    echo "Error: Failed to generate LaTeX files."
    exit 1
fi

echo "Step 2: Fixing and building LaTeX files..."
./fix_and_build_latex.sh

if [ $? -ne 0 ]; then
    echo "Error: Failed to fix and build LaTeX files."
    exit 1
fi

echo "PDF generation completed successfully!"
echo "PDF file is at: build/latex/pdfragsystem.pdf"

exit 0 