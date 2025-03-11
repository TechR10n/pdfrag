#!/bin/bash

# Exit on error
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Project root directory
PROJECT_ROOT=$(git rev-parse --show-toplevel || pwd)
DOCS_DIR="$PROJECT_ROOT/docs"
SPHINX_SOURCE_DIR="$DOCS_DIR/sphinx/source"

echo -e "${GREEN}Preparing markdown files for LaTeX processing...${NC}"

# Clean up any existing _latex files
find "$SPHINX_SOURCE_DIR" -name "*_latex*.md" -type f -delete

# Function to process a markdown file
process_markdown() {
    local file="$1"
    # Skip if file is already a _latex version
    if [[ "$file" == *"_latex"* ]]; then
        return
    fi
    echo "Processing $file..."
    python "$PROJECT_ROOT/app/scripts/fix_markdown.py" "$file"
}

# Process all markdown files in the Sphinx source directory
find "$SPHINX_SOURCE_DIR" -name "*.md" -type f | while read -r file; do
    process_markdown "$file"
done

# Convert SVG files to PDF
echo -e "\n${GREEN}Converting SVG files to PDF...${NC}"
find "$DOCS_DIR/puml/svg" -name "*.svg" -type f | while read -r file; do
    pdf_file="${file%.svg}.pdf"
    if [ ! -f "$pdf_file" ] || [ "$file" -nt "$pdf_file" ]; then
        echo "Converting $file to PDF..."
        if ! cairosvg "$file" -o "$pdf_file" 2>/dev/null; then
            echo -e "${YELLOW}Warning: Could not convert $file using cairosvg, trying Inkscape...${NC}"
            if command -v inkscape >/dev/null 2>&1; then
                inkscape --export-filename="$pdf_file" "$file" 2>/dev/null || {
                    echo -e "${RED}Error: Failed to convert $file using both cairosvg and Inkscape${NC}"
                    continue
                }
            else
                echo -e "${RED}Error: Inkscape not found. Please install it to convert problematic SVG files${NC}"
                continue
            fi
        fi
    fi
done

# Create symbolic links to PDF files if needed
echo -e "\n${GREEN}Creating symbolic links to PDF files...${NC}"
cd "$SPHINX_SOURCE_DIR"
find "$DOCS_DIR/puml/svg" -name "*.pdf" -type f | while read -r file; do
    base_name=$(basename "$file")
    if [ ! -f "$base_name" ]; then
        ln -sf "$file" "$base_name"
    fi
done

echo -e "\n${GREEN}Done preparing files for LaTeX processing.${NC}"

# Check for potential issues
echo -e "\n${YELLOW}Checking for potential LaTeX issues...${NC}"

# Check for special characters
echo "Checking for problematic characters..."
find "$SPHINX_SOURCE_DIR" -name "*_latex.md" -type f -exec grep -l "[^[:print:]]" {} \; || true

# Check for unescaped special characters that haven't been properly processed
echo "Checking for unescaped LaTeX special characters..."
find "$SPHINX_SOURCE_DIR" -name "*_latex.md" -type f -exec grep -l "[^\\][&%$#_{}~^\\]" {} \; || true

echo -e "\n${GREEN}All done! You can now proceed with LaTeX compilation.${NC}"
echo "If you encounter any issues, check the files listed above for problematic characters."

# Provide a summary of processed files
echo -e "\n${GREEN}Summary of processed files:${NC}"
find "$SPHINX_SOURCE_DIR" -name "*_latex.md" -type f | while read -r file; do
    echo "- $(basename "$file")"
done 