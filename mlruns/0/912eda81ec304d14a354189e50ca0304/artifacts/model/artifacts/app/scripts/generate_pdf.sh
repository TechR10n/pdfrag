#!/bin/bash
# Script to generate PDF from Markdown with proper handling of SVG images

# Check if input file is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <markdown_file>"
    exit 1
fi

MARKDOWN_FILE=$1
DOCS_DIR="docs"

# Make scripts executable
chmod +x app/scripts/convert_svg_to_pdf.py
chmod +x app/scripts/convert_md_to_pdf.py

# Step 1: Convert SVG files to PDF
echo "Converting SVG files to PDF..."
python app/scripts/convert_svg_to_pdf.py $DOCS_DIR $MARKDOWN_FILE

# Step 2: Generate PDF from Markdown
echo "Generating PDF from Markdown..."
python app/scripts/convert_md_to_pdf.py $MARKDOWN_FILE --keep-tex

echo "Done!" 