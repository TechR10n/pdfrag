#!/bin/bash
# Script to generate comprehensive documentation for the project

# Check if this script is being run with Python
if [[ "$0" == *python* ]]; then
    echo "Error: This is a bash script and should be run directly, not with Python."
    echo "Please run it as: ./app/scripts/generate_docs.sh [--format FORMAT]"
    echo "Or: bash app/scripts/generate_docs.sh [--format FORMAT]"
    exit 1
fi

# Parse command line arguments
FORMAT="all"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --format|-f)
      FORMAT="$2"
      shift 2
      ;;
    *)
      # If it's a positional argument without a flag, assume it's the format
      if [[ "$1" =~ ^[a-zA-Z]+$ ]]; then
        FORMAT="$1"
      fi
      shift
      ;;
  esac
done

VALID_FORMATS=("html" "pdf" "markdown" "all")

# Validate format
if [[ ! " ${VALID_FORMATS[@]} " =~ " ${FORMAT} " ]]; then
    echo "Invalid format: $FORMAT"
    echo "Valid formats: html, pdf, markdown, all"
    echo "Usage: $0 [--format FORMAT]"
    exit 1
fi

echo "Generating documentation in format: $FORMAT"

# Make scripts executable
chmod +x app/scripts/setup_sphinx.py
chmod +x app/scripts/build_docs.py
chmod +x app/scripts/convert_svg_to_pdf.py

# Step 1: Convert SVG files to PDF for better compatibility
echo "Converting SVG files to PDF..."
python app/scripts/convert_svg_to_pdf.py docs

# Step 2: Copy dev_notes.md to Sphinx source directory
echo "Copying dev_notes.md to Sphinx source directory..."
mkdir -p docs/sphinx/source
cp dev_notes.md docs/sphinx/source/

# Step 3: Set up Sphinx documentation if not already set up
if [ ! -f docs/sphinx/source/conf.py ]; then
    echo "Setting up Sphinx documentation..."
    python app/scripts/setup_sphinx.py
fi

# Step 4: Update API documentation
echo "Updating API documentation..."
python app/scripts/build_docs.py --api-only

# Check if timeout command is available
if command -v timeout >/dev/null 2>&1; then
    HAS_TIMEOUT=true
else
    HAS_TIMEOUT=false
    echo "Warning: 'timeout' command not found. PDF generation will not have timeout protection."
fi

# Step 5: Build documentation in the requested format(s)
if [ "$FORMAT" == "all" ]; then
    echo "Building documentation in all formats..."
    python app/scripts/build_docs.py --format html --clean
    
    echo "Building PDF documentation..."
    if [ "$HAS_TIMEOUT" = true ]; then
        # Use timeout command to prevent hanging
        timeout 300 python app/scripts/build_docs.py --format pdf || {
            echo "PDF generation timed out after 5 minutes."
            echo "This is likely due to a LaTeX package issue or a complex document."
            echo "HTML documentation should still be available."
        }
    else
        # Run without timeout
        python app/scripts/build_docs.py --format pdf
    fi
    
    python app/scripts/build_docs.py --format markdown
    python app/scripts/build_docs.py --format epub
elif [ "$FORMAT" == "pdf" ]; then
    echo "Building PDF documentation..."
    if [ "$HAS_TIMEOUT" = true ]; then
        # Use timeout command to prevent hanging
        timeout 300 python app/scripts/build_docs.py --format pdf --clean || {
            echo "PDF generation timed out after 5 minutes."
            echo "This is likely due to a LaTeX package issue or a complex document."
        }
    else
        # Run without timeout
        python app/scripts/build_docs.py --format pdf --clean
    fi
else
    echo "Building documentation in $FORMAT format..."
    python app/scripts/build_docs.py --format $FORMAT --clean
fi

echo "Documentation generation complete!"
echo "You can find the generated documentation in the docs/sphinx/build directory."
echo ""
echo "API documentation has been automatically generated from docstrings in your Python modules."
echo "To improve the API documentation, add detailed docstrings to your Python code." 