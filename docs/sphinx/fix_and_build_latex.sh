#!/bin/bash

# Script to fix LaTeX file issues and compile with local pdflatex
# Usage: ./fix_and_build_latex.sh

# Set variables
LATEX_DIR="build/latex"
MAIN_TEX="pdfragsystem.tex"
FULL_PATH="$LATEX_DIR/$MAIN_TEX"

# Check if the LaTeX directory exists
if [ ! -d "$LATEX_DIR" ]; then
    echo "Error: LaTeX directory not found at $LATEX_DIR"
    echo "Please run 'make latexonly' first to generate the LaTeX files."
    exit 1
fi

# Check if the main TeX file exists
if [ ! -f "$FULL_PATH" ]; then
    echo "Error: Main TeX file not found at $FULL_PATH"
    exit 1
fi

echo "Fixing LaTeX file issues..."

# Create a temporary file for modifications
TMP_FILE=$(mktemp)

# Read the original file
cat "$FULL_PATH" > "$TMP_FILE"

# Fix 1: The \Bbbk command redefinition issue
# Insert after fontenc line
awk '
/\\usepackage\[T1\]{fontenc}/ {
    print $0
    print "% Fix for \\Bbbk command redefinition error"
    print "\\let\\Bbbk\\relax"
    print ""
    next
}
{ print }
' "$TMP_FILE" > "$FULL_PATH"

# Fix 2: Fix potential issues with other command redefinitions
# Insert after the \Bbbk fix
awk '
/% Fix for \\Bbbk command redefinition error/ {
    print $0
    print "\\let\\Bbbk\\relax"
    print "% Fix for other potential command redefinitions"
    print "\\let\\openbox\\relax"
    print "\\let\\mathbb\\relax"
    print ""
    next
}
{ print }
' "$FULL_PATH" > "$TMP_FILE"
cat "$TMP_FILE" > "$FULL_PATH"

# Fix 3: Fix potential issues with hyperref and unicode characters
# Insert after hyperref package
awk '
/\\usepackage{hyperref}/ {
    print $0
    print "% Fix for hyperref unicode issues"
    print "\\hypersetup{unicode=true}"
    print ""
    next
}
{ print }
' "$FULL_PATH" > "$TMP_FILE"
cat "$TMP_FILE" > "$FULL_PATH"

# Fix 4: Add a patch for potential font issues
# Insert before begin document
awk '
/\\begin{document}/ {
    print "% Fix for potential font issues"
    print "\\DeclareTextCommandDefault{\\textquotesingle}{\\textquotesingle}"
    print "\\DeclareTextCommandDefault{\\textasciigrave}{\\textasciigrave}"
    print ""
    print $0
    next
}
{ print }
' "$FULL_PATH" > "$TMP_FILE"
cat "$TMP_FILE" > "$FULL_PATH"

# Clean up temporary file
rm "$TMP_FILE"

echo "LaTeX file fixed. Compiling with pdflatex..."

# Change to the LaTeX directory
cd "$LATEX_DIR"

# Run pdflatex with error handling
echo "First pdflatex run..."
pdflatex -interaction=nonstopmode "$MAIN_TEX" > pdflatex_run1.log 2>&1

# Check for errors in the first run
if grep -q "Fatal error" pdflatex_run1.log; then
    echo "Error: First pdflatex run failed with fatal errors."
    echo "Check pdflatex_run1.log for details."
    exit 1
fi

# Run pdflatex a second time to resolve references
echo "Second pdflatex run..."
pdflatex -interaction=nonstopmode "$MAIN_TEX" > pdflatex_run2.log 2>&1

# Check if compilation was successful
if [ -f "${MAIN_TEX%.tex}.pdf" ]; then
    echo "PDF compilation successful!"
    echo "PDF file is at: $LATEX_DIR/${MAIN_TEX%.tex}.pdf"
else
    echo "Error: PDF compilation failed."
    echo "Check pdflatex_run2.log for details."
    exit 1
fi

exit 0 