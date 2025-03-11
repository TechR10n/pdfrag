#!/bin/bash
# Simple script to set up LaTeX for Sphinx PDF generation on macOS

# Check if BasicTeX or MacTeX is installed
if ! command -v pdflatex &> /dev/null; then
    echo "LaTeX not found. Installing BasicTeX via Homebrew..."
    echo "This is the minimal TeX distribution needed for Sphinx PDF generation."
    read -p "Continue with installation? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if ! command -v brew &> /dev/null; then
            echo "Homebrew not found. Please install Homebrew first:"
            echo "/bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
            exit 1
        fi
        brew install --cask basictex
    else
        echo "Installation cancelled. You can install it manually later:"
        echo "brew install --cask basictex"
        echo "Or download the full MacTeX distribution from https://tug.org/mactex/"
        exit 1
    fi
fi

# Ensure the TeX bin directory is in PATH
if [[ ":$PATH:" != *":/Library/TeX/texbin:"* ]]; then
    echo "Adding /Library/TeX/texbin to PATH for this session"
    export PATH="/Library/TeX/texbin:$PATH"
fi

echo "Updating TeX Live package manager..."
sudo tlmgr update --self

echo "Installing required LaTeX packages for Sphinx..."
sudo tlmgr install \
    fncychap \
    tabulary \
    capt-of \
    needspace \
    framed \
    titlesec \
    varwidth \
    wrapfig \
    parskip \
    multirow \
    threeparttable \
    float \
    enumitem \
    babel-english \
    latexmk \
    cmap \
    collection-fontsrecommended \
    sphinx

echo "LaTeX setup complete! You can now generate PDF documentation with:"
echo "cd docs/sphinx && make latexpdf"
echo "or use the build_pdf.py script which will automatically use LaTeX if available." 