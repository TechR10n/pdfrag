# PDF Generation with Custom LaTeX Fixes

This directory contains scripts to generate PDF documentation with custom LaTeX fixes for the PDF RAG System.

## Background

The standard Sphinx PDF generation process using `make latexpdf` encounters LaTeX errors due to command redefinitions and other issues. To address this, we've created a two-step process:

1. Generate only the LaTeX files without attempting to build the PDF
2. Apply fixes to the LaTeX files and then compile them using the local macOS pdflatex

## Available Scripts

### `build_pdf_fixed.sh`

This is the main script that runs the entire process in one go:
- Generates LaTeX files using Sphinx
- Applies fixes to the LaTeX files
- Compiles the fixed LaTeX files using pdflatex

Usage:
```bash
./build_pdf_fixed.sh
```

### `fix_and_build_latex.sh`

This script only performs the second part of the process:
- Applies fixes to the LaTeX files
- Compiles the fixed LaTeX files using pdflatex

Usage:
```bash
./fix_and_build_latex.sh
```

### Makefile Target: `latexonly`

A new Makefile target that only generates the LaTeX files without attempting to build the PDF:

```bash
make latexonly
```

## Fixes Applied

The `fix_and_build_latex.sh` script applies the following fixes to the LaTeX files:

1. **\Bbbk Command Redefinition**: Prevents the error "LaTeX Error: Command `\Bbbk' already defined"
2. **Other Command Redefinitions**: Prevents similar errors with other commands like \openbox and \mathbb
3. **Hyperref Unicode Issues**: Adds proper Unicode support for hyperref
4. **Font Issues**: Adds declarations for text commands that might be missing

## Troubleshooting

If you encounter issues during PDF generation:

1. Check the log files in the `build/latex` directory:
   - `pdflatex_run1.log` for errors in the first pdflatex run
   - `pdflatex_run2.log` for errors in the second pdflatex run

2. If you need to add more fixes to the LaTeX files, edit the `fix_and_build_latex.sh` script and add more `sed` commands to apply the necessary fixes.

3. Make sure you have a complete TeX installation on your macOS system. You can install it using:
   ```bash
   brew install --cask mactex
   ```
   or the smaller version:
   ```bash
   brew install --cask basictex
   ``` 