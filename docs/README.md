# PDF RAG System Documentation

This directory contains the documentation for the PDF RAG System.

## Building Documentation

To build the documentation, simply run:

```bash
# From the project root
./build_docs.py
```

This script will:
1. Install all required dependencies
2. Convert SVG files to PDF format (for LaTeX compatibility)
3. Build HTML documentation
4. Build PDF documentation

## Documentation Outputs

The documentation is built in two formats:

- **HTML**: Interactive web-based documentation
  - Located at: `docs/sphinx/build/html/index.html`
  
- **PDF**: Printable documentation 
  - Located at: `docs/sphinx/build/pdfragsystem.pdf`

## Adding Documentation

To add new documentation:

1. Create Markdown (`.md`) or reStructuredText (`.rst`) files in `docs/sphinx/source/`
2. Add references to new files in `docs/sphinx/source/index.rst`

## Including Images

When adding images to documentation:

- **SVG images**: Place in `docs/sphinx/source/_static/` 
  - Use standard Markdown or RST image syntax
  - Example: `![Alt text](/_static/image.svg)`
  - The build script handles SVG-to-PDF conversion automatically

- **Other image formats**: Place in `docs/sphinx/source/_static/`
  - Supported formats include: PNG, JPG, GIF

## Troubleshooting

If you encounter issues with the documentation build:

1. **LaTeX errors**: Make sure you have LaTeX installed
   - The script will fall back to rinohtype if LaTeX is not available
   - For higher quality PDFs, install LaTeX:
     - MacOS: `brew install --cask mactex`
     - Ubuntu: `apt-get install texlive-latex-extra`
     - Windows: Install MiKTeX or TeX Live

2. **SVG conversion issues**: 
   - The script uses Inkscape (preferred) or cairosvg
   - Install Inkscape for best results: `brew install inkscape`

3. **Missing dependencies**:
   - The script automatically installs required Python packages
   - If you get import errors, try running: `pip install -r requirements.txt`

## Documentation Formats

The documentation system supports multiple output formats:

1. **HTML** - Interactive web-based documentation
2. **PDF** - High-quality printable documentation

## Installation Options

### Option 1: Direct Installation (Recommended)

Use the provided script to install all Sphinx dependencies directly:

```bash
./install_sphinx_deps.sh
```

This script installs all required packages including rinohtype and its dependencies.

### Option 2: Using requirements.txt

Alternatively, you can install from the project's requirements.txt:

```bash
pip install -r ../requirements.txt
```

### MacOS-Specific Setup (Optional)

For optimal PDF generation on macOS:

1. Run the setup script to install BasicTeX and required packages:
   ```bash
   ./setup_macos_tex.sh
   ```

2. Install Inkscape for SVG to PDF conversion (optional but recommended):
   ```bash
   brew install inkscape
   ```

## Building Documentation

### From Command Line

Navigate to the Sphinx directory:

```bash
cd sphinx
```

To build HTML documentation only:

```bash
make html
```

To build PDF documentation:

```bash
# Using LaTeX (preferred if installed):
make latexpdf

# Using rinohtype (fallback, no LaTeX required):
make rinohpdf

# Automatically choose the best method:
make pdf
```

To build all formats (HTML + PDF):

```bash
make alldocs
```

### Using the Helper Script

The provided Python script automates the building process:

```bash
cd sphinx
python build_pdf.py
```

This script:
1. Checks for required dependencies
2. Validates LaTeX installation (if present)
3. Builds documentation using the best available method
4. Reports where to find the output files

## Viewing Documentation

After building, you can find the documentation at:

- HTML: `sphinx/build/html/index.html`
- PDF (LaTeX): `sphinx/build/latex/pdfragsystem.pdf`
- PDF (rinohtype): `sphinx/build/rinoh/pdfragsystem.pdf`

## Maintenance

The documentation is built using [Sphinx](https://www.sphinx-doc.org/). The configuration is in `sphinx/source/conf.py`.

To add new documentation pages:

1. Create new `.rst` or `.md` files in the `sphinx/source/` directory
2. Add them to the table of contents in `sphinx/source/index.rst`

## SVG Images

When using SVG images in your documentation:

- For HTML output: SVGs are used directly
- For PDF output: SVGs are automatically converted to PDF format
  - The build script handles this conversion automatically
  - No manual steps are required 