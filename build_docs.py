#!/usr/bin/env python3
"""
Build script for the PDF RAG System documentation.
This script handles the entire documentation build process:
1. Installing required dependencies
2. Converting SVG files to PDF for LaTeX compatibility
3. Building HTML documentation
4. Building PDF documentation using LaTeX
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import tempfile
import glob
import re

def ensure_dependencies():
    """Install required dependencies for documentation."""
    print("Checking and installing dependencies...")
    
    # Required packages list
    required_packages = [
        "sphinx==7.3.7",
        "sphinx-rtd-theme==2.0.0",
        "myst-parser==3.0.0",
        "rinohtype==0.5.4",
        "cairosvg==2.7.1"
    ]
    
    # Install required packages
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade"] + required_packages, check=True)
    
    # Check for LaTeX installation
    have_latex = shutil.which('pdflatex') is not None
    if not have_latex:
        print("LaTeX not found. PDF output will use rinohtype instead.")
        print("For higher quality PDFs, install LaTeX:")
        print("  - MacOS: brew install --cask mactex")
        print("  - Ubuntu: apt-get install texlive-latex-extra")
        print("  - Windows: Install MiKTeX or TeX Live")

def find_svg_files():
    """Find all SVG files in the documentation directories."""
    print("Finding SVG files...")
    
    # Add all directories that might contain SVG files
    search_dirs = [
        Path("docs/sphinx/source"),
        Path("docs/puml/svg"),
        Path("puml/svg")
    ]
    
    svg_files = []
    for directory in search_dirs:
        if directory.exists():
            svg_files.extend(list(directory.glob("**/*.svg")))
    
    print(f"Found {len(svg_files)} SVG files")
    return svg_files

def prepare_static_directory():
    """Create and prepare the static directory for images."""
    static_dir = Path("docs/sphinx/source/_static")
    if not static_dir.exists():
        static_dir.mkdir(parents=True)
    return static_dir

def convert_svg_to_pdf(svg_files, static_dir):
    """Convert SVG files to PDF format for LaTeX compatibility."""
    print("Converting SVG files to PDF for LaTeX...")
    
    # Try to import cairosvg
    try:
        import cairosvg
        have_cairosvg = True
    except ImportError:
        have_cairosvg = False
        print("Warning: cairosvg not available, will try Inkscape instead")
    
    # Process each SVG file
    for svg_file in svg_files:
        # Get the base name for the PDF file
        pdf_filename = svg_file.stem + '.pdf'
        pdf_path = static_dir / pdf_filename
        
        print(f"Converting {svg_file} to {pdf_path}")
        
        # Try conversion methods in order of preference
        converted = False
        
        # Method 1: Inkscape (highest quality)
        if shutil.which('inkscape') and not converted:
            try:
                result = subprocess.run(
                    ['inkscape', '--export-filename', str(pdf_path), str(svg_file)],
                    capture_output=True,
                    check=True
                )
                converted = True
            except subprocess.CalledProcessError as e:
                print(f"Inkscape conversion failed: {e}")
        
        # Method 2: CairoSVG (fallback)
        if have_cairosvg and not converted:
            try:
                cairosvg.svg2pdf(url=str(svg_file), write_to=str(pdf_path))
                converted = True
            except Exception as e:
                print(f"CairoSVG conversion failed: {e}")
        
        if not converted:
            print(f"WARNING: Could not convert {svg_file} to PDF!")

def cleanup_latex_md_files():
    """Remove all *_latex.md files as they are no longer needed."""
    print("Cleaning up redundant LaTeX markdown files...")
    source_dir = Path("docs/sphinx/source")
    for latex_md in source_dir.glob("*_latex.md"):
        print(f"Removing {latex_md}")
        latex_md.unlink()

def modify_conf_for_unified_build():
    """Modify conf.py to properly handle markdown for both HTML and LaTeX."""
    conf_path = Path("docs/sphinx/source/conf.py")
    
    # Check if conf.py exists
    if not conf_path.exists():
        print(f"ERROR: {conf_path} not found!")
        return
    
    # Read the existing conf.py content
    with open(conf_path, "r") as f:
        conf_content = f.read()
    
    # Add the necessary configurations if not already present
    updates_needed = []
    
    if "sphinx.ext.imgconverter" not in conf_content:
        updates_needed.append("\n# Add imgconverter for SVG handling in LaTeX output")
        updates_needed.append("extensions.append('sphinx.ext.imgconverter')")
    
    if "latex_engine = 'pdflatex'" not in conf_content:
        updates_needed.append("\n# LaTeX configuration")
        updates_needed.append("latex_engine = 'pdflatex'")
    
    if "latex_elements =" not in conf_content:
        updates_needed.append("latex_elements = {")
        updates_needed.append("    'preamble': r'''")
        updates_needed.append("\\usepackage{graphicx}")
        updates_needed.append("\\usepackage{adjustbox}")
        updates_needed.append("    '''")
        updates_needed.append("}")
    
    # Add the patch for handling markdown image paths in LaTeX
    if "markdown_image_paths = " not in conf_content:
        updates_needed.append("\n# Fix image paths for markdown in LaTeX output")
        updates_needed.append("def setup(app):")
        updates_needed.append("    app.connect('source-read', process_markdown_images)")
        updates_needed.append("")
        updates_needed.append("def process_markdown_images(app, docname, source):")
        updates_needed.append("    if app.builder.format == 'latex':")
        updates_needed.append("        source[0] = source[0].replace('](../../puml/svg/', '](_static/')")
    
    # Apply the updates if needed
    if updates_needed:
        with open(conf_path, "a") as f:
            for line in updates_needed:
                f.write(f"{line}\n")
        print("Updated conf.py with unified build configuration")
    else:
        print("conf.py already configured for unified build")

def build_documentation():
    """Build the documentation using Sphinx."""
    sphinx_dir = Path("docs/sphinx")
    source_dir = sphinx_dir / "source"
    build_dir = sphinx_dir / "build"
    
    # Create build directory if it doesn't exist
    if not build_dir.exists():
        build_dir.mkdir(parents=True)
    
    # Build HTML documentation
    print("\nBuilding HTML documentation...")
    subprocess.run(
        ["sphinx-build", "-b", "html", str(source_dir), str(build_dir / "html")],
        check=True
    )
    print("✓ HTML documentation built successfully!")
    
    # Build PDF documentation
    print("\nBuilding PDF documentation...")
    
    # First clean the LaTeX build directory if it exists
    latex_dir = build_dir / "latex"
    if latex_dir.exists():
        shutil.rmtree(latex_dir)
    
    # Build LaTeX files
    subprocess.run(
        ["sphinx-build", "-b", "latex", str(source_dir), str(latex_dir)],
        check=True
    )
    
    # Check if pdflatex is available
    if shutil.which('pdflatex'):
        print("Using LaTeX to build PDF...")
        
        # Copy PDF files to LaTeX build directory for direct access
        static_dir = source_dir / "_static"
        for pdf_file in static_dir.glob("*.pdf"):
            pdf_dest = latex_dir / pdf_file.name
            print(f"Copying {pdf_file} to {pdf_dest}")
            shutil.copy(pdf_file, pdf_dest)
        
        # Save the original directory
        original_dir = os.getcwd()
        
        # Change to the LaTeX directory to run the commands
        os.chdir(latex_dir)
        try:
            # First try with latexmk (best approach)
            subprocess.run(["latexmk", "-pdf", "pdfragsystem.tex"], check=True)
        except subprocess.CalledProcessError:
            # Fall back to manual pdflatex runs
            print("latexmk failed, trying manual pdflatex...")
            for _ in range(3):  # Multiple runs to resolve references
                subprocess.run(["pdflatex", "-interaction=nonstopmode", "pdfragsystem.tex"], check=True)
        
        # Change back to the original directory
        os.chdir(original_dir)
    else:
        # Fall back to rinohtype if LaTeX is not available
        print("LaTeX not available, using rinohtype instead...")
        subprocess.run(
            ["sphinx-build", "-b", "rinoh", str(source_dir), str(build_dir / "rinoh")],
            check=True
        )
    
    # Print location of output files
    print("\nDocumentation build complete!")
    print(f"HTML: {build_dir}/html/index.html")
    
    # Check for PDF file in all possible locations
    pdf_locations = [
        build_dir / "pdfragsystem.pdf",
        latex_dir / "pdfragsystem.pdf", 
        build_dir / "rinoh" / "pdfragsystem.pdf"
    ]
    
    pdf_path = None
    for loc in pdf_locations:
        if loc.exists():
            pdf_path = loc
            break
    
    if pdf_path:
        # If PDF exists but not in build root, copy it there for easier access
        if pdf_path != build_dir / "pdfragsystem.pdf":
            shutil.copy(pdf_path, build_dir / "pdfragsystem.pdf")
            pdf_path = build_dir / "pdfragsystem.pdf"
        print(f"PDF: {pdf_path}")
    else:
        print("⚠ PDF file not found. Check LaTeX build logs for errors.")

def main():
    """Main function for the documentation build process."""
    print("=== PDF RAG System Documentation Builder ===")
    
    # Ensure dependencies are installed
    ensure_dependencies()
    
    # Clean up redundant LaTeX markdown files
    cleanup_latex_md_files()
    
    # Find SVG files
    svg_files = find_svg_files()
    
    # Prepare static directory for images
    static_dir = prepare_static_directory()
    
    # Convert SVG files to PDF
    if svg_files:
        convert_svg_to_pdf(svg_files, static_dir)
    
    # Modify Sphinx configuration for unified builds
    modify_conf_for_unified_build()
    
    # Build the documentation
    build_documentation()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 