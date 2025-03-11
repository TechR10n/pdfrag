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
                print(f"  ✓ Converted using Inkscape")
            except subprocess.CalledProcessError as e:
                print(f"  ✗ Inkscape failed: {e}")
        
        # Method 2: cairosvg (good fallback)
        if have_cairosvg and not converted:
            try:
                cairosvg.svg2pdf(url=str(svg_file), write_to=str(pdf_path))
                converted = True
                print(f"  ✓ Converted using cairosvg")
            except Exception as e:
                print(f"  ✗ cairosvg failed: {e}")
        
        # Check if conversion succeeded
        if not converted:
            print(f"  ⚠ WARNING: Could not convert {svg_file} to PDF")
        elif not pdf_path.exists():
            print(f"  ⚠ WARNING: PDF file {pdf_path} was not created")

def modify_latex_configuration():
    """Add SVG handling to conf.py for proper LaTeX output."""
    conf_path = Path("docs/sphinx/source/conf.py")
    
    # Check if conf.py exists
    if not conf_path.exists():
        print(f"ERROR: {conf_path} not found!")
        return
    
    # Read the existing conf.py content
    with open(conf_path, "r") as f:
        conf_content = f.read()
    
    # Check if the LaTeX handling is already there
    if "sphinx.ext.imgconverter" in conf_content:
        print("LaTeX image handling already configured in conf.py")
        return
    
    # Add SVG handling for LaTeX output
    with open(conf_path, "a") as f:
        f.write("\n\n# Add imgconverter for SVG handling in LaTeX output\n")
        f.write("extensions.append('sphinx.ext.imgconverter')\n")
        f.write("\n# LaTeX configuration for SVG handling\n")
        f.write("latex_engine = 'pdflatex'\n")
        f.write("latex_elements = {\n")
        f.write("    'preamble': r'''\n")
        f.write("\\usepackage{graphicx}\n")
        f.write("'''\n")
        f.write("}\n")
    
    print("Updated conf.py with LaTeX image handling configuration")

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
        
        # Run LaTeX to build PDF (with multiple passes for references)
        os.chdir(latex_dir)
        try:
            # First try with latexmk (best approach)
            subprocess.run(["latexmk", "-pdf", "pdfragsystem.tex"], check=True)
        except subprocess.CalledProcessError:
            # Fall back to manual pdflatex runs
            print("latexmk failed, trying manual pdflatex...")
            for _ in range(3):  # Multiple runs to resolve references
                subprocess.run(["pdflatex", "-interaction=nonstopmode", "pdfragsystem.tex"], check=True)
        
        # Copy resulting PDF to the build root for easy access
        if (latex_dir / "pdfragsystem.pdf").exists():
            shutil.copy(latex_dir / "pdfragsystem.pdf", build_dir / "pdfragsystem.pdf")
            print("✓ PDF documentation built with LaTeX!")
        else:
            print("⚠ PDF file not found after LaTeX build")
    else:
        # Fall back to rinohtype if LaTeX is not available
        print("LaTeX not available, using rinohtype instead...")
        subprocess.run(
            ["sphinx-build", "-b", "rinoh", str(source_dir), str(build_dir / "rinoh")],
            check=True
        )
        print("✓ PDF documentation built with rinohtype!")
    
    # Print location of output files
    print("\nDocumentation build complete!")
    print(f"HTML: {build_dir}/html/index.html")
    if (build_dir / "pdfragsystem.pdf").exists():
        print(f"PDF: {build_dir}/pdfragsystem.pdf")
    elif (build_dir / "rinoh" / "pdfragsystem.pdf").exists():
        print(f"PDF: {build_dir}/rinoh/pdfragsystem.pdf")

def main():
    """Main function for the documentation build process."""
    print("=== PDF RAG System Documentation Builder ===")
    
    # Ensure dependencies are installed
    ensure_dependencies()
    
    # Find SVG files
    svg_files = find_svg_files()
    
    # Prepare static directory for images
    static_dir = prepare_static_directory()
    
    # Convert SVG files to PDF
    if svg_files:
        convert_svg_to_pdf(svg_files, static_dir)
    
    # Modify Sphinx configuration for LaTeX
    modify_latex_configuration()
    
    # Build the documentation
    build_documentation()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 