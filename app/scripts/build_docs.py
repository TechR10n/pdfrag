#!/usr/bin/env python3
"""
Script to build Sphinx documentation in different formats (HTML, PDF, Markdown).
"""

import os
import sys
import subprocess
import argparse
import shutil
from pathlib import Path

def run_command(command, cwd=None):
    """Run a shell command and print output."""
    try:
        result = subprocess.run(
            command, 
            check=True, 
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {' '.join(command)}")
        print(f"Error message: {e.stderr}")
        return False

def build_docs(format_type="html", clean=False, api_only=False):
    """Build Sphinx documentation in the specified format."""
    sphinx_dir = Path("docs/sphinx")
    
    if not sphinx_dir.exists():
        print(f"Error: Sphinx directory not found at {sphinx_dir}")
        print("Please run setup_sphinx.py first to set up the documentation.")
        return False
    
    # Check if make is available
    make_command = "make"
    if os.name == "nt":  # Windows
        if os.path.exists(sphinx_dir / "make.bat"):
            make_command = str(sphinx_dir / "make.bat")
        else:
            print("Warning: make.bat not found. Using 'make' command.")
    
    # Clean build directory if requested
    if clean:
        print("Cleaning build directory...")
        run_command([make_command, "clean"], cwd=sphinx_dir)
    
    # If API only, run sphinx-apidoc to update API documentation
    if api_only:
        print("Updating API documentation...")
        api_dir = sphinx_dir / "source" / "api"
        if not api_dir.exists():
            os.makedirs(api_dir)
        
        # Run sphinx-apidoc to generate API documentation
        # Use -M to put module documentation before member documentation
        # Use -e to put documentation for each module on its own page
        # Use -f to force overwriting existing files
        # Use -d 4 to set the maximum depth of the TOC
        # Use -P to include private members
        # Use --implicit-namespaces to handle namespace packages
        run_command([
            "sphinx-apidoc",
            "-o", str(api_dir),
            "-f", "-e", "-M", "-d", "4", "-P", "--implicit-namespaces",
            "app"  # Path to the package
        ])
        
        # Create a custom index.rst file for the API documentation
        with open(api_dir / "index.rst", "w") as f:
            f.write("""API Reference
============

This section contains the API reference for the PDF RAG System.

.. toctree::
   :maxdepth: 2

   modules
""")
        
        # Ensure utils modules are included
        utils_modules = [
            "app.utils.pdf_ingestion",
            "app.utils.text_chunking",
            "app.utils.vector_db"
        ]
        
        # Create individual module files for important modules if they don't exist
        for module_path in utils_modules:
            module_name = module_path.split(".")[-1]
            module_file = api_dir / f"{module_name}.rst"
            
            if not module_file.exists():
                print(f"Creating documentation for {module_path}...")
                with open(module_file, "w") as f:
                    f.write(f"""{module_name} module
{'=' * (len(module_name) + 7)}

.. automodule:: {module_path}
   :members:
   :undoc-members:
   :show-inheritance:
""")
        
        # Update modules.rst to include these modules
        modules_file = api_dir / "modules.rst"
        if modules_file.exists():
            with open(modules_file, "r") as f:
                content = f.read()
            
            # Check if we need to add the utils modules
            if "app.utils" not in content:
                # Find the toctree directive
                toctree_pos = content.find(".. toctree::")
                if toctree_pos != -1:
                    # Find the end of the toctree entries
                    lines = content.split("\n")
                    toctree_start = -1
                    toctree_end = -1
                    
                    for i, line in enumerate(lines):
                        if ".. toctree::" in line:
                            toctree_start = i
                        elif toctree_start != -1 and line and not line.startswith(" "):
                            toctree_end = i
                            break
                    
                    if toctree_end == -1:
                        toctree_end = len(lines)
                    
                    # Add the utils modules to the toctree
                    for module_path in utils_modules:
                        module_name = module_path.split(".")[-1]
                        lines.insert(toctree_end, f"   {module_name}")
                    
                    # Write the updated content
                    with open(modules_file, "w") as f:
                        f.write("\n".join(lines))
        
        # Create app.utils.rst if it doesn't exist
        utils_file = api_dir / "app.utils.rst"
        if not utils_file.exists():
            with open(utils_file, "w") as f:
                f.write("""app.utils package
==============

.. toctree::
   :maxdepth: 4

   pdf_ingestion
   text_chunking
   vector_db

""")
        
        print("API documentation updated.")
        return True
    
    # Build documentation in the specified format
    print(f"Building documentation in {format_type} format...")
    
    if format_type == "pdf":
        # For PDF, we need to run latexpdf
        # First, ensure all required LaTeX packages are installed
        if sys.platform == "darwin":  # macOS
            print("Checking LaTeX installation on macOS...")
            # Check if pdflatex is available
            try:
                result = subprocess.run(["pdflatex", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
                print(f"Found pdflatex: {result.stdout.decode().splitlines()[0]}")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("pdflatex not found. Please install MacTeX from https://tug.org/mactex/")
                return False
            except subprocess.TimeoutExpired:
                print("pdflatex check timed out. Continuing anyway...")
        
        print("Running LaTeX build with timeout protection...")
        
        # Initialize success variable
        success = False
        
        # Run latexpdf with a timeout to prevent hanging
        try:
            # First, try to build the LaTeX files without running pdflatex
            run_command([make_command, "latex", "SPHINXOPTS=-v"], cwd=sphinx_dir)
            
            # Now run pdflatex directly on the generated .tex file with a timeout
            latex_dir = sphinx_dir / "build/latex"
            tex_files = list(latex_dir.glob("*.tex"))
            
            if tex_files:
                main_tex_file = tex_files[0]
                print(f"Running pdflatex on {main_tex_file}...")
                
                # Run pdflatex twice to resolve references
                pdflatex_success = True
                for i in range(2):
                    try:
                        subprocess.run(
                            ["pdflatex", main_tex_file.name], 
                            cwd=latex_dir,
                            check=True, 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE,
                            timeout=60  # 60 second timeout
                        )
                    except subprocess.TimeoutExpired:
                        print(f"pdflatex run {i+1} timed out. PDF may be incomplete.")
                        pdflatex_success = False
                        break
                    except subprocess.CalledProcessError:
                        print(f"pdflatex run {i+1} failed. PDF may be incomplete.")
                        pdflatex_success = False
                        break
                
                # Check if PDF was generated
                pdf_files = list(latex_dir.glob("*.pdf"))
                if pdf_files:
                    print(f"PDF documentation built successfully: {pdf_files[0]}")
                    # Copy the PDF to a more accessible location
                    output_pdf = sphinx_dir / "build/pdfragsystem.pdf"
                    shutil.copy(pdf_files[0], output_pdf)
                    print(f"PDF copied to: {output_pdf}")
                    success = True
                else:
                    print("PDF generation failed. Check the LaTeX directory for errors.")
                    success = False
            else:
                print("No .tex files found in the LaTeX build directory.")
                success = False
                
        except Exception as e:
            print(f"Error during PDF generation: {str(e)}")
            print("Trying alternative approach...")
            success = False
            
            # Try the direct make latexpdf command as a fallback
            try:
                subprocess.run(
                    [make_command, "latexpdf"], 
                    cwd=sphinx_dir,
                    check=True, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    timeout=120  # 2 minute timeout
                )
                
                # Check for PDF files
                pdf_path = sphinx_dir / "build/latex/pdfragsystem.pdf"
                if pdf_path.exists():
                    print(f"PDF documentation built successfully: {pdf_path}")
                    success = True
                else:
                    # Try to find the PDF file
                    pdf_files = list(sphinx_dir.glob("build/latex/*.pdf"))
                    if pdf_files:
                        print(f"PDF documentation built successfully: {pdf_files[0]}")
                        success = True
                    else:
                        print(f"PDF file not found at expected location: {pdf_path}")
                        print("Check the latex directory for the generated PDF.")
                        success = False
            except subprocess.TimeoutExpired:
                print("PDF generation timed out after 2 minutes.")
                print("This could be due to a LaTeX package issue or a complex document.")
                print("Try running the following command manually:")
                print(f"cd {sphinx_dir} && make latexpdf")
                success = False
            except Exception as e:
                print(f"Error during alternative PDF generation: {str(e)}")
                success = False
    elif format_type == "markdown":
        # For Markdown, we need a custom target
        success = run_command([make_command, "markdown"], cwd=sphinx_dir)
        if success:
            md_dir = sphinx_dir / "build/markdown"
            if md_dir.exists():
                print(f"Markdown documentation built successfully in: {md_dir}")
            else:
                print(f"Markdown directory not found at expected location: {md_dir}")
    else:
        # For HTML and other formats, use the format name directly
        success = run_command([make_command, format_type], cwd=sphinx_dir)
        if success:
            build_dir = sphinx_dir / f"build/{format_type}"
            if build_dir.exists():
                print(f"{format_type.upper()} documentation built successfully in: {build_dir}")
                
                # For HTML, print the path to the index.html file
                if format_type == "html":
                    index_path = build_dir / "index.html"
                    if index_path.exists():
                        print(f"You can view the HTML documentation by opening: {index_path}")
                        print(f"Or by running: open {index_path}")
            else:
                print(f"{format_type.upper()} directory not found at expected location: {build_dir}")
    
    return success

def main():
    """Main function to parse arguments and build documentation."""
    parser = argparse.ArgumentParser(description="Build Sphinx documentation in different formats.")
    parser.add_argument(
        "--format", "-f",
        choices=["html", "pdf", "markdown", "epub"],
        default="html",
        help="Output format for the documentation (default: html)"
    )
    parser.add_argument(
        "--clean", "-c",
        action="store_true",
        help="Clean the build directory before building"
    )
    parser.add_argument(
        "--api-only", "-a",
        action="store_true",
        help="Only update API documentation, don't build"
    )
    
    args = parser.parse_args()
    
    build_docs(args.format, args.clean, args.api_only)

if __name__ == "__main__":
    main() 