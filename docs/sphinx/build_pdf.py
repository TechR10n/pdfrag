#!/usr/bin/env python3
"""
Build Sphinx documentation with optimal settings for macOS.
This script checks for LaTeX installation and uses the best available PDF generator.
"""

import os
import sys
import subprocess
import shutil
import importlib.util
from pathlib import Path

def check_dependencies():
    """Verify that all required dependencies are installed."""
    print("Checking dependencies...")
    
    # Check Sphinx dependencies
    python_deps = {
        "sphinx": "Core documentation generator",
        "sphinx_rtd_theme": "Read the Docs theme",
        "myst_parser": "Markdown support",
        "sphinx_markdown_tables": "Markdown tables support",
    }
    
    # Check for rinohtype separately with special handling
    has_rinohtype = False
    try:
        # Try different import approaches
        try:
            import rinoh
            has_rinohtype = True
            print(f"✓ rinohtype: Alternative PDF generator (fallback)")
        except ImportError:
            try:
                import rinohtype
                has_rinohtype = True
                print(f"✓ rinohtype: Alternative PDF generator (fallback)")
            except ImportError:
                # Check if the module exists but isn't importable directly
                if importlib.util.find_spec("rinoh") or importlib.util.find_spec("rinohtype"):
                    has_rinohtype = True
                    print(f"✓ rinohtype: Alternative PDF generator (fallback)")
                else:
                    print(f"✗ rinohtype: Alternative PDF generator (fallback)")
    except Exception as e:
        print(f"✗ rinohtype: Alternative PDF generator (fallback) - Error: {e}")
    
    # Check other dependencies
    try:
        import sphinxcontrib.bibtex
        print(f"✓ sphinxcontrib.bibtex: Bibliography support")
    except ImportError:
        print(f"✗ sphinxcontrib.bibtex: Bibliography support")
    
    missing = []
    for dep, desc in python_deps.items():
        try:
            module_name = dep.replace("-", "_")
            __import__(module_name)
            print(f"✓ {dep}: {desc}")
        except ImportError:
            missing.append(dep)
            print(f"✗ {dep}: {desc}")
    
    # Check for LaTeX installation
    has_latex = shutil.which("pdflatex") is not None
    if has_latex:
        print("✓ pdflatex: Found LaTeX installation (preferred for PDF generation)")
    else:
        print("✗ pdflatex: LaTeX not found (will use rinohtype for PDF generation)")
        # If no LaTeX and no rinohtype, that's a problem
        if not has_rinohtype:
            missing.append("rinohtype")
    
    if missing:
        print("\nMissing Python dependencies. Install with:")
        print("pip install -r ../../requirements.txt")
        # Add direct installation command for rinohtype if it's missing
        if "rinohtype" in missing:
            print("\nOr install rinohtype directly:")
            print("pip install rinohtype==0.5.4")
        return False
    
    return True

def check_latex_deps():
    """Check if necessary LaTeX packages are available (macOS TeX Live/MacTeX)."""
    if not shutil.which("pdflatex"):
        return False
    
    print("\nChecking LaTeX packages...")
    packages_to_check = [
        "fncychap",
        "tabulary",
        "capt-of",
        "needspace",
        "sphinx"
    ]
    
    missing_packages = []
    for package in packages_to_check:
        # Use kpsewhich to check if the package is installed
        result = subprocess.run(
            ["kpsewhich", f"{package}.sty"], 
            capture_output=True, 
            text=True
        )
        if result.returncode != 0 or not result.stdout.strip():
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing LaTeX packages: {', '.join(missing_packages)}")
        print("\nTo install missing packages:")
        print("sudo tlmgr install " + " ".join(missing_packages))
        return False
    
    return True

def build_docs():
    """Build documentation using the Makefile."""
    print("\nBuilding documentation...")
    
    # Check if we have LaTeX or rinohtype
    has_latex = shutil.which("pdflatex") is not None
    has_rinohtype = False
    try:
        import rinoh
        has_rinohtype = True
    except ImportError:
        try:
            import rinohtype
            has_rinohtype = True
        except ImportError:
            pass
    
    # Determine which build command to use
    if has_latex:
        print("Using LaTeX for PDF generation (high quality)...")
        build_cmd = ["make", "alldocs"]
    elif has_rinohtype:
        print("Using rinohtype for PDF generation...")
        build_cmd = ["make", "alldocs"]
    else:
        print("No PDF generator available. Building HTML only...")
        build_cmd = ["make", "html"]
    
    # Run the build command
    result = subprocess.run(build_cmd, check=False)
    
    if result.returncode == 0:
        print("\nDocumentation built successfully!")
        
        # Check which PDF was generated
        latex_pdf = Path("build/latex/pdfragsystem.pdf")
        rinoh_pdf = Path("build/rinoh/pdfragsystem.pdf")
        
        if latex_pdf.exists():
            pdf_path = latex_pdf
            pdf_type = "LaTeX (high quality)"
        elif rinoh_pdf.exists():
            pdf_path = rinoh_pdf
            pdf_type = "rinohtype"
        else:
            print("\nDocumentation is available at:")
            print(f"- HTML: build/html/index.html")
            print("- PDF: Not generated")
            return True
        
        print(f"\nDocumentation is available at:")
        print(f"- HTML: build/html/index.html")
        print(f"- PDF ({pdf_type}): {pdf_path}")
        return True
    else:
        print("\nError building documentation.")
        print("Check the output above for errors.")
        return False

if __name__ == "__main__":
    # Change to the directory containing this script
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print("=" * 80)
    print(" PDF RAG System Documentation Builder ".center(80, "="))
    print("=" * 80)
    
    # Check dependencies first
    if not check_dependencies():
        print("\nPlease install missing dependencies before proceeding.")
        sys.exit(1)
    
    # Check LaTeX packages if LaTeX is installed
    if shutil.which("pdflatex"):
        check_latex_deps()
    
    # Build the documentation
    if build_docs():
        print("\nDocumentation build completed!")
        sys.exit(0)
    else:
        print("\nDocumentation build failed.")
        sys.exit(1) 