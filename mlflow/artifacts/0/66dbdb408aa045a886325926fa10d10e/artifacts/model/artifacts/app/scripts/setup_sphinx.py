#!/usr/bin/env python3
"""
Script to set up Sphinx documentation for the project.
This script creates the necessary configuration files and directory structure.
"""

import os
import sys
import subprocess
from pathlib import Path
import shutil
import glob
import importlib
import pkgutil

def create_directory(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

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

def find_python_modules(base_path="app"):
    """Find all Python modules in the project."""
    modules = []
    base_path = Path(base_path)
    
    if not base_path.exists():
        print(f"Warning: Path {base_path} does not exist.")
        return modules
    
    for root, dirs, files in os.walk(base_path):
        # Skip __pycache__ directories
        if "__pycache__" in root:
            continue
        
        # Process Python files
        for file in files:
            if file.endswith(".py") and file != "__init__.py":
                file_path = Path(root) / file
                # Use the file path directly without trying to make it relative to cwd
                module_path = str(file_path).replace("/", ".").replace("\\", ".").replace(".py", "")
                modules.append(module_path)
    
    return modules

def generate_api_docs(modules, output_dir):
    """Generate API documentation for the given modules."""
    # Create API directory
    api_dir = output_dir / "api"
    create_directory(api_dir)
    
    # Create API index file
    api_index_content = """API Reference
============

This section contains the API reference for the PDF RAG System.

.. toctree::
   :maxdepth: 2

   modules
"""
    
    # Write API index file
    with open(api_dir / "index.rst", "w") as f:
        f.write(api_index_content)
    
    # Run sphinx-apidoc to generate module documentation
    run_command([
        "sphinx-apidoc",
        "-o", str(api_dir),
        "-f", "-e", "-M", "-d", "4",  # Force, separate modules, module first, max depth 4
        "app"  # Path to the package
    ])
    
    return api_dir / "index.rst"

def setup_sphinx_docs():
    """Set up Sphinx documentation for the project."""
    # Create docs directory structure
    docs_dir = Path("docs")
    sphinx_dir = docs_dir / "sphinx"
    source_dir = sphinx_dir / "source"
    build_dir = sphinx_dir / "build"
    
    create_directory(docs_dir)
    create_directory(sphinx_dir)
    create_directory(source_dir)
    create_directory(build_dir)
    create_directory(source_dir / "_static")
    create_directory(source_dir / "_templates")
    
    # Initialize Sphinx with sphinx-quickstart
    print("Initializing Sphinx documentation...")
    
    # Check if sphinx-quickstart is available
    try:
        subprocess.run(["sphinx-quickstart", "--version"], check=True, stdout=subprocess.PIPE)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("sphinx-quickstart not found. Installing Sphinx...")
        run_command([
            "pip", "install", 
            "sphinx", "sphinx-rtd-theme", "recommonmark", 
            "sphinx-markdown-tables", "myst-parser"
        ])
    
    # Find Python modules
    print("Finding Python modules...")
    modules = find_python_modules()
    print(f"Found {len(modules)} Python modules.")
    
    # Create conf.py
    conf_py_content = """# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('../../..'))

# -- Project information -----------------------------------------------------
project = 'PDF RAG System'
copyright = '2025, Ryan Hammang'
author = 'Ryan Hammang'
release = '1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx_markdown_tables',
    'myst_parser',
]

# Auto-generate API documentation
autosummary_generate = True
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'special-members': '__init__',
    'imported-members': True,
    'private-members': True,
}

# Make sure autodoc can find the modules
autodoc_mock_imports = []
autoclass_content = 'both'
autodoc_typehints = 'description'
autodoc_typehints_format = 'short'
autodoc_member_order = 'bysource'

# MyST Parser settings
myst_enable_extensions = [
    'colon_fence',
    'deflist',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {
    'navigation_depth': 4,
    'titles_only': False,
    'display_version': True,
}

# -- Options for Markdown support --------------------------------------------
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Options for PDF output --------------------------------------------------
latex_engine = 'pdflatex'
latex_elements = {
    'preamble': r'''
\\usepackage{underscore}
\\usepackage{graphicx}
\\usepackage[utf8]{inputenc}
\\usepackage{xcolor}
\\usepackage{fancyvrb}
\\usepackage{tabulary}
\\usepackage{amsmath}
\\usepackage{amssymb}
\\usepackage{capt-of}
\\usepackage{needspace}
\\usepackage{hyperref}
''',
    'figure_align': 'H',
    'pointsize': '11pt',
    'papersize': 'letterpaper',
    'extraclassoptions': 'openany,oneside',
    'babel': r'\\usepackage[english]{babel}',
    'maketitle': r'\\maketitle',
    'tableofcontents': r'\\tableofcontents',
    'fncychap': r'\\usepackage[Bjarne]{fncychap}',
    'printindex': r'\\printindex',
}

# -- Extension configuration -------------------------------------------------
intersphinx_mapping = {'python': ('https://docs.python.org/3', None)}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True
napoleon_custom_sections = None
"""
    
    with open(source_dir / "conf.py", "w") as f:
        f.write(conf_py_content)
    
    # Generate API documentation
    print("Generating API documentation...")
    api_index = generate_api_docs(modules, source_dir)
    
    # Create index.rst
    index_rst_content = """PDF RAG System Documentation
============================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   installation
   usage
   architecture
   api/index
   development
   appendix

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
"""
    
    with open(source_dir / "index.rst", "w") as f:
        f.write(index_rst_content)
    
    # Create introduction.md
    introduction_md_content = """# Introduction

This documentation covers the PDF RAG (Retrieval Augmented Generation) System, a local system for building and querying a knowledge base from PDF documents.

## Overview

The PDF RAG System allows users to:

1. Upload PDF documents to create a knowledge base
2. Query the knowledge base using natural language
3. Receive accurate responses with citations to the source documents
4. Manage and update the knowledge base

This system is built using Flask for the web interface and MLflow for experiment tracking and model management.
"""
    
    with open(source_dir / "introduction.md", "w") as f:
        f.write(introduction_md_content)
    
    # Create installation.md
    installation_md_content = """# Installation

This section covers how to install and set up the PDF RAG System.

## Requirements

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

## Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pdfrag.git
   cd pdfrag
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up the configuration:
   ```bash
   cp app/config/config.example.py app/config/config.py
   # Edit app/config/config.py with your settings
   ```

5. Run the application:
   ```bash
   python app/main.py
   ```

The application should now be running at http://localhost:5000.
"""
    
    with open(source_dir / "installation.md", "w") as f:
        f.write(installation_md_content)
    
    # Create usage.md
    usage_md_content = """# Usage Guide

This section provides a guide on how to use the PDF RAG System.

## Uploading Documents

1. Navigate to the Upload page
2. Select PDF files from your computer
3. Click the Upload button
4. Wait for the processing to complete

## Querying the Knowledge Base

1. Navigate to the Query page
2. Enter your question in the text box
3. Click the Submit button
4. View the response with citations

## Managing the Knowledge Base

1. Navigate to the Management page
2. View all uploaded documents
3. Remove documents if needed
4. Update the knowledge base after changes
"""
    
    with open(source_dir / "usage.md", "w") as f:
        f.write(usage_md_content)
    
    # Create architecture.md with PUML diagrams
    architecture_md_content = """# Architecture and Design

This section describes the architecture and design of the PDF RAG System.

## System Architecture

The system is composed of several components that work together to provide the RAG functionality.

"""
    
    # Find PUML files and add them to architecture.md
    puml_files = glob.glob("docs/puml/svg/*.svg")
    for puml_file in puml_files:
        diagram_name = os.path.basename(puml_file).replace(".svg", "")
        diagram_title = diagram_name.replace("_", " ").title()
        architecture_md_content += f"""
### {diagram_title} Diagram

![{diagram_title} Diagram]({os.path.relpath(puml_file, source_dir)})

"""
    
    with open(source_dir / "architecture.md", "w") as f:
        f.write(architecture_md_content)
    
    # Create development.md
    development_md_content = """# Development Guide

This document contains development notes and additional information for developers working on the PDF RAG System.

## Project Structure

```
project/
├── app/
│   ├── api/
│   ├── models/
│   ├── scripts/
│   ├── static/
│   └── templates/
├── docs/
│   ├── puml/
│   └── sphinx/
└── tests/
```

## Development Environment

### Prerequisites

- Python 3.8+
- pip
- virtualenv or conda

### Setup

1. Clone the repository
2. Create a virtual environment
3. Install dependencies
4. Run the development server

## Coding Standards

- Follow PEP 8 for Python code
- Use docstrings for all functions and classes
- Write unit tests for new features

## Deployment

### Local Deployment

Instructions for local deployment...

### Production Deployment

Instructions for production deployment...

## Future Improvements

- List of planned improvements and features
- Known limitations and how they might be addressed

"""
    
    with open(source_dir / "development.md", "w") as f:
        f.write(development_md_content)
    
    # Create appendix.md that includes dev_notes.md
    appendix_md_content = """# Appendix

This appendix contains additional information and resources.

## Development Notes

```{include} dev_notes.md
```

## References

- [Flask Documentation](https://flask.palletsprojects.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)

"""
    
    with open(source_dir / "appendix.md", "w") as f:
        f.write(appendix_md_content)
    
    # Create Makefile
    makefile_content = """# Minimal makefile for Sphinx documentation

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean markdown

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf $(BUILDDIR)/*

markdown:
	@$(SPHINXBUILD) -M markdown "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
"""
    
    with open(sphinx_dir / "Makefile", "w") as f:
        f.write(makefile_content)
    
    # Create make.bat for Windows
    make_bat_content = """@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=source
set BUILDDIR=build

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.https://www.sphinx-doc.org/
	exit /b 1
)

if "%1" == "" goto help
if "%1" == "clean" goto clean
if "%1" == "markdown" goto markdown

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:clean
rmdir /s /q %BUILDDIR%
goto end

:markdown
%SPHINXBUILD% -M markdown %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:end
popd
"""
    
    with open(sphinx_dir / "make.bat", "w") as f:
        f.write(make_bat_content)
    
    print("Sphinx documentation setup complete!")
    print(f"Documentation source files are in: {source_dir}")
    print("To build the documentation:")
    print(f"  cd {sphinx_dir}")
    print("  make html    # For HTML output")
    print("  make latexpdf  # For PDF output")
    print("  make markdown  # For Markdown output")

if __name__ == "__main__":
    setup_sphinx_docs() 