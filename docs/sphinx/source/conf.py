# Configuration file for the Sphinx documentation builder.

import os
import sys
import shutil
import glob
import subprocess
from pathlib import Path
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
    'sphinxcontrib.bibtex',
    'sphinx.ext.imgconverter',
]

# Configuration for sphinxcontrib.bibtex
bibtex_bibfiles = []  # Empty list since we're not using bibliography files yet
bibtex_default_style = 'plain'
bibtex_reference_style = 'author_year'

# Try to add rinohtype extension if available
try:
    import rinoh
    extensions.append('rinoh.frontend.sphinx')
    print("rinohtype extension loaded successfully")
except ImportError:
    try:
        import rinohtype
        extensions.append('rinohtype.frontend.sphinx')
        print("rinohtype extension loaded successfully (alternative import)")
    except ImportError:
        print("Warning: rinohtype not available. PDF generation with rinohtype will be disabled.")

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

# -- Options for PDF output with rinohtype -----------------------------------
# Only configure if rinohtype is available
try:
    import rinoh
    rinoh_documents = [
        {
            'doc': 'index',
            'target': 'pdfragsystem',
            'title': 'PDF RAG System Documentation',
            'author': 'Ryan Hammang',
            'toctree_only': False,
        }
    ]
except ImportError:
    pass

# -- Options for LaTeX PDF output -------------------------------------------
latex_engine = 'pdflatex'
latex_elements = {
    'preamble': r'''
\usepackage{graphicx}
\usepackage[utf8]{inputenc}
\usepackage{xcolor}
\usepackage{fancyvrb}
\usepackage{tabulary}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{capt-of}
\usepackage{needspace}
\usepackage{hyperref}
''',
    'figure_align': 'H',
    'pointsize': '11pt',
    'papersize': 'letterpaper',
    'extraclassoptions': 'openany,oneside',
    'babel': r'\usepackage[english]{babel}',
    'maketitle': r'\maketitle',
    'tableofcontents': r'\tableofcontents',
    'fncychap': r'\usepackage[Bjarne]{fncychap}',
    'printindex': r'\printindex',
}

# -- Simple SVG handling for LaTeX output -----------------------------------
# This approach simply tells Sphinx to use PDF instead of SVG for LaTeX output
def setup(app):
    pass

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
