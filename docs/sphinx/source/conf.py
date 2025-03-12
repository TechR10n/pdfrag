# Configuration file for the Sphinx documentation builder.

import os
import sys
from pathlib import Path

# Point to the project root (assuming conf.py is in docs/sphinx/source)
sys.path.insert(0, os.path.abspath('../..'))  # Adjust if needed

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
    'sphinx.ext.autosummary',
    'sphinx_markdown_tables',
    'myst_parser',
    'sphinxcontrib.bibtex',
    'sphinx.ext.imgconverter',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
]

# Configuration for sphinxcontrib.bibtex
bibtex_bibfiles = ['references.bib']
bibtex_default_style = 'plain'
bibtex_reference_style = 'author_year'

# Auto-generate API documentation
autosummary_generate = True
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': True,
    'special-members': '__init__,__call__',
    'show-inheritance': True,
}
autodoc_mock_imports = ['numpy', 'pandas', 'torch', 'sklearn']
autoclass_content = 'both'
autodoc_typehints = 'description'
autodoc_typehints_format = 'short'
autodoc_member_order = 'bysource'

# MyST Parser settings (for Markdown support)
myst_enable_extensions = [
    'colon_fence',
    'deflist',
    'tasklist',
    'smartquotes',
    'replacements',
    'dollarmath',
    'amsmath',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {
    'navigation_depth': 4,
    'titles_only': False,
    'display_version': True,
    'collapse_navigation': False,
    'sticky_navigation': True,
}

# -- Options for Markdown support --------------------------------------------
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
    '.txt': 'markdown',
}

# -- Options for LaTeX PDF output -------------------------------------------
latex_engine = 'pdflatex'

# Basic LaTeX settings
latex_additional_files = []
latex_logo = None
latex_show_pagerefs = True
latex_show_urls = 'inline'
latex_appendices = ['appendix']
latex_domain_indices = True
latex_use_xindy = False  # Use makeindex instead of xindy

# Book-specific LaTeX settings
latex_documents = [
    (
        'index',
        'pdfragsystem.tex',
        'PDF RAG System Documentation',
        'Ryan Hammang',
        'manual',
        True
    )
]

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

# -- Book-like chapter settings ----------------------------------------------
numfig = True
numfig_format = {
    'figure': 'Figure %s',
    'table': 'Table %s',
    'code-block': 'Listing %s',
    'section': 'Section %s',
}
numfig_secnum_depth = 2
todo_include_todos = True

# Add CSS file
def setup(app):
    app.add_css_file('custom.css')