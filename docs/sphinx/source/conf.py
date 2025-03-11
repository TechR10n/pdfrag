# Configuration file for the Sphinx documentation builder.

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
\usepackage{underscore}
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
