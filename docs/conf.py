import os
import sys
import feos

sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))
sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'feos'
copyright = '2022, Gernot Bauer, Philipp Rehner'
author = 'Gernot Bauer, Philipp Rehner'
release = feos.__version__

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
    'nbsphinx',
    'myst_parser',
    'sphinx_copybutton',
    'sphinx_design',
    'sphinx_inline_tabs',
]

# -- Options for Markdown files ----------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath"
]
myst_heading_anchors = 3

napoleon_numpy_docstring = True
autodoc_typehints = "both"
autosummary_generate = True
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# -- Options for HTML output -------------------------------------------------
html_theme = 'furo'
html_title = f'FeOs v{feos.__version__}'
html_favicon = '_static/favicon.svg'
html_static_path = ['_static']
html_theme_options = {
    'light_logo': 'logo_light.svg',
    'dark_logo': 'logo_dark.svg',
    'sidebar_hide_name': True,
}

