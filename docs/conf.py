import os
import sys
import sphinx_bootstrap_theme
import feos

sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))
sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'feos'
copyright = '2021, Gernot Bauer, Philipp Rehner'
author = 'Gernot Bauer, Philipp Rehner'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
    'nbsphinx',
]

napoleon_numpy_docstring = True
autodoc_typehints = "both"
autosummary_generate = True
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
source_suffix = {
    '.rst': 'restructuredtext',
}

# -- Options for HTML output -------------------------------------------------
html_theme = 'bootstrap'
html_theme_options = {
    'navbar_sidebarrel': True,
    'navbar_pagenav': True,
    'navbar_pagenav_name': "Page",
    'globaltoc_includehidden': "true",
    'navbar_class': "navbar",
    'navbar_fixed_top': "true",
    'source_link_position': "",
    'bootswatch_theme': "paper",
    'bootstrap_version': "3",
    'navbar_links': [("Python API", "api/index"), ("Python Examples", "examples/index"), ("Rust Guide", "rustguide/index"), ],
}
html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()
html_static_path = ['_static']
html_css_files = [
    'style.css',
]

