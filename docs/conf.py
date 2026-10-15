import os
import sys
import feos
import sphinx.ext.autosummary

sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))
sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'feos'
copyright = '2022, Gernot Bauer, Philipp Rehner'
author = 'Gernot Bauer, Philipp Rehner'

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


def hide_missing_constructor(app, what, name, obj, options, signature, return_annotation):
    # classes without a constructor (no `#[new]`) would otherwise be shown as `Class()`
    if what == "class" and getattr(obj, "__text_signature__", None) is None:
        return "", None


# autosummary turns empty signatures into `()`, keep them empty instead
_mangle_signature = sphinx.ext.autosummary.mangle_signature
sphinx.ext.autosummary.mangle_signature = lambda sig, max_chars=30: (
    _mangle_signature(sig, max_chars) if sig else ""
)


def setup(app):
    app.connect("autodoc-process-signature", hide_missing_constructor)


exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# -- Options for HTML output -------------------------------------------------
html_theme = 'furo'
html_title = f'FeOs v{feos.__version__}'

