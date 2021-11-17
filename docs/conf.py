import os
import sys
import sphinx_bootstrap_theme

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
    'myst_parser',
    'nbsphinx',
]

napoleon_numpy_docstring = True
autodoc_typehints = "both"
autosummary_generate = True
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
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

def process_signature(app, what, name, obj, options, signature, return_annotation):
    print("PROCESS SIG", what, name, obj, signature)
    return ("", "")


def before_process_signature(app, obj, bound_method):
    from inspect import signature
    print("BERFORE SIG", obj, bound_method)
    try:
        print("   sig", signature(obj))
    except Exception as e:
        print("DID NOT WORK", e)

def process_docstring(app, what, name, obj, options, lines):
    print("PROCESS DOCSTRING", what, name, obj, options, lines)

# def setup(app):
#     app.connect("autodoc-process-docstring", process_docstring)
#     app.connect("autodoc-before-process-signature", before_process_signature)
#     app.connect("autodoc-process-signature", process_signature)

def setup(app):
    import inspect
    # Add custom signature inspector support *argument-clinic* signatures.
    def inspector(app, what, name, obj, options, signature, return_annotation):
        if signature is not None:
            return signature, return_annotation
        try:
            sig = inspect.signature(obj)
            return str(sig), return_annotation
        except:
            return None, return_annotation

    app.connect('autodoc-process-signature', inspector)
