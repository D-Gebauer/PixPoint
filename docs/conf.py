"""Sphinx configuration for the PixPoint documentation (pixpoint.readthedocs.io)."""

import importlib.metadata
import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "PixPoint"
author = "David Gebauer"
copyright = "2026, David Gebauer"
try:
    release = importlib.metadata.version("pixpoint")
except importlib.metadata.PackageNotFoundError:  # building from a bare checkout
    import pixpoint

    release = pixpoint.__version__
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]

# Markdown everywhere; the docstrings are Google style (napoleon).
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist", "fieldlist"]
myst_heading_anchors = 3

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = False
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {"members": True, "undoc-members": False, "show-inheritance": True}
# The GPU stack and the heavy optional imports are not needed to document the API.
autodoc_mock_imports = ["cupy", "cupyx"]
autosummary_generate = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "healpy": ("https://healpy.readthedocs.io/en/latest", None),
    "h5py": ("https://docs.h5py.org/en/stable", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = f"PixPoint {release}"
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "https://github.com/D-Gebauer/PixPoint",
    "source_branch": "main",
    "source_directory": "docs/",
    "light_css_variables": {"color-brand-primary": "#1e66f5", "color-brand-content": "#1e66f5"},
    "dark_css_variables": {"color-brand-primary": "#89b4fa", "color-brand-content": "#89b4fa"},
}
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
