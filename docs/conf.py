"""Sphinx configuration."""

project = "AM_Creep_Analysis"
author = "Matthew Nakamura"
copyright = "2025, Matthew Nakamura"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinxarg.ext",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
