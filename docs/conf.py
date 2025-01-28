# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
from datetime import datetime
from pathlib import Path

import m2r2
import psifx


# Process docstrings as Markdown.
def docstring(app, what, name, obj, options, lines):
    lines[:] = m2r2.convert("\n".join(lines)).splitlines()


def setup(app):
    app.connect("autodoc-process-docstring", docstring)


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "psifx"
release = version = psifx.__version__
copyright = f"{datetime.now().year}, UNIL"
author = "Guillaume Rochette, Matthew Vowels, Mathieu Rochat"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # Base sphinx automatic docs.
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    # Add links to source code.
    "sphinx.ext.viewcode",
    # Enable mathjax.
    "sphinx.ext.mathjax",
    # Autodocument argparse.
    "sphinxarg.ext",
    # Autodocument variables.
    "sphinx_toolbox.more_autodoc.variables",
    # Adds a `copy` button to code blocks.
    "sphinx_copybutton",
    # Parse markdown docs.
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = []

# Add typehints to both signature and description.
autodoc_typehints = "both"

# Remove parent module names from autodocs.
add_module_names = False

# Prefix document name to avoid clashes.
autosectionlabel_prefix_document = True

# MyST extensions.
myst_enable_extensions = [
    "substitution",
]

# Markdown macros, accessed as {{ variable_name }}
myst_substitutions = {
    "PSIFX_VERSION": psifx.__version__,
}

# Codeblocks theme, which contrasts better with our background colour.
pygments_dark_style = "github-dark"

suppress_warnings = [
    "myst.domains",
    "autosectionlabel.*",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_title = f"{project} - {release}"
html_theme = "furo"
html_favicon = "https://emoji.aranja.com/static/emoji-data/img-apple-160/1f9d0.png"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

# Make the background colour lighter.
html_theme_options = {
    "dark_css_variables": {
        "color-background-primary": "#1f2226",
    },
}
