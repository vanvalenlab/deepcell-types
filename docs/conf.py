# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'deepcell-types'
copyright = '%Y, Van Valen Lab at the California Institute of Technology (Caltech)'
author = 'Xuefei (Julie) Wang'
release = '0.0.1-dev'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_nb',  # Text-based jupyter notebooks
    'numpydoc',  # Docstring format
    'sphinx.ext.autosummary',  # Reference guide
    'sphinx.ext.autodoc',  # Docstring summaries
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Execution conf
nb_execution_timeout = 300  # seconds
nb_execution_show_tb = True  # print tracebacks to stderr


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_title = "deepcell-types"
html_theme_options = {
    "github_url": "https://github.com/vanvalenlab/deepcell-types",
}
