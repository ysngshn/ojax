# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'OJAX'
copyright = '2024, Yuesong Shen'
author = 'Yuesong Shen'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_readme',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']


# -- Custom configs

# sphinx-book-theme config
html_theme_options = {
    "repository_url": "https://github.com/ysngshn/ojax",
    "path_to_docs": "docs/source",
    "use_source_button": True,
    "use_repository_button": True,
    "use_edit_page_button": True,
    "use_issues_button": True,
}



# autodoc config
autodoc_member_order = 'bysource'


# document __infer_otree_field_type__
def autodoc_custom_skip(app, what, name, obj, would_skip, options):
    if name == "__infer_otree_field_type__":
        return False
    return would_skip


def setup(app):
    app.connect("autodoc-skip-member", autodoc_custom_skip)


# intersphinx config
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'jax': ('https://jax.readthedocs.io/en/latest/', None),
}

# sphinx_readme config
html_context = {
    'display_github': True,
    'github_user': 'ysngshn',
    'github_repo': 'ojax',
    "github_version": "main",
    "conf_py_path": "/docs/source/", # Path in the checkout to the docs root
}
html_baseurl = "https://ysngshn.github.io/ojax"
readme_src_files = "readme.rst"
readme_docs_url_type = "html"
