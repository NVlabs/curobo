# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# Standard Library
import os
import pathlib
import re
import subprocess
from datetime import date
from urllib.parse import urlparse

# Third Party
from sphinx.ext import apidoc

# CuRobo
import curobo

version = curobo.__version__
release = version

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

# Call doxygen if needed:


root = pathlib.Path(__file__).resolve().parent.parent
# sys.path.insert(0, root / "curobo")

# -- Derive repo URL from git remote -----------------------------------------

_DEFAULT_REPO_URL = "https://github.com/NVlabs/curobo"
try:
    _remote = subprocess.check_output(
        ["git", "remote", "get-url", "origin"], cwd=root, text=True
    ).strip()
    if _remote.endswith(".git"):
        _remote = _remote[:-4]
    if _remote.startswith("ssh://"):
        _parsed = urlparse(_remote)
        _remote = f"https://{_parsed.hostname}{_parsed.path}"
    elif _remote.startswith("git@"):
        _host, _, _path = _remote[4:].partition(":")
        _remote = f"https://{_host}/{_path}"
    repo_url = _remote
except Exception:
    repo_url = _DEFAULT_REPO_URL

# Derive "owner/repo" slug from repo_url for use in shields.io badges, etc.
_owner_repo = urlparse(repo_url).path.strip("/")

rst_epilog = f"""
.. |repo_url| replace:: {repo_url}
.. _changelog_link: {repo_url}/blob/main/CHANGELOG.md

.. |gh_stars| image:: https://img.shields.io/github/stars/{_owner_repo}?style=social
   :target: {repo_url}
   :alt: GitHub stars

.. |gh_license| image:: https://img.shields.io/github/license/{_owner_repo}
   :target: {repo_url}/blob/main/LICENSE
   :alt: License: Apache 2.0

.. |gh_release| image:: https://img.shields.io/github/v/release/{_owner_repo}
   :target: {repo_url}/releases
   :alt: Latest release
"""

# -- Project information -----------------------------------------------------

project = "cuRobo"
copyright = f"2023-{date.today().year}, NVIDIA"
author = "NVIDIA"

# -- Run sphinx-apidoc -------------------------------------------------------
# This hack is necessary since RTD does not issue `sphinx-apidoc` before running
# `sphinx-build -b html docs _build/docs`.
# See Issue: https://github.com/[[rtfd/readthedocs.org/issues/1139

output_dir = os.path.join(root, "docs", "api")
module_dir = os.path.join(root, "curobo")
exclude_patterns = [
    module_dir + "/content",
    module_dir + "/test",
]

import curobo._src.runtime as curobo_runtime

curobo_runtime.torch_compile = False
os.environ["PYTORCH_JIT"] = "0"

apidoc_args = [
    "--implicit-namespaces",
    "--force",
    "--separate",
    "--module-first",
    "-o",
    f"{output_dir}",
    f"{module_dir}",
] + exclude_patterns

try:
    apidoc.main(apidoc_args)
    print("Running `sphinx-apidoc` complete!")
except Exception as e:
    print(f"ERROR: Running `sphinx-apidoc` failed!\n{e}")

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions that are shipped with Sphinx (named 'sphinx.ext.*') or your
# custom ones.

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.graphviz",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.intersphinx",
    # Third-party extensions:
    "sphinx_copybutton",
    "sphinx_reredirects",
    "sphinx_autodoc_typehints",  # better typehints
    # "sphinx_github_style",
    "myst_parser",
]

# redirects = {"tutorials/1_robot_configuration.html": "../tutorials/robot_configuration.html"}
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# Variables exposed to Jinja templates (e.g. sidebar/external-links.html).
html_context = {
    "repo_url": repo_url,
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.


intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "trimesh": ("https://trimesh.org/", None),
    "warp": ("https://nvidia.github.io/warp", None),
}


# Prefix autosectionlabel with document path to avoid duplicate label warnings
autosectionlabel_prefix_document = True

# List of warning types to suppress
suppress_warnings = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
# html_theme = 'bizstyle'
# html_theme = 'agogo'
# html_theme = "sphinx_rtd_theme"

html_theme = "furo"

html_title = "cuRobo"  # f"cuRobo {version}"
html_show_sphinx = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_extra_path = ["_html_extra"]

html_css_files = ["custom_furo.css"]
html_js_files = ["version-switcher.js"]
html_logo = "_static/logo-light-mode.png"

# Furo's default sidebar plus our custom external links block.
html_sidebars = {
    "**": [
        "sidebar/scroll-start.html",
        "sidebar/brand.html",
        "sidebar/external-links.html",
        "sidebar/search.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
        "sidebar/variant-selector.html",
    ],
}

# -- Options for extensions --------------------------------------------------

# sphinx.ext.autodoc options
# --------------------------
autoclass_content = "both"
autodoc_class_signature = "mixed"
autodoc_member_order = "bysource"
autodoc_typehints = "signature"
autodoc_warningiserror = False
suppress_warnings.extend(["autodoc"])
autodoc_default_options = {
    "members": True,
    "private-members": False,
    "inherited-members": False,
    "show-inheritance": True,
    "special-members": "__init__",
    "undoc-members": True,
}

# Important for type aliases
autodoc_typehints = "description"
autodoc_typehints_format = "fully-qualified"

# Add support for documenting type aliases
napoleon_attr_annotations = True  # For Google-style docstrings

maximum_signature_line_length = 40

linkcode_url = repo_url
linkcode_link_text = "View Source"
linkcode_blob = "main"
# mathjax options
# ---------------
# NOTE (roflaherty): See
# https://www.sphinx-doc.org/en/master/usage/extensions/math.html#confval-mathjax_config
# http://docs.mathjax.org/en/latest/options/index.html#configuring-mathjax
# https://stackoverflow.com/a/60497853
mathjax4_config = {"tex": {"macros": {}}}

with open("mathsymbols.tex", "r") as f:
    for line in f:
        macros = re.findall(r"\\(DeclareRobustCommand|newcommand){\\(.*?)}(\[(\d)\])?{(.+)}", line)
        for macro in macros:
            if len(macro[2]) == 0:
                mathjax4_config["tex"]["macros"][macro[1]] = "{" + macro[4] + "}"
            else:
                mathjax4_config["tex"]["macros"][macro[1]] = ["{" + macro[4] + "}", int(macro[3])]

# sphinx.ext.todo options
# -----------------------
todo_include_todos = True

# sphinx_rtd_theme options
# ------------------------

if html_theme == "sphinx_rtd_theme":
    html_theme_options = {
        "navigation_depth": 1,
        "collapse_navigation": True,
        "titles_only": False,
        "sticky_navigation": True,
    }
elif html_theme == "furo":
    html_logo = None
    html_theme_options = {
        # "top_of_page_button": None,
        "source_repository": repo_url,
        "source_branch": "main",
        "source_directory": "docs/",
        "top_of_page_buttons": ["view", "edit"],
        "announcement": (
            f"<p>cuRobo is now <a href='{repo_url}'>open source</a> under Apache 2.0</p>"
        ),
        "light_css_variables": {
            # "admonition-title-font-size": "100%",
            # "admonition-font-size": "100%",
            "color-api-pre-name": "#76b900",  # "#76b900",
            "color-api-name": "#76b900",  # "#76b900",
            "color-admonition-title--seealso": "#ffffff",
            "color-admonition-title-background--seealso": "#448aff",
            "color-admonition-title-background--note": "#76b900",
            "color-admonition-title--note": "#ffffff",
            # "font-stack": "Lato, proxima-nova, Helvetica Neue",
            "color-brand-primary": "#76b900",
            "color-sidebar-background": "#f5fff5",
            # "color-brand-content": "#76b900",
        },
        "dark_css_variables": {
            "color-admonition-title-background--note": "#535353",
            "color-admonition-title--note": "#ffffff",
            "color-brand-primary": "#76b900",
            "color-brand-content": "#76b900",
            "color-sidebar-background": "#000000",
        },
        "light_logo": "logo-light-mode.png",
        "dark_logo": "logo-dark-mode.png",
        "footer_icons": [
            {
                "name": "GitHub",
                "url": repo_url,
                "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
                "class": "",
            },
        ],
    }
# autosummary files:
autosummary_generate = True
numpydoc_show_class_members = False

# python clean:
add_module_names = False
add_function_parentheses = False

graphviz_output_format = "svg"
