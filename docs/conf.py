# -- Project information -----------------------------------------------------
"""Sphinx configuration for mockdock documentation."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from mockdock import __version__ as release
except Exception:
    release = "0.2.0"

project = "mockdock"
author = "mockdock contributors"
copyright = f"{datetime.now():%Y}, {author}"
version = release

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

# Heavy / optional native deps: mock so autodoc can import the package in CI/docs env
autodoc_mock_imports = [
    "rdkit",
    "rdkit.Chem",
    "rdkit.Chem.AllChem",
    "rdkit.Chem.Scaffolds",
    "rdkit.Chem.Scaffolds.MurckoScaffold",
    "rdkit.Chem.QED",
    "rdkit.Chem.Descriptors",
    "rdkit.Chem.rdFMCS",
    "rdkit.Chem.FilterCatalog",
    "rdkit.DataStructs",
    "polars",
    "vina",
    "prody",
    "meeko",
    "gemmi",
    "molscrub",
    "chembl_webresource_client",
    "chembl_webresource_client.new_client",
    "scipy",
    "scipy.spatial",
    "scipy.spatial.distance",
    "matplotlib",
    "matplotlib.pyplot",
    "seaborn",
    "tqdm",
    "aiohttp",
    "asttokens",
    "comm",
    "pyarrow",
    "click",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "polars": ("https://docs.pola.rs/api/python/stable/", None),
}

# -- Options for HTML output -------------------------------------------------
html_theme = "shibuya"
html_static_path = ["_static"]
html_title = "mockdock"
html_copy_source = False
html_show_sourcelink = False

html_theme_options = {
    "accent_color": "amber",
    "github_url": "https://github.com/Popov-Lab-UNC/mockdock",
    "globaltoc_expand_depth": 1,
    "nav_links": [
        {"title": "Installation", "url": "installation"},
        {"title": "Running Benchmarks", "url": "running"},
        {"title": "Evaluating Results", "url": "evaluation"},
        {"title": "Benchmark Reference & Guide", "url": "reference/index"},
        {"title": "API Reference", "url": "api/index"},
    ],
}

html_context = {
    "source_type": "github",
    "source_user": "Popov-Lab-UNC",
    "source_repo": "mockdock",
    "source_version": "main",
    "source_docs_path": "/docs/",
}
