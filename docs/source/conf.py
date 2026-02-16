from datetime import datetime
from pathlib import Path
import os
import sys

HERE = Path(__file__).parent.resolve()
REPO_ROOT = HERE.parent.parent
SRC_ROOT = REPO_ROOT / "src"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

project = "bayesgm"
author = "Qiao Liu"
copyright = f"{datetime.now():%Y}, {author}"
html_title = "bayesgm Documentation"
master_doc = "index"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "nbsphinx",
    "recommonmark",
    "sphinx_markdown_tables",
]

autosummary_generate = True
autodoc_member_order = "bysource"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
nbsphinx_execute = "never"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "logo.png"
html_theme_options = {
    "navigation_depth": 4,
    "logo_only": True,
}
