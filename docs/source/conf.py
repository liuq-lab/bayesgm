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
    "sphinx.ext.intersphinx",
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
napoleon_use_param = True
nbsphinx_execute = "never"

# On RTD we don't need TensorFlow execution; mock heavyweight modules to keep
# autodoc importable and stable.
if os.environ.get("READTHEDOCS") == "True":
    autodoc_mock_imports = [
        "tensorflow",
        "tensorflow_probability",
        "keras",
        "bayesgm.models.networks",
    ]

# -- Intersphinx mapping for external libraries ----------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "tensorflow": ("https://www.tensorflow.org/api_docs/python", "https://github.com/GPflow/tensorflow-intersphinx/raw/master/tf2_py_objects.inv"),
}

# -- Suppress false cross-reference warnings from NumPy-style docstrings ---
nitpicky = False
nitpick_ignore = [
    ("py:class", "optional"),
    ("py:class", "array-like"),
]
nitpick_ignore_regex = [
    # "default=..." values parsed out of type fields
    (r"py:class", r"default=.*"),
    # NumPy / TensorFlow shorthand type names
    (r"py:class", r"np\.ndarray"),
    (r"py:class", r"tf\.Tensor"),
    # Miscellaneous type fragments from docstrings
    (r"py:class", r"optional"),
    (r"py:class", r"floats"),
    # Tuple-unpacking labels in shape descriptions, e.g. (H, W)
    (r"py:class", r"H"),
    (r"py:class", r"W"),
    (r"py:class", r"row"),
    (r"py:class", r"col"),
]

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
