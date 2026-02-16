from conf import *  # noqa: F401,F403

project = "bayesgm - BGM"
html_title = "BGM Documentation"
master_doc = "bgm/index"

exclude_patterns = exclude_patterns + [  # type: ignore[name-defined]
    "index.rst",
    "causalbgm/**",
    "getting-started/quickstart_causalbgm.md",
]
