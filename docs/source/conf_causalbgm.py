from conf import *  # noqa: F401,F403

project = "bayesgm - CausalBGM"
html_title = "CausalBGM Documentation"
master_doc = "causalbgm/index"

exclude_patterns = exclude_patterns + [  # type: ignore[name-defined]
    "index.rst",
    "bgm/**",
    "getting-started/quickstart_bgm.md",
]
