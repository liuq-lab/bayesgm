# Docs Layout

This folder now supports three Read the Docs build targets:

1. Unified `bayesgm` docs (BGM + CausalBGM)
   - Config: `.readthedocs.yaml`
   - Sphinx config: `docs/source/conf.py`

2. BGM-only docs
   - Config: `.readthedocs-bgm.yaml`
   - Sphinx config: `docs/source/conf_bgm.py`

3. CausalBGM-only docs
   - Config: `.readthedocs-causalbgm.yaml`
   - Sphinx config: `docs/source/conf_causalbgm.py`

Main content sections live in:

- `docs/source/getting-started/`
- `docs/source/bgm/`
- `docs/source/causalbgm/`
