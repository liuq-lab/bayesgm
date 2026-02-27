from importlib import import_module
from typing import TYPE_CHECKING

__version__ = "1.0.0"

if TYPE_CHECKING:
    from . import cli, datasets, models, utils
    from .datasets import (
        Base_sampler,
        Semi_Twins_sampler,
        Semi_acic_sampler,
        Sim_Colangelo_sampler,
        Sim_Hirano_Imbens_sampler,
        Sim_Sun_sampler,
    )
    from .models.causalbgm import CausalBGM, FullMCMCCausalBGM, IdentifiableCausalBGM

_SYMBOL_TO_MODULE = {
    "CausalBGM": "bayesgm.models.causalbgm",
    "IdentifiableCausalBGM": "bayesgm.models.causalbgm",
    "FullMCMCCausalBGM": "bayesgm.models.causalbgm",
    "Base_sampler": "bayesgm.datasets",
    "Sim_Hirano_Imbens_sampler": "bayesgm.datasets",
    "Sim_Sun_sampler": "bayesgm.datasets",
    "Sim_Colangelo_sampler": "bayesgm.datasets",
    "Semi_Twins_sampler": "bayesgm.datasets",
    "Semi_acic_sampler": "bayesgm.datasets",
}

_MODULE_ATTRIBUTES = {
    "models": "bayesgm.models",
    "datasets": "bayesgm.datasets",
    "utils": "bayesgm.utils",
    "cli": "bayesgm.cli",
}

__all__ = [
    "CausalBGM",
    "IdentifiableCausalBGM",
    "FullMCMCCausalBGM",
    "Base_sampler",
    "Sim_Hirano_Imbens_sampler",
    "Sim_Sun_sampler",
    "Sim_Colangelo_sampler",
    "Semi_Twins_sampler",
    "Semi_acic_sampler",
]


def __getattr__(name):
    if name in _SYMBOL_TO_MODULE:
        module = import_module(_SYMBOL_TO_MODULE[name])
        return getattr(module, name)
    if name in _MODULE_ATTRIBUTES:
        return import_module(_MODULE_ATTRIBUTES[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__) | set(_MODULE_ATTRIBUTES))
