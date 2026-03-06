"""
Anafibre: Analytical mode solver for cylindrical step-index fibers.

Author: Sebastian Golat
Affiliation: King's College London
Created: August 2025

Description:
Anafibre provides functions to compute the guided modes of cylindrical waveguides by 
solving dispersion relations and calculating the corresponding electromagnetic fields analytically.
It supports both dielectric and magnetic fibers, and enables visualisation of mode profiles and 
dispersion characteristics.

License: MIT
"""

import importlib

__version__     = "0.1.2"
__author__      = "Sebastian Golat"
__email__       = "sebastian.golat@gmail.com"
__description__ = "Analytical mode solver for cylindrical step-index fibers"
__license__     = "MIT"

from .fibre import StepIndexFibre, RefractiveIndexMaterial
from .fields import GuidedMode

_LAZY_SUBMODULES = {
    "plotting",
    "fields",
    "dispersion",
    "utils",
    "fibre",
}


def animate_fields_xy(*args, **kwargs):
    from .plotting import animate_fields_xy as _animate_fields_xy
    return _animate_fields_xy(*args, **kwargs)


def display_modes(*args, **kwargs):
    from .utils import display_modes as _display_modes
    return _display_modes(*args, **kwargs)


def display_anim(*args, **kwargs):
    from .utils import display_anim as _display_anim
    return _display_anim(*args, **kwargs)


def __getattr__(name):
    if name in _LAZY_SUBMODULES:
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_SUBMODULES))


__all__ = [
    "StepIndexFibre",
    "GuidedMode",
    "RefractiveIndexMaterial",
    "animate_fields_xy",
    "display_modes",
    "display_anim",
    "plotting",
    "fields",
    "dispersion",
    "utils",
    "fibre",
]
