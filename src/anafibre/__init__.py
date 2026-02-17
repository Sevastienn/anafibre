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

__version__ = "0.1.0"
__author__ = "Sebastian Golat"
__email__ = "sebastian.golat@gmail.com"
__description__ = "Analytical mode solver for cylindrical step-index fibers"
__license__ = "MIT"

from .fibre import StepIndexFibre, RefractiveIndexMaterial
from .fields import GuidedMode
from .plotting import animate_fields_xy
from .utils import repr_html_modes
from IPython.display import display_html, HTML, display

__all__ = ["StepIndexFibre", "GuidedMode", "RefractiveIndexMaterial", "animate_fields_xy", "repr_html_modes", "display_modes", "display_anim"]

def display_modes(*modes):
    display_html(repr_html_modes(modes), raw=True)

def display_anim(anim):
    display(HTML(anim.to_jshtml()))