"""
Anafibre: Analytical mode solver for cylindrical step-index fibers.

Author: Sebastian Golat
Affiliation: King's College London
Created: June 2024

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
from IPython.display import display_html

__all__ = ["StepIndexFibre", "GuidedMode", "RefractiveIndexMaterial"]

def display_modes(*modes):
    display_html(GuidedMode._repr_html_multi(modes), raw=True)