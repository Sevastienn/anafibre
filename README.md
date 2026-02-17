<!-- 
<h1>
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./assets/logos/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="./assets/logos/logo-light.svg">
    <img alt="anafibre logo" src="./assets/logos/logo-light.svg" width="150">
  </picture>
</p>
</h1><br> -->
<h1>
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Sevastienn/anafibre/refs/heads/main/assets/logos/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/Sevastienn/anafibre/refs/heads/main/assets/logos/logo-light.svg">
    <img alt="anafibre logo" src="https://raw.githubusercontent.com/Sevastienn/anafibre/refs/heads/main/assets/logos/logo-light.svg" width="150">
  </picture>
</p>
</h1><br>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/anafibre.svg)](https://pypi.org/project/anafibre/)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![ORCID](https://img.shields.io/badge/ORCID-0000--0003--3947--7634-A6CE39?logo=orcid&logoColor=white)](https://orcid.org/0000-0003-3947-7634)
<!-- [![arXiv](https://img.shields.io/badge/arXiv-2512.01784-B31B1B?logo=data:image/svg+xml;base64,PHN2ZyBpZD0ibG9nb21hcmsiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgdmlld0JveD0iMCAwIDE3LjczMiAyNC4yNjkiPjxnIGlkPSJ0aW55Ij48cGF0aCBkPSJNNTczLjU0OSwyODAuOTE2bDIuMjY2LDIuNzM4LDYuNjc0LTcuODRjLjM1My0uNDcuNTItLjcxNy4zNTMtMS4xMTdhMS4yMTgsMS4yMTgsMCwwLDAtMS4wNjEtLjc0OGgwYS45NTMuOTUzLDAsMCwwLS43MTIuMjYyWiIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTU2Ni45ODQgLTI3MS41NDgpIiBmaWxsPSIjYmRiOWI0Ii8+PHBhdGggZD0iTTU3OS41MjUsMjgyLjIyNWwtMTAuNjA2LTEwLjE3NGExLjQxMywxLjQxMywwLDAsMC0uODM0LS41LDEuMDksMS4wOSwwLDAsMC0xLjAyNy42NmMtLjE2Ny40LS4wNDcuNjgxLjMxOSwxLjIwNmw4LjQ0LDEwLjI0MmgwbC02LjI4Miw3LjcxNmExLjMzNiwxLjMzNiwwLDAsMC0uMzIzLDEuMywxLjExNCwxLjExNCwwLDAsMCwxLjA0LjY5QS45OTIuOTkyLDAsMCwwLDU3MSwyOTNsOC41MTktNy45MkExLjkyNCwxLjkyNCwwLDAsMCw1NzkuNTI1LDI4Mi4yMjVaIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgtNTY2Ljk4NCAtMjcxLjU0OCkiIGZpbGw9IiNiMzFiMWIiLz48cGF0aCBkPSJNNTg0LjMyLDI5My45MTJsLTguNTI1LTEwLjI3NSwwLDBMNTczLjUzLDI4MC45bC0xLjM4OSwxLjI1NGEyLjA2MywyLjA2MywwLDAsMCwwLDIuOTY1bDEwLjgxMiwxMC40MTlhLjkyNS45MjUsMCwwLDAsLjc0Mi4yODIsMS4wMzksMS4wMzksMCwwLDAsLjk1My0uNjY3QTEuMjYxLDEuMjYxLDAsMCwwLDU4NC4zMiwyOTMuOTEyWiIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTU2Ni45ODQgLTI3MS41NDgpIiBmaWxsPSIjYmRiOWI0Ii8+PC9nPjwvc3ZnPg==)](https://arxiv.org/abs/2512.01784) -->


**Anafibre** is an analytical mode solver for cylindrical step-index optical fibres. It computes guided modes by solving dispersion relations and evaluating corresponding electromagnetic fields analytically.

## Features

- 🔬 **Analytical solutions** for guided modes in cylindrical fibres
- 🌈 **Mode visualisation** with plotting utilities for field components
- 📊 **Dispersion analysis** helpers and effective index calculations
- ⚡ **Fast computation** of propagation constants with [SciPy](https://github.com/scipy/scipy)-based root finding
- 🎯 **Flexible materials** support via fixed indices, callables or [refractiveindex.info](https://refractiveindex.info/) database 
- 📐 **Optional unit support** through [Astropy](https://github.com/astropy/astropy)

## Installation

### Install from PyPI
```bash
pip install anafibre
```

### Optional extras
- Units support using [`astropy.units.Quantity`](https://docs.astropy.org/en/stable/units/quantity.html)
  ```bash
  pip install "anafibre[units]"
  ```
- [refractiveindex.info](https://refractiveindex.info/) database support
  ```bash
  pip install "anafibre[refractiveindex]"
  ```
- All optional features (units + [refractiveindex.info](https://refractiveindex.info/))
  ```bash
  pip install "anafibre[all]"
  ```



### From Source (development)

```bash
git clone https://github.com/Sevastienn/anafibre.git
cd anafibre
pip install -e .
```
<!-- 
### Core dependencies

- numpy >= 1.20.0
- scipy >= 1.7.0
- matplotlib >= 3.5.0
- IPython >= 7.0.0

### Optional extras
- astropy >= 4.0.0 (for unit support) - install with `pip install anafibre[units]`
- refractiveindex >= 0.1.0 (for refractive index database) - install with `pip install anafibre[refractiveindex]` -->


<!-- 
## Quick Start

```python
import numpy as np
import anafibre as fib

# Create a step-index fibre
fibre = fib.StepIndexFibre(
    core_radius=250e-9,
    n_core=2.00,
    n_clad=1.33
)

# Get fundamental mode (HE11) at 500 nm
wl = 500e-9
HE11 = fibre.HE(ell=1, n=1, wl=wl)

# Try to get some higher-order modes:
TM01 = HE21 = TE01 = EH11 = None
try:
    TM01 = fibre.TM(n=1, wl=wl)          
    HE21 = fibre.HE(ell=2, n=1, wl=wl)   
    TE01 = fibre.TE(n=1, wl=wl)          
    EH11 = fibre.EH(ell=1, n=1, wl=wl)   
except (ValueError, RuntimeError):
    pass  # Some modes may not exist for this fibre

# show only what we actually got
fib.display_modes(*[m for m in (HE11, TM01, HE21, TE01, EH11) if m is not None])

# Calculate field distributions
mode = HE11  # Use the mode from above
x = np.linspace(-2*fibre.core_radius, 2*fibre.core_radius, 100)
y = np.linspace(-2*fibre.core_radius, 2*fibre.core_radius, 100)
X, Y = np.meshgrid(x, y)

E = mode.E(x=X, y=Y)  # Electric field
H = mode.H(x=X, y=Y)  # Magnetic field
``` -->

## Core API Overview

Anafibre has two main objects:
- `StepIndexFibre` — defines the waveguide (geometry + materials)
- `GuidedMode` — represents a single solved eigenmode

The typical workflow is:
```python
fibre = StepIndexFibre(...)
mode = fibre.HE(...)
E = mode.E(x=X, y=Y)
```

---
### `StepIndexFibre`

Defines the fibre geometry and material parameters and provides dispersion utilities.

#### Required inputs

- `core_radius` (float in meters or `astropy.units.Quantity`)
- One of:
  - `core`, `clad` as `RefractiveIndexMaterial`
  - `n_core`, `n_clad` (float or callable *λ→ε*(*λ*))
  - `eps_core`, `eps_clad` (float or callable *λ→ε*(*λ*))
#### Optional inputs
- `mu_core`, `mu_clad` (float or callable *λ→ε*(*λ*))

#### Example

  ```python
  fibre = fib.StepIndexFibre(core_radius=250e-9, n_core=2.00, n_clad=1.33)
  ```

#### Provides
- **Mode constructors** for HE<sub>ℓn&nbsp;</sub>, EH<sub>ℓn&nbsp;</sub>, TE<sub>0n&nbsp;</sub>, and TM<sub>0n</sub> modes
  ```python
  fibre.HE(ell, n, wl, a_plus=..., a_minus=...)
  fibre.EH(...)
  fibre.TE(n, wl, a=...)
  fibre.TM(...)
  ```
  Each returns a `GuidedMode` object.

- **Dispersion utilities** to find *V, b, k<sub>z </sub>,*&nbsp;and *n*<sub>eff</sub>
  ```python
  fibre.V(wavelength)
  fibre.b(ell, m, V=..., wavelength=...)
  fibre.kz(...)
  fibre.neff(...)
  ```
---
### `GuidedMode`

Represents a guided mode with methods to calculate fields and properties. It is created using `StepIndexFibre` mode constructors.

<!-- Example:
```python
# Create the fundamental modes with different polarisations:
HE11L = fibre.HE(ell=1, n=1, wl=500e-9, a_plus=0, a_minus=1)
HE11x = fibre.HE(ell=1, n=1, wl=500e-9, a_plus=1/np.sqrt(2), a_minus=1/np.sqrt(2))
HE11y = fibre.HE(ell=1, n=1, wl=500e-9, a_plus=1j/np.sqrt(2), a_minus=-1j/np.sqrt(2))
``` -->



#### Provides
- Field evaluation in (ρ,ϕ,z) or (x,y,z) coordinates, when z is not provided z=0 is assumed
  ```python
  E = mode.E(rho=Rho, phi=Phi, z=Z)
  H = mode.H(rho=Rho, phi=Phi, z=Z)
  E = mode.E(x=X, y=Y, z=Z)
  H = mode.H(x=X, y=Y, z=Z)
  ```
 Both return arrays with a shape (..., 3) corresponding to the Cartesian vector components.
- Jacobians (gradients) of the fields
  ```python
  J_E = mode.gradE(rho=Rho, phi=Phi, z=Z)
  J_H = mode.gradH(rho=Rho, phi=Phi, z=Z)
  J_E = mode.gradE(x=X, y=Y, z=Z)
  J_H = mode.gradH(x=X, y=Y, z=Z)
  ```
 Both return arrays with a shape of (..., 3, 3), corresponding to the Cartesian tensor components.
- Power evaluated via numerical integration
  ```python
  P = mode.Power()
  ```


### Visualisation
The package ships with a built-in plotting utility that creates time-resolved animations of the electromagnetic field in the transverse cross-section of the fibre:
```python
from anafibre.plotting import animate_fields_xy
from IPython.display import HTML
anim = animate_fields_xy(modes=mode, show=("E",),figsize=(5,5))
display(HTML(anim.to_jshtml()))
```


## Citation

If you use Anafibre in your research, please cite:

```bibtex
@misc{anafibre2026,
  author  = {Golat, Sebastian},
  title   = {{Anafibre: Analytical mode solver for cylindrical step-index fibres}},
  year    = {2026},
  note    = {{Python package}},
  url     = {https://github.com/Sevastienn/anafibre},
  version = {0.1.0}}
```
