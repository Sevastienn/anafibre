# Anafibre

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Anafibre** is an analytical mode solver for cylindrical step-index optical fibres. It provides functions to compute the guided modes of cylindrical waveguides by solving dispersion relations and calculating the corresponding electromagnetic fields analytically.

## Features

- 🔬 **Analytical Solutions**: Compute guided modes using exact analytical expressions
- 🌈 **Mode Visualisation**: Rich visualisation of mode profiles and field distributions  
- 📊 **Dispersion Analysis**: Calculate dispersion characteristics and effective indices
- ⚡ **Fast Computation**: Efficient numerical methods for finding propagation constants
- 🎯 **Flexible Design**: Support for both dielectric and magnetic fibres
- 📐 **Unit Support**: Optional integration with Astropy units for dimensional analysis

## Installation

### From Source (recommended for development)

```bash
git clone https://github.com/Sevastienn/anafibre.git
cd anafibre
pip install -e .
```

### Dependencies

The package requires the following core dependencies:
- numpy >= 1.20.0
- scipy >= 1.7.0  
- matplotlib >= 3.5.0
- IPython >= 7.0.0

Optional dependencies:
- astropy >= 4.0.0 (for unit support) - install with `pip install anafibre[units]`
- refractiveindex >= 0.1.0 (for refractive index database) - install with `pip install anafibre[refractiveindex]`

### Future PyPI Release

The package will be available on PyPI soon:

```bash
pip install anafibre
```

## Quick Start

```python
import numpy as np
import anafibre as af

# Create a step-index fibre
core_radius = 4.1e-6  # 4.1 μm
core_index = 1.46
clad_index = 1.45

fibre = af.StepIndexFibre(
    core_radius=core_radius,
    n_core=core_index,
    n_clad=clad_index
)

# Find guided modes at 1550 nm
wavelength = 1550e-9  # 1550 nm

# Get fundamental mode (HE11)
fundamental_mode = fibre.HE(ell=1, n=1, wl=wavelength)

# Display mode information  
print(f"Fundamental mode: {fundamental_mode.mode_label()}")
print(f"Effective index: {fundamental_mode.neff:.6f}")

# For multi-mode fibres, get additional modes
try:
    te01 = fibre.TE(n=1, wl=wavelength)  # TE01 mode
    tm01 = fibre.TM(n=1, wl=wavelength)  # TM01 mode
    he21 = fibre.HE(ell=2, n=1, wl=wavelength)  # HE21 mode
except:
    pass  # Mode may not exist for this fibre

# Calculate field distributions
mode = fundamental_mode  # Use the mode from above
x = np.linspace(-2*core_radius, 2*core_radius, 100)
y = np.linspace(-2*core_radius, 2*core_radius, 100)
X, Y = np.meshgrid(x, y)

E_field = mode.E(x=X, y=Y)  # Electric field
H_field = mode.H(x=X, y=Y)  # Magnetic field
```

## Key Classes and Functions

### StepIndexFibre

The main class for defining step-index optical fibres:

```python
fiber = af.StepIndexFibre(
    core_radius=4.1e-6,     # Core radius in meters
    n_core=1.46,            # Core refractive index  
    n_clad=1.45,            # Cladding refractive index
    n_substrate=None        # Optional substrate index
)
```

### GuidedMode

Represents a guided mode with methods to calculate fields and properties:

```python
# Create specific modes
mode = fibre.HE(ell=1, n=1, wl=1550e-9)  # Fundamental HE11 mode
te_mode = fibre.TE(n=1, wl=1550e-9)      # TE01 mode  
tm_mode = fibre.TM(n=1, wl=1550e-9)      # TM01 mode

# Mode properties
print(f"Effective index: {mode.neff}")
print(f"Propagation constant: {mode.beta}")
print(f"Mode label: {mode.mode_label()}")

# Field calculations
E = mode.E(rho=rho, phi=phi)  # Cylindrical coordinates
E = mode.E(x=x, y=y)          # Cartesian coordinates
```

### Visualization

```python
import matplotlib.pyplot as plt
from anafibre.plotting import plot_mode_profile

# Plot mode field profile
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
plot_mode_profile(mode, component='E', ax=ax[0])
plot_mode_profile(mode, component='H', ax=ax[1])
plt.show()
```

## Optional Dependencies

- **astropy**: For unit handling and dimensional analysis
  ```bash
  pip install astropy
  ```

## Examples

Check out the `notebooks/` directory for detailed examples:

- Basic fibre mode calculations
- Dispersion analysis
- Mode field visualisation  
- Multi-mode fibre analysis

## Requirements

- Python ≥ 3.8
- NumPy ≥ 1.20.0
- SciPy ≥ 1.7.0
- Matplotlib ≥ 3.5.0
- refractiveindex ≥ 0.1.0
- IPython ≥ 7.0.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use Anafibre in your research, please cite:

```bibtex
@software{anafibre2024,
  author = {Golat, Sebastian},
  title = {Anafibre: Analytical mode solver for cylindrical step-index fibres},
  year = {2024},
  url = {https://github.com/Sevastienn/anafibre},
  version = {0.1.0}
}
```

## Author

**Sebastian Golat**
- Email: sebastian.golat@gmail.com
- Affiliation: King's College London

## Acknowledgments

- Developed at King's College London
- Inspired by classical optical fibre theory and numerical methods for waveguide analysis
