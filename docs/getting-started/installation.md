# Installation

This page covers the fastest way to install `anafibre` and check that it works.

## Requirements

- Python 3.10 or newer
- `pip` (usually included with Python)

## Install from PyPI

Core install:

```bash
pip install anafibre
```

## Optional extras

Install only what you need:

- `units`: support for unit-aware workflows (Astropy)
- `refractiveindex`: refractive index database integration
- `ipython`: richer interactive display in notebooks/shells
- `all`: all optional features above

```bash
pip install "anafibre[units]"
pip install "anafibre[refractiveindex]"
pip install "anafibre[ipython]"
pip install "anafibre[all]"
```

## Verify installation

Run a quick import check:

```bash
python -c "import anafibre; print('anafibre installed successfully')"
```

If that succeeds, continue to [Quickstart](quickstart.md).

## Install from source (contributors)

From the repository root:

```bash
git clone https://github.com/Sevastienn/anafibre.git
cd anafibre
pip install -e .
```

With all optional dependencies:

```bash
pip install -e .[all]
```
