# anafibre Documentation

`anafibre` is an analytical mode solver for cylindrical step-index fibres.

It provides closed-form guided-mode fields and dispersion tools for dielectric and magnetic fibres, with a workflow built around two main objects:

- `StepIndexFibre`: material profile, dispersion relations, and mode construction.
- `GuidedMode`: field evaluation, gradients, power, and Stokes/polarisation diagnostics.

## What You Can Do

- Build fibres from `n`, `eps`, and `mu` profiles.
- Solve and label guided modes (`HE`, `EH`, `TE`, `TM`).
- Evaluate `E`, `H`, `gradE`, and `gradH` on points or grids.
- Compute effective indices, propagation constants, and mode diagnostics.
- Visualise field components and polarisation with built-in plotting helpers.

## Minimal Example

```python
import numpy as np
import anafibre as fib

fibre = fib.StepIndexFibre(core_radius=250e-9, n_core=2.00, n_clad=1.33)
wl = 700e-9

mode = fibre.HE(ell=1, n=1, wl=wl, a_plus=1/np.sqrt(2), a_minus=1/np.sqrt(2))
E0 = mode.E(x=0, y=0)

print(mode.mode_label())
print(f"n_eff = {mode.neff:.6f}")
print("E(0,0) =", E0)
```

## Documentation Map

- Start here: [Installation](getting-started/installation.md)
- Learn by example: [Quickstart](getting-started/quickstart.md)
- API landing page: [API Reference](api/index.md)
- Background theory: [How It Works](explanation/how-it-works.md)
