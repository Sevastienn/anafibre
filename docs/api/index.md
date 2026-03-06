# API Reference

This section documents the public `anafibre` API. Content on module pages is generated from source docstrings via `mkdocstrings`.

## Package Entry Points

The package root (`anafibre`) re-exports the primary types for typical usage:

- `StepIndexFibre`
- `GuidedMode`
- `RefractiveIndexMaterial`

## Modules

- [dispersion](dispersion.md): Dispersion-relation helpers and related numerics.
- [fibre](fibre.md): Fibre/material models, dispersion relations, and mode constructors.
- [fields](fields.md): Guided-mode field evaluation, polarisation, and coordinate utilities.
- [plotting](plotting.md): Plotting and field-animation utilities.
- [utils](utils.md): High-level display and convenience helpers.

## Notes

- `fibre` and `fields` contain the core solver interfaces.
- `dispersion`, `plotting`, and `utils` provide supporting workflows.
- Use the module pages above for complete signatures and docstrings.
