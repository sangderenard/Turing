# Tensor Backends

## Quick Setup

Refer to `../ENV_SETUP_OPTIONS.md` for setup instructions.

This directory hosts the implementations of tensor operations for different numerical libraries.

### Backend Policy

- The **NumPy backend** is the default and canonical implementation.
- **Do not import, install, or rely on PyTorch (`torch`)**. The dependency is too heavy for this project.
- If an operation is missing, implement it in the NumPy backend rather than reaching for PyTorch or other large frameworks.

**Important:** Each backend must implement `_apply_operator` from `AbstractTensor`. This single method handles all arithmetic primitives. Avoid creating additional bespoke operator helpers – Python's magic methods already route standard arithmetic through `_apply_operator`.

Follow the repository coding standards and remember to run the test suite after modifications.

## Development Ethos

`AbstractTensor` mirrors a subset of the PyTorch API while accommodating idioms
from NumPy and plain Python lists.  Operators are overloaded to hide backend
specifics, letting contributors work with whichever library they know best.
When behaviours diverge between libraries, **NumPy is the authoritative
reference**.

Implementation priority follows this order:
1. Get the abstract interface fully specified.
2. Maintain feature parity across **NumPy** and pure Python backends.
3. Address backend-specific gaps without introducing heavy dependencies.
4. Expand the C backend only after the above are complete unless a simple stub
   can be filled quickly.
