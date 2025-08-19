# Tensors Package

This package provides an abstract tensor interface and backend implementations.
Optional extras let you install JAX, C, and NumPy backends.

For a guided overview of the architecture, see `EXPLAINER.md`.
## Design Philosophy

The `AbstractTensor` class behaves as much like PyTorch as possible while still accepting
operations that match NumPy, JAX, or pure Python expectations.  Methods are
carefully overloaded so that users coming from any of those libraries find familiar
interfaces.  When semantics differ between libraries we default to PyTorch's
behavior.  The `ShapeAccessor` helper demonstrates this approach by mimicking
`shape` access in a library‑agnostic way.

Development priorities follow a strict order:
1. Finalize operations in the abstract interface.
2. Ensure features work identically across Torch, NumPy, JAX, and pure Python backends.
3. Fill gaps in individual backends.
4. Only after the above are complete do we expand the optional C backend, except for trivial
   stub replacements when time permits.
