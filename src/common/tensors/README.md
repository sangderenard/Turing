# Tensors Package

This package provides an abstract tensor interface and backend implementations.
Optional extras let you install JAX, C, and NumPy backends.

For a guided overview of the architecture, see `EXPLAINER.md`.
## Design Philosophy

The `AbstractTensor` class behaves as much like PyTorch as possible while still accepting
operations that match NumPy, JAX, or pure Python expectations.  Methods are
carefully overloaded so that users coming from any of those libraries find familiar
interfaces.  When semantics differ between libraries we default to PyTorch's
behavior.  (Old: The `ShapeAccessor` helper demonstrated this approach by mimicking)
`shape` access in a library‑agnostic way.

Development priorities follow a strict order:
1. Finalize operations in the abstract interface.
2. Ensure features work identically across Torch, NumPy, JAX, and pure Python backends.
3. Fill gaps in individual backends.
4. Only after the above are complete do we expand the optional C backend, except for trivial
   stub replacements when time permits.

## Autograd strict mode and whitelisting

The autograd engine can enforce that every tensor contributing to a loss has
registered backward rules and is connected to the loss graph.  Enable this
behaviour by setting the environment variable `AUTOGRAD_STRICT=1`.  When active,
any missing backward implementation or disconnected input raises an error.

Tensors carrying a `label` annotation can be ignored by providing a comma or
pipe separated list of regular expressions via
`AUTOGRAD_STRICT_ALLOW_LABELS`.

Strict mode may be tuned or disabled at runtime:

```python
from src.common.tensors.autograd import autograd

autograd.strict = False                  # disable strict checks
autograd.whitelist(tensor_a, tensor_b)   # ignore specific tensors
autograd.whitelist_labels(r"demo.*")    # allow labels matching the pattern
```

Labels can be attached using `autograd.tape.annotate(tensor, label="name")`.
