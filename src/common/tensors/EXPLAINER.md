# Tensor Project Explainer

This document offers a concise guide to the `tensors` package. The project defines
an abstraction layer over multiple numerical libraries so higher level code can
operate on tensors without binding to a specific backend.

## Key Concepts

- **AbstractTensor** – Base class that implements standard tensor operations via
  the `_apply_operator` hook. Backends subclass this to provide concrete
  implementations.
- **Faculty** – Enum representing capability tiers. The environment detection
  logic chooses the highest available faculty (Pure Python, NumPy, Torch, JAX,
  etc.).
- **Backend Modules** – Each backend (`pure_backend.py`, `numpy_backend.py`,
  `torch_backend.py`, `jax_backend.py`, `accelerator_backends/c_backend.py`,
  `accelerator_backends/opengl_backend.py`)
  implements tensor primitives using its library of choice.
- **Conversion Registry** – Utility functions allow converting tensors between
  backends when possible.

## Typical Workflow

1. Call `get_tensor_operations()` to retrieve an `AbstractTensor` subclass
   appropriate for the current environment.
2. Instantiate tensors via methods like `full()` or `zeros()` from the returned
   operations class.
3. Use the methods defined on `AbstractTensor` to manipulate data. All common
   arithmetic operators route through `_apply_operator` so behavior stays
   consistent across backends.
4. Optional convenience helpers in `faculty.py` allow you to inspect available
   faculties or force a specific backend using the `TENSOR_FACULTY`
   environment variable.

## Relationship to Other Projects

This abstraction layer allows other projects to run with or without heavy numerical dependencies. The experimental `tensorprinting` package also builds upon these classes to explore novel tensor visualization techniques.

## Project Ethos

The abstraction aims to feel like PyTorch while remaining approachable for users
accustomed to NumPy, JAX, or simple Python lists.  We overload functions just
enough to mask backend differences – as seen with the `shape` accessor – so
code reads naturally no matter which library you prefer.  When behaviour differs
between frameworks we follow PyTorch's lead.

We prioritize work in this order:
1. Define and test operations in the abstract base.
2. Keep all Python backends in sync with those definitions.
3. Fill missing pieces of a particular backend when gaps appear.
4. Only then extend the optional C backend, aside from easy stub completions.

## Further Reading

See `abstraction_functions.md` for an auto-generated list of methods and pending
stubs. Consult each backend module for implementation details.
