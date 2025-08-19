# c_backend

This subdirectory documents the design rules for the C backend implementation.

* **Algorithm code must be written in C**, not Python. All heavy computation should be compiled via `cffi` or similar tooling.
* **Support arbitrary tensor dimensions.** Functions should not assume one- or two-dimensional inputs.
* **Mirror the behavior of PyTorch**. Operations implemented in C must produce the same results as equivalent Torch functions for the supported dtypes.

Follow the repository's general coding standards and keep tests passing.

Development of this backend waits until the abstract interface and all Python
backends (Torch, NumPy, JAX, pure) are stable.  Only quick fixes or trivial
stub completions are accepted before then.
