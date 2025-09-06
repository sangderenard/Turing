# src/common/tensors/pyopengl_handler.py
"""PyOpenGL array handler for :class:`AbstractTensor`.

This module registers a lightweight shim so that instances of
``AbstractTensor`` (and their NumPy backend) can be passed directly to
PyOpenGL calls without explicitly converting them to ``numpy`` arrays.

If PyOpenGL or the system GL libraries are unavailable the registration is a
no-op, keeping the import side-effects minimal in headless environments.
"""

from __future__ import annotations

try:  # pragma: no cover - optional OpenGL dependency
    from OpenGL.arrays import arraydatatype, numpymodule
except Exception:  # noqa: BLE001 - tolerate missing GL libraries
    arraydatatype = None  # type: ignore[assignment]
    numpymodule = None  # type: ignore[assignment]


def install_pyopengl_handlers() -> None:
    """Install PyOpenGL handlers for ``AbstractTensor`` types.

    The handler delegates to NumPy's implementation but extracts the underlying
    ``numpy`` ``ndarray`` without forcing an intermediate ``np.asarray``
    conversion, satisfying the "no explicit numpy conversion" requirement.
    """

    if arraydatatype is None or numpymodule is None:  # pragma: no cover - optional
        return

    # Import here to avoid circular dependencies
    from .abstraction import AbstractTensor
    from .numpy_backend import NumPyTensorOperations

    class _AbstractTensorHandler(numpymodule.NumpyHandler):
        @classmethod
        def asArray(cls, value, typeCode=None):
            arr = getattr(value, "data", value)  # direct view of backing array
            return super().asArray(arr, typeCode)

    reg = arraydatatype.ArrayDatatype.getRegistry()
    handler = _AbstractTensorHandler()

    # Register for both wrapper + backend instances (traceback showed both)
    reg.register(handler, types=(AbstractTensor, NumPyTensorOperations))
