#!/usr/bin/env python3
"""Tensor backends and abstraction layer."""
from __future__ import annotations

try:
    from .abstraction import (
        AbstractTensor,
        get_tensor_operations,
    )
    from .faculty import Faculty, DEFAULT_FACULTY, FORCE_ENV, detect_faculty
    from .pure_backend import PurePythonTensorOperations
except Exception:
    import sys
    print("Failed to import optional tensor backends")
    sys.exit(1)
# --- END HEADER ---

PyTorchTensorOperations = None
try:  # optional torch backend
    from .torch_backend import PyTorchTensorOperations  # type: ignore
except Exception:  # pragma: no cover - torch missing
    PyTorchTensorOperations = None  # type: ignore

NumPyTensorOperations = None
try:  # optional numpy backend
    from .numpy_backend import NumPyTensorOperations  # type: ignore
except Exception:  # pragma: no cover - numpy missing
    NumPyTensorOperations = None  # type: ignore

JAXTensorOperations = None
try:  # optional jax backend
    from .jax_backend import JAXTensorOperations  # type: ignore
except Exception:  # pragma: no cover - jax missing
    JAXTensorOperations = None  # type: ignore

CTensorOperations = None
try:  # optional C backend
    from .accelerator_backends.c_backend import CTensorOperations  # type: ignore
except Exception:  # pragma: no cover - c backend missing
    CTensorOperations = None  # type: ignore

OpenGLTensorOperations = None
try:  # optional OpenGL backend
    from .accelerator_backends.opengl_backend import OpenGLTensorOperations  # type: ignore
except Exception:  # pragma: no cover - opengl missing
    OpenGLTensorOperations = None  # type: ignore

AcceleratorCoordinator = None
try:
    from .accelerator_backends.coordinator import AcceleratorCoordinator  # type: ignore
except Exception:  # pragma: no cover - missing dependencies
    AcceleratorCoordinator = None  # type: ignore

__all__ = [
    "AbstractTensor",
    "get_tensor_operations",
    "PyTorchTensorOperations",
    "NumPyTensorOperations",
    "PurePythonTensorOperations",
    "Faculty",
    "DEFAULT_FACULTY",
    "FORCE_ENV",
    "detect_faculty",
]

if JAXTensorOperations is not None:
    __all__.append("JAXTensorOperations")

if CTensorOperations is not None:
    __all__.append("CTensorOperations")

if OpenGLTensorOperations is not None:
    __all__.append("OpenGLTensorOperations")

if AcceleratorCoordinator is not None:
    __all__.append("AcceleratorCoordinator")
