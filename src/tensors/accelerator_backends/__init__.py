#!/usr/bin/env python3
"""Accelerator-specific tensor backends."""
from __future__ import annotations

try:
    from .c_backend import CTensorOperations
    from .opengl_backend import OpenGLTensorOperations
    from .rust_backend import RustTensorOperations
except Exception:
    import sys
    print("Accelerator backends failed to import")
    sys.exit(1)
# --- END HEADER ---

__all__ = ["CTensorOperations", "OpenGLTensorOperations", "RustTensorOperations"]
