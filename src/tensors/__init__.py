"""Tensor backends and abstraction layer."""
from __future__ import annotations

# Only import abstraction and utilities that do not import backends
try:
    from .abstraction import (
        AbstractTensor,
        get_tensor_operations,
    )
    from .faculty import Faculty, DEFAULT_FACULTY, FORCE_ENV, detect_faculty
except Exception:
    import sys
    print("Failed to import core tensor abstraction or faculty utilities")
    sys.exit(1)

# Do not import any backend modules here. Backend registration is handled via the registry pattern.
