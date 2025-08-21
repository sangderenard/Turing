"""Tensor backends and abstraction layer."""
from __future__ import annotations

DEBUG = False
# Only import abstraction and utilities that do not import backends
try:
    from .abstraction import (
        AbstractTensor,
        get_tensor_operations,
    )
    from .faculty import Faculty, DEFAULT_FACULTY, FORCE_ENV, detect_faculty
except Exception as e:
    import sys
    import traceback
    print("Failed to import core tensor abstraction or faculty utilities")
    print(f"Exception: {e}")
    traceback.print_exc()
    sys.exit(1)

# Do not import any backend modules here. Backend registration is handled via the registry pattern.
