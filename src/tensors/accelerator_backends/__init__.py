"""Accelerator-specific tensor backends."""



"""Accelerator-specific tensor backends registry entrypoint.

This module intentionally does NOT import any backend implementations directly.
Backends must register themselves via the registry pattern in abstraction.py.
"""

__all__ = []
