"""Python exceptions a native module reports through ``turing_validation_error``.

The native entry points record one integer code.  A code that stands for a
Python exception is raised as that exception (still a ``RuntimeError``, so
callers that caught the generic failure keep catching it).
"""

from __future__ import annotations

#: ``pop`` from an empty list, or a list read/write outside ``-len <= i < len``.
LIST_INDEX_OUT_OF_RANGE = 71


class NativeIndexError(RuntimeError, IndexError):
    """``IndexError`` raised from a native validation code."""


def validation_fault(code: int, runtime_message: str) -> RuntimeError:
    """The exception for one recorded validation code."""

    if int(code) == LIST_INDEX_OUT_OF_RANGE:
        return NativeIndexError(f"list index out of range ({runtime_message})")
    return RuntimeError(runtime_message)
