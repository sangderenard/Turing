"""Python exceptions a native module reports through ``turing_validation_error``.

The native entry points record one integer code.  A code that stands for a
Python exception is raised as that exception (still a ``RuntimeError``, so
callers that caught the generic failure keep catching it).
"""

from __future__ import annotations

#: ``pop`` from an empty list, or a list read/write outside ``-len <= i < len``.
LIST_INDEX_OUT_OF_RANGE = 71

#: A subscript read of a mapping key it does not hold (``required_key_absent``).
MAPPING_KEY_ABSENT = 84


class NativeIndexError(RuntimeError, IndexError):
    """``IndexError`` raised from a native validation code."""


class NativeKeyError(RuntimeError, KeyError):
    """``KeyError`` raised from a native validation code."""

    def __str__(self) -> str:
        return RuntimeError.__str__(self)


def validation_fault(code: int, runtime_message: str) -> RuntimeError:
    """The exception for one recorded validation code."""

    if int(code) == LIST_INDEX_OUT_OF_RANGE:
        return NativeIndexError(f"list index out of range ({runtime_message})")
    if int(code) == MAPPING_KEY_ABSENT:
        return NativeKeyError(f"key not in mapping ({runtime_message})")
    return RuntimeError(runtime_message)
