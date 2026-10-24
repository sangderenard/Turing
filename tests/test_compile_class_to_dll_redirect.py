"""``compile_class_to_dll`` is deprecated, loudly, and redirected.

The adapter used to call the legacy ``compile_ast_fortran_c_shell``, which
derives no record ABI for the receiver.  Program::

    def __init__(self, c, n):
        self.n = 0
        if c:
            self.n = n

Wrong record: every field cell was float64 and both stores ran at entry,
outside the ``if``, so ``get()`` returned 0 for ``c`` true, and an int above
2**53 could not survive the float64 cell.  The redirect lowers through the
canonical repository-SSA compiler with every method seeded, so the store
lands in its arm in an int64 column.
"""

from __future__ import annotations

import ctypes
import os
import warnings
from pathlib import Path

import pytest

from src.compiler.compile_section_to_dll import compile_class_to_dll
from src.compiler.compiler_entrypoints import CANONICAL_SOURCE_COMPILER
from src.compiler.ssa_fortran_backend import fortran_compiler


CONDITIONAL = '''
class Toy:
    def __init__(self, c: bool, n: int) -> None:
        self.n = 0
        if c:
            self.n = n

    def get(self) -> int:
        return self.n
'''


def _compile(tmp_path):
    with warnings.catch_warnings(record=True) as caught:
        # The adapter forces its own warning on; the caller's filter to
        # ignore everything must not hide it.
        warnings.simplefilter("ignore")
        handle = compile_class_to_dll(CONDITIONAL, "Toy", str(tmp_path / "out"))
    return handle, caught


@pytest.mark.skipif(
    fortran_compiler() is None, reason="the class route emits through Fortran",
)
def test_conditional_field_store_and_wide_int_round_trip(tmp_path):
    handle, caught = _compile(tmp_path)
    deprecations = [
        str(item.message) for item in caught
        if issubclass(item.category, DeprecationWarning)
    ]
    assert any(
        "compile_class_to_dll is deprecated" in message
        and CANONICAL_SOURCE_COMPILER in message
        for message in deprecations
    ), deprecations

    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(Path(fortran_compiler()).resolve().parent))
    library = ctypes.CDLL(str(handle.executable_path))
    i64 = ctypes.c_int64
    init = library.Toy____init__
    init.restype = None
    init.argtypes = [
        ctypes.c_int32, ctypes.POINTER(i64), ctypes.c_bool, i64,
        ctypes.POINTER(i64),
    ]
    get = library.Toy__get
    get.restype = None
    get.argtypes = [ctypes.c_int32, ctypes.POINTER(i64), ctypes.POINTER(i64)]
    for c in (False, True):
        for n in (7, 2 ** 60 + 1):
            column = (i64 * 1)()
            receipt = i64()
            init(1, column, c, n, ctypes.byref(receipt))
            result = i64()
            get(1, column, ctypes.byref(result))
            assert result.value == (n if c else 0), (c, n, result.value)
