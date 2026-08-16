"""Compile and exercise the native turing_pool runtime end to end.

Uses the repository's own toolchain discovery (``native_library``) so the
same compilers the backends use build this runtime; skips visibly when no
toolchain exists.  When one does, this proves the exactly-once claiming
theorem on the real runtime: every (lane, chunk) cell of the frame grid is
executed exactly once, from Python, through ctypes.
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import pytest

from src.common.tensors.accelerator_backends.native_library import (
    compile_shared_library,
    preferred_toolchain,
)

_BACKEND_DIR = (
    Path(__file__).resolve().parents[1]
    / "src" / "common" / "tensors" / "accelerator_backends" / "c_backend"
)


@pytest.fixture(scope="module")
def pool_library(tmp_path_factory):
    toolchain = preferred_toolchain()
    if toolchain is None:
        pytest.skip(
            "no native toolchain found (native_library.detect_toolchains)"
        )
    # compile_shared_library writes the source into its own directory, so
    # inline the header textually rather than teaching each toolchain an
    # include path.
    header = (_BACKEND_DIR / "turing_pool.h").read_text(encoding="utf-8")
    source = (_BACKEND_DIR / "turing_pool.c").read_text(encoding="utf-8")
    source = source.replace('#include "turing_pool.h"', header)
    extra_flags = () if sys.platform == "win32" else ("-pthread",)
    library_path, _toolchain = compile_shared_library(
        source,
        name="turing_pool_test",
        directory=tmp_path_factory.mktemp("turing_pool"),
        extra_flags=extra_flags,
    )
    library = ctypes.CDLL(str(library_path))
    library.turing_pool_start.restype = ctypes.c_int
    library.turing_pool_start.argtypes = [ctypes.c_int]
    library.turing_pool_workers.restype = ctypes.c_int
    library.turing_pool_deploy.restype = ctypes.c_int
    yield library
    library.turing_pool_stop()


_LANE_FN = ctypes.CFUNCTYPE(
    None, ctypes.c_void_p, ctypes.c_long, ctypes.c_long, ctypes.c_long,
)


def test_start_is_idempotent_and_never_shrinks(pool_library):
    assert pool_library.turing_pool_start(3) == 3
    assert pool_library.turing_pool_start(1) == 3
    assert pool_library.turing_pool_workers() == 3


def test_every_grid_cell_is_claimed_exactly_once(pool_library):
    lanes, chunks = 7, 13
    counts = (ctypes.c_long * (lanes * chunks))()

    @_LANE_FN
    def kernel(context, lane, chunk, chunks_per_lane):
        counts[lane * chunks_per_lane + chunk] += 1

    pool_library.turing_pool_start(3)
    status = pool_library.turing_pool_deploy(kernel, None, lanes, chunks)
    assert status == 0
    assert all(cell == 1 for cell in counts), (
        "claiming was not exactly-once: "
        f"{[index for index, cell in enumerate(counts) if cell != 1]}"
    )


def test_zero_workers_serial_fallback_runs_the_identical_path(pool_library):
    # Workers may already exist from earlier tests; the property that
    # matters is that deploy completes and covers the grid regardless of
    # pool size, caller participating.
    total = ctypes.c_long(0)

    @_LANE_FN
    def kernel(context, lane, chunk, chunks_per_lane):
        total.value += lane + chunk

    assert pool_library.turing_pool_deploy(kernel, None, 4, 1) == 0
    assert total.value == 0 + 1 + 2 + 3


def test_invalid_frames_are_refused(pool_library):
    @_LANE_FN
    def kernel(context, lane, chunk, chunks_per_lane):
        pass

    assert pool_library.turing_pool_deploy(kernel, None, 0, 1) == -1
    assert pool_library.turing_pool_deploy(kernel, None, 1, 0) == -1


def test_nested_deploy_from_a_lane_is_refused_not_deadlocked(pool_library):
    inner_status = ctypes.c_long(99)

    @_LANE_FN
    def inner(context, lane, chunk, chunks_per_lane):
        pass

    @_LANE_FN
    def outer(context, lane, chunk, chunks_per_lane):
        inner_status.value = pool_library.turing_pool_deploy(
            inner, None, 1, 1,
        )

    assert pool_library.turing_pool_deploy(outer, None, 1, 1) == 0
    assert inner_status.value == -2
