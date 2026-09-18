from __future__ import annotations

import hashlib

import numpy as np

from src.compiler.numpy_mathematical_library import (
    NUMPY_LIBRARY_SCHEMA,
    emit_numpy_mathematical_library,
)


def _load(source):
    namespace = {}
    exec(compile(source, "<generated-numpy-math>", "exec"), namespace)
    return namespace


def test_compiler_manifests_standalone_numpy_math_deterministically():
    source, receipt = emit_numpy_mathematical_library()
    repeated_source, repeated_receipt = emit_numpy_mathematical_library()
    assert source == repeated_source
    assert receipt == repeated_receipt
    assert receipt["schema"] == NUMPY_LIBRARY_SCHEMA
    assert receipt["module_source_sha256"] == hashlib.sha256(
        source.encode("utf-8")
    ).hexdigest()
    assert "src.common.tensors" not in source
    assert "_compiled_gemm" in source
    assert "a @ b" in source
    assert "np.sum" in source


def test_standalone_numpy_class_realizes_every_blas_graph():
    source, _receipt = emit_numpy_mathematical_library()
    library = _load(source)["load"]()
    assert library.libraries == ("blas",)
    assert library.blas.methods == ("scal", "axpy", "dot", "gemv", "gemm", "rot")

    x = np.asarray([1.0, 2.0, 3.0])
    y = np.asarray([4.0, 5.0, 6.0])
    assert np.allclose(library.blas.scal(x, 2.0), 2.0 * x)
    assert np.allclose(library.blas.axpy(x, y, 2.0), 2.0 * x + y)
    assert library.blas.dot(x, y) == np.dot(x, y)

    a = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    b = np.asarray([[2.0, 0.0], [1.0, 2.0]])
    assert np.allclose(library.blas.gemv(a, x[:2]), a @ x[:2])
    assert np.allclose(library.blas.gemm(a, b), a @ b)
    rx, ry = library.blas.rot(x, y, 0.8, 0.6)
    assert np.allclose(rx, 0.8 * x + 0.6 * y)
    assert np.allclose(ry, 0.8 * y - 0.6 * x)
