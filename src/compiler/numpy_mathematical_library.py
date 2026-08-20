"""Manifest the canonical mathematical library as standalone NumPy Python.

The numerical bodies in the generated module are not maintained as a second
BLAS implementation.  They are recorded through ``AbstractTensor`` and
rendered by the tape-to-source compiler's NumPy dialect.  The small class
shell supplies the stable public calling convention around those compiled
functions and has no Turing dependency at runtime.
"""

from __future__ import annotations

import ast
import hashlib
import json
from typing import Any, Callable

import numpy as np

from ..common.tensors.accelerator_backends.tape_to_source import emit_tape_source
from ..common.tensors.autograd import autograd
from ..common.tensors.mathematical_library import AbstractTensorBLAS, BLAS_LIBRARY
from ..common.tensors.numpy_backend import NumPyTensorOperations


NUMPY_LIBRARY_SCHEMA = "turing.numpy-mathematical-library.v1"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tensor(value: Any):
    return NumPyTensorOperations.tensor(np.asarray(value, dtype=np.float64))


def _trace_cases() -> dict[str, tuple[dict[str, Any], Callable[[dict[str, Any]], Any]]]:
    blas = AbstractTensorBLAS(NumPyTensorOperations)
    return {
        "scal": (
            {"x": _tensor([1, 2, 3]), "alpha": _tensor(2.0)},
            lambda p: blas.scal(p["x"], p["alpha"]),
        ),
        "axpy": (
            {
                "x": _tensor([1, 2, 3]), "y": _tensor([4, 5, 6]),
                "alpha": _tensor(2.0),
            },
            lambda p: blas.axpy(p["x"], p["y"], p["alpha"]),
        ),
        "dot": (
            {"x": _tensor([1, 2, 3]), "y": _tensor([4, 5, 6])},
            lambda p: blas.dot(p["x"], p["y"]),
        ),
        "gemv": (
            {
                "a": _tensor([[1, 2], [3, 4]]), "x": _tensor([2, 3]),
                "y": _tensor([5, 7]), "alpha": _tensor(1.5),
                "beta": _tensor(0.25),
            },
            lambda p: blas.gemv(
                p["a"], p["x"], y=p["y"],
                alpha=p["alpha"], beta=p["beta"],
            ),
        ),
        "gemm": (
            {
                "a": _tensor([[1, 2], [3, 4]]),
                "b": _tensor([[2, 0], [1, 2]]),
                "c": _tensor([[1, 1], [1, 1]]),
                "alpha": _tensor(1.5), "beta": _tensor(0.25),
            },
            lambda p: blas.gemm(
                p["a"], p["b"], c=p["c"],
                alpha=p["alpha"], beta=p["beta"],
            ),
        ),
        "rot": (
            {
                "x": _tensor([1, 2]), "y": _tensor([3, 4]),
                "c": _tensor(0.8), "s": _tensor(0.6),
            },
            lambda p: blas.rot(p["x"], p["y"], p["c"], p["s"]),
        ),
    }


def _compiled_functions() -> tuple[list[str], dict[str, dict[str, str]]]:
    functions = []
    records = {}
    for method in BLAS_LIBRARY.methods:
        inputs, invoke = _trace_cases()[method.name]
        with autograd.forward_capture() as tape:
            result = invoke(inputs)
        outputs = (
            {"x": result[0], "y": result[1]}
            if isinstance(result, tuple) else {"result": result}
        )
        emitted = emit_tape_source(
            tape, inputs, outputs, backend="numpy",
            function_name=f"_compiled_{method.name}",
        )
        tree = ast.parse(emitted)
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
        source = ast.unparse(function) + "\n"
        functions.append(source)
        records[method.name] = {
            "identity": method.identity,
            "role_source_sha256": method.source_sha256,
            "compiled_source_sha256": _sha(source),
        }
    return functions, records


_CLASS_SHELL = '''
class NumPyBLAS:
    """Standalone NumPy realization of the compiler's canonical BLAS graph."""
    methods = ("scal", "axpy", "dot", "gemv", "gemm", "rot")

    @staticmethod
    def scal(x, alpha, *, y=None):
        return _compiled_scal(np.asarray(x), np.asarray(alpha))

    @staticmethod
    def axpy(x, y, alpha):
        return _compiled_axpy(np.asarray(x), np.asarray(y), np.asarray(alpha))

    @staticmethod
    def dot(x, y):
        return _compiled_dot(np.asarray(x), np.asarray(y))

    @staticmethod
    def gemv(a, x, *, y=None, alpha=1.0, beta=0.0):
        a, x = np.asarray(a), np.asarray(x)
        if y is None:
            y = np.zeros(a.shape[0], dtype=np.result_type(a, x, float))
        return _compiled_gemv(
            a, x, np.asarray(y), np.asarray(alpha), np.asarray(beta),
        )

    @staticmethod
    def gemm(a, b, *, c=None, alpha=1.0, beta=0.0):
        a, b = np.asarray(a), np.asarray(b)
        if c is None:
            c = np.zeros((a.shape[0], b.shape[1]), dtype=np.result_type(a, b, float))
        return _compiled_gemm(
            a, b, np.asarray(c), np.asarray(alpha), np.asarray(beta),
        )

    @staticmethod
    def rot(x, y, c, s):
        return _compiled_rot(
            np.asarray(x), np.asarray(y), np.asarray(c), np.asarray(s),
        )


class NumPyMathematicalLibrary:
    """Standalone NumPy mathematical-library product."""
    libraries = ("blas",)

    def __init__(self):
        self.blas = NumPyBLAS()
        self.manifest = COMPILER_MANIFEST

    def install(self, host, attribute="math"):
        hook = getattr(host, "install_mathematical_library", None)
        if hook is not None:
            return hook(self)
        setattr(host, str(attribute), self)
        return self

    def close(self):
        return None


def load():
    return NumPyMathematicalLibrary()
'''


def emit_numpy_mathematical_library() -> tuple[str, dict[str, Any]]:
    """Return deterministic standalone source and its compiler receipt."""

    functions, records = _compiled_functions()
    receipt: dict[str, Any] = {
        "schema": NUMPY_LIBRARY_SCHEMA,
        "compiler": "abstract-tensor-tape-to-source:numpy",
        "methods": records,
    }
    receipt_source = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
    source = (
        '"""Generated standalone NumPy Turing mathematical library."""\n'
        "from __future__ import annotations\n"
        "import numpy as np\n\n"
        f"COMPILER_MANIFEST = {receipt_source}\n\n"
        + "\n".join(functions)
        + _CLASS_SHELL
    )
    ast.parse(source)
    receipt["module_source_sha256"] = _sha(source)
    return source, receipt


__all__ = [
    "NUMPY_LIBRARY_SCHEMA",
    "emit_numpy_mathematical_library",
]
