"""The math hierarchy is derived from authored semantic material."""

from __future__ import annotations

import hashlib
import inspect
import json
import math

from src.common.tensors.blas import BLAS_ROLES
from src.common.tensors.abstraction_methods import trigonometry as trig_surface
from src.common.tensors.mathematical_library import (
    BLAS_LIBRARY,
    CATALOG_SCHEMA,
    LINALG_LIBRARY,
    TURING_MATHEMATICAL_LIBRARY,
    TRIGONOMETRY_LIBRARY,
)


def test_blas_sublibrary_discovers_every_authored_method_in_order():
    assert tuple(method.name for method in BLAS_LIBRARY.methods) == tuple(BLAS_ROLES)
    assert TURING_MATHEMATICAL_LIBRARY.library("blas") is BLAS_LIBRARY
    assert BLAS_LIBRARY.method("gemm").identity == "blas.gemm"


def test_trigonometry_is_a_second_canonical_sublibrary():
    assert TURING_MATHEMATICAL_LIBRARY.library(
        "trigonometry"
    ) is TRIGONOMETRY_LIBRARY
    existing = tuple(
        name for name, function in vars(trig_surface).items()
        if inspect.isfunction(function)
        and tuple(inspect.signature(function).parameters)[:1] == ("self",)
    )
    assert tuple(method.name for method in TRIGONOMETRY_LIBRARY.methods) == existing
    assert TRIGONOMETRY_LIBRARY.method("sin").identity == "trigonometry.sin"


def test_existing_linalg_namespace_is_the_third_canonical_sublibrary():
    assert TURING_MATHEMATICAL_LIBRARY.library("linalg") is LINALG_LIBRARY
    assert tuple(method.name for method in LINALG_LIBRARY.methods) == (
        "eye", "dot", "norm", "cross", "trace", "det", "solve", "inv",
        "eigh", "cholesky",
    )
    assert all(
        method.installation == "namespace_operator:linalg"
        for method in LINALG_LIBRARY.methods
    )


def test_method_signatures_are_derived_from_authored_source():
    gemm = BLAS_LIBRARY.method("gemm")
    assert [item.to_mapping() for item in gemm.parameters] == [
        {"name": "A", "kind": "buffer", "access": "read"},
        {"name": "B", "kind": "buffer", "access": "read"},
        {"name": "C", "kind": "buffer", "access": "read_write"},
        {"name": "alpha", "kind": "scalar", "access": "read"},
        {"name": "beta", "kind": "scalar", "access": "read"},
        {"name": "m", "kind": "extent", "access": "read"},
        {"name": "n", "kind": "extent", "access": "read"},
        {"name": "k", "kind": "extent", "access": "read"},
    ]
    assert gemm.result == {"kind": "parameter", "parameter": "C"}
    assert gemm.abstract_operators == ("matmul",)
    dot = BLAS_LIBRARY.method("dot")
    assert dot.result == {"kind": "scalar"}
    rot = BLAS_LIBRARY.method("rot")
    assert [item.access for item in rot.parameters[:2]] == [
        "read_write", "read_write",
    ]


def test_catalog_mapping_is_self_contained_and_canonicalizable():
    mapping = TURING_MATHEMATICAL_LIBRARY.to_mapping()
    assert mapping["schema"] == CATALOG_SCHEMA
    gemm = mapping["libraries"][0]["methods"][4]
    assert hashlib.sha256(gemm["source"].encode("utf-8")).hexdigest() == gemm[
        "source_sha256"
    ]
    first = json.dumps(mapping, sort_keys=True, separators=(",", ":"))
    second = json.dumps(
        TURING_MATHEMATICAL_LIBRARY.to_mapping(),
        sort_keys=True,
        separators=(",", ":"),
    )
    assert first == second


def test_abstract_tensor_receives_the_same_outer_blas_hierarchy():
    from src.common.tensors.abstraction import AbstractTensor

    assert AbstractTensor.math.libraries == ("blas", "trigonometry", "linalg")
    assert AbstractTensor.math.blas is AbstractTensor.blas
    assert AbstractTensor.blas.methods == tuple(BLAS_ROLES)
    assert AbstractTensor.trigonometry.methods == tuple(
        method.name for method in TRIGONOMETRY_LIBRARY.methods
    )
    assert AbstractTensor.math.linalg is AbstractTensor.linalg
    x = AbstractTensor.get_tensor([1.0, 2.0, 3.0])
    y = AbstractTensor.get_tensor([4.0, 5.0, 6.0])
    assert AbstractTensor.blas.scal(x, 2.0).tolist() == [2.0, 4.0, 6.0]
    assert AbstractTensor.blas.axpy(x, y, 2.0).tolist() == [6.0, 9.0, 12.0]
    assert AbstractTensor.blas.dot(x, y).item() == 32.0
    rx, ry = AbstractTensor.blas.rot(x, y, 0.8, 0.6)
    assert rx.tolist() == [3.2, 4.6, 6.0]
    assert all(abs(a - b) < 1.0e-12 for a, b in zip(
        ry.tolist(), [2.6, 2.8, 3.0],
    ))
    np_values = [0.25, 0.5]
    assert all(abs(a - b) < 1.0e-12 for a, b in zip(
        AbstractTensor.trigonometry.sin(np_values).tolist(),
        [math.sin(value) for value in np_values],
    ))


def test_abstract_tensor_semantic_namespace_can_be_restored():
    from src.common.tensors.abstraction import AbstractTensor

    semantic = AbstractTensor.math
    assert AbstractTensor.use_semantic_mathematical_library() is semantic
    assert AbstractTensor.math is semantic
    assert AbstractTensor.blas is semantic.blas
    assert AbstractTensor.trigonometry is semantic.trigonometry
