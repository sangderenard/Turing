"""The math hierarchy is derived from authored semantic material."""

from __future__ import annotations

import hashlib
import json

from src.common.tensors.blas import BLAS_ROLES
from src.common.tensors.mathematical_library import (
    BLAS_LIBRARY,
    CATALOG_SCHEMA,
    TURING_MATHEMATICAL_LIBRARY,
)


def test_blas_sublibrary_discovers_every_authored_method_in_order():
    assert tuple(method.name for method in BLAS_LIBRARY.methods) == tuple(BLAS_ROLES)
    assert TURING_MATHEMATICAL_LIBRARY.library("blas") is BLAS_LIBRARY
    assert BLAS_LIBRARY.method("gemm").identity == "blas.gemm"


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

    assert AbstractTensor.math.libraries == ("blas",)
    assert AbstractTensor.math.blas is AbstractTensor.blas
    assert AbstractTensor.blas.methods == tuple(BLAS_ROLES)
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


def test_abstract_tensor_semantic_namespace_can_be_restored():
    from src.common.tensors.abstraction import AbstractTensor

    semantic = AbstractTensor.math
    assert AbstractTensor.use_semantic_mathematical_library() is semantic
    assert AbstractTensor.math is semantic
    assert AbstractTensor.blas is semantic.blas
