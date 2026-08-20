"""The BLAS server is one verifiable product on native, Python, and Web."""

from __future__ import annotations

import hashlib
import importlib.util
import json

import numpy as np
import pytest

from src.compiler.blas_server import (
    MATRIX_SCHEMA,
    SERVER_SCHEMA,
    build_blas_server,
)
from src.compiler.kernel_bank import open_blas_bank


def _load_generated(path):
    spec = importlib.util.spec_from_file_location("generated_blas_server", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_blas_server_packages_and_dispatches_multiple_specializations(tmp_path):
    product = build_blas_server(
        open_blas_bank(tmp_path / "bank"),
        ((17, 17, 17), (25, 25, 25)),
        tmp_path / "product",
        contract="fast",
        cores=2,
        candidate_sizes=(8,),
    )

    manifest = json.loads(product.manifest_path.read_text(encoding="utf-8"))
    matrix_bytes = product.matrix_path.read_bytes()
    matrix = json.loads(matrix_bytes)
    digest = hashlib.sha256(matrix_bytes).hexdigest()
    assert manifest["schema"] == SERVER_SCHEMA
    assert matrix["schema"] == MATRIX_SCHEMA
    assert manifest["product_id"] == digest
    assert manifest["server_matrix_sha256"] == digest
    assert manifest["methods"] == ["scal", "axpy", "dot", "gemv", "gemm", "rot"]
    assert manifest["deployed_roles"] == [
        "blas.scal", "blas.axpy", "blas.dot", "blas.gemv", "blas.gemm", "blas.rot",
    ]
    assert manifest["surface_roles"]["webgpu"] == manifest["deployed_roles"]
    assert [item["name"] for item in matrix["library"]["methods"]] == manifest[
        "methods"
    ]
    assert matrix["surface_methods"]["python"] == manifest["methods"]
    assert matrix["surface_methods"]["webgpu"] == manifest["methods"]
    assert set(matrix["webgpu_prebakes"]) == {
        "scal", "axpy", "dot", "gemv", "rot",
    }
    assert all(matrix["webgpu_prebakes"].values())
    assert manifest["surfaces"]["native"]["python_runtime_dependency"] is False
    assert manifest["surfaces"]["web"]["python_runtime_dependency"] is False
    assert [item["shape"] for item in matrix["variants"]] == [
        [17, 17, 17], [25, 25, 25],
    ]

    generated = _load_generated(product.python_loader)
    server = generated.load(product.directory)
    rng = np.random.default_rng(811)
    try:
        assert server.shapes == ((17, 17, 17), (25, 25, 25))
        assert server.methods == ("scal", "axpy", "dot", "gemv", "gemm", "rot")
        assert server.deployed_methods == server.methods
        assert server.supports("gemm") and server.supports("axpy")
        x = rng.standard_normal(19)
        y = rng.standard_normal(19)
        assert np.allclose(server.scal(x, 1.25), 1.25 * x)
        assert np.allclose(server.axpy(x, y, 1.25), 1.25 * x + y)
        assert abs(server.dot(x, y) - float(x @ y)) < 1.0e-10
        a = rng.standard_normal((13, 19))
        initial = rng.standard_normal(13)
        assert np.allclose(
            server.gemv(a, x, y=initial, alpha=1.25, beta=0.5),
            1.25 * (a @ x) + 0.5 * initial,
        )
        rx, ry = server.rot(x, y, 0.8, 0.6)
        assert np.allclose(rx, 0.8 * x + 0.6 * y)
        assert np.allclose(ry, 0.8 * y - 0.6 * x)
        for size in (17, 25):
            a = rng.standard_normal((size, size))
            b = rng.standard_normal((size, size))
            c = rng.standard_normal((size, size))
            actual = server.gemm(a, b, c=c, alpha=1.25, beta=0.5)
            expected = 1.25 * (a @ b) + 0.5 * c
            assert np.max(np.abs(actual - expected)) < 1.0e-10
        with pytest.raises(KeyError, match="not prebaked"):
            server.gemm(np.eye(19), np.eye(19))
    finally:
        server.close()

    assert matrix_bytes in product.wasm_path.read_bytes()
    javascript = product.javascript_path.read_text(encoding="utf-8")
    assert "WebAssembly.instantiate" in javascript
    assert 'crypto.subtle.digest("SHA-256"' in javascript
    assert "navigator.gpu.requestAdapter" in javascript
    assert "binding:3" in javascript
    assert "alpha,beta" in javascript
    assert 'join("\\n")' in javascript
    assert "get methods()" in javascript
    assert "deployedMethods" in javascript
    assert "async scal(" in javascript
    assert "async axpy(" in javascript
    assert "async dot(" in javascript
    assert "async gemv(" in javascript
    assert "async rot(" in javascript
    assert (product.directory / "native" / "turing_blas_server.h").is_file()
    assert (product.directory / "README.md").is_file()
    assert not (product.directory / ".build").exists()

    for relative, record in manifest["artifacts"].items():
        artifact = product.directory / relative
        assert artifact.is_file()
        assert artifact.stat().st_size == record["bytes"]
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == record["sha256"]


def test_blas_server_refuses_to_replace_an_unowned_directory(tmp_path):
    output = tmp_path / "not-a-product"
    output.mkdir()
    (output / "notes.txt").write_text("keep me", encoding="utf-8")
    with pytest.raises(RuntimeError, match="refusing to replace"):
        build_blas_server(
            open_blas_bank(tmp_path / "bank"), (17,), output,
            cores=1, candidate_sizes=(8,),
        )
    assert (output / "notes.txt").read_text(encoding="utf-8") == "keep me"
