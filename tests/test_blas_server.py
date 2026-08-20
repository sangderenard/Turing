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
