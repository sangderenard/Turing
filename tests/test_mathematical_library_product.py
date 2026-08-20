"""The outer product synchronizes its BLAS subunit on every surface."""

from __future__ import annotations

import hashlib
import importlib.util
import json

import numpy as np

from src.compiler.kernel_bank import open_blas_bank
from src.compiler.mathematical_library_product import (
    MATRIX_SCHEMA,
    PRODUCT_SCHEMA,
    build_mathematical_library_product,
)


def _load_generated(path):
    spec = importlib.util.spec_from_file_location("generated_turing_math", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_outer_math_product_owns_a_synchronized_blas_subunit(tmp_path):
    product = build_mathematical_library_product(
        open_blas_bank(tmp_path / "bank"), (17,), tmp_path / "math",
        contract="fast", cores=2, candidate_sizes=(8,),
    )
    manifest = json.loads(product.manifest_path.read_text(encoding="utf-8"))
    matrix_bytes = product.matrix_path.read_bytes()
    matrix = json.loads(matrix_bytes)
    assert manifest["schema"] == PRODUCT_SCHEMA
    assert matrix["schema"] == MATRIX_SCHEMA
    assert hashlib.sha256(matrix_bytes).hexdigest() == manifest["product_id"]
    assert list(matrix["products"]) == ["blas"]
    assert manifest["libraries"]["blas"]["methods"] == [
        "scal", "axpy", "dot", "gemv", "gemm", "rot",
    ]
    coverage = {
        item["method"]: item for item in matrix["products"]["blas"]["coverage"]
    }
    assert coverage["gemm"]["realizations"]["native"]["status"] == "packaged"
    assert coverage["gemm"]["realizations"]["webgpu"]["status"] == "packaged"
    assert coverage["axpy"]["realizations"]["native"]["status"] == "packaged"
    assert coverage["axpy"]["realizations"]["python"]["status"] == "packaged"
    assert coverage["axpy"]["realizations"]["webgpu"]["status"] == "packaged"

    generated = _load_generated(product.python_loader)
    math = generated.load(product.directory)
    try:
        assert math.libraries == ("blas",)
        assert math.blas.methods == (
            "scal", "axpy", "dot", "gemv", "gemm", "rot",
        )
        rng = np.random.default_rng(311)
        a = rng.standard_normal((17, 17))
        b = rng.standard_normal((17, 17))
        assert np.max(np.abs(math.blas.gemm(a, b) - a @ b)) < 1.0e-10
        class Host:
            pass

        assert math.install(Host) is math
        assert Host.math is math
        from src.common.tensors.abstraction import AbstractTensor

        semantic = AbstractTensor.math
        assert math.install(AbstractTensor) is math
        assert AbstractTensor.compiled_math is math
        assert AbstractTensor.math.product is math
        tensor = AbstractTensor.get_tensor([1.0, 2.0, 3.0])
        assert AbstractTensor.blas.scal(tensor, 2.0).tolist() == [2.0, 4.0, 6.0]
        assert AbstractTensor.use_semantic_mathematical_library() is semantic
        assert AbstractTensor.math is semantic
    finally:
        math.close()

    assert matrix_bytes in product.wasm_path.read_bytes()
    javascript = product.javascript_path.read_text(encoding="utf-8")
    assert "class TuringMathematicalLibrary" in javascript
    assert "this.blas=blas" in javascript
    assert "../libraries/blas/web/blas-server.js" in javascript
    header = product.directory / manifest["surfaces"]["native"]["header"]
    assert "turing_blas_server.h" in header.read_text(encoding="utf-8")
    native_blas = manifest["surfaces"]["native"]["libraries"]["blas"]
    assert (product.directory / native_blas["library"]).is_file()
    assert (product.directory / native_blas["header"]).is_file()
    for relative, record in manifest["artifacts"].items():
        artifact = product.directory / relative
        assert artifact.stat().st_size == record["bytes"]
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == record["sha256"]
