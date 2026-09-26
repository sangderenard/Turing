"""A prebaked GEMM matrix becomes a standalone native pooled product."""

from __future__ import annotations

import ctypes

import numpy as np

from src.compiler.kernel_bank import open_blas_bank
from src.compiler.native_gemm_product import compile_native_gemm_product


def test_prebaked_gemm_product_compiles_runs_and_publishes_edges(tmp_path):
    size, tile = 65, 32
    bank = open_blas_bank(tmp_path / "bank")
    product = compile_native_gemm_product(
        bank, {"m": size, "n": size, "k": size}, tmp_path / "product",
        contract="fast", cores=2, candidate_sizes=(tile,),
    )

    library = ctypes.CDLL(str(product.library_path))
    function = getattr(library, product.function_name)
    pointer = ctypes.POINTER(ctypes.c_double)
    function.argtypes = [
        pointer, pointer, pointer, ctypes.c_double, ctypes.c_double,
    ]
    function.restype = ctypes.c_int
    serial = getattr(library, product.function_name + "_serial")
    serial.argtypes = function.argtypes
    serial.restype = ctypes.c_int
    shutdown = getattr(library, product.function_name + "_shutdown")
    shutdown.restype = None
    rng = np.random.default_rng(19)
    a = np.ascontiguousarray(rng.standard_normal((size, size)))
    b = np.ascontiguousarray(rng.standard_normal((size, size)))
    original = np.ascontiguousarray(rng.standard_normal((size, size)))
    result = original.copy()
    serial_result = original.copy()
    alpha, beta = 1.25, 0.5
    try:
        status = function(
            a.ctypes.data_as(pointer), b.ctypes.data_as(pointer),
            result.ctypes.data_as(pointer), alpha, beta,
        )
        serial_status = serial(
            a.ctypes.data_as(pointer), b.ctypes.data_as(pointer),
            serial_result.ctypes.data_as(pointer), alpha, beta,
        )
    finally:
        shutdown()

    expected = alpha * (a @ b) + beta * original
    assert status == 0
    assert serial_status == 0
    assert np.max(np.abs(result - expected)) < 1e-9
    assert np.max(np.abs(serial_result - expected)) < 1e-9
    assert product.manifest["python_runtime_dependency"] is False
    assert product.manifest["launch"] == {
        "workers": 1, "chunk_size": 2, "lane_count": 9,
        "join": "barrier",
    }
    assert product.manifest["compiler_decision"]["tile"] == tile
    assert product.manifest["deployment_choice"]["strategy"] == "pool"
    source = product.source_path.read_text(encoding="utf-8")
    assert "turing_pool_start(1)" in source
    assert "turing_pool_deploy_span(" in source
    assert product.manifest["serial_control"].endswith("_serial")
    assert product.manifest["core"]["symbol"] in source
