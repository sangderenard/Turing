from __future__ import annotations

import json

from src.compiler.webgpu_benchmark_bundle import (
    build_webgpu_benchmark_manifest,
    write_webgpu_benchmark_bundle,
)


def test_manifest_is_deterministic_and_owns_compiler_emitted_kernels():
    first = build_webgpu_benchmark_manifest(
        counts=(256, 512), gemm_sizes=(16, 32),
    )
    second = build_webgpu_benchmark_manifest(
        counts=(512, 256), gemm_sizes=(32, 16),
    )

    assert first == second
    assert len([k for k in first["kernels"] if k["kind"] == "elementwise"]) == 42
    assert len([k for k in first["kernels"] if k["kind"] == "gemm"]) == 4
    assert all("@compute" in kernel["source"] for kernel in first["kernels"])
    assert all(len(kernel["source_sha256"]) == 64 for kernel in first["kernels"])


def test_bundle_is_a_single_page_webgpu_runner_with_public_manifest(tmp_path):
    bundle = write_webgpu_benchmark_bundle(
        tmp_path, counts=(256,), gemm_sizes=(16,),
    )
    page = bundle.page_path.read_text(encoding="utf-8")
    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))

    assert "navigator.gpu.requestAdapter" in page
    assert "timestamp-query" in page
    assert "queue.onSubmittedWorkDone" in page
    assert "copyBufferToBuffer" in page
    assert "Benchmark selected operation" in page
    assert "__TURING_BENCHMARK_MANIFEST__" not in page
    assert manifest["manifest_sha256"] == bundle.manifest["manifest_sha256"]
