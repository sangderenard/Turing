"""WebGPU compute payoff: chosen from the public-entry module's planned regions."""

from __future__ import annotations

import pytest

from src.compiler.deployment_classification import (
    WebGPUComputeCostModel,
    webgpu_compute_payoff,
)


@pytest.fixture(scope="module")
def lowered():
    from tools.build_pairwise_webgpu_demo import lower_block

    return {size: lower_block(*size) for size in ((4, 4), (24, 256), (256, 256))}


def test_pairwise_block_pays_at_realistic_sizes_not_tiny(lowered):
    verdicts = {size: webgpu_compute_payoff(*lowered[size]) for size in lowered}
    assert not verdicts[(4, 4)].pays
    assert verdicts[(24, 256)].pays
    assert verdicts[(256, 256)].pays
    big = verdicts[(24, 256)]
    assert big.data_parallel_extent == 24 * 256
    assert big.bytes_out == 24 * 4
    assert verdicts[(256, 256)].speedup > big.speedup


def test_cost_model_is_the_only_threshold(lowered):
    strict = WebGPUComputeCostModel(min_speedup=1e6)
    assert not webgpu_compute_payoff(*lowered[(256, 256)], model=strict).pays


def test_pairwise_shader_is_emitted_from_ssa():
    from tools.build_pairwise_webgpu_demo import emit_shader

    shader = emit_shader(24, 256)
    assert shader.complete
    assert "atan2(" in shader.source and "sqrt(" in shader.source
    assert "for (var j: u32 = 0u; j < 256u" in shader.source
