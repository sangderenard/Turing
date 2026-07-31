from __future__ import annotations

import numpy as np
import pytest

from src.common.tensors.accelerator_backends.tensor_torture import (
    TortureTier,
    eager_backend_result,
    tensor_torture_cases,
)
from src.common.tensors.numpy_backend import NumPyTensorOperations
from src.common.tensors.torch_backend import PyTorchTensorOperations, torch


CASES = tensor_torture_cases()


def _assert_outputs(case, actual):
    expected = case.numpy_reference()
    assert actual.keys() == expected.keys()
    for name in expected:
        np.testing.assert_allclose(
            actual[name],
            expected[name],
            rtol=case.rtol,
            atol=case.atol,
            equal_nan=True,
            err_msg=f"{case.name}:{name}",
        )


def test_torture_corpus_has_three_distinct_topology_tiers():
    tiers = {case.tier for case in CASES}
    assert tiers == {
        TortureTier.ISOLATED,
        TortureTier.GRAB_BAG,
        TortureTier.ADVANCED,
    }
    assert len({case.semantic_digest for case in CASES}) == len(CASES)


def test_large_torture_tier_matches_requested_frame_batch_scale():
    large = {
        case.name: case
        for case in tensor_torture_cases(include_large=True)
        if case.tier is TortureTier.LARGE
    }

    # The bespoke frame/reduction/matmul cases remain.
    assert {
        "large_frame_batch",
        "large_reduction",
        "large_matmul",
    } <= set(large)
    assert large["large_frame_batch"].inputs["left"].shape == (8, 512, 512)
    assert large["large_reduction"].inputs["frames"].shape == (8, 512, 512)
    assert large["large_matmul"].inputs["left"].shape == (256, 256)


def test_large_tier_repeats_every_operator_at_scale():
    """Which backend wins depends on size, so the operators must be measured
    at more than one shape.  The large tier reuses the isolated definitions
    rather than inventing new ones, so a tier difference is a size difference."""

    cases = tensor_torture_cases(include_large=True)
    isolated = {
        case.name for case in cases if case.tier is TortureTier.ISOLATED
    }
    large = {
        case.name: case
        for case in cases
        if case.tier is TortureTier.LARGE
    }

    # Every isolated operator reappears at scale under a suffixed name.
    assert {f"{name}_large" for name in isolated} <= set(large)

    scaled = large["add_large"]
    assert scaled.inputs["left"].shape == (1024, 1024)
    assert scaled.inputs["left"].size == 1024 * 1024
    # Matmul is sized separately because its cost is cubic.
    assert large["matmul_large"].inputs["left"].shape == (256, 256)


def test_scaled_operator_cases_stay_numerically_well_formed():
    """A reshape target hard-coded for 24 elements would not survive scaling."""

    large = {
        case.name: case
        for case in tensor_torture_cases(include_large=True)
        if case.tier is TortureTier.LARGE
    }
    result = large["reshape_transpose_large"].numpy_reference()["result"]
    assert result.size == 1024 * 1024
    assert result.ndim == 3


def test_precompile_observer_preserves_integer_dtype_for_indexing():
    from src.common.tensors.abstraction import AbstractTensor
    from src.common.tensors.accelerator_backends.precompile_observer_backend import (
        PrecompileObserverTensorOperations,
    )

    with AbstractTensor.use_backend("precompile_observer"):
        values = AbstractTensor.tensor([10, 20, 30], dtype=np.int64)
        indices = AbstractTensor.tensor([2, 0], dtype=np.int64)
        selected = values[indices]

    assert isinstance(values, PrecompileObserverTensorOperations)
    assert values.data.dtype == np.dtype(np.int64)
    assert indices.data.dtype == np.dtype(np.int64)
    np.testing.assert_equal(selected.data, np.asarray([30, 10]))


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_numpy_abstract_tensor_backend_matches_raw_numpy(case):
    _assert_outputs(
        case,
        eager_backend_result(case, NumPyTensorOperations),
    )


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_torch_abstract_tensor_backend_matches_raw_numpy(case):
    _assert_outputs(
        case,
        eager_backend_result(case, PyTorchTensorOperations),
    )
