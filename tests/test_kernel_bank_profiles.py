"""The bank's auto-collected performance charts and source-derived layout.

Two facts the deployment strategy layer is starved without, both
established here:

* **Timing**: every admitted variant is profiled AT BUILD TIME with a
  LAUNCH average (preparation + dispatch, measured by clearing the
  execution cache) and a COMPUTE average (warm steady-state repeats),
  recorded in its manifest and readable as a chart
  (``KernelBank.performance_chart``). The single cold admission timing
  that existed before conflated the two, which is exactly the bias that
  made admission-probe throughput unusable as strategy evidence.

* **Per-item data size**: how a custom loop's data partitions across the
  items of a deployment matrix. Derived from the AUTHORED SOURCE -- the
  loop bounds and index arithmetic are the single authority
  (``derive_extents_from_source`` / ``derive_size_parameters_from_source``)
  -- never from a hand-maintained mirror table, which would drift the
  moment a kernel changes. ``KernelSpec.item_data(axis, sizes)`` answers
  what one index of an axis consumes of every parameter: split parameters
  (an A row, a C row) versus shared ones (all of B), which is the byte
  math a lane/chunk hand-off needs. Undeclared or underivable extents
  REFUSE -- a partitioner guessing byte ranges is how buffers get torn.

The cross-checks are dynamic on purpose: derived extents are asserted
against what ``example_inputs`` actually allocates -- real code checking
real code, no restated expectations.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from src.compiler.kernel_bank import (
    BankRefusal,
    KernelBank,
    KernelSpec,
    blas_kernel_specs,
    derive_extents_from_source,
    derive_size_parameters_from_source,
)


@pytest.fixture(scope="module")
def specs():
    return blas_kernel_specs()


@pytest.fixture(scope="module")
def bank(tmp_path_factory, specs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        built = KernelBank(tmp_path_factory.mktemp("profile_bank"), specs)
        built.get("dot")  # one admitted variant is enough for the charts
        return built


def test_every_blas_extent_matches_what_example_inputs_allocates(specs):
    rng = np.random.default_rng(0)
    for spec in specs.values():
        sizes = {
            parameter: 5 + 2 * index
            for index, parameter in enumerate(spec.size_parameters)
        }
        sample = spec.example_inputs(sizes, rng)
        for parameter, dims in spec.extents.items():
            if not dims:
                continue
            expected = 1
            for dim in dims:
                expected *= sizes[dim]
            assert np.size(sample[parameter]) == expected, (
                spec.name, parameter, dims,
            )


def test_size_parameters_are_read_from_the_loops_not_a_table(specs):
    # The derivation and the arity-supplied parameter order must agree on
    # membership for every kernel -- checked against the live source.
    for spec in specs.values():
        derived = derive_size_parameters_from_source(
            spec.source, spec.function_name
        )
        assert set(derived) <= set(spec.parameter_order), spec.name
        assert derived == spec.size_parameters


def test_item_data_splits_owned_rows_and_shares_the_rest(specs):
    gemm = specs["gemm"]
    sizes = {"m": 6, "n": 7, "k": 9}
    partition = gemm.item_data("m", sizes)
    # One m-item owns one A row (k elements) and one C row (n elements);
    # B is shared whole. Derived from the kernel's own index arithmetic.
    assert partition["split"] == {"A": 9, "C": 7}
    assert partition["shared"] == {"B": 63}


def test_underivable_indexing_refuses_instead_of_guessing():
    with pytest.raises(BankRefusal) as excinfo:
        derive_extents_from_source(
            "def f(a, n):\n"
            "    for i in range(n):\n"
            "        a[i * i] = 0.0\n"
            "    return a\n",
            "f",
        )
    assert "cannot derive" in str(excinfo.value)


def test_item_data_without_declared_extents_refuses():
    spec = KernelSpec(
        name="bare", source="def f(x):\n    return x\n",
        function_name="f", reference=lambda x: x,
        parameter_order=("x",), size_parameters=(),
        example_inputs=lambda sizes, rng: {"x": 0.0},
    )
    with pytest.raises(BankRefusal):
        spec.item_data("n", {"n": 4})


def test_an_admitted_variant_carries_a_launch_and_compute_profile(bank):
    rows = bank.performance_chart("dot")
    assert rows, "an admitted variant must be profiled at build"
    row = rows[-1]
    assert row["compute_avg_seconds"] > 0
    assert row["launch_avg_seconds"] >= 0
    assert row["cold_avg_seconds"] >= row["compute_avg_seconds"]
    assert row["sizes"]  # the chart states what was measured


def test_a_refused_variant_has_no_profile_row(bank):
    # gemm specialized below the unroll limit is the standing refused
    # variant (test_compiled_linalg.py pins the underlying defect).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(BankRefusal):
            bank.get("gemm", specialized={"m": 4, "n": 4, "k": 4})
    charted = {
        tuple(sorted(row["specialized"].items()))
        for row in bank.performance_chart("gemm")
    }
    assert (("k", 4), ("m", 4), ("n", 4)) not in charted
