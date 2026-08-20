"""The bank's auto-collected performance charts and source-derived layout.

Two facts the deployment strategy layer is starved without, both
established here:

* **Timing**: every admitted variant is profiled AT BUILD TIME with a
  COMPUTE average (warm steady-state medians) and TWO distinct launch
  costs: FIRST launch (library load + first dispatch -- read from the
  admission probe, the only first call a variant ever has) and RELAUNCH
  (fresh-execution preparation, re-measured by clearing the execution
  cache). Recorded in the manifest, readable as a chart
  (``KernelBank.performance_chart``). The single cold admission timing
  that existed before conflated all three, which is exactly the bias
  that made admission-probe throughput unusable as strategy evidence.

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

import json
import shutil
import warnings

import numpy as np
import pytest

from src.compiler.kernel_bank import (
    BankRefusal,
    KernelBank,
    KernelSpec,
    blas_kernel_specs,
    derive_extents_from_source,
    derive_access_signature_from_source,
    derive_size_parameters_from_source,
    parameter_layout_permutation,
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


def test_gemm_access_signature_exposes_the_profitable_unit_strides(specs):
    signature = derive_access_signature_from_source(
        specs["gemm"].source, "gemm",
    )
    by_parameter = {}
    for access in signature:
        by_parameter.setdefault(access["parameter"], []).append(
            dict(access["loop_strides"])
        )

    assert any(strides["p"] == "1" for strides in by_parameter["A"])
    assert any(strides["j"] == "1" for strides in by_parameter["B"])
    assert any(strides["j"] == "1" for strides in by_parameter["C"])


def test_specialized_gemm_prebakes_parameter_shapes_and_strides(specs):
    layout = parameter_layout_permutation(
        specs["gemm"], {"m": 6, "n": 7, "k": 9},
    )
    arrays = {
        row["parameter"]: row
        for row in layout["parameters"] if row["kind"] == "array"
    }
    assert layout["parameter_order"] == ["A", "B", "C", "alpha", "beta"]
    assert arrays["A"]["shape"] == [6, 9]
    assert arrays["A"]["row_major_strides"] == [9, 1]
    assert arrays["B"]["row_major_strides"] == [7, 1]
    assert arrays["C"]["row_major_strides"] == [7, 1]
    assert arrays["B"]["flat_offset"] == 54
    assert layout["total_array_elements"] == 54 + 63 + 42


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
    # The two launch costs are distinct facts: FIRST launch (library load
    # and first dispatch, read from the admission probe -- the only first
    # call there ever is) and RELAUNCH (fresh-execution preparation,
    # re-measurable by clearing the cache). Conflating them charted a
    # specialized core at launch=0 because its one true first launch was
    # paid before profiling began.
    assert row["first_launch_seconds"] >= 0
    assert row["relaunch_avg_seconds"] >= 0
    assert row["cold_avg_seconds"] >= row["compute_avg_seconds"]
    assert row["sizes"]  # the chart states what was measured


def test_tiny_specialized_gemm_is_admitted_and_profiled(bank):
    # Nested carried-recurrence preservation outranks unrolling, so the old
    # blanket refusal at/below the unroll threshold is no longer necessary.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        variant = bank.get(
            "gemm", contract="fast",
            specialized={"m": 4, "n": 4, "k": 4},
        )
    assert variant.specialized == {"m": 4, "n": 4, "k": 4}
    charted = {
        tuple(sorted(row["specialized"].items()))
        for row in bank.performance_chart("gemm")
    }
    assert (("k", 4), ("m", 4), ("n", 4)) in charted


def test_a_loaded_variant_refuses_parameter_identity_drift(
    bank, specs, tmp_path,
):
    copied_root = tmp_path / "tampered_bank"
    shutil.copytree(bank.root, copied_root)
    manifest_path = next(copied_root.glob("dot/*/manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identifiers = manifest["binding"]["parameter_ids_at_build"]
    first_name = next(iter(identifiers))
    identifiers[first_name] = int(identifiers[first_name]) + 1
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    reloaded = KernelBank(copied_root, specs)
    with pytest.raises(BankRefusal, match="deterministic parameter identity"):
        reloaded.get("dot", compile_missing=False)
