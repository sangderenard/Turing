"""A conditional is ordered after the regions producing its predicate.

``exchange_time_bound`` (``src/common/dt_system/participants.py``) tests
``bool(binding.any().item())``.  The control ordering treated a conditional's
membership as its arm regions only, so the region computing ``binding.any()``
could land after the ``if`` whose predicate reads it: the lowered function
read a value defined in a later block (concordance ``use-not-dominated``;
found in the woodshop dt system).  The conditional now declares its
predicate-producing regions and ordering treats them as prerequisites.
"""

from __future__ import annotations

import pathlib
import warnings

import pytest

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable


CONTRACT = (
    pathlib.Path(__file__).resolve().parents[1]
    / "extraction_contracts"
    / "program_extraction.yaml"
)

_SOURCE = (
    "from src.common.tensors.abstraction import AbstractTensor\n"
    "from src.common.dt_system.participants import StepSpans, exchange_time_bound\n\n"
    "def train(value, limit):\n"
    "    z = AbstractTensor.zeros(2)\n"
    "    spans = StepSpans(\n"
    "        pub_exchange_time=AbstractTensor.ones(2) * value,\n"
    "        pub_exchange_time_present=AbstractTensor.ones(2),\n"
    "        pub_contract=AbstractTensor.ones(2),\n"
    "        pub_dt_limit=z, pub_dt_limit_present=z,\n"
    "        pub_values=z, pub_present=z, pub_limits=z, pub_limits_present=z,\n"
    "    )\n"
    "    return exchange_time_bound(spans, 0.5, limit{current}) * 1.0\n"
)


@pytest.mark.parametrize("current", ["", ", value * 0.1"])
def test_predicate_regions_precede_their_conditional(current):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module, _outputs, _exports = lower_ast_source_to_ssa(
            _SOURCE.format(current=current), "train",
            name="predorder", extraction_contract=CONTRACT,
        )
    table = CorrelationTable.build(module)
    undominated = [
        finding for finding in table.findings(module)
        if finding.kind == "use-not-dominated"
    ]
    assert undominated == []


def test_item_capture_depends_on_its_operand_producer():
    """``float(x.item())`` is ordered after the region producing ``x``.

    ``_propose_dt_pen``: ``float(AbstractTensor.maximum(ratios.max(),
    1.0).item())``.  The region captured the ``item`` node's id; the producer
    region publishes the operand.  Dependency signatures now resolve an item
    feed through ``item_operand``; before, the consumer was scheduled first
    and read a value defined later in the same block.
    """

    source = (
        "from src.common.dt_system.dt_controller import Targets, _propose_dt_pen\n"
        "from src.common.dt_system.dt_scaler import Metrics\n\n"
        "def train(value, limit):\n"
        "    targets = Targets(cfl=0.5, div_max=1e3, mass_max=1e3,"
        " energy_exchange_fraction=0.25)\n"
        "    metrics = Metrics(value, 0.0, 0.0, 0.0)\n"
        "    return _propose_dt_pen(metrics, targets, limit, None) * 1.0\n"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module, _outputs, _exports = lower_ast_source_to_ssa(
            source, "train", name="itemorder", extraction_contract=CONTRACT,
        )
    table = CorrelationTable.build(module)
    assert [
        finding for finding in table.findings(module)
        if finding.kind == "use-not-dominated"
    ] == []


def test_call_projection_element_is_its_region_publication():
    """A region computing an element of a call's record result owns that value.

    ``run_superstep``: ``metrics = step_with_dt_control_used(...)`` then
    ``if float(metrics.control_values[0].item()) > 0.0``.  The call's
    projection walk also claimed the element read ``control_values[0]``,
    which the planner computes in a region; the region reading it bound to
    the call, so it and the ``if`` ran before the element's producer
    (``use-not-dominated``; found in the woodshop dt system).  The walk now
    descends only through declared aggregates.
    """

    source = (
        "from src.common.tensors.abstraction import AbstractTensor\n"
        "from src.common.dt_system.dt_scaler import Metrics\n\n"
        "def measure(value):\n"
        "    metrics = Metrics(value, 0.0, 0.0, 0.0)\n"
        "    metrics.control_values[0] = value - 0.5\n"
        "    return metrics\n\n"
        "def train(value, limit):\n"
        "    total = AbstractTensor.tensor(0.0)\n"
        "    last = 0.0\n"
        "    while total.item() < limit:\n"
        "        metrics = measure(value)\n"
        "        if float(metrics.control_values[0].item()) > 0.0:\n"
        "            last = last + 1.0\n"
        "        total += value\n"
        "    return total * last\n"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module, _outputs, _exports = lower_ast_source_to_ssa(
            source, "train", name="projorder", extraction_contract=CONTRACT,
        )
    table = CorrelationTable.build(module)
    assert [
        finding for finding in table.findings(module)
        if finding.kind == "use-not-dominated"
    ] == []
