"""Identity operators inside loops, checked by native LLVM execution.

Each program reads a loop-carried value through an operation that gives
one value a second key, and each computed a wrong answer or was refused
before that operation was recorded on the identity book:

* ``item-carried-inner-range`` -- ``total_value = float(total.item())`` in
  a while body whose ``total`` is carried.  No region computes the rank-0
  ``item``; the region captured the item's id, which the loop does not
  carry, and a later id-keyed rewrite (``_rebind_recorded_scalar_
  identities``) replaced it with ``total``'s pre-loop definition, so
  ``total_value`` froze at its first value (silent wrong answer, 65.9 for
  35.9).  Now the planner records ``item_operand`` and the control lowering
  reads the operand by its binding in each site's loop generation (merge).
* ``item-in-break-test`` -- ``if cap.item() > 100.0: break`` with ``cap``
  carried.  The receiver ``cap`` was read by the attribute load; the
  rebuilt call took the same occurrence as its ``operand`` without the
  binding, so the control lowering found no binding for a carried value
  (refused).  Now that second read position is a fork of the first.
* ``dt-controller-shape`` -- the ``run_superstep`` loop in miniature: two
  bindings seeded from one pre-loop if-merge, an iteration-cap break, a
  ``float(total.item())`` read, and an inner ``for boundary in (0.7, 1.9)``
  clamp with a break.  Besides the item merge, the LLVM ``Const`` emitter
  wrote the float tuple as ``i32 int(item)`` into one slot, so every
  boundary read 0.0 and the clamp never fired (silent wrong answer).

* ``evaporation-keeps-carried-update`` -- a set comprehension (evaporated)
  before a while whose carried ``cap`` is last updated by an if-merge on a
  hoisted test.  Evaporation's dataflow collection kept retained loops'
  bodies but not their carried bindings; the update (read only by the
  backedge ledger) was deleted and the loop's control regions no longer
  nested (refused; in the woodshop, ``dt_cap`` read its pre-loop value).

* ``continue-arm-and-fallthrough-update`` -- ``if retries < 2: dt = dt * 0.5;
  ...; continue`` then ``dt = minimum(dt * 1.5, 1.0)``.  Four defects: the
  planner took the next ``dt`` version (the fall-through update) as the
  conditional's merge; continue-site values were never recorded, so
  ``dt * 0.5`` had no consumer; the fold's dead-call pruning deleted the
  update (read only by the loop's ledger); wrong answers throughout.
* ``continue-then-call-update`` -- the plan builder overrode the call's
  edge to the pre-branch ``dt`` with the source-latest ``dt`` definition,
  the continue arm's ``dt * 0.5``.
* ``method-keyword-carried`` -- ``ctrl.pi_update(dt_prev=dt, ...)``: the
  keyword wrapper (and its Name) is absent from the function subgraph, so
  the read had no row; it is now a fact of the occurrence (refused).

All run through LLVM, the lane the dt-system builds use.
"""

from __future__ import annotations

import pathlib
import tempfile
import warnings

import numpy as np
import pytest

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_llvm_backend import (
    compile_artifact,
    emit_ssa_function_to_llvm,
    prepare_artifact_execution,
)


CONTRACT = (
    pathlib.Path(__file__).resolve().parents[1]
    / "extraction_contracts"
    / "program_extraction.yaml"
)

_HEADER = "from src.common.tensors.abstraction import AbstractTensor\n\n"

_PROGRAMS = {
    "item-carried-inner-range": (
        _HEADER
        + "def helper(a):\n"
        "    return a\n\n"
        "def train(value, limit):\n"
        "    cap = AbstractTensor.tensor(value)\n"
        "    last = AbstractTensor.tensor(0.0)\n"
        "    total = AbstractTensor.tensor(0.0)\n"
        "    while (limit - total).item() > 0.0:\n"
        "        step = AbstractTensor.minimum(cap, limit - total)\n"
        "        total_value = float(total.item())\n"
        "        for k in range(2):\n"
        "            boundary = 0.7 + 1.2 * k\n"
        "            if boundary > total_value + 1e-9:\n"
        "                step = AbstractTensor.minimum(\n"
        "                    step, AbstractTensor.tensor(boundary - total_value))\n"
        "        proposal = step * 3.0\n"
        "        total = total + step\n"
        "        cap = AbstractTensor.minimum(cap, proposal)\n"
        "        last = proposal\n"
        "    return total + cap * 10.0 + last * 100.0\n",
        ((0.3, 2.9), (0.25, 5.0), (2.0, 0.4)),
    ),
    "dt-controller-shape": (
        _HEADER
        + "def helper(a):\n"
        "    return a\n\n"
        "def train(value, limit):\n"
        "    cap = AbstractTensor.tensor(value)\n"
        "    if value > 0.5:\n"
        "        cap = AbstractTensor.minimum(cap, AbstractTensor.tensor(0.5))\n"
        "    last = cap\n"
        "    total = AbstractTensor.tensor(0.0)\n"
        "    iters = 0\n"
        "    boundaries = (0.7, 1.9)\n"
        "    while (limit - total).item() > 0.0:\n"
        "        if iters >= 50:\n"
        "            break\n"
        "        iters += 1\n"
        "        step = AbstractTensor.minimum(cap, limit - total)\n"
        "        total_value = float(total.item())\n"
        "        for boundary in boundaries:\n"
        "            if boundary > total_value + 1e-9:\n"
        "                step = AbstractTensor.minimum(\n"
        "                    step, AbstractTensor.tensor(boundary - total_value))\n"
        "                break\n"
        "        proposal = step * 3.0\n"
        "        total = total + step\n"
        "        cap = AbstractTensor.minimum(cap, proposal)\n"
        "        last = proposal\n"
        "    return total + cap * 10.0 + last * 100.0\n",
        ((0.3, 2.9), (0.25, 5.0), (2.0, 0.4)),
    ),
    "evaporation-keeps-carried-update": (
        _HEADER
        + "def helper(a):\n"
        "    return a\n\n"
        "def train(value, limit):\n"
        "    cap = AbstractTensor.tensor(value)\n"
        "    total = AbstractTensor.tensor(0.0)\n"
        "    has_max = value > 0.2\n"
        "    if has_max:\n"
        "        cap = AbstractTensor.minimum(cap, AbstractTensor.tensor(1.0))\n"
        "    extra = tuple(sorted({float(v) for v in (2.0, 1.0) if v > 0.5}))\n"
        "    while (limit - total).item() > 0.0:\n"
        "        step = AbstractTensor.minimum(cap, limit - total)\n"
        "        total = total + step\n"
        "        cap = cap * 2.0\n"
        "        if has_max:\n"
        "            cap = AbstractTensor.minimum(cap, AbstractTensor.tensor(1.0))\n"
        "    return total * 1.0\n",
        ((0.3, 2.9), (0.1, 5.0), (2.0, 0.4)),
    ),
    "continue-arm-and-fallthrough-update": (
        _HEADER
        + "def helper(a):\n"
        "    return a\n\n"
        "def train(value, limit):\n"
        "    dt = AbstractTensor.tensor(value)\n"
        "    total = AbstractTensor.tensor(0.0)\n"
        "    retries = 0\n"
        "    while (limit - total).item() > 0.0:\n"
        "        if retries < 2:\n"
        "            dt = dt * 0.5\n"
        "            retries += 1\n"
        "            continue\n"
        "        total = total + dt\n"
        "        dt = AbstractTensor.minimum(dt * 1.5, AbstractTensor.tensor(1.0))\n"
        "    return total * 1.0\n",
        ((0.3, 2.9), (0.25, 5.0), (2.0, 0.4)),
    ),
    "continue-then-call-update": (
        _HEADER
        + "def pi_update(dt_prev, dt_pen):\n"
        "    return AbstractTensor.minimum(dt_prev * 1.5, dt_pen)\n\n"
        "def train(value, limit):\n"
        "    dt = AbstractTensor.tensor(value)\n"
        "    total = AbstractTensor.tensor(0.0)\n"
        "    retries = 0\n"
        "    while (limit - total).item() > 0.0:\n"
        "        if retries < 2:\n"
        "            dt = dt * 0.5\n"
        "            retries += 1\n"
        "            continue\n"
        "        total = total + dt\n"
        "        dt_next = pi_update(dt, AbstractTensor.tensor(1.0))\n"
        "        dt = dt_next\n"
        "    return total * 1.0\n",
        ((0.3, 2.9), (0.25, 5.0), (2.0, 0.4)),
    ),
    "method-keyword-carried": (
        _HEADER
        + "class Ctl:\n"
        "    def __init__(self, gain):\n"
        "        self.gain = gain\n\n"
        "    def pi_update(self, dt_prev, dt_pen, osc=False):\n"
        "        return AbstractTensor.minimum(dt_prev * self.gain, dt_pen)\n\n"
        "def train(value, limit):\n"
        "    ctrl = Ctl(1.5)\n"
        "    dt = AbstractTensor.tensor(value)\n"
        "    total = AbstractTensor.tensor(0.0)\n"
        "    while (limit - total).item() > 0.0:\n"
        "        total = total + dt\n"
        "        dt = ctrl.pi_update(dt_prev=dt, dt_pen=AbstractTensor.tensor(1.0), osc=False)\n"
        "    return total * 1.0\n",
        ((0.3, 2.9), (0.25, 5.0), (2.0, 0.4)),
    ),
    "item-in-break-test": (
        _HEADER
        + "def propose(step):\n"
        "    return step * 3.0\n\n"
        "def train(value, limit):\n"
        "    cap = AbstractTensor.tensor(value)\n"
        "    total = AbstractTensor.tensor(0.0)\n"
        "    while (limit - total).item() > 0.0:\n"
        "        if cap.item() > 100.0:\n"
        "            break\n"
        "        total = total + cap\n"
        "        cap = propose(cap)\n"
        "    return total + cap\n",
        ((0.3, 2.9), (0.25, 5.0), (60.0, 1000.0)),
    ),
}


@pytest.mark.parametrize("label", sorted(_PROGRAMS))
def test_loop_identity_operators_compute_the_authored_answer(label):
    source, probes = _PROGRAMS[label]
    namespace: dict = {}
    exec(compile(source, "<authored>", "exec"), namespace)
    authored = namespace["train"]
    prefix = "identop_" + label.replace("-", "_")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module, outputs, _exports = lower_ast_source_to_ssa(
            source, "train", name=prefix, extraction_contract=CONTRACT,
        )
    root = module.functions[f"{prefix}__train"]
    parameters = dict(root.metadata["parameter_names"])
    artifact = emit_ssa_function_to_llvm(module, root.name)
    assert artifact.shortfalls == (), artifact.shortfalls
    native = compile_artifact(
        artifact, directory=pathlib.Path(tempfile.mkdtemp()) / prefix,
    )
    produced, expected = [], []
    for value, limit in probes:
        execution = prepare_artifact_execution(native, {
            parameters["value"]: np.array([value]),
            parameters["limit"]: np.array([limit]),
        })
        execution.run()
        produced.append(float(np.asarray(
            execution.buffers[outputs[root.name][0].id]
        ).reshape(-1)[0]))
        expected.append(float(authored(value, limit)))
    assert produced == pytest.approx(expected, rel=1e-12, abs=1e-12)
