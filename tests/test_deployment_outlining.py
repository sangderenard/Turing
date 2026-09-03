"""Outlined iteration lanes become real native pool deploys, provably.

The synthetic program is the canonical retained-loop shape precompile_to_ssa
mints: deploy marker, header Phi/Lt/CondBr, member body writing one
induction-indexed slot, latch Add/Br, join at the exit.  The tests prove the
whole seam the vehicle build uses: SSA outlining -> C emission with
``turing_pool_deploy_span`` -> a compiled artifact linked against
``turing_pool.c`` -> numerically identical results to the serial build.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.compiler.deployment_outlining import (
    outline_independent_iteration_lanes,
)
from src.compiler.ssa_c_backend import emit_ssa_module_to_c
from src.transmogrifier.ssa import (
    BasicBlock,
    Function,
    IRModule,
    Instr,
    SSADeploymentLane,
    SSADeploymentRegion,
    SSAValue,
)

EXTENT = 16
REGION = 0


def _member(instruction: Instr) -> Instr:
    attributes = dict(instruction.attributes or {})
    attributes["deployment_memberships"] = ((REGION, 0),)
    instruction.attributes = attributes
    return instruction


def _loop_module(
    *, shared_append: bool = False, invariant_append: bool = False,
) -> IRModule:
    buffer = SSAValue(0, "float64", shape=(EXTENT,))
    start = SSAValue(1, "int64")
    stop = SSAValue(2, "int64")
    step = SSAValue(3, "int64")
    induction = SSAValue(4, "int64")
    next_induction = SSAValue(5, "int64")
    condition = SSAValue(6, "bool")
    address = SSAValue(7, "ptr")
    loaded = SSAValue(8, "float64")
    one = SSAValue(9, "float64")
    summed = SSAValue(10, "float64")
    shared_handle = SSAValue(11, "int64")
    append_result = SSAValue(12, "int64")

    body_instrs = [
        _member(Instr("GetElementPtr", [buffer, induction], address)),
        _member(Instr("Load", [address], loaded)),
        _member(Instr("Add", [loaded, one], summed)),
        _member(Instr("Store", [summed, address], None)),
    ]
    if shared_append:
        body_instrs.append(_member(Instr(
            "Call", [shared_handle, summed], append_result,
            attributes={"callee": "ssa_sequence_11_append"},
        )))
    if invariant_append:
        # Every iteration appends the same lane-invariant value: order is
        # unobservable, so the pass accepts it as a guarded critical block.
        body_instrs.append(_member(Instr(
            "Call", [shared_handle, one], append_result,
            attributes={"callee": "ssa_sequence_11_append"},
        )))
    body_instrs.append(_member(Instr(
        "Br", [], None, attributes={"target": "loop_latch"},
    )))

    root = Function("pooled_fill_root", [buffer], {
        "entry": BasicBlock("entry", [
            Instr("Const", [], start, attributes={"value": 0}),
            Instr("Const", [], stop, attributes={"value": EXTENT}),
            Instr("Const", [], step, attributes={"value": 1}),
            Instr("Const", [], one, attributes={"value": 1.0}),
            Instr("Const", [], shared_handle, attributes={"value": 11}),
            Instr("Deploy", [], None, attributes={
                "deployment_frame": True, "region_id": REGION,
            }),
            Instr("Br", [], None, attributes={"target": "loop_header"}),
        ], ["loop_header"]),
        "loop_header": BasicBlock("loop_header", [
            Instr("Phi", [start, next_induction], induction, attributes={
                "incoming_blocks": ("entry", "loop_latch"),
            }),
            Instr("Lt", [induction, stop], condition),
            Instr("CondBr", [condition], None, attributes={
                "true_target": "loop_body", "false_target": "loop_exit",
            }),
        ], ["loop_body", "loop_exit"]),
        "loop_body": BasicBlock("loop_body", body_instrs, ["loop_latch"]),
        "loop_latch": BasicBlock("loop_latch", [
            Instr("Add", [induction, step], next_induction),
            Instr("Br", [], None, attributes={"target": "loop_header"}),
        ], ["loop_header"]),
        "loop_exit": BasicBlock("loop_exit", [
            Instr("Join", [], None, attributes={
                "deployment_frame": True, "region_id": REGION,
            }),
            Instr("Ret", [], None),
        ], []),
    }, metadata={"output_names": ()})
    region = SSADeploymentRegion(
        region_id=REGION,
        function=root.name,
        kind="parallel_candidate",
        schedule="independent_iterations",
        lanes=(SSADeploymentLane(0, (("loop_body", 0),), ()),),
        iteration_space=("0", str(EXTENT), "1"),
        origin="retained_loop",
        deploy_site=("entry", 5),
        join_site=("loop_exit", 0),
    )
    functions = {root.name: root}
    if shared_append or invariant_append:
        value = SSAValue(20, "float64")
        result = SSAValue(21, "int64")
        handle = SSAValue(22, "int64")
        functions["ssa_sequence_11_append"] = Function(
            "ssa_sequence_11_append", [handle, value], {
                "entry": BasicBlock("entry", [
                    Instr("Ret", [result], None),
                ]),
            },
        )
    return IRModule(functions, deployment_table={root.name: (region,)})


def test_outlining_moves_the_lane_into_a_callable_closure():
    module = _loop_module()
    report = outline_independent_iteration_lanes(module)
    assert report.complete, report.refused
    record, = report.outlined
    assert record.outline_name in module.functions
    outline = module.functions[record.outline_name]
    # Induction first, then the remaining live-ins (buffer, the 1.0
    # constant); the parent's lane is now a single call plus the
    # continuation branch.
    assert [int(value.id) for value in outline.args] == [4, 0, 9]
    parent = module.functions["pooled_fill_root"]
    call, branch = parent.blocks["loop_body"].instrs
    assert call.op == "Call"
    assert call.attributes["callee"] == record.outline_name
    assert branch.attributes["target"] == "loop_latch"
    # The region record now names the outline as the lane's callable root.
    lane = module.deployment_table["pooled_fill_root"][0].lanes[0]
    assert lane.callees[0] == record.outline_name


def test_shared_sequence_append_is_a_named_refusal():
    module = _loop_module(shared_append=True)
    report = outline_independent_iteration_lanes(module)
    assert not report.outlined
    (_function, region_id, reason), = report.refused
    assert region_id == REGION
    assert "ordered-join" in reason
    assert "%t11" in reason


def test_invariant_shared_append_is_guarded_not_refused():
    module = _loop_module(invariant_append=True)
    report = outline_independent_iteration_lanes(module)
    assert report.complete, report.refused
    record, = report.outlined
    assert record.guarded_blocks == ("loop_body",)
    outline = module.functions[record.outline_name]
    assert outline.metadata["pool_effect_guarded_blocks"] == ("loop_body",)
    artifact = emit_ssa_module_to_c(module, "pooled_fill_root")
    assert artifact.pool_required
    assert "turing_pool_effect_lock();" in artifact.source
    assert "turing_pool_effect_unlock();" in artifact.source


def test_pooled_emission_carries_deploy_span_and_serial_fallback():
    module = _loop_module()
    report = outline_independent_iteration_lanes(module)
    assert report.complete, report.refused
    artifact = emit_ssa_module_to_c(module, "pooled_fill_root")
    assert artifact.complete, artifact.shortfalls
    assert artifact.pool_required
    assert artifact.pooled_regions == (("pooled_fill_root", REGION),)
    assert "turing_pool_deploy_span(" in artifact.source
    assert "turing_pool_start(" in artifact.source
    # The serial loop remains in the text as the fallback.
    assert "goto L_impl_loop_header" in artifact.source


def test_compute_selection_judges_lanes_automatically():
    from src.compiler.deployment_compute_selection import select_compute_lanes

    eligible_module = _loop_module()
    outline_independent_iteration_lanes(eligible_module)
    verdict, = select_compute_lanes(eligible_module).verdicts
    assert verdict.eligible, verdict.reasons

    guarded_module = _loop_module(invariant_append=True)
    outline_independent_iteration_lanes(guarded_module)
    refused, = select_compute_lanes(guarded_module).verdicts
    assert not refused.eligible
    assert any("internal functions" in reason for reason in refused.reasons)
    assert any("effect-locked" in reason for reason in refused.reasons)


@pytest.mark.parametrize("outlined", (False, True))
def test_compiled_pooled_loop_matches_serial_numerics(tmp_path, outlined):
    module = _loop_module()
    if outlined:
        report = outline_independent_iteration_lanes(module)
        assert report.complete, report.refused
    artifact = emit_ssa_module_to_c(module, "pooled_fill_root")
    assert artifact.complete, artifact.shortfalls
    assert artifact.pool_required is outlined
    artifact.compile(tmp_path / ("pooled" if outlined else "serial"))
    initial = (np.arange(EXTENT) * 10.0).astype(np.float64)
    execution = artifact.prepare_execution({0: initial.copy()}).run()
    np.testing.assert_allclose(
        np.asarray(execution.buffers[0]).reshape(-1), initial + 1.0,
    )
