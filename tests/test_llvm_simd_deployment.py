from __future__ import annotations

import math

import pytest

from src.common.tensors.accelerator_backends.llvm_simd_deployment import (
    compile_scale1_reduction,
    host_simd_lane_width,
    host_simd_width_bits,
)
from src.compiler.control_source import (
    ControlDeploymentLane,
    ControlDeploymentRegion,
    ControlProgram,
    StatementBlock,
)
from src.compiler.deployment_frame import DeploymentJoin, DeploymentJoinMode
from src.compiler.precompile_to_ssa import lower_control_program_to_ssa
from src.transmogrifier.ssa import SSADeploymentRegion
from src.transmogrifier.operator_defs import operator_signatures, role_schemas
from src.transmogrifier.ssa_registry import Handler, SSARegistry


def _join(*, reassociate: bool = True) -> DeploymentJoin:
    return DeploymentJoin(
        DeploymentJoinMode.REDUCE,
        "Add",
        allow_reassociation=reassociate,
    )


def _ssa_region(**changes) -> SSADeploymentRegion:
    values = dict(
        region_id=0,
        function="reduce",
        kind="parallel_candidate",
        schedule="independent_lanes",
        scale=1,
        join=_join(),
    )
    values.update(changes)
    return SSADeploymentRegion(**values)


def test_process_graph_and_ssa_share_structural_deploy_join_vocabulary():
    assert {"Deploy", "Join"} <= operator_signatures.keys()
    assert role_schemas["Deploy"]["down"]["lanes"] == "many"
    assert role_schemas["Join"]["up"]["lanes"] == "many"
    assert SSARegistry.name_map["deploy"] is Handler.Deploy
    assert SSARegistry.name_map["join"] is Handler.Join


def test_control_deployment_contract_survives_into_ssa():
    control = ControlProgram(
        StatementBlock(("__scheduled_region_3__",)),
        region_indices=(3,),
        deployment_regions=(ControlDeploymentRegion(
            region_id=8,
            kind="parallel_candidate",
            schedule="independent_lanes",
            lanes=(ControlDeploymentLane(0, region_indices=(3,)),),
            scale=1,
            join=_join(),
        ),),
    )

    function, shortfalls = lower_control_program_to_ssa(control)

    assert shortfalls == ()
    region, = function.metadata["deployment_regions"]
    assert region.frame == control.deployment_regions[0].frame
    assert region.join.reduction_operator == "Add"


def test_host_adaptive_simd_executes_vector_tail_and_scalar_fallback():
    artifact = compile_scale1_reduction(_ssa_region())
    lanes = host_simd_lane_width(64)

    assert artifact.lane_width == lanes
    assert artifact.vector_bits == host_simd_width_bits()
    for length in (0, 1, max(1, lanes - 1), lanes, lanes * 3 + 1):
        values = [((index % 7) - 3) / 8.0 for index in range(length)]
        assert math.isclose(
            artifact.execute(values),
            artifact.execute_serial(values),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    if lanes > 1:
        assert artifact.vectorized

    scalar = compile_scale1_reduction(_ssa_region(), force_vector_bits=64)
    assert scalar.lane_width == 1
    assert not scalar.vectorized
    assert scalar.execute([1.0, 2.0, 3.0]) == 6.0


def test_host_width_never_assumes_avx2():
    from llvmlite import binding as llvm

    features = llvm.get_host_cpu_features()
    if not bool(features.get("avx2", False)):
        # AVX2 is not a legality condition: SSE2 can run two float64 lanes and
        # AVX1 can run four. This host's bdver2 tuning deliberately chooses 2.
        assert host_simd_width_bits() <= 256
        artifact = compile_scale1_reduction(_ssa_region())
        assert artifact.execute([1.0, 2.0, 3.0]) == 6.0
        if host_simd_width_bits() <= 128:
            assert "ymm" not in artifact.assembly.lower()


@pytest.mark.parametrize(
    "region, message",
    [
        (_ssa_region(scale=2), "scale 1"),
        (_ssa_region(join=_join(reassociate=False)), "reassociation"),
        (_ssa_region(carried_aliases=((1, 0),)), "aliases"),
        (
            _ssa_region(join=DeploymentJoin(
                DeploymentJoinMode.REDUCE, "Mul", True
            )),
            "Add only",
        ),
    ],
)
def test_illegal_frames_take_the_explicit_serial_boundary(region, message):
    with pytest.raises(ValueError, match=message):
        compile_scale1_reduction(region)
