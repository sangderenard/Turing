import pytest

from src.compiler.loop_ir import (
    IterableDomain,
    LoopDomainKind,
    LoopPolicy,
    LoopValue,
    RangeDomain,
    SemanticLoop,
)
from src.compiler.hierarchical_plan import (
    PlanClosure,
    PlanLine,
    render_plan_ascii,
)


def test_semantic_loop_rejects_domain_kind_mismatch():
    with pytest.raises(TypeError, match="range loop requires RangeDomain"):
        SemanticLoop(
            loop_id=1,
            domain_kind=LoopDomainKind.RANGE,
            domain=IterableDomain(LoopValue(2), (("item", 3),)),
            body_node_ids=(4,),
        )


def test_semantic_range_loop_preserves_policy_separately_from_semantics():
    loop = SemanticLoop(
        loop_id=1,
        domain_kind=LoopDomainKind.RANGE,
        domain=RangeDomain(
            LoopValue(literal=1),
            LoopValue(value_id=8),
            LoopValue(literal=2),
        ),
        body_node_ids=(4, 5),
        policy=LoopPolicy(
            unroll_limit=16,
            allow_parallel_iterations=True,
            require_resident_state=True,
        ),
    )

    assert loop.domain.stop.value_id == 8
    assert loop.policy.allow_parallel_iterations
    assert loop.policy.require_resident_state


def test_semantic_range_rejects_zero_step():
    with pytest.raises(ValueError, match="step cannot be zero"):
        SemanticLoop(
            loop_id=1,
            domain_kind=LoopDomainKind.RANGE,
            domain=RangeDomain(
                LoopValue(literal=0),
                LoopValue(literal=8),
                LoopValue(literal=0),
            ),
            body_node_ids=(2,),
        )


def test_hierarchical_plan_is_structural_and_ascii_is_only_a_view():
    plan = PlanClosure(
        "outer",
        (1,),
        (
            PlanLine.create("load", inputs=(1,), outputs=(2,)),
            PlanClosure(
                "loop_body",
                (2,),
                (PlanLine.create("add", inputs=(2, 3), outputs=(4,)),),
            ),
        ),
    )

    rendered = render_plan_ascii(plan)
    assert "closure outer captures=[1]" in rendered
    assert "closure loop_body captures=[2]" in rendered
    assert "add in=[2,3] out=[4]" in rendered
