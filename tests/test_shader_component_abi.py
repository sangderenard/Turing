import pytest

from src.common.tensors.fused_ir import FusedProgram, OpStep
from src.compiler.glsl_component_backend import (
    emit_glsl_compute_component,
    emit_glsl_fragment_component,
)
from src.compiler.precompile_to_ssa import lower_fused_program_to_ssa
from src.compiler.shader_component_abi import (
    ComponentAssemblyPlan,
    ComponentSentinels,
    ExternalComponentLink,
    LinkScope,
    LinkTransport,
    component_abi_from_layout,
    validate_hierarchical_component_plan,
)
from src.compiler.hierarchical_plan import PlanCall, PlanClosure
from src.compiler.shader_stages import BufferBinding, ShaderIOLayout
from src.compiler.ssa_webgpu_backend import emit_module
from src.transmogrifier.ssa import IRModule


def _program(feed=1, result=3):
    return FusedProgram(
        version=1,
        feeds={feed, 2},
        steps=[OpStep(0, "add", [feed, 2], {}, result)],
        outputs={"result": result},
    )


def _wgsl(program):
    function, shortfalls = lower_fused_program_to_ssa(program, function_name="kernel")
    assert not shortfalls
    returned = next(
        instruction.args
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op in {"Ret", "ret", "Return", "return"}
    )
    return emit_module(
        IRModule({"kernel": function}),
        name="kernel",
        outputs={"kernel": returned},
        count=8,
    )


def _logical_ports(component):
    return tuple(
        (port.slot, port.role, port.dtype, port.value_id)
        for port in component.ports
    )


def test_glsl_compute_and_webgpu_publish_the_same_logical_component_abi():
    program = _program()
    glsl = emit_glsl_compute_component(program, name="kernel", local_size=32)
    wgsl = _wgsl(program)

    assert "#version 430" in glsl.source
    assert "layout(local_size_x = 32)" in glsl.source
    assert _logical_ports(glsl.component_abi) == _logical_ports(wgsl.component_abi)
    assert glsl.component_abi.language == "glsl-430"
    assert wgsl.component_abi.language == "wgsl"
    assert glsl.api.metadata["component_abi"]["schema"] == "turing.shader-component.v1"


def test_fragment_shader_has_the_same_sentinel_and_port_contract():
    fragment = emit_glsl_fragment_component(_program(), name="present")

    assert "#version 300 es" in fragment.source
    assert fragment.component_abi.stage == "fragment"
    assert fragment.component_abi.sentinels.port_count == len(fragment.component_abi.ports)
    assert fragment.api.metadata["component_abi"]["sentinels"]["word_layout"][4:6] == [
        "ready", "error",
    ]


def test_sentinel_rejects_corrupt_or_cross_version_component_headers():
    valid = ComponentSentinels.for_ports(3)
    valid.validate(3)

    with pytest.raises(ValueError, match="incompatible or corrupt"):
        ComponentSentinels(port_count=3, checksum=0).validate(3)


def test_local_and_online_links_share_validation_and_lower_to_ssa_boundaries():
    producer = emit_glsl_compute_component(_program(), name="producer").component_abi
    # This second ABI has the same feed slots; connect producer output slot 2
    # to consumer feed slot 0.
    consumer = _wgsl(_program(feed=1, result=4)).component_abi
    consumer = type(consumer)(
        component_id="consumer",
        language=consumer.language,
        stage=consumer.stage,
        entrypoint=consumer.entrypoint,
        ports=consumer.ports,
        sentinels=consumer.sentinels,
        decorations=consumer.decorations,
    )
    local = ExternalComponentLink(
        "local-edge", "producer", 2, "consumer", 0,
        LinkScope.LOCAL, LinkTransport.SHARED_ARENA,
    )
    plan = ComponentAssemblyPlan((producer, consumer), (local,))

    assert plan.shell_waves() == (("producer",), ("consumer",))
    boundary = plan.ssa_boundaries()[0]
    assert boundary.scope == "system-local"
    assert boundary.transport == "shared-arena"
    assert boundary.sentinel_required
    lowered = plan.lower_to_ssa()
    link_function = lowered.module.functions["external_link_0"]
    call = link_function.blocks["entry"].instrs[0]
    assert call.op == "Call"
    assert call.attributes["external"] is True
    assert call.attributes["sentinel_generation_rule"] == (
        "consumer-generation=producer-generation"
    )
    assert lowered.shell_waves == (("producer",), ("consumer",))
    manifest = plan.to_mapping()
    assert manifest["schema"] == "turing.shader-component-assembly.v1"
    assert manifest["links"][0]["sentinel_policy"] == "required"
    assert manifest["shell_waves"] == [["producer"], ["consumer"]]

    online = ExternalComponentLink(
        "online-edge", "producer", 2, "consumer", 0,
        LinkScope.ONLINE, LinkTransport.ONLINE_MESSAGE,
        alias=False, endpoint="program://consumer/v1",
    )
    online_plan = ComponentAssemblyPlan((producer, consumer), (online,))
    assert online_plan.ssa_boundaries()[0].scope == "online-cross-program"


def test_feedback_cycle_requires_a_versioned_sentinel_boundary():
    producer = emit_glsl_compute_component(_program(), name="producer").component_abi
    consumer = _wgsl(_program(feed=1, result=4)).component_abi
    consumer = type(consumer)(
        component_id="consumer", language=consumer.language,
        stage=consumer.stage, entrypoint=consumer.entrypoint,
        ports=consumer.ports, sentinels=consumer.sentinels,
        decorations=consumer.decorations,
    )
    forward = ExternalComponentLink(
        "forward", "producer", 2, "consumer", 0,
        LinkScope.LOCAL, LinkTransport.SHARED_ARENA,
    )
    feedback = ExternalComponentLink(
        "feedback", "consumer", 2, "producer", 0,
        LinkScope.LOCAL, LinkTransport.COMPILED_ARTIFACT,
        alias=False, feedback=True,
    )
    plan = ComponentAssemblyPlan((producer, consumer), (forward, feedback))

    assert plan.shell_waves() == (("producer",), ("consumer",))
    feedback_call = plan.lower_to_ssa().module.functions["external_link_1"].blocks["entry"].instrs[0]
    assert feedback_call.attributes["feedback"] is True

    with pytest.raises(ValueError, match="non-aliasing sentinel"):
        ExternalComponentLink(
            "bad-feedback", "consumer", 2, "producer", 0,
            LinkScope.LOCAL, LinkTransport.SHARED_ARENA,
            feedback=True,
        )


def test_external_link_validation_fails_closed():
    with pytest.raises(ValueError, match="require an endpoint"):
        ExternalComponentLink(
            "bad", "a", 0, "b", 0,
            LinkScope.ONLINE, LinkTransport.ONLINE_MESSAGE,
        )


def test_existing_multi_shell_call_plan_is_verified_against_component_ports():
    caller = component_abi_from_layout(
        "root-shell", "glsl-430",
        ShaderIOLayout(
            "compute",
            feeds=(BufferBinding("root_in", "feed", "f32", 0, 10),),
            outputs=(BufferBinding("root_out", "output", "f32", 1, 11),),
        ),
    )
    callee = component_abi_from_layout(
        "child-shell", "wgsl",
        ShaderIOLayout(
            "compute",
            feeds=(BufferBinding("child_in", "feed", "f32", 0, 20),),
            outputs=(BufferBinding("child_out", "output", "f32", 1, 21),),
        ),
    )
    child_plan = PlanClosure("child", (20,), (), closure_id=1)
    root_plan = PlanClosure("root", (10,), (
        PlanCall(
            99, child_plan,
            argument_value_ids=(10,), result_value_ids=(11,),
            argument_bindings=((10, 20),),
            result_bindings=((21, 11),),
        ),
    ), closure_id=0)

    bindings = validate_hierarchical_component_plan(
        root_plan, (caller, callee), {0: "root-shell", 1: "child-shell"},
    )

    assert bindings[0].callsite_id == 99
    assert bindings[0].argument_slots == ((0, 0),)
    assert bindings[0].result_slots == ((1, 1),)
