from src.common.tensors.accelerator_backends.glsl_backend import (
    GLSL_OPS,
    emit_multi_output_program_source,
)
from src.common.tensors.fused_ir import FusedProgram, Meta, OpStep
from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.compression.jpeg.frame import (
    encode_color_jfif,
    encode_ycbcr_jfif,
)
from src.common.tensors.compression.jpeg.transform import rgb_to_ycbcr
from src.compiler.process_graph_fusion import (
    BackendFusionProfile,
    dispatch_region_to_fused_program,
    fused_program_to_process_graph,
    plan_process_graph_dispatches,
)
from src.compiler.torch_process_graph import compile_process_graph_torch


def _branched_program():
    return FusedProgram(
        version=1,
        feeds={1, 2},
        steps=[
            OpStep(0, "add", [1, 2], result_id=3),
            OpStep(1, "sin", [3], result_id=4),
            OpStep(2, "cos", [3], result_id=5),
        ],
        outputs={"sine": 4, "cosine": 5},
        meta={
            value_id: Meta(shape=(8,), dtype="float32")
            for value_id in range(1, 6)
        },
    )


def test_process_graph_planner_keeps_shared_multi_output_producer_in_one_region():
    graph = fused_program_to_process_graph(_branched_program())
    plan = plan_process_graph_dispatches(
        graph,
        BackendFusionProfile("glsl", frozenset(GLSL_OPS)),
    )

    assert len(plan.regions) == 1
    region = plan.regions[0]
    assert region.node_ids == (3, 4, 5)
    assert region.input_ids == (1, 2)
    assert dict(region.outputs) == {"sine": 4, "cosine": 5}
    assert region.binding_count == 4

    lowered = dispatch_region_to_fused_program(graph, region)
    assert [step.op_name for step in lowered.steps] == ["add", "sin", "cos"]
    assert lowered.outputs == {"sine": 4, "cosine": 5}


def test_multi_output_glsl_emitter_writes_all_results_in_one_shader():
    source = emit_multi_output_program_source(
        _branched_program(),
        feed_shapes={1: (8,), 2: (8,)},
        output_shape=(8,),
    )

    assert source.count("void main()") == 1
    assert "binding = 2" in source
    assert "binding = 3" in source
    assert "outbuf0[gid]" in source
    assert "outbuf1[gid]" in source
    assert source.count("sin(") == 1
    assert source.count("cos(") == 1


def test_precomputed_ycbcr_entry_matches_the_rgb_encoder():
    samples = [
        [
            (
                (row * 17 + column * 3) % 256,
                (row * 5 + column * 19) % 256,
                (row * 11 + column * 13) % 256,
            )
            for column in range(16)
        ]
        for row in range(16)
    ]
    with AbstractTensor.use_backend("numpy"):
        rgb = AbstractTensor.tensor(samples)
        planes = rgb_to_ycbcr(rgb)
        expected = encode_color_jfif(rgb)
        actual = encode_ycbcr_jfif(planes)

    assert actual == expected


def test_torch_process_graph_compiler_uses_the_same_selected_region():
    torch = __import__("torch")
    graph = fused_program_to_process_graph(_branched_program())
    compilation = compile_process_graph_torch(
        graph,
        # This exercises Dynamo/AOT without requiring Triton in the test
        # environment. Production defaults to the optimizing Inductor backend.
        compiler_backend="aot_eager",
    )

    assert len(compilation.plan.regions) == 1
    assert len(compilation.regions) == 1
    left = torch.linspace(-1.0, 1.0, 8)
    right = torch.linspace(0.25, 0.75, 8)
    outputs = compilation.regions[0]({1: left, 2: right})

    torch.testing.assert_close(outputs["sine"], torch.sin(left + right))
    torch.testing.assert_close(outputs["cosine"], torch.cos(left + right))


def test_dynamic_scalar_dependencies_remain_runtime_feeds():
    from src.common.tensors.accelerator_backends.c_primitive_program import (
        compile_elementwise_tape,
    )
    from src.common.tensors.autograd import autograd
    from src.common.tensors.numpy_backend import NumPyTensorOperations as NT

    with autograd.forward_capture() as tape:
        values = NT.tensor([1.0, 2.0, 3.0])
        control = NT.tensor([0.25])
        bounded = control.minimum(1.0).maximum(0.0)
        output = values + bounded * 2.0
    captured = compile_elementwise_tape(
        tape,
        output,
        dynamic_scalar_ids=(id(control),),
    )

    assert id(control) in captured.program.feeds
    assert any(step.result_id == id(bounded) for step in captured.program.steps)
