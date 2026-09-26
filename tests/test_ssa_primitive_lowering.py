from src.compiler.bitops_process_graph import expand_bitops_process_graph
from src.compiler.ssa_builder import process_graph_to_ssa_instrs
from src.compiler.ssa_primitive_lowering import lower_ssa_to_fused_program
from src.compiler.precompile_to_ssa import lower_fused_program_to_ssa
from src.common.tensors.accelerator_backends.c_backend import CTensor
from src.common.tensors.accelerator_backends.c_primitive_program import (
    execute_fused_program,
)
from src.common.tensors.accelerator_backends.glsl_backend import emit_program_source
from src.transmogrifier.graph.graph_express2 import ProcessGraph


def _ssa(source, *, bitops=False):
    graph = ProcessGraph(materialize_memory=False)
    graph.build_from_ast(source)
    if bitops:
        graph = expand_bitops_process_graph(graph, bit_width=8)
    return process_graph_to_ssa_instrs(graph, schedule="asap")


def test_elementwise_ast_ssa_lowers_to_shared_c_glsl_program():
    result = lower_ssa_to_fused_program(
        _ssa(
            """
def kernel(x, y):
    return (x + y) * 3
"""
        )
    )
    program = result.require_complete()
    assert len(program.feeds) == 2
    assert [step.op_name for step in program.steps] == ["add", "mul"]
    assert program.steps[1].attrs["right_scalar"] == 3.0

    c_result = execute_fused_program(
        program,
        (
            CTensor.from_list([1.0, 2.0], (2,)),
            CTensor.from_list([10.0, 20.0], (2,)),
        )
    )
    assert c_result.tolist() == [33.0, 66.0]

    shader = emit_program_source(program)
    assert "float s2 = s0 + s1;" in shader
    assert "float s3 = s2 * float(3.0);" in shader


def test_natural_tensor_calls_lower_to_one_glsl_program():
    result = lower_ssa_to_fused_program(
        _ssa(
            """
def kernel(x, y):
    return tanh((x + y).sin())
"""
        )
    )
    program = result.require_complete()
    assert [step.op_name for step in program.steps] == ["add", "sin", "tanh"]

    shader = emit_program_source(program)
    assert "float s2 = s0 + s1;" in shader
    assert "float s3 = sin(s2);" in shader
    assert "float s4 = tanh(s3);" in shader


def test_random_spellings_normalize_to_one_fused_operator():
    graph = ProcessGraph(materialize_memory=False)
    graph.build_from_ast(
        """
def kernel():
    a = random.random()
    b = np.random.randn(4)
    c = torch.randint(0, 8, size=(4,))
    return c
"""
    )

    random_nodes = [
        data
        for _, data in graph.G.nodes(data=True)
        if data.get("op") == "random_source"
    ]

    assert len(random_nodes) == 3
    assert {node["attributes"]["distribution"] for node in random_nodes} == {
        "uniform",
        "normal",
        "integer",
    }
    assert {node["attributes"]["request"] for node in random_nodes} == {
        "random.random",
        "np.random.randn",
        "torch.randint",
    }

    numerical_graph = ProcessGraph(materialize_memory=False)
    numerical_graph.build_from_ast(
        """
def kernel():
    return np.random.randn(4)
"""
    )
    fused = lower_ssa_to_fused_program(
        process_graph_to_ssa_instrs(numerical_graph, schedule="asap")
    ).require_complete()
    assert [step.op_name for step in fused.steps] == ["random_source"]

    function, shortfalls = lower_fused_program_to_ssa(fused)
    assert shortfalls == ()
    calls = [
        instruction
        for instruction in function.blocks["entry"].instrs
        if instruction.op == "Call"
    ]
    assert {call.attributes["callee"] for call in calls} == {
        "xoroshiro128ss_fill_double"
    }


def test_bitops_nand_graph_lowers_without_bypassing_provenance():
    result = lower_ssa_to_fused_program(
        _ssa(
            """
def kernel(x, y):
    return x ^ y
""",
            bitops=True,
        )
    )
    program = result.require_complete()
    ops = [step.op_name for step in program.steps]
    assert "mul" in ops
    assert "logical_not" in ops
    assert "bitxor" not in ops


def test_shape_changing_bitops_report_a_boundary_instead_of_cheating():
    result = lower_ssa_to_fused_program(
        _ssa(
            """
def kernel(x, y):
    return x + y
""",
            bitops=True,
        )
    )
    assert not result.complete
    assert result.program is None
    assert any(issue.op in {"zeros", "slice", "concat", "mu"} for issue in result.issues)
