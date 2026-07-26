from src.compiler.ast_process_graph import ast_to_process_graph
from src.compiler.bitops_process_graph import expand_bitops_process_graph
from src.compiler.ssa_builder import process_graph_to_ssa_instrs
from src.compiler.ssa_primitive_lowering import lower_ssa_to_primitive_program
from src.common.tensors.accelerator_backends.c_backend import CTensor
from src.common.tensors.accelerator_backends.glsl_backend import (
    GlslProgram,
    emit_program_source,
)


def _ssa(source, *, bitops=False):
    graph = ast_to_process_graph(source)
    if bitops:
        graph = expand_bitops_process_graph(graph, bit_width=8)
    return process_graph_to_ssa_instrs(graph, schedule="asap")


def test_elementwise_ast_ssa_lowers_to_shared_c_glsl_program():
    result = lower_ssa_to_primitive_program(
        _ssa(
            """
def kernel(x, y):
    return (x + y) * 3
"""
        )
    )
    program = result.require_complete()
    assert program.feed_count == 2
    assert [instruction.op for instruction in program.instructions] == ["add", "mul"]
    assert program.instructions[1].right_scalar == 3.0

    c_result = program.execute(
        (
            CTensor.from_list([1.0, 2.0], (2,)),
            CTensor.from_list([10.0, 20.0], (2,)),
        )
    )
    assert c_result.tolist() == [33.0, 66.0]

    shader = emit_program_source(GlslProgram.from_c_program(program))
    assert "float s2 = s0 + s1;" in shader
    assert "float s3 = s2 * float(3.0);" in shader


def test_bitops_nand_graph_lowers_without_bypassing_provenance():
    result = lower_ssa_to_primitive_program(
        _ssa(
            """
def kernel(x, y):
    return x ^ y
""",
            bitops=True,
        )
    )
    program = result.require_complete()
    ops = [instruction.op for instruction in program.instructions]
    assert "mul" in ops
    assert "logical_not" in ops
    assert "bitxor" not in ops


def test_shape_changing_bitops_report_a_boundary_instead_of_cheating():
    result = lower_ssa_to_primitive_program(
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
