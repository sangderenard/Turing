from src.compiler.ast_process_graph import ast_to_process_graph
from src.compiler.ssa_builder import process_graph_to_ssa_instrs
from src.bitbitbuffer import BitBitBuffer
from src.transmogrifier.process_op import BitQuantaSpec, ProcessOp


SOURCE = """
def kernel(x, y, n):
    z = (x + y) * 3
    if z > n:
        z = z ^ n
    return z
"""


def test_ast_semantics_preserve_dataflow_and_control_merge():
    graph = ast_to_process_graph(SOURCE, filename="fixture.py")
    ops = [data["process_op"].op for _, data in graph.G.nodes(data=True)]

    assert ops == [
        "input",
        "input",
        "input",
        "add",
        "const",
        "mul",
        "gt",
        "bitxor",
        "select",
        "return",
    ]
    select_id = next(
        nid
        for nid, data in graph.G.nodes(data=True)
        if data["process_op"].op == "select"
    )
    assert [role for _, role in graph.G.nodes[select_id]["parents"]] == [
        "condition",
        "if_true",
        "if_false",
    ]


def test_process_op_roundtrips_without_live_python_objects():
    graph = ast_to_process_graph(SOURCE)
    payload = next(
        data["process_op"]
        for _, data in graph.G.nodes(data=True)
        if data["process_op"].op == "const"
    )
    restored = ProcessOp.from_dict(payload.to_dict())
    assert restored == payload
    assert restored.constant == 3


def test_bitbit_quanta_accounting_is_serializable_without_copying_storage():
    buffer = BitBitBuffer(mask_size=4, bitsforbits=8)
    buffer.register_pid_buffer(left=0, right=4, stride=1, label="value")
    accounting = BitQuantaSpec.from_bitbit_buffer(buffer)
    payload = ProcessOp("input", bit_quanta=accounting)
    restored = ProcessOp.from_dict(payload.to_dict())

    assert restored.bit_quanta.quanta == 4
    assert restored.bit_quanta.data_bits == 32
    assert restored.bit_quanta.pid_domains == ("value",)


def test_ssa_preserves_roles_constants_attributes_and_source():
    graph = ast_to_process_graph(SOURCE, filename="fixture.py")
    instrs = process_graph_to_ssa_instrs(graph, schedule="asap")

    const = next(instr for instr in instrs if instr.op == "const")
    select = next(instr for instr in instrs if instr.op == "select")

    assert const.attributes["value"] == 3
    assert const.source_span["filename"] == "fixture.py"
    assert select.arg_roles == ["condition", "if_true", "if_false"]
    assert select.attributes["variable"] == "z"
