from src.compiler.ssa_builder import process_graph_to_ssa_instrs
from src.bitbitbuffer import BitBitBuffer
from src.transmogrifier.graph.graph_express2 import ProcessGraph


SOURCE = """
def kernel(x, y, n):
    z = (x + y) * 3
    if z > n:
        z = z ^ n
    return z
"""

def _graph(source=SOURCE, *, filename=None):
    graph = ProcessGraph(materialize_memory=False)
    graph.build_from_ast(source, filename=filename)
    return graph


def test_ast_semantics_preserve_dataflow_and_control_merge():
    graph = _graph(filename="fixture.py")
    ops = [data["op"] for _, data in graph.G.nodes(data=True)]

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
        if data["op"] == "select"
    )
    assert [role for _, role in graph.G.nodes[select_id]["parents"]] == [
        "condition",
        "if_true",
        "if_false",
    ]


def test_semantic_nodes_store_serializable_metadata_without_live_python_objects():
    graph = _graph()
    payload = next(
        data
        for _, data in graph.G.nodes(data=True)
        if data["op"] == "const"
    )
    assert payload["expr_obj"] is None
    assert payload["constant"] == 3
    assert payload["schema_version"] == 1


def test_bitbit_quanta_accounting_is_serializable_without_copying_storage():
    buffer = BitBitBuffer(mask_size=4, bitsforbits=8)
    buffer.register_pid_buffer(left=0, right=4, stride=1, label="value")
    accounting = buffer.quanta_metadata()
    assert accounting["quanta"] == 4
    assert accounting["data_bits"] == 32
    assert accounting["pid_domains"] == ("value",)


def test_ssa_preserves_roles_constants_attributes_and_source():
    graph = _graph(filename="fixture.py")
    instrs = process_graph_to_ssa_instrs(graph, schedule="asap")

    const = next(instr for instr in instrs if instr.op == "const")
    select = next(instr for instr in instrs if instr.op == "select")

    assert const.attributes["value"] == 3
    assert const.source_span["filename"] == "fixture.py"
    assert select.arg_roles == ["condition", "if_true", "if_false"]
    assert select.attributes["variable"] == "z"
