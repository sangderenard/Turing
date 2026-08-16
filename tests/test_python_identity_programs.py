import ast
from pathlib import Path

from src.compiler.extraction_contract import ExtractionContract
from src.transmogrifier.graph.graph_express2 import ProcessGraph
from src.transmogrifier.graph.python_identity_programs import (
    PYTHON_IDENTITY_PROGRAMS,
    resolve_python_identity,
)


CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "extraction_contracts"
    / "program_extraction.yaml"
)


def _call_by_identity(graph, identity):
    return next(
        (node_id, data)
        for node_id, data in graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.Call)
        and data.get("attributes", {}).get("extraction_identity") == identity
    )


def test_xor_python_identities_have_explicit_replacement_kinds():
    assert PYTHON_IDENTITY_PROGRAMS
    assert resolve_python_identity("math.sqrt").direct_operator == "sqrt"
    assert resolve_python_identity("random.gauss").direct_operator == "random_source"
    assert resolve_python_identity("builtins.range").object_type == "arithmetic_sequence"
    assert resolve_python_identity("builtins.id").object_type == "stable_object_identity"
    assert resolve_python_identity("builtins.isinstance").object_type == "schema_type_guard"
    assert resolve_python_identity("builtins.super").object_type == "oop_super_dispatch"
    assert resolve_python_identity("builtins.list").object_type == "resident_sequence"
    assert resolve_python_identity("random.seed").object_type == "prng_state"
    assert resolve_python_identity("logging.Logger.debug").kind == "compile_time"
    assert resolve_python_identity(
        "src.common.dt_system.debug.dbg"
    ).kind == "compile_time"
    restore = resolve_python_identity(
        "src.common.dt_system.dt_controller._restore_type"
    )
    assert restore.direct_operator == "cast_like"
    assert restore.steps[0].inputs == ("$arg0", "$arg1")


def test_abstract_tensor_extrema_are_existing_graph_operators():
    maximum = resolve_python_identity(
        "src.common.tensors.abstraction_methods.elementwise.maximum"
    )
    minimum = resolve_python_identity(
        "src.common.tensors.abstraction_methods.elementwise.minimum"
    )

    assert maximum.direct_operator == "maximum"
    assert minimum.direct_operator == "minimum"
    assert maximum.steps[0].inputs == ("$receiver", "$arg0")
    assert minimum.steps[0].inputs == ("$receiver", "$arg0")


def test_python_introspection_is_schema_work_not_interpreter_work():
    getattr_program = resolve_python_identity("builtins.getattr")
    setattr_program = resolve_python_identity("builtins.setattr")

    assert getattr_program.kind == "program"
    assert getattr_program.steps[0].operator == "GetAttr"
    assert setattr_program.kind == "program"
    assert setattr_program.steps[0].operator == "SetAttr"
    assert setattr_program.effects == ("write:receiver",)


def test_len_is_defined_as_shape_leading_axis_not_numel():
    program = resolve_python_identity("builtins.len")

    assert program.kind == "program"
    assert tuple(step.operator for step in program.steps) == (
        "shape", "Constant", "Indexed",
    )
    assert program.output == "result"


def test_math_sqrt_materializes_as_native_operator_with_argument_edge():
    graph = ProcessGraph(materialize_memory=False)
    graph.build_from_ast(
        "import math\ndef kernel(x):\n    return math.sqrt(x)\n",
        resolve_unresolved_parents=True,
        pursuit_roots=("kernel",),
        parent_include=ExtractionContract(CONTRACT),
    )

    node_id, node = _call_by_identity(graph, "math.sqrt")
    assert node["type"] == "sqrt"
    assert node["attributes"]["python_replacement_kind"] == "operator"
    assert graph.G.in_degree(node_id) == 1
    assert "resolved_ast_parent" not in node["attributes"]


def test_composite_and_object_identities_are_attached_to_call_nodes():
    graph = ProcessGraph(materialize_memory=False)
    graph.build_from_ast(
        "def kernel(items):\n    return len(items)\n",
        resolve_unresolved_parents=True,
        pursuit_roots=("kernel",),
        parent_include=ExtractionContract(CONTRACT),
    )

    _, node = _call_by_identity(graph, "builtins.len")
    replacement = node["attributes"]["python_identity_program"]
    assert node["type"] == "Call"
    assert replacement["kind"] == "program"
    assert tuple(step["operator"] for step in replacement["steps"]) == (
        "shape", "Constant", "Indexed",
    )
