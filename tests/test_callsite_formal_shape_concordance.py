"""Callsite shape facts follow authored identities into specialized graphs."""

import ast
import inspect
from types import SimpleNamespace

import networkx as nx
import numpy as np

from src.compiler.glsl_deployment_strategy import (
    _concorded_value_type_descriptor,
    _invalidate_tensor_descriptor_dependents,
    _concorded_tensor_result_class,
    _descriptor_states_tensor_contract,
    _descriptor_loses_runtime_rank,
    _fold_callsite_structural_values,
    _publish_callsite_return_members,
    _proven_formal_shape,
    _publish_formal_shape,
    _program_abi_tensor_fact,
    _post_source_value_record_view,
    _publish_parameter_record_views,
    _publish_source_record_constructor_views,
    _post_region_value_obligation,
    _refuse_unsettled_host_scalars,
    _tensor_descriptor,
    _tensor_descriptor_rule,
    outstanding_region_value_obligations,
)
from src.compiler.tensor_ssa_lowering import _settle_operand_shapes
from src.compiler.planner_concordance_posts import (
    _callsite_descriptor_receipt,
    _post_copy_planner_specializations,
)
from src.compiler.concordance_declarations import (
    CANONICAL_VALUE,
    INGEST_SOURCE,
    INGESTION,
    INGESTION_VALUE,
    NodeFact,
    PLANNER_SPECIALIZATION_PAGE,
    PROVEN_LITERAL,
    READ_SCOPE_FORK,
    SOURCE_ACTIVITY_SCOPE,
    SOURCE_VALUE_RECORD_VIEW,
    SOURCE_VALUE_CLASS,
    VALUE_TYPE_ABI,
    ValueTypeABIAlternative,
    ValueTypeABIFact,
)
from src.compiler.identity_concordance import (
    Mode,
    Derived,
    Novel,
    begin_identity_book,
    concordant_shape_transformation_state,
    current_identity_book,
    descriptor_from_shape_transformation_state,
    end_identity_book,
    proven_shape_of,
    record_proven_shape,
    record_shape_transformation,
    shape_scope_of,
)
from src.transmogrifier.ssa import SSAValue
from src.transmogrifier.graph.graph_express2 import ProcessGraph
from src.common.tensors.topological_reducer import node_identity_cell


def test_dynamic_rank_states_a_callsite_return_tensor_contract():
    """A runtime span is a typed return even without static extents."""

    assert _descriptor_states_tensor_contract({
        "shape": (), "dtype": "int64", "rank": 1,
        "metadata_state": "dynamic",
    })
    assert not _descriptor_states_tensor_contract({
        "shape": (), "dtype": "unknown", "rank": 0,
    })


def test_outer_callsite_settles_nested_dynamic_return_before_reading_it(
    monkeypatch,
):
    """A specialized callee reads returns after settling its own call edges."""

    import src.compiler.glsl_deployment_strategy as deployment
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    source = '''
from src.common.tensors.abstraction import AbstractTensor

def make(count: int):
    return AbstractTensor.arange(count).to_dtype("int64")

def carried(count: int):
    value = make(count)
    bit = 1
    while bit < count:
        value = (value + bit).to_dtype("int64")
        bit *= 2
    return value

def root(count: int):
    return carried(count)
'''
    observed = []
    original = deployment._structured_output_descriptor

    def record_nested_return(graph, value_id):
        descriptor = original(graph, value_id)
        if (
            str(graph.G.graph.get("function_name") or "").startswith(
                "carried"
            )
            and any(
                frame.function == "call_result_descriptor"
                for frame in inspect.stack()[1:8]
            )
        ):
            observed.append(descriptor)
        return descriptor

    class StopAfterSpecialization(Exception):
        pass

    def stop_after_specialization(*_args, **_kwargs):
        raise StopAfterSpecialization

    monkeypatch.setattr(
        deployment, "_structured_output_descriptor", record_nested_return,
    )
    monkeypatch.setattr(
        deployment, "_build_shell_hierarchy_plan", stop_after_specialization,
    )
    try:
        lower_ast_source_to_ssa(
            source, "root", name="nested_dynamic_return_fixture",
            runtime_closure_only=True,
            extraction_contract="extraction_contracts/program_extraction.yaml",
            progress=lambda _: None,
        )
    except StopAfterSpecialization:
        pass

    assert observed
    assert all(descriptor == {
        "shape": (), "dtype": "int64", "rank": 1,
        "metadata_state": "dynamic",
    } for descriptor in observed)


def test_constant_array_descriptor_enters_the_value_concordance():
    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph["function_name"] = "pointer_lane"
    graph.G.add_node(
        3, type="Constant", op="const", value_id=3,
        constant=np.zeros(96, dtype=np.float64),
        attributes={"structural_specialization": True},
        parents=[], children=[],
    )

    _book, token = begin_identity_book()
    try:
        assert _tensor_descriptor(graph, 3) == {
            "shape": (96,), "dtype": "float64", "rank": 1,
        }
        assert proven_shape_of(shape_scope_of(graph), 3) == (96,)
    finally:
        end_identity_book(token)


def test_builtin_scalar_cast_declares_rank_zero_result_descriptor():
    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph["function_name"] = "scalar_cast_result"
    graph.G.add_node(
        1, type="Input", op="input", value_id=1,
        attributes={"binding_name": "value", "binding_kind": "parameter"},
        parents=[], children=[],
    )
    graph.G.add_node(
        2, type="float", op="float", value_id=2,
        attributes={"extraction_identity": "builtins.float"},
        parents=[(1, "arg:0")], children=[],
    )
    graph.G.add_node(
        3, type="Constant", op="const", value_id=3, constant=1.0,
        attributes={"value": 1.0}, parents=[], children=[],
    )
    graph.G.add_node(
        4, type="min", op="min", value_id=4,
        expr_obj=ast.parse("min(float(value), 1.0)", mode="eval").body,
        attributes={"extraction_identity": "builtins.min"},
        parents=[(2, "arg:0"), (3, "arg:1")], children=[],
    )
    graph.G.add_edge(1, 2)
    graph.G.add_edges_from(((2, 4), (3, 4)))

    _book, token = begin_identity_book()
    try:
        assert _tensor_descriptor(graph, 2) == {
            "shape": (), "dtype": "float64", "rank": 0,
        }
        assert _tensor_descriptor(graph, 4) == {
            "shape": (), "dtype": "float64", "rank": 0,
        }
    finally:
        end_identity_book(token)


def test_dynamic_arange_cast_retains_rank_and_authored_dtype():
    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph["function_name"] = "dynamic_arange_cast"
    graph.G.add_node(
        1, type="Input", op="input", value_id=1,
        attributes={"binding_name": "count", "binding_kind": "parameter"},
        parents=[], children=[],
    )
    graph.G.add_node(
        2, type="arange", op="arange", value_id=2,
        expr_obj=ast.parse("AbstractTensor.arange(count)", mode="eval").body,
        attributes={"tensor_candidate": "arange"},
        parents=[(1, "arg:0")], children=[],
    )
    graph.G.add_node(
        3, type="Constant", op="const", value_id=3, constant="int64",
        attributes={"value": "int64"}, parents=[], children=[],
    )
    graph.G.add_node(
        4, type="to_dtype", op="to_dtype", value_id=4,
        expr_obj=ast.parse(
            'AbstractTensor.arange(count).to_dtype("int64")', mode="eval",
        ).body,
        attributes={"tensor_candidate": "to_dtype"},
        parents=[(2, "operand"), (3, "arg:0")], children=[],
    )
    graph.G.add_edges_from(((1, 2), (2, 4), (3, 4)))

    _book, token = begin_identity_book()
    try:
        assert _tensor_descriptor(graph, 2) == {
            "shape": (), "dtype": "unknown", "rank": 1,
            "metadata_state": "dynamic",
        }
        assert _tensor_descriptor(graph, 4) == {
            "shape": (), "dtype": "int64", "rank": 1,
            "metadata_state": "dynamic",
        }
    finally:
        end_identity_book(token)


def test_dynamic_arange_with_integer_bound_states_integer_ordinals():
    """Runtime integer bounds determine arange's element dtype."""

    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph["function_name"] = "canonical_codes_from_lengths"
    graph.G.add_node(
        1, type="Input", op="input", value_id=1,
        tensor={"shape": (), "dtype": "int64", "rank": 0},
        attributes={"binding_name": "count", "binding_kind": "parameter"},
        parents=[],
    )
    graph.G.add_node(
        2, type="arange", op="arange", value_id=2,
        attributes={"tensor_candidate": "arange"},
        parents=[(1, "arg:0")],
    )

    _book, token = begin_identity_book()
    try:
        assert _tensor_descriptor_rule(graph, 2) == {
            "shape": (), "dtype": "int64", "rank": 1,
            "metadata_state": "dynamic",
        }
    finally:
        end_identity_book(token)


def test_specialized_arange_replaces_provisional_unknown_dtype():
    """Exact bound evidence outranks the catalogue's unknown dtype."""

    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph.update({
        "function_name": "canonical_codes_from_lengths",
        "planner_tensor_descriptors": {"lengths": {
            "shape": (256,), "dtype": "int64", "rank": 1,
        }},
    })
    graph.G.add_node(
        1, type="Input", op="input", value_id=1,
        tensor={"shape": (), "dtype": "int64", "rank": 0},
        attributes={"binding_name": "count", "binding_kind": "parameter"},
        parents=[],
    )
    graph.G.add_node(
        2, type="arange", op="arange", value_id=2,
        tensor={"shape": (256,), "dtype": "unknown", "rank": 1},
        attributes={"tensor_candidate": "arange"},
        parents=[(1, "arg:0")],
    )
    graph.G.add_edge(1, 2)

    _book, token = begin_identity_book()
    try:
        assert _tensor_descriptor(graph, 2) == {
            "shape": (256,), "dtype": "int64", "rank": 1,
        }
    finally:
        end_identity_book(token)


def test_unsqueeze_reads_exact_dtype_after_stale_shape_only_proof():
    """A prior shape proof cannot erase a later exact call-result dtype."""

    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph["function_name"] = "canonical_codes_from_lengths"
    graph.G.add_node(
        6, type="Call", op="Call", value_id=6,
        tensor={"shape": (256,), "dtype": "int64", "rank": 1},
        parents=[],
    )
    graph.G.add_node(
        63, type="Constant", op="const", value_id=63,
        constant=1, attributes={"value": 1}, parents=[],
    )
    graph.G.add_node(
        64, type="unsqueeze", op="unsqueeze", value_id=64,
        attributes={"tensor_candidate": "unsqueeze"},
        parents=[(6, "operand"), (63, "arg:0")],
    )
    graph.G.add_edges_from(((6, 64), (63, 64)))

    _book, token = begin_identity_book()
    try:
        source = _book.post(
            CANONICAL_VALUE, (("canonical_codes_fixture", 0), 6),
            ("Call", "Call", "symbol_index"),
            stage=INGESTION,
            provenance=Novel(INGEST_SOURCE, ()),
            mode=Mode.CONCORD,
        )
        record_proven_shape(
            shape_scope_of(graph), 6, (256,), "unknown", 1,
            sources=(source,),
        )
        assert _tensor_descriptor(graph, 64) == {
            "shape": (256, 1), "dtype": "int64", "rank": 2,
        }
    finally:
        end_identity_book(token)


def test_referenced_call_header_accepts_exact_specialized_dtype():
    """A shape-only callee header cannot erase its exact call-edge dtype."""

    output = SSAValue(
        7, dtype="unknown", shape=(256,),
        accounting={"program_abi_rank": 1},
    )
    declaration = SimpleNamespace(metadata={
        "native_interface": {"return_container_kind": None},
        "declared_native_outputs": (output,),
    })
    host = SimpleNamespace(
        functions={"arange_root": declaration}, metadata={},
    )
    entry = SimpleNamespace(metadata={
        "host_ssa_module": host,
        "host_ssa_root": "arange_root",
        "host_ssa_outputs": {"arange_root": (output,)},
    })
    table = SimpleNamespace(entry=lambda _reference: entry)
    graph = ProcessGraph(materialize_memory=False, function_table=table)
    graph.G.add_node(
        6, type="Call", op="Call", value_id=6,
        tensor={"shape": (256,), "dtype": "int64", "rank": 1},
        attributes={"method_ref": 1}, parents=[],
    )

    _book, token = begin_identity_book()
    try:
        descriptor = _tensor_descriptor_rule(graph, 6)
        assert descriptor is not None
        assert descriptor["shape"] == (256,)
        assert descriptor["rank"] == 1
        assert descriptor["dtype"] == "int64"
    finally:
        end_identity_book(token)


def test_unknown_linked_boundary_does_not_erase_exact_local_dtype():
    """Unknown is absence of evidence, not a competing element class."""

    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph.update({
        "function_name": "canonical_codes_from_lengths",
        "linked_value_abi": {6: {
            "shape": (256,), "dtype": "unknown", "rank": 1,
        }},
    })
    graph.G.add_node(
        6, type="Call", op="Call", value_id=6,
        tensor={"shape": (256,), "dtype": "int64", "rank": 1},
        parents=[],
    )

    _book, token = begin_identity_book()
    try:
        assert _tensor_descriptor_rule(graph, 6) == {
            "shape": (256,), "dtype": "int64", "rank": 1,
        }
    finally:
        end_identity_book(token)


def test_scalar_tensor_index_preserves_base_dtype_and_removes_axis():
    """A runtime scalar index is an exact tensor transformation edge."""

    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph["function_name"] = "canonical_codes_from_lengths"
    graph.G.add_node(
        120, type="reshape", op="reshape", value_id=120,
        tensor={"shape": (12,), "dtype": "bool", "rank": 1},
        parents=[],
    )
    graph.G.add_node(
        51, type="Sub", op="Sub", value_id=51,
        tensor={"shape": (), "dtype": "int64", "rank": 0},
        parents=[],
    )
    graph.G.add_node(
        123, type="Indexed", op="Indexed", value_id=123,
        attributes={"source_type": "Subscript"},
        parents=[(120, "base"), (51, "index")],
    )
    graph.G.add_edges_from(((120, 123), (51, 123)))

    _book, token = begin_identity_book()
    try:
        assert _tensor_descriptor(graph, 123) == {
            "shape": (), "dtype": "bool", "rank": 0,
        }
    finally:
        end_identity_book(token)


def test_stack_reads_exact_append_writer_for_empty_initializer():
    """A retained append, not an empty initializer, states stack rows."""

    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph.update({
        "function_name": "canonical_codes_from_lengths",
        "source_sequence_mutation_records": {42: {
            "sequence_value_id": 41,
            "argument_value_ids": (39,),
            "operator": "append",
        }},
    })
    graph.G.add_node(
        41, type="List", op=None, value_id=41,
        attributes={
            "producer_kind": "aggregate", "aggregate_kind": "list",
            "aggregate_leaf_value_ids": (), "sequence_writable": True,
        }, parents=[],
    )
    graph.G.add_node(
        39, type="Phi", op="Phi", value_id=39,
        tensor={"shape": (), "dtype": "int64", "rank": 0},
        parents=[],
    )
    graph.G.add_node(
        42, type="Call", op="Call", value_id=42,
        parents=[(41, "operand"), (39, "arg:0")],
    )
    graph.G.add_node(
        91, type="stack", op="stack", value_id=91,
        attributes={"tensor_candidate": "stack", "dim": 0},
        parents=[(41, "arg:0")],
    )
    graph.G.add_edges_from(((41, 42), (39, 42), (41, 91)))

    _book, token = begin_identity_book()
    try:
        assert _tensor_descriptor_rule(graph, 91) == {
            "shape": (), "dtype": "int64", "rank": 1,
            "metadata_state": "dynamic",
        }
    finally:
        end_identity_book(token)


def test_dynamic_cat_reads_exact_aggregate_members_and_retains_span_rank():
    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph["function_name"] = "dynamic_cat"
    for node_id in (1, 2):
        graph.G.add_node(
            node_id, type="Input", op="input", value_id=node_id,
            tensor={
                "shape": (), "dtype": "int64", "rank": 1,
                "metadata_state": "dynamic",
            },
            attributes={
                "binding_name": f"part_{node_id}",
                "binding_kind": "parameter",
            },
            parents=[], children=[],
        )
    graph.G.add_node(
        3, type="Tuple", op=None, value_id=3,
        expr_obj=ast.parse("(part_1, part_2)", mode="eval").body,
        attributes={
            "producer_kind": "aggregate", "aggregate_kind": "tuple",
            "aggregate_leaf_value_ids": (1, 2),
        },
        parents=[(1, "elts"), (2, "elts")], children=[],
    )
    graph.G.add_node(
        4, type="Constant", op="const", value_id=4, constant=0,
        attributes={"value": 0}, parents=[], children=[],
    )
    graph.G.add_node(
        5, type="cat", op="cat", value_id=5,
        expr_obj=ast.parse(
            "AbstractTensor.cat((part_1, part_2), dim=0)", mode="eval",
        ).body,
        attributes={"tensor_candidate": "cat"},
        parents=[(3, "arg:0"), (4, "kw:dim")], children=[],
    )
    graph.G.add_edges_from(((1, 3), (2, 3), (3, 5), (4, 5)))

    _book, token = begin_identity_book()
    try:
        assert _tensor_descriptor(graph, 5) == {
            "shape": (), "dtype": "int64", "rank": 1,
            "metadata_state": "dynamic",
        }
    finally:
        end_identity_book(token)


def test_scatter_retains_destination_storage_contract_before_explicit_cast():
    """Dynamic index/value rows do not erase scatter destination storage."""

    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph["function_name"] = "_code_lengths"
    graph.G.add_node(
        0, type="zeros", op="zeros", value_id=0,
        tensor={"shape": (256,), "dtype": "float64", "rank": 1},
        parents=[],
    )
    dynamic_int_vector = {
        "shape": (), "dtype": "int64", "rank": 1,
        "metadata_state": "dynamic",
    }
    for node_id in (1, 2):
        graph.G.add_node(
            node_id, type="Input", op="input", value_id=node_id,
            tensor=dict(dynamic_int_vector), parents=[],
        )
    graph.G.add_node(
        3, type="scatter", op="scatter", value_id=3,
        attributes={"tensor_candidate": "scatter"},
        parents=[(0, "arg:0"), (1, "arg:1"), (2, "arg:2")],
    )
    graph.G.add_node(
        4, type="Constant", op="const", value_id=4,
        constant="int64", attributes={"value": "int64"}, parents=[],
    )
    graph.G.add_node(
        5, type="to_dtype", op="to_dtype", value_id=5,
        attributes={"tensor_candidate": "to_dtype"},
        parents=[(3, "operand"), (4, "arg:0")],
    )
    graph.G.add_edges_from(
        ((0, 3), (1, 3), (2, 3), (3, 5), (4, 5))
    )

    _book, token = begin_identity_book()
    try:
        assert _tensor_descriptor_rule(graph, 3) == {
            "shape": (256,), "dtype": "float64", "rank": 1,
        }
        assert _tensor_descriptor_rule(graph, 5) == {
            "shape": (256,), "dtype": "int64", "rank": 1,
        }
    finally:
        end_identity_book(token)


def test_declared_scalar_record_field_is_a_rank_zero_descriptor():
    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph["function_name"] = "scalar_record_field"
    record = {
        "identity": "example.Targets",
        "fields": {
            "limit": {
                "storage": "scalar", "dtype": "float64", "rank": 0,
            },
        },
    }
    graph.G.graph["program_abi"] = {"records": {"Targets": record}}
    graph.G.graph["parameter_record_abi"] = {"targets": record}
    graph.G.add_node(
        1, type="Input", op="input", value_id=1,
        attributes={"binding_name": "targets", "binding_kind": "parameter"},
        parents=[], children=[],
    )
    graph.G.add_node(
        2, type="GetAttr", op="GetAttr", value_id=2,
        expr_obj=ast.parse("targets.limit", mode="eval").body,
        attributes={"attribute": "limit"},
        parents=[(1, "value")], children=[],
    )
    graph.G.add_node(
        3, type="Constant", op="const", value_id=3, constant=2.0,
        attributes={"value": 2.0},
        parents=[], children=[],
    )
    graph.G.add_node(
        4, type="Div", op="Div", value_id=4,
        expr_obj=ast.parse("measure / targets.limit", mode="eval").body,
        attributes={}, parents=[(3, "lhs"), (2, "rhs")], children=[],
    )
    graph.G.add_edges_from(((1, 2), (3, 4), (2, 4)))

    _book, token = begin_identity_book()
    try:
        expected = {"shape": (), "dtype": "float64", "rank": 0}
        assert _tensor_descriptor(graph, 2) == expected
        assert _tensor_descriptor(graph, 4) == expected
    finally:
        end_identity_book(token)


def test_numeric_literal_views_carry_shape_into_matmul_concordance():
    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph["function_name"] = "literal_projection"
    graph.G.add_node(
        1, type="Constant", op="const", value_id=1,
        constant=[float(index) for index in range(8)],
        attributes={"value": [float(index) for index in range(8)]},
        parents=[],
    )
    for node_id, value in ((2, 0), (4, 2), (6, 0), (8, 1)):
        graph.G.add_node(
            node_id, type="Constant", op="const", value_id=node_id,
            constant=value, attributes={"value": value}, parents=[],
        )
    graph.G.add_node(
        3, type="unsqueeze", op="unsqueeze", value_id=3,
        attributes={"tensor_candidate": "unsqueeze"},
        parents=[(1, "operand"), (2, "arg:0")],
    )
    graph.G.add_node(
        5, type="unsqueeze", op="unsqueeze", value_id=5,
        attributes={"tensor_candidate": "unsqueeze"},
        parents=[(3, "operand"), (4, "arg:0")],
    )
    graph.G.add_node(
        7, type="unsqueeze", op="unsqueeze", value_id=7,
        attributes={"tensor_candidate": "unsqueeze"},
        parents=[(1, "operand"), (6, "arg:0")],
    )
    graph.G.add_node(
        9, type="unsqueeze", op="unsqueeze", value_id=9,
        attributes={"tensor_candidate": "unsqueeze"},
        parents=[(7, "operand"), (8, "arg:0")],
    )
    graph.G.add_node(
        10, type="matmul", op="matmul", value_id=10,
        attributes={}, parents=[(5, "lhs"), (9, "rhs")],
    )
    graph.G.add_edges_from((
        (1, 3), (2, 3), (3, 5), (4, 5),
        (1, 7), (6, 7), (7, 9), (8, 9), (5, 10), (9, 10),
    ))

    _book, token = begin_identity_book()
    try:
        assert _tensor_descriptor(graph, 10) == {
            "shape": (1, 8, 8), "dtype": "float64", "rank": 3,
        }
        assert proven_shape_of(shape_scope_of(graph), 10) == (1, 8, 8)
    finally:
        end_identity_book(token)


def test_specialized_graph_reads_authored_formal_shape_row():
    """A specialization suffix must not create a second formal identity."""

    graph = nx.DiGraph()
    graph.graph["function_name"] = (
        "artifact___first_occurrence__specialized_5171c8115542"
    )
    specialized = SimpleNamespace(G=graph)

    _book, token = begin_identity_book()
    try:
        _publish_formal_shape(
            "_first_occurrence",
            "mask",
            {"shape": (2,), "dtype": "float64"},
            "_pivot_mask",
        )

        assert _proven_formal_shape(specialized, "mask") == {
            "shape": (2,),
            "dtype": "float64",
            "rank": 1,
        }
        page = current_identity_book().page("formal_shape")
        assert page.rows() == (("_first_occurrence", "mask"),)
    finally:
        end_identity_book(token)


def test_parametric_formal_shape_repeated_caller_is_not_a_new_mutation():
    """A settled conflict changes only when a genuinely new callsite is read."""

    _book, token = begin_identity_book()
    try:
        book = current_identity_book()
        for value_id, name in enumerate(("first", "second", "third"), 1):
            book.post(
                CANONICAL_VALUE, (("callers", 0), value_id),
                ("Input", "input", name),
                stage=INGESTION,
                provenance=Novel(INGEST_SOURCE, ()),
                mode=Mode.CONCORD,
            )
        page = book.page("formal_shape")
        first = book.latest_ref(
            CANONICAL_VALUE, (("callers", 0), 1),
        )
        second = book.latest_ref(
            CANONICAL_VALUE, (("callers", 0), 2),
        )
        third = book.latest_ref(
            CANONICAL_VALUE, (("callers", 0), 3),
        )

        assert _publish_formal_shape(
            "callee", "value",
            {"shape": (2,), "dtype": "float64"}, "first",
            (first,),
        )
        assert _publish_formal_shape(
            "callee", "value",
            {"shape": (3,), "dtype": "float64"}, "second",
            (second,),
        )
        conflict_history_length = len(page.history(("callee", "value")))

        assert not _publish_formal_shape(
            "callee", "value",
            {"shape": (3,), "dtype": "float64"}, "second",
            (second,),
        )
        assert len(page.history(("callee", "value"))) == conflict_history_length

        assert _publish_formal_shape(
            "callee", "value",
            {"shape": (4,), "dtype": "float64"}, "third",
            (third,),
        )
        third_history_length = len(page.history(("callee", "value")))
        assert not _publish_formal_shape(
            "callee", "value",
            {"shape": (2,), "dtype": "float64"}, "first",
            (first,),
        )
        assert len(page.history(("callee", "value"))) == third_history_length
    finally:
        end_identity_book(token)


def test_exact_polymorphic_shape_conflicts_at_current_concordance_depth():
    """A later exact specialization must not inherit a peer's deeper shape."""

    _book, token = begin_identity_book()
    try:
        record_proven_shape("_row", 6, (2, 2), "float64", 4)
        record_proven_shape("_row", 6, (2, 1), "float64", None)

        assert proven_shape_of("_row", 6) is None
        page = current_identity_book().page("proven_shape")
        assert page.latest(("_row", 6))[0] == "conflicting"
    finally:
        end_identity_book(token)


def test_polymorphic_helper_keeps_exact_specialization_operand_shape():
    """A peer specialization's proven extent cannot restamp a local value."""

    _book, token = begin_identity_book()
    try:
        record_proven_shape("_row", 6, (2, 2), "float64", 2)
        page = current_identity_book().page("value_shape")
        page.set(
            ("_row", 0), 0,
            ("from matrix caller", (2, 2), "float64", "span"),
        )
        page.set(
            ("_row", 0), 1,
            ("from column caller", (2, 1), "float64", "span"),
        )
        local = SSAValue(6, dtype="float64", shape=(2, 1))

        column = SimpleNamespace(
            name="artifact___row__specialized_column", blocks={}, metadata={},
        )
        _settle_operand_shapes(column, (local,))

        assert local.shape == (2, 1)
    finally:
        end_identity_book(token)


def test_region_operand_reads_later_unanimous_tensor_descriptor_shape():
    """A captured tensor projection cannot remain scalar at its consumer."""

    _book, token = begin_identity_book()
    try:
        operand = SSAValue(710, dtype="float64", shape=())
        region = SimpleNamespace(
            name="artifact__planned_region_267", blocks={}, metadata={},
        )
        descriptor = SimpleNamespace(shape=(144, 288))

        _settle_operand_shapes(
            region, (operand,), lambda value: (
                descriptor if int(value.id) == 710 else None
            ),
        )

        assert operand.shape == (144, 288)
    finally:
        end_identity_book(token)


def test_existing_call_projection_is_enriched_by_later_exact_descriptor():
    """A resident projection id does not freeze an incomplete first round."""

    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph["function_name"] = "solve"
    graph.G.add_node(
        27, type="Call", op="Call", value_id=27,
        attributes={}, parents=[], children=[],
    )
    initial = (
        {"shape": (2, 2), "dtype": "float64"},
        {"shape": (), "dtype": "float64"},
        None,
    )
    exact = (
        initial[0], initial[1],
        {"shape": (2, 2), "dtype": "float64"},
    )

    _book, token = begin_identity_book()
    try:
        assert _publish_callsite_return_members(
            graph, 27, initial, "tuple",
        )
        leaves = tuple(
            graph.G.nodes[27]["attributes"]["aggregate_leaf_value_ids"]
        )
        permutation = leaves[2]
        assert graph.G.nodes[permutation]["tensor"] == {}

        assert _publish_callsite_return_members(
            graph, 27, exact, "tuple",
        )

        assert tuple(
            graph.G.nodes[27]["attributes"]["aggregate_leaf_value_ids"]
        ) == leaves
        assert graph.G.nodes[permutation]["tensor"] == exact[2]
        # Descriptor revisions now live on the shared declared member page,
        # keyed by caller-copy scope, not the retired action-log name page.
        page = current_identity_book().page("callsite_return_member")
        history = page.history((("function", "solve"), 27, 2, permutation))
        assert history == (
            (0, None),
            (1, ((2, 2), "float64", 2, None, None, None, None)),
        )
    finally:
        end_identity_book(token)


def test_polymorphic_specialization_rederives_intermediate_shape_locally():
    """A peer's authored-row proof cannot enter an exact descriptor graph."""

    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph.update({
        "function_name": "_row",
        "planner_tensor_descriptors": {
            "matrix": {"shape": (2, 1), "dtype": "float64"},
            "selector": {"shape": (2, 1), "dtype": "float64"},
        },
    })
    graph.G.add_node(
        0, type="Input", op="input", value_id=0,
        tensor={"shape": (2, 1), "dtype": "float64"},
        attributes={"binding_name": "matrix"}, parents=[], children=[],
    )
    graph.G.add_node(
        1, type="Input", op="input", value_id=1,
        tensor={"shape": (2, 1), "dtype": "float64"},
        attributes={"binding_name": "selector"}, parents=[], children=[],
    )
    graph.G.add_node(
        2, type="Mul", op="mul", value_id=2,
        parents=[(0, "lhs"), (1, "rhs")], children=[(3, "operand")],
        attributes={"tensor_candidate": "mul"},
    )
    graph.G.add_node(
        3, type="sum", op="sum", value_id=3,
        parents=[(2, "operand")], children=[],
        attributes={"tensor_candidate": "sum", "dim": -2},
    )
    graph.G.add_edges_from(((0, 2), (1, 2), (2, 3)))

    _book, token = begin_identity_book()
    try:
        record_proven_shape(shape_scope_of(graph), 2, (2, 2), "float64", 2)
        _publish_formal_shape(
            "_row", "matrix",
            {"shape": (2, 2), "dtype": "float64"}, "matrix_lane",
        )
        _publish_formal_shape(
            "_row", "matrix",
            {"shape": (2, 1), "dtype": "float64"}, "column_lane",
        )

        assert _tensor_descriptor(graph, 3) == {
            "shape": (1,), "dtype": "float64", "rank": 1,
        }
        history = current_identity_book().page("proven_shape").history(
            (shape_scope_of(graph), 2)
        )
        assert history[-1][1][0] == "conflicting"
    finally:
        end_identity_book(token)


def test_axis_reduction_keeps_tracked_result_when_operand_rank_is_unsettled():
    """A transient scalar operand fact cannot erase a reduction descriptor."""

    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph.update({
        "function_name": "reduce_rows",
        "planner_tensor_descriptors": {"value": {
            "shape": (), "dtype": "float64",
        }},
    })
    graph.G.add_node(
        0, type="Input", op="input", value_id=0,
        tensor={"shape": (), "dtype": "float64"},
        attributes={"binding_name": "value"}, parents=[], children=[
            (1, "operand"),
        ],
    )
    graph.G.add_node(
        1, type="sum", op="sum", value_id=1,
        tensor={"shape": (8,), "dtype": "float64"},
        parents=[(0, "operand")], children=[],
        attributes={"tensor_candidate": "sum", "dim": 2},
    )
    graph.G.add_edge(0, 1)

    _book, token = begin_identity_book()
    try:
        assert _tensor_descriptor(graph, 1) == {
            "shape": (8,), "dtype": "float64", "rank": 1,
        }
    finally:
        end_identity_book(token)


def test_leading_axis_reduction_uses_dynamic_sequence_row_shape():
    """A dynamic row count does not hide the statically known row extents."""

    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph["function_name"] = "reduce_dynamic_rows"
    graph.G.add_node(
        0, type="Input", op="input", value_id=0,
        tensor={
            "shape": (),
            "dtype": "float64",
            "rank": 2,
            "metadata_state": "dynamic",
            "sequence_row_shape": (3,),
        },
        attributes={"binding_name": "rows"}, parents=[], children=[
            (1, "operand"),
        ],
    )
    graph.G.add_node(
        1, type="mean", op="mean", value_id=1,
        parents=[(0, "operand")], children=[],
        attributes={"tensor_candidate": "mean", "dim": 0},
    )
    graph.G.add_edge(0, 1)

    _book, token = begin_identity_book()
    try:
        assert _tensor_descriptor(graph, 1) == {
            "shape": (3,), "dtype": "float64", "rank": 1,
        }
        assert proven_shape_of(shape_scope_of(graph), 1) == (3,)
    finally:
        end_identity_book(token)


def test_dependency_change_invalidates_and_then_reproves_concordance_shape():
    """Clearing a graph cache also withdraws its stale identity-level proof."""

    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph["function_name"] = "helper"
    graph.G.add_node(1, type="Call", op="Call", value_id=1, parents=[])
    graph.G.add_node(
        2, type="Add", op="add", value_id=2,
        parents=[(1, "lhs")], tensor={"shape": (2,), "dtype": "float64"},
    )
    graph.G.add_edge(1, 2)

    _book, token = begin_identity_book()
    try:
        record_proven_shape(shape_scope_of(graph), 2, (2,), "float64", 4)
        _invalidate_tensor_descriptor_dependents(
            graph, (1,), "call-result-specialization-changed",
        )

        assert "tensor" not in graph.G.nodes[2]
        assert proven_shape_of(shape_scope_of(graph), 2) is None
        assert current_identity_book().page("proven_shape").latest(
            (shape_scope_of(graph), 2)
        ) == (
            "invalidated", 1, "call-result-specialization-changed",
        )

        record_proven_shape(shape_scope_of(graph), 2, (1,), "float64", 2)
        assert proven_shape_of(shape_scope_of(graph), 2) == (1,)
    finally:
        end_identity_book(token)


def test_a_callsite_boundary_keeps_its_runtime_rank_on_an_untyped_formal():
    # ``_require_tensor(self.codes, "codes")``: an unannotated formal bound
    # at the callsite to a rank-1 runtime-sized span.  Its empty shape is
    # runtime extents, not rank 0.
    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph["function_name"] = "_require_tensor"
    span = {
        "shape": (), "dtype": "int64", "rank": 1, "metadata_state": "dynamic",
    }
    graph.G.graph["planner_tensor_descriptors"] = {"value": dict(span)}
    graph.G.add_node(
        1, type="Input", op="input", value_id=1,
        attributes={
            "binding_name": "value", "binding_kind": "parameter",
            "program_abi_storage": "scalar", "program_abi_rank": 0,
        },
        parents=[], children=[],
    )

    _book, token = begin_identity_book()
    try:
        descriptor = _tensor_descriptor(graph, 1)
        assert descriptor["rank"] == 1
        assert descriptor["metadata_state"] == "dynamic"
    finally:
        end_identity_book(token)


def test_runtime_rank_one_span_broadcast_with_scalar_stays_runtime_rank_one():
    """The static broadcaster must not reinterpret empty dynamic extents."""

    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph["function_name"] = "runtime_span_scalar_product"
    graph.G.add_node(
        1, type="Input", op="input", value_id=1,
        tensor={
            "shape": (), "dtype": "int64", "rank": 1,
            "metadata_state": "dynamic",
        },
        attributes={"binding_name": "values", "binding_kind": "parameter"},
        parents=[], children=[],
    )
    graph.G.add_node(
        2, type="Constant", op="const", value_id=2, constant=4,
        attributes={"value": 4}, parents=[], children=[],
    )
    graph.G.add_node(
        3, type="Mul", op="Mul", value_id=3, attributes={},
        parents=[(1, "lhs"), (2, "rhs")], children=[],
    )

    _book, token = begin_identity_book()
    try:
        assert _tensor_descriptor(graph, 3) == {
            "shape": (), "dtype": "int64", "rank": 1,
            "metadata_state": "dynamic",
        }
    finally:
        end_identity_book(token)


def test_axis_reduction_of_dynamic_rank_two_span_keeps_dynamic_rank_one():
    """A known axis transforms runtime rank even without static extents."""

    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph["function_name"] = "dynamic_membership_reduction"
    graph.G.add_node(
        1, type="Input", op="input", value_id=1,
        tensor={
            "shape": (), "dtype": "int64", "rank": 2,
            "metadata_state": "dynamic",
        },
        attributes={"binding_name": "membership", "binding_kind": "parameter"},
        parents=[], children=[],
    )
    graph.G.add_node(
        2, type="Constant", op="const", value_id=2, constant=0,
        attributes={"value": 0}, parents=[], children=[],
    )
    graph.G.add_node(
        3, type="sum", op="sum", value_id=3,
        attributes={"tensor_candidate": "sum", "dim": 0, "keepdim": False},
        parents=[(1, "operand"), (2, "kw:dim")], children=[],
    )
    graph.G.add_edges_from(((1, 3), (2, 3)))

    _book, token = begin_identity_book()
    try:
        assert _tensor_descriptor_rule(graph, 3) == {
            "shape": (), "dtype": "int64", "rank": 1,
            "metadata_state": "dynamic",
        }
    finally:
        end_identity_book(token)


def test_dynamic_flatten_unsqueeze_and_shape_reshape_keep_exact_contract():
    """Runtime extents do not hide the closed rank laws of tensor views."""

    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph.update({
        "function_name": "dynamic_view_chain",
        "planner_tensor_descriptors": {"values": {
            "shape": (), "dtype": "int64", "rank": 1,
            "metadata_state": "dynamic",
        }},
    })
    graph.G.add_node(
        0, type="Input", op="input", value_id=0,
        tensor={
            "shape": (), "dtype": "int64", "rank": 1,
            "metadata_state": "dynamic",
        },
        attributes={"binding_name": "values"}, parents=[], children=[],
    )
    graph.G.add_node(
        1, type="flatten", op="flatten", value_id=1,
        tensor={"shape": (), "dtype": "int64", "rank": 0},
        attributes={"tensor_candidate": "flatten"},
        parents=[(0, "operand")], children=[],
    )
    graph.G.add_node(
        2, type="Constant", op="constant", value_id=2, constant=1,
        attributes={"value": 1}, parents=[], children=[],
    )
    graph.G.add_node(
        3, type="unsqueeze", op="unsqueeze", value_id=3,
        tensor={"shape": (), "dtype": "int64", "rank": 0},
        attributes={"tensor_candidate": "unsqueeze"},
        parents=[(1, "operand"), (2, "arg:0")], children=[],
    )
    graph.G.add_node(
        4, type="GetAttr", op="GetAttr", value_id=4,
        attributes={"attribute": "shape"},
        parents=[(0, "value")], children=[],
    )
    graph.G.add_node(
        5, type="reshape", op="reshape", value_id=5,
        tensor={"shape": (), "dtype": "int64", "rank": 0},
        attributes={"tensor_candidate": "reshape"},
        parents=[(3, "operand"), (4, "arg:0")], children=[],
    )
    graph.G.add_edges_from(((0, 1), (1, 3), (2, 3), (0, 4), (3, 5), (4, 5)))

    _book, token = begin_identity_book()
    try:
        assert _tensor_descriptor_rule(graph, 1) == {
            "shape": (), "dtype": "int64", "rank": 1,
            "metadata_state": "dynamic",
        }
        assert _tensor_descriptor_rule(graph, 3) == {
            "shape": (), "dtype": "int64", "rank": 2,
            "metadata_state": "dynamic",
        }
        assert _tensor_descriptor_rule(graph, 5) == {
            "shape": (), "dtype": "int64", "rank": 1,
            "metadata_state": "dynamic",
        }
        graph.G.nodes[4].update({
            "type": "Constant", "op": "const", "constant": (4,),
            "attributes": {"value": (4,), "structural_specialization": True},
            "parents": [],
        })
        assert _tensor_descriptor_rule(graph, 5) == {
            "shape": (4,), "dtype": "int64", "rank": 1,
        }
    finally:
        end_identity_book(token)


def test_reshape_reads_folded_shape_identity_from_concordance_after_edge_retired():
    """A graph rewrite cannot erase reshape's recorded shape identity."""

    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph["function_name"] = "lookup"
    graph.G.add_node(
        30, type="Input", op="input", value_id=30,
        tensor={
            "shape": (), "dtype": "int64", "rank": 1,
            "metadata_state": "dynamic",
        },
        parents=[], children=[],
    )
    graph.G.add_node(
        67, type="Constant", op="const", value_id=67, constant=(),
        attributes={"value": (), "structural_specialization": True},
        parents=[], children=[],
    )
    graph.G.add_node(
        87, type="reshape", op="reshape", value_id=87,
        tensor={"shape": (), "dtype": "int64", "rank": 0},
        attributes={"tensor_candidate": "reshape"},
        # The late structural rewrite has retired arg:0 from this mutable
        # view.  Its exact copy-scoped edge remains on the concordance.
        parents=[(30, "operand"), (67, "arg:0")], children=[],
    )
    graph.G.add_edges_from(((30, 87), (67, 87)))

    book, token = begin_identity_book()
    try:
        scope = book.mint_scope("lookup|shape", READ_SCOPE_FORK)
        graph.G.graph.update(
            ingestion_value_scope=scope,
            lexical_read_scope=scope,
            operand_position_scope=scope,
            shape_scope=scope,
        )
        source = book.post(
            INGESTION_VALUE, (scope, 67),
            NodeFact("GetAttr", "GetAttr", "values.shape"),
            stage=READ_SCOPE_FORK,
            provenance=Novel(INGEST_SOURCE, ()),
            mode=Mode.CONCORD,
        )
        record_shape_transformation(
            scope, 67, scope, 87,
            stage="graph_tensor_descriptor", operation="reshape",
            role="arg:0",
            source_state=((), "int64", 1, "dynamic", None),
            target_state=((), "int64", 1, "dynamic", None),
            source_cells=(source,),
        )

        assert _tensor_descriptor_rule(graph, 87) == {
            "shape": (), "dtype": "int64", "rank": 1,
            "metadata_state": "dynamic",
        }
    finally:
        end_identity_book(token)


def test_exact_record_view_is_bound_to_forwarded_method_receiver_formal():
    """A constructor-specialized record crosses the next exact call edge."""

    caller = ProcessGraph(materialize_memory=False)
    caller.G.graph["function_name"] = "encode"
    caller.G.add_node(
        7, type="Call", op="Call", value_id=7,
        attributes={"program_abi_record_identity": "Table"},
        parents=[], children=[],
    )
    callee = ProcessGraph(materialize_memory=False)
    callee.G.graph.update({
        "function_name": "lookup",
        "identity_table": {"self": (0,)},
        "parameter_record_abi": {
            "self": {
                "identity": "Table",
                "fields": {
                    "lengths": {
                        "storage": "span", "dtype": "unknown", "rank": 1,
                    },
                },
            },
        },
    })
    callee.G.add_node(
        0, type="Input", op="input", value_id=0,
        attributes={
            "binding_name": "self", "binding_kind": "parameter",
            "program_abi_record_identity": "Table",
        },
        parents=[], children=[],
    )
    callee.G.add_node(
        1, type="GetAttr", op="GetAttr", value_id=1,
        attributes={"attribute": "lengths"},
        parents=[(0, "value")], children=[],
    )
    callee.G.add_edge(0, 1)
    callee.G.add_node(
        2, type="Input", op="input", value_id=2,
        tensor={
            "shape": (), "dtype": "int64", "rank": 1,
            "metadata_state": "dynamic",
        },
        attributes={"binding_name": "indices", "binding_kind": "parameter"},
        parents=[], children=[],
    )
    callee.G.add_node(
        3, type="Indexed", op="Indexed", value_id=3,
        parents=[(1, "base"), (2, "index")], children=[],
    )
    callee.G.add_edges_from(((1, 3), (2, 3)))
    exact = {
        "identity": "Table",
        "fields": {
            "lengths": {
                "storage": "span", "dtype": "int64", "shape": (),
                "rank": 1, "metadata_state": "dynamic",
            },
        },
    }

    book, token = begin_identity_book()
    try:
        for owner, label in ((caller, "record_caller"), (callee, "record_callee")):
            scope = book.mint_scope(label, READ_SCOPE_FORK)
            owner.G.graph.update({
                "ingestion_value_scope": scope,
                "lexical_read_scope": scope,
                "operand_position_scope": scope,
            })
            for node_id, data in owner.G.nodes(data=True):
                book.post(
                    INGESTION_VALUE, (scope, int(node_id)),
                    NodeFact(data.get("type"), data.get("op"), data.get("label")),
                    stage=READ_SCOPE_FORK,
                    provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD,
                )
        # Settle the generic field descriptor first.  Publishing the exact
        # receiver record must withdraw this stale state and recompute it.
        assert _tensor_descriptor(callee, 1) is None
        caller_view = _post_source_value_record_view(
            caller, 7, exact, sources=(node_identity_cell(caller, 7),),
        )
        assert _publish_parameter_record_views(
            callee, {"self": exact}, {"self": (caller_view,)},
        ) == 1
        assert descriptor_from_shape_transformation_state(
            concordant_shape_transformation_state(
                shape_scope_of(callee), 3,
            )
        ) == {
            "shape": (), "dtype": "int64", "rank": 1,
            "metadata_state": "dynamic",
        }
        assert _tensor_descriptor(callee, 1) == {
            "shape": (), "dtype": "int64", "rank": 1,
            "metadata_state": "dynamic",
        }
        formal_view = book.latest_ref(
            SOURCE_VALUE_RECORD_VIEW, (shape_scope_of(callee), 0),
        )
        assert caller_view in {
            source for source, _stage in book.edges_into(formal_view)
        }
    finally:
        end_identity_book(token)


def test_constructor_record_view_carries_exact_optional_absence():
    """A specialized ``None`` constructor actual owns field presence."""

    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph.update({
        "function_name": "from_code_lengths",
        "function_parameters": ("symbols",),
        "identity_table": {"symbols": (1,)},
        "program_abi": {
            "records": {
                "Table": {
                    "identity": "Table",
                    "fields": {
                        "symbols": {
                            "storage": "span", "dtype": "unknown",
                            "rank": 1, "optional": True,
                        },
                    },
                },
            },
        },
    })
    graph.G.add_node(
        1, type="Input", op="input", value_id=1,
        attributes={"binding_name": "symbols", "binding_kind": "parameter"},
        parents=[], children=[(5, "kw:symbols")],
    )
    graph.G.add_node(
        5, type="Call", op="Call", value_id=5,
        attributes={"class_ref": "Table"},
        parents=[(1, "kw:symbols")], children=[],
    )
    graph.G.add_edge(1, 5, role="kw:symbols")

    book, token = begin_identity_book()
    try:
        scope = book.mint_scope("constructor_optional", READ_SCOPE_FORK)
        graph.G.graph.update({
            "ingestion_value_scope": scope,
            "lexical_read_scope": scope,
            "operand_position_scope": scope,
            "shape_scope": scope,
        })
        ingestion_cells = tuple(
            book.post(
                INGESTION_VALUE, (scope, int(node_id)),
                NodeFact(data.get("type"), data.get("op"), data.get("label")),
                stage=READ_SCOPE_FORK,
                provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD,
            )
            for node_id, data in graph.G.nodes(data=True)
        )
        activity_scope = book.mint_scope(
            "lookup_optional_activity", READ_SCOPE_FORK,
        )
        book.post(
            SOURCE_ACTIVITY_SCOPE, (scope,), activity_scope,
            stage=READ_SCOPE_FORK,
            provenance=Derived(ingestion_cells), mode=Mode.CONCORD,
        )
        _post_copy_planner_specializations(graph, {"symbols": None}, {})

        assert _publish_source_record_constructor_views(graph) == 1
        record = book.page(SOURCE_VALUE_RECORD_VIEW).latest((scope, 5))
        assert record["fields"]["symbols"]["present"] is False
        view_cell = book.latest_ref(SOURCE_VALUE_RECORD_VIEW, (scope, 5))
        planner_cell = book.latest_ref(
            PLANNER_SPECIALIZATION_PAGE, (scope, "symbols"),
        )
        assert planner_cell in {
            source for source, _stage in book.edges_into(view_cell)
        }
    finally:
        end_identity_book(token)


def test_optional_presence_fold_reads_exact_receiver_record_view():
    """A presence slot folds only from its receiver's concordance row."""

    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph.update({
        "function_name": "lookup",
        "function_parameters": ("self",),
        "identity_table": {"self": (0,)},
    })
    graph.G.add_node(
        0, type="Input", op="input", value_id=0,
        attributes={"binding_name": "self", "binding_kind": "parameter"},
        parents=[], children=[],
    )
    graph.G.add_node(
        1, type="Input", op="input", value_id=1,
        tensor={"shape": (), "dtype": "bool"},
        attributes={
            "binding_kind": "program_abi_optional_presence",
            "program_abi_parameter": "self",
            "program_abi_field": "symbols",
            "program_abi_optional_presence": True,
            "program_abi_optional_present_when": True,
        },
        parents=[], children=[(8, "operand")],
    )
    graph.G.add_node(
        8, type="logical_not", op="logical_not", value_id=8,
        tensor={"shape": (), "dtype": "bool"}, attributes={},
        parents=[(1, "operand")], children=[], expr_obj=None,
    )
    graph.G.add_edge(1, 8, role="operand")
    exact = {
        "identity": "Table",
        "fields": {
            "symbols": {
                "storage": "span", "dtype": "unknown", "rank": 1,
                "optional": True, "present": False,
            },
        },
    }

    book, token = begin_identity_book()
    try:
        scope = book.mint_scope("lookup_optional", READ_SCOPE_FORK)
        graph.G.graph.update({
            "ingestion_value_scope": scope,
            "lexical_read_scope": scope,
            "operand_position_scope": scope,
            "shape_scope": scope,
        })
        ingestion_cells = tuple(
            book.post(
                INGESTION_VALUE, (scope, int(node_id)),
                NodeFact(data.get("type"), data.get("op"), data.get("label")),
                stage=READ_SCOPE_FORK,
                provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD,
            )
            for node_id, data in graph.G.nodes(data=True)
        )
        activity_scope = book.mint_scope(
            "lookup_optional_activity", READ_SCOPE_FORK,
        )
        book.post(
            SOURCE_ACTIVITY_SCOPE, (scope,), activity_scope,
            stage=READ_SCOPE_FORK,
            provenance=Derived(ingestion_cells), mode=Mode.CONCORD,
        )
        record_cell = _post_source_value_record_view(
            graph, 0, exact, sources=(node_identity_cell(graph, 0),),
        )

        _fold_callsite_structural_values(graph)

        assert graph.G.nodes[1]["type"] == "Constant"
        assert graph.G.nodes[1]["constant"] is False
        assert graph.G.nodes[8]["type"] == "Constant"
        assert graph.G.nodes[8]["constant"] is True
        literal_cell = book.latest_ref(PROVEN_LITERAL, ("lookup", 1))
        assert record_cell in {
            source for source, _stage in book.edges_into(literal_cell)
        }
        negation_cell = book.latest_ref(PROVEN_LITERAL, ("lookup", 8))
        assert literal_cell in {
            source for source, _stage in book.edges_into(negation_cell)
        }
    finally:
        end_identity_book(token)


def test_tensor_normalization_reads_rank_from_linked_dynamic_span():
    """An empty ABI shape plus rank one is a span, not a scalar."""

    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph.update({
        "function_name": "_tensor",
        "linked_value_abi": {
            0: {
                "storage": "span", "shape": (), "rank": 1,
                "dtype": "int64",
            },
        },
    })
    graph.G.add_node(
        0, type="Input", op="input", value_id=0,
        attributes={"binding_name": "values"},
        parents=[], children=[(1, "arg:0")],
    )
    graph.G.add_node(
        1, type="Call", op="Call", value_id=1,
        attributes={"tensor_operation": "tensor"},
        parents=[(0, "arg:0")], children=[],
    )
    graph.G.add_edge(0, 1)

    _book, token = begin_identity_book()
    try:
        assert _tensor_descriptor(graph, 0) == {
            "shape": (), "dtype": "int64", "rank": 1,
            "metadata_state": "dynamic",
        }
        assert _tensor_descriptor(graph, 1) == {
            "shape": (), "dtype": "int64", "rank": 1,
            "metadata_state": "dynamic",
        }
    finally:
        end_identity_book(token)


def test_tensor_of_concorded_list_is_dynamic_vector_before_abi_publication():
    """The source class supplies rank; the authored cast supplies dtype."""

    graph = ProcessGraph(materialize_memory=False)
    scope = ("fixture", "tensor_list")
    graph.G.graph.update({
        "function_name": "_tensor", "source_numeric_scope": scope,
    })
    graph.G.add_node(
        7, type="Call", op="Call", value_id=7, parents=[],
        attributes={"static_python_reference": "builtins.list"},
    )
    graph.G.add_node(
        8, type="Call", op="tensor", value_id=8,
        parents=[(7, "arg:0")],
        attributes={"tensor_operation": "tensor"},
    )
    graph.G.add_node(
        9, type="Constant", op="const", value_id=9,
        constant="int64", attributes={"value": "int64"}, parents=[],
    )
    graph.G.add_node(
        10, type="Call", op="to_dtype", value_id=10,
        parents=[(8, "receiver"), (9, "arg:0")],
        attributes={"tensor_operation": "to_dtype"},
    )
    graph.G.add_edges_from(((7, 8), (8, 10), (9, 10)))

    book, token = begin_identity_book()
    try:
        book.post(
            SOURCE_VALUE_CLASS, (scope, 7),
            ("builtins.list", 1, "resolved builtin aggregate result"),
            stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()),
            mode=Mode.CONCORD,
        )
        assert _tensor_descriptor_rule(graph, 8) == {
            "shape": (), "dtype": "unknown", "rank": 1,
            "metadata_state": "dynamic",
        }
        assert _tensor_descriptor_rule(graph, 10) == {
            "shape": (), "dtype": "int64", "rank": 1,
            "metadata_state": "dynamic",
        }
    finally:
        end_identity_book(token)


def test_dynamic_integer_tensor_index_preserves_base_dtype_and_index_rank():
    """``values[order]`` is a dynamic gather, not an untyped subscript."""

    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph["function_name"] = "sort_values"
    dynamic_int_vector = {
        "shape": (), "dtype": "int64", "rank": 1,
        "metadata_state": "dynamic",
    }
    graph.G.add_node(
        0, type="Input", op="input", value_id=0,
        tensor=dict(dynamic_int_vector), parents=[], children=[(2, "base")],
    )
    graph.G.add_node(
        1, type="Input", op="input", value_id=1,
        tensor=dict(dynamic_int_vector), parents=[], children=[(2, "index")],
    )
    graph.G.add_node(
        2, type="Indexed", op="Indexed", value_id=2,
        parents=[(0, "base"), (1, "index")], children=[],
    )
    graph.G.add_edges_from(((0, 2), (1, 2)))

    _book, token = begin_identity_book()
    try:
        assert _tensor_descriptor_rule(graph, 2) == dynamic_int_vector
    finally:
        end_identity_book(token)


def test_tensor_phi_records_only_an_agreed_union_descriptor():
    """A homogeneous tensor union is exact; a mixed union stays unresolved."""

    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph["function_name"] = "conditional_tensor"
    dynamic_int_vector = {
        "shape": (), "dtype": "int64", "rank": 1,
        "metadata_state": "dynamic",
    }
    graph.G.add_node(
        0, type="Input", op="input", value_id=0,
        tensor={"shape": (), "dtype": "bool", "rank": 0}, parents=[],
    )
    graph.G.add_node(
        1, type="Input", op="input", value_id=1,
        tensor=dict(dynamic_int_vector), parents=[],
    )
    graph.G.add_node(
        2, type="Input", op="input", value_id=2,
        tensor=dict(dynamic_int_vector), parents=[],
    )
    graph.G.add_node(
        3, type="Phi", op="phi", value_id=3,
        parents=[(0, "test"), (1, "body"), (2, "orelse")], children=[],
    )
    graph.G.add_edges_from(((0, 3), (1, 3), (2, 3)))

    _book, token = begin_identity_book()
    try:
        assert _tensor_descriptor_rule(graph, 3) == dynamic_int_vector
        graph.G.nodes[2]["tensor"] = {
            **dynamic_int_vector, "dtype": "float64",
        }
        assert _tensor_descriptor_rule(graph, 3) is None
    finally:
        end_identity_book(token)


def test_terminal_obligation_closes_from_settled_tensor_phi_descriptor():
    """A stale scalar question yields to the phi's later shape receipt."""

    graph = ProcessGraph(materialize_memory=False)
    dynamic_int_vector = {
        "shape": (), "dtype": "int64", "rank": 1,
        "metadata_state": "dynamic",
    }
    graph.G.graph.update({
        "function_name": "lz_encode", "canonical_value_ids": True,
    })
    graph.G.add_node(
        0, type="Input", op="input", value_id=0,
        tensor={"shape": (), "dtype": "bool", "rank": 0}, parents=[],
    )
    for node_id in (1, 2):
        graph.G.add_node(
            node_id, type="Input", op="input", value_id=node_id,
            tensor=dict(dynamic_int_vector), parents=[],
        )
    graph.G.add_node(
        3, type="Phi", op="phi", value_id=3,
        parents=[(0, "test"), (1, "body"), (2, "orelse")],
    )
    graph.G.add_edges_from(((0, 3), (1, 3), (2, 3)))

    book, token = begin_identity_book()
    try:
        scope = book.mint_scope("tensor_phi_obligation", READ_SCOPE_FORK)
        graph.G.graph.update({
            "ingestion_value_scope": scope,
            "lexical_read_scope": scope,
            "operand_position_scope": scope,
            "source_numeric_scope": scope,
        })
        ingestion_cells = tuple(
            book.post(
                INGESTION_VALUE, (scope, node_id),
                NodeFact(data.get("type"), data.get("op"), data.get("label")),
                stage=READ_SCOPE_FORK,
                provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD,
            )
            for node_id, data in graph.G.nodes(data=True)
        )
        activity_scope = book.mint_scope(
            "tensor_phi_obligation_activity", READ_SCOPE_FORK,
        )
        book.post(
            SOURCE_ACTIVITY_SCOPE, (scope,), activity_scope,
            stage=READ_SCOPE_FORK,
            provenance=Derived(ingestion_cells), mode=Mode.CONCORD,
        )
        _post_region_value_obligation(graph, 3)
        assert outstanding_region_value_obligations(graph) == (3,)
        _refuse_unsettled_host_scalars(graph, {})
        assert outstanding_region_value_obligations(graph) == ()
    finally:
        end_identity_book(token)


def test_dynamic_zeros_shape_tuple_states_rank_before_explicit_cast():
    """Runtime extents retain authored rank without guessing default dtype."""

    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph["function_name"] = "_zeros"
    graph.G.add_node(
        0, type="Input", op="input", value_id=0,
        tensor={"shape": (), "dtype": "int64", "rank": 0}, parents=[],
    )
    shape_expression = ast.Tuple(
        elts=[ast.Name(id="count", ctx=ast.Load())], ctx=ast.Load(),
    )
    graph.G.add_node(
        1, type="Tuple", op=None, value_id=1, expr_obj=shape_expression,
        parents=[(0, "elts")], children=[(2, "arg:0")],
    )
    graph.G.add_node(
        2, type="Call", op="zeros", value_id=2,
        parents=[(1, "arg:0")], children=[],
        attributes={"tensor_operation": "zeros"},
    )
    graph.G.add_edges_from(((0, 1), (1, 2)))

    _book, token = begin_identity_book()
    try:
        assert _tensor_descriptor_rule(graph, 2) == {
            "shape": (), "dtype": "unknown", "rank": 1,
            "metadata_state": "dynamic",
        }
    finally:
        end_identity_book(token)


def test_unknown_dtype_cat_and_phi_preserve_rank_for_explicit_cast():
    """Structural union evidence remains unknown until authored to_dtype."""

    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph["function_name"] = "compact"
    unknown_vector = {
        "shape": (), "dtype": "unknown", "rank": 1,
        "metadata_state": "dynamic",
    }
    for node_id in (0, 1):
        graph.G.add_node(
            node_id, type="Input", op="input", value_id=node_id,
            tensor=dict(unknown_vector), parents=[],
        )
    graph.G.add_node(
        2, type="Tuple", op=None, value_id=2,
        expr_obj=ast.Tuple(elts=[ast.Name(id="a"), ast.Name(id="b")]),
        parents=[(0, "elts"), (1, "elts")],
    )
    graph.G.add_node(
        3, type="Call", op="cat", value_id=3,
        parents=[(2, "arg:0")], attributes={"tensor_operation": "cat"},
    )
    graph.G.add_node(
        4, type="Input", op="input", value_id=4,
        tensor={"shape": (), "dtype": "bool", "rank": 0}, parents=[],
    )
    graph.G.add_node(
        5, type="Phi", op="phi", value_id=5,
        parents=[(4, "test"), (3, "body"), (0, "orelse")],
    )
    graph.G.add_node(
        6, type="Constant", op="const", value_id=6,
        constant="int64", attributes={"value": "int64"}, parents=[],
    )
    graph.G.add_node(
        7, type="Call", op="to_dtype", value_id=7,
        parents=[(5, "operand"), (6, "arg:0")],
        attributes={"tensor_operation": "to_dtype"},
    )
    graph.G.add_edges_from((
        (0, 2), (1, 2), (2, 3), (4, 5), (3, 5), (0, 5), (5, 7), (6, 7),
    ))

    _book, token = begin_identity_book()
    try:
        assert _tensor_descriptor_rule(graph, 3) == unknown_vector
        assert _tensor_descriptor_rule(graph, 5) == unknown_vector
        assert _tensor_descriptor_rule(graph, 7) == {
            "shape": (), "dtype": "int64", "rank": 1,
            "metadata_state": "dynamic",
        }
    finally:
        end_identity_book(token)


def test_basic_multi_axis_index_preserves_dtype_and_remaining_rank():
    """The two authored edges of ``pairs[:, 0]`` describe one vector."""

    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph["function_name"] = "encode_octets"
    graph.G.add_node(
        0, type="Input", op="input", value_id=0,
        tensor={
            "shape": (), "dtype": "int64", "rank": 2,
            "metadata_state": "dynamic",
        },
        parents=[],
    )
    graph.G.add_node(
        1, type="Constant", op="const", value_id=1,
        constant=slice(None), attributes={"value": slice(None)}, parents=[],
    )
    graph.G.add_node(
        2, type="Constant", op="const", value_id=2,
        constant=0, attributes={"value": 0}, parents=[],
    )
    graph.G.add_node(
        3, type="Indexed", op="Indexed", value_id=3,
        parents=[(0, "base"), (1, "index"), (2, "index")],
    )
    graph.G.add_edges_from(((0, 3), (1, 3), (2, 3)))

    _book, token = begin_identity_book()
    try:
        assert _tensor_descriptor_rule(graph, 3) == {
            "shape": (), "dtype": "int64", "rank": 1,
            "metadata_state": "dynamic",
        }
    finally:
        end_identity_book(token)


def test_singleton_value_type_abi_supplies_dynamic_region_descriptor():
    """A copied bytes value is typed by its exact concordance ABI row."""

    graph = ProcessGraph(materialize_memory=False)
    scope = ("bytes_copy", 0)
    graph.G.graph.update({
        "function_name": "encode_octets",
        "lexical_read_scope": scope,
    })
    graph.G.add_node(65, type="Call", op="Call", value_id=65, parents=[])

    book, token = begin_identity_book()
    try:
        canonical = book.post(
            CANONICAL_VALUE, (scope, 65), ("bytes(header)",),
            stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()),
            mode=Mode.CONCORD,
        )
        fact = ValueTypeABIFact(("builtins.bytes",), (), (
            ValueTypeABIAlternative(
                "builtins.bytes", "span", "int64", (-1,), 1,
                (("primitive_class_abi", "bytes"),), (canonical,),
            ),
        ))
        book.post(
            VALUE_TYPE_ABI, canonical.row, fact, stage=INGESTION,
            provenance=Derived((canonical,)), mode=Mode.CONCORD,
        )
        assert _concorded_value_type_descriptor(graph, 65) == {
            "shape": (), "dtype": "int64", "rank": 1,
            "metadata_state": "dynamic",
        }
    finally:
        end_identity_book(token)


def test_topk_indices_projection_and_ensure_tensor_keep_dynamic_vector():
    """Named top-k indices are int64; ensure_tensor follows its argument."""

    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph["function_name"] = "lz_encode"
    dynamic_float_vector = {
        "shape": (), "dtype": "float64", "rank": 1,
        "metadata_state": "dynamic",
    }
    dynamic_int_vector = {
        "shape": (), "dtype": "int64", "rank": 1,
        "metadata_state": "dynamic",
    }
    graph.G.add_node(
        0, type="Input", op="input", value_id=0,
        tensor=dict(dynamic_float_vector), parents=[],
    )
    graph.G.add_node(
        1, type="Input", op="input", value_id=1,
        tensor=dict(dynamic_int_vector), parents=[],
    )
    graph.G.add_node(
        2, type="topk", op="topk", value_id=2,
        parents=[(0, "arg:0"), (5, "kw:k")], attributes={
            "tensor_candidate": "topk", "dim": 0,
        },
    )
    graph.G.add_node(
        5, type="Constant", op="const", value_id=5,
        parents=[], attributes={},
    )
    graph.G.add_node(
        3, type="GetAttr", op="GetAttr", value_id=3,
        parents=[(2, "value")], attributes={"attribute": "indices"},
    )
    graph.G.add_node(
        4, type="ensure_tensor", op="ensure_tensor", value_id=4,
        parents=[(1, "operand"), (3, "arg:0")],
        attributes={"tensor_candidate": "ensure_tensor"},
    )
    graph.G.add_edges_from(((0, 2), (5, 2), (2, 3), (1, 4), (3, 4)))

    _book, token = begin_identity_book()
    try:
        assert _tensor_descriptor_rule(graph, 3) == dynamic_int_vector
        assert _tensor_descriptor_rule(graph, 4) == dynamic_int_vector
    finally:
        end_identity_book(token)


def test_runtime_bounded_slice_preserves_dynamic_span_dtype_and_rank():
    """A runtime slice changes extent, not rank or element type."""

    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph["function_name"] = "lz_encode"
    dynamic_int_vector = {
        "shape": (), "dtype": "int64", "rank": 1,
        "metadata_state": "dynamic",
    }
    graph.G.add_node(
        0, type="Input", op="input", value_id=0,
        tensor=dict(dynamic_int_vector), parents=[],
    )
    graph.G.add_node(
        1, type="Input", op="input", value_id=1,
        tensor={"shape": (), "dtype": "int64", "rank": 0}, parents=[],
    )
    graph.G.add_node(
        2, type="Slice", op="Slice", value_id=2,
        parents=[(1, "upper")],
    )
    graph.G.add_node(
        3, type="Indexed", op="Indexed", value_id=3,
        parents=[(0, "base"), (2, "index")],
    )
    graph.G.add_edges_from(((1, 2), (0, 3), (2, 3)))

    _book, token = begin_identity_book()
    try:
        assert _tensor_descriptor_rule(graph, 3) == dynamic_int_vector
    finally:
        end_identity_book(token)


def test_literal_slice_preserves_dynamic_span_dtype_and_rank():
    """A folded literal slice has the same contract as an AST Slice node."""

    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph["function_name"] = "lz_encode"
    dynamic_int_vector = {
        "shape": (), "dtype": "int64", "rank": 1,
        "metadata_state": "dynamic",
    }
    graph.G.add_node(
        0, type="Input", op="input", value_id=0,
        tensor=dict(dynamic_int_vector), parents=[],
    )
    graph.G.add_node(
        1, type="Constant", op="const", value_id=1,
        constant=slice(None, -1, None),
        attributes={"value": slice(None, -1, None)}, parents=[],
    )
    graph.G.add_node(
        2, type="Indexed", op="Indexed", value_id=2,
        parents=[(0, "base"), (1, "index")],
    )
    graph.G.add_edges_from(((0, 2), (1, 2)))

    _book, token = begin_identity_book()
    try:
        assert _tensor_descriptor_rule(graph, 2) == dynamic_int_vector
    finally:
        end_identity_book(token)


def test_dynamic_reshape_inferred_extent_stays_runtime_sized():
    """The empty extent tuple of a runtime span is not a count of one."""

    graph = ProcessGraph(materialize_memory=False)
    graph.G.graph["function_name"] = "lz_encode"
    graph.G.add_node(
        0, type="Input", op="input", value_id=0,
        tensor={
            "shape": (), "dtype": "int64", "rank": 2,
            "metadata_state": "dynamic",
        }, parents=[],
    )
    graph.G.add_node(
        1, type="Constant", op="const", value_id=1, constant=-1,
        attributes={"value": -1}, parents=[],
    )
    graph.G.add_node(
        2, type="reshape", op="reshape", value_id=2,
        expr_obj=ast.parse("value.reshape(-1)", mode="eval").body,
        attributes={"tensor_candidate": "reshape", "source_type": "Call"},
        parents=[(0, "operand"), (1, "arg:0")],
    )
    graph.G.add_node(
        3, type="ndims", op="ndims", value_id=3,
        expr_obj=ast.parse("value.ndims()", mode="eval").body,
        attributes={"tensor_candidate": "ndims", "source_type": "Call"},
        parents=[(0, "operand")],
    )
    graph.G.add_edges_from(((0, 2), (1, 2), (0, 3)))
    graph.roots = [2, 3]

    _book, token = begin_identity_book()
    try:
        scope = _book.mint_scope(
            "dynamic_reshape_fixture", READ_SCOPE_FORK,
        )
        graph.G.graph.update(
            ingestion_value_scope=scope,
            lexical_read_scope=scope,
            operand_position_scope=scope,
        )
        ingestion_cells = []
        for node_id, data in graph.G.nodes(data=True):
            ingestion_cells.append(_book.post(
                INGESTION_VALUE, (scope, node_id),
                NodeFact(data.get("type"), data.get("op"), data.get("label")),
                stage=READ_SCOPE_FORK,
                provenance=Novel(INGEST_SOURCE, ()),
                mode=Mode.CONCORD,
            ))
        activity_scope = _book.mint_scope(
            "dynamic_reshape_activity", READ_SCOPE_FORK,
        )
        _book.post(
            SOURCE_ACTIVITY_SCOPE, (scope,), activity_scope,
            stage=READ_SCOPE_FORK,
            provenance=Derived(tuple(ingestion_cells)),
            mode=Mode.CONCORD,
        )
        _fold_callsite_structural_values(graph)
        assert graph.G.nodes[2]["tensor"] == {
            "shape": (), "dtype": "int64", "rank": 1,
            "metadata_state": "dynamic",
        }
        assert (
            graph.G.nodes[3].get("constant")
            if graph.G.nodes[3].get("constant") is not None
            else (graph.G.nodes[3].get("attributes") or {}).get("value")
        ) == 2
    finally:
        end_identity_book(token)


def test_authored_tensor_result_class_opens_descriptor_specialization_gate():
    """A sequence argument cannot suppress an exact tensor return class."""

    graph = ProcessGraph(materialize_memory=False)
    scope = ("fixture", "caller")
    graph.G.graph.update({
        "function_name": "caller", "source_numeric_scope": scope,
    })
    graph.G.add_node(5, type="Call", op="Call", value_id=5)

    book, token = begin_identity_book()
    try:
        declared = book.post(
            SOURCE_VALUE_CLASS, (scope, 5),
            ("AbstractTensor", 1, "authored call result declaration"),
            stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()),
            mode=Mode.CONCORD,
        )
        assert _concorded_tensor_result_class(graph, 5) == (
            True, declared,
        )
    finally:
        end_identity_book(token)


def test_callsite_descriptor_receipt_records_dynamic_rank_and_state():
    scalar = {"shape": (), "dtype": "int64", "rank": 0}
    vector = {
        "shape": (), "dtype": "int64", "rank": 1,
        "metadata_state": "dynamic",
    }

    assert _callsite_descriptor_receipt(scalar) != (
        _callsite_descriptor_receipt(vector)
    )


def test_structural_descriptor_write_cannot_collapse_dynamic_span_to_scalar():
    dynamic_vector = {
        "shape": (), "dtype": "int64", "rank": 1,
        "metadata_state": "dynamic",
    }
    scalar = {"shape": (), "dtype": "int64", "rank": 0}

    assert _descriptor_loses_runtime_rank(dynamic_vector, scalar)
    assert not _descriptor_loses_runtime_rank(scalar, dynamic_vector)


def test_dynamic_tensor_descriptor_projects_to_span_program_abi():
    descriptor = {
        "shape": (), "dtype": "int64", "rank": 1,
        "metadata_state": "dynamic",
    }

    fact = _program_abi_tensor_fact(descriptor, "AbstractTensor")
    assert (fact.storage, fact.dtype, fact.shape, fact.rank) == (
        "span", "int64", None, 1,
    )
