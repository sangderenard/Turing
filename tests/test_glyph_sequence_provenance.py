from __future__ import annotations

import ast
import contextlib
import io
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.common.tensors.topological_reducer import (
    node_identity_cell,
    reduce_abstract_tensor_topology,
)
from src.compiler.concordance_declarations import (
    ANNOTATION_DECLARATION,
    CALL_BINDING,
    CLASS_FIELD_DECLARATION,
    STRING_GLYPH_VOCABULARY_MINT,
    CallResolution,
    IDENTITY_TRANSITION,
    INGESTION,
    INGEST_SOURCE,
    NAME_BINDING,
    PARAMETER_ANNOTATION,
    SOURCE_AGGREGATE_LAYOUT,
    SOURCE_ENUM_GLYPH_TABLE,
    SOURCE_GLYPH_SEQUENCE,
    SOURCE_OPTIONAL_GLYPH_CARRY,
    SOURCE_PAGE_FACT_CLASS,
    SOURCE_VALUE_CLASS,
    SourceAggregateLayout,
    SourceGlyphComponent,
    SourceGlyphSequence,
    SourceGlyphUnionAlternative,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.control_source import (
    CallBlock,
    ControlSequenceMutation,
    ControlProgram,
    SequenceBlock,
    SequenceMutationBlock,
)
from src.compiler.ast_control_normalization import (
    _normalize_top_level_guard_returns,
)
from src.compiler.fortran_c_shell import (
    _verify_optional_glyph_work,
    lower_ast_source_to_ssa,
)
from src.compiler.glsl_deployment_strategy import (
    _post_source_optional_glyph_carries,
)
from src.compiler.glyph_sequence_format import (
    decimal_integer_glyph_tokens,
    prefixed_lower_hex_integer_glyph_tokens,
    quoted_string_glyph_tokens,
    static_literal_repr_glyph_tokens,
)
from src.compiler.identity_concordance import (
    CorrelationTable,
    Derived,
    Mode,
    Novel,
    PROVEN_SHAPE_PAGE,
    SHAPE_STATE_PAGE,
    begin_identity_book,
    current_identity_book,
    end_identity_book,
)
from src.compiler.resident_sequence_materialization import (
    _source_glyph_sequence_materializations,
    _string_token_glyph_materializations,
)
from src.compiler.precompile_to_ssa import lower_control_program_to_ssa
from src.compiler.string_table import string_token
from src.transmogrifier.graph.graph_express2 import ProcessGraph


def _source_glyph_cell(book, canonical_cell):
    return next(
        source for source, _stage in book.edges_into(canonical_cell)
        if source.page is SOURCE_GLYPH_SEQUENCE
        and source.row != canonical_cell.row
    )


@pytest.mark.parametrize(
    ("value", "spelling"),
    (
        ("plain", "'plain'"),
        ("has'quote", '"has\'quote"'),
        ('both\'"quotes', "'both\\\'\"quotes'"),
        ("line\nfeed", "'line\\nfeed'"),
        ("nul\0", "'nul\\x00'"),
    ),
)
def test_quoted_string_glyphs_are_authored_and_lossless(value, spelling):
    assert quoted_string_glyph_tokens(value) == [
        string_token(character) for character in spelling
    ]


@pytest.mark.parametrize(
    ("value", "spelling"),
    ((None, "None"), (True, "True"), (False, "False"), (-17, "-17")),
)
def test_static_literal_repr_uses_the_authored_glyph_faculty(value, spelling):
    assert static_literal_repr_glyph_tokens(value) == [
        string_token(character) for character in spelling
    ]


def test_static_literal_repr_refuses_undeclared_host_formatting():
    assert static_literal_repr_glyph_tokens(1.25) is None


def test_value_union_glyph_recipe_retains_exact_per_arm_sequences():
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        scope = book.mint_scope("value-union-glyph")
        graph.G.graph["source_glyph_sequence_scope"] = scope
        for value_id in (10, 20, 21, 30, 40):
            graph.G.add_node(value_id, value_id=value_id, attributes={})
        # The test is the ingestion of these two values: each class fact is
        # a root, as a source construct's span is.
        for value_id, classes in ((10, ("builtins.tuple",)), (30, ("builtins.NoneType",))):
            book.post(
                SOURCE_VALUE_CLASS, (scope, value_id), classes,
                stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()),
                mode=Mode.CONCORD,
            )
        tuple_source = book.latest_ref(SOURCE_VALUE_CLASS, (scope, 10))
        none_source = book.latest_ref(SOURCE_VALUE_CLASS, (scope, 30))
        fact = SourceGlyphSequence((SourceGlyphComponent(
            kind="formatted", component_value_id=30,
            operand_value_id=10, glyph_tokens=(), conversion=ord("r"),
            format_spec_value_id=None, projection_kind="value_union",
            record_identity=None,
            union_alternatives=(
                SourceGlyphUnionAlternative(
                    "builtins.NoneType", 20, (none_source,),
                ),
                SourceGlyphUnionAlternative(
                    "builtins.tuple", 21, (tuple_source,),
                ),
            ),
        ),))
        # The recipe stands on the two class facts it names.
        book.post(
            SOURCE_GLYPH_SEQUENCE, (scope, 40), fact, stage=INGESTION,
            provenance=Derived((tuple_source, none_source)),
            mode=Mode.CONCORD,
        )
        declarations, _initializations, mutations = (
            _source_glyph_sequence_materializations(graph.G)
        )
        assert (30, "duplicates", 1, True) in declarations
        dispatch = next(
            mutation for mutation in mutations
            if mutation.operator == "glyph_union_select"
        )
        assert dispatch.sequence_value_id == 30
        assert dispatch.argument_value_ids == (10, 20, 21)
        assert dispatch.argument_kind == "value_union_glyph_projection"
    finally:
        end_identity_book(token)


def test_joined_string_publishes_canonical_glyph_sequence_recipe():
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        with contextlib.redirect_stdout(io.StringIO()):
            graph.build_from_ast(ast.parse(
                "def render(value):\n"
                "    return f'a{value!r}b'\n"
            ))
        reduce_abstract_tensor_topology(graph)

        function = graph.function_table.entry("render").graph
        joined_id, joined = next(
            (node_id, data)
            for node_id, data in function.G.nodes(data=True)
            if isinstance(data.get("expr_obj"), ast.JoinedStr)
        )
        row = joined["attributes"]["source_glyph_sequence_row"]
        assert row[1] == joined_id
        fact = book.page(SOURCE_GLYPH_SEQUENCE).latest(row)
        assert tuple(item.kind for item in fact.components) == (
            "literal", "formatted", "literal",
        )
        assert fact.components[0].glyph_tokens == (string_token("a"),)
        assert fact.components[1].conversion == ord("r")
        assert fact.components[1].operand_value_id == next(
            parent
            for parent, role in function.G.nodes[
                fact.components[1].component_value_id
            ]["parents"]
            if role == "value"
        )
        formatted = function.G.nodes[
            fact.components[1].component_value_id
        ]["attributes"]
        assert formatted["producer_kind"] == "formatted_glyph_sequence"
        assert formatted["source_glyph_component"] == (row, 1)
        assert fact.components[2].glyph_tokens == (string_token("b"),)
        assert joined["attributes"]["sequence_element_dtypes"] == (
            "int64",
        )
        target = book.latest_ref(SOURCE_GLYPH_SEQUENCE, row)
        assert target is not None
        assert book.edges_into(target)
    finally:
        end_identity_book(token)


def test_repeated_same_operand_formatted_occurrences_keep_canonical_slots():
    book, token = begin_identity_book()
    try:
        source = (
            "def render(value):\n"
            "    return f'{value!r}:{value!r}'\n"
        )
        facts = []
        rows = []
        for _replay in range(2):
            graph = ProcessGraph(materialize_memory=False)
            with contextlib.redirect_stdout(io.StringIO()):
                graph.build_from_ast(ast.parse(source))
            reduce_abstract_tensor_topology(graph)
            function = graph.function_table.entry("render").graph
            joined = next(
                data for _node_id, data in function.G.nodes(data=True)
                if isinstance(data.get("expr_obj"), ast.JoinedStr)
            )
            row = joined["attributes"]["source_glyph_sequence_row"]
            rows.append(row)
            assert row[0] == function.G.graph["source_glyph_sequence_scope"]
            facts.append(book.page(SOURCE_GLYPH_SEQUENCE).latest(row))

        assert facts[0] == facts[1]
        assert rows[0] != rows[1]
        formatted = tuple(
            component for component in facts[0].components
            if component.kind == "formatted"
        )
        assert len(formatted) == 2
        assert formatted[0].component_value_id != formatted[1].component_value_id
        assert formatted[0].operand_value_id == formatted[1].operand_value_id
    finally:
        end_identity_book(token)


def test_mapping_items_nested_key_keeps_string_class_from_store_layout():
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        with contextlib.redirect_stdout(io.StringIO()):
            graph.build_from_ast(ast.parse(
                "class Table:\n"
                "    kind: str\n"
                "\n"
                "def render(table: Table):\n"
                "    live: dict[tuple[str, int], int] = {}\n"
                "    live[table.kind, 3] = 4\n"
                "    for (kind, row_id), descriptor in live.items():\n"
                "        text = f'{kind}'\n"
                "    return 0\n"
            ))
        reduce_abstract_tensor_topology(graph)

        function = graph.function_table.entry("render").graph.G
        scope = function.graph["source_aggregate_layout_scope"]
        store_id = next(
            int(node_id)
            for node_id, data in function.nodes(data=True)
            if data.get("type") == "IndexedStore"
        )
        layout = book.page(SOURCE_AGGREGATE_LAYOUT).latest(
            (scope, store_id)
        )
        assert isinstance(layout, SourceAggregateLayout)
        assert layout.aggregate_kind == "dict"
        assert len(layout.members) == 3
        assert all(member.source_cells for member in layout.members)

        kind_id = function.graph["identity_table"]["kind"][-1]
        kind_rows = tuple(
            row for row in book.page(SOURCE_VALUE_CLASS).rows()
            if int(row[1]) == int(kind_id)
        )
        assert len(kind_rows) == 1
        kind_row = kind_rows[0]
        kind_class = book.page(SOURCE_VALUE_CLASS).latest(kind_row)
        assert kind_class[:2] == ("builtins.str", 1)
        kind_cell = book.latest_ref(SOURCE_VALUE_CLASS, kind_row)
        assert kind_cell is not None
        assert book.edges_into(kind_cell)

        joined = next(
            data for _node_id, data in function.nodes(data=True)
            if isinstance(data.get("expr_obj"), ast.JoinedStr)
        )
        glyphs = book.page(SOURCE_GLYPH_SEQUENCE).latest(
            joined["attributes"]["source_glyph_sequence_row"]
        )
        formatted = next(
            component for component in glyphs.components
            if component.kind == "formatted"
        )
        assert formatted.operand_value_id == kind_id
        assert formatted.projection_kind == "string_token"
    finally:
        end_identity_book(token)


def test_authored_decimal_formatter_emits_reversible_glyph_tokens():
    for value in (0, 7, 10, -42, 2**63 - 1, -(2**63)):
        assert decimal_integer_glyph_tokens(value) == [
            string_token(character) for character in str(value)
        ]


def test_authored_prefixed_hex_formatter_emits_reversible_glyph_tokens():
    for value in (0, 7, 16, -42, 2**63 - 1, -(2**63)):
        assert prefixed_lower_hex_integer_glyph_tokens(value) == [
            string_token(character) for character in format(value, "#x")
        ]


def test_constant_repr_component_stays_folded_without_runtime_call():
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        with contextlib.redirect_stdout(io.StringIO()):
            graph.build_from_ast(ast.parse(
                "def render():\n"
                "    return f'{42!r}:{\"café\"!r}'\n"
            ))
        reduce_abstract_tensor_topology(graph)
        function = graph.function_table.entry("render").graph
        joined = next(
            data for _node_id, data in function.G.nodes(data=True)
            if isinstance(data.get("expr_obj"), ast.JoinedStr)
        )
        fact = book.page(SOURCE_GLYPH_SEQUENCE).latest(
            joined["attributes"]["source_glyph_sequence_row"]
        )
        assert tuple(component.kind for component in fact.components) == (
            "literal", "literal", "literal",
        )
        assert tuple(
            component.glyph_tokens for component in fact.components
        ) == tuple(
            tuple(string_token(character) for character in spelling)
            for spelling in ("42", ":", "'café'")
        )
        assert not any(
            (data.get("attributes") or {}).get("extraction_action")
            == "use_native"
            for _node_id, data in function.G.nodes(data=True)
        )
    finally:
        end_identity_book(token)


def test_lexical_reads_cite_the_exact_binding_version_at_each_occurrence():
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        with contextlib.redirect_stdout(io.StringIO()):
            graph.build_from_ast(ast.parse(
                "def select():\n"
                "    x = 1\n"
                "    first = x\n"
                "    x = 2\n"
                "    second = x\n"
                "    return first, second\n"
            ))
        reduce_abstract_tensor_topology(graph)
        page = book.page("lexical_read_binding")
        reads = [
            book.latest_ref(page, row)
            for row in page.rows()
            if len(row) == 4 and row[1] == "occurrence"
            and page.latest(row) == "x"
        ]
        assert len(reads) == 2
        binding_sources = {
            source.row[2]
            for read in reads
            for source, _stage in book.edges_into(read)
            if source.page is NAME_BINDING and source.row[1] == "x"
        }
        assert binding_sources == {0, 1}
    finally:
        end_identity_book(token)


def test_literal_glyph_row_is_canonical_derivation_of_ingestion_fact():
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        with contextlib.redirect_stdout(io.StringIO()):
            graph.build_from_ast(ast.parse(
                "def render(flag: bool):\n"
                "    detail = None\n"
                "    if flag:\n"
                "        detail = 'fixed'\n"
                "    return detail\n"
            ))
        reduce_abstract_tensor_topology(graph)
        function = graph.function_table.entry("render").graph
        literal = next(
            data for _node_id, data in function.G.nodes(data=True)
            if isinstance(data.get("expr_obj"), ast.Constant)
            and data["expr_obj"].value == "fixed"
        )
        assert "source_glyph_sequence_row" not in literal["attributes"]
        row = literal["attributes"]["source_literal_glyph_sequence_row"]
        canonical_cell = book.latest_ref(SOURCE_GLYPH_SEQUENCE, row)
        source_cell = _source_glyph_cell(book, canonical_cell)
        assert source_cell.row[1] != row[1]
        assert book.edges_into(source_cell)
    finally:
        end_identity_book(token)


def test_integer_formatted_value_has_authored_call_and_unit_dependency():
    policy = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "records": {},
        "bindings": [],
        "values": [{
            "function": "render",
            "parameter": "value",
            "storage": "scalar",
            "dtype": "int64",
            "rank": 0,
            "python_type": "builtins.int",
        }],
    })
    captured = {}

    def capture(graph):
        root = next(
            entry.graph for entry in graph.function_table
            if entry.name == "render"
        )
        formatter = next(
            entry for entry in graph.function_table
            if entry.qualified_name == (
                "src.compiler.glyph_sequence_format."
                "decimal_integer_glyph_tokens"
            )
        )
        call_id, call = next(
            (node_id, data)
            for node_id, data in root.G.nodes(data=True)
            if (data.get("attributes") or {}).get(
                "callee_resolution"
            ) == "source-glyph-formatter"
        )
        glyph_row = next(
            (data.get("attributes") or {})["source_glyph_sequence_row"]
            for _node_id, data in root.G.nodes(data=True)
            if (data.get("attributes") or {}).get(
                "source_glyph_sequence_row"
            ) is not None
        )
        book = current_identity_book()
        captured.update({
            "root": root,
            "formatter": formatter,
            "call_id": int(call_id),
            "call": call,
            "glyph_row": glyph_row,
            "glyph_fact": book.page(SOURCE_GLYPH_SEQUENCE).latest(
                glyph_row
            ),
            "glyph_cell": book.latest_ref(
                SOURCE_GLYPH_SEQUENCE, glyph_row,
            ),
            "formatter_result_cell": node_identity_cell(root, int(call_id)),
            "glyph_sources": {
                source for source, _stage in book.edges_into(
                    book.latest_ref(SOURCE_GLYPH_SEQUENCE, glyph_row)
                )
            },
            "binding": book.page(CALL_BINDING).latest((
                tuple(root.G.graph["lexical_read_scope"]), int(call_id),
            )),
        })

    plans = []
    with contextlib.redirect_stdout(io.StringIO()):
        module, _outputs, _exports = lower_ast_source_to_ssa(
            "def render(value: int):\n"
            "    return f'n={value!r}'\n",
            "render",
            name="glyph_formatter_topology",
            extraction_contract=policy,
            resolved_process_graph_sink=capture,
            compilation_unit_plan_sink=plans.append,
            stop_after_compilation_unit_plan=True,
            progress=lambda _message: None,
        )

    call = captured["call"]
    call_id = captured["call_id"]
    formatter = captured["formatter"]
    assert call.get("expr_obj") is None
    assert call["attributes"]["callee_ref"] == formatter.reference.address
    assert call["parents"] == [(0, "arg:0")]
    fact = captured["glyph_fact"]
    formatted = next(
        component for component in fact.components
        if component.kind == "formatted"
    )
    assert formatted.component_value_id != call_id
    assert formatted.formatted_sequence_value_id == call_id
    assert captured["formatter_result_cell"] in captured["glyph_sources"]
    assert captured["binding"].callee == formatter.reference.address
    assert captured["binding"].resolution is CallResolution.CALLSITE_SHELL

    plan = plans[0]
    reference_to_unit = {
        int(reference): int(unit)
        for reference, unit in plan["reference_to_unit"].items()
    }
    root_ref = next(
        reference for reference, unit in reference_to_unit.items()
        if "render" in plan["units"][unit]["qualified_names"]
    )
    assert reference_to_unit[formatter.reference.address] in plan["units"][
        reference_to_unit[root_ref]
    ]["dependency_units"]


def test_unannotated_formal_links_the_integer_formatter_once_its_callsite_class_settles():
    """``def f(x): return f"{x!r}"`` called once as ``f(a)`` with ``a: int``.

    The formal ``x`` has no annotation: its class row is published from the
    call actual by the call-edge settlement, which runs after ``f``'s recipe
    was authored.  The recipe is linked again once that fact exists: the
    formatted component names the authored decimal formatter call, the
    call derives from the recipe cell and the formal's class cell, the
    revised recipe derives from the call, and the call has its binding.
    """

    policy = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "records": {},
        "bindings": [],
        "values": [{
            "function": "render",
            "parameter": "a",
            "storage": "scalar",
            "dtype": "int64",
            "rank": 0,
            "python_type": "builtins.int",
        }],
    })
    captured = {}

    def capture(graph):
        callee = next(
            entry.graph for entry in graph.function_table
            if entry.name == "f"
        )
        formatter = next(
            entry for entry in graph.function_table
            if entry.qualified_name == (
                "src.compiler.glyph_sequence_format."
                "decimal_integer_glyph_tokens"
            )
        )
        book = current_identity_book()
        glyph_row = next(
            (data.get("attributes") or {})["source_glyph_sequence_row"]
            for _node_id, data in callee.G.nodes(data=True)
            if (data.get("attributes") or {}).get(
                "source_glyph_sequence_row"
            ) is not None
        )
        formal_id = next(
            int(node_id)
            for node_id, data in callee.G.nodes(data=True)
            if data.get("type") == "Input"
            and (data.get("attributes") or {}).get("binding_name") == "x"
        )
        glyph_cell = book.latest_ref(SOURCE_GLYPH_SEQUENCE, glyph_row)
        glyph_fact = book.page(SOURCE_GLYPH_SEQUENCE).latest(glyph_row)
        formatted = next(
            component for component in glyph_fact.components
            if component.kind == "formatted"
        )
        call_id = formatted.formatted_sequence_value_id
        call_cell = (
            None if call_id is None
            else node_identity_cell(callee, int(call_id))
        )
        formal_class_rows = {
            row: book.page(SOURCE_VALUE_CLASS).latest(row)
            for row in book.page(SOURCE_VALUE_CLASS).rows()
            if int(row[1]) == formal_id
            and row[0] == callee.G.graph["authored_function_identity"]
        }
        captured.update({
            "callee": callee,
            "formatter": formatter,
            "glyph_row": glyph_row,
            "glyph_fact": glyph_fact,
            "glyph_sources": {
                source for source, _stage in book.edges_into(glyph_cell)
            },
            "formal_id": formal_id,
            "formal_class_rows": formal_class_rows,
            "formal_class_cells": {
                row: book.latest_ref(SOURCE_VALUE_CLASS, row)
                for row in formal_class_rows
            },
            "call_cell": call_cell,
            "call_sources": set() if call_cell is None else {
                source for source, _stage in book.edges_into(call_cell)
            },
            "binding": None if call_id is None else book.page(
                CALL_BINDING
            ).latest((
                tuple(callee.G.graph["lexical_read_scope"]), int(call_id),
            )),
        })

    plans = []
    with contextlib.redirect_stdout(io.StringIO()):
        lower_ast_source_to_ssa(
            "def f(x):\n"
            "    return f'v={x!r}'\n"
            "\n"
            "def render(a: int):\n"
            "    return f(a)\n",
            "render",
            name="glyph_formatter_settled_formal",
            extraction_contract=policy,
            resolved_process_graph_sink=capture,
            compilation_unit_plan_sink=plans.append,
            stop_after_compilation_unit_plan=True,
            progress=lambda _message: None,
        )

    callee = captured["callee"]
    formatter = captured["formatter"]
    # The formal's class came from the call edge, not an annotation.
    (class_row, class_fact), = captured["formal_class_rows"].items()
    assert class_fact[0] == "builtins.int"
    assert str(class_fact[2]).startswith("call actual ")
    class_cell = captured["formal_class_cells"][class_row]
    assert class_cell is not None

    fact = captured["glyph_fact"]
    formatted = next(
        component for component in fact.components
        if component.kind == "formatted"
    )
    assert formatted.operand_value_id == captured["formal_id"]
    assert formatted.projection_kind == "scalar"
    call_id = formatted.formatted_sequence_value_id
    assert call_id is not None and call_id != formatted.component_value_id
    call = callee.G.nodes[int(call_id)]
    assert call["type"] == "Call"
    assert call["attributes"]["callee_ref"] == formatter.reference.address
    assert call["attributes"]["callee_resolution"] == "source-glyph-formatter"
    assert call["attributes"]["source_glyph_component"] == (
        captured["glyph_row"], fact.components.index(formatted),
    )
    assert call["parents"] == [(captured["formal_id"], "arg:0")]
    assert callee.G.nodes[formatted.component_value_id]["attributes"][
        "source_glyph_component"
    ] == (captured["glyph_row"], fact.components.index(formatted))

    # Provenance: the call stands on the recipe cell and the class cell it
    # was selected from; the revised recipe stands on the call.
    call_cell = captured["call_cell"]
    call_sources = captured["call_sources"]
    assert class_cell in call_sources
    assert any(
        source.page is SOURCE_GLYPH_SEQUENCE for source in call_sources
    )
    assert call_cell in captured["glyph_sources"]
    binding = captured["binding"]
    assert binding.callee == formatter.reference.address
    assert binding.resolution is CallResolution.CALLSITE_SHELL

    plan = plans[0]
    reference_to_unit = {
        int(reference): int(unit)
        for reference, unit in plan["reference_to_unit"].items()
    }
    callee_ref = next(
        reference for reference, unit in reference_to_unit.items()
        if "f" in plan["units"][unit]["qualified_names"]
    )
    assert reference_to_unit[formatter.reference.address] in plan["units"][
        reference_to_unit[callee_ref]
    ]["dependency_units"]


def test_decimal_digit_tuple_shape_keeps_canonical_provenance():
    """The pursued digit table's shape derives from its constant identity."""

    policy = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "records": {}, "bindings": [],
        "values": [{
            "function": "render", "parameter": "value",
            "storage": "scalar", "dtype": "int64", "rank": 0,
            "python_type": "builtins.int",
        }],
    })
    with contextlib.redirect_stdout(io.StringIO()):
        module, _outputs, _exports = lower_ast_source_to_ssa(
            "def render(value: int):\n"
            "    return f'value={value}'\n",
            "render", name="decimal_digit_shape_provenance",
            extraction_contract=policy, progress=lambda _message: None,
        )

    book = module.metadata["identity_book"]
    shape_page = book.page(PROVEN_SHAPE_PAGE)
    row = next(
        row for row in shape_page.rows()
        if "decimal_digit_glyph_token|shape" in str(row)
        and shape_page.latest(row)[1] == (10,)
    )
    shape_ref = book.latest_ref(PROVEN_SHAPE_PAGE, row)
    assert shape_ref is not None
    sources = book.edges_into(shape_ref)
    assert len(sources) == 1
    assert sources[0][0].page == SHAPE_STATE_PAGE
    assert not [
        finding for finding in CorrelationTable.build(module).findings(module)
        if finding.kind == "unsourced-fact"
        and finding.function == "proven_shape"
        and "decimal_digit_glyph_token|shape" in finding.detail
    ]


def test_integer_intrinsic_result_uses_concorded_typed_ssa_formatters():
    policy = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "records": {},
        "bindings": [],
        "values": [{
            "function": "render",
            "parameter": "value",
            "storage": "scalar",
            "dtype": "int64",
            "rank": 0,
            "python_type": "builtins.int",
        }],
    })
    captured = {}

    def capture(process):
        root = next(
            entry.graph for entry in process.function_table
            if entry.name == "render"
        )
        calls = tuple(
            (int(node_id), data)
            for node_id, data in root.G.nodes(data=True)
            if (data.get("attributes") or {}).get(
                "callee_resolution"
            ) == "source-glyph-formatter"
        )
        book = current_identity_book()
        class_scope = tuple(root.G.graph["authored_function_identity"])
        captured["names"] = tuple(
            str(data.get("label")) for _node_id, data in calls
        )
        captured["operand_classes"] = tuple(
            book.page(SOURCE_VALUE_CLASS).latest((
                class_scope,
                next(
                    int(parent)
                    for parent, role in data.get("parents") or ()
                    if str(role) == "arg:0"
                ),
            ))
            for _node_id, data in calls
        )

    with contextlib.redirect_stdout(io.StringIO()):
        module, _outputs, _exports = lower_ast_source_to_ssa(
            "def render(value: int):\n"
            "    return f'n={int(value)} h={int(value):#x}'\n",
            "render",
            name="typed_ssa_integer_repr",
            extraction_contract=policy,
            resolved_process_graph_sink=capture,
            progress=lambda _message: None,
        )

    assert captured["names"] == (
        "decimal_integer_glyph_tokens",
        "prefixed_lower_hex_integer_glyph_tokens",
    )
    assert all(
        fact is not None and fact[0] == "builtins.int"
        for fact in captured["operand_classes"]
    )
    # The linked decimal/hex helper chain exercises call-record reordering,
    # planned-region calls, and caller/callee hierarchy projections.  Every
    # one must retain its exact callsite, RegionSignature, or equivalence-
    # class source rather than landing as a raw compiler fact.
    assert not [
        finding for finding in CorrelationTable.build(module).findings(module)
        if finding.kind == "unsourced-fact"
        and finding.function in {
            "call_record",
            "formal_actual_occurrence_concordance",
            "hierarchy_value",
            "scalar_integer_width",
            "transformation_decision",
        }
    ]


def test_optional_authored_tuple_repr_carries_into_exact_finding_row_append():
    policy = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    )
    source = (
        "from dataclasses import dataclass\n"
        "@dataclass(frozen=True)\n"
        "class Finding:\n"
        "    kind: str\n"
        "    function: str\n"
        "    value_id: int | None\n"
        "    detail: str\n"
        "class Publication:\n"
        "    left: int\n"
        "    right: int\n"
        "    has_scope: bool\n"
        "def render(fact: Publication):\n"
        "    found = []\n"
        "    detail = None\n"
        "    if fact.has_scope:\n"
        "        read_scope = (fact.left, fact.right)\n"
        "        detail = f'scope={read_scope!r}'\n"
        "    if detail is not None:\n"
        "        found.append(Finding('bad', 'render', None, detail))\n"
        "    return found\n"
    )
    captured = {}

    def capture(graph):
        captured["root"] = next(
            entry.graph for entry in graph.function_table
            if entry.name == "render"
        )

    with contextlib.redirect_stdout(io.StringIO()):
        lower_ast_source_to_ssa(
            source, "render", name="optional_tuple_field_finding",
            extraction_contract=policy,
            resolved_process_graph_sink=capture,
            progress=lambda _message: None,
        )

    root = captured["root"]
    joined = next(
        data for _node_id, data in root.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.JoinedStr)
    )
    assert joined["attributes"]["source_glyph_sequence_row"]
    optional = next(
        data for _node_id, data in root.G.nodes(data=True)
        if (data.get("attributes") or {}).get(
            "source_optional_glyph_carry_row"
        ) is not None
    )
    assert optional["attributes"]["producer_kind"] == (
        "optional_glyph_sequence"
    )
    constructor = next(
        data for _node_id, data in root.G.nodes(data=True)
        if (data.get("attributes") or {}).get("class_ref") == "Finding"
    )
    assert constructor["attributes"][
        "program_abi_record_identity"
    ] == "Finding"
    destination = next(
        data for _node_id, data in root.G.nodes(data=True)
        if (data.get("attributes") or {}).get(
            "program_abi_sequence_row_identity"
        ) == "Finding"
    )
    assert destination["attributes"]["aggregate_kind"] == "list"
    assert not any(
        (data.get("attributes") or {}).get("extraction_action")
        == "use_native"
        for _node_id, data in root.G.nodes(data=True)
    )


def test_unknown_runtime_tuple_repr_refuses_without_host_boundary():
    """A tuple declaration without member edges is not a repr recipe."""

    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        with contextlib.redirect_stdout(io.StringIO()):
            graph.build_from_ast(ast.parse(
                "class Publication:\n"
                "    read_scope: tuple | None\n"
                "def render(fact: Publication):\n"
                "    if fact.read_scope is not None:\n"
                "        return f'scope={fact.read_scope!r}'\n"
                "    return 'absent'\n"
            ))
        reduce_abstract_tensor_topology(graph)
        function = graph.function_table.entry("render").graph
        joined = next(
            data for _node_id, data in function.G.nodes(data=True)
            if isinstance(data.get("expr_obj"), ast.JoinedStr)
        )
        fact = book.page(SOURCE_GLYPH_SEQUENCE).latest(
            joined["attributes"]["source_glyph_sequence_row"]
        )
        formatted = next(
            component for component in fact.components
            if component.kind == "formatted"
        )
        assert formatted.formatted_sequence_value_id is None
        assert not formatted.projection_components
        assert not any(
            (data.get("attributes") or {}).get("extraction_action")
            == "use_native"
            for _node_id, data in function.G.nodes(data=True)
        )
        with pytest.raises(
            ValueError, match="formatted glyph component has no resident producer"
        ):
            _source_glyph_sequence_materializations(function.G)
    finally:
        end_identity_book(token)


def test_unsupported_format_conversions_do_not_create_host_boundary():
    policy = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    )
    captured = {}

    def capture(graph):
        root = next(
            entry.graph for entry in graph.function_table
            if entry.name == "render"
        )
        captured["root"] = root

    with contextlib.redirect_stdout(io.StringIO()):
        lower_ast_source_to_ssa(
            "def render(value: float):\n"
            "    return f'{value!s}', f'{value!a}', f'{value:0.2f}'\n",
            "render",
            name="unsupported_runtime_format_conversions",
            extraction_contract=policy,
            resolved_process_graph_sink=capture,
            stop_after_compilation_unit_plan=True,
            progress=lambda _message: None,
        )

    root = captured["root"]
    assert not any(
        (data.get("attributes") or {}).get("extraction_action")
        == "use_native"
        for _node_id, data in root.G.nodes(data=True)
    )


def test_glyph_recipe_materializes_through_resident_sequence_operations():
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        with contextlib.redirect_stdout(io.StringIO()):
            graph.build_from_ast(ast.parse(
                "def render(value: str):\n"
                "    return f'a{value}b'\n"
            ))
        reduce_abstract_tensor_topology(graph)
        function = graph.function_table.entry("render").graph
        joined_id, joined = next(
            (node_id, data)
            for node_id, data in function.G.nodes(data=True)
            if isinstance(data.get("expr_obj"), ast.JoinedStr)
        )
        fact = book.page(SOURCE_GLYPH_SEQUENCE).latest(
            joined["attributes"]["source_glyph_sequence_row"]
        )

        declarations, initializations, mutations = (
            _source_glyph_sequence_materializations(function.G)
        )
        component_ids = tuple(
            int(component.component_value_id) for component in fact.components
        )
        assert (joined_id, "duplicates", 1, True) in declarations
        assert all(
            any(item[0] == component_id for item in declarations)
            for component_id in component_ids
        )
        literal_initializations = tuple(
            item for item in initializations
            if str(item[1]).startswith("literal_glyphs=")
        )
        assert len(literal_initializations) == 2
        assert tuple(item.operator for item in mutations) == (
            "clear", "extend", "extend", "extend",
        )
        assert tuple(
            item.argument_value_ids for item in mutations[1:]
        ) == tuple((component_id,) for component_id in component_ids)
        assert all(item.source_contract_cell is not None for item in mutations)
    finally:
        end_identity_book(token)


def test_string_slice_and_len_formatting_have_exact_resident_producers():
    policy = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    )
    captured = {}

    def capture(graph):
        root = next(
            entry.graph for entry in graph.function_table
            if entry.name == "truncate"
        )
        joined = next(
            data for _node_id, data in root.G.nodes(data=True)
            if (data.get("attributes") or {}).get(
                "source_glyph_sequence_row"
            ) is not None
        )
        row = joined["attributes"]["source_glyph_sequence_row"]
        fact = current_identity_book().page(
            SOURCE_GLYPH_SEQUENCE
        ).latest(row)
        captured.update({
            "root": root,
            "fact": fact,
            "materializations": _source_glyph_sequence_materializations(
                root.G
            ),
        })

    with contextlib.redirect_stdout(io.StringIO()):
        lower_ast_source_to_ssa(
            "def truncate(text: str):\n"
            "    if len(text) <= 240:\n"
            "        return text\n"
            "    return f'{text[:240]} ... ({len(text)} chars)'\n",
            "truncate",
            name="truncate_glyph_components",
            extraction_contract=policy,
            resolved_process_graph_sink=capture,
            stop_after_compilation_unit_plan=True,
            progress=lambda _message: None,
        )

    formatted = tuple(
        component for component in captured["fact"].components
        if component.kind == "formatted"
    )
    assert tuple(component.projection_kind for component in formatted) == (
        "string_token", "scalar",
    )
    sliced, length = formatted
    assert sliced.formatted_sequence_value_id is None
    assert captured["root"].G.nodes[sliced.operand_value_id][
        "attributes"
    ]["producer_kind"] == "string_slice"
    assert length.formatted_sequence_value_id is not None
    length_formatter = captured["root"].G.nodes[
        length.formatted_sequence_value_id
    ]
    assert length_formatter["attributes"]["callee_resolution"] == (
        "source-glyph-formatter"
    )
    declarations, _initializations, _mutations = captured[
        "materializations"
    ]
    declared = {int(value_id) for value_id, _policy, _columns, _write in declarations}
    assert sliced.component_value_id in declared
    assert length.formatted_sequence_value_id in declared


def test_record_repr_publishes_declared_field_projection_order():
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        tree = ast.parse(
            "from dataclasses import dataclass\n"
            "@dataclass(frozen=True)\n"
            "class Fact:\n"
            "    count: int\n"
            "    label: str\n"
            "def render(fact: Fact):\n"
            "    return f'{fact!r}'\n"
        )
        with contextlib.redirect_stdout(io.StringIO()):
            graph.build_from_ast(tree)
        reduce_abstract_tensor_topology(graph)
        function = graph.function_table.entry("render").graph
        joined = next(
            data for _node_id, data in function.G.nodes(data=True)
            if isinstance(data.get("expr_obj"), ast.JoinedStr)
        )
        fact = book.page(SOURCE_GLYPH_SEQUENCE).latest(
            joined["attributes"]["source_glyph_sequence_row"]
        )
        component, = fact.components
        assert component.projection_kind == "record"
        assert component.record_identity == "Fact"
        assert component.projection_names == ("count", "label")
        canonical_cell = book.latest_ref(
            SOURCE_GLYPH_SEQUENCE,
            joined["attributes"]["source_glyph_sequence_row"],
        )
        source_cell = _source_glyph_cell(book, canonical_cell)
        incoming = {
            source for source, _stage in book.edges_into(source_cell)
        }
        assert any(source.page is SOURCE_VALUE_CLASS for source in incoming)
        module_scope = tree._turing_source_module
        assert {
            (module_scope, "Fact", "count"),
            (module_scope, "Fact", "label"),
        } == {
            source.row for source in incoming
            if source.page is CLASS_FIELD_DECLARATION
        }
        assert sum(
            source.page is IDENTITY_TRANSITION for source in incoming
        ) >= 2
    finally:
        end_identity_book(token)


def test_annotated_parameter_field_uses_declared_string_projection():
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        tree = ast.parse(
            "class Page:\n"
            "    name: str\n"
            "def render(page: Page):\n"
            "    return f'{page.name}'\n"
        )
        with contextlib.redirect_stdout(io.StringIO()):
            graph.build_from_ast(tree)
        reduce_abstract_tensor_topology(graph)
        function = graph.function_table.entry("render").graph
        joined = next(
            data for _node_id, data in function.G.nodes(data=True)
            if isinstance(data.get("expr_obj"), ast.JoinedStr)
        )
        fact = book.page(SOURCE_GLYPH_SEQUENCE).latest(
            joined["attributes"]["source_glyph_sequence_row"]
        )
        component, = fact.components
        assert component.projection_kind == "string_token"
        canonical_cell = book.latest_ref(
            SOURCE_GLYPH_SEQUENCE,
            joined["attributes"]["source_glyph_sequence_row"],
        )
        source_cell = _source_glyph_cell(book, canonical_cell)
        incoming = {
            source for source, _stage in book.edges_into(source_cell)
        }
        module_scope = tree._turing_source_module
        assert any(
            source.page is PARAMETER_ANNOTATION
            and source.row == (module_scope, "render", "page")
            for source in incoming
        )
        assert any(
            source.page is CLASS_FIELD_DECLARATION
            and source.row == (module_scope, "Page", "name")
            for source in incoming
        )
    finally:
        end_identity_book(token)


def test_string_partition_member_keeps_declared_string_class():
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        tree = ast.parse(
            "class Claim:\n"
            "    key: str\n"
            "def render(claim: Claim):\n"
            "    kind, _, identity = claim.key.partition(':')\n"
            "    return f'{kind}'\n"
        )
        with contextlib.redirect_stdout(io.StringIO()):
            graph.build_from_ast(tree)
        reduce_abstract_tensor_topology(graph)
        function = graph.function_table.entry("render").graph
        partition_call = next(
            (node_id, data)
            for node_id, data in function.G.nodes(data=True)
            if isinstance(data.get("expr_obj"), ast.Call)
            and isinstance(data["expr_obj"].func, ast.Attribute)
            and data["expr_obj"].func.attr == "partition"
        )
        _call_id, call_data = partition_call
        member_ids = tuple(sorted(
            int(child)
            for child, role in call_data.get("children", ())
            if str(role) == "base"
            and str(function.G.nodes[child].get("type")) == "Indexed"
        ))
        assert len(member_ids) == 3
        class_page = book.page(SOURCE_VALUE_CLASS)
        class_rows = {
            int(row[1]): row
            for row in class_page.rows()
            if len(row) == 2 and int(row[1]) in member_ids
        }
        assert set(class_rows) == set(member_ids)
        assert all(
            class_page.latest(class_rows[member_id])[:2]
            == ("builtins.str", 1)
            for member_id in member_ids
        )
        for member_id in member_ids:
            cell = book.latest_ref(SOURCE_VALUE_CLASS, class_rows[member_id])
            incoming = {
                source for source, _stage in book.edges_into(cell)
            }
            source_class = next(
                source
                for source in incoming
                if source.page is SOURCE_VALUE_CLASS
            )
            source_incoming = {
                source for source, _stage in book.edges_into(source_class)
            }
            assert {
                source.row[2]
                for source in source_incoming
                if source.page is IDENTITY_TRANSITION
            } >= {"operand", "base", "index"}

        joined = next(
            data for _node_id, data in function.G.nodes(data=True)
            if isinstance(data.get("expr_obj"), ast.JoinedStr)
        )
        fact = book.page(SOURCE_GLYPH_SEQUENCE).latest(
            joined["attributes"]["source_glyph_sequence_row"]
        )
        component, = fact.components
        assert component.projection_kind == "string_token"
        assert component.operand_value_id in member_ids
        assert component.formatted_sequence_value_id is None
    finally:
        end_identity_book(token)


def test_annotated_local_field_uses_declared_string_projection():
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        tree = ast.parse(
            "class RowField:\n"
            "    name: str\n"
            "def render(value):\n"
            "    declared: RowField\n"
            "    declared = value\n"
            "    return f'{declared.name!r}'\n"
        )
        with contextlib.redirect_stdout(io.StringIO()):
            graph.build_from_ast(tree)
        reduce_abstract_tensor_topology(graph)
        function = graph.function_table.entry("render").graph
        joined = next(
            data for _node_id, data in function.G.nodes(data=True)
            if isinstance(data.get("expr_obj"), ast.JoinedStr)
        )
        fact = book.page(SOURCE_GLYPH_SEQUENCE).latest(
            joined["attributes"]["source_glyph_sequence_row"]
        )
        component, = fact.components
        assert component.projection_kind == "string_token"
        canonical_cell = book.latest_ref(
            SOURCE_GLYPH_SEQUENCE,
            joined["attributes"]["source_glyph_sequence_row"],
        )
        source_cell = _source_glyph_cell(book, canonical_cell)
        incoming = {
            source for source, _stage in book.edges_into(source_cell)
        }
        module_scope = tree._turing_source_module
        assert any(
            source.page is ANNOTATION_DECLARATION
            and source.row == (module_scope, "render", "declared")
            for source in incoming
        )
        assert any(
            source.page is CLASS_FIELD_DECLARATION
            and source.row == (module_scope, "RowField", "name")
            for source in incoming
        )
    finally:
        end_identity_book(token)


def test_enum_name_projection_uses_authored_resident_lookup_table_after_rehydration():
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        tree = ast.parse(
            "from enum import Enum\n"
            "class Kind(Enum):\n"
            "    LEFT = 'left'\n"
            "    RIGHT = 'right'\n"
            "class Row:\n"
            "    kind: Kind\n"
            "def render(row: Row):\n"
            "    return f'{row.kind.name}'\n"
        )
        with contextlib.redirect_stdout(io.StringIO()):
            graph.build_from_ast(tree)
        enum_row = (tree._turing_source_module, "Kind")
        enum_cell = book.latest_ref(SOURCE_ENUM_GLYPH_TABLE, enum_row)
        enum_fact = book.page(SOURCE_ENUM_GLYPH_TABLE).cell(
            enum_cell.row, enum_cell.column,
        )
        assert tuple(
            member.member_identity for member in enum_fact.members
        ) == ("LEFT", "RIGHT")
        assert tuple(
            member.runtime_value for member in enum_fact.members
        ) == (string_token("left"), string_token("right"))

        reduce_abstract_tensor_topology(graph)
        function = graph.function_table.entry("render").graph
        joined = next(
            data for _node_id, data in function.G.nodes(data=True)
            if isinstance(data.get("expr_obj"), ast.JoinedStr)
        )
        glyph_fact = book.page(SOURCE_GLYPH_SEQUENCE).latest(
            joined["attributes"]["source_glyph_sequence_row"]
        )
        component, = glyph_fact.components
        assert component.projection_kind == "enum_name"
        assert component.record_identity == "Kind"
        assert component.enum_table_cell == enum_cell

        declarations, initializations, mutations = (
            _string_token_glyph_materializations(function.G, None)
        )
        mutation, = mutations
        assert mutation.argument_kind == "enum_name_glyph_projection"
        assert mutation.operator == "glyph_token_lookup_copy"
        assert mutation.argument_value_ids[0] == component.operand_value_id
        vocabulary_id = mutation.argument_value_ids[1]
        assert (vocabulary_id, "unique", 3, False) in declarations
        table = next(
            value for sequence_id, value, columns in initializations
            if sequence_id == vocabulary_id and columns == 3
        )
        assert str(string_token("left")) in table
        assert str(string_token("L")) in table
        vocabulary_cell = mutation.vocabulary_contract_cell
        # The vocabulary's id is minted by the book from the Enum's source
        # glyph table: the mint edge, not a Derived edge, names the cell.
        assert book.mint_of(vocabulary_cell) == (
            STRING_GLYPH_VOCABULARY_MINT, (enum_cell,),
        )
        assert book.edges_into(vocabulary_cell) == ()

        # Compilation-unit control can cross a tile/pickle boundary after the
        # exact vocabulary Ref is minted.  Its Page is rehydrated by value;
        # the cell key must still resolve against this book's retained fact.
        rehydrated = pickle.loads(pickle.dumps(mutation))
        assert rehydrated.vocabulary_contract_cell.page is not (
            book.registry.pages["string_glyph_vocabulary"]
        )
        lowered, shortfalls = lower_control_program_to_ssa(
            ControlProgram(SequenceBlock((
                SequenceMutationBlock(rehydrated),
            ))),
            function_name="rehydrated_enum_glyph_vocabulary",
            sequence_declarations=(
                *declarations,
                (
                    int(rehydrated.sequence_value_id),
                    "duplicates", 1, True,
                ),
            ),
            sequence_initializations=initializations,
        )
        assert shortfalls == ()
        assert any(
            instruction.attributes.get("ssa_sequence_operation")
            == "glyph_token_lookup_copy"
            and instruction.attributes.get("vocabulary_sequence_id")
            == vocabulary_id
            for block in lowered.blocks.values()
            for instruction in block.instrs
        )
    finally:
        end_identity_book(token)


def test_tuple_repr_publishes_exact_element_identities():
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        with contextlib.redirect_stdout(io.StringIO()):
            graph.build_from_ast(ast.parse(
                "def render(left, right):\n"
                "    pair = (left, right)\n"
                "    return f'{pair!r}'\n"
            ))
        reduce_abstract_tensor_topology(graph)
        function = graph.function_table.entry("render").graph
        joined = next(
            data for _node_id, data in function.G.nodes(data=True)
            if isinstance(data.get("expr_obj"), ast.JoinedStr)
        )
        fact = book.page(SOURCE_GLYPH_SEQUENCE).latest(
            joined["attributes"]["source_glyph_sequence_row"]
        )
        component, = fact.components
        operand = function.G.nodes[component.operand_value_id]
        assert component.projection_kind == "tuple"
        assert component.projection_value_ids == tuple(
            operand["attributes"]["aggregate_leaf_value_ids"]
        )
        assert len(component.projection_value_ids) == 2
        canonical_cell = book.latest_ref(
            SOURCE_GLYPH_SEQUENCE,
            joined["attributes"]["source_glyph_sequence_row"],
        )
        source_cell = _source_glyph_cell(book, canonical_cell)
        incoming = {
            source for source, _stage in book.edges_into(source_cell)
        }
        # JoinedStr component order, FormattedValue operand position, and
        # both ordered tuple-member positions are all explicit graph cells.
        assert sum(
            source.page is IDENTITY_TRANSITION for source in incoming
        ) >= 4
    finally:
        end_identity_book(token)


def test_borrowed_page_latest_is_record_after_exact_none_arm_refinement():
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        with contextlib.redirect_stdout(io.StringIO()):
            graph.build_from_ast(ast.parse(
                "from dataclasses import dataclass\n"
                "@dataclass(frozen=True)\n"
                "class Fact:\n"
                "    count: int\n"
                "from src.compiler.identity_concordance import (\n"
                "    RowField, RowFieldKind as K, declare_page,\n"
                ")\n"
                "FACTS = declare_page(\n"
                "    'facts', (RowField('index', K.INDEX),), Fact,\n"
                ")\n"
                "def render(book):\n"
                "    with book.borrowed('facts') as facts:\n"
                "        fact = facts.latest((0,))\n"
                "        if fact is None:\n"
                "            return 'absent'\n"
                "        return f'{fact!r}'\n"
            ))
        reduce_abstract_tensor_topology(graph)
        function = graph.function_table.entry("render").graph
        joined = next(
            data for _node_id, data in function.G.nodes(data=True)
            if isinstance(data.get("expr_obj"), ast.JoinedStr)
        )
        fact = book.page(SOURCE_GLYPH_SEQUENCE).latest(
            joined["attributes"]["source_glyph_sequence_row"]
        )
        component, = fact.components
        assert component.projection_kind == "record"
        assert component.record_identity == "Fact"
        assert component.projection_names == ("count",)
    finally:
        end_identity_book(token)


def test_page_rows_and_isinstance_else_arm_carry_declared_schema():
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        with contextlib.redirect_stdout(io.StringIO()):
            graph.build_from_ast(ast.parse(
                "from dataclasses import dataclass\n"
                "@dataclass(frozen=True)\n"
                "class Fact:\n"
                "    count: int\n"
                "from src.compiler.identity_concordance import (\n"
                "    RowField, RowFieldKind as K, declare_page,\n"
                ")\n"
                "FACTS = declare_page(\n"
                "    'facts', (RowField('scope', K.SCOPE), "
                "RowField('index', K.INDEX)), Fact,\n"
                ")\n"
                "def render(book):\n"
                "    with book.borrowed('facts') as facts:\n"
                "        for row in facts.rows():\n"
                "            fact = facts.latest(row)\n"
                "            if not isinstance(fact, Fact):\n"
                "                continue\n"
                "            count = fact.count\n"
                "    return 0\n"
            ))
        reduce_abstract_tensor_topology(graph)
        function = graph.function_table.entry("render").graph
        rows = next(
            data for _node_id, data in function.G.nodes(data=True)
            if isinstance(data.get("expr_obj"), ast.Call)
            and ast.unparse(data["expr_obj"]) == "facts.rows()"
        )
        assert rows["attributes"]["sequence_element_row_fields"] == (
            ("scope", "SCOPE"), ("index", "INDEX"),
        )
        loop_row = next(
            data for _node_id, data in function.G.nodes(data=True)
            if (data.get("attributes") or {}).get("producer_kind")
            == "page_row"
        )
        declaration_cell = loop_row["attributes"][
            "source_page_declaration_cell"
        ]
        assert declaration_cell.page == SOURCE_PAGE_FACT_CLASS
        declaration = current_identity_book().page(
            SOURCE_PAGE_FACT_CLASS
        ).cell(declaration_cell.row, declaration_cell.column)
        assert declaration.row_fields == (
            ("scope", "SCOPE"), ("index", "INDEX"),
        )
        assert "source_page_row_fields" not in loop_row["attributes"]
        assert "source_page_fact_class" not in loop_row["attributes"]
        count = next(
            data for _node_id, data in function.G.nodes(data=True)
            if isinstance(data.get("expr_obj"), ast.Attribute)
            and data["expr_obj"].attr == "count"
        )
        assert count["attributes"]["source_class_refinement_cells"], count
        assert count["tensor"] == {"shape": (), "dtype": "int64"}
    finally:
        end_identity_book(token)


def test_page_scope_str_call_builds_resident_tuple_glyph_recipe():
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        with contextlib.redirect_stdout(io.StringIO()):
            graph.build_from_ast(ast.parse(
                "from dataclasses import dataclass\n"
                "@dataclass(frozen=True)\n"
                "class Fact:\n"
                "    count: int\n"
                "from src.compiler.identity_concordance import (\n"
                "    RowField, RowFieldKind as K, declare_page,\n"
                ")\n"
                "FACTS = declare_page(\n"
                "    'facts', (RowField('scope', K.SCOPE), "
                "RowField('index', K.INDEX)), Fact,\n"
                ")\n"
                "def render(book):\n"
                "    with book.borrowed('facts') as facts:\n"
                "        for row in facts.rows():\n"
                "            return str(row[0])\n"
                "    return 'absent'\n"
            ))
        reduce_abstract_tensor_topology(graph)
        function = graph.function_table.entry("render").graph
        rendered = next(
            data for _node_id, data in function.G.nodes(data=True)
            if isinstance(data.get("expr_obj"), ast.Call)
            and ast.unparse(data["expr_obj"]) == "str(row[0])"
        )
        fact = book.page(SOURCE_GLYPH_SEQUENCE).latest(
            rendered["attributes"]["source_glyph_sequence_row"]
        )
        assert tuple(component.projection_kind for component in fact.components) == (
            "literal", "string_token", "literal", "scalar", "literal",
        )
        scalar = fact.components[3]
        assert function.G.nodes[scalar.operand_value_id]["tensor"] == {
            "shape": (), "dtype": "int64",
        }
        # The helper call is linked when the whole-program pursuit brings the
        # authored decimal formatter into the FunctionTable.  This isolated
        # reducer fixture proves the page schema drives the exact projections,
        # without inventing a stand-in formatter address.
    finally:
        end_identity_book(token)


def test_page_latest_key_declares_exact_optional_field_tuple_repr():
    """The consuming Page key, not the bare tuple annotation, owns layout."""

    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        with contextlib.redirect_stdout(io.StringIO()):
            graph.build_from_ast(ast.parse(
                "from dataclasses import dataclass\n"
                "@dataclass(frozen=True)\n"
                "class Publication:\n"
                "    read_scope: tuple | None\n"
                "from src.compiler.identity_concordance import (\n"
                "    RowField, RowFieldKind as K, declare_page,\n"
                ")\n"
                "PUBLICATIONS = declare_page(\n"
                "    'publications', (RowField('index', K.INDEX),), "
                "Publication,\n"
                ")\n"
                "SCOPES = declare_page(\n"
                "    'scopes', (RowField('label', K.LABEL), "
                "RowField('serial', K.INDEX)), bool,\n"
                ")\n"
                "def render(book):\n"
                "    with book.borrowed('publications') as publications, "
                "book.borrowed('scopes') as scopes:\n"
                "        fact = publications.latest((0,))\n"
                "        if not isinstance(fact, Publication):\n"
                "            return 'absent'\n"
                "        if (fact.read_scope is not None "
                "and scopes.latest(fact.read_scope) is not True):\n"
                "            return f'{fact.read_scope!r}'\n"
                "    return 'valid'\n"
            ))
        reduce_abstract_tensor_topology(graph)
        function = graph.function_table.entry("render").graph
        joined = next(
            data for _node_id, data in function.G.nodes(data=True)
            if isinstance(data.get("expr_obj"), ast.JoinedStr)
        )
        fact = book.page(SOURCE_GLYPH_SEQUENCE).latest(
            joined["attributes"]["source_glyph_sequence_row"]
        )
        formatted = next(
            component for component in fact.components
            if component.kind == "formatted"
        )
        assert formatted.projection_kind == "tuple"
        assert tuple(
            component.projection_kind
            for component in formatted.projection_components
        ) == (
            "literal", "string_token", "literal", "scalar", "literal",
        )
        operand = function.G.nodes[formatted.operand_value_id]
        declaration_cell = operand["attributes"][
            "source_page_declaration_cell"
        ]
        declaration = book.page(SOURCE_PAGE_FACT_CLASS).cell(
            declaration_cell.row, declaration_cell.column,
        )
        assert declaration.page_name == "scopes"
        assert declaration.class_identity == "builtins.bool"
        assert declaration.row_fields == (
            ("label", "LABEL"), ("serial", "INDEX"),
        )
        assert not any(
            (data.get("attributes") or {}).get("extraction_action")
            == "use_native"
            for _node_id, data in function.G.nodes(data=True)
        )
    finally:
        end_identity_book(token)


def test_page_latest_or_absence_repr_owns_nested_optional_glyph_carry():
    """The original OR keeps record formatting behind exact presence."""

    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        with contextlib.redirect_stdout(io.StringIO()):
            graph.build_from_ast(ast.parse(
                "from dataclasses import dataclass\n"
                "@dataclass(frozen=True)\n"
                "class Region:\n"
                "    schedule_preference: str\n"
                "    serial: int\n"
                "from src.compiler.identity_concordance import (\n"
                "    RowField, RowFieldKind as K, declare_page,\n"
                ")\n"
                "REGIONS = declare_page(\n"
                "    'regions', (RowField('index', K.INDEX),), Region,\n"
                ")\n"
                "def render(book, preference: str):\n"
                "    with book.borrowed('regions') as regions:\n"
                "        region = regions.latest((0,))\n"
                "        if (region is None or "
                "region.schedule_preference != preference):\n"
                "            return f'mismatch: {region!r}'\n"
                "    return 'valid'\n"
            ))
        reduce_abstract_tensor_topology(graph)
        function = graph.function_table.entry("render").graph
        merged_id, merged = next(
            (int(node_id), data)
            for node_id, data in function.G.nodes(data=True)
            if (data.get("attributes") or {}).get(
                "source_optional_glyph_dispatch"
            ) is not None
        )
        row = merged["attributes"]["source_optional_glyph_carry_row"]
        fact = book.page(SOURCE_OPTIONAL_GLYPH_CARRY).latest(row)
        assert fact.merged_value_id == merged_id
        assert fact.initial_value_id is None
        assert fact.predicate_expect_true is False
        assert fact.owner_conditional_id == merged["attributes"][
            "source_optional_glyph_dispatch"
        ]["owner_conditional_id"]
        assert fact.true_arm.sequence_value_id == fact.true_value_id
        assert fact.false_arm.sequence_value_id == fact.false_value_id
        assert fact.true_arm.presence is True
        assert fact.false_arm.presence is True
        assert fact.present_projection_sequence_ids
        absent = book.page(SOURCE_GLYPH_SEQUENCE).cell(
            fact.false_arm.presence_source.row,
            fact.false_arm.presence_source.column,
        )
        assert tuple(
            token
            for component in absent.components
            for token in component.glyph_tokens
        ) == tuple(string_token(character) for character in "None")
        joined = next(
            data for _node_id, data in function.G.nodes(data=True)
            if isinstance(data.get("expr_obj"), ast.JoinedStr)
        )
        joined_fact = book.page(SOURCE_GLYPH_SEQUENCE).latest(
            joined["attributes"]["source_glyph_sequence_row"]
        )
        assert any(
            component.formatted_sequence_value_id == merged_id
            and component.projection_kind == "optional_glyph_sequence"
            for component in joined_fact.components
        )
    finally:
        end_identity_book(token)


def test_optional_glyph_work_refuses_partial_or_unrecognized_extraction():
    fact = SimpleNamespace(
        merged_value_id=30,
        present_callsite_ids=(11,),
        present_projection_sequence_ids=(21, 22),
    )
    call = CallBlock(11, SequenceBlock(()))
    first = SequenceMutationBlock(ControlSequenceMutation(
        21, "clear", (), 21, argument_kind="glyph_sequence_clear",
    ))
    second = SequenceMutationBlock(ControlSequenceMutation(
        22, "clear", (), 22, argument_kind="glyph_sequence_clear",
    ))
    with pytest.raises(ValueError, match="partial or not directly owned"):
        _verify_optional_glyph_work(fact, (call, first), ())
    with pytest.raises(ValueError, match="unrecognized=1"):
        _verify_optional_glyph_work(
            fact, (call, first, second), (second,),
        )


def test_isinstance_true_arm_owns_record_repr_refinement():
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        with contextlib.redirect_stdout(io.StringIO()):
            graph.build_from_ast(ast.parse(
                "from dataclasses import dataclass\n"
                "@dataclass(frozen=True)\n"
                "class Fact:\n"
                "    count: int\n"
                "def render(value):\n"
                "    if isinstance(value, Fact):\n"
                "        return f'{value!r}'\n"
                "    return 'invalid'\n"
            ))
        reduce_abstract_tensor_topology(graph)
        function = graph.function_table.entry("render").graph
        joined = next(
            data for _node_id, data in function.G.nodes(data=True)
            if isinstance(data.get("expr_obj"), ast.JoinedStr)
        )
        fact = book.page(SOURCE_GLYPH_SEQUENCE).latest(
            joined["attributes"]["source_glyph_sequence_row"]
        )
        component, = fact.components
        assert component.projection_kind == "record"
        assert component.record_identity == "Fact"
    finally:
        end_identity_book(token)


def test_refined_record_getattrs_keep_declared_member_repr_classes():
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        with contextlib.redirect_stdout(io.StringIO()):
            graph.build_from_ast(ast.parse(
                "from dataclasses import dataclass\n"
                "@dataclass(frozen=True)\n"
                "class Fact:\n"
                "    count: int\n"
                "    label: str\n"
                "def render(value):\n"
                "    if not isinstance(value, Fact):\n"
                "        return 'invalid'\n"
                "    return f'{value.count!r}:{value.label!r}'\n"
            ))
        reduce_abstract_tensor_topology(graph)
        function = graph.function_table.entry("render").graph
        joined = next(
            data for _node_id, data in function.G.nodes(data=True)
            if isinstance(data.get("expr_obj"), ast.JoinedStr)
        )
        fact = book.page(SOURCE_GLYPH_SEQUENCE).latest(
            joined["attributes"]["source_glyph_sequence_row"]
        )
        formatted = tuple(
            component for component in fact.components
            if component.kind == "formatted"
        )
        assert tuple(
            component.projection_kind for component in formatted
        ) == ("scalar", "string_token")
        canonical_cell = book.latest_ref(
            SOURCE_GLYPH_SEQUENCE,
            joined["attributes"]["source_glyph_sequence_row"],
        )
        source_cell = _source_glyph_cell(book, canonical_cell)
        incoming = {
            source for source, _stage in book.edges_into(source_cell)
        }
        assert {
            source.row[2] for source in incoming
            if source.page is CLASS_FIELD_DECLARATION
        } == {"count", "label"}
    finally:
        end_identity_book(token)


def test_record_repr_builds_resident_projection_sequence_end_to_end():
    policy = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "records": {"Fact": {"fields": {
            "count": {"storage": "scalar", "dtype": "int64", "rank": 0},
            "label": {"storage": "scalar", "dtype": "int64", "rank": 0},
        }}},
        "bindings": [{
            "function": "render", "parameter": "fact", "record": "Fact",
        }],
        "values": [],
    })
    source = (
        "from dataclasses import dataclass\n"
        "@dataclass(frozen=True)\n"
        "class Fact:\n"
        "    count: int\n"
        "    label: str\n"
        "def render(fact: Fact):\n"
        "    return f'{fact!r}'\n"
    )
    with contextlib.redirect_stdout(io.StringIO()):
        module, _outputs, exports = lower_ast_source_to_ssa(
            source, "render", name="record_glyph_projection",
            extraction_contract=policy, progress=lambda _message: None,
        )
    assert exports == (
        "record_glyph_projection__render",
    )
    root = module.functions[exports[0]]
    operations = tuple(
        instruction
        for block in root.blocks.values()
        for instruction in block.instrs
    )
    assert any(
        (instruction.attributes or {}).get("ssa_sequence_operation")
        == "extend" for instruction in operations
    )
    assert not any(
        (instruction.attributes or {}).get("external_reference")
        for instruction in operations
    )


def test_tuple_repr_builds_resident_projection_sequence_end_to_end():
    policy = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "records": {}, "bindings": [],
        "values": [{
            "function": "render", "parameter": name,
            "storage": "scalar", "dtype": "int64", "rank": 0,
            "python_type": "builtins.int",
        } for name in ("left", "right")],
    })
    source = (
        "def render(left: int, right: int):\n"
        "    pair = (left, right)\n"
        "    return f'{pair!r}'\n"
    )
    with contextlib.redirect_stdout(io.StringIO()):
        module, _outputs, exports = lower_ast_source_to_ssa(
            source, "render", name="tuple_glyph_projection",
            extraction_contract=policy, progress=lambda _message: None,
        )
    root = module.functions[exports[0]]
    operations = tuple(
        instruction
        for block in root.blocks.values()
        for instruction in block.instrs
    )
    assert sum(
        (instruction.attributes or {}).get("ssa_sequence_operation")
        == "extend" for instruction in operations
    ) >= 3
    assert not any(
        (instruction.attributes or {}).get("external_reference")
        for instruction in operations
    )


def test_typed_identity_page_name_is_resident_glyph_producer():
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        with contextlib.redirect_stdout(io.StringIO()):
            graph.build_from_ast(ast.parse(
                "class IdentityPage:\n"
                "    name: str\n"
                "def replace(page: IdentityPage):\n"
                "    return f'{page.name}: row'\n"
            ))
        reduce_abstract_tensor_topology(graph)
        function = graph.function_table.entry("replace").graph
        joined = next(
            data for _node_id, data in function.G.nodes(data=True)
            if isinstance(data.get("expr_obj"), ast.JoinedStr)
        )
        fact = book.page(SOURCE_GLYPH_SEQUENCE).latest(
            joined["attributes"]["source_glyph_sequence_row"]
        )
        formatted = next(
            component for component in fact.components
            if component.kind == "formatted"
        )
        assert formatted.projection_kind == "string_token"
        declarations, _initializations, mutations = (
            _source_glyph_sequence_materializations(function.G)
        )
        assert any(
            declaration[0] == formatted.component_value_id
            for declaration in declarations
        )
        assert any(
            mutation.argument_kind == "glyph_sequence_component"
            and mutation.argument_value_ids == (
                formatted.component_value_id,
            )
            for mutation in mutations
        )
    finally:
        end_identity_book(token)


def test_optional_glyph_carry_chains_presence_from_exact_source_cells():
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        with contextlib.redirect_stdout(io.StringIO()):
            graph.build_from_ast(ast.parse(
                "def render(first: bool, second: bool, value: int):\n"
                "    detail = None\n"
                "    if first:\n"
                "        detail = 'fixed'\n"
                "    if second:\n"
                "        detail = f'value={value}'\n"
                "    return detail\n"
            ))
        reduce_abstract_tensor_topology(graph)
        function = graph.function_table.entry("render").graph
        candidates = {}
        for node_id, data in function.G.nodes(data=True):
            attributes = data.get("attributes") or {}
            control_id = attributes.get("source_conditional_id")
            parents = {
                str(role): int(function.G.nodes[parent].get(
                    "value_id", parent
                ))
                for parent, role in data.get("parents") or ()
                if parent in function.G
            }
            if (
                str(data.get("type") or data.get("op") or "").casefold()
                != "phi"
                or control_id is None
                or attributes.get("initial_value_id") is None
                or not {"body", "orelse"}.issubset(parents)
            ):
                continue
            candidates.setdefault(int(control_id), []).append((
                int(parents["body"]),
                int(parents["orelse"]),
                int(attributes["initial_value_id"]),
                int(data.get("value_id", node_id)),
            ))
        cells = _post_source_optional_glyph_carries(
            function,
            tuple(
                (control_id, tuple(aliases))
                for control_id, aliases in candidates.items()
            ),
        )
        assert len(cells) == 2
        facts = [
            book.page(SOURCE_OPTIONAL_GLYPH_CARRY).latest(cell.row)
            for cell in cells
        ]
        for cell, fact in zip(cells, facts):
            incoming = {
                source for source, _stage in book.edges_into(cell)
            }
            assert {
                fact.true_arm.presence_source,
                fact.false_arm.presence_source,
                fact.initial_arm.presence_source,
            }.issubset(incoming)
        inner = next(
            fact for fact in facts
            if fact.true_arm.presence is True
            and fact.false_arm.presence is False
        )
        outer = next(fact for fact in facts if fact is not inner)
        inherited = next(
            arm for arm in (outer.true_arm, outer.false_arm)
            if arm.presence is None
        )
        assert inherited.value_id == inner.merged_value_id
        assert inherited.presence_source.row == (
            cells[0].row[0], inner.merged_value_id,
        )
        assert outer.payload_sequence_value_id == outer.merged_value_id
        merged = function.G.nodes[outer.merged_value_id]["attributes"]
        assert merged["producer_kind"] == "optional_glyph_sequence"
        assert merged["sequence_element_dtypes"] == ("int64",)
    finally:
        end_identity_book(token)


def test_dynamic_scalar_string_token_is_not_coerced_to_glyph_sequence():
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        with contextlib.redirect_stdout(io.StringIO()):
            graph.build_from_ast(ast.parse(
                "def render(flag: bool, incumbent, value: int):\n"
                "    detail = incumbent\n"
                "    if flag:\n"
                "        detail = f'value={value}'\n"
                "    return detail\n"
            ))
        reduce_abstract_tensor_topology(graph)
        function = graph.function_table.entry("render").graph
        candidate = next(
            (
                int((data.get("attributes") or {})["source_conditional_id"]),
                ((
                    int(parents["body"]), int(parents["orelse"]),
                    int((data.get("attributes") or {})["initial_value_id"]),
                    int(data.get("value_id", node_id)),
                ),),
            )
            for node_id, data in function.G.nodes(data=True)
            for parents in ({
                str(role): int(function.G.nodes[parent].get(
                    "value_id", parent
                ))
                for parent, role in data.get("parents") or ()
                if parent in function.G
            },)
            if str(data.get("type") or data.get("op") or "").casefold()
            == "phi"
            and (data.get("attributes") or {}).get(
                "source_conditional_id"
            ) is not None
            and (data.get("attributes") or {}).get("initial_value_id")
            is not None
            and {"body", "orelse"}.issubset(parents)
        )
        assert _post_source_optional_glyph_carries(
            function, (candidate,)
        ) == ()
        assert tuple(book.page(SOURCE_OPTIONAL_GLYPH_CARRY).rows()) == ()
    finally:
        end_identity_book(token)


def test_two_arm_return_phi_publishes_optional_glyph_contract():
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        tree = ast.parse(
            "def render(flag: bool, value: int) -> str | None:\n"
            "    if flag:\n"
            "        return f'value={value}'\n"
            "    return None\n"
        )
        assert _normalize_top_level_guard_returns(tree, ("render",))
        with contextlib.redirect_stdout(io.StringIO()):
            graph.build_from_ast(tree)
        reduce_abstract_tensor_topology(graph)
        function = graph.function_table.entry("render").graph
        source_scope = (
            function.G.graph.get("source_numeric_scope")
            or function.G.graph["authored_function_identity"]
        )
        # The carry names its arms' glyph cells, which are copy-local, so it
        # is keyed by the copy's canonical glyph scope (never the authored
        # scope two retained copies would share).
        carry_scopes = {
            tuple(source_scope),
            tuple(function.G.graph["source_glyph_sequence_scope"]),
        }
        facts = tuple(
            book.page(SOURCE_OPTIONAL_GLYPH_CARRY).latest(row)
            for scope in carry_scopes
            for row in book.page(SOURCE_OPTIONAL_GLYPH_CARRY).scope_rows(scope)
        )
        fact = next(
            candidate for candidate in facts
            if candidate.initial_value_id is None
        )
        assert {fact.true_arm.presence, fact.false_arm.presence} == {
            True, False,
        }
        present_arm = next(
            arm for arm in (fact.true_arm, fact.false_arm)
            if arm.presence is True
        )
        present_attributes = function.G.nodes[
            present_arm.value_id
        ]["attributes"]
        present_row = (
            present_attributes.get("source_glyph_sequence_row")
            or present_attributes.get("source_literal_glyph_sequence_row")
        )
        assert present_arm.presence_source == book.latest_ref(
            SOURCE_GLYPH_SEQUENCE, tuple(present_row),
        )
        optional_cell = book.latest_ref(
            SOURCE_OPTIONAL_GLYPH_CARRY,
            tuple(function.G.nodes[fact.merged_value_id]["attributes"][
                "source_optional_glyph_carry_row"
            ]),
        )
        assert optional_cell.row[0] == tuple(
            function.G.graph["source_glyph_sequence_scope"]
        )
        assert node_identity_cell(
            function, fact.conditional_value_id,
        ) in {
            source for source, _stage in book.edges_into(optional_cell)
        }
        merged = function.G.nodes[fact.merged_value_id]["attributes"]
        assert merged["producer_kind"] == "optional_glyph_sequence"
        assert merged["sequence_element_dtypes"] == ("int64",)
    finally:
        end_identity_book(token)


def test_local_list_of_tuple_annotation_publishes_exact_row_dtypes():
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        with contextlib.redirect_stdout(io.StringIO()):
            graph.build_from_ast(ast.parse(
                "def first(rows_source):\n"
                "    rows: list[tuple[int, str]] = rows_source\n"
                "    return rows[0]\n"
            ))
        reduce_abstract_tensor_topology(graph)
        function = graph.function_table.entry("first").graph
        annotated = next(
            data["attributes"]
            for _node_id, data in function.G.nodes(data=True)
            if (data.get("attributes") or {}).get("producer_kind")
            == "annotated_aggregate_binding"
        )
        assert annotated["sequence_column_count"] == 2
        assert annotated["sequence_element_dtypes"] == ("int64", "int64")
    finally:
        end_identity_book(token)


def _reduced_joined_components(source, entry, which=-1):
    """Reduce ``source`` and return the formatted components of one f-string."""

    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        with contextlib.redirect_stdout(io.StringIO()):
            graph.build_from_ast(ast.parse(source))
        reduce_abstract_tensor_topology(graph)
        function = graph.function_table.entry(entry).graph
        joined = sorted(
            (
                data for _node_id, data in function.G.nodes(data=True)
                if isinstance(data.get("expr_obj"), ast.JoinedStr)
                and (data.get("attributes") or {}).get(
                    "source_glyph_sequence_row"
                ) is not None
            ),
            key=lambda data: data["expr_obj"].lineno,
        )[which]
        fact = book.page(SOURCE_GLYPH_SEQUENCE).latest(
            joined["attributes"]["source_glyph_sequence_row"]
        )
        classes = book.page(SOURCE_VALUE_CLASS)
        formatted = tuple(
            component for component in fact.components
            if component.kind == "formatted"
        )
        return formatted, {
            int(component.operand_value_id): next((
                classes.latest(row) for row in classes.rows()
                if len(row) == 2
                and int(row[1]) == int(component.operand_value_id)
            ), None)
            for component in formatted
        }
    finally:
        end_identity_book(token)


def test_str_call_result_has_str_class_even_without_a_glyph_recipe():
    # ``str(thing)`` is a ``str`` by the builtin's own semantics, whatever
    # ``thing`` is: the consuming f-string takes the string_token path.
    formatted, classes = _reduced_joined_components(
        "def render(thing: object):\n"
        "    text = str(thing)\n"
        "    return f'v={text}'\n",
        "render",
    )
    component, = formatted
    assert component.projection_kind == "string_token"
    assert classes[int(component.operand_value_id)][:2] == (
        "builtins.str", 1,
    )
    assert classes[int(component.operand_value_id)][2] == (
        "builtins.str call result"
    )


def test_joined_string_result_has_str_class_for_a_consuming_fstring():
    formatted, classes = _reduced_joined_components(
        "def render(count: int):\n"
        "    inner = f'n={count}'\n"
        "    return f'<{inner}>'\n",
        "render",
    )
    component, = formatted
    assert component.projection_kind == "string_token"
    assert classes[int(component.operand_value_id)][:2] == (
        "builtins.str", 1,
    )


def test_type_name_attribute_has_str_class():
    formatted, classes = _reduced_joined_components(
        "def render(owner):\n"
        "    return f'{type(owner).__name__!r} has no member'\n",
        "render",
    )
    component, = formatted
    assert component.projection_kind == "string_token"
    assert component.conversion == ord("r")
    assert classes[int(component.operand_value_id)][:2] == (
        "builtins.str", 1,
    )


def test_bare_primitive_local_annotation_declares_the_local_class():
    formatted, classes = _reduced_joined_components(
        "def render(items: list[str]):\n"
        "    label: str = items[0]\n"
        "    return f'{label}'\n",
        "render",
    )
    component, = formatted
    assert component.projection_kind == "string_token"
    assert classes[int(component.operand_value_id)][:2] == (
        "builtins.str", 1,
    )


def test_primitive_union_local_annotation_is_not_given_a_class():
    # ``str | None`` has no single class: nothing is invented for it.
    formatted, classes = _reduced_joined_components(
        "def render(items: list[str]):\n"
        "    label: str | None = items[0]\n"
        "    return f'{label}'\n",
        "render",
    )
    component, = formatted
    assert component.projection_kind == "scalar"
    assert classes[int(component.operand_value_id)] is None


def test_string_slice_class_is_read_through_the_canonical_identity():
    formatted, classes = _reduced_joined_components(
        "def render(text: str):\n"
        "    return f'{text[2:-2]}'\n",
        "render",
    )
    component, = formatted
    assert component.projection_kind == "string_token"
    assert classes[int(component.operand_value_id)][2] == (
        "authored string slice"
    )
