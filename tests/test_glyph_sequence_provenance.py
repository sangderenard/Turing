from __future__ import annotations

import ast
import contextlib
import io
from pathlib import Path

from src.common.tensors.topological_reducer import (
    node_identity_cell,
    reduce_abstract_tensor_topology,
)
from src.compiler.concordance_declarations import (
    ANNOTATION_DECLARATION,
    CALL_BINDING,
    CLASS_FIELD_DECLARATION,
    CallResolution,
    IDENTITY_TRANSITION,
    NAME_BINDING,
    PARAMETER_ANNOTATION,
    SOURCE_AGGREGATE_LAYOUT,
    SOURCE_ENUM_GLYPH_TABLE,
    SOURCE_GLYPH_SEQUENCE,
    SOURCE_OPTIONAL_GLYPH_CARRY,
    SOURCE_VALUE_CLASS,
    SourceAggregateLayout,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.ast_control_normalization import (
    _normalize_top_level_guard_returns,
)
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.glsl_deployment_strategy import (
    _post_source_optional_glyph_carries,
)
from src.compiler.glyph_sequence_format import (
    decimal_integer_glyph_tokens,
    prefixed_lower_hex_integer_glyph_tokens,
    utf8_repr_bytes_to_glyph_tokens,
)
from src.compiler.identity_concordance import (
    begin_identity_book,
    current_identity_book,
    end_identity_book,
)
from src.compiler.resident_sequence_materialization import (
    _source_glyph_sequence_materializations,
    _string_token_glyph_materializations,
)
from src.compiler.string_table import string_token
from src.transmogrifier.graph.graph_express2 import ProcessGraph


def _source_glyph_cell(book, canonical_cell):
    return next(
        source for source, _stage in book.edges_into(canonical_cell)
        if source.page is SOURCE_GLYPH_SEQUENCE
        and source.row != canonical_cell.row
    )


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
        assert kind_class[:2] == ("str", 1)
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


def test_utf8_repr_decoder_produces_glyph_identities_not_byte_values():
    for spelling in ("'ascii'", "'café'", "'🧪'"):
        encoded = list(spelling.encode("utf-8"))
        tokens = utf8_repr_bytes_to_glyph_tokens(encoded)
        assert tokens == [string_token(character) for character in spelling]
        assert tokens != encoded


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
            (data.get("attributes") or {}).get("extraction_identity")
            == "src.compiler.glyph_sequence_format.runtime_repr_utf8"
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
        lower_ast_source_to_ssa(
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
        lower_ast_source_to_ssa(
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
        fact is not None and fact[0] == "int"
        for fact in captured["operand_classes"]
    )


def test_unresolved_repr_uses_external_utf8_then_authored_glyph_decoder():
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
            "dtype": "float64",
            "rank": 0,
            "python_type": "builtins.float",
        }],
    })
    captured = {}

    def capture(graph):
        root = next(
            entry.graph for entry in graph.function_table
            if entry.name == "render"
        )
        byte_id, byte_call = next(
            (int(node_id), data)
            for node_id, data in root.G.nodes(data=True)
            if (data.get("attributes") or {}).get("extraction_identity")
            == "src.compiler.glyph_sequence_format.runtime_repr_utf8"
        )
        decoder_id, decoder_call = next(
            (int(node_id), data)
            for node_id, data in root.G.nodes(data=True)
            if (data.get("attributes") or {}).get("callee_resolution")
            == "source-glyph-utf8-decoder"
        )
        joined = next(
            data for _node_id, data in root.G.nodes(data=True)
            if (data.get("attributes") or {}).get(
                "source_glyph_sequence_row"
            ) is not None
        )
        row = joined["attributes"]["source_glyph_sequence_row"]
        captured.update({
            "byte_id": byte_id,
            "byte_call": byte_call,
            "decoder_id": decoder_id,
            "decoder_call": decoder_call,
            "fact": current_identity_book().page(
                SOURCE_GLYPH_SEQUENCE
            ).latest(row),
            "boundaries": tuple(root.G.graph.get(
                "extraction_boundary_calls", ()
            )),
        })

    with contextlib.redirect_stdout(io.StringIO()):
        lower_ast_source_to_ssa(
            "def render(value: float):\n"
            "    return f'x={value!r}'\n",
            "render",
            name="runtime_repr_glyph_topology",
            extraction_contract=policy,
            resolved_process_graph_sink=capture,
            stop_after_compilation_unit_plan=True,
            progress=lambda _message: None,
        )

    byte_id = captured["byte_id"]
    decoder_id = captured["decoder_id"]
    byte_call = captured["byte_call"]
    decoder_call = captured["decoder_call"]
    assert isinstance(byte_call["expr_obj"], ast.Call)
    assert byte_call["parents"] == [(0, "arg:0")]
    assert byte_call["attributes"]["aggregate_kind"] == "bytes"
    assert byte_call["attributes"]["sequence_element_dtypes"] == (
        "int64",
    )
    assert decoder_call["parents"] == [(byte_id, "arg:0")]
    assert decoder_call["attributes"]["aggregate_kind"] == "list"
    formatted = next(
        component for component in captured["fact"].components
        if component.kind == "formatted"
    )
    assert formatted.formatted_sequence_value_id == decoder_id
    assert any(
        boundary.get("source_glyph_component")
        == byte_call["attributes"]["source_glyph_component"]
        and (boundary.get("extraction_contract") or {}).get("action")
        == "use_native"
        for boundary in captured["boundaries"]
    )


def test_runtime_repr_boundary_does_not_admit_other_format_conversions():
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
        (data.get("attributes") or {}).get("extraction_identity")
        == "src.compiler.glyph_sequence_format.runtime_repr_utf8"
        or (data.get("attributes") or {}).get("callee_resolution")
        == "source-glyph-utf8-decoder"
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
            class_page.latest(class_rows[member_id])[:2] == ("str", 1)
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


def test_enum_name_projection_uses_authored_resident_lookup_table():
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
        incoming = {
            source for source, _stage in book.edges_into(vocabulary_cell)
        }
        assert enum_cell in incoming
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
        assert loop_row["attributes"]["source_page_row_fields"] == (
            ("scope", "SCOPE"), ("index", "INDEX"),
        )
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
        "runtime_repr_utf8_glyph_projection__render",
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


def test_runtime_repr_utf8_decode_builds_resident_sequence_end_to_end():
    policy = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "records": {}, "bindings": [],
        "values": [{
            "function": "render", "parameter": "value",
            "storage": "scalar", "dtype": "float64", "rank": 0,
            "python_type": "builtins.float",
        }],
    })
    with contextlib.redirect_stdout(io.StringIO()):
        module, _outputs, exports = lower_ast_source_to_ssa(
            "def render(value: float):\n"
            "    return f'value={value!r}'\n",
            "render", name="runtime_repr_utf8_glyph_projection",
            extraction_contract=policy, progress=lambda _message: None,
        )
    root = module.functions[exports[0]]
    operations = tuple(
        instruction
        for block in root.blocks.values()
        for instruction in block.instrs
    )
    external = next(
        instruction for instruction in operations
        if (instruction.attributes or {}).get("external_reference")
        and (instruction.attributes or {}).get("extraction_identity")
        == "src.compiler.glyph_sequence_format.runtime_repr_utf8"
    )
    assert external.attributes["external_result_sequence"] == "utf8"
    assert any(
        (instruction.attributes or {}).get("callee", "").endswith(
            "utf8_repr_bytes_to_glyph_tokens"
        )
        for instruction in operations
        if instruction.op == "Call"
    )
    assert any(
        (instruction.attributes or {}).get("ssa_sequence_operation")
        == "extend"
        for instruction in operations
    )


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
        facts = tuple(
            book.page(SOURCE_OPTIONAL_GLYPH_CARRY).latest(row)
            for row in book.page(SOURCE_OPTIONAL_GLYPH_CARRY).scope_rows(
                tuple(source_scope)
            )
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
        assert present_arm.presence_source == book.latest_ref(
            SOURCE_GLYPH_SEQUENCE,
            (tuple(source_scope), present_arm.value_id),
        )
        optional_cell = book.latest_ref(
            SOURCE_OPTIONAL_GLYPH_CARRY,
            (tuple(source_scope), fact.merged_value_id),
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
