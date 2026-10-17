import ast
import contextlib
import io
from pathlib import Path

import pytest

from src.common.tensors.topological_reducer import (
    reduce_abstract_tensor_topology,
)
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.fortran_c_shell import (
    _accepted_record_access_entries,
    _accepted_record_parameter_receipt,
    _accepted_record_sequence_members,
    _accepted_record_sequence_view,
    _authored_complete_record_schemas,
    _concordant_function_aliases,
    _concordant_function_resident,
    _field_slot_ops,
    _publish_concordant_function_aliases,
    _retire_concordant_sequence_aliases,
    _retire_concorded_record_identity_aliases,
    _retire_dead_planning_aliases,
    lower_ast_source_to_ssa,
)
from src.compiler.identity_concordance import (
    IdentityPage,
    begin_identity_book,
    concord_sequence_row_dtypes,
    commit_sequence_contract,
    commit_sequence_row_layout,
    committed_sequence_row_dtypes,
    committed_sequence_row_layout,
    committed_sequence_contract,
    end_identity_book,
    resolved_concordant_alias_bindings,
)
from src.compiler.ir_sequence_tables import (
    lower_sequence_contains,
    lower_sequence_equal,
)
from src.compiler.concordance_declarations import (
    SOURCE_SORT_CONTRACT, SourceSortContract,
)
from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm
from src.transmogrifier.graph.graph_express2 import ProcessGraph
from src.transmogrifier.ssa import (
    Function,
    SSAValue,
    SSARecordDescriptor,
    SSARecordFieldDescriptor,
    SSARecordFieldStorage,
    SSARecordTable,
    SSASequenceDescriptor,
    SSASequenceTable,
)


def test_single_column_resident_membership_scans_the_value_column():
    descriptor = SSASequenceDescriptor(
        10, (11,), 12, 13, column_dtypes=("int64",), key_columns=(),
    )

    lowering = lower_sequence_contains(
        descriptor, function_name="resident_string_contains",
        first_value_id=20,
    )

    function = lowering.functions[0]
    assert function.metadata["key_columns"] == ()
    assert function.metadata["membership_columns"] == (0,)
    assert len(function.args) == 4
    assert function.args[-1].dtype == "int64"


def test_keyed_membership_accepts_every_declared_key_component():
    descriptor = SSASequenceDescriptor(
        30, (31, 32, 33), 34, 35,
        column_dtypes=("int64", "bool", "float64"),
        key_columns=(0, 2),
    )

    lowering = lower_sequence_contains(
        descriptor, function_name="compound_key_contains",
        first_value_id=40,
    )

    function = lowering.functions[0]
    assert function.metadata["membership_columns"] == (0, 2)
    assert tuple(value.dtype for value in function.args[-2:]) == (
        "int64", "float64",
    )


def test_resident_sequence_equality_compares_length_and_contents():
    left = SSASequenceDescriptor(
        50, (51,), 52, 53, column_dtypes=("int64",), key_columns=(),
    )
    right = SSASequenceDescriptor(
        60, (61,), 62, 63, column_dtypes=("int64",), key_columns=(),
    )

    lowering = lower_sequence_equal(
        left, right, function_name="resident_string_equal",
        first_value_id=70,
    )

    function = lowering.functions[0]
    assert function.metadata["ssa_sequence_operation"] == "equal"
    assert function.metadata["left_sequence_id"] == 50
    assert function.metadata["right_sequence_id"] == 60
    operations = tuple(
        instruction.op
        for block in function.blocks.values()
        for instruction in block.instrs
    )
    assert operations.count("Load") == 4
    assert operations.count("Eq") == 2
    assert "LAnd" not in operations


def test_sort_policy_is_published_at_source_and_carried_through_relabel():
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        with contextlib.redirect_stdout(io.StringIO()):
            graph.build_from_ast(ast.parse(
                "def run():\n"
                "    rows = [(2, 3), (1, 4)]\n"
                "    rows.sort(key=lambda row: row[1], reverse=True)\n"
                "    return rows\n"
            ))
        reduce_abstract_tensor_topology(graph)
        function = graph.function_table.entry("run").graph
        call = next(
            data for _node_id, data in function.G.nodes(data=True)
            if (data.get("attributes") or {}).get(
                "source_sort_contract_row"
            ) is not None
        )
        row = call["attributes"]["source_sort_contract_row"]
        fact = book.page(SOURCE_SORT_CONTRACT).latest(row)
        assert fact == SourceSortContract(
            "sort_key_column", (), 1, True,
            destination_value_id=6,
        )
    finally:
        end_identity_book(token)


def test_sequence_contract_concordance_keeps_one_physical_row_contract():
    book, token = begin_identity_book()
    try:
        first = commit_sequence_contract(
            "kernel", 3, "unique", 2, False,
            source="authored Mapping[str, float]",
        )
        repeated = commit_sequence_contract(
            "kernel", 3, "unique", 2, True,
            source="SSA sequence declaration",
        )
        assert first.policy == repeated.policy == "unique"
        assert first.column_count == repeated.column_count == 2
        assert repeated.writable is True
        assert committed_sequence_contract("kernel", 3) == repeated
        assert len(book.page("sequence_contract_concordance").history(
            ("kernel", 3)
        )) == 2
    finally:
        end_identity_book(token)


def test_sequence_contract_concordance_refuses_width_redefinition():
    _book, token = begin_identity_book()
    try:
        commit_sequence_contract(
            "kernel", 3, "unique", 2, False,
            source="authored Mapping[str, float]",
        )
        with pytest.raises(ValueError, match=(
            "authored Mapping.*committed unique/2.*runtime.*duplicates/1"
        )):
            commit_sequence_contract(
                "kernel", 3, "duplicates", 1, False,
                source="runtime tuple view",
            )
    finally:
        end_identity_book(token)


def test_sequence_row_dtype_concordance_refines_equivalent_unknown_arena():
    book, token = begin_identity_book()
    try:
        assert concord_sequence_row_dtypes(
            "kernel", {29: ("float64",)}, source="declaration"
        ) == ("float64",)
        assert concord_sequence_row_dtypes(
            "kernel",
            {29: ("float64",), 33: ("unknown",)},
            source="conditional replace",
        ) == ("float64",)
        assert committed_sequence_row_dtypes(
            "kernel", 33
        ) == ("float64",)
        assert len(book.page(
            "sequence_row_dtype_concordance"
        ).history(("kernel", 33))) == 1
    finally:
        end_identity_book(token)


def test_sequence_row_dtype_concordance_refuses_known_disagreement():
    _book, token = begin_identity_book()
    try:
        with pytest.raises(ValueError, match="column 0"):
            concord_sequence_row_dtypes(
                "kernel",
                {29: ("float64",), 33: ("complex128",)},
                source="conditional replace",
            )
    finally:
        end_identity_book(token)


def test_alias_concordance_resolves_a_chain_split_across_pages():
    planning = IdentityPage("planning_value_concordance")
    planning.bind_alias("kernel", 440, 447)
    control = IdentityPage("control_value_concordance")
    control.bind_alias("kernel", 447, 56)

    assert resolved_concordant_alias_bindings(
        "kernel", control.alias_bindings("kernel"), page=planning,
    ) == {440: 56, 447: 56}


def test_alias_concordance_compares_terminal_residents():
    planning = IdentityPage("planning_value_concordance")
    planning.bind_alias("kernel", 277, 446)
    planning.bind_alias("kernel", 446, 56)

    assert resolved_concordant_alias_bindings(
        "kernel", {277: 56}, page=planning,
    ) == {277: 56, 446: 56}

    conflicting = IdentityPage("planning_value_concordance")
    conflicting.bind_alias("kernel", 277, 446)
    with pytest.raises(ValueError, match="terminal residents"):
        resolved_concordant_alias_bindings(
            "kernel", {277: 56}, page=conflicting,
        )


def test_function_alias_publication_commits_transitive_terminal_resident():
    function = Function("solve", [], {})
    function.metadata["value_aliases"] = {69: 58}

    book, token = begin_identity_book()
    try:
        page = book.page("planning_value_concordance")
        page.bind_alias("solve", 64, 58)
        page.bind_alias("solve", 69, 64)

        assert _concordant_function_aliases(function) == {64: 58, 69: 58}
        published = _publish_concordant_function_aliases(
            function, {71: 86}, provenance="test_new_alias",
        )

        assert published == {64: 58, 69: 58, 71: 86}
        assert function.metadata["value_aliases"] == published
        assert page.latest(("solve", 69)) == 58
    finally:
        end_identity_book(token)


def test_sequence_row_layout_is_one_direct_identity_book_row():
    book, token = begin_identity_book()
    try:
        scope = ("rows|shape", 0)
        committed = commit_sequence_row_layout(
            scope, 264, ((3,),), ("float64",),
            source="collection value edge",
        )

        assert committed_sequence_row_layout(scope, 264) == committed
        # A sibling copy's scope never reads this copy's layout.
        assert committed_sequence_row_layout(("rows|shape", 1), 264) is None
        assert book.page("sequence_row_layout_concordance").rows() == (
            (scope, 264),
        )
    finally:
        end_identity_book(token)


def test_concordant_sequence_descriptor_alias_retires_duplicate_members():
    book, token = begin_identity_book()
    try:
        function = Function("solve", [], {})
        sequences = SSASequenceTable(owner="solve", book=book)
        resident = SSASequenceDescriptor(10, (11,), 12, 13)
        duplicate = SSASequenceDescriptor(20, (21,), 22, 23)
        sequences.register(resident)
        sequences.register(duplicate)
        records = SSARecordTable(owner="solve", book=book)
        records.register(SSARecordDescriptor(30, "State", (
            SSARecordFieldDescriptor(
                "items", SSARecordFieldStorage.SEQUENCE,
                storage_identity="State.items", sequence_id=20,
            ),
        )))
        _publish_concordant_function_aliases(
            function, {20: 10, 21: 11, 22: 12, 23: 13},
            provenance="exact_sequence_argument",
        )

        assert _retire_concordant_sequence_aliases(
            {"solve": function}, {"solve": sequences}, {"solve": records},
        ) == 1
        assert sequences.by_id(20) is None
        assert sequences.by_id(10) == resident
        assert sequences.member_claims(11) == ((10, ("column", 0)),)
        field = records.records[30].fields[0]
        assert field.sequence_id == 10
    finally:
        end_identity_book(token)


def test_output_concordance_closes_nonphysical_record_planning_alias():
    book, token = begin_identity_book()
    try:
        function = Function("solve", [], {})
        records = SSARecordTable(owner="solve", book=book)
        for record_id in (1, 2, 3):
            records.register(SSARecordDescriptor(record_id, "State"))
        _publish_concordant_function_aliases(
            function, {2: 1}, provenance="linked_record_projection",
        )
        output_page = book.page("output_identity_concordance")
        output_page.bind_alias("solve", 1, 3)
        output_page.bind_alias("solve", 2, 3)

        assert _retire_concorded_record_identity_aliases(
            {"solve": function}, {"solve": records}, book,
        ) == 1
        assert function.metadata["value_aliases"] == {}
        planning = book.page("planning_value_concordance")
        assert planning.latest(("solve", 2)) is None
        assert planning.alias_bindings("solve") == {}
    finally:
        end_identity_book(token)


def test_finalization_closes_terminal_alias_with_no_surviving_identity():
    book, token = begin_identity_book()
    try:
        function = Function("solve", [], {})
        _publish_concordant_function_aliases(
            function, {20: 21}, provenance="provisional_aggregate_slot",
        )

        assert _retire_dead_planning_aliases(
            {"solve": function}, {}, {}, {}, book,
        ) == 1
        assert function.metadata["value_aliases"] == {}
        planning = book.page("planning_value_concordance")
        assert planning.latest(("solve", 20)) is None
        assert planning.alias_bindings("solve") == {}
    finally:
        end_identity_book(token)


def test_imported_function_aliases_use_the_accepted_component_book():
    owner, owner_token = begin_identity_book()
    try:
        planning = owner.page("planning_value_concordance")
        planning.bind_alias("imported", 5, 7)
        outputs = owner.page("output_identity_concordance")
        outputs.bind_alias("imported", 3, 7)
    finally:
        end_identity_book(owner_token)

    function = Function("imported", [], {})
    function.metadata.update({
        "value_aliases": ((5, 7),),
        "output_identity_aliases": ((3, 7),),
        "accepted_ssa_component": {"identity_book": owner},
    })

    ambient, ambient_token = begin_identity_book()
    try:
        assert _concordant_function_aliases(
            function, include_output_identities=True,
        ) == {3: 7, 5: 7}
        assert not ambient.page(
            "output_identity_concordance"
        ).scope_rows("imported")
    finally:
        end_identity_book(ambient_token)


def test_imported_record_demand_reads_only_the_accepted_owner_book():
    from src.compiler.identity_concordance import IdentityBook

    owner = IdentityBook()
    scope = ("record_field_access", 0)
    symbol = "LiveScopeProjection_excludes__excludes"
    parameter_key = (symbol, "self")
    field_key = (
        parameter_key, "excluded", "LiveScopeProjection.excluded",
    )
    read_path = (("source-node", 4),)
    sequence_path = ((
        "local-sequence-view", symbol, 1, 4,
    ),)
    sequence_view = (("opaque_ref",), 1, (0,), False, "fixed", False)
    owner.page("record_parameter_value").concord(
        (scope, (symbol, 1)), parameter_key,
    )
    owner.page("record_field_access").concord(
        (scope, field_key), frozenset({"read", "sequence"}),
    )
    owner.page("record_field_access_path").concord(
        (scope, field_key, "read"), read_path,
    )
    owner.page("record_field_access_path").concord(
        (scope, field_key, "sequence"), sequence_path,
    )
    owner.page("record_field_sequence_view").concord(
        (scope, "LiveScopeProjection.excluded"), sequence_view,
    )
    sequence_scope = (symbol, 0)
    owner.page("sequence_descriptor").concord(
        (sequence_scope, 4), ("accepted-descriptor", 4),
    )
    for value_id, roles in {
        4: ((4, ("column", 0)), (4, ("handle",))),
        10: ((4, ("length_address_id",)),),
        11: ((4, ("capacity_value_id",)),),
        12: ((4, ("status_address_id",)),),
        13: ((4, ("live_flags_value_id",)),),
    }.items():
        owner.page("sequence_member").concord(
            (sequence_scope, value_id), roles,
        )
    function = Function(
        symbol, [
            SSAValue(4, "opaque_ref", shape=()),
            SSAValue(10, "int64", shape=(1,)),
            SSAValue(11, "int64", shape=()),
            SSAValue(12, "int", shape=(1,)),
            SSAValue(13, "bool", shape=(1,)),
        ], {},
        metadata={"accepted_ssa_component": {"identity_book": owner}},
    )

    ambient, ambient_token = begin_identity_book()
    try:
        key, parameter_owner, parameter_scope = (
            _accepted_record_parameter_receipt(function, symbol, 1)
        )
        assert (key, parameter_owner, parameter_scope) == (
            parameter_key, owner, scope,
        )
        assert _accepted_record_access_entries(function) == ((
            field_key, frozenset({"read", "sequence"}), owner, scope,
        ),)
        assert parameter_owner.page("record_field_access_path").latest(
            (parameter_scope, field_key, "read")
        ) == read_path
        assert parameter_owner.page("record_field_access_path").latest(
            (parameter_scope, field_key, "sequence")
        ) == sequence_path
        assert _accepted_record_sequence_view(
            function, "LiveScopeProjection.excluded",
        ) == sequence_view
        assert _accepted_record_sequence_members(
            function, "LiveScopeProjection.excluded",
        ) == (
            (("capacity_value_id",), 11, "int64", ()),
            (("column", 0), 4, "opaque_ref", ()),
            (("handle",), 4, "opaque_ref", ()),
            (("length_address_id",), 10, "int64", (1,)),
            (("live_flags_value_id",), 13, "bool", (1,)),
            (("status_address_id",), 12, "int", (1,)),
        )
        assert all(
            not ambient.page(page_name).rows()
            for page_name in (
                "record_parameter_value", "record_field_access",
                "record_field_access_path", "record_field_sequence_view",
                "sequence_descriptor", "sequence_member",
            )
        )
    finally:
        end_identity_book(ambient_token)


def test_imported_output_alias_audit_uses_accepted_component_book():
    from src.compiler.identity_concordance import CorrelationTable, IdentityBook
    from src.transmogrifier.ssa import IRModule

    owner = IdentityBook()
    owner.page("output_identity_concordance").bind_alias(
        "imported", 40, 2305843010213693979,
    )
    function = Function(
        "imported", [], {},
        metadata={
            "output_identity_aliases": ((40, 2305843010213693979),),
            "accepted_ssa_component": {"identity_book": owner},
        },
    )
    module = IRModule({"imported": function})
    module.metadata["identity_book"] = IdentityBook()

    findings = CorrelationTable.build(module).findings(module)

    assert not tuple(
        finding for finding in findings
        if finding.kind == "alias-not-concorded"
    )


def test_sequence_alias_with_live_length_reads_rebinds_all_descriptor_members(tmp_path):
    from src.transmogrifier.ssa import SSAValue, BasicBlock, Instr
    from src.compiler.ssa_llvm_backend import compile_artifact, prepare_artifact_execution
    import numpy as np
    book, token = begin_identity_book()
    try:
        resident = SSASequenceDescriptor(10, (10,), 12, 13, column_dtypes=('int64',))
        duplicate = SSASequenceDescriptor(20, (10,), 22, 23, column_dtypes=('int64',))
        values = {i: SSAValue(i, 'int64') for i in (10, 12, 13, 22, 23)}
        output = SSAValue(30, 'int64')
        call = Instr('Call', [values[10], values[22], values[23]], output,
                     attributes={'callee': 'consumer'})
        function = Function('solve', list(values.values()), {
            'entry': BasicBlock('entry', [call, Instr('Ret', [output], None)])})
        sequences = SSASequenceTable(owner='solve', book=book)
        sequences.register(resident)
        sequences.register(duplicate)
        _publish_concordant_function_aliases(function, {20: 10}, provenance='exact_handle')
        assert _retire_concordant_sequence_aliases(
            {'solve': function}, {'solve': sequences}, {},
        ) == 1
        assert [v.id for v in call.args] == [10, 12, 13]
        assert sequences.by_id(20) is None
        assert {v.id for v in function.args} == {10, 12, 13}
        zero, address, length = SSAValue(31, 'int64'), SSAValue(32, 'ptr'), SSAValue(33, 'int64')
        consumer = Function('consumer', [SSAValue(10, 'int64'), SSAValue(22, 'int64', (1,)),
                                         SSAValue(23, 'int64')], {
            'entry': BasicBlock('entry', [Instr('Const', [], zero, attributes={'value': 0}),
                Instr('GetElementPtr', [SSAValue(22, 'int64', (1,)), zero], address),
                Instr('Load', [address], length), Instr('Ret', [length], None)])})
        from src.transmogrifier.ssa import IRModule
        module = IRModule({'solve': function, 'consumer': consumer}, sequence_tables={
            'solve': sequences, 'consumer': SSASequenceTable({20: duplicate}),
        })
        artifact = emit_ssa_function_to_llvm(module, 'solve')
        assert artifact.complete, artifact.shortfalls
        native = compile_artifact(artifact, directory=tmp_path / 'native')
        for count in (0, 3, 7):
            execution = prepare_artifact_execution(native, {
                10: np.arange(8, dtype=np.int64), 12: np.array([count], dtype=np.int64),
                13: np.array([8], dtype=np.int64),
            }).run()
            assert execution.buffers[30].item() == count
    finally:
        end_identity_book(token)


def test_finalization_does_not_mistake_callee_output_identity_for_local_alias():
    from src.transmogrifier.ssa import SSAValue, BasicBlock, Instr
    book, token = begin_identity_book()
    try:
        output = SSAValue(1040, 'float64', (22,))
        producer = Function('producer', [output], {
            'entry': BasicBlock('entry', [Instr('Ret', [output], None)])})
        projected = SSAValue(1041, 'float64', (22,), accounting={
            'ssa_call_result_from': ('producer', 1040),
        })
        function = Function('solve', [], {'entry': BasicBlock('entry', [
            Instr('Load', [SSAValue(99, 'ptr')], projected),
            Instr('Ret', [projected], None)])})
        _publish_concordant_function_aliases(function, {1041: 1040}, provenance='region_output')
        assert _retire_dead_planning_aliases(
            {'solve': function, 'producer': producer}, {}, {}, {}, book,
        ) == 1
        assert function.metadata['value_aliases'] == {}
        assert function.blocks['entry'].instrs[-1].args == [projected]
    finally:
        end_identity_book(token)


def test_output_settlement_keeps_frame_arena_as_physical_resident():
    function = Function("get_state", [], {})
    function.metadata["value_aliases"] = {
        **{value_id: 2 for value_id in range(50, 60)},
        60: 2,
    }

    book, token = begin_identity_book()
    try:
        page = book.page("planning_value_concordance")
        for alias, resident in function.metadata["value_aliases"].items():
            page.bind_alias("get_state", alias, resident)

        assert _concordant_function_resident(function, 60) == 2
        published = _publish_concordant_function_aliases(
            function,
            {value_id: 2 for value_id in range(50, 61) if value_id != 2},
            provenance="test_frame_arena",
        )
        assert published[60] == 2
        assert page.latest(("get_state", 60)) == 2
        assert page.latest(("get_state", 2)) is None
    finally:
        end_identity_book(token)


def test_authored_mapping_contract_governs_specialized_parameter_view():
    # The test is the ingestion: the graph is built and reduced on the book
    # ``_field_slot_ops`` then reads its parameter rows from.
    _book, token = begin_identity_book()
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(ast.parse("""
from typing import Mapping

def kernel(values: Mapping[str, float]):
    return tuple(values.items())
"""))
    reduce_abstract_tensor_topology(graph)
    executable = graph.function_table.entry("kernel").graph.G
    parameter_id = int(executable.graph["identity_table"]["values"][0])
    parameter = executable.nodes[parameter_id]
    parameter.setdefault("attributes", {}).update({
        "binding_name": "values",
        "binding_kind": "parameter",
        "aggregate_kind": "tuple",
        "sequence_column_count": 1,
        "sequence_writable": False,
    })

    try:
        declarations = _field_slot_ops(
            executable, contract_scope="kernel-specialized"
        )[8]
        assert (
            parameter_id, "unique", 2, False
        ) in declarations
    finally:
        end_identity_book(token)


def test_annotated_list_slice_iteration_does_not_become_a_keyed_table():
    # The test is the ingestion: the graph is built and reduced on the book
    # ``_field_slot_ops`` then reads its parameter rows from.
    _book, token = begin_identity_book()
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(ast.parse("""
from typing import Any

def product(factors: list[Any]):
    held = factors[0]
    for factor in factors[1:]:
        held = factor
    return held
"""))
    reduce_abstract_tensor_topology(graph)
    executable = graph.function_table.entry("product").graph.G
    parameter_id = int(executable.graph["identity_table"]["factors"][0])

    try:
        declarations = _field_slot_ops(
            executable, contract_scope="product"
        )[8]
        parameter_declarations = tuple(
            declaration
            for declaration in declarations
            if int(declaration[0]) == parameter_id
        )
        assert parameter_declarations
        assert {
            (policy, int(columns))
            for _sequence_id, policy, columns, _writable
            in parameter_declarations
        } == {("duplicates", 1)}
    finally:
        end_identity_book(token)


def test_source_record_schema_resolves_native_tensor_to_abstract_tensor():
    schemas = _authored_complete_record_schemas(ast.parse("""
from dataclasses import dataclass
from typing import Mapping
import torch

@dataclass(frozen=True)
class Reading:
    voltage_v: Mapping[str, torch.Tensor]
"""))

    field = schemas["Reading"]["fields"]["voltage_v"]
    assert field["value_tensor"] is True
    assert field["value_python_type"] == "AbstractTensor"
    assert "value_record" not in field


def test_source_record_schema_resolves_direct_native_tensor_to_span():
    schemas = _authored_complete_record_schemas(ast.parse("""
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class Reading:
    values: torch.Tensor
"""))

    field = schemas["Reading"]["fields"]["values"]
    assert field == {
        "storage": "span",
        "dtype": "unknown",
        "rank": 1,
        "mutable": False,
        "python_type": "AbstractTensor",
    }


def test_frozenset_and_nested_set_membership_keep_authored_resident_contracts():
    module, _outputs, exports = lower_ast_source_to_ssa(
        """
from dataclasses import dataclass, field

READ_THROUGH_EXCLUDED_PAGES: frozenset[str] = frozenset({
    "scope_origin", "planner_specialization",
})

@dataclass
class Registry:
    private_pages: set[str] = field(default_factory=set)

@dataclass
class Book:
    registry: Registry

def excluded(book: Book, name: str):
    return (
        name in READ_THROUGH_EXCLUDED_PAGES
        or name in book.registry.private_pages
    )
""",
        "excluded",
        name="nested_set_membership_contract",
        extraction_contract=(
            Path(__file__).resolve().parents[1]
            / "extraction_contracts"
            / "program_extraction.yaml"
        ),
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _message: None,
    )

    assert exports == ("nested_set_membership_contract__excluded",)
    root = module.functions[exports[0]]
    contains = tuple(
        instruction
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get("ssa_sequence_operation") == "contains"
    )
    assert len(contains) == 2
    table = module.sequence_tables[root.name]
    for instruction in contains:
        descriptor = table.by_id(int(instruction.attributes["sequence_id"]))
        assert descriptor.column_dtypes == ("int64",)
        assert descriptor.key_columns == (0,)


def test_record_frozenset_any_membership_keeps_reference_key_contract():
    from src.compiler.identity_concordance import CorrelationTable

    module, _outputs, exports = lower_ast_source_to_ssa(
        """
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class Projection:
    excluded: frozenset[Any] = frozenset()
    universe: frozenset[Any] = frozenset()
    held: frozenset[Any] = frozenset()

    def excludes(self, node: Any) -> bool:
        return node in self.excluded or (
            node in self.universe and node not in self.held
        )
""",
        name="record_frozenset_reference_key_contract",
        extraction_contract=(
            Path(__file__).resolve().parents[1]
            / "extraction_contracts"
            / "program_extraction.yaml"
        ),
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _message: None,
    )

    root = module.functions[exports[0]]
    record = root.metadata["parameter_record_abi"]["self"]
    assert {
        name: (field["storage"], field["dtype"], field["aggregate_kind"])
        for name, field in record["fields"].items()
    } == {
        "excluded": ("span", "opaque_ref", "frozenset"),
        "universe": ("span", "opaque_ref", "frozenset"),
        "held": ("span", "opaque_ref", "frozenset"),
    }
    assert len(module.sequence_tables[root.name].sequences) == 3
    assert {
        int(sequence_id): (
            descriptor.column_dtypes, descriptor.key_columns,
        )
        for sequence_id, descriptor
        in module.sequence_tables[root.name].sequences.items()
    } == {
        # The sequence arena and the queried node retain the same fixed-width
        # opaque identity-handle ABI.
        int(sequence_id): (("opaque_ref",), (0,))
        for sequence_id
        in module.sequence_tables[root.name].sequences
    }
    record_table = module.record_tables[root.name]
    descriptor = next(
        descriptor
        for descriptor in record_table.records.values()
        if descriptor.identity == "Projection"
    )
    physical_members = {
        int(value_id)
        for field in descriptor.fields
        for value_id in field.value_ids
    }
    arguments = {int(value.id): value for value in root.args}
    assert physical_members
    assert physical_members <= set(arguments)
    assert all(
        (arguments[value_id].accounting or {}).get(
            "program_abi_parameter"
        ) == "self"
        for value_id in physical_members
    )
    owned = {
        int(value.id)
        for value in root.args
        if (value.accounting or {}).get("program_abi_parameter") == "self"
    }
    exact_record_bindings = [
        int(record_id)
        for record_id, candidate in record_table.records.items()
        if candidate.identity == record["identity"]
        and {
            int(value_id)
            for field in candidate.fields
            for value_id in field.value_ids
        }
        and {
            int(value_id)
            for field in candidate.fields
            for value_id in field.value_ids
        } <= owned
    ]
    assert exact_record_bindings == [int(descriptor.record_id)]
    findings = CorrelationTable.build(module).findings(module)
    assert not [
        finding
        for finding in findings
        if finding.kind == "key-column-not-integral"
    ]
    unsourced_record_findings = [
        finding
        for finding in findings
        if finding.kind == "unsourced-fact"
        and finding.function in {
            "record_descriptor",
            "record_member",
            "record_field_access",
            "record_field_access_path",
            "record_field_sequence_view",
        }
    ]
    assert not unsourced_record_findings, unsourced_record_findings


def test_any_scalar_parameter_keeps_concrete_reference_transport():
    module, _outputs, exports = lower_ast_source_to_ssa(
        """
from typing import Any

def same(left: Any, right: Any):
    return left == right
""",
        "same",
        name="any_reference_transport",
        extraction_contract=(
            Path(__file__).resolve().parents[1]
            / "extraction_contracts"
            / "program_extraction.yaml"
        ),
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _message: None,
    )

    root = module.functions[exports[0]]
    authored = {
        int(value.id): str(value.dtype)
        for value in root.args
        if int(value.id) in {0, 1}
    }
    assert authored == {0: "opaque_ref", 1: "opaque_ref"}


def test_authored_nested_mapping_value_sort_keeps_projected_target_dtype():
    module, _outputs, _exports = lower_ast_source_to_ssa(
        """
from dataclasses import dataclass, field
from typing import Any

@dataclass
class Page:
    column_positions: dict[int, int] = field(default_factory=dict)
    row_columns: dict[Any, list[int]] = field(default_factory=dict)

    def reorder(self):
        for resident in self.row_columns.values():
            resident.sort(key=self.column_positions.__getitem__)
""",
        name="nested_value_sort_contract",
        extraction_contract=(
            Path(__file__).resolve().parents[1]
            / "extraction_contracts"
            / "program_extraction.yaml"
        ),
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _message: None,
    )

    root = next(
        function for function in module.functions.values()
        if any(
            instruction.attributes.get("ssa_sequence_operation") == "sort"
            for block in function.blocks.values()
            for instruction in block.instrs
        )
    )
    sort_call = next(
        instruction
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get("ssa_sequence_operation") == "sort"
    )
    sorted_sequence = module.sequence_tables[root.name].by_id(
        int(sort_call.attributes["sequence_id"])
    )
    assert sorted_sequence.column_dtypes == ("int64",)


def test_enumerate_element_projection_none_lowers_in_keyed_reindex():
    # ``enumerate(self.columns)`` binds its element with projection None
    # (the source's only column); lowering the loop must not int() it.
    module, _outputs, _exports = lower_ast_source_to_ssa(
        """
from dataclasses import dataclass, field
from typing import Any

@dataclass
class Page:
    cells: dict[tuple[Any, int], Any] = field(default_factory=dict)
    columns: list[int] = field(default_factory=list)
    column_positions: dict[int, int] = field(default_factory=dict)
    row_columns: dict[Any, list[int]] = field(default_factory=dict)

    def reindex(self):
        self.column_positions = {
            column: position for position, column in enumerate(self.columns)
        }
        self.row_columns = {}
        for row, column in self.cells:
            if column not in self.column_positions:
                self.column_positions[column] = len(self.columns)
                self.columns.append(column)
            self.row_columns.setdefault(row, []).append(column)
        for columns in self.row_columns.values():
            columns.sort(key=self.column_positions.__getitem__)
""",
        name="enumerate_none_projection_contract",
        extraction_contract=(
            Path(__file__).resolve().parents[1]
            / "extraction_contracts"
            / "program_extraction.yaml"
        ),
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _message: None,
    )
    assert module.functions


def test_mapping_values_generator_retains_record_and_nested_span_contracts():
    module, _outputs, exports = lower_ast_source_to_ssa(
        """
from dataclasses import dataclass
from itertools import chain
from typing import Any, Iterable

@dataclass(frozen=True)
class Segment:
    kind: str
    owner: str
    offset: int
    stored: int
    raw: int
    cells: int
    rows: int
    ordinals: Iterable[int] | None = None

def detached_ordinals(page: Any) -> list:
    detached: dict[str, Segment] | None = page.__dict__.get("_detached")
    if not detached:
        return []
    return sorted(chain.from_iterable(
        segment.ordinals for segment in detached.values()
    ))
""",
        "detached_ordinals",
        name="mapping_value_record_generator",
        extraction_contract=(
            Path(__file__).resolve().parents[1]
            / "extraction_contracts"
            / "program_extraction.yaml"
        ),
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _message: None,
    )

    root = module.functions[exports[0]]
    table = module.sequence_tables[root.name]
    values = table.by_id(11)
    ordinals = table.by_id(9)
    assert values is not None
    assert values.column_dtypes == (
        "int64", "int64", "int64", "int64", "int64", "int64", "int64",
        "bool", "int64",
    )
    assert ordinals is not None
    assert ordinals.column_dtypes == ("int64",)
    assert not ordinals.writable


def test_authored_tuple_loop_row_remains_integer_table_key_handle():
    module, _outputs, _exports = lower_ast_source_to_ssa(
        """
from dataclasses import dataclass, field
from typing import Any

@dataclass
class IdentityPage:
    cells: dict[tuple[Any, int], Any] = field(default_factory=dict)
    row_columns: dict[tuple, list[int]] = field(default_factory=dict)
    scopes: dict[Any, dict[tuple, None]] = field(default_factory=dict)

def reindex(page: Any):
    state = page.__dict__
    row_columns: dict[tuple, list[int]] = state["row_columns"]
    scopes: dict[Any, dict[tuple, None]] = state["scopes"]
    row: tuple
    column: int
    for row, column in state["cells"]:
        row_columns.setdefault(row, []).append(column)
        scopes.setdefault(row[0], {}).setdefault(row, None)

def replace(page: IdentityPage):
    reindex(page)
""",
        "replace",
        name="tuple_loop_row_handle_contract",
        extraction_contract=(
            Path(__file__).resolve().parents[1]
            / "extraction_contracts"
            / "program_extraction.yaml"
        ),
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _message: None,
    )

    root = next(
        function for function in module.functions.values()
        if function.name.endswith("__reindex")
    )
    table_store = next(
        instruction
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get("ssa_sequence_operation")
        == "table_store"
        and module.sequence_tables[root.name].by_id(
            int(instruction.attributes["sequence_id"])
        ).column_dtypes == ("int64", "int64")
    )
    descriptor = module.sequence_tables[root.name].by_id(
        int(table_store.attributes["sequence_id"])
    )
    assert descriptor.column_dtypes == ("int64", "int64")
    assert tuple(value.dtype for value in table_store.args[-2:]) == (
        "int64", "int64",
    )


def test_reassigned_keyed_field_rebinds_to_declared_parts_and_dict_predicate():
    from src.compiler.project_compilation_product import (
        _unexplained_root_argument_ids,
    )

    module, _outputs, exports = lower_ast_source_to_ssa(
        """
from dataclasses import dataclass, field
from typing import Any

@dataclass
class Page:
    columns: list[int] = field(default_factory=list)
    cells: dict[tuple[Any, int], Any] = field(default_factory=dict)
    column_positions: dict[int, int] = field(default_factory=dict)
    row_columns: dict[Any, list[int]] = field(default_factory=dict)

    def reindex(self):
        if self.__dict__.get("_compressed_ram"):
            self.columns.append(0)
        self.column_positions = {
            column: position for position, column in enumerate(self.columns)
        }
        self.row_columns = {}
        for row, column in self.cells:
            if column not in self.column_positions:
                self.column_positions[column] = len(self.columns)
                self.columns.append(column)
            self.row_columns.setdefault(row, []).append(column)
        for resident in self.row_columns.values():
            resident.sort(key=self.column_positions.__getitem__)
""",
        name="reassigned_keyed_field_contract",
        extraction_contract=(
            Path(__file__).resolve().parents[1]
            / "extraction_contracts"
            / "program_extraction.yaml"
        ),
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _message: None,
    )

    root = module.functions[exports[0]]
    assert _unexplained_root_argument_ids(root) == ()
    truth_id, source_name, transform = next(
        row for row in root.metadata["scalar_source_transforms"]
        if row[2] == "dict_truth:_compressed_ram"
    )
    assert source_name == "self"
    assert next(
        value for value in root.args if int(value.id) == int(truth_id)
    ).dtype == "bool"
    keyed_parts = {
        str((value.accounting or {}).get("program_abi_keyed_part")): int(value.id)
        for value in root.args
        if (value.accounting or {}).get("program_abi_keyed_owner")
        == "column_positions"
    }
    assert set(keyed_parts) >= {"length", "keys", "values"}
    keyed_calls = tuple(
        instruction
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get("ssa_sequence_operation")
        in {"contains", "table_store"}
        and instruction.args
        and int(instruction.args[0].id) == keyed_parts["keys"]
    )
    assert keyed_calls
    for instruction in keyed_calls:
        assert tuple(int(value.id) for value in instruction.args[:4]) == (
            keyed_parts["keys"],
            keyed_parts["values"],
            keyed_parts["length"],
            keyed_parts["length"],
        )
    assert not any(
        str(value.dtype or "unknown") == "unknown"
        for value in root.args
    )


def test_keyed_tensor_field_commits_nested_span_sequence_contract():
    module, _outputs, _exports = lower_ast_source_to_ssa(
        """
from dataclasses import dataclass
from typing import Mapping
import torch

@dataclass(frozen=True)
class Reading:
    voltage_v: Mapping[str, torch.Tensor]

def sample(reading: Reading, key: str):
    return reading.voltage_v[key]
""",
        "sample",
        name="keyed_tensor_contract",
        extraction_contract=(
            Path(__file__).resolve().parents[1]
            / "extraction_contracts"
            / "program_extraction.yaml"
        ),
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _message: None,
    )

    root = module.functions["keyed_tensor_contract__sample"]
    descriptor = next(
        item
        for item in module.sequence_tables[root.name].sequences.values()
        if item.key_columns
    )
    assert descriptor.child_table_pool is not None
    pool = descriptor.child_table_pool
    assert pool.shape_value_id is not None
    assert pool.rank_value_id is not None
    assert pool.shape_stride_value_id is not None
    assert len({
        pool.length_value_id,
        pool.shape_value_id,
        pool.rank_value_id,
        pool.shape_stride_value_id,
    }) == 4
    assert descriptor.child_table_pool.handle_column == 1
    assert descriptor.column_dtypes == ("int64", "int64")
    assert descriptor.child_table_pool.column_dtypes == ("unknown",)
    assert descriptor.child_table_pool.key_columns == ()


def test_keyed_tensor_lookup_feeds_tensor_property_region_as_span():
    module, _outputs, _exports = lower_ast_source_to_ssa(
        """
from dataclasses import dataclass
from typing import Mapping
import torch

@dataclass(frozen=True)
class Reading:
    voltage_v: Mapping[str, torch.Tensor]

def sample(reading: Reading, key: str):
    selected = reading.voltage_v[key]
    return selected.real.sum()
""",
        "sample",
        name="keyed_tensor_region",
        extraction_contract=(
            Path(__file__).resolve().parents[1]
            / "extraction_contracts"
            / "program_extraction.yaml"
        ),
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _message: None,
    )

    root = module.functions["keyed_tensor_region__sample"]
    row_base = next(
        instruction.res
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get("binding") == "keyed_tensor_row_base"
    )
    region_call = next(
        instruction
        for block in root.blocks.values()
        for instruction in block.instrs
        if module.functions.get(str(instruction.attributes.get("callee") or "")) is not None
        and module.functions[str(instruction.attributes["callee"])].metadata.get("source_region_integral")
        and any(int(value.id) == int(row_base.id) for value in instruction.args)
    )
    assert row_base.id in {value.id for value in region_call.args}
    # Region numbers are planner-local, not an API. Dynamic span consumers
    # also receive their exact shape/rank operands through this call edge.
    assert {int(row_base.accounting[key]) for key in (
        'tensor_shape_value_id', 'tensor_rank_value_id',
    )} <= {int(value.id) for value in region_call.args}
    assert row_base.shape == ()
    assert row_base.accounting["program_abi_storage"] == "span"
    assert row_base.accounting["program_abi_rank"] == 1
    assert row_base.accounting["tensor_metadata_state"] == "dynamic"
    produced_ids = {
        int(instruction.res.id)
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.res is not None
    }
    assert {
        int(row_base.accounting["tensor_shape_value_id"]),
        int(row_base.accounting["tensor_rank_value_id"]),
        int(row_base.accounting["tensor_element_count_value_id"]),
    } <= produced_ids
    sequence = next(
        descriptor
        for descriptor in module.sequence_tables[root.name].sequences.values()
        if descriptor.child_table_pool is not None
    )
    pool = sequence.child_table_pool
    assert pool is not None
    shape_address = next(
        instruction
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get("binding") == "keyed_tensor_row_shape"
    )
    rank_address = next(
        instruction
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get("binding")
        == "keyed_tensor_row_rank_address"
    )
    assert int(shape_address.args[0].id) == pool.shape_value_id
    assert int(rank_address.args[0].id) == pool.rank_value_id
    assert int(row_base.accounting["tensor_shape_value_id"]) == int(
        shape_address.res.id
    )

    region = module.functions[region_call.attributes["callee"]]
    descriptor = module.tensor_tables[region.name].by_id(row_base.id)
    assert descriptor is not None
    assert descriptor.metadata_state == "dynamic"
    assert descriptor.shape_value_id == row_base.accounting[
        "tensor_shape_value_id"
    ]
    assert descriptor.rank_value_id == row_base.accounting[
        "tensor_rank_value_id"
    ]
    assert descriptor.element_count_value_id == row_base.accounting[
        "tensor_element_count_value_id"
    ]
    numerical = [
        instruction
        for block in region.blocks.values()
        for instruction in block.instrs
        if instruction.op not in {"Ret", "ret"}
    ]
    real = next(instruction for instruction in numerical if instruction.op == "real")
    assert real.args[0].accounting["program_abi_rank"] == 1
    assert real.args[0].accounting["program_abi_storage"] == "span"

    lookup = next(
        instruction
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get("ssa_sequence_operation") == "lookup"
    )
    helper = module.functions[lookup.attributes["callee"]]
    assert lookup.args[0].dtype == "int64"
    assert helper.args[0].dtype == "int64"
    helper_descriptor = module.sequence_tables[helper.name].by_id(
        int(helper.metadata["sequence_id"])
    )
    assert helper_descriptor.column_dtypes[0] == "int64"


def test_keyed_rank_two_tensor_index_uses_concorded_runtime_shape():
    module, _outputs, _exports = lower_ast_source_to_ssa(
        """
from dataclasses import dataclass
from typing import Mapping
import torch

@dataclass(frozen=True)
class Reading:
    values: Mapping[str, torch.Tensor]

def sample(reading: Reading, key: str, index: int):
    selected = reading.values[key]
    return selected[:, index]
""",
        "sample",
        name="keyed_tensor_rank_two_index",
        extraction_contract=(
            Path(__file__).resolve().parents[1]
            / "extraction_contracts"
            / "program_extraction.yaml"
        ),
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _message: None,
    )

    root = module.functions["keyed_tensor_rank_two_index__sample"]
    row_base = next(
        instruction.res
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get("binding") == "keyed_tensor_row_base"
    )
    region_call = next(
        instruction
        for block in root.blocks.values()
        for instruction in block.instrs
        if module.functions.get(str(instruction.attributes.get("callee") or "")) is not None
        and module.functions[str(instruction.attributes["callee"])].metadata.get("source_region_integral")
        and any(int(value.id) == int(row_base.id) for value in instruction.args)
    )
    region = module.functions[region_call.attributes["callee"]]
    shape_id = int(row_base.accounting["tensor_shape_value_id"])
    rank_id = int(row_base.accounting["tensor_rank_value_id"])
    assert row_base.accounting["program_abi_rank"] == 2
    assert {shape_id, rank_id} <= {
        int(value.id) for value in region.args
    }
    assert {shape_id, rank_id} <= {
        int(value.id) for value in region_call.args
    }

    instructions = [
        instruction
        for block in region.blocks.values()
        for instruction in block.instrs
    ]
    selection = next(
        instruction for instruction in instructions
        if instruction.attributes.get("callee") == "index_select_double"
    )
    assert int(selection.args[2].id) == shape_id
    assert int(selection.args[3].id) == rank_id
    assert not any(
        instruction.op in {"Indexed", "indexed"}
        or isinstance(instruction.attributes.get("value"), slice)
        for instruction in instructions
    )
    assert any(
        instruction.attributes.get("binding") == "dynamic-index-vector"
        for instruction in instructions
    )

    page = module.metadata["identity_book"].page(
        "tensor_shape_concordance"
    )
    contract = page.latest((
        root.metadata["tensor_shape_concordance_scope"],
        int(row_base.id),
    ))
    assert contract["source"] == "control-keyed-tensor-lookup"
    assert contract["program_abi_rank"] == 2
    assert int(contract["tensor_shape_value_id"]) == shape_id
    assert int(contract["tensor_rank_value_id"]) == rank_id
    emitted = emit_ssa_function_to_llvm(
        module,
        root.name,
        entry_name="keyed_tensor_rank_two_index_sample",
    )
    assert emitted.shortfalls == ()


def test_source_record_schema_keeps_resolved_mapping_value_record():
    schemas = _authored_complete_record_schemas(ast.parse("""
from dataclasses import dataclass
from typing import Mapping

@dataclass(frozen=True)
class Sample:
    value: float

@dataclass(frozen=True)
class Reading:
    samples: Mapping[str, Sample]
"""))

    assert schemas["Reading"]["fields"]["samples"]["value_record"] == "Sample"


def test_source_record_schema_keeps_record_sequence_row_identity():
    schemas = _authored_complete_record_schemas(ast.parse("""
from dataclasses import dataclass

@dataclass(frozen=True)
class Claim:
    key: str

@dataclass(frozen=True)
class ValueRow:
    claims: list[Claim]
"""))

    assert schemas["ValueRow"]["fields"]["claims"] == {
        "storage": "table",
        "rank": 1,
        "mutable": True,
        "aggregate_kind": "list",
        "row_record": "Claim",
    }
