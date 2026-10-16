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
from src.compiler.concordance_declarations import (
    SOURCE_SORT_CONTRACT, SourceSortContract,
)
from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm
from src.transmogrifier.graph.graph_express2 import ProcessGraph
from src.transmogrifier.ssa import (
    Function,
    SSARecordDescriptor,
    SSARecordFieldDescriptor,
    SSARecordFieldStorage,
    SSARecordTable,
    SSASequenceDescriptor,
    SSASequenceTable,
)


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

    _book, token = begin_identity_book()
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

    _book, token = begin_identity_book()
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
