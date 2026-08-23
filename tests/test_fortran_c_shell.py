from __future__ import annotations

import ctypes
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.compiler.compiled_program_api import (
    CompiledProgramAPI,
    EntryPoint,
    Parameter,
)
from src.compiler.fortran_c_shell import (
    compile_ast_fortran_c_shell,
    compile_fortran_module_c_shell,
)
from src.compiler.fortran_c_shell import emit_fortran_c_shell_source
from src.compiler.ssa_fortran_backend import FortranModule, fortran_compiler
from src.compiler.ssa_fortran_backend import FortranEmissionError
from src.compiler.ssa_fortran_backend import compile_module
from src.compiler.shell_io import (
    ShellIOManifest, ShellIORequest, SystemPort, attach_shell_io,
)


def test_pure_region_dce_preserves_an_exact_required_source_feed():
    from src.compiler.ir_identities import drop_dead_pure_region_calls
    from src.transmogrifier.ssa import BasicBlock, Function, Instr, SSAValue

    region_value = SSAValue(131, "int64")
    region = Function("probe__planned_region_13", [], {
        "entry": BasicBlock("entry", [
            Instr("Const", [], region_value, attributes={"value": 7}),
            Instr("Ret", [region_value], None),
        ]),
    })
    aggregate = SSAValue(200)
    index = SSAValue(201, "int64")
    pointer = SSAValue(202, "ptr")
    projected = SSAValue(131, "int64")
    caller = Function("probe", [], {
        "entry": BasicBlock("entry", [
            Instr(
                "Call", [], aggregate,
                attributes={
                    "callee": region.name,
                    "result_convention": "ssa.aggregate",
                },
            ),
            Instr("Const", [], index, attributes={"value": 0}),
            Instr("GetElementPtr", [aggregate, index], pointer),
            Instr("Load", [pointer], projected),
        ]),
    }, metadata={"required_source_value_ids": (131,)})
    functions = {caller.name: caller, region.name: region}

    assert drop_dead_pure_region_calls(functions) == 0
    assert region.name in functions
    assert any(
        instruction.res is projected
        for instruction in caller.blocks["entry"].instrs
    )


def test_pure_region_dce_discards_only_an_outputless_uncalled_integral():
    from src.compiler.ir_identities import drop_dead_pure_region_calls
    from src.transmogrifier.ssa import BasicBlock, Function, Instr, SSAValue

    token = SSAValue(87, "int64")
    owner = Function("probe", [], {"entry": BasicBlock("entry", [])})
    region = Function("probe__planned_region_4", [], {
        "entry": BasicBlock("entry", [
            Instr(
                "string_token", [], token,
                attributes={"text": b"\x00asm", "token": 17},
            ),
        ]),
    }, metadata={
        "source_region_integral": {
            "owner": owner.name,
            "identity_token_chain": (
                "source-region", owner.name, "closure:53", "region_4",
            ),
            "output_value_ids": (),
        },
    })
    functions = {owner.name: owner, region.name: region}

    assert drop_dead_pure_region_calls(functions) == 1
    assert region.name not in functions
    assert owner.metadata["discarded_outputless_source_regions"] == ({
        "function": region.name,
        "identity_token_chain": (
            "source-region", owner.name, "closure:53", "region_4",
        ),
        "reason": "uncalled-pure-no-published-outputs",
    },)


def test_linked_method_record_fields_use_outer_authored_parameter_identity():
    from src.compiler.fortran_c_shell import (
        _linked_authored_parameter_aliases,
    )

    caller = SimpleNamespace(metadata={
        "parameter_names": (("body", 10),),
        "parameter_record_abi": {"body": {"identity": "Builder"}},
    })
    callee = SimpleNamespace(metadata={
        "parameter_names": (),
        "parameter_record_abi": {"self": {"identity": "Builder"}},
    })
    caller_graph = SimpleNamespace(graph={
        "identity_table": {"body": (10, 110)},
        "function_parameters": ("body",),
    })
    callee_graph = SimpleNamespace(graph={
        "identity_table": {"self": (3, 103)},
        "function_parameters": ("self",),
    })

    assert _linked_authored_parameter_aliases(
        caller, callee, caller_graph, callee_graph, ((10, 3),)
    ) == {"self": "body"}
    # Same-spelling SSA versions are not formal aliases.
    assert _linked_authored_parameter_aliases(
        caller, callee, caller_graph, callee_graph, ((110, 103),)
    ) == {}
    projected_callee_graph = SimpleNamespace(graph={
        "identity_table": {}, "function_parameters": (),
    })
    callee_records = SimpleNamespace(records={
        3: SimpleNamespace(identity="Builder"),
    })
    assert _linked_authored_parameter_aliases(
        caller,
        callee,
        caller_graph,
        projected_callee_graph,
        ((10, 3),),
        None,
        callee_records,
    ) == {"self": "body"}


def test_linked_sequence_argument_binds_its_complete_physical_descriptor():
    from src.compiler.fortran_c_shell import _bind_sequence_storage_members

    callee = SimpleNamespace(
        column_value_ids=(0,),
        length_address_id=10,
        capacity_value_id=11,
        status_address_id=12,
        live_flags_value_id=13,
    )
    caller = SimpleNamespace(
        column_value_ids=(191,),
        length_address_id=370,
        capacity_value_id=371,
        status_address_id=372,
        live_flags_value_id=373,
    )
    bindings = {}

    assert _bind_sequence_storage_members(bindings, callee, caller)
    assert bindings == {0: 191, 10: 370, 11: 371, 12: 372, 13: 373}


def test_authored_text_parameter_declares_its_utf8_sequence_view():
    from src.compiler.fortran_c_shell import (
        _authored_text_parameter_transforms,
    )

    graph = SimpleNamespace(graph={
        "identity_table": {"function_name": (8,), "memory_pages": (12,)},
        "function_parameter_annotations": {
            "function_name": "str",
            "memory_pages": "int",
        },
    })

    assert _authored_text_parameter_transforms(graph) == (
        (8, 8, "function_name", "utf8"),
    )


def test_optional_record_none_predicate_uses_row_handle_sentinel():
    from src.compiler.control_source import ControlExpression
    from src.compiler.fortran_c_shell import (
        _rewrite_optional_row_handle_none_predicate,
    )

    predicate = ControlExpression("eq", (
        ControlExpression("value", value_id=165),
        ControlExpression("const", value_id=180, literal=None),
    ))

    rewritten = _rewrite_optional_row_handle_none_predicate(predicate, (165,))

    assert rewritten.op == "eq"
    assert rewritten.operands[0].value_id == 165
    assert rewritten.operands[1].literal == -1


def test_mapping_pop_none_rewrites_predicate_without_record_parameters():
    import networkx as nx

    from src.compiler.control_source import (
        ConditionalBlock, ControlExpression, ControlProgram, ControlSequenceMutation,
        LoopBlock, SequenceBlock, StatementBlock,
    )
    from src.compiler.fortran_c_shell import (
        _record_sequence_projection_bindings,
    )

    graph = nx.DiGraph()
    graph.graph["parameter_sequence_record_abi"] = {}
    mutation = ControlSequenceMutation(
        20,
        "pop",
        (21, 22),
        23,
        policy="unique",
        argument_kind="mapping_pop_default_none",
    )
    predicate = ControlExpression("eq", (
        ControlExpression("value", value_id=23),
        ControlExpression("const", literal=None),
    ))
    control = ControlProgram(SequenceBlock((
        LoopBlock(
            "item", "0", "1", "1", SequenceBlock(()),
            sequence_mutations=(mutation,),
        ),
        ConditionalBlock(24, StatementBlock(("consume",)),
                         predicate_expression=predicate),
    )))

    rewritten, bindings, fields = _record_sequence_projection_bindings(
        graph, control,
    )
    conditional = rewritten.root.blocks[1]

    assert bindings == ()
    assert fields == ()
    assert conditional.predicate_expression.operands[1].literal == -1


def test_sequence_query_producer_is_scheduled_before_earlier_consumer():
    from src.compiler.control_source import (
        ConditionalBlock, ControlExpression, LoopBlock, SequenceBlock,
        SequenceQueryBlock, StatementBlock,
    )
    from src.compiler.fortran_c_shell import (
        _schedule_sequence_query_dependencies,
    )

    consumer = ConditionalBlock(
        180,
        StatementBlock(("consume",)),
        predicate_expression=ControlExpression("eq", (
            ControlExpression("value", value_id=165),
            ControlExpression("const", literal=-1),
        )),
    )
    producer = SequenceBlock((
        LoopBlock(
            "row", "0", "n", "1", StatementBlock(("produce",)),
            source_loop_node_id=164,
        ),
        SequenceQueryBlock(
            result_value_id=165,
            sequence_value_id=213,
            operation="first_or_default",
            default_value_id=62,
            producer_loop_node_id=164,
        ),
    ))

    scheduled = _schedule_sequence_query_dependencies(SequenceBlock((
        consumer, producer, StatementBlock(("tail",)),
    )))

    assert scheduled.blocks == (
        producer.blocks[0], producer.blocks[1], consumer,
        StatementBlock(("tail",)),
    )


def test_sequence_query_scheduler_repairs_a_query_separated_before_its_loop():
    from src.compiler.control_source import (
        ConditionalBlock, ControlExpression, LoopBlock, SequenceBlock,
        SequenceQueryBlock, StatementBlock,
    )
    from src.compiler.fortran_c_shell import (
        _schedule_sequence_query_dependencies,
    )

    consumer = ConditionalBlock(
        180,
        StatementBlock(("consume",)),
        predicate_expression=ControlExpression("eq", (
            ControlExpression("value", value_id=165),
            ControlExpression("const", literal=-1),
        )),
    )
    query = SequenceQueryBlock(
        result_value_id=165,
        sequence_value_id=213,
        operation="first_or_default",
        default_value_id=62,
        producer_loop_node_id=164,
    )
    producer = LoopBlock(
        "row", "0", "n", "1", StatementBlock(("produce",)),
        source_loop_node_id=164,
    )
    tail = StatementBlock(("tail",))

    scheduled = _schedule_sequence_query_dependencies(SequenceBlock((
        consumer, query, tail, producer,
    )))

    assert scheduled.blocks == (producer, query, consumer, tail)


def test_joined_inline_list_recovers_dynamic_elements_from_consuming_call():
    import ast
    import networkx as nx

    from src.compiler.fortran_c_shell import _joined_list_literal_mutations

    expression = ast.parse("_vector([uleb(index)])", mode="eval").body
    element = expression.args[0].elts[0]
    graph = nx.DiGraph()
    graph.add_node(
        100,
        value_id=32,
        type="Constant",
        expr_obj=None,
        parents=(),
        attributes={"aggregate_kind": "list", "value": []},
    )
    graph.add_node(
        200, value_id=44, type="Call", expr_obj=element, parents=(),
    )
    graph.add_node(
        300,
        value_id=45,
        type="Call",
        expr_obj=expression,
        parents=((100, "arg:0"),),
    )

    mutations = _joined_list_literal_mutations(graph, (32,))

    assert len(mutations) == 1
    assert mutations[0].sequence_value_id == 32
    assert mutations[0].argument_value_ids == (44,)
    assert mutations[0].effect_node_id == 200


def test_singleton_bytes_view_retains_its_dynamic_scalar_materialization():
    import ast
    import networkx as nx

    from src.compiler.fortran_c_shell import _sequence_concat_ops

    singleton = ast.parse("[value]", mode="eval").body
    graph = nx.DiGraph()
    graph.add_node(100, value_id=102, type="Lookup", parents=())
    graph.add_node(
        101,
        value_id=103,
        type="List",
        expr_obj=singleton,
        parents=((100, "elts:0"),),
        attributes={
            "aggregate_kind": "list",
            "aggregate_leaf_value_ids": (102,),
        },
    )
    graph.add_node(
        108,
        value_id=108,
        type="Call",
        parents=((101, "arg:0"),),
        attributes={
            "aggregate_kind": "bytes",
            "producer_kind": "aggregate_materialization",
            "aggregate_leaf_value_ids": (103,),
        },
    )

    operations, aliases, singleton_values = _sequence_concat_ops(graph)

    assert operations == ()
    assert aliases == ((108, 103),)
    assert singleton_values == {108: 102}


def test_retained_control_expression_recursively_keeps_bitwise_masks():
    import ast
    import networkx as nx

    from src.compiler.fortran_c_shell import _graph_control_expression

    graph = nx.DiGraph()
    graph.add_node(0, type="Input", parents=())
    graph.add_node(
        1, type="Const", parents=(), attributes={"value": 0x40},
    )
    graph.add_node(
        2, type="BitAnd", op="bitand", expr_obj=ast.parse("x & 64").body[0].value,
        parents=((0, "lhs"), (1, "rhs")),
    )
    graph.add_node(
        3, type="LogicalNot", op="logical_not",
        parents=((2, "operand"),),
    )

    expression = _graph_control_expression(graph, 3)

    assert expression.op == "not"
    assert expression.operands[0].op == "bitand"
    assert expression.operands[0].operands[0].op == "value"
    assert expression.operands[0].operands[0].value_id == 0
    assert expression.operands[0].operands[1].op == "const"
    assert expression.operands[0].operands[1].literal == 0x40


class _ArenaState:
    def __init__(self):
        self.values = np.arange(4.0)


def test_whole_object_library_refuses_numerical_projection_fallback(
    tmp_path, monkeypatch
):
    from src.common.tensors.accelerator_backends import aot_compile
    import src.compiler.fortran_c_shell as shell_module

    observed = {}

    def fake_compile(*args, **kwargs):
        observed.update(kwargs)
        return SimpleNamespace(
            public_output_value_ids={}, public_input_value_ids={}
        )

    monkeypatch.setattr(aot_compile, "compile_ast_aot", fake_compile)
    monkeypatch.setattr(
        shell_module, "_emit_class_surface_module",
        lambda *args, **kwargs: (None, ()),
    )

    with pytest.raises(FortranEmissionError, match="refusing to substitute"):
        compile_ast_fortran_c_shell(
            "def surface():\n    return object\n",
            "surface", {}, tmp_path, library=True,
        )

    assert observed["require_planned_shells"] is True
    assert observed["project_captured_hierarchy"] is False


def test_whole_object_qualified_name_is_mangled_before_fortran_emission(
    tmp_path, monkeypatch
):
    from src.common.tensors.accelerator_backends import aot_compile
    import src.compiler.fortran_c_shell as shell_module

    observed = {}
    monkeypatch.setattr(
        aot_compile,
        "compile_ast_aot",
        lambda *args, **kwargs: SimpleNamespace(
            public_output_value_ids={}, public_input_value_ids={}
        ),
    )

    def observe_name(_compilation, artifact_name, **_kwargs):
        observed["artifact_name"] = artifact_name
        return None, ()

    monkeypatch.setattr(
        shell_module, "_emit_class_surface_module", observe_name
    )

    with pytest.raises(FortranEmissionError, match="refusing to substitute"):
        compile_ast_fortran_c_shell(
            "def surface():\n    return object\n",
            "surface",
            {},
            tmp_path,
            name="ProcessGraph.build_from_ast",
            library=True,
        )

    assert observed["artifact_name"] == "ProcessGraph_build_from_ast"


def test_whole_object_sequence_field_is_real_record_storage_not_scalar_slot():
    from src.common.tensors.accelerator_backends.aot_compile import (
        compile_ast_aot,
    )
    from src.compiler.fortran_c_shell import _class_surface_ssa_program
    from src.compiler.ssa_fortran_backend import emit_module

    source = """
class Bucket:
    def __init__(self):
        self.items = []

    def fill(self, value):
        for index in range(4):
            self.items.append(value)
        return self.items

def run(value):
    bucket = Bucket()
    return bucket.fill(value)
"""
    compilation = compile_ast_aot(
        source,
        "run",
        {"value": 2.5},
        precompile_only=True,
        bake_mode="whole_program",
        require_planned_shells=True,
        project_captured_hierarchy=False,
    )
    module, outputs, exports = _class_surface_ssa_program(
        compilation, "bucket_record_probe"
    )

    fill_name = "bucket_record_probe__fill"
    init_name = "bucket_record_probe____init__"
    fill_record = next(iter(module.record_tables[fill_name].records.values()))
    init_record = next(iter(module.record_tables[init_name].records.values()))
    assert fill_record.fields[0].storage.value == "sequence"
    assert init_record.fields[0].storage.value == "sequence"
    assert fill_record.fields[0].storage_identity == "Bucket.items"
    assert init_record.fields[0].storage_identity == "Bucket.items"
    fill_sequence = module.sequence_tables[fill_name].by_id(
        fill_record.fields[0].sequence_id
    )
    assert set(fill_record.fields[0].value_ids) == {
        *fill_sequence.column_value_ids,
        fill_sequence.length_address_id,
        fill_sequence.capacity_value_id,
        fill_sequence.status_address_id,
    }
    assert any(
        instruction.op == "Store"
        and instruction.attributes.get("binding")
        == "ssa_sequence_initialize_length"
        for block in module.functions[init_name].blocks.values()
        for instruction in block.instrs
    )
    emitted = emit_module(
        module,
        name="bucket_record_probe",
        outputs=outputs,
        extra_roots=exports,
    )
    assert emitted.complete, [item.format() for item in emitted.shortfalls]


def test_whole_object_factory_dict_aliases_share_one_record_table_storage():
    from src.common.tensors.accelerator_backends.aot_compile import (
        compile_ast_aot,
    )
    from src.compiler.fortran_c_shell import _class_surface_ssa_program
    from src.compiler.ssa_fortran_backend import emit_module

    source = """
class DiGraph:
    adjlist_outer_dict_factory = dict

    def __init__(self):
        self._adj = self.adjlist_outer_dict_factory()
        self._succ = self._adj

def run():
    return DiGraph()
"""
    compilation = compile_ast_aot(
        source,
        "run",
        {},
        precompile_only=True,
        bake_mode="whole_program",
        require_planned_shells=True,
        project_captured_hierarchy=False,
    )
    module, outputs, exports = _class_surface_ssa_program(
        compilation, "digraph_alias_probe"
    )
    init_name = "digraph_alias_probe____init__"
    record = next(iter(module.record_tables[init_name].records.values()))
    fields = {field.name: field for field in record.fields}

    assert fields["_adj"].storage.value == "sequence"
    assert fields["_succ"].storage.value == "sequence"
    assert fields["_adj"].sequence_id == fields["_succ"].sequence_id
    assert fields["_adj"].value_ids == fields["_succ"].value_ids
    assert fields["_adj"].storage_identity == "DiGraph._adj"
    assert fields["_succ"].storage_identity == "DiGraph._adj"
    emitted = emit_module(
        module,
        name="digraph_alias_probe",
        outputs=outputs,
        extra_roots=exports,
    )
    assert emitted.complete, [item.format() for item in emitted.shortfalls]


def test_whole_object_dict_membership_is_source_linked_ssa_scan():
    from src.common.tensors.accelerator_backends.aot_compile import (
        compile_ast_aot,
    )
    from src.compiler.fortran_c_shell import _class_surface_ssa_program
    from src.compiler.ssa_fortran_backend import emit_module

    source = """
class Store:
    def __init__(self):
        self.table = {}

    def has(self, key):
        return key in self.table

def run(key):
    return Store().has(key)
"""
    compilation = compile_ast_aot(
        source,
        "run",
        {"key": 1},
        precompile_only=True,
        bake_mode="whole_program",
        require_planned_shells=True,
        project_captured_hierarchy=False,
    )
    module, outputs, exports = _class_surface_ssa_program(
        compilation, "dict_membership_probe"
    )
    has_name = "dict_membership_probe__has"
    calls = [
        instruction
        for block in module.functions[has_name].blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get("ssa_sequence_operation") == "contains"
    ]
    assert len(calls) == 1
    assert calls[0].attributes["source_linked"] is True
    assert "ssa_sequence_2_contains" in module.functions
    assert any(
        instruction.op == "Ret" and instruction.args == [calls[0].res]
        for block in module.functions[has_name].blocks.values()
        for instruction in block.instrs
    )
    emitted = emit_module(
        module,
        name="dict_membership_probe",
        outputs=outputs,
        extra_roots=exports,
    )
    assert emitted.complete, [item.format() for item in emitted.shortfalls]


def test_constructed_record_arena_is_initialized_before_method_call():
    from src.common.tensors.accelerator_backends.aot_compile import (
        compile_ast_aot,
    )
    from src.compiler.fortran_c_shell import _class_surface_ssa_program

    source = """
class Store:
    def __init__(self):
        self.table = {}

    def has(self, key):
        return key in self.table

def run(key):
    return Store().has(key)
"""
    compilation = compile_ast_aot(
        source,
        "run",
        {"key": 1},
        precompile_only=True,
        bake_mode="whole_program",
        require_planned_shells=True,
        project_captured_hierarchy=False,
    )
    module, _outputs, _exports = _class_surface_ssa_program(
        compilation, "constructed_record_order"
    )
    run_name = "constructed_record_order__run"
    calls = [
        instruction
        for block in module.functions[run_name].blocks.values()
        for instruction in block.instrs
        if instruction.op == "Call"
        and instruction.attributes.get("source_linked")
    ]
    assert [instruction.attributes["callee"] for instruction in calls] == [
        "constructed_record_order____init__",
        "constructed_record_order__has",
    ]
    record = next(iter(module.record_tables[run_name].records.values()))
    storage_ids = record.fields[0].value_ids
    assert tuple(value.id for value in calls[0].args) == storage_ids
    assert tuple(value.id for value in calls[1].args[1:]) == storage_ids
    assert not any(
        item.resolution == "unresolved"
        for records in module.call_table.values()
        for item in records
    )


def test_two_constructed_records_have_distinct_arenas_and_constructor_values():
    from src.common.tensors.accelerator_backends.aot_compile import (
        compile_ast_aot,
    )
    from src.compiler.fortran_c_shell import _class_surface_ssa_program

    source = """
class Bucket:
    def __init__(self, value):
        self.items = []
        self.items.append(value)

def run(left, right):
    first = Bucket(left)
    second = Bucket(right)
"""
    compilation = compile_ast_aot(
        source,
        "run",
        {"left": 1, "right": 2},
        precompile_only=True,
        bake_mode="whole_program",
        require_planned_shells=True,
        project_captured_hierarchy=False,
    )
    module, _outputs, _exports = _class_surface_ssa_program(
        compilation, "two_constructed_records"
    )
    run_name = "two_constructed_records__run"
    records = tuple(module.record_tables[run_name].records.values())
    assert len(records) == 2
    first_storage = records[0].fields[0].value_ids
    second_storage = records[1].fields[0].value_ids
    assert set(first_storage).isdisjoint(second_storage)
    calls = [
        instruction
        for block in module.functions[run_name].blocks.values()
        for instruction in block.instrs
        if instruction.op == "Call"
        and instruction.attributes.get("callee")
        == "two_constructed_records____init__"
    ]
    assert len(calls) == 2
    assert calls[0].args[0].id == 1
    assert calls[1].args[0].id == 2
    assert set(value.id for value in calls[0].args[1:]) == set(first_storage)
    assert set(value.id for value in calls[1].args[1:]) == set(second_storage)


def test_non_escaping_constructor_inside_loop_reuses_arena_in_loop_body():
    from src.common.tensors.accelerator_backends.aot_compile import (
        compile_ast_aot,
    )
    from src.compiler.fortran_c_shell import _class_surface_ssa_program

    source = """
class Bucket:
    def __init__(self):
        self.items = []

def run(count):
    for index in range(count):
        bucket = Bucket()
"""
    compilation = compile_ast_aot(
        source,
        "run",
        {"count": 2},
        precompile_only=True,
        bake_mode="whole_program",
        require_planned_shells=True,
        project_captured_hierarchy=False,
    )
    module, _outputs, _exports = _class_surface_ssa_program(
        compilation, "loop_constructor"
    )
    records = module.call_table["loop_constructor__run"]
    assert len(records) == 1
    record = records[0]
    assert record.callee_name == "__init__"
    assert record.enclosing_loop_ids
    assert record.resolution == "native_call"
    entry_calls = [
        instruction
        for instruction in module.functions[
            "loop_constructor__run"
        ].blocks["entry"].instrs
        if instruction.op == "Call"
    ]
    assert not entry_calls
    body_calls = [
        instruction
        for instruction in module.functions[
            "loop_constructor__run"
        ].blocks["loop_body"].instrs
        if instruction.op == "Call"
        and instruction.attributes.get("callee")
        == "loop_constructor____init__"
    ]
    assert len(body_calls) == 1
    assert not any(
        instruction.op == "Call"
        and instruction.attributes.get("callee")
        == "loop_constructor____init__"
        for name, block in module.functions[
            "loop_constructor__run"
        ].blocks.items()
        if name != "loop_body"
        for instruction in block.instrs
    )


def test_escaping_constructor_inside_loop_uses_child_table_instance_pool():
    from src.common.tensors.accelerator_backends.aot_compile import (
        compile_ast_aot,
    )
    from src.compiler.fortran_c_shell import _class_surface_ssa_program

    source = """
class Bucket:
    def __init__(self):
        self.items = []

def run(count):
    buckets = []
    for index in range(count):
        buckets.append(Bucket())
    return buckets
"""
    compilation = compile_ast_aot(
        source,
        "run",
        {"count": 2},
        precompile_only=True,
        bake_mode="whole_program",
        require_planned_shells=True,
        project_captured_hierarchy=False,
    )
    module, outputs, exports = _class_surface_ssa_program(
        compilation, "escaping_loop_constructor"
    )
    constructors = [
        record
        for record in module.call_table["escaping_loop_constructor__run"]
        if record.callee_name == "__init__"
    ]
    assert len(constructors) == 1
    assert constructors[0].resolution == "native_call"
    assert constructors[0].decomposition is None
    destination = module.sequence_tables[
        "escaping_loop_constructor__run"
    ].by_id(3)
    assert destination.child_table_pool is not None
    assert destination.child_table_pool.handle_column == 0
    body = module.functions[
        "escaping_loop_constructor__run"
    ].blocks["loop_body"]
    calls = [
        instruction.op == "Call"
        and instruction.attributes.get("callee")
        for instruction in body.instrs
        if instruction.op == "Call"
    ]
    assert calls[:2] == [
        "escaping_loop_constructor____init__",
        "ssa_sequence_3_append",
    ]
    append = next(
        instruction for instruction in body.instrs
        if instruction.attributes.get("ssa_sequence_operation") == "append"
    )
    induction = next(
        instruction.res
        for instruction in module.functions[
            "escaping_loop_constructor__run"
        ].blocks["loop_header"].instrs
        if instruction.op == "Phi"
    )
    assert append.args[-1] == induction
    from src.compiler.ssa_fortran_backend import emit_module

    emitted = emit_module(
        module,
        name="escaping_loop_constructor",
        outputs=outputs,
        extra_roots=exports,
    )
    assert emitted.complete, [item.format() for item in emitted.shortfalls]


def test_multi_field_loop_record_uses_grouped_distinct_sequence_pools():
    from src.common.tensors.accelerator_backends.aot_compile import (
        compile_ast_aot,
    )
    from src.compiler.fortran_c_shell import _class_surface_ssa_program

    source = """
class Pair:
    def __init__(self):
        self.left = []
        self.right = []

def run(count):
    out = []
    for index in range(count):
        out.append(Pair())
    return out
"""
    compilation = compile_ast_aot(
        source,
        "run",
        {"count": 2},
        precompile_only=True,
        bake_mode="whole_program",
        require_planned_shells=True,
        project_captured_hierarchy=False,
    )
    module, outputs, exports = _class_surface_ssa_program(
        compilation, "multi_field_loop_record"
    )
    record = next(iter(module.record_tables[
        "multi_field_loop_record__run"
    ].records.values()))
    fields = {field.name: field for field in record.fields}
    assert fields["left"].storage_identity == "Pair.left"
    assert fields["right"].storage_identity == "Pair.right"
    assert set(fields["left"].value_ids).isdisjoint(
        fields["right"].value_ids
    )
    assert record.instance_pool is not None
    pools = {
        field.storage_identity: field.sequence_pool
        for field in record.instance_pool.fields
    }
    assert set(pools) == {"Pair.left", "Pair.right"}
    assert set(pools["Pair.left"].column_value_ids).isdisjoint(
        pools["Pair.right"].column_value_ids
    )
    constructor = next(
        item
        for item in module.call_table["multi_field_loop_record__run"]
        if item.callee_name == "__init__"
    )
    assert constructor.resolution == "native_call"
    assert constructor.decomposition is None
    from src.compiler.ssa_fortran_backend import emit_module

    emitted = emit_module(
        module,
        name="multi_field_loop_record",
        outputs=outputs,
        extra_roots=exports,
    )
    assert emitted.complete, [item.format() for item in emitted.shortfalls]


def test_mixed_scalar_sequence_loop_record_uses_one_handle_without_id_alias():
    from src.common.tensors.accelerator_backends.aot_compile import (
        compile_ast_aot,
    )
    from src.compiler.fortran_c_shell import _class_surface_ssa_program
    from src.compiler.ssa_fortran_backend import emit_module

    source = """
class Mixed:
    def __init__(self, value):
        self.value = value
        self.items = []

def run(count):
    out = []
    for index in range(count):
        out.append(Mixed(index))
    return out
"""
    compilation = compile_ast_aot(
        source,
        "run",
        {"count": 2},
        precompile_only=True,
        bake_mode="whole_program",
        require_planned_shells=True,
        project_captured_hierarchy=False,
    )
    module, outputs, exports = _class_surface_ssa_program(
        compilation, "mixed_loop_record"
    )
    function = module.functions["mixed_loop_record__run"]
    assert len({value.id for value in function.args}) == len(function.args)
    record = next(iter(module.record_tables[
        "mixed_loop_record__run"
    ].records.values()))
    pool_fields = {
        field.storage_identity: field
        for field in record.instance_pool.fields
    }
    assert pool_fields["Mixed.items"].sequence_pool is not None
    assert pool_fields["Mixed.value"].scalar_value_id is not None
    body = function.blocks["loop_body"]
    assert any(
        instruction.attributes.get("binding")
        == "record_instance_pool_scalar"
        for instruction in body.instrs
    )
    emitted = emit_module(
        module,
        name="mixed_loop_record",
        outputs=outputs,
        extra_roots=exports,
    )
    assert emitted.complete, [item.format() for item in emitted.shortfalls]


def test_nested_loop_record_recursively_pools_leaf_storage():
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_fortran_backend import emit_module
    from src.transmogrifier.ssa import SSARecordFieldStorage

    source = """
class Inner:
    def __init__(self):
        self.items = []

class Outer:
    def __init__(self):
        self.inner = Inner()

def run(count):
    out = []
    for index in range(count):
        out.append(Outer())
    return out
"""
    module, outputs, exports = lower_ast_source_to_ssa(
        source,
        "run",
        name="nested_loop_record",
    )
    records = module.record_tables["nested_loop_record__run"].records
    outer = next(record for record in records.values()
                 if record.identity == "Outer")
    inner = records[outer.fields[0].record_id]
    assert outer.fields[0].storage is SSARecordFieldStorage.RECORD
    assert outer.fields[0].value_ids == ()
    assert inner.identity == "Inner"
    assert inner.fields[0].storage is SSARecordFieldStorage.SEQUENCE
    assert outer.instance_pool is not None
    assert [field.storage_identity for field in outer.instance_pool.fields] == [
        "Inner.items"
    ]
    calls = {
        (caller, item.callee_symbol): item
        for caller, items in module.call_table.items()
        for item in items
    }
    assert calls[(
        "nested_loop_record__run",
        "nested_loop_record__Outer____init__",
    )].resolution == "native_call"
    assert calls[(
        "nested_loop_record__Outer____init__",
        "nested_loop_record__Inner____init__",
    )].resolution == "native_call"
    emitted = emit_module(
        module,
        name="nested_loop_record",
        outputs=outputs,
        extra_roots=exports,
    )
    assert emitted.complete, [item.format() for item in emitted.shortfalls]


@pytest.mark.skipif(
    fortran_compiler() is None,
    reason="no Fortran compiler installed",
)
def test_compiled_whole_object_dict_membership_scans_caller_record(tmp_path):
    from src.common.tensors.accelerator_backends.aot_compile import (
        compile_ast_aot,
    )
    from src.compiler.fortran_c_shell import _class_surface_ssa_program
    from src.compiler.ssa_fortran_backend import emit_module

    source = """
class Store:
    def __init__(self):
        self.table = {}

    def has(self, key):
        return key in self.table

def run(key):
    return Store().has(key)
"""
    compilation = compile_ast_aot(
        source,
        "run",
        {"key": 1.0},
        precompile_only=True,
        bake_mode="whole_program",
        require_planned_shells=True,
        project_captured_hierarchy=False,
    )
    module, outputs, exports = _class_surface_ssa_program(
        compilation, "native_dict_membership"
    )
    emitted = emit_module(
        module,
        name="native_dict_membership",
        outputs=outputs,
        extra_roots=exports,
    )
    library_path = compile_module(
        emitted, directory=Path(tmp_path).resolve(), standalone=False
    )
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(Path(fortran_compiler()).resolve().parent))
    library = ctypes.CDLL(str(library_path))
    function = library.native_dict_membership__has
    double_pointer = ctypes.POINTER(ctypes.c_double)
    int_pointer = ctypes.POINTER(ctypes.c_int32)
    bool_pointer = ctypes.POINTER(ctypes.c_bool)
    function.argtypes = [
        ctypes.c_int32,
        double_pointer,
        double_pointer,
        int_pointer,
        ctypes.c_int32,
        int_pointer,
        ctypes.c_double,
        bool_pointer,
    ]
    keys = np.asarray([1.0, 3.0, 7.0, 0.0], dtype=np.float64)
    values = np.asarray([10.0, 30.0, 70.0, 0.0], dtype=np.float64)
    length = np.asarray([3], dtype=np.int32)
    status = np.asarray([0], dtype=np.int32)
    result = np.asarray([False], dtype=np.bool_)

    def contains(query):
        result[0] = False
        function(
            4,
            keys.ctypes.data_as(double_pointer),
            values.ctypes.data_as(double_pointer),
            length.ctypes.data_as(int_pointer),
            4,
            status.ctypes.data_as(int_pointer),
            query,
            result.ctypes.data_as(bool_pointer),
        )
        return bool(result[0])

    assert contains(1.0)
    assert contains(3.0)
    assert not contains(2.0)
    assert contains(7.0)


def test_whole_object_table_lookup_and_store_are_key_search_helpers():
    from src.common.tensors.accelerator_backends.aot_compile import (
        compile_ast_aot,
    )
    from src.compiler.fortran_c_shell import _class_surface_ssa_program

    source = """
class Store:
    def __init__(self):
        self.table = {}

    def lookup(self, key):
        return self.table[key]

    def store(self, key, value):
        self.table[key] = value
        return value

def run(key, value):
    store = Store()
    store.store(key, value)
    return store.lookup(key)
"""
    compilation = compile_ast_aot(
        source,
        "run",
        {"key": 1.0, "value": 2.0},
        precompile_only=True,
        bake_mode="whole_program",
        require_planned_shells=True,
        project_captured_hierarchy=False,
    )
    module, _outputs, _exports = _class_surface_ssa_program(
        compilation, "table_key_search"
    )
    lookup = module.functions["table_key_search__lookup"]
    store = module.functions["table_key_search__store"]
    assert any(
        instruction.attributes.get("ssa_sequence_operation") == "lookup"
        for block in lookup.blocks.values()
        for instruction in block.instrs
    )
    assert any(
        instruction.attributes.get("ssa_sequence_operation") == "table_store"
        for block in store.blocks.values()
        for instruction in block.instrs
    )
    assert not any(
        instruction.op in {"Indexed", "IndexedStore"}
        for function in (lookup, store)
        for block in function.blocks.values()
        for instruction in block.instrs
    )


def test_whole_object_if_assignment_lowers_to_conditional_phi_merge():
    from src.common.tensors.accelerator_backends.aot_compile import (
        compile_ast_aot,
    )
    from src.compiler.fortran_c_shell import _class_surface_ssa_program

    source = """
def choose(flags, mask, add):
    if add & mask:
        flags &= ~mask
    return flags | add
"""
    compilation = compile_ast_aot(
        source,
        "choose",
        {"flags": 7, "mask": 3, "add": 1},
        precompile_only=True,
        bake_mode="whole_program",
        require_planned_shells=True,
        project_captured_hierarchy=False,
    )
    module, _outputs, _exports = _class_surface_ssa_program(
        compilation, "conditional_merge"
    )
    function = module.functions["conditional_merge__choose"]
    instructions = [
        instruction
        for block in function.blocks.values()
        for instruction in block.instrs
    ]
    conditional = next(
        instruction for instruction in instructions
        if instruction.op == "CondBr"
    )
    merged = next(
        instruction for instruction in instructions
        if instruction.op == "Phi"
        and instruction.attributes.get("binding") == "conditional_carried"
    )
    assert [value.id for value in conditional.args] == [4]
    assert [value.id for value in merged.args] == [6, 2]
    assert merged.res.id == 3
    assert any(
        instruction.op == "Call" and [value.id for value in instruction.args] == [3, 0]
        for instruction in instructions
    )


def test_nested_if_threads_inner_phi_into_outer_phi():
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    module, outputs, _exports = lower_ast_source_to_ssa(
        "def nested(x, outer, inner):\n"
        "    value = x\n"
        "    if outer:\n"
        "        value = x + 1.0\n"
        "        if inner:\n"
        "            value = value * 2.0\n"
        "    return value\n",
        "nested",
        name="nested_conditional",
    )
    function = module.functions["nested_conditional__nested"]
    phis = [
        instruction
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Phi"
        and instruction.attributes.get("binding") == "conditional_carried"
    ]

    assert len(phis) == 2
    inner, outer = sorted(phis, key=lambda instruction: instruction.res.id)
    assert outer.args[0].id == inner.res.id
    assert outputs[function.name] == (outer.res,)
    assert not function.metadata.get("structural_output_shortfalls")


def test_iterable_loop_target_is_accounted_as_coordinator_lowered():
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    module, _outputs, _exports = lower_ast_source_to_ssa(
        "def total(values):\n"
        "    result = 0\n"
        "    for value in values:\n"
        "        result += value\n"
        "    return result\n",
        "total",
        name="iterable_target_accounting",
    )
    function = module.functions["iterable_target_accounting__total"]
    iterable_targets = {
        int(instruction.res.id)
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.res is not None
        and instruction.op == "Load"
        and instruction.attributes.get("binding") in {
            "iterable", "projected_iterable", "static_iterable",
            "closure_iterable",
        }
    }

    assert iterable_targets
    assert iterable_targets.issubset(set(
        map(int, function.metadata["lowered_source_value_ids"])
    ))


def test_annotated_string_sequence_is_a_typed_token_arena():
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_fortran_backend import emit_module

    module, outputs, exports = lower_ast_source_to_ssa(
        "from typing import Sequence\n"
        "def count_x(values: Sequence[str]) -> int:\n"
        "    total = 0\n"
        "    for value in values:\n"
        "        if value == 'x':\n"
        "            total += 1\n"
        "    return total\n",
        "count_x",
        name="annotated_string_sequence",
    )
    function = module.functions["annotated_string_sequence__count_x"]
    parameter_ids = dict(function.metadata["parameter_names"])
    sequence_id = int(parameter_ids["values"])
    descriptor = module.sequence_tables[function.name].sequences[sequence_id]

    assert descriptor.column_dtypes == ("int64",)
    assert descriptor.writable is False
    emitted = emit_module(
        module,
        name="annotated_string_sequence",
        outputs=outputs,
        extra_roots=exports,
    )
    assert emitted.complete, [item.format() for item in emitted.shortfalls]


def test_sequence_of_authored_records_projects_fields_inside_loop():
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.project_compilation_product import (
        _unexplained_root_argument_ids,
    )

    module, _outputs, _exports = lower_ast_source_to_ssa(
        "from dataclasses import dataclass\n"
        "from typing import Sequence\n"
        "@dataclass\n"
        "class Row:\n"
        "    kind: str\n"
        "    value: int\n"
        "def total(rows: Sequence[Row]) -> int:\n"
        "    result = 0\n"
        "    for row in rows:\n"
        "        if row.kind == 'keep':\n"
        "            result += row.value\n"
        "    return result\n",
        "total",
        name="record_row_projection",
    )
    function = module.functions["record_row_projection__total"]

    assert _unexplained_root_argument_ids(function) == ()
    projected_fields = {
        field_name
        for _value_id, field_name, _dtype
        in function.metadata["record_sequence_projection_fields"]
    }
    assert projected_fields == {"kind", "value"}


def test_filtered_authored_record_sequence_keeps_row_provenance():
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.project_compilation_product import (
        _unexplained_root_argument_ids,
    )

    module, _outputs, _exports = lower_ast_source_to_ssa(
        "from dataclasses import dataclass\n"
        "from typing import Sequence\n"
        "@dataclass\n"
        "class Row:\n"
        "    kind: str\n"
        "    value: int\n"
        "def total(rows: Sequence[Row]) -> int:\n"
        "    selected = [row for row in rows if row.kind == 'keep']\n"
        "    result = 0\n"
        "    for row in selected:\n"
        "        result += row.value\n"
        "    return result\n",
        "total",
        name="filtered_record_row_projection",
    )
    function = module.functions["filtered_record_row_projection__total"]
    unexplained = _unexplained_root_argument_ids(function)
    definitions = {
        int(instruction.res.id)
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.res is not None
    }
    projected_fields = {
        int(value_id): field_name
        for value_id, field_name, _dtype
        in function.metadata["record_sequence_projection_fields"]
    }

    assert set(projected_fields.values()) == {"kind", "value"}
    assert set(projected_fields).issubset(definitions)
    assert not set(projected_fields).intersection(unexplained)
    assert unexplained == ()


def test_pursued_re_compile_uses_one_external_multikey_table_contract():
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    contract = (
        Path(__file__).resolve().parents[1]
        / "extraction_contracts"
        / "program_extraction.yaml"
    )
    source = (
        "def entry(pattern, flags):\n"
        "    return compile_re(pattern, flags)\n"
    )
    module, outputs, exports = lower_ast_source_to_ssa(
        source,
        "entry",
        python_bindings={"compile_re": re._compile},
        name="re_compile_probe",
        extraction_contract=contract,
    )
    compile_function = module.functions["re_compile_probe___compile"]
    operations = [
        instruction.attributes.get("ssa_sequence_operation")
        for block in compile_function.blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get("ssa_sequence_operation")
    ]
    assert "lookup" in operations
    assert "table_store" in operations
    assert "table_delete_first" in operations


def test_pursued_networkx_remove_node_emits_complete_nested_table_ssa():
    import networkx as nx

    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_fortran_backend import emit_module

    contract = (
        Path(__file__).resolve().parents[1]
        / "extraction_contracts"
        / "program_extraction.yaml"
    )
    source = (
        "def entry(graph, node):\n"
        "    return remove(graph, node)\n"
    )
    module, outputs, exports = lower_ast_source_to_ssa(
        source,
        "entry",
        python_bindings={"remove": nx.DiGraph.remove_node},
        name="networkx_remove_probe",
        extraction_contract=contract,
    )
    emitted = emit_module(
        module,
        name="networkx_remove_probe",
        outputs=outputs,
        extra_roots=exports,
    )

    assert emitted.complete, [item.format() for item in emitted.shortfalls]
    remove = module.functions["networkx_remove_probe__remove_node"]
    operations = [
        instruction.attributes.get("ssa_sequence_operation")
        for block in remove.blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get("ssa_sequence_operation")
    ]
    assert operations.count("child_table_delete") == 2
    assert operations.count("table_delete") == 3
    argument_ids = {int(argument.id) for argument in remove.args}
    record_value_ids = {
        int(value_id)
        for record in module.record_tables[remove.name].records.values()
        for field in record.fields
        for value_id in field.value_ids
    }
    nested_descriptors = [
        descriptor
        for descriptor in module.sequence_tables[remove.name].sequences.values()
        if descriptor.child_table_pool is not None
    ]
    assert nested_descriptors
    for descriptor in nested_descriptors:
        pool = descriptor.child_table_pool
        assert pool is not None
        pool_ids = {
            *map(int, pool.column_value_ids),
            int(pool.length_value_id),
            int(pool.capacity_value_id),
            int(pool.row_stride_value_id),
            *(
                (int(pool.status_value_id),)
                if pool.status_value_id is not None else ()
            ),
            *(
                (int(pool.live_flags_value_id),)
                if pool.live_flags_value_id is not None else ()
            ),
        }
        assert pool_ids <= argument_ids
        assert pool_ids <= record_value_ids


def test_pursued_re_compile_closure_emits_complete_fortran_ssa():
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_fortran_backend import emit_module

    contract = (
        Path(__file__).resolve().parents[1]
        / "extraction_contracts"
        / "program_extraction.yaml"
    )
    source = (
        "def entry(pattern, flags):\n"
        "    return compile_re(pattern, flags)\n"
    )
    module, outputs, exports = lower_ast_source_to_ssa(
        source,
        "entry",
        python_bindings={"compile_re": re._compile},
        name="re_compile_probe",
        extraction_contract=contract,
    )
    emitted = emit_module(
        module,
        name="re_compile_probe",
        outputs=outputs,
        extra_roots=exports,
    )

    assert emitted.complete, [item.format() for item in emitted.shortfalls]
    assert len(module.functions) >= 100
    compile_function = module.functions["re_compile_probe___compile"]
    operations = [
        instruction.attributes.get("ssa_sequence_operation")
        for block in compile_function.blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get("ssa_sequence_operation")
    ]
    assert "lookup" in operations
    assert "table_store" in operations
    assert "table_delete_first" in operations
    assert any(
        instruction.attributes.get("ssa_sequence_operation") == "fill"
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
    )
    assert any(
        instruction.attributes.get("ssa_sequence_operation") == "append_slice"
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
    )


def test_whole_object_table_delete_is_source_linked_tombstone_helper():
    from src.common.tensors.accelerator_backends.aot_compile import (
        compile_ast_aot,
    )
    from src.compiler.fortran_c_shell import _class_surface_ssa_program

    source = """
class Store:
    def __init__(self):
        self.table = {}

    def delete(self, key):
        del self.table[key]

def run(key):
    Store().delete(key)
"""
    compilation = compile_ast_aot(
        source,
        "run",
        {"key": 1.0},
        precompile_only=True,
        bake_mode="whole_program",
        require_planned_shells=True,
        project_captured_hierarchy=False,
    )
    module, _outputs, _exports = _class_surface_ssa_program(
        compilation, "table_delete"
    )
    function = module.functions["table_delete__delete"]
    calls = [
        instruction
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get("ssa_sequence_operation") == "table_delete"
    ]
    assert len(calls) == 1
    descriptor = module.sequence_tables[function.name].by_id(
        calls[0].attributes["sequence_id"]
    )
    assert descriptor.live_flags_value_id is not None
    assert calls[0].attributes["source_linked"] is True
    assert not any(
        instruction.op in {"delitem", "IndexedStore"}
        for block in function.blocks.values()
        for instruction in block.instrs
    )


def test_loop_indexed_table_lookup_stays_inside_retained_loop_body():
    from src.common.tensors.accelerator_backends.aot_compile import (
        compile_ast_aot,
    )
    from src.compiler.fortran_c_shell import _class_surface_ssa_program

    source = """
class Store:
    def __init__(self):
        self.table = {}

    def values_for(self, keys):
        for key in keys:
            value = self.table[key]

def run(keys):
    Store().values_for(keys)
"""
    compilation = compile_ast_aot(
        source,
        "run",
        {"keys": [1.0, 2.0]},
        precompile_only=True,
        bake_mode="whole_program",
        require_planned_shells=True,
        project_captured_hierarchy=False,
    )
    module, _outputs, _exports = _class_surface_ssa_program(
        compilation, "loop_table_lookup"
    )
    function = module.functions["loop_table_lookup__values_for"]
    lookup_blocks = [
        block.name
        for block in function.blocks.values()
        if any(
            instruction.attributes.get("ssa_sequence_operation") == "lookup"
            for instruction in block.instrs
        )
    ]
    assert lookup_blocks == ["loop_body"]


@pytest.mark.skipif(
    fortran_compiler() is None,
    reason="no Fortran compiler installed",
)
def test_compiled_table_lookup_store_and_delete_share_caller_record(tmp_path):
    from src.common.tensors.accelerator_backends.aot_compile import (
        compile_ast_aot,
    )
    from src.compiler.fortran_c_shell import _class_surface_ssa_program
    from src.compiler.ssa_fortran_backend import emit_module

    source = """
class Store:
    def __init__(self):
        self.table = {}

    def lookup(self, key):
        return self.table[key]

    def store(self, key, value):
        self.table[key] = value
        return value

    def delete(self, key):
        del self.table[key]

def run(key, value):
    return value
"""
    compilation = compile_ast_aot(
        source,
        "run",
        {"key": 1.0, "value": 2.0},
        precompile_only=True,
        bake_mode="whole_program",
        require_planned_shells=True,
        project_captured_hierarchy=False,
    )
    module, outputs, exports = _class_surface_ssa_program(
        compilation, "native_table_record"
    )
    emitted = emit_module(
        module,
        name="native_table_record",
        outputs=outputs,
        extra_roots=exports,
    )
    library_path = compile_module(
        emitted, directory=Path(tmp_path).resolve(), standalone=False
    )
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(Path(fortran_compiler()).resolve().parent))
    library = ctypes.CDLL(str(library_path))
    lookup = library.native_table_record__lookup
    store = library.native_table_record__store
    delete = library.native_table_record__delete
    double_pointer = ctypes.POINTER(ctypes.c_double)
    int_pointer = ctypes.POINTER(ctypes.c_int32)
    bool_pointer = ctypes.POINTER(ctypes.c_bool)
    lookup.argtypes = [
        ctypes.c_int32, double_pointer, double_pointer, int_pointer,
        ctypes.c_int32, int_pointer, bool_pointer, ctypes.c_double, double_pointer,
    ]
    store.argtypes = [
        ctypes.c_int32, double_pointer, double_pointer, int_pointer,
        ctypes.c_int32, int_pointer, bool_pointer, ctypes.c_double, double_pointer,
    ]
    delete.argtypes = [
        ctypes.c_int32, double_pointer, double_pointer, int_pointer,
        ctypes.c_int32, int_pointer, bool_pointer, ctypes.c_double,
    ]
    keys = np.asarray([1.0, 3.0, 0.0, 0.0], dtype=np.float64)
    values = np.asarray([10.0, 30.0, 0.0, 0.0], dtype=np.float64)
    length = np.asarray([2], dtype=np.int32)
    status = np.asarray([0], dtype=np.int32)
    live = np.asarray([True, True, False, False], dtype=np.bool_)
    scalar = np.asarray([0.0], dtype=np.float64)

    def invoke_store(key, value):
        scalar[0] = value
        store(
            4, keys.ctypes.data_as(double_pointer),
            values.ctypes.data_as(double_pointer),
            length.ctypes.data_as(int_pointer), 4,
            status.ctypes.data_as(int_pointer),
            live.ctypes.data_as(bool_pointer), key,
            scalar.ctypes.data_as(double_pointer),
        )

    def invoke_lookup(key):
        scalar[0] = -1.0
        lookup(
            4, keys.ctypes.data_as(double_pointer),
            values.ctypes.data_as(double_pointer),
            length.ctypes.data_as(int_pointer), 4,
            status.ctypes.data_as(int_pointer),
            live.ctypes.data_as(bool_pointer), key,
            scalar.ctypes.data_as(double_pointer),
        )
        return float(scalar[0]), int(status[0])

    def invoke_delete(key):
        delete(
            4, keys.ctypes.data_as(double_pointer),
            values.ctypes.data_as(double_pointer),
            length.ctypes.data_as(int_pointer), 4,
            status.ctypes.data_as(int_pointer),
            live.ctypes.data_as(bool_pointer), key,
        )

    invoke_store(3.0, 33.0)
    assert (values[1], length[0], status[0]) == (33.0, 2, 3)
    invoke_store(7.0, 70.0)
    assert (keys[2], values[2], length[0], status[0]) == (7.0, 70.0, 3, 1)
    assert invoke_lookup(7.0) == (70.0, 1)
    assert invoke_lookup(2.0) == (0.0, 0)
    invoke_delete(3.0)
    assert status[0] == 4
    assert not live[1]
    assert invoke_lookup(3.0) == (0.0, 0)
    length[0] = 4
    invoke_store(9.0, 90.0)
    assert status[0] == 2
    assert 9.0 not in keys


@pytest.mark.skipif(
    fortran_compiler() is None,
    reason="no Fortran compiler installed",
)
def test_ast_c_shell_resolves_and_rotates_dotted_object_arenas(tmp_path):
    artifact = compile_ast_fortran_c_shell(
        "def frame(state):\n"
        "    state.values += 1.0\n"
        "    return state.values\n",
        "frame",
        {"state": _ArenaState()},
        tmp_path,
        output_names=("values_out",),
        state_feedback={"state.values": "values_out"},
    )

    payload = json.loads(artifact.run(frames=2).stdout)

    assert payload["outputs"]["values_out"] == {
        "first": 2.0,
        "sum": 14.0,
    }


@pytest.mark.skipif(
    fortran_compiler() is None,
    reason="no Fortran compiler installed",
)
def test_ast_c_shell_uses_captured_early_return_identity(tmp_path):
    artifact = compile_ast_fortran_c_shell(
        "def restore(value, ref):\n"
        "    if isinstance(ref, AbstractTensor):\n"
        "        return value\n"
        "    return float(value.item())\n"
        "\n"
        "def frame(dt):\n"
        "    return restore(dt * 2.0, dt)\n",
        "frame",
        {"dt": np.asarray([0.05], dtype=np.float64)},
        tmp_path,
        python_bindings={"AbstractTensor": AbstractTensor},
        output_names=("value_out",),
    )

    payload = json.loads(artifact.run().stdout)

    assert payload["outputs"]["value_out"] == {
        "first": pytest.approx(0.1),
        "sum": pytest.approx(0.1),
    }
    assert "source_name: dt" in artifact.api_path.read_text(encoding="utf-8")


@pytest.mark.skipif(
    fortran_compiler() is None,
    reason="no Fortran compiler installed",
)
def test_generated_c_shell_launches_fortran_and_applies_feedback(tmp_path):
    source = """
module affine_fortran
  use, intrinsic :: iso_c_binding
  implicit none
contains
  subroutine affine(extent_4, x, y) bind(C, name="affine")
    integer(c_int), value :: extent_4
    real(c_double), intent(in) :: x(extent_4)
    real(c_double), intent(out) :: y(extent_4)
    y = x * 2.0_c_double + 1.0_c_double
  end subroutine affine
end module affine_fortran
"""
    api = CompiledProgramAPI(
        module="affine_fortran",
        language="fortran",
        entry="affine",
        entry_points=(EntryPoint(
            name="affine",
            symbol="affine",
            kind="numerical",
            parameters=(
                Parameter(
                    "extent_4", "extent", "int32", "int32_t",
                    "c_int32", "value",
                ),
                Parameter(
                    "x", "input", "float64", "double", "c_double",
                    "reference", (4,), "extent_4", "x",
                ),
                Parameter(
                    "y", "output", "float64", "double", "c_double",
                    "reference", (4,), "extent_4", "y",
                ),
            ),
        ),),
    )
    module = FortranModule("affine_fortran", source, api=api)

    artifact = compile_fortran_module_c_shell(
        module,
        {"x": np.arange(1.0, 9.0)},
        tmp_path,
        state_feedback={"x": "y"},
        extent_overrides={"extent_4": 8},
        name="affine_native",
    )
    payload = json.loads(artifact.run(frames=2).stdout)

    assert artifact.executable_path.is_file()
    assert artifact.final_outputs_path.is_file()
    assert payload["status"] == 1
    assert payload["frames"] == 2
    assert payload["outputs"]["y"] == {"first": 7.0, "sum": 168.0}
    assert payload["shell_ns_total"] > 0
    c_source = artifact.c_source_path.read_text(encoding="utf-8")
    assert "slots[0] = slots[1]" in c_source
    assert "memcpy(slots[0], slots[1]" not in c_source


@pytest.mark.skipif(
    fortran_compiler() is None,
    reason="no Fortran compiler installed",
)
def test_compiled_library_contract_records_selected_root_not_link_order(tmp_path):
    source = """
module linked_fortran
  use, intrinsic :: iso_c_binding
  implicit none
contains
  subroutine linked_helper() bind(C, name="linked_helper")
  end subroutine linked_helper
  subroutine authored_root() bind(C, name="authored_root")
  end subroutine authored_root
end module linked_fortran
"""
    entries = tuple(
        EntryPoint(name=name, symbol=name, kind="control", parameters=())
        for name in ("linked_helper", "authored_root")
    )
    module = FortranModule(
        "linked_fortran",
        source,
        api=CompiledProgramAPI(
            module="linked_fortran",
            language="fortran",
            entry="linked_helper",
            entry_points=entries,
        ),
    )

    artifact = compile_fortran_module_c_shell(
        module,
        {},
        tmp_path,
        entrypoint="authored_root",
        name="linked_native",
        library=True,
    )
    import yaml

    published = yaml.safe_load(artifact.api_path.read_text(encoding="utf-8"))

    assert artifact.entrypoint == "authored_root"
    assert published["entry"] == "authored_root"
    assert tuple(item["name"] for item in published["entry_points"]) == (
        "linked_helper",
        "authored_root",
    )
    assert published["metadata"]["packed_entrypoints"]["authored_root"] == {
        "schema": "turing.packed-pointer-array.v1",
        "symbol": "authored_root__packed",
        "parameter_count": 0,
    }
    assert "authored_root__packed" in artifact.c_source_path.read_text(
        encoding="utf-8"
    )
    # The ordinary typed ABI remains directly callable as advertised; the
    # packed pointer-array entry is an additional generic transport surface.
    import ctypes

    runtime_handles = []
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        for dependency in published.get("metadata", {}).get(
            "runtime_dependencies", ()
        ):
            parent = Path(str(dependency["path"])).resolve().parent
            runtime_handles.append(os.add_dll_directory(str(parent)))
    library = ctypes.CDLL(str(artifact.executable_path.resolve()))
    assert getattr(library, "authored_root") is not None
    assert getattr(library, "authored_root__packed") is not None
    # Keep Windows DLL search handles resident until after both lookups.
    assert runtime_handles or os.name != "nt"


@pytest.mark.skipif(
    fortran_compiler() is None,
    reason="no Fortran compiler installed",
)
def test_generated_c_shell_preserves_all_extent_arguments_in_abi_order(tmp_path):
    source = """
module two_extent_fortran
  use, intrinsic :: iso_c_binding
  implicit none
contains
  subroutine copy_with_scalar_extent(extent_1, extent_4, x, y) &
      bind(C, name="copy_with_scalar_extent")
    integer(c_int), value :: extent_1
    integer(c_int), value :: extent_4
    real(c_double), intent(in) :: x(extent_4)
    real(c_double), intent(out) :: y(extent_4)
    y = x + real(extent_1, c_double)
  end subroutine copy_with_scalar_extent
end module two_extent_fortran
"""
    api = CompiledProgramAPI(
        module="two_extent_fortran",
        language="fortran",
        entry="copy_with_scalar_extent",
        entry_points=(EntryPoint(
            name="copy_with_scalar_extent",
            symbol="copy_with_scalar_extent",
            kind="control",
            parameters=(
                Parameter(
                    "extent_1", "extent", "int32", "int32_t",
                    "c_int32", "value",
                ),
                Parameter(
                    "extent_4", "extent", "int32", "int32_t",
                    "c_int32", "value",
                ),
                Parameter(
                    "x", "input", "float64", "double", "c_double",
                    "reference", (4,), "extent_4", "x",
                ),
                Parameter(
                    "y", "output", "float64", "double", "c_double",
                    "reference", (4,), "extent_4", "y",
                ),
            ),
        ),),
    )
    module = FortranModule("two_extent_fortran", source, api=api)

    artifact = compile_fortran_module_c_shell(
        module,
        {"x": np.arange(8.0)},
        tmp_path,
        extent_overrides={"extent_4": 8},
        name="two_extent_native",
    )
    payload = json.loads(artifact.run().stdout)

    assert "int32_t, int32_t, double *" in artifact.c_source_path.read_text()
    assert payload["outputs"]["y"] == {"first": 1.0, "sum": 36.0}


def test_c_shell_allocates_dynamic_rank_two_workspace_from_explicit_extents():
    parameters = (
        Parameter(
            "extent_dynamic_0_1", "extent", "int32", "int32_t",
            "c_int32", "value",
        ),
        Parameter(
            "extent_dynamic_0_2", "extent", "int32", "int32_t",
            "c_int32", "value",
        ),
        Parameter(
            "scratch", "workspace", "float64", "double", "c_double",
            "reference", source_name="callee.scratch",
            extents=("extent_dynamic_0_1", "extent_dynamic_0_2"),
        ),
    )
    module = FortranModule(
        "workspace_fortran",
        "",
        api=CompiledProgramAPI(
            "workspace_fortran", "fortran", "workspace",
            (EntryPoint("workspace", "workspace", "control", parameters),),
        ),
    )

    with pytest.raises(ValueError, match="require explicit positive"):
        emit_fortran_c_shell_source(module)

    source = emit_fortran_c_shell_source(
        module,
        extent_overrides={
            "extent_dynamic_0_1": 3,
            "extent_dynamic_0_2": 4,
        },
    )

    assert "slots[0] = calloc(12, sizeof(double));" in source
    assert "short initial state at callee.scratch" not in source


@pytest.mark.skipif(
    fortran_compiler() is None,
    reason="no Fortran compiler installed",
)
def test_c_shell_converts_multidimensional_row_major_arenas_at_boundary(tmp_path):
    source = """
module layout_fortran
  use, intrinsic :: iso_c_binding
  implicit none
contains
  subroutine alter_cell(extent_2, extent_3, x, y) bind(C, name="alter_cell")
    integer(c_int), value :: extent_2
    integer(c_int), value :: extent_3
    real(c_double), intent(in) :: x(extent_2, extent_3)
    real(c_double), intent(out) :: y(extent_2, extent_3)
    y = x
    y(1, 2) = y(1, 2) + 100.0_c_double
  end subroutine alter_cell
end module layout_fortran
"""
    parameters = (
        Parameter(
            "extent_2", "extent", "int32", "int32_t",
            "c_int32", "value",
        ),
        Parameter(
            "extent_3", "extent", "int32", "int32_t",
            "c_int32", "value",
        ),
        Parameter(
            "x", "input", "float64", "double", "c_double",
            "reference", (2, 3), "extent_2", "x",
        ),
        Parameter(
            "y", "output", "float64", "double", "c_double",
            "reference", (2, 3), "extent_2", "y",
        ),
    )
    module = FortranModule(
        "layout_fortran",
        source,
        api=CompiledProgramAPI(
            "layout_fortran", "fortran", "alter_cell",
            (EntryPoint(
                "alter_cell", "alter_cell", "control", parameters,
            ),),
        ),
    )
    values = np.arange(6.0).reshape(2, 3)

    artifact = compile_fortran_module_c_shell(
        module, {"x": values}, tmp_path, name="layout_native",
    )
    artifact.run()
    result = np.fromfile(artifact.final_outputs_path, dtype=np.float64)

    expected = values.copy()
    expected[0, 1] += 100.0
    np.testing.assert_array_equal(result.reshape(2, 3), expected)
    c_source = artifact.c_source_path.read_text(encoding="utf-8")
    assert "logical_index" in c_source


def test_generated_c_shell_uses_win32_rgb_blit_from_shared_io_contract():
    parameters = (
        Parameter(
            "extent_4", "extent", "int32", "int32_t", "c_int32", "value",
        ),
        *(
            Parameter(
                channel, "output", "float64", "double", "c_double",
                "reference", (4,), "extent_4", channel,
            )
            for channel in ("red", "green", "blue")
        ),
    )
    shell_io = {
        "requirements": {
            "requests": [{
                "capability": "display_double_buffer",
                "optional": False,
                "attributes": {
                    "pixel_format": "rgb_f64_planar",
                    "width": 2,
                    "height": 2,
                    "title": "Generated RGB",
                },
            }],
            "bindings": [
                {
                    "resource": f"display.{channel}",
                    "entry_point": "rgb",
                    "parameter": channel,
                }
                for channel in ("red", "green", "blue")
            ],
            "options": [],
        },
        "abi": {},
    }
    api = CompiledProgramAPI(
        module="rgb_fortran",
        language="fortran",
        entry="rgb",
        entry_points=(EntryPoint(
            name="rgb", symbol="rgb", kind="numerical",
            parameters=parameters,
        ),),
        metadata={"shell_io": shell_io},
    )
    module = FortranModule("rgb_fortran", "", api=api)

    source = emit_fortran_c_shell_source(
        module, extent_overrides={"extent_4": 4}
    )

    assert "StretchDIBits(" in source
    assert "CreateWindowExA(" in source
    assert "PeekMessageA(" in source
    assert "SDL" not in source
    assert "pygame" not in source
    assert "turing_display_present_layered(" in source
    assert "#include <stdbool.h>" in source


def test_generated_c_shell_uses_win32_rgba_blit_from_shared_io_contract():
    """A single alpha-blended layer: one NULL-free entry in every array."""

    parameters = (
        Parameter(
            "extent_4", "extent", "int32", "int32_t", "c_int32", "value",
        ),
        *(
            Parameter(
                channel, "output", "float64", "double", "c_double",
                "reference", (4,), "extent_4", channel,
            )
            for channel in ("red", "green", "blue", "alpha")
        ),
    )
    shell_io = {
        "requirements": {
            "requests": [{
                "capability": "display_double_buffer",
                "optional": False,
                "attributes": {
                    "pixel_format": "rgba_f64_planar",
                    "width": 2,
                    "height": 2,
                    "title": "Generated RGBA",
                },
            }],
            "bindings": [
                {
                    "resource": f"display.{channel}",
                    "entry_point": "rgba",
                    "parameter": channel,
                }
                for channel in ("red", "green", "blue", "alpha")
            ],
            "options": [],
        },
        "abi": {},
    }
    api = CompiledProgramAPI(
        module="rgba_fortran",
        language="fortran",
        entry="rgba",
        entry_points=(EntryPoint(
            name="rgba", symbol="rgba", kind="numerical",
            parameters=parameters,
        ),),
        metadata={"shell_io": shell_io},
    )
    module = FortranModule("rgba_fortran", "", api=api)

    source = emit_fortran_c_shell_source(
        module, extent_overrides={"extent_4": 4}
    )

    assert "turing_display_present_layered(" in source
    assert "turing_display_alpha_layers[1] = {" in source
    assert "NULL" not in re.search(
        r"turing_display_alpha_layers\[1\] = \{ (.*?) \};", source
    ).group(1)


def test_generated_c_shell_uses_win32_layered_blit_from_shared_io_contract():
    """Two layers: an opaque base (no alpha binding) under an alpha overlay."""

    def _channel_parameter(name):
        return Parameter(
            name, "output", "float64", "double", "c_double",
            "reference", (4,), "extent_4", name,
        )

    parameters = (
        Parameter(
            "extent_4", "extent", "int32", "int32_t", "c_int32", "value",
        ),
        _channel_parameter("base_r"), _channel_parameter("base_g"),
        _channel_parameter("base_b"),
        _channel_parameter("overlay_r"), _channel_parameter("overlay_g"),
        _channel_parameter("overlay_b"), _channel_parameter("overlay_a"),
    )
    layer_bindings = {
        0: {"red": "base_r", "green": "base_g", "blue": "base_b"},
        1: {
            "red": "overlay_r", "green": "overlay_g", "blue": "overlay_b",
            "alpha": "overlay_a",
        },
    }
    shell_io = {
        "requirements": {
            "requests": [{
                "capability": "display_double_buffer",
                "optional": False,
                "attributes": {
                    "pixel_format": "rgba_f64_planar_layered",
                    "layer_count": 2,
                    "width": 2,
                    "height": 2,
                    "title": "Generated layered",
                },
            }],
            "bindings": [
                {
                    "resource": f"display.layer{layer}.{channel}",
                    "entry_point": "layered",
                    "parameter": parameter,
                }
                for layer, channels in layer_bindings.items()
                for channel, parameter in channels.items()
            ],
            "options": [],
        },
        "abi": {},
    }
    api = CompiledProgramAPI(
        module="layered_fortran",
        language="fortran",
        entry="layered",
        entry_points=(EntryPoint(
            name="layered", symbol="layered", kind="numerical",
            parameters=parameters,
        ),),
        metadata={"shell_io": shell_io},
    )
    module = FortranModule("layered_fortran", "", api=api)

    source = emit_fortran_c_shell_source(
        module, extent_overrides={"extent_4": 4}
    )

    assert "turing_display_present_layered(" in source
    assert "            2," in source
    alpha_array = re.search(
        r"turing_display_alpha_layers\[2\] = \{ (.*?) \};", source
    ).group(1)
    entries = [entry.strip() for entry in alpha_array.split(",")]
    assert entries[0] == "NULL"
    assert entries[1] != "NULL"


def test_generated_c_shell_reads_declared_file_port_into_bound_abi_parameters():
    parameters = (
        Parameter(
            "subject_bytes", "input", "u8", "uint8_t", "c_uint8",
            "reference", (16,), None, "binary_bytes",
        ),
        Parameter(
            "subject_length", "input", "i64", "int64_t", "c_int64",
            "value", source_name="binary_length",
        ),
    )
    api = attach_shell_io(CompiledProgramAPI(
        "machine", "fortran", "load_subject",
        (EntryPoint("load_subject", "load_subject", "control", parameters),),
    ), ShellIOManifest(
        (ShellIORequest.create("files"),),
        system_ports=(SystemPort.create(
            "subject-binary", "file", "input", entry_point="load_subject",
            fields={"data": "binary_bytes", "length": "binary_length"},
            attributes={"maximum_bytes": 16},
        ),),
    ))
    module = FortranModule("machine", "", api=api)

    source = emit_fortran_c_shell_source(module)

    assert 'turing_argument_value(argc, argv, "--file-subject-binary")' in source
    assert "turing_read_file(" in source
    assert "(uint8_t *)slots[0], 16" in source
    assert "*((int64_t *)slots[1]) = (int64_t)loaded_bytes" in source
    assert "short initial state at binary_bytes" not in source


def test_generated_c_shell_exports_native_file_broker_without_python():
    parameter = Parameter(
        "status", "output", "i64", "int64_t", "c_int64",
        "reference", (1,), None, "status",
    )
    api = attach_shell_io(CompiledProgramAPI(
        "file_broker", "fortran", "run",
        (EntryPoint("run", "run", "control", (parameter,)),),
    ), ShellIOManifest((ShellIORequest.create("files"),)))
    module = FortranModule("file_broker", "", api=api)

    source = emit_fortran_c_shell_source(module)

    for operation in (
        "open", "read", "write", "seek", "tell", "flush", "close",
        "stat_size",
    ):
        assert f"turing_shell_file_{operation}(" in source
    assert "TURING_FILE_HANDLE_CAPACITY" in source
    assert "turing_shell_file_close_all();" in source
    assert "static int turing_shell_file_fail(int status)" in source
    assert "Python.h" not in source
    assert "PyObject" not in source


def _native_shell_write_plan():
    return {
        "schema": "turing.python-shell-file-context.v1",
        "function": "write_payload",
        "scope": "file-scope:1:1:1",
        "operations": (
            {
                "operation": "open", "sequence": 0,
                "arguments": (
                    {"kind": "name", "name": "path"},
                    {"kind": "literal", "value": "wb"},
                ),
                "result": "file_handle",
            },
            {
                "operation": "write", "sequence": 1,
                "arguments": (
                    {"kind": "name", "name": "file_handle"},
                    {"kind": "name", "name": "payload"},
                ),
                "result": "count",
            },
            {
                "operation": "flush", "sequence": 2,
                "arguments": ({"kind": "name", "name": "file_handle"},),
                "result": None,
            },
            {
                "operation": "close", "sequence": 3,
                "arguments": ({"kind": "name", "name": "file_handle"},),
                "result": None,
            },
        ),
    }


def _native_shell_write_module():
    source = """
module native_shell_write_fortran
  use, intrinsic :: iso_c_binding
  implicit none
contains
  subroutine write_payload(path_data, path_length, payload_data, &
      payload_length, count) bind(C, name="write_payload")
    integer(c_int8_t), intent(in) :: path_data(512)
    integer(c_int64_t), value :: path_length
    integer(c_int8_t), intent(in) :: payload_data(8)
    integer(c_int64_t), value :: payload_length
    integer(c_int64_t), intent(out) :: count
    count = 0_c_int64_t
  end subroutine write_payload
end module native_shell_write_fortran
"""
    parameters = (
        Parameter(
            "path_data", "input", "u8", "uint8_t", "c_uint8",
            "reference", (512,), None, "path",
        ),
        Parameter(
            "path_length", "input", "i64", "int64_t", "c_int64",
            "value", source_name="path_length", source_transform="sequence_length",
        ),
        Parameter(
            "payload_data", "input", "u8", "uint8_t", "c_uint8",
            "reference", (8,), None, "payload",
        ),
        Parameter(
            "payload_length", "input", "i64", "int64_t", "c_int64",
            "value", source_name="payload_length", source_transform="sequence_length",
        ),
        Parameter(
            "count", "output", "i64", "int64_t", "c_int64",
            "reference", source_name="count",
        ),
    )
    api = CompiledProgramAPI(
        "native_shell_write_fortran", "fortran", "write_payload",
        (EntryPoint(
            "write_payload", "write_payload", "control", parameters,
        ),),
        metadata={
            "shell_io": {
                "boundary_plan_schema": "turing.shell-boundary-plan.v1",
                "boundary_plans": (_native_shell_write_plan(),),
            },
        },
    )
    api = attach_shell_io(
        api, ShellIOManifest((ShellIORequest.create("files"),)),
    )
    return FortranModule("native_shell_write_fortran", source, api=api)


def test_native_shell_consumes_file_boundary_plan_outside_backend_entry():
    module = _native_shell_write_module()

    source = emit_fortran_c_shell_source(module)

    assert "turing_shell_file_open(" in source
    assert "turing_shell_file_write(" in source
    assert "turing_shell_file_flush(" in source
    assert "turing_shell_file_close(" in source
    assert source.index("write_payload(") < source.index(
        "turing_shell_file_open(", source.index("int main(")
    )
    assert "return turing_shell_file_fail(10);" in source
    assert "return turing_shell_file_fail(11);" in source
    assert "turing_shell_file_" not in module.source


@pytest.mark.skipif(
    fortran_compiler() is None,
    reason="no Fortran compiler installed",
)
def test_native_shell_executes_file_plan_around_plain_backend_entry(tmp_path):
    module = _native_shell_write_module()
    destination = (tmp_path / "written-by-shell-plan.bin").resolve()
    encoded_path = os.fsencode(destination)
    path_data = np.zeros(512, dtype=np.uint8)
    path_data[:len(encoded_path)] = np.frombuffer(encoded_path, dtype=np.uint8)
    payload_data = np.frombuffer(b"shell-ok", dtype=np.uint8).copy()

    artifact = compile_fortran_module_c_shell(
        module,
        {
            "path": path_data,
            "path_length": np.asarray([len(encoded_path)], dtype=np.int64),
            "payload": payload_data,
            "payload_length": np.asarray([len(payload_data)], dtype=np.int64),
        },
        tmp_path / "build",
        name="native_shell_boundary_plan",
    )
    payload = json.loads(artifact.run().stdout)

    assert payload["outputs"]["count"] == {"first": 8.0, "sum": 8.0}
    assert destination.read_bytes() == b"shell-ok"
    assert "turing_shell_file_" not in module.source


def test_generated_c_shell_loads_and_publishes_inout_state():
    parameters = (
        Parameter(
            "state", "inout", "float64", "double", "c_double",
            "reference", (4,), "extent_4", "state.values",
        ),
        Parameter(
            "state_snapshot", "input", "float64", "double", "c_double",
            "reference", (4,), "extent_4", "state.values",
        ),
    )
    api = CompiledProgramAPI(
        "resident_state", "fortran", "advance",
        (EntryPoint("advance", "advance", "control", parameters),),
    )
    module = FortranModule("resident_state", "", api=api)

    source = emit_fortran_c_shell_source(module)

    assert "fread(slots[0], sizeof(double), 4, state)" in source
    assert '\\"state.values\\":{\\"first\\":%.17g,\\"sum\\":%.17g}' in source
    assert "((double *)slots[0])[0], sum" in source
    assert "((double *)slots[1])[0], sum" not in source
    assert 'strcmp(argv[argument_index], "--stream-frames") == 0' in source
    assert '\\"event\\":\\"frame\\"' in source
    assert "fflush(stdout);" in source


@pytest.mark.skipif(
    fortran_compiler() is None,
    reason="no Fortran compiler installed",
)
def test_native_c_file_handler_runs_fortran_with_exact_bytes_and_length(tmp_path):
    source = """
module file_subject_fortran
  use, intrinsic :: iso_c_binding
  implicit none
contains
  subroutine inspect_subject(subject_bytes, subject_length, result) &
      bind(C, name="inspect_subject")
    integer(c_int8_t), intent(in) :: subject_bytes(16)
    integer(c_int64_t), value :: subject_length
    integer(c_int64_t), intent(out) :: result
    result = int(subject_bytes(1), c_int64_t) + subject_length
  end subroutine inspect_subject
end module file_subject_fortran
"""
    parameters = (
        Parameter(
            "subject_bytes", "input", "u8", "uint8_t", "c_uint8",
            "reference", (16,), None, "binary_bytes",
        ),
        Parameter(
            "subject_length", "input", "i64", "int64_t", "c_int64",
            "value", source_name="binary_length",
        ),
        Parameter(
            "result", "output", "i64", "int64_t", "c_int64",
            "reference", source_name="result",
        ),
    )
    api = attach_shell_io(CompiledProgramAPI(
        "file_subject_fortran", "fortran", "inspect_subject",
        (EntryPoint("inspect_subject", "inspect_subject", "control", parameters),),
    ), ShellIOManifest(
        (ShellIORequest.create("files"),),
        system_ports=(SystemPort.create(
            "subject-binary", "file", "input", entry_point="inspect_subject",
            fields={"data": "binary_bytes", "length": "binary_length"},
            attributes={"maximum_bytes": 16},
        ),),
    ))
    module = FortranModule("file_subject_fortran", source, api=api)
    subject = tmp_path / "subject.exe"
    subject.write_bytes(b"MZ\x00\xff")

    artifact = compile_fortran_module_c_shell(
        module, {}, tmp_path / "build", name="file_subject_native",
    )
    payload = json.loads(artifact.run(files={"subject-binary": subject}).stdout)

    assert payload["outputs"]["result"] == {"first": 81.0, "sum": 81.0}
