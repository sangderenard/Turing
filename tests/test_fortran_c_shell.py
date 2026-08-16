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


def test_pursued_re_compile_uses_one_external_multikey_table_contract():
    import ast
    import contextlib
    import io

    from src.common.tensors.topological_reducer import (
        reduce_abstract_tensor_topology,
    )
    from src.compiler.fortran_c_shell import _field_slot_ops
    from src.transmogrifier.graph.graph_express2 import ProcessGraph

    graph = ProcessGraph(materialize_memory=False)
    graph.python_bindings = {"compile_re": re._compile}
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(
            ast.parse(
                "def entry(pattern, flags):\n"
                "    return compile_re(pattern, flags)\n"
            ),
            resolve_unresolved_parents=True,
        )
    reduce_abstract_tensor_topology(graph)
    operations = _field_slot_ops(
        graph.function_table.entry("_compile").graph.G
    )

    declarations = operations[8]
    lookups = operations[10]
    stores = operations[11]
    deletions = operations[12]
    compiled_cache = tuple(
        declaration for declaration in declarations
        if declaration[2] == 4
    )
    assert len(compiled_cache) == 1
    sequence_id, policy, column_count, writable = compiled_cache[0]
    assert (policy, column_count, writable) == ("unique", 4, True)
    assert len(lookups) == len(stores) == len(deletions) == 1
    assert len(lookups[0][1]) == 3
    assert len(stores[0][1]) == 3
    assert lookups[0][2] == stores[0][3] == deletions[0][2] == sequence_id
    assert deletions[0][3] == "external._cache"


def test_pursued_networkx_remove_node_emits_complete_nested_table_ssa():
    import networkx as nx

    from src.common.tensors.accelerator_backends.aot_compile import (
        compile_ast_aot,
    )
    from src.compiler.fortran_c_shell import _class_surface_ssa_program
    from src.compiler.ssa_fortran_backend import emit_module

    source = (
        "def entry(graph, node):\n"
        "    return remove(graph, node)\n"
    )
    compilation = compile_ast_aot(
        source,
        "entry",
        {"graph": 1.0, "node": 1.0},
        python_bindings={"remove": nx.DiGraph.remove_node},
        precompile_only=True,
        bake_mode="whole_program",
        require_planned_shells=True,
        project_captured_hierarchy=False,
    )
    module, outputs, exports = _class_surface_ssa_program(
        compilation, "networkx_remove_probe"
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
    from src.common.tensors.accelerator_backends.aot_compile import (
        compile_ast_aot,
    )
    from src.compiler.fortran_c_shell import _class_surface_ssa_program
    from src.compiler.ssa_fortran_backend import emit_module

    source = (
        "def entry(pattern, flags):\n"
        "    return compile_re(pattern, flags)\n"
    )
    compilation = compile_ast_aot(
        source,
        "entry",
        {"pattern": 1.0, "flags": 0.0},
        python_bindings={"compile_re": re._compile},
        precompile_only=True,
        bake_mode="whole_program",
        require_planned_shells=True,
        project_captured_hierarchy=False,
    )
    module, outputs, exports = _class_surface_ssa_program(
        compilation, "re_compile_probe"
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
    assert "turing_display_present(" in source
    assert "#include <stdbool.h>" in source


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
