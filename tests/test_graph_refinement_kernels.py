"""Seconds-sized proofs of the actual source compiler's graph kernels."""
from pathlib import Path
import dataclasses
import json
import pickle
import subprocess
import sys

import numpy as np

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import concordance_report
from src.compiler.ssa_llvm_backend import (
    compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
)


def test_store_value_is_read_only_but_its_destination_is_not():
    from src.compiler.ssa_call_input_adapters import _read_only_feed
    from src.transmogrifier.ssa import BasicBlock, Function, Instr, SSAValue
    value, address = SSAValue(0, "int64"), SSAValue(1, "ptr")
    function = Function("store", [value, address], {"entry": BasicBlock("entry", [
        Instr("Store", [value, address], None), Instr("Ret", [], None),
    ])})
    assert _read_only_feed(function, value.id)
    assert not _read_only_feed(function, address.id)


def test_address_index_is_not_an_alias_of_the_written_arena():
    from src.compiler.ssa_call_input_adapters import _read_only_feed
    from src.transmogrifier.ssa import BasicBlock, Function, Instr, SSAValue
    arena, index, value, address = SSAValue(0, "int64", (5,)), SSAValue(1, "float64"), SSAValue(2, "int64"), SSAValue(3, "ptr")
    function = Function("store_indexed", [arena, index, value], {"entry": BasicBlock("entry", [
        Instr("GetElementPtr", [arena, index], address),
        Instr("Store", [value, address], None), Instr("Ret", [], None),
    ])})
    assert _read_only_feed(function, index.id)
    assert not _read_only_feed(function, arena.id)


def test_loop_entry_reads_use_the_recorded_entry_even_when_only_one_entry_has_the_seed():
    from types import SimpleNamespace
    from src.compiler.precompile_to_ssa import _ControlSSABuilder
    from src.transmogrifier.ssa import SSAValue
    key = ("control", 111)
    before, reserved, current, completed, leaked = [SSAValue(value, "int64") for value in range(5)]
    rows = {
        "loop_entry_state": {(*key, 84): (("position",), 84)},
        "loop_carried_entry": {(*key, "position"): 0},
    }
    book = SimpleNamespace(page=lambda name: SimpleNamespace(latest=lambda row: rows[name].get(row)))
    builder = object.__new__(_ControlSSABuilder)
    builder._book = lambda: book
    builder.enclosing_loop_states = [key]
    builder.loop_frames = {key: {
        "carried": [(90, 84, before, reserved, current)],
        "latch": False, "updates": {0: completed},
    }}
    builder.external_value = lambda value_id: leaked
    builder.external_values = {90: leaked}
    assert builder._read(84, {"position"}, "heap guard") is current
    builder.loop_frames[key]["latch"] = True
    assert builder._read(84, {"position"}, "heap latch guard") is completed


def test_capture_audit_refuses_a_source_and_target_with_different_value_identities():
    from types import SimpleNamespace
    from src.compiler.concordance_declarations import BindingFact
    from src.compiler.identity_concordance import CorrelationTable, begin_identity_book, end_identity_book
    book, token = begin_identity_book()
    try:
        scope = ("capture_test", 0)
        book.page("name_binding").set((scope, "size", 0), 0, BindingFact(84, True, (None,) * 4, ""))
        book.page("name_binding").set((scope, "position", 0), 0, BindingFact(90, True, (None,) * 4, ""))
        book.page("name_binding_capture").set((scope, "position", 0), 0, ("size", 0, 117))
        module = SimpleNamespace(metadata={"identity_book": book}, functions={})
        findings = CorrelationTable._name_binding_capture_findings(module)
        assert len(findings) == 1 and findings[0].kind == "name-binding-capture-disagreement"
        book.page("name_binding").set((scope, "position", 0), 1, BindingFact(84, True, (None,) * 4, ""))
        assert CorrelationTable._name_binding_capture_findings(module) == []
    finally:
        end_identity_book(token)


def test_capture_reads_index_the_sealed_binding_scope_once():
    from types import SimpleNamespace
    from src.compiler.precompile_to_ssa import _ControlSSABuilder
    from src.transmogrifier.ssa import SSAValue
    key, scope = ("control", 117), ("capture", 0)
    row = (scope, "position", 0)
    rows = {
        "loop_entry_state": {(*key, 84): (("size",), 84)},
        "loop_carried_entry": {(*key, "size"): 0},
        "name_binding_capture": {row: ("size", 0, 117)},
        "name_binding": {row: SimpleNamespace(value_id=84), (scope, "size", 0): SimpleNamespace(value_id=84)},
    }
    scans = []
    def page(name):
        return SimpleNamespace(latest=lambda key: rows[name].get(key),
            scope_rows=lambda owner: (scans.append(owner) or tuple(rows[name])))
    builder = object.__new__(_ControlSSABuilder)
    builder._book = lambda: SimpleNamespace(page=page)
    builder.lexical_read_scope = scope
    builder.enclosing_loop_states = [key]
    before, updated, current = [SSAValue(value, "int64") for value in range(3)]
    builder.loop_frames = {key: {"carried": [(116, 84, before, updated, current)], "latch": False}}
    builder.external_value = lambda value_id: before
    builder.external_values = {84: before}
    for _ in range(3):
        assert builder._read(84, {"position"}, "capture") is current
    assert len(scans) == 1


def test_indexed_element_type_reaches_its_callers_through_the_existing_fixed_point():
    from src.compiler.identity_concordance import begin_identity_book, end_identity_book
    from src.compiler.ssa_result_type_resolution import settle_call_result_types
    from src.transmogrifier.ssa import BasicBlock, Function, Instr, SSAValue
    base, index, address, loaded = (
        SSAValue(0, "int64", (5,)), SSAValue(1, "int64"),
        SSAValue(2, "ptr"), SSAValue(3, "float64"),
    )
    read = Function("read", [base], {"entry": BasicBlock("entry", [
        Instr("Const", [], index, attributes={"value": 0}),
        Instr("GetElementPtr", [base, index], address, attributes={"lowered_from": "Indexed"}),
        Instr("Load", [address], loaded, attributes={"lowered_from": "Indexed"}),
        Instr("Ret", [loaded], None),
    ])})
    actual, result = SSAValue(0, "int64", (5,)), SSAValue(1, "float64")
    root = Function("root", [actual], {"entry": BasicBlock("entry", [
        Instr("Call", [actual], result, attributes={"callee": "read"}),
        Instr("Ret", [result], None),
    ])})
    functions = {"root": root, "read": read}

    def values(function):
        return {value.id: value for value in [
            *function.args,
            *(item.res for block in function.blocks.values() for item in block.instrs if item.res is not None),
        ]}

    book, token = begin_identity_book()
    try:
        settle_call_result_types(functions, lambda name, function: function.blocks["entry"].instrs[-1].args, values)
        assert loaded.dtype == "int64"
        assert result.dtype == "int64"
    finally:
        end_identity_book(token)


def test_integer_load_is_a_value_before_an_aliasing_store(tmp_path):
    from src.transmogrifier.ssa import BasicBlock, Function, IRModule, Instr, SSAValue
    arena = SSAValue(0, "int64", shape=(2,))
    zero, one = SSAValue(1, "int64"), SSAValue(2, "int64")
    first, second = SSAValue(3, "ptr"), SSAValue(4, "ptr")
    old, replacement = SSAValue(5, "int64"), SSAValue(6, "int64")
    function = Function("integer_pop", [arena], {"entry": BasicBlock("entry", [
        Instr("Const", [], zero, attributes={"value": 0}),
        Instr("Const", [], one, attributes={"value": 1}),
        Instr("GetElementPtr", [arena, zero], first),
        Instr("Load", [first], old),
        Instr("GetElementPtr", [arena, one], second),
        Instr("Load", [second], replacement),
        Instr("Store", [replacement, first], None),
        Instr("Ret", [old], None),
    ])})
    artifact = emit_ssa_function_to_llvm(IRModule({function.name: function}), function.name)
    assert artifact.complete, artifact.shortfalls
    native = compile_artifact(artifact, directory=tmp_path)
    execution = prepare_artifact_execution(native, {arena.id: np.array([17, 31], dtype=np.int64)}).run()
    assert int(execution.buffers[old.id]) == 17


def compile_kernel(tmp_path, source=None, *, entrypoint="topological_order", names=None):
    repository = Path(__file__).resolve().parents[1]
    if source is None:
        source = (repository / "src/compiler/graph_refinement_kernels.py").read_text()
    names = names or ("n", "edge_count", "offsets", "targets", "indegree", "heap", "output", "status")
    contract = ExtractionContract(repository / "extraction_contracts/program_extraction.yaml")
    contract = contract.with_program_abi({
        "records": {}, "bindings": [], "values": [
            {"function": entrypoint, "parameter": name,
             "storage": "scalar" if name in {"n", "edge_count"} else "span",
             "dtype": "int64", "rank": 0 if name in {"n", "edge_count"} else 1,
             "python_type": "builtins.int" if name in {"n", "edge_count"}
                            else "src.common.tensors.abstraction.AbstractTensor",
             "mutable": name not in {"offsets", "targets"}}
            for name in names
        ],
    }).with_execution_file(repository / "extraction_contracts/vehicle_full_native_execution.yaml")
    module, _, _ = lower_ast_source_to_ssa(
        source, entrypoint, name="compiler_graph_kernel",
        extraction_contract=contract, runtime_closure_only=True,
        python_bindings={},
    )
    report = concordance_report(module)
    print(report, flush=True)
    root = module.functions[f"compiler_graph_kernel__{entrypoint}"]
    assert report.splitlines()[0].endswith(", 0 finding(s)")
    artifact = emit_ssa_function_to_llvm(module, root.name)
    assert artifact.complete, artifact.shortfalls
    artifact = compile_artifact(artifact, directory=tmp_path)
    return artifact, dict(root.metadata["parameter_names"]), names


def execute_kernel(compiled, inputs=None):
    artifact, parameters, names = compiled
    arrays = [np.array(values, dtype=np.int64) for values in (inputs if inputs is not None else (
        5, 5, [0, 1, 3, 4, 5, 5], [3, 2, 3, 4, 4], [0] * 5,
        [0] * 5, [0] * 5, [0],
    ))]
    # A failing native compiler micro-test must not corrupt pytest's process
    # or leave Windows' crash reporter holding it indefinitely. The parent
    # retains the audit; the child receives only the runtime artifact/arenas.
    saved = artifact.library_path.parent / "native_graph_probe.pkl"
    saved.write_bytes(pickle.dumps((dataclasses.replace(artifact, emission=None), parameters, names, arrays)))
    worker = Path(__file__).resolve().parents[1] / "tools/compiler_probes/graph_kernel_native_worker.py"
    completed = subprocess.run([sys.executable, "-u", str(worker), str(saved)],
                               capture_output=True, text=True, timeout=10)
    print(completed.stdout, flush=True)
    assert completed.returncode == 0, f"native graph worker exit={completed.returncode}: {completed.stderr[-2000:]}"
    payload = next(line.removeprefix("GRAPH_KERNEL_ARRAYS=") for line in completed.stdout.splitlines()
                   if line.startswith("GRAPH_KERNEL_ARRAYS="))
    arrays = [np.asarray(value, dtype=np.int64) for value in json.loads(payload)]
    print("kernel input arenas", [array.tolist() for array in arrays], flush=True)
    return arrays


def test_own_compiler_indegree_kernel_has_isolated_parallel_arenas(tmp_path):
    import ast
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    repository = Path(__file__).resolve().parents[1]
    authored = (repository / "src/compiler/graph_refinement_kernels.py").read_text()
    function = next(node for node in ast.parse(authored).body
                    if isinstance(node, ast.FunctionDef) and node.name == "indegree_counts")
    source = ast.get_source_segment(authored, function)
    artifact, parameters, names = compile_kernel(tmp_path, source,
        entrypoint="indegree_counts", names=("edge_count", "targets", "indegree"))
    artifact.entry()  # Load shared code once, before any worker starts.
    assert "@turing_validation_error(" not in artifact.llvm_ir
    rendezvous = Barrier(2)

    def run(targets):
        spans = (np.array(len(targets), dtype=np.int64), np.array(targets, dtype=np.int64), np.zeros(5, dtype=np.int64))
        execution = prepare_artifact_execution(artifact, {parameters[name]: value for name, value in zip(names, spans)})
        rendezvous.wait(timeout=5)
        execution.run()
        return spans[2]

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, targets) for targets in ([3, 2, 3, 4, 4], [0, 0, 1])]
        results = [future.result(timeout=5) for future in futures]
    np.testing.assert_array_equal(results[0], [0, 0, 1, 2, 2])
    np.testing.assert_array_equal(results[1], [2, 1, 0, 0, 0])


def test_public_compiler_lowers_and_executes_integer_graph_kernel(tmp_path):
    arrays = execute_kernel(compile_kernel(tmp_path))
    np.testing.assert_array_equal(arrays[6], [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(arrays[7], [5])


def test_graph_kernel_empty_cycle_and_both_heap_directions(tmp_path):
    compiled = compile_kernel(tmp_path)
    cases = (
        (0, [0], [], []),
        (6, [0] * 7, [], [0, 1, 2, 3, 4, 5]),
        (6, [0, 1, 1, 2, 2, 2, 2], [1, 0], [2, 0, 1, 3, 4, 5]),
        (3, [0, 1, 2, 2], [1, 0], [2]),
    )
    for n, offsets, targets, expected in cases:
        capacity = max(1, n)
        arrays = execute_kernel(compiled, (
            n, len(targets), offsets, targets or [0], [0] * capacity,
            [0] * capacity, [0] * capacity, [0],
        ))
        np.testing.assert_array_equal(arrays[6][:len(expected)], expected)
        np.testing.assert_array_equal(arrays[7], [len(expected)])


def test_graph_kernel_initial_frontier_preserves_conditional_carried_count(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    source = (repository / "src/compiler/graph_refinement_kernels.py").read_text()
    source = source.split("    emitted =", 1)[0] + "    status[0] = size\n    return status\n"
    arrays = execute_kernel(compile_kernel(tmp_path, source))
    np.testing.assert_array_equal(arrays[4], [0, 0, 1, 2, 2])
    np.testing.assert_array_equal(arrays[5][:2], [0, 1])
    np.testing.assert_array_equal(arrays[7], [2])


def test_graph_kernel_first_heap_pop_keeps_the_selected_value(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    source = (repository / "src/compiler/graph_refinement_kernels.py").read_text()
    source = source.split("        edge = offsets[node]", 1)[0] + "    status[0] = output[0]\n    return status\n"
    source = source.replace("while size > 0:", "while size > 0 and emitted < 1:")
    arrays = execute_kernel(compile_kernel(tmp_path, source))
    np.testing.assert_array_equal(arrays[5][:2], [1, 1])
    np.testing.assert_array_equal(arrays[7], [0])


def test_graph_kernel_first_two_pops_publish_the_next_frontier(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    source = (repository / "src/compiler/graph_refinement_kernels.py").read_text()
    source = source.replace("while size > 0:", "while size > 0 and emitted < 2:")
    arrays = execute_kernel(compile_kernel(tmp_path, source))
    np.testing.assert_array_equal(arrays[6][:2], [0, 1])
    np.testing.assert_array_equal(arrays[4], [0, 0, 0, 0, 2])
    np.testing.assert_array_equal(arrays[5][:2], [2, 3])
    np.testing.assert_array_equal(arrays[7], [2])


def test_heap_push_keeps_size_distinct_from_position_with_the_same_seed(tmp_path):
    source = """def push(n, heap, status):
    size = n
    position = size
    parent = (position - 1) // 2
    while position > 0:
        if heap[parent] <= 0:
            break
        heap[position] = heap[parent]
        position = parent
        parent = (position - 1) // 2
    heap[position] = 0
    size = size + 1
    status[0] = size
    return status
"""
    compiled = compile_kernel(tmp_path, source, entrypoint="push", names=("n", "heap", "status"))
    arrays = execute_kernel(compiled, (5, [1, 2, 3, 4, 5, 0], [0]))
    np.testing.assert_array_equal(arrays[1], [0, 2, 1, 4, 5, 3])
    np.testing.assert_array_equal(arrays[2], [6])


def test_break_before_assignment_does_not_take_another_bindings_available_value(tmp_path):
    source = """def stop(n, status):
    child = n
    position = n - n
    while child < n + 4:
        child = child + 1
        if child > n:
            break
        position = child
    status[0] = position
    return status
"""
    compiled = compile_kernel(tmp_path, source, entrypoint="stop", names=("n", "status"))
    arrays = execute_kernel(compiled, (5, [0]))
    np.testing.assert_array_equal(arrays[1], [0])


def test_alias_captured_inside_a_loop_reads_that_iterations_source_binding(tmp_path):
    source = """def repeat(n, status):
    size = n - n
    iteration = n - n
    while iteration < n:
        position = size
        if position > 0:
            status[0] = status[0] + 1
        size = size + 1
        iteration = iteration + 1
    return status
"""
    compiled = compile_kernel(tmp_path, source, entrypoint="repeat", names=("n", "status"))
    arrays = execute_kernel(compiled, (3, [0]))
    np.testing.assert_array_equal(arrays[1], [2])


def test_nested_conditional_rebinding_keeps_another_names_shared_source(tmp_path):
    source = """def choose(n, status):
    incumbent = n
    proposed = incumbent
    if n > 0:
        if n > 1:
            proposed = n + 3
        else:
            proposed = n + 2
    status[0] = incumbent
    status[1] = proposed
    if proposed != incumbent:
        status[2] = 1
    return status
"""
    compiled = compile_kernel(tmp_path, source, entrypoint="choose", names=("n", "status"))
    for n, expected in ((-3, [-3, -3, 0]), (1, [1, 3, 1]), (5, [5, 8, 1])):
        arrays = execute_kernel(compiled, (n, [0, 0, 0]))
        np.testing.assert_array_equal(arrays[1], expected)


def test_carried_arm_rejects_true_only_binding_on_false_edge():
    from src.compiler.precompile_to_ssa import _ControlSSABuilder
    from src.transmogrifier.ssa import BasicBlock, Instr, SSAValue
    before, child = SSAValue(0, 'ptr'), SSAValue(1, 'ptr')
    builder = object.__new__(_ControlSSABuilder)
    builder.arguments = [before]
    builder.blocks = {
        'entry': BasicBlock('entry', [], successors=['true', 'false']),
        'true': BasicBlock('true', [Instr('PtrCast', [before], child)], successors=['merge']),
        'false': BasicBlock('false', [], successors=['merge']),
        'merge': BasicBlock('merge'),
    }
    builder.external_values = {17: before, 462: child}
    builder.constant_value_ids = set()
    builder.concorded_value_aliases = {148: 17, 462: 17}
    builder._binding_cell = lambda _id: 'binding'
    builder.current = builder.blocks['true']
    assert builder._carried_name_arm(462, 148, before, None, path='true') == (child, 'binding')
    builder.current = builder.blocks['false']
    assert builder._carried_name_arm(462, 148, before, None, path='false') == (before, None)


def test_conditional_preservation_audit_reads_the_existing_name_binding_facts():
    from types import SimpleNamespace
    from src.compiler.concordance_declarations import BindingFact
    from src.compiler.identity_concordance import CorrelationTable, begin_identity_book, end_identity_book
    from src.transmogrifier.ssa import BasicBlock, Function, Instr, SSAValue
    book, token = begin_identity_book()
    try:
        scope = ('conditional_test', 0)
        names = book.page('name_binding')
        for name, version, value_id in (('incumbent', 0, 366), ('proposed', 0, 366), ('proposed', 1, 374)):
            names.set((scope, name, version), 0, BindingFact(value_id, True, (None,) * 4, ''))
        phi = Instr('Phi', [SSAValue(366, 'int64')], SSAValue(374, 'int64'), attributes={
            'preserved_name_binding_value_ids': (366,), 'preserved_name_binding_scope': scope,
            'preserved_name_binding_merge_id': 374,
        })
        module = SimpleNamespace(metadata={'identity_book': book}, functions={
            'choose': Function('choose', [], {'entry': BasicBlock('entry', [phi])})})
        assert CorrelationTable._conditional_binding_preservation_findings(module) == []
        phi.attributes['preserved_name_binding_value_ids'] = (374,)
        findings = CorrelationTable._conditional_binding_preservation_findings(module)
        assert len(findings) == 1 and findings[0].kind == 'conditional-binding-preservation-disagreement'
    finally:
        end_identity_book(token)
