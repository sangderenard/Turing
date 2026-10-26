"""A selected mapping value mutates the resident child pool, not a new list."""
from pathlib import Path
import numpy as np
import pytest
import subprocess
import sys

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import concordance_report

ROOT = Path(__file__).resolve().parents[1]


def test_setdefault_append_uses_the_selected_child_row_storage():
    source = (ROOT / "tests/record_view_corpus/c638_mapping_child_append.py").read_text()
    module, _, _ = lower_ast_source_to_ssa(
        source, "root", name="mapping_child_append",
        extraction_contract=ROOT / "extraction_contracts/program_extraction.yaml",
        progress=lambda _: None, progress_bars=False,
    )
    name = next(n for n in module.functions
                if "__put__specialized" in n and "__ssa_sequence_" not in n)
    function = module.functions[name]
    instructions = [i for b in function.blocks.values() for i in b.instrs]
    operations = [i.attributes.get("ssa_sequence_operation") for i in instructions]
    assert "setdefault_lookup" in operations
    assert "table_store" in operations
    append = next(i for i in instructions if i.attributes.get("ssa_sequence_operation") == "append")
    descriptors = module.sequence_tables[name].sequences
    parent = next(d for d in descriptors.values() if d.child_table_pool is not None)
    child = descriptors[int(append.attributes["sequence_id"])]
    pool = parent.child_table_pool
    assert pool.key_columns == ()
    assert len(pool.column_value_ids) == 1
    assert child.capacity_value_id == pool.row_stride_value_id
    definitions = {i.res.id: i for i in instructions if i.res is not None}
    view = definitions[child.column_value_ids[0]]
    assert view.op == "GetElementPtr"
    assert view.args[0].id == pool.column_value_ids[0]
    length = definitions[child.length_address_id]
    assert length.op == "GetElementPtr"
    assert length.args[0].id == pool.length_value_id
    assert append.args[0].id == child.column_value_ids[0]
    phi = next(i for i in instructions if i.op == "Phi"
               and i.attributes.get("binding") == "ssa_mapping_setdefault_result")
    phi_block = next(name for name, block in function.blocks.items() if phi in block.instrs)
    predecessors = {name for name, block in function.blocks.items() if phi_block in block.successors}
    assert set(phi.attributes["incoming_blocks"]) == predecessors
    assert not any(i.attributes.get("binding") == "ssa_local_sequence_initialize"
                   and i.attributes.get("sequence_id") == child.sequence_id for i in instructions)
    report = concordance_report(module)
    assert "0 finding(s)" in report, report
    # Reproduce the old writer's impossible edge: the original missing arm
    # now reaches a validation block, whose tail owns the merge edge.
    missing = next(name for name in function.blocks if name.startswith("mapping_setdefault_missing"))
    incoming = list(phi.attributes["incoming_blocks"])
    original_incoming = tuple(incoming)
    incoming[-1] = missing
    phi.attributes["incoming_blocks"] = tuple(incoming)
    assert "[phi-edge-not-predecessor]" in concordance_report(module)
    phi.attributes["incoming_blocks"] = original_incoming
    owner = next(block for block in function.blocks.values() if view in block.instrs)
    owner.instrs.remove(view)
    function.blocks[next(iter(function.blocks))].instrs.insert(0, view)
    assert "[child-selection-not-guarded]" in concordance_report(module)


def test_reassigned_append_parameter_does_not_publish_its_old_annotation():
    source = (ROOT / "tests/record_view_corpus/c638_mapping_child_append.py").read_text()
    source = source.replace("        self.table.setdefault", "        value = 1.5\n        self.table.setdefault")
    with pytest.raises(ValueError, match="reassigned parameter binding"):
        lower_ast_source_to_ssa(source, "root", name="reassigned_child_payload",
            extraction_contract=ROOT / "extraction_contracts/program_extraction.yaml",
            progress=lambda _: None, progress_bars=False)


def test_nonempty_child_default_does_not_silently_initialize_an_empty_row():
    source = (ROOT / "tests/record_view_corpus/c638_mapping_child_append.py").read_text().replace("key, []", "key, [9]")
    with pytest.raises(ValueError, match="proved empty mutable default"):
        lower_ast_source_to_ssa(source, "root", name="nonempty_child_default",
            extraction_contract=ROOT / "extraction_contracts/program_extraction.yaml",
            progress=lambda _: None, progress_bars=False)


def _exercise_caller_owned_child_pool(backend, tmp_path, *, undersized_pool=False, undersized_outer=False):
    source = (ROOT / "tests/record_view_corpus/c639_caller_mapping_child.py").read_text()
    module, outputs, exports = lower_ast_source_to_ssa(
        source, "root", name="caller_child_pool",
        extraction_contract=ROOT / "extraction_contracts/program_extraction.yaml",
        progress=lambda _: None, progress_bars=False,
    )
    root = next(n for n in exports if n.endswith("__root"))
    parameters = dict(module.functions[root].metadata["parameter_names"])
    field = next(field for record in module.record_tables[root].records.values()
                 if record.identity == "Index" for field in record.fields if field.name == "table")
    descriptor = module.sequence_tables[root].by_id(field.sequence_id)
    pool = descriptor.child_table_pool
    # These are declared public Index.table members, not invented private
    # workspaces. The test supplies four outer rows and four elements per
    # child. The six transactions below use two rows and at most two elements.
    feeds = {parameters[name]: np.zeros(1, dtype="bool" if name == "write" else "int64")
             for name in ("write", "key", "value", "query")}
    feeds[parameters["index"]] = np.zeros(1, dtype="int64")
    # The declared public reference union carries its explicit type tag.
    # Use the same Index record identity that owns the table field above.
    for value_id, tag_id in module.functions[root].metadata.get("native_value_type_tags", ()):
        if value_id == parameters["index"]:
            fact = module.functions[root].metadata["native_value_type_abi"][value_id]
            record = module.record_tables[root].records[value_id]
            feeds[tag_id] = np.array([dict(fact.tags)[record.identity]], dtype="int64")
    for column in descriptor.column_value_ids:
        feeds[column] = np.zeros(4, dtype="int64")
    feeds[descriptor.length_address_id] = np.array([0], dtype="int64")
    feeds[descriptor.capacity_value_id] = np.array([0 if undersized_outer else 4], dtype="int64")
    feeds[descriptor.status_address_id] = np.zeros(1, dtype="int32")
    for column in pool.column_value_ids:
        feeds[column] = np.zeros(16, dtype="int64")
    feeds[pool.length_value_id] = np.zeros(4, dtype="int64")
    feeds[pool.capacity_value_id] = np.array([3 if undersized_pool else 16], dtype="int64")
    feeds[pool.row_stride_value_id] = np.array([4], dtype="int64")
    feeds[pool.status_value_id] = np.zeros(4, dtype="int32")
    if undersized_pool or undersized_outer:
        for column in descriptor.column_value_ids:
            feeds[column][:] = 123
        for column in pool.column_value_ids:
            feeds[column][:] = 111
        feeds[pool.length_value_id][:] = 77
        feeds[pool.status_value_id][:] = 91
    scalar_members = {}
    for record_id, record in module.record_tables[root].records.items():
        if record.identity in {"builtins.int", "builtins.bool"} and record_id in feeds:
            for member in record.fields:
                feeds[member.value_ids[0]] = feeds[record_id]
                scalar_members.setdefault(record_id, []).append(member.value_ids[0])
    if backend == "c":
        from src.compiler.ssa_c_backend import emit_ssa_to_c
        artifact = emit_ssa_to_c(module, root)
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path / "native")
        execution = artifact.prepare_execution(feeds)
    elif backend == "llvm":
        from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution
        artifact = emit_ssa_function_to_llvm(module, root)
        assert artifact.complete, artifact.shortfalls
        execution = prepare_artifact_execution(compile_artifact(artifact, directory=tmp_path / "native"), feeds)
    else:
        from src.compiler.ssa_fortran_backend import emit_module, FortranCoreNative, fortran_compiler
        from src.compiler.fortran_c_shell import compile_fortran_module_c_shell
        if fortran_compiler() is None:
            pytest.skip("Fortran compiler unavailable")
        artifact = emit_module(module, name="caller_child_pool", outputs={root: outputs[root]}, extra_roots=(root,), progress=lambda _: None)
        assert artifact.complete, artifact.shortfalls
        built = compile_fortran_module_c_shell(artifact, {}, tmp_path / "native", entrypoint=root, library=True)
        execution = FortranCoreNative(built.executable_path, artifact.api.entry_point(root), artifact.api).prepare_execution(feeds)
    def set_inputs(write, key, value, query):
        for name, payload in zip(("write", "key", "value", "query"), (write, key, value, query)):
            value_id = parameters[name]
            execution.buffers[value_id][0] = payload
            for member in scalar_members.get(value_id, ()):
                if member in execution.buffers:
                    execution.buffers[member][0] = payload

    if undersized_pool or undersized_outer:
        set_inputs(True, 3, 99, 3)
        before = {value_id: execution.buffers[value_id].copy() for value_id in field.value_ids}
        print("BOUNDS_EXECUTION_BEGIN", backend, flush=True)
        if backend == "fortran":
            # ERROR STOP must run only in the child process below.
            execution.run()
            raise AssertionError("undersized Fortran pool did not stop")
        with pytest.raises(RuntimeError, match="75"):
            execution.run()
        for value_id, snapshot in before.items():
            assert np.array_equal(execution.buffers[value_id], snapshot), value_id
        print("BOUNDS_NO_MUTATION", backend, flush=True)
        return

    big = 2**53 + 1
    transactions = [
        (False, 3, 0, 3, 0), (True, 3, big, 3, big),
        (False, 3, 0, 9, 0), (True, 3, 7, 3, big + 7),
        (True, 9, 11, 9, 11), (False, 3, 0, 3, big + 7),
    ]
    for write, key, value, query, expected in transactions:
        set_inputs(write, key, value, query)
        result = execution.run()
        assert result.buffers[outputs[root][0].id].item() == expected
    assert execution.buffers[descriptor.length_address_id].item() == 2
    assert execution.buffers[pool.length_value_id][:2].tolist() == [2, 1]
    assert execution.buffers[pool.column_value_ids[0]][:2].tolist() == [big, 7]
    assert execution.buffers[pool.column_value_ids[0]][4] == 11
    report = concordance_report(module)
    assert "0 finding(s)" in report, report


@pytest.mark.parametrize("backend", ["c", "llvm", "fortran"])
def test_caller_owned_child_pool_found_missing_and_repeated_mutations(backend, tmp_path):
    _exercise_caller_owned_child_pool(backend, tmp_path)


@pytest.mark.parametrize("backend", ["c", "llvm", "fortran"])
@pytest.mark.parametrize("exhaustion", ["pool", "outer"])
def test_undersized_flattened_pool_refuses_without_initializing_or_appending(backend, exhaustion, tmp_path):
    probe = (
        "from pathlib import Path\n"
        "from tests.test_mapping_child_sequence_views import _exercise_caller_owned_child_pool\n"
        f"_exercise_caller_owned_child_pool({backend!r}, Path({str(tmp_path)!r}), undersized_pool={exhaustion == 'pool'}, undersized_outer={exhaustion == 'outer'})\n"
    )
    result = subprocess.run([sys.executable, "-u", "-c", probe], cwd=ROOT,
                            capture_output=True, text=True, timeout=60)
    if backend == "fortran":
        assert result.returncode != 0, result.stdout + result.stderr
        assert "BOUNDS_EXECUTION_BEGIN fortran" in result.stdout
        assert "ERROR STOP 75" in result.stderr, result.stdout + result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert "BOUNDS_NO_MUTATION " + backend in result.stdout


def test_empty_default_indexed_child_requires_presence_and_bounds_proof():
    source = (ROOT / "tests/record_view_corpus/c639_caller_mapping_child.py").read_text()
    source = source.replace("        total = 0\n        for value in self.table.get(key, ()):\n            total += value\n        return total",
                            "        return self.table.get(key, ())[0]")
    from src.compiler.precompile_to_ssa import SSALoweringIncompleteError
    with pytest.raises(SSALoweringIncompleteError, match="optional-child-index"):
        lower_ast_source_to_ssa(source, "root", name="indexed_optional_child",
            extraction_contract=ROOT / "extraction_contracts/program_extraction.yaml",
            progress=lambda _: None, progress_bars=False)


@pytest.mark.parametrize("backend", ["c", "llvm"])
def test_empty_default_length_is_explicitly_incomplete_until_result_linking_is_proved(backend):
    source = (ROOT / "tests/record_view_corpus/c639_caller_mapping_child.py").read_text()
    source = source.replace("        total = 0\n        for value in self.table.get(key, ()):\n            total += value\n        return total",
                            "        return len(self.table.get(key, ()))")
    module, _, exports = lower_ast_source_to_ssa(source, "root", name="length_optional_child",
        extraction_contract=ROOT / "extraction_contracts/program_extraction.yaml",
        progress=lambda _: None, progress_bars=False)
    root = next(n for n in exports if n.endswith("__root"))
    if backend == "c":
        from src.compiler.ssa_c_backend import emit_ssa_to_c
        artifact = emit_ssa_to_c(module, root)
    else:
        from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm
        artifact = emit_ssa_function_to_llvm(module, root)
    # The len-only method's source return is not yet linked. Its explicit
    # Call shortfall must prevent execution; no first-child length is used
    # as the missing tuple's length.
    assert not artifact.complete
    assert any(item.operation == "Call" for item in artifact.shortfalls)
