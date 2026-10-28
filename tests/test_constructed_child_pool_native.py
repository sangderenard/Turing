"""Real source constructor and repeated native child-list growth."""
from pathlib import Path

import numpy as np
import pytest

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import concordance_report

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("backend", ["c", "llvm", "fortran"])
def test_constructor_and_two_mapping_fields_grow_from_their_context(backend, tmp_path):
    source = (ROOT / "tests/record_view_corpus/c640_computed_mapping_child.py").read_text()
    source = source[:source.index("def root(")] + '''def root(count: int):
    index = Index()
    for i in range(count):
        index.put(1, 2, 3, 2, i % 3)
    return index.read(1, 2, 3, 2, 0)
'''
    module, outputs, exports = lower_ast_source_to_ssa(
        source, "root", name="constructed_pool_growth",
        extraction_contract=ROOT / "extraction_contracts/program_extraction.yaml",
        progress=lambda _: None, progress_bars=False)
    report = concordance_report(module)
    assert "0 finding(s)" in report, report
    constructor = next(name for name in module.sequence_tables if name.endswith("____init__"))
    descriptors = tuple(module.sequence_tables[constructor].sequences.values())
    domain = next(item for item in descriptors if item.column_dtypes == ("uint8",))
    assert domain.heap_owner_value_id is not None
    assert domain.capacity_policy.value == "dynamic"
    parents = [item for item in descriptors if item.child_table_pool is not None]
    assert len(parents) == 2
    assert all(item.heap_owner_value_id is not None and item.capacity_policy.value == "dynamic" for item in parents)
    entry = next(name for name in exports if name.endswith("__root"))
    count_id = dict(module.functions[entry].metadata["parameter_names"])["count"]
    feeds = {count_id: np.zeros(1, dtype="int64")}
    if backend == "c":
        from src.compiler.ssa_c_backend import emit_ssa_to_c
        artifact = emit_ssa_to_c(module, entry)
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path / "native")
        execution = artifact.prepare_execution(feeds)
    elif backend == "llvm":
        from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution
        artifact = emit_ssa_function_to_llvm(module, entry)
        assert artifact.complete, artifact.shortfalls
        execution = prepare_artifact_execution(compile_artifact(artifact, directory=tmp_path / "native"), feeds)
    else:
        from src.compiler.ssa_fortran_backend import emit_module, FortranCoreNative, fortran_compiler
        from src.compiler.fortran_c_shell import compile_fortran_module_c_shell
        if fortran_compiler() is None:
            pytest.skip("Fortran compiler unavailable")
        artifact = emit_module(module, name="constructed_pool_growth", outputs={entry: outputs[entry]}, extra_roots=(entry,), progress=lambda _: None)
        assert artifact.complete, artifact.shortfalls
        built = compile_fortran_module_c_shell(artifact, {}, tmp_path / "native", entrypoint=entry, library=True)
        execution = FortranCoreNative(built.executable_path, artifact.api.entry_point(entry), artifact.api).prepare_execution(feeds)
    result = outputs[entry][0].id
    for count in (0, 1, 2, 7, 33):
        execution.buffers[count_id][0] = count
        execution.run()
        expected = sum(1114112 + i for i in range(count)) + sum(1114112 + i for i in range(0, count, 3))
        assert int(execution.buffers[result].item()) == expected
