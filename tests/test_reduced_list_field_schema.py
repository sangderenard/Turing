"""Unannotated list fields use reduced member identities, never defaults."""
from pathlib import Path

import pytest

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import current_identity_book
from src.compiler.concordance_declarations import SOURCE_RECORD_VIEW_SCHEMA


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("literal,dtype", [
    ("[0, 7]", "int64"),
    ("[False, True]", "bool"),
    ("[0.0, 7.0]", "float64"),
    ("[0, True]", None),
    ("[]", None),
])
def test_list_field_schema_reads_exact_reduced_members(literal, dtype):
    source = f'''class Clock:
    def __init__(self):
        self.epoch = 0
        self.clock = {literal}
class Page:
    def __init__(self, clock: list[int], book: Clock):
        self.clock = clock
        self.book = book
def root(n: int):
    projected = Clock()
    for index in range(n):
        page = Page(clock=projected.clock, book=projected)
    return projected.epoch
'''
    inspected = []

    def inspect(graph):
        for entry in graph.function_table:
            target = getattr(getattr(entry, "graph", None), "G", None)
            if target is None:
                continue
            schema = target.graph["program_abi"]["records"]["Clock"]
            fields = schema["fields"]
            assert fields["epoch"]["dtype"] == "int64"
            if dtype is None:
                assert "clock" not in fields
            else:
                assert fields["clock"] == {
                    "storage": "span", "dtype": dtype, "rank": 1,
                    "mutable": True, "aggregate_kind": "list",
                }
                row = tuple(schema["source_record_view_schema"])
                cell = current_identity_book().latest_ref(SOURCE_RECORD_VIEW_SCHEMA, row)
                fact = current_identity_book().page(SOURCE_RECORD_VIEW_SCHEMA).cell(
                    cell.row, cell.column,
                )
                assert "clock" in dict(fact[1])
            inspected.append(entry.name)

    lower_ast_source_to_ssa(
        source, "root", name="reduced_list_field",
        extraction_contract=ROOT / "extraction_contracts/program_extraction.yaml",
        resolved_process_graph_sink=inspect, stop_after_compilation_unit_plan=True,
        progress=lambda _: None, progress_bars=False,
    )
    assert inspected


@pytest.mark.parametrize("lane", ["C", "LLVM", "Fortran"])
@pytest.mark.parametrize("imported", [False, True])
def test_nested_clock_list_keyword_loop_executes_native(tmp_path, monkeypatch, lane, imported):
    import numpy as np
    from src.compiler.identity_concordance import CorrelationTable
    from src.compiler.ssa_c_backend import emit_ssa_to_c
    from src.compiler.ssa_llvm_backend import (
        emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution,
    )
    from src.compiler.ssa_fortran_backend import (
        emit_module, FortranCoreNative, fortran_compiler,
    )
    from src.compiler.fortran_c_shell import compile_fortran_module_c_shell

    source = '''class Clock:
    def __init__(self):
        self.clock = [9007199254740993]
class Page:
    def __init__(self, clock: list[int], book: Clock):
        self.clock = clock
        self.book = book
def root(n: int):
    projected = Clock()
    for index in range(n):
        page = Page(clock=projected.clock, book=projected)
    return projected.clock[0] + n
'''
    contract = ROOT / "extraction_contracts/program_extraction.yaml"
    if imported:
        import hashlib
        from src.compiler.extraction_contract import ExtractionContract
        module_name = "list_clock_" + hashlib.sha256(str(tmp_path).encode()).hexdigest()[:8]
        definitions, entry = source.split("def root(", 1)
        path = tmp_path / (module_name + ".py")
        path.write_text(definitions, encoding="utf-8")
        monkeypatch.syspath_prepend(str(tmp_path))
        source = f"from {module_name} import Clock, Page\n" + "def root(" + entry
        contract = ExtractionContract(contract).with_sources([(module_name, path)])
    module, outputs, exports = lower_ast_source_to_ssa(
        source, "root", name="native_clock_list_keyword",
        extraction_contract=contract,
        progress=lambda _: None, progress_bars=False,
    )
    assert CorrelationTable.build(module).findings(module) == []
    # Poison the physical field surface while preserving its valid handle.
    # The old audit missed precisely this retirement disagreement.
    from dataclasses import replace
    from types import SimpleNamespace
    table = module.record_tables[exports[0]]
    records = dict(table.records)
    record = next(record for record in records.values() if record.identity == "Page")
    field = next(field for field in record.fields if field.sequence_id is not None)
    poisoned = replace(field, value_ids=(*field.value_ids, 987654321))
    records[record.record_id] = replace(record, fields=tuple(
        poisoned if item is field else item for item in record.fields
    ))
    module.record_tables[exports[0]] = SimpleNamespace(records=records)
    assert any(f.kind == "record-sequence-member-disagreement"
               for f in CorrelationTable.build(module).findings(module))
    module.record_tables[exports[0]] = table
    root = module.functions[exports[0]]
    parameter = dict(root.metadata["parameter_names"])["n"]
    result = next(i.args[0].id for block in root.blocks.values()
                  for i in block.instrs if i.op == "Ret")
    if lane == "C":
        artifact = emit_ssa_to_c(module, exports[0])
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path)
        prepare = artifact.prepare_execution
    elif lane == "LLVM":
        artifact = emit_ssa_function_to_llvm(module, exports[0])
        assert artifact.complete, artifact.shortfalls
        built = compile_artifact(artifact, directory=tmp_path)
        prepare = lambda feeds: prepare_artifact_execution(built, feeds)
    else:
        if fortran_compiler() is None:
            pytest.skip("native Fortran compiler unavailable")
        artifact = emit_module(module, outputs={name: outputs[name] for name in exports},
                               extra_roots=exports)
        assert artifact.complete, artifact.shortfalls
        built = compile_fortran_module_c_shell(artifact, {}, tmp_path,
                                               entrypoint=exports[0], library=True)
        prepare = FortranCoreNative(built.executable_path,
                                   artifact.api.entry_point(exports[0]),
                                   artifact.api).prepare_execution
    for value in (0, 1, 7):
        actual = prepare({parameter: np.array([value], dtype=np.int64)}).run().buffers[result]
        assert int(np.asarray(actual).reshape(-1)[0]) == 2**53 + 1 + value
