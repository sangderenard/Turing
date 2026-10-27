"""Owned typed arrays retain one heap owner through record instance frames."""
from pathlib import Path
import numpy as np
import pytest
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa, compile_fortran_module_c_shell
from src.compiler.identity_concordance import CorrelationTable
from src.compiler.ssa_c_backend import emit_ssa_to_c
from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution
from src.compiler.ssa_fortran_backend import emit_module, FortranCoreNative, fortran_compiler

ISOLATION = """from array import array
class Ledger:
    def __init__(self):
        self.domains = array('B')
    def add(self, value: int):
        self.domains.append(value)
    def read(self, index: int):
        return self.domains[index]
def root(n: int):
    left = Ledger()
    right = Ledger()
    i = 0
    while i < n:
        left.add(i % 256)
        right.add((255 - i) % 256)
        i += 1
    return int(left.read(n - 1)) * 1000 + int(right.read(n - 1))
"""

TEMPORAL = """from array import array
class Ledger:
    def __init__(self):
        self.left = array('B')
        self.right = array('B')
    def step(self, value: int):
        before = len(self.left)
        other = len(self.right)
        self.left.append(value)
        after = len(self.left)
        self.right.append(value)
        return before * 1000 + after * 100 + other * 10 + len(self.right)
def root(domain: int):
    ledger = Ledger()
    first = ledger.step(domain)
    second = ledger.step(domain)
    return first * 10000 + second
"""

@pytest.mark.parametrize('case', ['guarded', 'isolation', 'temporal'])
@pytest.mark.parametrize('lane', ['C', 'LLVM', 'Fortran'])
def test_owned_array_record_native(case, lane, tmp_path):
    source = ((Path(__file__).parent / 'record_view_corpus/c632_guarded_record_length.py').read_text()
              if case == 'guarded' else ISOLATION if case == 'isolation' else TEMPORAL)
    module, outputs, exports = lower_ast_source_to_ssa(
        source, 'root', name='owned_array_' + case,
        extraction_contract=Path(__file__).resolve().parents[1] / 'extraction_contracts/program_extraction.yaml',
        progress=lambda _: None, progress_bars=False)
    assert CorrelationTable.build(module).findings(module) == []
    root = module.functions[exports[0]]
    parameters = dict(root.metadata['parameter_names'])
    returns = [ins.args for block in root.blocks.values() for ins in block.instrs if ins.op == 'Ret']
    assert len(returns) == 1 and len(returns[0]) == 1
    result = returns[0][0].id
    descriptors = tuple(module.sequence_tables[exports[0]].sequences.values())
    arrays = [d for d in descriptors if d.column_dtypes == ('uint8',)]
    assert len(arrays) == (1 if case == 'guarded' else 2)
    owners = [d.heap_owner_value_id for d in arrays]
    assert None not in owners and len(set(owners)) == len(owners)
    if lane == 'C':
        artifact = emit_ssa_to_c(module, exports[0])
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path / 'native')
        prepare = artifact.prepare_execution
    elif lane == 'LLVM':
        artifact = emit_ssa_function_to_llvm(module, exports[0])
        assert artifact.complete, artifact.shortfalls
        built = compile_artifact(artifact, directory=tmp_path / 'native')
        prepare = lambda feeds: prepare_artifact_execution(built, feeds)
    else:
        if fortran_compiler() is None:
            pytest.skip('native Fortran compiler unavailable')
        artifact = emit_module(module, name='owned_array_native', outputs={name: outputs[name] for name in exports}, extra_roots=exports, progress=lambda _: None)
        assert artifact.complete, artifact.shortfalls
        built = compile_fortran_module_c_shell(artifact, {}, tmp_path / 'native', entrypoint=exports[0], library=True)
        core = FortranCoreNative(built.executable_path, artifact.api.entry_point(exports[0]), artifact.api)
        prepare = core.prepare_execution
    for argument in ((1, 25, 260) if case == 'isolation' else (0, 255)):
        feeds = {parameters['n' if case == 'isolation' else 'domain']: np.array([argument], dtype=np.int64)}
        buffers = prepare(feeds).run().buffers
        expected = (100 if case == 'guarded' else 1011212 if case == 'temporal'
                    else ((argument - 1) % 256) * 1000 + (256 - argument) % 256)
        assert buffers[result].item() == expected


def test_record_sequence_heap_owner_is_audited():
    from dataclasses import replace
    from types import SimpleNamespace
    source = (Path(__file__).parent / 'record_view_corpus/c632_guarded_record_length.py').read_text()
    module, _, exports = lower_ast_source_to_ssa(
        source, 'root', name='owned_array_owner_audit',
        extraction_contract=Path(__file__).resolve().parents[1] / 'extraction_contracts/program_extraction.yaml',
        progress=lambda _: None, progress_bars=False)
    table = module.record_tables[exports[0]]
    sequence_table = module.sequence_tables[exports[0]]
    records = {}
    for key, record in table.records.items():
        fields = []
        for field in record.fields:
            sequence = None if field.sequence_id is None else sequence_table.by_id(field.sequence_id)
            owner = None if sequence is None else sequence.heap_owner_value_id
            fields.append(replace(field, value_ids=tuple(value for value in field.value_ids if value != owner)))
        records[key] = replace(record, fields=tuple(fields))
    module.record_tables[exports[0]] = SimpleNamespace(records=records)
    assert any(f.kind == 'record-sequence-owner-missing'
               for f in CorrelationTable.build(module).findings(module))


def test_retained_field_reference_requires_its_declaration(monkeypatch):
    from src.compiler import fortran_c_shell as shell
    original = shell._settle_linked_dynamic_sequences
    def malformed(graphs, edges, *args, **kwargs):
        for graph in graphs:
            for _, data in graph.nodes(data=True):
                attributes = data.get('attributes') or {}
                if attributes.get('field_state_cell') is not None:
                    attributes['field_declaration_cell'] = attributes['field_state_cell']
        return original(graphs, edges, *args, **kwargs)
    monkeypatch.setattr(shell, '_settle_linked_dynamic_sequences', malformed)
    source = (Path(__file__).parent / 'record_view_corpus/c632_guarded_record_length.py').read_text()
    with pytest.raises(ValueError, match='non-field declaration'):
        lower_ast_source_to_ssa(
            source, 'root', name='owned_array_bad_field_ref',
            extraction_contract=Path(__file__).resolve().parents[1] / 'extraction_contracts/program_extraction.yaml',
            progress=lambda _: None, progress_bars=False)
