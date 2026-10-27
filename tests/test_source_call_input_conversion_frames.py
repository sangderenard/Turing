"""Native argument conversions are views of the same source-call frame."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable, Derived, Mode
from src.compiler.concordance_declarations import CALL_RECORD, CALL_INPUT_CONVERSION, PHYSICAL_CALL_INPUT_ADAPTATION


SOURCE = '''def consume(begin:int,end:int)->int:
    return begin+end
def root(i:int,choose:bool):
    offsets=[False,True]
    begin=offsets[i]
    end=offsets[i+1]
    if choose:
        return consume(begin,end)+10
    return consume(begin,end)+20
'''


def _lower():
    root = Path(__file__).resolve().parents[1]
    return lower_ast_source_to_ssa(SOURCE, 'root', name='converted_call_frames',
        extraction_contract=root/'extraction_contracts/program_extraction.yaml',
        progress=lambda _:None, progress_bars=False)


def _calls(module, entry):
    function = module.functions[entry]
    return [(label, ins) for label, block in function.blocks.items() for ins in block.instrs
            if ins.op == 'Call' and ins.attributes.get('plan_callsite_id') is not None
            and ins.attributes.get('callee') in module.functions]


@pytest.mark.parametrize('backend', ['c', 'llvm', 'fortran'])
def test_native_duplicate_call_activations_keep_exact_converted_inputs(backend, tmp_path):
    module, outputs, exports = _lower()
    entry = exports[0]
    assert not CorrelationTable.build(module).findings(module)
    calls = _calls(module, entry)
    assert len(calls) == 2
    records = {record.callsite_id: record for record in module.call_table[entry]}
    for _block, call in calls:
        record = records[call.attributes['plan_callsite_id']]
        assert all(value.dtype == 'int64' for value in call.args)
        # The original bool loads remain the semantic arguments; only the
        # exact activation's physical bindings name the minted Cast results.
        semantic = dict((int(formal), int(source)) for source, formal in record.argument_bindings)
        physical = dict((int(formal), int(source)) for formal, _kind, source in record.frame_bindings)
        assert semantic and all(physical[formal] != source for formal, source in semantic.items())
    if backend == 'c':
        from src.compiler.ssa_c_backend import emit_ssa_to_c
        artifact = emit_ssa_to_c(module, entry)
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path)
        prepare = artifact.prepare_execution
    elif backend == 'llvm':
        from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution
        artifact = emit_ssa_function_to_llvm(module, entry)
        assert artifact.complete, artifact.shortfalls
        native = compile_artifact(artifact, directory=tmp_path)
        prepare = lambda feeds: prepare_artifact_execution(native, feeds)
    else:
        from src.compiler.ssa_fortran_backend import emit_module, FortranCoreNative
        from src.compiler.fortran_c_shell import compile_fortran_module_c_shell
        artifact = emit_module(module, outputs={entry:outputs[entry]}, extra_roots=exports, progress=lambda _:None)
        assert artifact.complete, artifact.shortfalls
        native = compile_fortran_module_c_shell(artifact, {}, tmp_path, entrypoint=entry, library=True)
        prepare = FortranCoreNative(native.executable_path, artifact.api.entry_point(entry), artifact.api).prepare_execution
    names = dict(module.functions[entry].metadata['parameter_names'])
    for choose, expected in ((True,11),(False,21)):
        result = prepare({names['i']:np.array([0],dtype='int64'),
                          names['choose']:np.array([choose],dtype='bool')}).run()
        assert result.buffers[outputs[entry][0].id].item() == expected


def test_audit_catches_native_argument_reset_to_semantic_source():
    module, _outputs, exports = _lower()
    entry = exports[0]
    _block, call = _calls(module, entry)[0]
    record = next(record for record in module.call_table[entry]
                  if record.callsite_id == call.attributes['plan_callsite_id'])
    formal = call.attributes['callee_input_ids'][0]
    original = next(source for source, target in record.argument_bindings if target == formal)
    source = next(ins.res for block in module.functions[entry].blocks.values() for ins in block.instrs
                  if ins.res is not None and ins.res.id == original)
    call.args[0] = source
    assert CorrelationTable._physical_call_input_conversion_findings(module)


def test_audit_catches_conversion_from_another_branch_activation():
    module, _outputs, exports = _lower()
    entry = exports[0]
    calls = _calls(module, entry)
    assert calls[0][0] != calls[1][0]
    first, second = calls[0][1], calls[1][1]
    first.args[0] = second.args[0]
    table = module.call_table
    records = tuple(table[entry])
    wrong = replace(records[0], frame_bindings=tuple(
        (formal, kind, int(second.args[0].id)) if formal == first.attributes['callee_input_ids'][0]
        else (formal, kind, source) for formal, kind, source in records[0].frame_bindings))
    table.assign(entry, (wrong,*records[1:]), sources=(records[0].cell,), stage=PHYSICAL_CALL_INPUT_ADAPTATION)
    assert CorrelationTable._physical_call_input_conversion_findings(module)


def test_audit_rejects_conversion_that_no_longer_dominates_its_call():
    module, _outputs, exports = _lower()
    entry = exports[0]
    label, call = _calls(module, entry)[0]
    block = module.functions[entry].blocks[label]
    cast = next(ins for ins in block.instrs
                if ins.res is not None and ins.res.id == call.args[0].id)
    block.instrs.remove(cast)
    block.instrs.insert(block.instrs.index(call)+1, cast)
    assert CorrelationTable._physical_call_input_conversion_findings(module)


def test_audit_requires_pinned_conversion_and_converted_value_causes():
    module, _outputs, exports = _lower()
    entry = exports[0]
    book = module.call_table.book
    records = tuple(module.call_table[entry])
    # Equal-looking physical bindings without the conversion's causal
    # receipt are not proof. Replacing this row severs that exact ancestry.
    source = book.latest_ref(CALL_INPUT_CONVERSION, next(iter(book.page(CALL_INPUT_CONVERSION).rows())))
    book.post(CALL_RECORD, (module.call_table.owner,entry), records,
              stage=PHYSICAL_CALL_INPUT_ADAPTATION, mode=Mode.REVISE,
              provenance=Derived((source,)))
    assert CorrelationTable._physical_call_input_conversion_findings(module)
