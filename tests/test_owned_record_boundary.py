"""Final ownership integration obeys explicit exports and fails closed."""
from pathlib import Path
import pytest
from tests.owned_record_test_support import lower_before_owned_record_boundary
from tests.test_owned_record_return_source import SOURCE
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.source_owned_record_transfer import integrate_owned_record_public_returns
from src.compiler.ssa_storage_requirements import public_owned_sequence_returns


def lower():
    return lower_before_owned_record_boundary(SOURCE,'root',extraction_contract=ExtractionContract(
        Path(__file__).resolve().parents[1]/'extraction_contracts/program_extraction.yaml'),
        progress=lambda _:None,progress_bars=False)


def test_boundary_requires_exports_and_is_idempotent():
    module,outputs,exports=lower(); name=exports[0]
    before=repr(outputs),tuple((k,repr(v)) for k,v in module.functions.items())
    assert integrate_owned_record_public_returns(module,outputs,())==()
    assert (repr(outputs),tuple((k,repr(v)) for k,v in module.functions.items()))==before
    helpers=tuple(k for k in module.functions if k!=name)
    assert helpers
    assert integrate_owned_record_public_returns(module,outputs,(*helpers,'not_an_export'))==()
    assert (repr(outputs),tuple((k,repr(v)) for k,v in module.functions.items()))==before
    assert public_owned_sequence_returns(module,name)==()
    assert len(integrate_owned_record_public_returns(module,outputs,exports))==1
    ports=public_owned_sequence_returns(module,name); assert len(ports)==1
    integrated=repr(outputs),tuple((k,repr(v)) for k,v in module.functions.items())
    assert integrate_owned_record_public_returns(module,outputs,exports)==()
    assert (repr(outputs),tuple((k,repr(v)) for k,v in module.functions.items()))==integrated
    assert public_owned_sequence_returns(module,name)==ports


def test_boundary_does_not_hide_failed_projection_admission(monkeypatch):
    module,outputs,exports=lower()
    monkeypatch.setattr('src.compiler.source_owned_record_projection_validation.publish_record_owned_public_returns',
                        lambda *_:())
    with pytest.raises(ValueError,match='lacks complete public admission'):
        integrate_owned_record_public_returns(module,outputs,exports)
    assert public_owned_sequence_returns(module,exports[0])==()


def test_real_boundary_rejects_declared_int_with_actual_string_writers():
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_c_backend import emit_ssa_module_to_c
    from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm
    from src.compiler.concordance_declarations import RECORD_OWNED_RETURN_PROJECTION
    text=SOURCE.replace('out.append(i)',"out.append('x')").replace(
        'out.append(-i)',"out.append('y')").replace('out.append(-7)',"out.append('seed')")
    module,outputs,exports=lower_ast_source_to_ssa(text,'root',extraction_contract=ExtractionContract(
        Path(__file__).resolve().parents[1]/'extraction_contracts/program_extraction.yaml'),
        progress=lambda _:None,progress_bars=False)
    name=exports[0]
    assert public_owned_sequence_returns(module,name)==()
    book=module.metadata['identity_book']
    assert not any(row[0]==name for row in book.page(RECORD_OWNED_RETURN_PROJECTION).rows())
    for emit in (emit_ssa_module_to_c,emit_ssa_function_to_llvm):
        artifact=emit(module,name)
        assert not artifact.complete
        assert any('ownership escape contract' in failure.reason for failure in artifact.shortfalls)
