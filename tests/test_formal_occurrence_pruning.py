from dataclasses import replace
import pytest
from src.compiler.identity_concordance import (begin_identity_book, end_identity_book,
    current_identity_book, publish_formal_actual_occurrences, CorrelationTable, Unresolved)
from src.compiler.concordance_declarations import FORMAL_ACTUAL_OCCURRENCE as OCC, PRUNED_CALLEE_FORMAL
from src.compiler.fortran_c_shell import _prune_unused_callee_formals
from src.compiler.source_formal_occurrence_pruning import occurrence_retirement_sources, live_occurrence_rows
from src.transmogrifier.ssa import SSAValue, Instr, BasicBlock, Function, IRModule
from tests.test_native_call_input_receipts import _post_value_roots

@pytest.fixture(autouse=True)
def book_context():
    book, token=begin_identity_book()
    try:yield book
    finally:end_identity_book(token)


def fixture_program(last=False):
    formals=[SSAValue(i,"int64") for i in (1,2,3)]
    live=(formals[:2] if last else (formals[0],formals[2]))
    result=SSAValue(4,"int64")
    callee=Function("callee",formals,{"entry":BasicBlock("entry",[Instr("Add",list(live),result),Instr("Ret",[result],None)])})
    actuals=[SSAValue(i,"int64") for i in (11,12,13)]
    calls=[Instr("Call",actuals.copy(),SSAValue(20+j,"int64"),attributes={"callee":"callee","callee_input_ids":(1,2,3)}) for j in range(2)]
    caller=Function("caller",actuals,{"entry":BasicBlock("entry",[*calls,Instr("Ret",[calls[-1].res],None)])})
    _post_value_roots("callee",*formals,result);_post_value_roots("caller",*actuals,*(c.res for c in calls))
    functions={"caller":caller,"callee":callee};book=current_identity_book()
    publish_formal_actual_occurrences(functions,book=book)
    assert _prune_unused_callee_formals(functions)==1
    module=IRModule(functions=functions,metadata={"identity_book":book})
    return module,calls


@pytest.mark.parametrize("last",[False,True])
def test_prune_retires_exact_middle_or_last_positions(last):
    module,calls=fixture_program(last);book=current_identity_book();page=book.page(OCC)
    retired=[book.latest_ref(OCC,row) for row in page.rows() if isinstance(page.latest(row),Unresolved)]
    assert len(retired)==(2 if last else 4)
    for cell in retired:assert occurrence_retirement_sources(book,cell,module.functions)
    live=live_occurrence_rows(book,tuple(page.rows()),module.functions)
    assert len(live)==4
    assert all(len(call.args)==2 for call in calls)
    assert CorrelationTable._formal_actual_occurrence_findings(module)==[]


def test_missing_replacement_and_wrong_current_call_refuse(monkeypatch):
    module,calls=fixture_program();book=current_identity_book();page=book.page(OCC)
    row=next(r for r in page.rows() if r[1]==3 and r[5]==2)
    cell=book.latest_ref(OCC,row);assert occurrence_retirement_sources(book,cell,module.functions)
    replacement=book.latest_ref(OCC,(*row[:5],1,row[6]));key=(replacement.row,replacement.column)
    previous=page.cells[key];page.cells[key]=999
    assert occurrence_retirement_sources(book,cell,module.functions) is None
    page.cells[key]=previous
    calls[0].args=list(reversed(calls[0].args))
    assert occurrence_retirement_sources(book,cell,module.functions) is None
    assert CorrelationTable._formal_actual_occurrence_findings(module)


def test_forged_prune_and_shifted_instruction_refuse():
    module,calls=fixture_program();book=current_identity_book();page=book.page(OCC)
    row=next(r for r in page.rows() if r[1]==3 and r[5]==2);cell=book.latest_ref(OCC,row)
    key=(cell.row,cell.column);prior=page.cells[key]
    page.cells[key]=replace(prior,read=tuple(r for r in prior.read if r.page!=PRUNED_CALLEE_FORMAL))
    assert occurrence_retirement_sources(book,cell,module.functions) is None
    with pytest.raises(ValueError,match="invalid positional"):
        live_occurrence_rows(book,[row],module.functions)
    page.cells[key]=prior
    module.functions["caller"].blocks["entry"].instrs.insert(0,Instr("Const",[],SSAValue(99,"int64"),attributes={"value":0}))
    assert occurrence_retirement_sources(book,cell,module.functions) is None


def test_removed_last_still_requires_exact_current_call_and_complete_census():
    module,calls=fixture_program(True);book=current_identity_book();page=book.page(OCC)
    row=next(r for r in page.rows() if r[1]==3);cell=book.latest_ref(OCC,row)
    assert occurrence_retirement_sources(book,cell,module.functions)
    calls[0].args[0]=SSAValue(999,"int64")
    assert occurrence_retirement_sources(book,cell,module.functions) is None


def test_repeated_exact_pruning_preserves_retirement_chain():
    module,calls=fixture_program();book=current_identity_book();callee=module.functions["callee"]
    callee.blocks["entry"].instrs=[Instr("Ret",[callee.args[-1]],None)]
    assert _prune_unused_callee_formals(module.functions)==1
    page=book.page(OCC)
    for row in page.rows():
        if isinstance(page.latest(row),Unresolved):
            assert occurrence_retirement_sources(book,book.latest_ref(OCC,row),module.functions)
    assert len(live_occurrence_rows(book,tuple(page.rows()),module.functions))==2
    assert CorrelationTable._formal_actual_occurrence_findings(module)==[]


def test_omitting_trailing_removed_formal_and_prune_receipt_refuses():
    book=current_identity_book()
    formals=[SSAValue(i,"int64") for i in (1,2,3)]
    actuals=[SSAValue(i,"int64") for i in (11,12,13)]
    callee=Function("callee",formals,{"entry":BasicBlock("entry",[Instr("Ret",[formals[1]],None)])})
    call=Instr("Call",actuals.copy(),SSAValue(20,"int64"),attributes={"callee":"callee"})
    caller=Function("caller",actuals,{"entry":BasicBlock("entry",[call])})
    _post_value_roots("callee",*formals);_post_value_roots("caller",*actuals,call.res)
    functions={"caller":caller,"callee":callee}
    publish_formal_actual_occurrences(functions,book=book)
    assert _prune_unused_callee_formals(functions)==2
    page=book.page(OCC);row=next(r for r in page.rows() if r[1]==2 and r[5]==1)
    cell=book.latest_ref(OCC,row);assert occurrence_retirement_sources(book,cell,functions)
    prior=page.cells[(row,cell.column)]
    kept=tuple(ref for ref in prior.read if not (ref.page in (OCC,PRUNED_CALLEE_FORMAL) and ref.row[1]==3))
    page.cells[(row,cell.column)]=replace(prior,read=kept)
    original_edges=book.edges_into
    book.edges_into=lambda ref: tuple((p,stage) for p,stage in original_edges(ref)
                                      if ref!=cell or p in kept)
    try:assert occurrence_retirement_sources(book,cell,functions) is None
    finally:book.edges_into=original_edges
