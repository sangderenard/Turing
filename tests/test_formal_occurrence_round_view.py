"""An adaptation round's positional occurrence census closes with that round.

Program: loop corpus c124/c209.  The per-round census named calls by block
index; Casts inserted by the round, and later dead-work reduction, moved the
calls.  Live rows then named a Cast or Ret as the call, and late frame-formal
classification indexed an absent operand (IndexError).
"""
from dataclasses import replace
import pytest
from src.compiler.identity_concordance import (begin_identity_book, end_identity_book,
    current_identity_book, publish_formal_actual_occurrences, concord_compiler_frame_formals,
    CorrelationTable, Derived, Mode, Unresolved)
from src.compiler.concordance_declarations import (FORMAL_ACTUAL_OCCURRENCE as OCC,
    PHYSICAL_CALL_INPUT_ADAPTATION, PHYSICAL_CALL_INPUT_ADAPTATION_ROUND, SSA_VALUE)
from src.compiler.source_formal_occurrence_pruning import (
    close_round_occurrences, occurrence_retirement_sources, live_occurrence_rows)
from src.transmogrifier.ssa import SSAValue, Instr, BasicBlock, Function, IRModule
from tests.test_native_call_input_receipts import _post_value_roots


@pytest.fixture(autouse=True)
def book_context():
    book, token = begin_identity_book()
    try:
        yield book
    finally:
        end_identity_book(token)


def moved_program():
    """One round's census, its closure, then an instruction ahead of the call."""
    formals = [SSAValue(i, "int64") for i in (1, 2)]
    result = SSAValue(4, "int64")
    callee = Function("callee", formals, {"entry": BasicBlock("entry", [
        Instr("Add", list(formals), result), Instr("Ret", [result], None)])})
    actuals = [SSAValue(i, "int64") for i in (11, 12)]
    call = Instr("Call", actuals.copy(), SSAValue(20, "int64"), attributes={"callee": "callee"})
    block = BasicBlock("entry", [call, Instr("Ret", [call.res], None)])
    caller = Function("caller", actuals, {"entry": block})
    converted = SSAValue(30, "int64")
    roots = _post_value_roots("caller", *actuals, call.res, converted)
    _post_value_roots("callee", *formals, result)
    functions = {"caller": caller, "callee": callee}
    book = current_identity_book()
    stated = publish_formal_actual_occurrences(functions, book=book)
    assert len(stated) == 2
    round_scope = book.mint_scope(("physical-call-input-adaptation", ("caller", "callee")))
    round_cell = book.post(PHYSICAL_CALL_INPUT_ADAPTATION_ROUND, (round_scope, 0), 1,
                           stage=PHYSICAL_CALL_INPUT_ADAPTATION, provenance=Derived(tuple(roots)),
                           mode=Mode.CONCORD)
    close_round_occurrences(book, stated, round_cell)
    # The round's input Cast moves the call to index 1.
    call.args[0] = converted
    block.instrs = [Instr("Cast", [actuals[0]], converted), *block.instrs]
    return IRModule(functions=functions, metadata={"identity_book": book}), stated


def test_closed_round_rows_are_not_live_and_later_census_restates():
    module, stated = moved_program()
    book = current_identity_book(); page = book.page(OCC)
    for row in stated:
        cell = book.latest_ref(OCC, row)
        assert isinstance(page.latest(row), Unresolved)
        assert occurrence_retirement_sources(book, cell, module.functions)
    assert live_occurrence_rows(book, tuple(page.rows()), module.functions) == ()
    # Late frame-formal classification takes its own census; no stale index.
    concord_compiler_frame_formals(module)
    live = live_occurrence_rows(book, tuple(page.rows()), module.functions)
    assert sorted((row[4], row[5], page.latest(row)) for row in live) == [(1, 0, 30), (1, 1, 12)]
    assert CorrelationTable._formal_actual_occurrence_findings(module) == []


def test_restated_closed_position_derives_from_its_closure():
    module, stated = moved_program()
    book = current_identity_book(); page = book.page(OCC)
    # The call returns to index 0; its census restates the closed rows.
    block = module.functions["caller"].blocks["entry"]
    block.instrs = block.instrs[1:]
    publish_formal_actual_occurrences(module.functions, book=book)
    for row in stated:
        assert isinstance(page.latest(row), int)
    assert CorrelationTable._formal_actual_occurrence_findings(module) == []


def test_closure_without_its_round_or_prior_statement_refuses():
    module, stated = moved_program()
    book = current_identity_book(); page = book.page(OCC)
    row = stated[0]; cell = book.latest_ref(OCC, row)
    key = (cell.row, cell.column); prior = page.cells[key]
    page.cells[key] = replace(prior, read=tuple(
        r for r in prior.read if r.page != PHYSICAL_CALL_INPUT_ADAPTATION_ROUND))
    assert occurrence_retirement_sources(book, cell, module.functions) is None
    with pytest.raises(ValueError, match="invalid positional"):
        live_occurrence_rows(book, [row], module.functions)
    assert CorrelationTable._formal_actual_occurrence_findings(module)
    page.cells[key] = prior
    with pytest.raises(Exception, match="round ledger"):
        close_round_occurrences(book, stated, book.latest_ref(SSA_VALUE, ("caller", 11)))
