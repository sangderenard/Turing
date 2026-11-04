"""A mixed call inventory cannot assign one actual's class to a formal."""

from pathlib import Path
import inspect

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.compression.octet_lz import _tensor
from src.compiler.concordance_declarations import (
    SOURCE_CALL_ACTUAL_FORMAL, SOURCE_VALUE_CLASS,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import ConcordanceRefusal, IdentityBook


CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "extraction_contracts" / "program_extraction.yaml"
)


def test_unannotated_formal_waits_for_all_reached_actual_classes(monkeypatch):
    source = """def consume(value):
    return len(value)
def root(data: bytes) -> int:
    left = consume(list(data))
    payload = bytes(data) + bytes(data)
    return left + consume(payload)
"""
    captured = []
    original = IdentityBook.post

    def capture(self, page, row, fact, *args, **kwargs):
        result = original(self, page, row, fact, *args, **kwargs)
        if page == SOURCE_CALL_ACTUAL_FORMAL and row[2][1] == "consume":
            captured.append(self)
        return result

    monkeypatch.setattr(IdentityBook, "post", capture)
    try:
        module, _outputs, _exports = lower_ast_source_to_ssa(
            source, "root", name="mixed_formal_classes",
            extraction_contract=str(CONTRACT),
            progress=lambda _message: None, progress_bars=False,
        )
    except ConcordanceRefusal:
        # The source-only call contract is already complete even if a later
        # native consumer refuses to lower a generic sequence operation.
        pass
    assert captured
    book = captured[0]
    edges = tuple(
        row for row in book.page(SOURCE_CALL_ACTUAL_FORMAL).rows()
        if row[2][1] == "consume"
    )
    assert len(edges) == 2
    formal_ids = {row[3] for row in edges}
    assert len(formal_ids) == 1
    classes = {
        book.page(SOURCE_VALUE_CLASS).latest((row[0],
            book.page(SOURCE_CALL_ACTUAL_FORMAL).latest(row)[0].row[1]))[0]
        for row in edges
    }
    assert classes == {"builtins.list", "builtins.bytes"}
    formal_scope = book.page(SOURCE_CALL_ACTUAL_FORMAL).latest(edges[0])[4].row[0]
    assert book.latest_ref(SOURCE_VALUE_CLASS,
                           (formal_scope, next(iter(formal_ids)))) is None


def test_tensor_formal_waits_for_delayed_bytes_add_class():
    # The first call's list class is present during formal propagation. The
    # second call's bytes Add class arrives at the whole-source flush.
    source = inspect.getsource(_tensor) + """
def root(packet: bytes, dictionary: bytes):
    a = _tensor([0])
    b = _tensor(packet + dictionary)
    return a, b
"""
    observed = {}

    def capture(graph):
        from src.compiler.identity_concordance import current_identity_book
        from src.compiler.concorded_literals import origin_cell

        book = current_identity_book()
        caller_scope = graph.function_table.entry("root").graph.G.graph[
            "authored_function_identity"]
        callee_scope = graph.function_table.entry("_tensor").graph.G.graph[
            "authored_function_identity"]
        rows = tuple(row for row in book.page(SOURCE_CALL_ACTUAL_FORMAL).rows()
                     if row[0] == caller_scope and row[2] == callee_scope)
        observed["rows"] = rows
        observed["actuals"] = tuple(
            book.page(SOURCE_CALL_ACTUAL_FORMAL).latest(row)[0] for row in rows)
        observed["classes"] = tuple(tuple(
            book.page(SOURCE_VALUE_CLASS).cell(target.row, target.column)
            for subject in (actual, origin_cell(book, actual))
            for target, _stage in book.edges_out_of(subject)
            if target.page == SOURCE_VALUE_CLASS)
            for actual in observed["actuals"])
        observed["formal"] = book.latest_ref(SOURCE_VALUE_CLASS,
                                             (callee_scope, rows[0][3]))

    lower_ast_source_to_ssa(
        source, "root", name="delayed_tensor_formal_class",
        python_bindings={"AbstractTensor": AbstractTensor},
        extraction_contract=ExtractionContract(CONTRACT),
        progress=lambda _message: None, progress_bars=False,
        resolved_process_graph_sink=capture,
        compilation_unit_plan_sink=lambda _plan: None,
        stop_after_compilation_unit_plan=True,
    )
    assert len(observed["rows"]) == 2
    assert {fact[0] for classes in observed["classes"] for fact in classes} == {
        "builtins.list", "builtins.bytes"}
    assert observed["formal"] is None
