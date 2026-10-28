"""Row-level test of the called inventory (compiler/called_inventory.py)."""

import pytest

from src.compiler.called_inventory import (
    called_class, distinct_signatures, post_root_calls,
)
from src.compiler.concordance_declarations import (
    CALLED_INVENTORY, ROOT_CALL_RECORD,
)
from src.compiler.identity_concordance import begin_identity_book, end_identity_book


@pytest.fixture
def book():
    book, token = begin_identity_book(None)
    try:
        yield book
    finally:
        end_identity_book(token)


def test_inventory_is_the_exact_class_set_and_cites_each_call_record(book):
    calls = [{"n": 3}, {"n": 2.5}, {"n": True}, {"n": 7}]
    post_root_calls(book, "root", calls, selected={"n": "float"})
    fact = book.page(CALLED_INVENTORY).latest(("root", "n"))
    assert fact.classes == ("bool", "float", "int")  # exactly found, no wider
    assert fact.selected == "float"
    cell = book.latest_ref(CALLED_INVENTORY, ("root", "n"))
    cited = {ref.row for ref, _stage in book.edges_into(cell)}
    assert cited == {("root", i) for i in range(4)}  # one edge per call record
    assert book.page(ROOT_CALL_RECORD).latest(("root", 1)) == (("n", "float"),)
    assert called_class(book, "root", "n")[0] == "float"


def test_single_class_selects_itself(book):
    post_root_calls(book, "root", [{"n": 1}, {"n": 2}])
    assert called_class(book, "root", "n")[0] == "int"


def test_several_classes_without_a_signature_refuse_by_name(book):
    post_root_calls(book, "root", [{"n": 1}, {"n": 2.0}])
    with pytest.raises(ValueError, match="formal n of root is called with classes"):
        called_class(book, "root", "n")


def test_empty_inventory_refuses_by_name(book):
    post_root_calls(book, "root", [{}] * 0)
    from src.compiler.concordance_declarations import CalledInventoryFact
    from src.compiler.concordance_declarations import INGESTION
    from src.compiler.identity_concordance import Derived, Mode
    cell = book.post(
        ROOT_CALL_RECORD, ("root", 0), (), stage=INGESTION,
        provenance=__import__("src.compiler.identity_concordance", fromlist=["Novel"]).Novel(
            __import__("src.compiler.concordance_declarations", fromlist=["CONTRACT_DEMAND"]).CONTRACT_DEMAND, ()),
        mode=Mode.CONCORD,
    )
    book.post(CALLED_INVENTORY, ("root", "n"), CalledInventoryFact((), None, True),
              stage=INGESTION, provenance=Derived((cell,)), mode=Mode.REVISE)
    with pytest.raises(ValueError, match="no call record or contract reaches formal n of root"):
        called_class(book, "root", "n")


def test_selection_outside_inventory_refuses(book):
    with pytest.raises(ValueError, match="not in its called inventory"):
        post_root_calls(book, "root", [{"n": 1}], selected={"n": "float"})


def test_distinct_signatures():
    sigs = distinct_signatures([{"a": 1, "b": 2}, {"a": 1.5, "b": 2}, {"a": 4, "b": 5}])
    assert sigs == [{"a": "int", "b": "int"}, {"a": "float", "b": "int"}]


def test_internal_call_edges_feed_the_inventory_and_are_cited():
    """A callee reached by int and float actuals: its CALLED_INVENTORY cell
    cites both call-edge cells, each Derived from the caller's class cell."""
    from src.compiler.concordance_declarations import CALL_EDGE_ACTUAL
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.identity_concordance import current_identity_book
    import src.compiler.called_inventory as inv

    seen = {}
    real = inv.post_call_edge_actual

    def spy(book, *args):
        real(book, *args)
        seen["book"] = book

    inv.post_call_edge_actual = spy
    try:
        source = (
            "def f(x):\n    return x * 2\n\n"
            "def root(n: int):\n    return f(n) + f(2.5)\n"
        )
        lower_ast_source_to_ssa(
            source, "root", name="inv_edges", progress=lambda _: None,
            extraction_contract="extraction_contracts/program_extraction.yaml",
            identity_book=None,
        )
    finally:
        inv.post_call_edge_actual = real
    book = seen["book"]
    fact = book.page(CALLED_INVENTORY).latest(("f", "x"))
    assert fact.classes == ("float", "int")
    cell = book.latest_ref(CALLED_INVENTORY, ("f", "x"))
    cited = [ref for ref, _stage in book.edges_into(cell)]
    assert len(cited) == 2
    assert {book.page(CALL_EDGE_ACTUAL).cell(r.row, r.column) for r in cited} == {"int", "float"}
