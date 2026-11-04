"""Exact raising-guard ranges at admitted intrinsic integer operands."""

from pathlib import Path
from types import SimpleNamespace

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import (
    GUARDED_CALL_ACTUAL_RANGE,
    SOURCE_EXCEPTION_OPERATION,
    SOURCE_SPAN,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable, current_identity_book
import src.compiler.glsl_deployment_strategy as deployment


class _Captured(Exception):
    def __init__(self, book):
        self.book = book


def _book_at_topk(monkeypatch, source):
    original = deployment.publish_intrinsic_named_result_leaves

    def capture(graph, node_id):
        expression = graph.G.nodes[node_id].get("expr_obj")
        if getattr(getattr(expression, "func", None), "attr", None) == "topk":
            raise _Captured(current_identity_book())
        return original(graph, node_id)

    monkeypatch.setattr(deployment, "publish_intrinsic_named_result_leaves", capture)
    policy = ExtractionContract(
        Path(__file__).resolve().parents[1] / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "records": {}, "bindings": [], "values": [
            {"function": "root", "parameter": "values", "storage": "span",
             "dtype": "int64", "rank": 1, "shape": [3],
             "python_type": "src.common.tensors.abstraction.AbstractTensor"},
            {"function": "root", "parameter": "k", "storage": "scalar",
             "dtype": "int64", "rank": 0, "python_type": "builtins.int"},
        ],
    })
    try:
        lower_ast_source_to_ssa(
            source, "root", python_bindings={"AbstractTensor": AbstractTensor},
            name="guarded_intrinsic_topk", extraction_contract=policy,
            tensor_ssa_reference=c_backend_repository_ssa_reference(),
            progress=lambda _item: None, progress_bars=False,
        )
    except _Captured as captured:
        return captured.book
    raise AssertionError("topk publication boundary was not reached")


def test_converted_raising_guards_bound_admitted_topk_actual(monkeypatch):
    book = _book_at_topk(monkeypatch, '''
def root(values, k: int):
    if k < 0:
        raise ValueError("negative k")
    if k > 3:
        raise ValueError("large k")
    return AbstractTensor.topk(values, k=k, dim=0).indices
''')
    rows = tuple(book.page(GUARDED_CALL_ACTUAL_RANGE).rows())
    assert len(rows) == 1
    cell = book.latest_ref(GUARDED_CALL_ACTUAL_RANGE, rows[0])
    assert book.page(GUARDED_CALL_ACTUAL_RANGE).cell(cell.row, cell.column) == (0, 3)
    sources = {source for source, _stage in book.edges_into(cell)}
    terminals = [source for source in sources if source.page == SOURCE_EXCEPTION_OPERATION]
    assert len(terminals) == 2
    for terminal in terminals:
        causes = {source for source, _stage in book.edges_into(terminal)}
        assert any(source.page == SOURCE_SPAN and
                   book.page(SOURCE_SPAN).cell(source.row, source.column).kind == "Raise"
                   for source in causes)
    assert not CorrelationTable._operator_safety_findings(
        SimpleNamespace(metadata={"identity_book": book}, functions={})
    )


def test_nonterminal_conditions_do_not_bound_topk_actual(monkeypatch):
    book = _book_at_topk(monkeypatch, '''
def root(values, k: int):
    if k < 0:
        marker = 1
    if k > 3:
        marker = 2
    return AbstractTensor.topk(values, k=k, dim=0).indices
''')
    assert not tuple(book.page(GUARDED_CALL_ACTUAL_RANGE).rows())
