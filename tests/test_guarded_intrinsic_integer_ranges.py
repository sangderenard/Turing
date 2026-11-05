"""Exact raising-guard ranges at admitted intrinsic integer operands."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import (
    FUNCTION_ADDRESS,
    GUARDED_CALL_ACTUAL_RANGE,
    PROCESS_GRAPH_OPERANDS,
    SOURCE_AGGREGATE_LAYOUT,
    SOURCE_EXCEPTION_OPERATION,
    SOURCE_FUNCTION_REACHABILITY,
    SOURCE_LITERAL_VALUE,
    SOURCE_INT_RANGE,
    SOURCE_INTRINSIC_ADMISSION,
    SOURCE_OPERATOR_KIND,
    SOURCE_SPAN,
    SOURCE_VALUE_CLASS,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable, current_identity_book
from src.compiler.ssa_c_backend import emit_ssa_module_to_c
import src.compiler.glsl_deployment_strategy as deployment


def _intrinsic_guard_rows(book):
    """Rows whose exact receipt cites an admitted intrinsic call."""
    return tuple(
        row for row in book.page(GUARDED_CALL_ACTUAL_RANGE).rows()
        if any(source.page == SOURCE_INTRINSIC_ADMISSION
               for source, _ in book.edges_into(
                   book.latest_ref(GUARDED_CALL_ACTUAL_RANGE, row)))
    )


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
    rows = tuple(_intrinsic_guard_rows(book))
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
    assert not tuple(_intrinsic_guard_rows(book))


def test_returned_tuple_member_joins_helper_and_caller_raising_guards(monkeypatch):
    book = _book_at_topk(monkeypatch, '''
def bounded(k: int):
    if k < 1:
        raise ValueError("small")
    return (0, 0, k)

def root(values, k: int):
    selected = bounded(k)[2]
    if selected > 3:
        raise ValueError("large")
    return AbstractTensor.topk(values, k=selected, dim=0).indices
''')
    rows = tuple(_intrinsic_guard_rows(book))
    assert len(rows) == 1
    cell = book.latest_ref(GUARDED_CALL_ACTUAL_RANGE, rows[0])
    assert book.page(GUARDED_CALL_ACTUAL_RANGE).cell(cell.row, cell.column) == (1, 3)
    causes = {source for source, _stage in book.edges_into(cell)}
    assert sum(source.page == SOURCE_EXCEPTION_OPERATION for source in causes) == 2
    assert any(source.page == PROCESS_GRAPH_OPERANDS for source in causes)
    assert any(source.page == SOURCE_LITERAL_VALUE for source in causes)
    assert any(source.page == SOURCE_AGGREGATE_LAYOUT for source in causes)
    assert any(source.page == SOURCE_FUNCTION_REACHABILITY for source in causes)
    assert any(source.page == FUNCTION_ADDRESS for source in causes)


def test_recursive_tuple_return_does_not_prove_a_topk_bound(monkeypatch):
    book = _book_at_topk(monkeypatch, '''
def bounded(k: int):
    if k < 1:
        raise ValueError("small")
    return (0, 0, bounded(k)[2])

def root(values, k: int):
    selected = bounded(k)[2]
    if selected > 3:
        raise ValueError("large")
    return AbstractTensor.topk(values, k=selected, dim=0).indices
''')
    assert not tuple(_intrinsic_guard_rows(book))


def test_exact_nonnegative_integer_power_bounds_returned_topk_count(monkeypatch):
    book = _book_at_topk(monkeypatch, '''
def bounded(k: int):
    if k < 1:
        raise ValueError("small")
    return (0, 0, k)

def root(values, k: int):
    selected = bounded(k)[2]
    if selected > 2 ** 12:
        raise ValueError("large")
    return AbstractTensor.topk(values, k=selected, dim=0).indices
''')
    rows = tuple(_intrinsic_guard_rows(book))
    assert len(rows) == 1
    guarded = book.latest_ref(GUARDED_CALL_ACTUAL_RANGE, rows[0])
    assert book.page(GUARDED_CALL_ACTUAL_RANGE).cell(
        guarded.row, guarded.column,
    ) == (1, 4096)
    powers = tuple(
        row for row in book.page(SOURCE_INT_RANGE).rows()
        if book.page(SOURCE_INT_RANGE).latest(row) == (4096, 4096)
        and book.page(SOURCE_VALUE_CLASS).latest(row) is not None
        and book.page(SOURCE_VALUE_CLASS).latest(row)[:2] == ("builtins.int", 1)
    )
    assert len(powers) == 1
    power = book.latest_ref(SOURCE_INT_RANGE, powers[0])
    sources = {source for source, _stage in book.edges_into(power)}
    assert any(source.page == SOURCE_OPERATOR_KIND for source in sources)
    assert any(source.page == PROCESS_GRAPH_OPERANDS for source in sources)
    assert sum(source.page == SOURCE_INT_RANGE for source in sources) == 2


def test_negative_or_unproved_power_does_not_bound_topk_count(monkeypatch):
    for exponent in ("-1", "exponent"):
        book = _book_at_topk(monkeypatch, f'''
def bounded(k: int):
    if k < 1:
        raise ValueError("small")
    return (0, 0, k)

def root(values, k: int, exponent: int):
    selected = bounded(k)[2]
    if selected > 2 ** {exponent}:
        raise ValueError("large")
    return AbstractTensor.topk(values, k=selected, dim=0).indices
''')
        rows = _intrinsic_guard_rows(book)
        assert len(rows) == 1
        assert book.page(GUARDED_CALL_ACTUAL_RANGE).latest(rows[0]) == (1, None)


def test_power_range_from_other_function_does_not_bound_topk_count(monkeypatch):
    book = _book_at_topk(monkeypatch, '''
def known(exponent: int):
    return 2 ** exponent

def bounded(k: int):
    if k < 1:
        raise ValueError("small")
    return (0, 0, k)

def root(values, k: int, exponent: int):
    unrelated = known(12)
    selected = bounded(k)[2]
    if selected > 2 ** exponent:
        raise ValueError("large")
    return AbstractTensor.topk(values, k=selected, dim=0).indices
''')
    assert any(
        book.page(SOURCE_INT_RANGE).latest(row) == (4096, 4096)
        and "known" in str(row[0])
        for row in book.page(SOURCE_INT_RANGE).rows()
    )
    rows = _intrinsic_guard_rows(book)
    assert len(rows) == 1
    assert book.page(GUARDED_CALL_ACTUAL_RANGE).latest(rows[0]) == (1, None)


def test_zero_power_and_unproved_bool_base_keep_exact_range_rule(monkeypatch):
    for expression, expected in (("0 ** 0", (1, 1)), ("True ** 0", (1, None))):
        book = _book_at_topk(monkeypatch, f'''
def bounded(k: int):
    if k < 1:
        raise ValueError("small")
    return (0, 0, k)

def root(values, k: int):
    selected = bounded(k)[2]
    if selected > {expression}:
        raise ValueError("large")
    return AbstractTensor.topk(values, k=selected, dim=0).indices
''')
        rows = tuple(_intrinsic_guard_rows(book))
        assert len(rows) == 1
        cell = book.latest_ref(GUARDED_CALL_ACTUAL_RANGE, rows[0])
        assert book.page(GUARDED_CALL_ACTUAL_RANGE).cell(
            cell.row, cell.column,
        ) == expected


@pytest.mark.parametrize("upper", ("3", "2 ** 12"))
def test_returned_tuple_guard_selects_native_topk_indices(tmp_path, upper):
    source = f'''
def bounded(k: int):
    if k < 1:
        raise ValueError("small")
    return (0, 0, k)

def root(values: AbstractTensor, k: int):
    selected = bounded(k)[2]
    if selected > {upper}:
        raise ValueError("large")
    return AbstractTensor.topk(values, k=selected, dim=0).indices
'''
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
    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "root", python_bindings={"AbstractTensor": AbstractTensor},
        name="returned_tuple_guard_native", extraction_contract=policy,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _item: None, progress_bars=False,
    )
    assert not CorrelationTable.build(module).findings(module)
    entry = exports[0]
    artifact = emit_ssa_module_to_c(module, entry)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "returned_tuple_guard_native")
    root = module.functions[entry]
    names = dict(root.metadata["parameter_names"])
    returns = tuple(
        instruction.args[0].id
        for block in root.blocks.values()
        for instruction in block.instrs if instruction.op == "Ret"
    )
    assert len(set(returns)) == 1
    for count, expected in ((1, [0]), (2, [0, 2]), (3, [0, 2, 1])):
        buffers = artifact.prepare_execution({
            int(names["values"]): np.asarray([7, 2, 5], dtype=np.int64),
            int(names["k"]): np.asarray([count], dtype=np.int64),
        }).run().buffers
        assert np.asarray(buffers[returns[0]]).tolist() == expected
