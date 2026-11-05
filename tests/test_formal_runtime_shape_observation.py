"""A runtime span callsite withdraws a shared helper's static formal shape."""

from pathlib import Path
import inspect

import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.common.tensors.compression.huffman import _require_tensor
from src.compiler import glsl_deployment_strategy
from src.compiler.concordance_declarations import (
    CANONICAL_VALUE, FORMAL_SHAPE, INGEST_SOURCE, INGESTION,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.formal_shape_ledger import (
    _publish_formal_runtime_shape, _publish_formal_shape,
)
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import (
    Mode, Novel, Unresolved, begin_identity_book,
    current_identity_book, end_identity_book,
)


class _ObservedFormal(Exception):
    pass


def _observe_formal(monkeypatch, *, dynamic_symbols, forge_runtime=False):
    source = (
        "def root(frequencies: AbstractTensor, symbols: AbstractTensor) -> int:\n"
        "    first = _require_tensor(frequencies, 'frequencies')\n"
        "    second = _require_tensor(symbols, 'symbols')\n"
        "    return first.sum().item() + second.sum().item()\n"
    )
    values = [
        {"function": "root", "parameter": "frequencies", "storage": "span",
         "dtype": "int64", "rank": 1, "shape": (256,),
         "python_type": "src.common.tensors.abstraction.AbstractTensor"},
        {"function": "root", "parameter": "symbols", "storage": "span",
         "dtype": "int64", "rank": 1,
         **({} if dynamic_symbols else {"shape": (256,)}),
         "python_type": "src.common.tensors.abstraction.AbstractTensor"},
    ]
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({"values": values, "records": {}, "bindings": []})
    original = glsl_deployment_strategy._tensor_descriptor_rule
    argument_descriptor = glsl_deployment_strategy._callsite_argument_descriptor
    observed = []

    def watch(graph, node_id, *args, **kwargs):
        answer = original(graph, node_id, *args, **kwargs)
        if (graph.G.graph.get("function_name") != "_require_tensor"
                or int(node_id) not in graph.G):
            return answer
        node = graph.G.nodes[int(node_id)]
        if (node.get("type") != "Input"
                or (node.get("attributes") or {}).get("binding_name") != "value"):
            return answer
        local = node.get("tensor") or {}
        wanted = "dynamic" if dynamic_symbols else "static"
        local_state = str(local.get("metadata_state") or "static")
        if local_state != wanted or not isinstance(answer, dict):
            return answer
        book = current_identity_book()
        page = book.page("formal_shape")
        row = ("_require_tensor", "value")
        cell = book.latest_ref(FORMAL_SHAPE, row)
        if cell is None:
            return answer
        formal = page.cell(row, cell.column)
        if dynamic_symbols and not isinstance(formal, Unresolved):
            return answer
        observed.append((answer, formal, cell,
                         tuple(cause for cause, _ in book.edges_into(cell))))
        raise _ObservedFormal

    monkeypatch.setattr(glsl_deployment_strategy, "_tensor_descriptor_rule", watch)
    if forge_runtime:
        def forged_descriptor(caller, call_id, parent):
            answer = argument_descriptor(caller, call_id, parent)
            data = caller.G.nodes[int(parent)]
            if (data.get("attributes") or {}).get("binding_name") == "symbols":
                return {"shape": (), "dtype": "int64", "rank": 1,
                        "metadata_state": "dynamic"}
            return answer

        monkeypatch.setattr(
            glsl_deployment_strategy, "_callsite_argument_descriptor",
            forged_descriptor,
        )
    try:
        lower_ast_source_to_ssa(
            source, "root", name="formal_runtime_shape_observation",
            extraction_contract=contract,
            python_bindings={"AbstractTensor": AbstractTensor,
                             "_require_tensor": _require_tensor},
            tensor_ssa_reference=c_backend_repository_ssa_reference(),
            progress=lambda _: None, progress_bars=False,
        )
    except _ObservedFormal:
        pass
    assert observed
    return observed[-1]


def test_dynamic_callsite_withdraws_static_formal_shape(monkeypatch):
    answer, formal, cell, causes = _observe_formal(
        monkeypatch, dynamic_symbols=True,
    )
    assert answer["shape"] == ()
    assert answer["rank"] == 1
    assert answer["metadata_state"] == "dynamic"
    assert isinstance(formal, Unresolved)
    assert any(cause.page.name == "shape_transformation_state" for cause in causes)
    assert any(cause.page.name == "canonical_value" for cause in causes)


def test_static_calls_keep_proven_formal_shape(monkeypatch):
    answer, formal, _cell, _causes = _observe_formal(
        monkeypatch, dynamic_symbols=False,
    )
    assert answer["shape"] == (256,)
    assert formal[0:3] == ("proven", (256,), "int64")


def test_dynamic_claim_without_matching_shape_state_is_refused(monkeypatch):
    with pytest.raises(ValueError, match="runtime formal shape conflicts"):
        _observe_formal(monkeypatch, dynamic_symbols=False, forge_runtime=True)


def test_second_dynamic_caller_revises_unresolved_formal_with_its_source():
    _book, token = begin_identity_book()
    try:
        book = current_identity_book()
        callers = []
        for value_id, name in enumerate(("fixed", "runtime_a", "runtime_b"), 1):
            callers.append(book.post(
                CANONICAL_VALUE, (("authored_callers", 0), value_id),
                ("Input", "input", name), stage=INGESTION,
                provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD,
            ))
        assert _publish_formal_shape(
            "callee", "value", {"shape": (256,), "dtype": "int64"},
            "fixed", (callers[0],),
        )
        runtime = {"shape": (), "dtype": "int64", "rank": 1,
                   "metadata_state": "dynamic"}
        assert _publish_formal_runtime_shape(
            "callee", "value", runtime, "runtime_a", (callers[1],),
        )
        page = book.page("formal_shape")
        row = ("callee", "value")
        first_unresolved = book.latest_ref(FORMAL_SHAPE, row)
        assert isinstance(page.cell(row, first_unresolved.column), Unresolved)
        assert _publish_formal_runtime_shape(
            "callee", "value", runtime, "runtime_b", (callers[2],),
        )
        second_unresolved = book.latest_ref(FORMAL_SHAPE, row)
        assert second_unresolved.column == first_unresolved.column + 1
        causes = {cause for cause, _ in book.edges_into(second_unresolved)}
        assert first_unresolved in causes
        assert callers[2] in causes
        assert not _publish_formal_runtime_shape(
            "callee", "value", runtime, "runtime_b", (callers[2],),
        )
        assert book.latest_ref(FORMAL_SHAPE, row) == second_unresolved
    finally:
        end_identity_book(token)


@pytest.mark.parametrize("symbols_shape", [None, (128,)])
def test_specialized_copy_posts_its_formal_shape_observation(
    monkeypatch, symbols_shape,
):
    source = (
        "def use(symbols: AbstractTensor) -> AbstractTensor:\n"
        "    return _require_tensor(symbols, 'symbols').to_dtype('int64')\n"
        "def root(frequencies: AbstractTensor, symbols: AbstractTensor) -> int:\n"
        "    fixed = _require_tensor(frequencies, 'frequencies')\n"
        "    selected = use(symbols)\n"
        "    return fixed.sum().item() + selected.sum().item()\n"
    )
    values = [
        {"function": "root", "parameter": name, "storage": "span",
         "dtype": "int64", "rank": 1,
         **({} if shape is None else {"shape": shape}),
         "python_type": "src.common.tensors.abstraction.AbstractTensor"}
        for name, shape in (("frequencies", (256,)),
                            ("symbols", symbols_shape))
    ]
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({"values": values, "records": {}, "bindings": []})
    writer_name = (
        "_publish_formal_runtime_shape" if symbols_shape is None
        else "_publish_formal_shape"
    )
    original = getattr(glsl_deployment_strategy, writer_name)
    observed = []

    def capture(function, parameter, descriptor, caller, cells):
        frame = inspect.currentframe().f_back
        while frame is not None and frame.f_code.co_name != (
            "_propagate_callsite_tensor_specializations"
        ):
            frame = frame.f_back
        specialized = bool(
            frame is not None
            and frame.f_locals.get("publish_to_function_table") is False
        )
        changed = original(function, parameter, descriptor, caller, cells)
        if specialized and function == "_require_tensor" and caller == "use":
            book = current_identity_book()
            row = ("_require_tensor", "value")
            ref = book.latest_ref(FORMAL_SHAPE, row)
            fact = book.page(FORMAL_SHAPE).cell(row, ref.column)
            observed.append((changed, descriptor, fact, ref,
                             tuple(cause for cause, _ in book.edges_into(ref)),
                             book.page(FORMAL_SHAPE).latest(("unrelated", "value"))))
            raise _ObservedFormal
        return changed

    monkeypatch.setattr(glsl_deployment_strategy, writer_name, capture)
    with pytest.raises(_ObservedFormal):
        lower_ast_source_to_ssa(
            source, "root", name="specialized_formal_shape_observation",
            extraction_contract=contract,
            python_bindings={"AbstractTensor": AbstractTensor,
                             "_require_tensor": _require_tensor},
            tensor_ssa_reference=c_backend_repository_ssa_reference(),
            progress=lambda _: None, progress_bars=False,
        )
    changed, descriptor, fact, ref, causes, unrelated = observed[-1]
    assert changed
    assert isinstance(fact, Unresolved)
    assert ref.column >= 1
    assert unrelated is None
    assert any(cause.page.name == "canonical_value"
               and "'use'" in str(cause.row) for cause in causes)
    if symbols_shape is None:
        assert descriptor["metadata_state"] == "dynamic"
        assert descriptor["rank"] == 1
        assert any(cause.page.name == "shape_transformation_state"
                   and "use|shape" in str(cause.row) for cause in causes)
    else:
        assert descriptor["shape"] == (128,)
