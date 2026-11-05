"""A guarded tuple return reaches only its authored projected topk slot."""

from pathlib import Path

import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import GUARDED_CALL_ACTUAL_RANGE
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import current_identity_book
import src.compiler.glsl_deployment_strategy as deployment


class _ReachedTopk(Exception):
    pass


def _guarded_topk_rows(monkeypatch, *, helper_body, member):
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [
            {"function": "root", "parameter": "values", "storage": "span",
             "dtype": "int64", "rank": 1, "shape": [3],
             "python_type": "src.common.tensors.abstraction.AbstractTensor"},
            {"function": "root", "parameter": "k", "storage": "scalar",
             "dtype": "int64", "rank": 0, "python_type": "builtins.int"},
        ], "records": {}, "bindings": [],
    })
    source = (
        "def bounded(k: int):\n"
        f"{helper_body}"
        "\n"
        "def root(values: AbstractTensor, k: int):\n"
        f"    selected = bounded(k)[{member}]\n"
        "    if selected > 3:\n"
        "        raise ValueError('large')\n"
        "    return AbstractTensor.topk(values, k=selected, dim=0).indices\n"
    )
    captured = {}
    original = deployment.publish_intrinsic_named_result_leaves

    def at_topk(graph, node_id):
        expression = graph.G.nodes[node_id].get("expr_obj")
        if getattr(getattr(expression, "func", None), "attr", None) == "topk":
            book = current_identity_book()
            page = book.page(GUARDED_CALL_ACTUAL_RANGE)
            captured["rows"] = tuple(
                (row, page.latest(row), book.latest_ref(
                    GUARDED_CALL_ACTUAL_RANGE, row,
                ))
                for row in page.rows()
            )
            captured["book"] = book
            raise _ReachedTopk
        return original(graph, node_id)

    monkeypatch.setattr(deployment, "publish_intrinsic_named_result_leaves", at_topk)
    with pytest.raises(_ReachedTopk):
        lower_ast_source_to_ssa(
            source, "root", name="cross_call_topk_guard_projection",
            python_bindings={"AbstractTensor": AbstractTensor},
            extraction_contract=contract,
            tensor_ssa_reference=c_backend_repository_ssa_reference(),
            progress=lambda _: None, progress_bars=False,
        )
    return captured["book"], captured["rows"]


def test_other_tuple_member_does_not_inherit_guard(monkeypatch):
    _, rows = _guarded_topk_rows(
        monkeypatch,
        helper_body=(
            "    if k < 1:\n"
            "        raise ValueError('small')\n"
            "    return (k, 0, 0)\n"
        ),
        member=2,
    )
    assert all(fact != (1, 3) for _row, fact, _ref in rows)


def test_continuing_handler_does_not_narrow_return(monkeypatch):
    _, rows = _guarded_topk_rows(
        monkeypatch,
        helper_body=(
            "    if k < 1:\n"
            "        pass\n"
            "    return (0, 0, k)\n"
        ),
        member=2,
    )
    assert all(fact != (1, 3) for _row, fact, _ref in rows)
