"""Canonical native constructor defaults have exact source and SSA causes."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.compiler import hierarchical_plan
from src.compiler.concordance_declarations import (
    NUMERICAL_CONSTRUCTOR_CODE_REFERENCE,
    NUMERICAL_CONSTRUCTOR_DEFAULT,
    PROCESS_GRAPH_OPERANDS, SOURCE_INTRINSIC_ADMISSION,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import (
    CorrelationTable,
    IdentityBook,
    current_identity_book,
)


def test_empty_constructor_pages_do_not_load_native_ssa(monkeypatch):
    from src.common.tensors.accelerator_backends import c_backend_llvm_ssa

    def unexpected_reference():
        raise AssertionError("empty constructor pages do not require native SSA")

    monkeypatch.setattr(
        c_backend_llvm_ssa,
        "c_backend_repository_ssa_reference",
        unexpected_reference,
    )
    module = SimpleNamespace(metadata={"identity_book": IdentityBook()})

    assert CorrelationTable._numerical_constructor_default_findings(module) == []


@pytest.mark.parametrize(("call", "expected"), [
    ("AbstractTensor.zeros((count,), cls=type(like))", "float64"),
    ("AbstractTensor.zeros((count,), dtype='float64', cls=type(like))", None),
])
def test_native_constructor_default_at_real_source_region(monkeypatch, call, expected):
    from src.common.tensors.accelerator_backends import c_backend_llvm_ssa

    class ReachedZerosRegion(Exception):
        pass

    observed = []
    reference_loads = []
    original_reference = c_backend_llvm_ssa.c_backend_repository_ssa_reference
    original = hierarchical_plan.plan_region_to_ssa_instrs

    def observe_reference():
        reference_loads.append(True)
        return original_reference()

    monkeypatch.setattr(
        c_backend_llvm_ssa,
        "c_backend_repository_ssa_reference",
        observe_reference,
    )

    def inspect_region(region, **kwargs):
        if any(getattr(item, "opcode", None) == "zeros"
               for item in region.items):
            book = current_identity_book()
            rows = tuple(book.page(NUMERICAL_CONSTRUCTOR_DEFAULT).rows())
            if expected is None:
                assert not rows
            else:
                assert rows
                for row in rows:
                    fact = book.page(NUMERICAL_CONSTRUCTOR_DEFAULT).latest(row)
                    assert fact[0] == "zeros"
                    assert fact[-1] == expected
                    assert fact[4].page == NUMERICAL_CONSTRUCTOR_CODE_REFERENCE
                    assert fact[7]  # exact stored scalar SSA value
            module = SimpleNamespace(metadata={"identity_book": book})
            assert CorrelationTable._numerical_constructor_default_findings(module) == []
            if rows:
                page = book.page(NUMERICAL_CONSTRUCTOR_DEFAULT)
                row = rows[0]
                cell = book.latest_ref(NUMERICAL_CONSTRUCTOR_DEFAULT, row)
                original_fact = page.cells[(row, cell.column)]
                try:
                    page.cells[(row, cell.column)] = (*original_fact[:-1], "float32")
                    assert CorrelationTable._numerical_constructor_default_findings(module)
                finally:
                    page.cells[(row, cell.column)] = original_fact
                admission = original_fact[2]
                admission_page = book.page(SOURCE_INTRINSIC_ADMISSION)
                admitted = admission_page.cells[(admission.row, admission.column)]
                try:
                    altered = dict(admitted[1])
                    altered["parameters"] = {
                        **altered["parameters"], "lowering_namespace": "other",
                    }
                    admission_page.cells[(admission.row, admission.column)] = (
                        admitted[0], altered, admitted[2],
                    )
                    assert CorrelationTable._numerical_constructor_default_findings(module)
                finally:
                    admission_page.cells[(admission.row, admission.column)] = admitted
                operand = original_fact[3]
                operand_page = book.page(PROCESS_GRAPH_OPERANDS)
                edges = operand_page.cells[(operand.row, operand.column)]
                try:
                    operand_page.cells[(operand.row, operand.column)] = (
                        *edges, (edges[0][0], "kw:dtype"),
                    )
                    assert CorrelationTable._numerical_constructor_default_findings(module)
                finally:
                    operand_page.cells[(operand.row, operand.column)] = edges
            observed.append(len(rows))
            raise ReachedZerosRegion
        return original(region, **kwargs)

    monkeypatch.setattr(hierarchical_plan, "plan_region_to_ssa_instrs", inspect_region)
    policy = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "records": {}, "bindings": [],
        "values": [
            {"function": "root", "parameter": "like", "storage": "span",
             "dtype": "int64", "rank": 1,
             "python_type": "src.common.tensors.abstraction.AbstractTensor"},
            {"function": "root", "parameter": "count", "storage": "scalar",
             "dtype": "int64", "rank": 0, "python_type": "builtins.int"},
        ],
    })
    with pytest.raises(ReachedZerosRegion):
        lower_ast_source_to_ssa(
            f"def root(like, count):\n    return {call}\n",
            "root", python_bindings={"AbstractTensor": AbstractTensor},
            name="native_constructor_default", extraction_contract=policy,
            progress=lambda _: None, progress_bars=False,
        )
    assert len(observed) == 1
    assert bool(observed[0]) is bool(expected)
    assert bool(reference_loads) is bool(expected)
