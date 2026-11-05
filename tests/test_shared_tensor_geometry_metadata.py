"""Distinct tensor results can share one geometry across several Call ports."""

import numpy as np
import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import c_backend_repository_ssa_reference
from src.compiler.concordance_declarations import (
    AGGREGATE_PASSTHROUGH_REBINDING, SSA_BLOCK,
    TENSOR_METADATA_CALL_BINDING, TENSOR_METADATA_RESULT_BINDING,
    TENSOR_RUNTIME_BROADCAST_VALIDITY,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import (
    CorrelationTable, EDGE_PAGE, identity_book,
)
from src.compiler.ssa_c_backend import emit_ssa_module_to_c


def _assert_count_two_binding_forgeries(module, monkeypatch):
    """Each shared geometry actual retains its own producer and formal proof."""

    book = identity_book(module)
    page = book.page(TENSOR_METADATA_CALL_BINDING)
    row, = (
        row for row in page.rows()
        if len(page.latest(row)[0]) == 6
        and len(set(page.latest(row)[0])) == 3
        and len(set(page.latest(row)[1])) == 6
    )
    receipt = book.latest_ref(TENSOR_METADATA_CALL_BINDING, row)
    causes = {source for source, _ in book.edges_into(receipt)}
    assert len([source for source in causes
                if source.page == TENSOR_METADATA_RESULT_BINDING]) == 4
    assert len([source for source in causes
                if source.page == AGGREGATE_PASSTHROUGH_REBINDING]) == 3
    edges = book.page(EDGE_PAGE)
    original_scope_rows = edges.scope_rows
    for page_name in (
        TENSOR_METADATA_RESULT_BINDING,
        AGGREGATE_PASSTHROUGH_REBINDING,
    ):
        removed = next(
            source for source in causes
            if source.page == page_name
            and (page_name != TENSOR_METADATA_RESULT_BINDING
                 or source.column == book.latest_ref(
                     TENSOR_METADATA_RESULT_BINDING, source.row,
                 ).column)
        )
        with monkeypatch.context() as patch:
            patch.setattr(edges, "scope_rows", lambda scope, removed=removed: tuple(
                edge for edge in original_scope_rows(scope)
                if not (scope == receipt.key and edge[1] == removed.key)
            ))
            assert any(f.kind == "tensor-metadata-call-binding-disagreement"
                       and f.value_id == row[3]
                       for f in CorrelationTable._tensor_metadata_call_binding_findings(module))

    caller = module.functions[row[1]]
    callee = module.functions[row[4]]
    call, = (
        instruction for instruction in caller.blocks[row[2]].instrs
        if instruction.op == "Call" and instruction.res is not None
        and instruction.res.id == row[3]
    )
    formal_id = page.latest(row)[1][3]
    formal_index, = (index for index, formal in enumerate(callee.args)
                     if formal.id == formal_id)
    previous = call.args[formal_index]
    call.args[formal_index] = call.args[formal_index + 1]
    try:
        assert any(f.kind == "tensor-metadata-call-binding-disagreement"
                   and f.value_id == row[3]
                   for f in CorrelationTable._tensor_metadata_call_binding_findings(module))
    finally:
        call.args[formal_index] = previous
    assert CorrelationTable._tensor_metadata_call_binding_findings(module) == []


@pytest.mark.parametrize("count", (2, 7))
def test_shared_geometry_keeps_each_tensor_formal_binding(tmp_path, count, monkeypatch):
    lines = ["def root(values: AbstractTensor):"]
    lines += [f'    mask{i} = (values > {i}).to_dtype("int64")' for i in range(count)]
    lines += ["    return (" + " + ".join(f"mask{i}" for i in range(count)) + ").sum().item()"]
    contract = ExtractionContract(
        "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({"records": {}, "bindings": [], "values": [{
        "function": "root", "parameter": "values", "storage": "span",
        "dtype": "float64", "rank": 1,
        "python_type": "src.common.tensors.abstraction.AbstractTensor",
    }]})
    module, _, exports = lower_ast_source_to_ssa(
        "\n".join(lines) + "\n", "root", name=f"shared_geometry_{count}",
        python_bindings={"AbstractTensor": AbstractTensor},
        extraction_contract=contract,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    assert CorrelationTable.build(module).findings(module) == []
    if count == 2:
        _assert_count_two_binding_forgeries(module, monkeypatch)
    page = identity_book(module).page(TENSOR_METADATA_CALL_BINDING)
    assert any(len(set(actuals)) < len(actuals) and len(set(formals)) == len(formals)
               for row in page.rows() for actuals, formals in (page.latest(row),))
    artifact = emit_ssa_module_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / f"shared_geometry_{count}")
    root = module.functions[exports[0]]
    source_id = dict(root.metadata["parameter_names"])["values"]
    output_id, = (value.id for block in root.blocks.values()
                  for instruction in block.instrs if instruction.op == "Ret"
                  for value in instruction.args)
    for values in (np.empty(0), np.asarray((0.5,)),
                   np.asarray((-1, 0, 0.5, 1.5, 4, np.nan, np.inf)),
                   np.linspace(-3, 8, 64)):
        expected = sum(int((values > threshold).sum()) for threshold in range(count))
        actual = np.asarray(artifact.prepare_execution({source_id: values}).run().buffers[output_id])
        assert actual.dtype == np.dtype("int64")
        assert actual.item() == expected
    if count == 7:
        # Add36's pass block becomes Add37's guard source. The next
        # validity row must directly cite that exact SSA_BLOCK, so removing
        # the causal edge invalidates the preceding nonterminal pass too.
        book = identity_book(module)
        validity = book.page(TENSOR_RUNTIME_BROADCAST_VALIDITY)
        by_function = {}
        for row in validity.rows():
            by_function.setdefault(row[1], []).append(row)
        rows, = (sorted(group, key=lambda row: row[2])
                 for group in by_function.values() if len(group) == count - 1)
        first, second = rows[:2]
        pass_name = validity.latest(first)[-1]
        pass_cell = book.latest_ref(SSA_BLOCK, (first[0], first[1], pass_name))
        next_cell = book.latest_ref(TENSOR_RUNTIME_BROADCAST_VALIDITY, second)
        assert pass_cell in {source for source, _ in book.edges_into(next_cell)}
        edges = book.page(EDGE_PAGE)
        original = edges.scope_rows
        with monkeypatch.context() as patch:
            patch.setattr(edges, "scope_rows", lambda scope: tuple(
                edge for edge in original(scope)
                if not (scope == next_cell.key and edge[1] == pass_cell.key)
            ))
            findings = CorrelationTable._tensor_runtime_broadcast_validity_findings(module)
            assert any(finding.value_id == first[2] for finding in findings)
