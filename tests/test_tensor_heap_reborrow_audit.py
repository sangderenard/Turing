"""A discarded aggregate view keeps private storage in its owning caller."""

from pathlib import Path

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import (
    AGGREGATE_PASSTHROUGH_REBINDING, TENSOR_HEAP_CONSUMER_NONESCAPE,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable, identity_book


def test_retired_borrowed_view_requires_rebinding_and_no_escape():
    source = (
        "def root(values: AbstractTensor) -> int:\n"
        "    k = values.shape[0]\n"
        "    raw = AbstractTensor.topk(values, k=k, dim=0).indices\n"
        "    selected = values.ensure_tensor(raw)\n"
        "    return selected.sum().item()\n"
    )
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "records": {}, "bindings": [],
        "values": [{
            "function": "root", "parameter": "values", "storage": "span",
            "dtype": "int64", "rank": 1,
            "python_type": "src.common.tensors.abstraction.AbstractTensor",
        }],
    })
    module, _, _ = lower_ast_source_to_ssa(
        source, "root", name="topk_reborrow_audit",
        python_bindings={"AbstractTensor": AbstractTensor},
        extraction_contract=contract,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    book = identity_book(module)
    assert CorrelationTable.build(module).findings(module) == []
    page = book.page(TENSOR_HEAP_CONSUMER_NONESCAPE)
    rows = tuple(row for row in page.rows()
                 if any(layout[1] in page.latest(row)[2]
                        for layout in page.latest(row)[3]))
    assert len(rows) == 1
    row = rows[0]
    caller = module.functions[row[1]]
    call = next(
        instruction for block in caller.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Call" and instruction.res is None
        and instruction.attributes.get("aggregate_retired_result") is not None
        and instruction.attributes.get("callee") == page.latest(row)[0]
    )
    retired = call.attributes.pop("aggregate_retired_result")
    try:
        assert CorrelationTable._tensor_heap_consumer_nonescape_findings(module)
        assert CorrelationTable._tensor_heap_release_binding_findings(module)
    finally:
        call.attributes["aggregate_retired_result"] = retired

    alias_id = call.attributes["aggregate_output_passthrough_bindings"][0][1]
    binding_row = (row[0], alias_id)
    binding_page = book.page(AGGREGATE_PASSTHROUGH_REBINDING)
    binding_ref = book.latest_ref(AGGREGATE_PASSTHROUGH_REBINDING, binding_row)
    key = (binding_ref.row, binding_ref.column)
    original = binding_page.cells[key]
    try:
        binding_page.cells[key] = (original[0], original[1], original[2] + 1)
        assert CorrelationTable._tensor_heap_consumer_nonescape_findings(module)
    finally:
        binding_page.cells[key] = original

    terminal = next(instruction for block in caller.blocks.values()
                    for instruction in block.instrs if instruction.op == "Ret")
    previous = terminal.args[0]
    terminal.args[0] = next(value for block in caller.blocks.values()
                            for instruction in block.instrs
                            for value in instruction.args
                            if int(value.id) == page.latest(row)[1])
    try:
        assert CorrelationTable._tensor_heap_consumer_nonescape_findings(module)
        assert CorrelationTable._tensor_heap_release_binding_findings(module)
    finally:
        terminal.args[0] = previous
    assert CorrelationTable.build(module).findings(module) == []
