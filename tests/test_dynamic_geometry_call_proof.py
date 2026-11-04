"""Transported tensor extents require the live data actual/formal edge."""

from pathlib import Path

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler import tensor_ssa_lowering
from src.compiler.concordance_declarations import (
    TENSOR_METADATA_CALL_BINDING, TENSOR_METADATA_RESULT_BINDING,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import identity_book


def test_transported_geometry_requires_live_data_pair(monkeypatch):
    source = (
        "def root(left: AbstractTensor, right: AbstractTensor) -> AbstractTensor:\n"
        "    a = left.unsqueeze(1)\n"
        "    b = right.unsqueeze(0)\n"
        "    return a + b\n"
    )
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [
            {"function": "root", "parameter": name, "storage": "span",
             "dtype": "int64", "rank": 1,
             "python_type": "src.common.tensors.abstraction.AbstractTensor"}
            for name in ("left", "right")
        ],
        "records": {}, "bindings": [],
    })
    publish = tensor_ssa_lowering.publish_computed_region_result_metadata
    observed = []

    class AtPublication(Exception):
        pass

    def capture(module, **options):
        changed = publish(module, **options)
        if tuple(identity_book(module).page(TENSOR_METADATA_CALL_BINDING).rows()):
            observed.append(module)
            raise AtPublication
        return changed

    monkeypatch.setattr(
        tensor_ssa_lowering, "publish_computed_region_result_metadata", capture,
    )
    try:
        lower_ast_source_to_ssa(
            source, "root", name="dynamic_geometry_call_proof",
            extraction_contract=contract,
            python_bindings={"AbstractTensor": AbstractTensor},
            tensor_ssa_reference=c_backend_repository_ssa_reference(),
            progress=lambda _: None, progress_bars=False,
        )
    except AtPublication:
        pass
    assert observed
    module = observed[-1]
    book = identity_book(module)
    row, = tuple(book.page(TENSOR_METADATA_CALL_BINDING).rows())
    binding = book.latest_ref(TENSOR_METADATA_CALL_BINDING, row)
    caller = module.functions[row[1]]
    callee = module.functions[row[4]]
    call, = tuple(
        instruction for instruction in caller.blocks[row[2]].instrs
        if instruction.op == "Call" and instruction.res is not None
        and int(instruction.res.id) == row[3]
    )
    predecessors = tuple(
        cause for cause, _ in book.edges_into(binding)
        if cause.page == TENSOR_METADATA_RESULT_BINDING
    )
    assert len(predecessors) == 2
    candidates = tuple(
        predecessor.row[3] for predecessor in predecessors
        if tensor_ssa_lowering._dynamic_tensor_geometry_evidence(
            module, callee, predecessor.row[3],
        ) is not None
    )
    assert candidates
    value_id = candidates[0]
    actuals = tuple(call.args)
    position, = tuple(index for index, formal in enumerate(callee.args)
                      if int(formal.id) == value_id)
    other, = tuple(actual for index, actual in enumerate(actuals)
                   if index != position
                   and int(actual.id) in {item.row[3] for item in predecessors})
    call.args = tuple(other if index == position else actual
                      for index, actual in enumerate(actuals))
    try:
        assert tensor_ssa_lowering._dynamic_tensor_geometry_evidence(
            module, callee, value_id,
        ) is None
    finally:
        call.args = actuals
