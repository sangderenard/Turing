"""A float broadcast cites its selected native kernel and owned data carrier."""

from pathlib import Path
from dataclasses import replace

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import (
    SSA_VALUE, TENSOR_RUNTIME_BROADCAST_VALIDITY,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable, identity_book
import src.compiler.tensor_ssa_lowering as tensor_ssa_lowering


def test_float_weight_broadcast_requires_exact_kernel_and_output(monkeypatch):
    class AtBroadcastProof(BaseException):
        pass

    seen = []
    publish = tensor_ssa_lowering.publish_computed_region_result_metadata

    def inspect(module, *args, **kwargs):
        book = identity_book(module)
        page = book.page(TENSOR_RUNTIME_BROADCAST_VALIDITY)
        rows = tuple(row for row in page.rows()
                     if row[1].endswith("__planned_region_4") and row[2] == 23)
        if not rows or "before-adaptation" in seen:
            return publish(module, *args, **kwargs)
        assert len(rows) == 1
        function = module.functions[rows[0][1]]
        result = next(instruction for block in function.blocks.values()
                      for instruction in block.instrs
                      if instruction.res is not None
                      and int(instruction.res.id) == rows[0][2])
        assert result.op == "Call"
        assert result.attributes.get("callee") == "binary_double"
        assert not CorrelationTable._tensor_runtime_broadcast_validity_findings(module)

        output_argument = result.attributes["ssa_output_argument"]
        result.attributes["ssa_output_argument"] = 1
        assert CorrelationTable._tensor_runtime_broadcast_validity_findings(module)
        result.attributes["ssa_output_argument"] = output_argument

        opcode_id = int(result.args[4].id)
        opcode = next(instruction for block in function.blocks.values()
                      for instruction in block.instrs
                      if instruction.op == "Const" and instruction.res is not None
                      and int(instruction.res.id) == opcode_id)
        original_opcode = opcode.attributes["constant"]
        opcode.attributes["constant"] = original_opcode + 1
        assert CorrelationTable._tensor_runtime_broadcast_validity_findings(module)
        opcode.attributes["constant"] = original_opcode
        assert not CorrelationTable._tensor_runtime_broadcast_validity_findings(module)
        book = identity_book(module)
        rhs_id = book.page(TENSOR_RUNTIME_BROADCAST_VALIDITY).latest(rows[0])[2]
        rhs_ref = book.latest_ref(SSA_VALUE, (rows[0][0], rhs_id))
        rhs_page = book.page(SSA_VALUE)
        rhs_fact = rhs_page.cell(rhs_ref.row, rhs_ref.column)
        assert rhs_fact.physical_dtype == "float64"
        rhs_page.cells[(rhs_ref.row, rhs_ref.column)] = replace(
            rhs_fact, physical_dtype="int64",
        )
        assert CorrelationTable._tensor_runtime_broadcast_validity_findings(module)
        rhs_page.cells[(rhs_ref.row, rhs_ref.column)] = rhs_fact
        assert CorrelationTable._tensor_runtime_broadcast_validity_findings(module) == []
        seen.append("direct-physical")
        raise AtBroadcastProof

    monkeypatch.setattr(
        tensor_ssa_lowering, "publish_computed_region_result_metadata", inspect,
    )
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [{
            "function": "root", "parameter": "bits", "storage": "span",
            "dtype": "int64", "rank": 1,
            "python_type": "src.common.tensors.abstraction.AbstractTensor",
        }],
        "records": {}, "bindings": [],
    })
    source = """def root(bits: AbstractTensor):
    weights = bits.ensure_tensor((128, 64, 32, 16, 8, 4, 2, 1))
    shaped = bits.reshape(-1, 8).to_dtype('float64')
    return (shaped * weights.unsqueeze(0)).sum(dim=1).to_dtype('int64').sum().item()
"""
    try:
        lower_ast_source_to_ssa(
            source, "root", name="float_runtime_broadcast_audit_probe",
            extraction_contract=contract,
            python_bindings={"AbstractTensor": AbstractTensor},
            tensor_ssa_reference=c_backend_repository_ssa_reference(),
            progress=lambda _: None, progress_bars=False,
        )
    except AtBroadcastProof:
        pass
    assert seen == ["direct-physical"]
