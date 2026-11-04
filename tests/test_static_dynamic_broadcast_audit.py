"""A fixed weight view and a runtime span need one complete broadcast proof."""

from pathlib import Path

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import (
    KERNEL_INPUT_CONVERSION, TENSOR_RUNTIME_BROADCAST_VALIDITY,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable, identity_book
import src.compiler.tensor_ssa_lowering as tensor_ssa_lowering


def test_static_weight_broadcast_audit_checks_shape_and_live_call(monkeypatch):
    class AtBroadcastProof(Exception):
        pass

    seen = []
    publish = tensor_ssa_lowering.publish_computed_region_result_metadata

    def inspect(module, *args, **kwargs):
        book = identity_book(module)
        page = book.page(TENSOR_RUNTIME_BROADCAST_VALIDITY)
        rows = tuple(row for row in page.rows()
                     if row[1].endswith("__planned_region_3") and row[2] == 20)
        if not rows:
            return publish(module, *args, **kwargs)
        assert len(rows) == 1
        fact = page.latest(rows[0])
        function = module.functions[rows[0][1]]
        assert not CorrelationTable._tensor_runtime_broadcast_validity_findings(
            module,
        )

        shape = next(instruction for block in function.blocks.values()
                     for instruction in block.instrs
                     if instruction.op == "Const" and instruction.res is not None
                     and int(instruction.res.id) == fact[8])
        values = shape.attributes["values"]
        shape.attributes["values"] = (2, 8)
        assert CorrelationTable._tensor_runtime_broadcast_validity_findings(
            module,
        )
        shape.attributes["values"] = values

        rhs = next(instruction for block in function.blocks.values()
                   for instruction in block.instrs
                   if instruction.op == "Call" and instruction.res is not None
                   and instruction.attributes.get("callee") == "broadcast_double"
                   and int(instruction.args[0].id) == fact[2])
        actuals = rhs.args
        wrong = list(actuals)
        wrong[2], wrong[3] = wrong[3], wrong[2]
        rhs.args = tuple(wrong)
        assert CorrelationTable._tensor_runtime_broadcast_validity_findings(
            module,
        )
        rhs.args = actuals
        assert not CorrelationTable._tensor_runtime_broadcast_validity_findings(
            module,
        )

        conversion_row = next(
            row for row in book.page(KERNEL_INPUT_CONVERSION).rows()
            if row[0] == rows[0][0] and row[2:] == (20, 0)
        )
        converted_id = book.page(KERNEL_INPUT_CONVERSION).latest(
            conversion_row,
        )[1]
        counted = next(
            instruction for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.op == "Call" and instruction.res is not None
            and int(instruction.res.id) == converted_id
            and instruction.attributes.get("callee")
            == "cast_int64_to_double_values"
        )
        counted_actuals = counted.args
        counted.args = (*counted_actuals[:2], counted_actuals[0])
        assert CorrelationTable._tensor_runtime_broadcast_validity_findings(
            module,
        )
        counted.args = counted_actuals
        assert not CorrelationTable._tensor_runtime_broadcast_validity_findings(
            module,
        )
        seen.append(rows[0])
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
    return (bits.reshape(-1, 8) * weights.unsqueeze(0)).sum(dim=1)
"""
    try:
        lower_ast_source_to_ssa(
            source, "root", name="static_dynamic_broadcast_audit_probe",
            extraction_contract=contract,
            python_bindings={"AbstractTensor": AbstractTensor},
            tensor_ssa_reference=c_backend_repository_ssa_reference(),
            progress=lambda _: None, progress_bars=False,
        )
    except AtBroadcastProof:
        pass
    assert len(seen) == 1
