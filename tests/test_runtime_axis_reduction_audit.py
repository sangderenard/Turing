"""A dynamic axis reduction keeps its shape-copy and adapted output proof."""

from pathlib import Path

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import TENSOR_REDUCTION_DOMAIN
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable, identity_book


def test_axis_runtime_audit_checks_shape_copy_and_output_bridge(monkeypatch):
    class AtReductionProof(BaseException):
        pass

    seen = []
    audit = CorrelationTable._tensor_reduction_domain_findings

    def inspect(module):
        findings = audit(module)
        if seen:
            return findings
        book = identity_book(module)
        page = book.page(TENSOR_REDUCTION_DOMAIN)
        rows = tuple(row for row in page.rows()
                     if row[0].endswith("__planned_region_5") and row[1] == 27)
        if not rows:
            return findings
        function = module.functions[rows[0][0]]
        casts = tuple(instruction for block in function.blocks.values()
                      for instruction in block.instrs
                      if instruction.op == "Cast" and instruction.res is not None
                      and int(instruction.res.id) == 27
                      and instruction.attributes.get(
                          "physical_call_output_conversion") is True)
        assert findings == []
        fact = page.latest(rows[0])
        assert fact[0] == "axis-runtime"
        calls = tuple(instruction for block in function.blocks.values()
                      for instruction in block.instrs
                      if instruction.op == "Call"
                      and instruction.attributes.get("callee")
                      == "reduce_dim_double")
        assert len(calls) == 1
        if casts:
            assert len(casts) == 1
            bridge = casts[0]
            bridge.attributes["physical_call_output_conversion"] = False
            assert audit(module)
            bridge.attributes["physical_call_output_conversion"] = True
        else:
            assert calls[0].res is not None
            assert int(calls[0].res.id) == 27
            assert int(calls[0].args[1].id) != 27
            carrier = calls[0].args[1]
            calls[0].args[1] = calls[0].args[0]
            assert audit(module)
            calls[0].args[1] = carrier

        shape_id = fact[8]
        stores = tuple(instruction for block in function.blocks.values()
                       for instruction in block.instrs
                       if instruction.op == "Store" and len(instruction.args) == 2
                       and any(address.res is not None
                               and int(address.res.id) == int(instruction.args[1].id)
                               and address.op == "GetElementPtr"
                               and int(address.args[0].id) == shape_id
                               for owner in function.blocks.values()
                               for address in owner.instrs))
        assert len(stores) == 1
        original_value = stores[0].args[0]
        source_count = next(value for value in function.args
                            if int(value.id) == fact[7])
        stores[0].args[0] = source_count
        assert audit(module)
        stores[0].args[0] = original_value

        assert audit(module) == []
        seen.append(rows[0])
        raise AtReductionProof

    monkeypatch.setattr(
        CorrelationTable, "_tensor_reduction_domain_findings",
        staticmethod(inspect),
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
            source, "root", name="runtime_axis_reduction_audit_probe",
            extraction_contract=contract,
            python_bindings={"AbstractTensor": AbstractTensor},
            tensor_ssa_reference=c_backend_repository_ssa_reference(),
            progress=lambda _: None, progress_bars=False,
        )
    except AtReductionProof:
        pass
    assert len(seen) == 1
