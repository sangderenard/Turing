"""A planned region keeps its own runtime metadata SSA identities."""

from pathlib import Path

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import TENSOR_METADATA_RESULT_BINDING
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import (
    CorrelationTable, identity_book,
)


def test_region_product_does_not_inherit_caller_metadata_locations(monkeypatch):
    class AtFrameProof(BaseException):
        pass

    observed = []
    original_findings = CorrelationTable.findings

    def inspect(self, module, *args, **kwargs):
        # The selected authored broadcast helper is compiled first and has
        # its own planned_region_3. This checkpoint belongs to the public
        # source's region, where Mul20 and its metadata pair were written.
        region = module.functions.get(
            "dynamic_mixed_frame_scope_probe__root__planned_region_3"
        )
        if region is None:
            return original_findings(self, module, *args, **kwargs)
        product = next(instruction for block in region.blocks.values()
                       for instruction in block.instrs
                       if instruction.res is not None
                       and int(instruction.res.id) == 20)
        book = identity_book(module)
        row = next(row for row in book.page(TENSOR_METADATA_RESULT_BINDING).rows()
                   if row[2] == region.name and row[3] == 20)
        fact = book.page(TENSOR_METADATA_RESULT_BINDING).latest(row)
        local_ids, caller_ids = fact[1:]
        assert local_ids != caller_ids
        assert all(value_id not in caller_ids for value_id in (
            (product.res.accounting or {}).get("tensor_shape_value_id"),
            (product.res.accounting or {}).get("tensor_rank_value_id"),
            (product.res.accounting or {}).get("tensor_element_count_value_id"),
        ))
        observed.append((local_ids, caller_ids))
        raise AtFrameProof

    monkeypatch.setattr(CorrelationTable, "findings", inspect)
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
            source, "root", name="dynamic_mixed_frame_scope_probe",
            extraction_contract=contract,
            python_bindings={"AbstractTensor": AbstractTensor},
            tensor_ssa_reference=c_backend_repository_ssa_reference(),
            progress=lambda _: None, progress_bars=False,
        )
    except AtFrameProof:
        pass
    assert len(observed) == 1
