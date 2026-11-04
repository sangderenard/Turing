"""Region admission reads its existing scoped scalar physical receipt."""

from dataclasses import replace
from pathlib import Path

import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler import ssa_record_return_state, tensor_ssa_lowering


@pytest.mark.parametrize("with_scalar_receipts", [True, False])
def test_mixed_tensor_region_admission_uses_posted_scalar_receipts(
    monkeypatch, with_scalar_receipts,
):
    source = """def root(bits: AbstractTensor, width: int) -> AbstractTensor:
    count = bits.shape[0]
    positions = AbstractTensor.arange(width, cls=type(bits))
    capacity = count * width
    return bits + positions + capacity
"""
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [
            {"function": "root", "parameter": "bits", "storage": "span",
             "dtype": "int64", "rank": 2,
             "python_type": "src.common.tensors.abstraction.AbstractTensor"},
            {"function": "root", "parameter": "width", "storage": "scalar",
             "dtype": "int64", "rank": 0, "shape": [],
             "python_type": "builtins.int"},
        ],
        "records": {}, "bindings": [],
    })
    original = ssa_record_return_state.posted_ssa_value_fact
    withheld = []

    def read_fact(function, value_id, **kwargs):
        fact = original(function, value_id, **kwargs)
        if (
            not with_scalar_receipts
            and function.metadata.get("source_region_integral")
            and any(int(formal.id) == value_id and not formal.accounting
                    for formal in function.args)
            and fact is not None and fact.storage == "scalar" and fact.rank == 0
        ):
            withheld.append((function.name, value_id))
            return replace(fact, physical_dtype=None, storage=None, rank=None)
        return fact

    monkeypatch.setattr(ssa_record_return_state, "posted_ssa_value_fact", read_fact)

    class ObservedAdmission(Exception):
        pass

    observed = []

    def observe_before_result_publication(module, **_options):
        # This checkpoint checks admission only. Dynamic result geometry is
        # a separate, still-audited obligation; do not bypass the final audit
        # and report this module as a completed source/native product.
        regions = [function for function in module.functions.values()
                   if function.metadata.get("source_region_integral")
                   and any(instruction.op == "Mul"
                           for block in function.blocks.values()
                           for instruction in block.instrs)]
        assert len(regions) == 1
        function = regions[0]
        result_ids = function.metadata["source_region_integral"]["output_value_ids"]
        table = module.tensor_tables.get(function.name)
        observed.append(table is not None and all(
            table.by_id(int(value_id)) is not None for value_id in result_ids
        ))
        raise ObservedAdmission

    monkeypatch.setattr(
        tensor_ssa_lowering, "publish_computed_region_result_metadata",
        observe_before_result_publication,
    )
    with pytest.raises(ObservedAdmission):
        lower_ast_source_to_ssa(
            source, "root", name="preflight_scalar_receipt",
            extraction_contract=contract,
            python_bindings={"AbstractTensor": AbstractTensor},
            tensor_ssa_reference=c_backend_repository_ssa_reference(),
            progress=lambda _: None, progress_bars=False,
        )
    assert observed == [with_scalar_receipts]
    if not with_scalar_receipts:
        assert withheld
