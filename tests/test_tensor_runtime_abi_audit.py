"""The C output extent indices must follow proved tensor metadata ports."""

from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import (
    ArtifactFact, ArtifactPart, Backend, EMISSION_ARTIFACT,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable
from src.compiler.ssa_c_backend import emit_ssa_module_to_c
from src.compiler.ssa_storage_requirements import public_tensor_capacity_sources


@pytest.mark.parametrize("shape,rank,extent_field", (
    ([3], 1, "logical_length_from"),
    ([3, 4], 2, "logical_shape_from"),
))
def test_tensor_result_abi_rejects_forged_port_indices(
    shape, rank, extent_field,
):
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "records": {}, "bindings": [],
        "values": [
            {"function": "root", "parameter": "values", "storage": "span",
             "dtype": "int64", "rank": rank, "shape": shape,
             "python_type": "src.common.tensors.abstraction.AbstractTensor"},
            {"function": "root", "parameter": "k", "storage": "scalar",
             "dtype": "int64", "rank": 0, "python_type": "builtins.int"},
        ],
    })
    source = (
        "def root(values, k: int):\n"
        "    if k < 0:\n        raise ValueError('negative k')\n"
        "    if k > 3:\n        raise ValueError('large k')\n"
        "    return AbstractTensor.topk(values, k=k, dim=0).indices\n"
    )
    module, _outputs, _exports = lower_ast_source_to_ssa(
        source, "root", python_bindings={"AbstractTensor": AbstractTensor},
        name="topk_abi_audit", extraction_contract=contract,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    entry = "topk_abi_audit__root"
    (relation,) = public_tensor_capacity_sources(module, entry)
    artifact = emit_ssa_module_to_c(module, entry)
    assert artifact.runtime_abi()["schema"] == "turing-runtime-abi-v2"
    assert CorrelationTable._runtime_result_abi_findings(module) == []

    book = module.metadata["identity_book"]
    page = book.page(EMISSION_ARTIFACT)
    cell = book.latest_ref(EMISSION_ARTIFACT, (
        entry, Backend.C_MODULE, ArtifactPart.RUNTIME_ABI,
    ))
    original = page.cell(cell.row, cell.column)
    for field in ("capacity_from", extent_field):
        forged = deepcopy(original.location[0])
        output = next(buffer for buffer in forged["buffers"]
                      if buffer["value_id"] == relation.output_data_id)
        output[field] = next(
            buffer["index"] for buffer in forged["buffers"]
            if buffer["index"] != output[field]
        )
        payload = repr(forged).encode("utf-8")
        page.cells[(cell.row, cell.column)] = ArtifactFact(
            hashlib.sha256(payload).hexdigest(), len(payload), (forged,),
        )
        try:
            assert CorrelationTable._runtime_result_abi_findings(module)
        finally:
            page.cells[(cell.row, cell.column)] = original
