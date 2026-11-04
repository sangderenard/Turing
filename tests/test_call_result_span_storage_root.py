"""A returned tensor member is not a record-row payload candidate."""

import numpy as np
import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable
from src.compiler.ssa_c_backend import emit_ssa_module_to_c


@pytest.mark.parametrize("indexed_view", (False, True))
def test_indexed_and_scalar_used_returned_span_keeps_call_result_storage(
    tmp_path, indexed_view,
):
    # The same returned tensor is indexed and consumed by sum. The former
    # storage survey classified this as a possible row payload and demanded
    # a local numerical producer for the structural call-result projection.
    source = """def derive(x: AbstractTensor):
    return x.to_dtype('int64'), x + 1

def root(x: AbstractTensor, indices: AbstractTensor):
    weights, active = derive(x)
    if_view
""".replace(
        "    if_view",
        "    leaves = weights[indices]\n    return leaves[0] + leaves.sum().item()"
        if indexed_view else
        "    return weights[0] + weights.sum().item()",
    )
    contract = ExtractionContract(
        "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [{
            "function": "root", "parameter": parameter, "storage": "span",
            "dtype": "int64", "rank": 1, "shape": shape,
            "python_type": "src.common.tensors.abstraction.AbstractTensor",
        } for parameter, shape in (("x", [4]), ("indices", [2]))],
        "records": {}, "bindings": [],
    })
    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "root", name=f"call_result_span_storage_root_{indexed_view}",
        python_bindings={"AbstractTensor": AbstractTensor},
        extraction_contract=contract,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    assert exports
    assert CorrelationTable.build(module).findings(module) == []
    artifact = emit_ssa_module_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "call_result_span")
    names = dict(module.functions[exports[0]].metadata["parameter_names"])
    feeds = {
        int(names["x"]): np.asarray((1, 2, 3, 4), dtype=np.int64),
    }
    if indexed_view:
        feeds[int(names["indices"])] = np.asarray((1, 3), dtype=np.int64)
    buffers = artifact.prepare_execution(feeds).run().buffers
    result = tuple(value for key, value in buffers.items() if key not in feeds)
    assert len(result) == 1
    assert result[0].item() == (8 if indexed_view else 11)
