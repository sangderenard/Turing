"""The admitted named top-k pair crosses source, region, SSA, and C."""

from pathlib import Path

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


@pytest.mark.parametrize("field,expected", (
    ("values", [2**53 + 2, 2**53 + 1]),
    ("indices", [2, 1]),
))
def test_public_topk_named_leaf_has_one_native_aggregate_call(
    tmp_path, field, expected,
):
    policy = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "records": {}, "bindings": [],
        "values": [{
            "function": "root", "parameter": "values", "storage": "span",
            "dtype": "int64", "rank": 1, "shape": [3],
            "python_type": "src.common.tensors.abstraction.AbstractTensor",
        }],
    })
    module, _outputs, _exports = lower_ast_source_to_ssa(
        "def root(values):\n"
        f"    return AbstractTensor.topk(values,k=2,dim=0).{field}\n",
        "root", python_bindings={"AbstractTensor": AbstractTensor},
        name="topk_public", extraction_contract=policy,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    root = module.functions["topk_public__root"]
    region = module.functions["topk_public__root__planned_region_0"]
    kernel = module.functions["topk_int64_dim"]
    calls = tuple(
        instruction for block in region.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Call"
        and instruction.attributes.get("callee") == kernel.name
    )
    assert len(calls) == 1
    call = calls[0]
    assert len(call.args) == 5
    assert call.attributes["callee_input_ids"] == tuple(
        argument.id for argument in kernel.args[:5]
    )
    named = dict(kernel.metadata["named_outputs"])
    assert call.attributes["callee_output_ids"] == (
        named["values"], named["indices"],
    )
    assert call.res.dtype == "ssa.aggregate"
    assert not any(
        instruction.op == "Alloca"
        for block in region.blocks.values()
        for instruction in block.instrs
    )
    assert CorrelationTable.build(module).findings(module) == []

    artifact = emit_ssa_module_to_c(module, root.name)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / f"topk_{field}")
    abi = artifact.runtime_abi()
    assert tuple(item["shape"] for item in abi["buffers"]) == ([3], [2])
    input_id, output_id = (item["value_id"] for item in abi["buffers"])
    execution = artifact.prepare_execution({
        input_id: np.array([2**53, 2**53 + 1, 2**53 + 2], dtype=np.int64),
    }).run()
    assert execution.buffers[output_id].tolist() == expected
