"""A runtime int64 tensor is converted one element at a time for double kernels."""

from pathlib import Path

import numpy as np

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable
from src.compiler.ssa_c_backend import emit_ssa_module_to_c


def test_weighted_dynamic_int64_span_uses_counted_float64_conversion(tmp_path):
    source = """def root(bits: AbstractTensor):
    weights = bits.ensure_tensor((128, 64, 32, 16, 8, 4, 2, 1))
    return (bits.reshape(-1, 8) * weights.unsqueeze(0)).sum(dim=1).sum().item()
"""
    contract = ExtractionContract(
        Path("extraction_contracts/program_extraction.yaml")
    ).with_program_abi({
        "values": [{"function": "root", "parameter": "bits",
                    "storage": "span", "dtype": "int64", "rank": 1,
                    "python_type": "src.common.tensors.abstraction.AbstractTensor"}],
        "records": {}, "bindings": [],
    })
    module, _, exports = lower_ast_source_to_ssa(
        source, "root", name="counted_span_promotion",
        extraction_contract=contract,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    assert CorrelationTable.build(module).findings(module) == []
    cast_calls = [
        instruction for function in module.functions.values()
        for block in function.blocks.values() for instruction in block.instrs
        if instruction.op == "Call"
        and instruction.attributes.get("callee") == "cast_int64_to_double_values"
    ]
    assert len(cast_calls) == 1
    assert len(cast_calls[0].args) == 3
    artifact = emit_ssa_module_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "counted_span_promotion")
    bit_id = dict(module.functions[exports[0]].metadata["parameter_names"])["bits"]
    weights = np.asarray((128, 64, 32, 16, 8, 4, 2, 1), dtype=np.float64)
    for count in (16, 64):
        data = np.arange(count, dtype=np.int64)
        feed = {bit_id: data}
        output = [value for key, value in artifact.prepare_execution(feed).run().buffers.items()
                  if key not in feed]
        expected = (data.reshape(-1, 8).astype(np.float64) * weights).sum(axis=1).sum()
        assert len(output) == 1
        assert output[0].item() == expected
