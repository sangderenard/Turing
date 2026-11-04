"""A runtime tensor dimension is scalar integer evidence for arange."""

import numpy as np
import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import c_backend_repository_ssa_reference
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable
from src.compiler.ssa_c_backend import emit_ssa_module_to_c


@pytest.mark.parametrize("bound, offset", (
    ("weights.shape[0]", 0),
    ("int(weights.shape[0])", 0),
    ("weights.shape[0] + 1", 1),
), ids=("projection", "conversion", "arithmetic"))
def test_runtime_shape_dimension_types_arange_before_region_lowering(tmp_path, bound, offset):
    contract = ExtractionContract("extraction_contracts/program_extraction.yaml").with_program_abi({
        "values": [{"function": "root", "parameter": "weights", "storage": "span",
                    "dtype": "int64", "rank": 1,
                    "python_type": "src.common.tensors.abstraction.AbstractTensor"}],
        "records": {}, "bindings": [],
    })
    module, _, exports = lower_ast_source_to_ssa(
        "def root(weights: AbstractTensor) -> int:\n"
        f"    count = {bound}\n"
        "    index = AbstractTensor.arange(count, cls=type(weights))\n"
        "    return index.sum().item()\n",
        "root", name="shape_bound_arange",
        python_bindings={"AbstractTensor": AbstractTensor}, extraction_contract=contract,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    assert CorrelationTable.build(module).findings(module) == []
    assert any(instruction.attributes.get("callee") == "create_arange_int64"
               for function in module.functions.values() for block in function.blocks.values()
               for instruction in block.instrs)
    artifact = emit_ssa_module_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "shape_bound_arange")
    root = module.functions[exports[0]]
    parameters = dict(root.metadata["parameter_names"])
    output, = [value.id for block in root.blocks.values() for instruction in block.instrs
               if instruction.op == "Ret" for value in instruction.args]
    for count in (0, 1, 4, 7):
        result = artifact.prepare_execution({
            int(parameters["weights"]): np.arange(count, dtype=np.int64),
        }).run().buffers[output]
        assert np.asarray(result).dtype == np.dtype(np.int64)
        limit = count + offset
        assert np.asarray(result).item() == limit * (limit - 1) // 2
