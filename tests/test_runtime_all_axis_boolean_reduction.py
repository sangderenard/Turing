"""Runtime all-axis Boolean reductions use the selected flat tensor domain."""

from pathlib import Path

import numpy as np
import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_c_backend import emit_ssa_module_to_c


@pytest.mark.parametrize("operation", ("all", "any"))
def test_runtime_all_axis_boolean_reduction_native(tmp_path, operation):
    source = (
        "def root(values: AbstractTensor) -> bool:\n"
        f"    return values.{operation}().item()\n"
    )
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [{
            "function": "root", "parameter": "values", "storage": "span",
            "dtype": "float64", "rank": 1,
            "python_type": "src.common.tensors.abstraction.AbstractTensor",
        }],
        "records": {}, "bindings": [],
    })
    module, _, exports = lower_ast_source_to_ssa(
        source, "root", name=f"runtime_{operation}_axis_probe",
        python_bindings={"AbstractTensor": AbstractTensor},
        extraction_contract=contract,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    artifact = emit_ssa_module_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / operation)
    root = module.functions[exports[0]]
    parameter = dict(root.metadata["parameter_names"])["values"]
    output, = (
        value.id for block in root.blocks.values()
        for instruction in block.instrs if instruction.op == "Ret"
        for value in instruction.args
    )
    for values in (
        np.asarray([], dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
        np.asarray([1.0, -2.0], dtype=np.float64),
        np.asarray([1.0, 0.0, np.nan], dtype=np.float64),
        np.asarray([np.nan], dtype=np.float64),
    ):
        expected = bool(getattr(np, operation)(values))
        actual = artifact.prepare_execution({parameter: values}).run().buffers[output]
        assert bool(np.asarray(actual).item()) == expected
