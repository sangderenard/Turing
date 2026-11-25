"""A list fill (``[0] * n``) is resident storage on the LLVM lane.

GLSL ``uint active[4]`` reads as ``active: list['u32'] = [0] * 4``.  Three
wrong records, each fixed at its source:

- the annotation replaced the value's ``sequence_replication`` birth, so the
  fill lowered as an elementwise ``Mul`` of ``[0]`` (one row) and
  ``active[2] = 5`` faulted out of range;
- the fill helper's count formal is int64 but its call passed ``int``, and the
  concordance refused the flip between them;
- a list handed to a helper that stores into it folded to its initializer
  (``active[2]`` read 0, the helper's store lost).
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

_HEADER = "from src.common.tensors.abstraction import AbstractTensor\n\n"

_PROGRAMS = {
    "plain_fill_store": ('''def main(out: AbstractTensor):
    a = [0] * 5
    a[2] = 3
    out[0] = float(a[2] * 10 + a[4] + len(a))
''', 35.0),
    "annotated_fill_store": ('''def main(out: AbstractTensor):
    active: list['uint32'] = [0] * 4
    active[2] = 5
    out[0] = float(active[2])
''', 5.0),
    "fill_written_by_helper": ('''def begin(active: list['uint32']):
    active[2] = 5

def main(out: AbstractTensor):
    active: list['uint32'] = [0] * 4
    begin(active)
    out[0] = float(active[2])
''', 5.0),
}


@pytest.mark.parametrize("name", sorted(_PROGRAMS))
def test_list_fill_runs_on_llvm(name):
    from src.common.tensors.abstraction import AbstractTensor
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm,
        prepare_artifact_execution,
    )

    body, expected = _PROGRAMS[name]
    contract = ExtractionContract(
        ROOT / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({"records": {}, "bindings": [], "values": [{
        "function": "*", "parameter": "out", "storage": "span",
        "dtype": "float32", "rank": 1, "shape": [1],
        "python_type": "src.common.tensors.abstraction.AbstractTensor",
    }]}).with_execution_file(
        ROOT / "extraction_contracts/vehicle_full_native_execution.yaml"
    )
    module, _outputs, exports = lower_ast_source_to_ssa(
        _HEADER + body, "main", extraction_contract=contract,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        runtime_closure_only=True, progress=lambda _: None,
        progress_bars=False,
    )
    entry = exports[0]
    names = dict(module.functions[entry].metadata["parameter_names"])
    emitted = emit_ssa_function_to_llvm(module, entry)
    assert not emitted.shortfalls, emitted.shortfalls
    artifact = compile_artifact(emitted, directory=tempfile.mkdtemp())
    out = np.zeros(1, dtype=np.float32)
    prepare_artifact_execution(artifact, {names["out"]: out}).run()
    assert float(out[0]) == expected
