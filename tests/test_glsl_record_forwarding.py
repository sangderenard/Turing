"""Vector records through helper calls on the GLSL route (``source_language='glsl'``).

* The reader's tree is ordinary Python, so its graph has Python semantics:
  ``b.x`` of a record parameter read 0 when the graph was built as "glsl".
* A record forwarded whole to another helper (the Studio tracer's
  ``splat_segment_multiscale`` -> ``splat_segment_spectral``) crosses as its
  fields: the forwarder's whole-record formal is correlation, not a slot.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

_OUTER = {
    "single": ("float outer(float a, vec2 b){ return b.x + a; }", 3.0),
    "forward": ("float outer(float a, vec2 b){ return inner(b) + a; }", 3.0),
    "forward_and_read": (
        "float outer(float a, vec2 b){ return inner(b) + a + b.y; }", 6.0),
}


@pytest.mark.parametrize("name", sorted(_OUTER))
def test_vector_record_through_helpers_on_the_glsl_route(name):
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

    outer, expected = _OUTER[name]
    source = f'''#version 430
layout(local_size_x=1) in;
layout(std430, binding=0) buffer O {{ float o[]; }};
float inner(vec2 b){{ return b.x; }}
{outer}
void main(){{ vec2 p = vec2(2.0, 3.0); o[0] = outer(1.0, p); }}
'''
    contract = ExtractionContract(
        ROOT / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({"records": {}, "bindings": [], "values": [{
        "function": "*", "parameter": "o", "storage": "span",
        "dtype": "float32", "rank": 1, "shape": [1],
        "python_type": "src.common.tensors.abstraction.AbstractTensor",
    }]}).with_execution_file(
        ROOT / "extraction_contracts/vehicle_full_native_execution.yaml"
    )
    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "main", source_language="glsl", extraction_contract=contract,
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
    prepare_artifact_execution(artifact, {names["o"]: out}).run()
    assert float(out[0]) == expected
