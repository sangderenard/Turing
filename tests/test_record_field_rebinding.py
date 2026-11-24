"""A record field read is the operand the constructor was built from.

``u = vec2(u.x * k, u.y * k)`` (GLSL ``u *= k`` on a vec2, the Studio
tracer's GGX sample) used to resolve ``u.x`` by re-reading the constructor's
argument AST in the current environment, where ``u`` already names that
constructor: an unbounded recursion.  A name rebound after construction
(``p = V(a); a = 5``) must likewise not change ``p.x``.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

_HEADER = '''from dataclasses import dataclass
from src.common.tensors.abstraction import AbstractTensor

@dataclass
class vec2:
    x: 'float32'
    y: 'float32'

'''

_PROGRAMS = {
    "self_rebound_local": ('''def main(out: AbstractTensor):
    a: 'float32' = 2.0
    u: vec2 = vec2(a, 3.0)
    u = vec2(u.x * a, u.y * a)
    out[0] = u.x + u.y
''', 10.0),
    "self_rebound_parameter": ('''def scale(u: vec2, k: 'float32') -> 'float32':
    u = vec2(u.x * k, u.y * k)
    return u.x + u.y

def main(out: AbstractTensor):
    a: 'float32' = 2.0
    out[0] = scale(vec2(a, 3.0), a)
''', 10.0),
    "argument_rebound_after": ('''def main(out: AbstractTensor):
    a: 'float32' = 2.0
    p: vec2 = vec2(a, a)
    a = 5.0
    out[0] = p.x + a
''', 7.0),
}


@pytest.mark.parametrize("name", sorted(_PROGRAMS))
def test_record_field_reads_the_constructed_operand(name):
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
