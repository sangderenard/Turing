"""GLSL copies vectors, structs and arrays; the Python it reads into must too.

GLSL assigns, passes (``in``/``inout``) and returns aggregates by value.
Python binds objects, so the reader spells each GLSL copy as a new object
(``_value_copy``).  Read as plain bindings, ``vec3 a = b; a.x = 9.0`` also
wrote ``b.x``, and the Studio tracer's records escaped through every helper
that returned or rebound one (its ``rd.x`` was refused as stored by a
call).  Each program here would read a different value with Python
aliasing than with GLSL copies; the expected value is GLSL's.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

_HEADER = '''#version 430
layout(local_size_x=1) in;
layout(std430, binding=0) buffer O { float o[]; };
'''

_PROGRAMS = {
    # Assignment copies: writing ``a`` leaves ``b``.
    "assigned_then_written": (
        "void main(){ vec3 b = vec3(1.0, 2.0, 3.0); vec3 a = b; "
        "a.x = 9.0; o[0] = b.x + b.y; }\n", 3.0),
    # A returned parameter is a copy: writing the result leaves the actual.
    "returned_parameter_then_written": (
        "vec3 keep(vec3 d){ return d; }\n"
        "void main(){ vec3 b = vec3(1.0, 2.0, 3.0); vec3 c = keep(b); "
        "c.y = 7.0; o[0] = b.x + b.y + c.y; }\n", 10.0),
    # An ``in`` parameter the helper writes is the helper's own copy.
    "in_parameter_written_by_helper": (
        "float scaled(vec3 d){ d.x = d.x * 10.0; return d.x; }\n"
        "void main(){ vec3 b = vec3(1.0, 2.0, 3.0); float s = scaled(b); "
        "o[0] = s + b.x; }\n", 11.0),
    # A struct copy copies its vector member too.
    "struct_with_vector_member_assigned": (
        "struct Hit { vec3 p; float t; };\n"
        "void main(){ Hit h = Hit(vec3(1.0, 2.0, 3.0), 4.0); Hit g = h; "
        "g.p.x = 9.0; o[0] = h.p.x + g.p.x + g.t; }\n", 14.0),
    # The tracer's shape: a helper records its direction in a global and
    # returns; the caller then writes its own direction.
    "global_recorded_from_parameter": (
        "vec3 recorded;\nvoid begin(vec3 d){ recorded = d; }\n"
        "void main(){ vec3 rd = vec3(1.0, 2.0, 3.0); begin(rd); "
        "rd.y = 8.0; o[0] = recorded.y + rd.y; }\n", 10.0),
}


@pytest.mark.parametrize("name", sorted(_PROGRAMS))
def test_glsl_aggregate_copies_run_on_llvm(name):
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
        "function": "*", "parameter": "o", "storage": "span",
        "dtype": "float32", "rank": 1, "shape": [1],
        "python_type": "src.common.tensors.abstraction.AbstractTensor",
    }]}).with_execution_file(
        ROOT / "extraction_contracts/vehicle_full_native_execution.yaml"
    )
    module, _outputs, exports = lower_ast_source_to_ssa(
        _HEADER + body, "main", source_language="glsl",
        extraction_contract=contract,
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
