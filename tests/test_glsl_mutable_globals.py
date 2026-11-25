"""GLSL inout parameters and mutable globals, compiled and run on LLVM.

The Studio tracer records ``studio_response_direction`` in one helper and
copies it into its response row in another.  ``main`` owns that
per-invocation state; helpers take it after their authored parameters and
hand written values back, the same way an ``inout`` parameter is returned.
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
    # A returned int read back through a cast (``float(h)``).
    "inout_int": ("void bump(inout int h){ h += 1; }\n"
                  "void main(){ int h = 0; bump(h); o[0] = float(h); }\n", 1.0),
    # An inout record overwritten before any read of it.
    "inout_vec3_overwritten": (
        "void put(inout vec3 r, vec3 d){ r = d; }\n"
        "void main(){ vec3 r = vec3(0.0); put(r, vec3(1.0, 2.5, 3.0)); o[0] = r.y; }\n",
        2.5),
    "global_int": ("int hits = 0;\nvoid begin(){ hits += 1; }\n"
                   "void main(){ begin(); o[0] = float(hits); }\n", 1.0),
    "global_vec3": ("vec3 recorded;\nvoid begin(vec3 d){ recorded = d; }\n"
                    "void main(){ begin(vec3(1.0, 2.5, 3.0)); o[0] = recorded.y; }\n",
                    2.5),
    # Written in one helper, read in another: the response-row shape.
    "global_vec3_read_by_another_helper": (
        "vec3 recorded;\nvoid begin(vec3 d){ recorded = d; }\n"
        "float row(){ return recorded.y; }\n"
        "void main(){ begin(vec3(1.0, 2.5, 3.0)); o[0] = row(); }\n", 2.5),
    # One helper writes a record global and an int global: begin returns
    # (recorded, hits), member 0 being its own formal ``d``.
    "global_vec3_and_int": (
        "vec3 recorded; int hits = 0;\n"
        "void begin(vec3 d){ recorded = d; hits += 1; }\n"
        "void main(){ begin(vec3(1.0, 2.5, 3.0)); "
        "o[0] = recorded.y + float(hits); }\n", 3.5),
    # An array global written by a helper (``[0] * 4`` handed to begin).
    "global_uint_array": (
        "uint active[4];\nvoid begin(){ active[2] = 5u; }\n"
        "void main(){ begin(); o[0] = float(active[2]); }\n", 5.0),
    # A caller local that shadows a global is not the global (issue
    # e24175de): the helper reads the global's own binding, 0.
    "global_shadowed_by_caller_local": (
        "int g = 0;\nint get(){ return g; }\n"
        "void main(){ int g = 5; o[0] = float(get()); }\n", 0.0),
    # The same through an intermediate helper that declares the local.
    "global_shadowed_in_intermediate_helper": (
        "int g = 0;\nint get(){ return g; }\n"
        "int mid(){ int g = 5; return get(); }\n"
        "void main(){ o[0] = float(mid()); }\n", 0.0),
    # The write-back lands in the global, not in the shadowing local:
    # get() sees bump's 1, the local keeps its 5 (1 + 10 * 5).
    "global_written_under_shadowing_local": (
        "int g = 0;\nvoid bump(){ g += 1; }\nint get(){ return g; }\n"
        "void main(){ int g = 5; bump(); o[0] = float(get()) + 10.0 * float(g); }\n",
        51.0),
    # A local array written and read in place.
    "local_float_array": (
        "void main(){ float a[4]; a[1] = 2.5; o[0] = a[1] + a[3]; }\n", 2.5),
}


# Lowered correctly (``mid`` hands ``get`` its carried ``g``), but ``mid``
# carries ``g`` inout and forwards it, a shape the compiler refuses on main
# today (issue 45a22a5d318b429d9c4fa29b2f8f5390).  The refusal is pinned to
# that issue's own diagnostic, so another concordance regression in this
# program fails; when the program lowers, ``pytest.raises`` fails too:
# remove the entry when that issue is fixed.
_KNOWN_REFUSALS = {
    "global_shadowed_in_intermediate_helper": (
        r"post ssa_block_geometry \('main__mid__specialized_[0-9a-f]+@control:0'"
        r".*REVISE without a changed source"),
}


@pytest.mark.parametrize("name", sorted(_PROGRAMS))
def test_inout_and_mutable_globals_run_on_llvm(name):
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

    def lower():
        return lower_ast_source_to_ssa(
            _HEADER + body, "main", source_language="glsl",
            extraction_contract=contract,
            python_bindings={"AbstractTensor": AbstractTensor},
            tensor_ssa_reference=c_backend_repository_ssa_reference(),
            runtime_closure_only=True, progress=lambda _: None,
            progress_bars=False,
        )

    refusal = _KNOWN_REFUSALS.get(name)
    if refusal is not None:
        from src.compiler.identity_concordance import ConcordanceRefusal

        with pytest.raises(ConcordanceRefusal, match=refusal):
            lower()
        pytest.xfail("issue 45a22a5d318b429d9c4fa29b2f8f5390")
    module, _outputs, exports = lower()
    entry = exports[0]
    names = dict(module.functions[entry].metadata["parameter_names"])
    emitted = emit_ssa_function_to_llvm(module, entry)
    assert not emitted.shortfalls, emitted.shortfalls
    artifact = compile_artifact(emitted, directory=tempfile.mkdtemp())
    out = np.zeros(1, dtype=np.float32)
    prepare_artifact_execution(artifact, {names["o"]: out}).run()
    assert float(out[0]) == expected


def test_carried_global_names_its_own_binding_not_a_shadowing_local():
    """Issue e24175de: the argument and the write-back target are the global."""
    import ast

    from src.compiler.glsl_ast_ingestion import lower_glsl_source_to_ast

    source = _HEADER + (
        "int g = 0;\nvoid bump(){ g += 1; }\n"
        "int mid(){ int g = 5; bump(); return g; }\n"
        "void main(){ int g = 7; bump(); o[0] = float(mid()) + float(g); }\n")
    lowered = lower_glsl_source_to_ast(source)
    assert not lowered.shortfalls, lowered.shortfalls
    calls = [node for node in ast.walk(lowered.module)
             if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
             and isinstance(node.value.func, ast.Name) and node.value.func.id == "bump"]
    assert len(calls) == 2, ast.unparse(lowered.module)
    for call in calls:
        assert [ast.unparse(arg) for arg in call.value.args] == ["g"], ast.unparse(call)
        assert [ast.unparse(target) for target in call.targets] == ["g"], ast.unparse(call)
