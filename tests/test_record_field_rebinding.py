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
    # ``q`` rebound to a call's returned tuple member names the record the
    # call returned (here its own formal ``v``), not the earlier ``q``.
    "rebound_to_returned_member": ('''def ident(v: vec2, k: 'int32') -> tuple[vec2, 'int32']:
    k += 1
    return (v, k)

def main(out: AbstractTensor):
    q: vec2 = vec2(0.0, 0.0)
    n: 'int32' = 1
    q, n = ident(vec2(2.0, 3.0), n)
    out[0] = q.y + float(n)
''', 5.0),
    # The GLSL global shape: the callee rebinds its formal ``q`` to ``v``
    # and returns it beside an int.
    "rebound_formal_returned_as_member": ('''def ident(v: vec2, q: vec2, k: 'int32') -> tuple[vec2, 'int32']:
    q = v
    k += 1
    return (q, k)

def main(out: AbstractTensor):
    q: vec2 = vec2(0.0, 0.0)
    n: 'int32' = 1
    q, n = ident(vec2(2.0, 3.0), q, n)
    out[0] = q.y + float(n)
''', 5.0),
    # GLSL block scope read as Python: ``vec3 rd = ...`` declared in the
    # Studio bounce loop is a fresh record every iteration.  A record built
    # in the loop body and read there is not carried by the loop.
    "built_in_loop_body": ('''def main(out: AbstractTensor):
    t: 'float32' = 1.0
    for i in range(3):
        r: vec2 = vec2(t, 2.0)
        t = t + r.x
    out[0] = t
''', 8.0),
    # The same fresh record handed to a helper in the loop: each unrolled
    # copy of the call is the callee's call, with the callee's result type.
    "built_in_loop_body_passed_to_helper": ('''def peek(r: vec2) -> 'float32':
    return r.y

def main(out: AbstractTensor):
    t: 'float32' = 1.0
    for i in range(3):
        r: vec2 = vec2(t, 2.0)
        t = t + r.x + peek(r)
    out[0] = t
''', 22.0),
    # The tracer's ``emit_pkt.z *= w`` under an ``if``: a field of a record
    # built in the loop body, written on one arm.  The arm that leaves it
    # alone holds the constructor's operand, not a read of the record.
    "conditional_write_in_while_body": ('''def main(out: AbstractTensor):
    t: 'float32' = 1.0
    i: 'int32' = 0
    while i < 3:
        r: vec2 = vec2(t, 2.0)
        if t > 1.5:
            r.y *= 2.0
        t = t + r.y
        i += 1
    out[0] = t
''', 11.0),
    # GLSL ``vec2 copy(vec2 v){ return v; }`` read with value copies: the
    # helper returns a new record of its formal's own fields.
    "returned_copy_of_formal": ('''def ident(v: vec2) -> vec2:
    return vec2(v.x, v.y)

def main(out: AbstractTensor):
    p: vec2 = vec2(2.0, 3.0)
    q: vec2 = ident(p)
    out[0] = q.y
''', 3.0),
}

# A record from before the loop that a helper in the loop body stores into
# is carried by the loop: its field read has no reloaded value and is
# refused by name.
_CARRIED_STORED_RECORD = '''def bump(r: vec2):
    r.x = r.x + 1.0

def main(out: AbstractTensor):
    t: 'float32' = 0.0
    r: vec2 = vec2(1.0, 2.0)
    for i in range(3):
        t = t + r.x
        bump(r)
    out[0] = t
'''


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


def test_record_carried_into_a_storing_loop_is_refused_by_name():
    from src.common.tensors.abstraction import AbstractTensor
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    contract = ExtractionContract(
        ROOT / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({"records": {}, "bindings": [], "values": [{
        "function": "*", "parameter": "out", "storage": "span",
        "dtype": "float32", "rank": 1, "shape": [1],
        "python_type": "src.common.tensors.abstraction.AbstractTensor",
    }]}).with_execution_file(
        ROOT / "extraction_contracts/vehicle_full_native_execution.yaml"
    )
    with pytest.raises(ValueError, match="record_stored_by_call"):
        lower_ast_source_to_ssa(
            _HEADER + _CARRIED_STORED_RECORD, "main",
            extraction_contract=contract,
            python_bindings={"AbstractTensor": AbstractTensor},
            tensor_ssa_reference=c_backend_repository_ssa_reference(),
            runtime_closure_only=True, progress=lambda _: None,
            progress_bars=False,
        )
