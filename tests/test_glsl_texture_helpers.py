"""Texture memory crossing helper calls, compiled and run on LLVM.

One program exercises the faults the Studio response shader met:

* ``splat(q, 5u)`` forwards its declared ``uint`` on to ``atomic_add``: the
  width is the formal's own ``planner_specialization`` row.
* ``main`` reads ``q``'s fields again after the call: the constructor
  literals stay control-owned, so the call's field arguments are defined
  before it (not loaded from a later region).
* ``atomic_add(layer, (q.z, q.y, q.x), v)``: the tuple index is three axes
  (rank 0 element), addressed by its members, and the retired tuple
  descriptor ends its storage lifetime.
"""

import ast
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

_SOURCE = '''#version 430
layout(local_size_x=1) in;
layout(r32ui, binding = 0) uniform uimage3D uLayer;
layout(binding = 8) uniform usampler3D uFwd;
layout(std430, binding=0) buffer Out { uint outv[]; };
void splat(ivec3 q, uint v){ uint prior = imageAtomicAdd(uLayer, q, v); outv[0] = prior; }
void main(){ ivec3 q = ivec3(1, 0, 2); splat(q, 5u); outv[1] = texelFetch(uFwd, q, 0).r; }
'''


def _contract(values):
    from src.compiler.extraction_contract import ExtractionContract

    return ExtractionContract(
        ROOT / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({"records": {}, "bindings": [], "values": values}).with_execution_file(
        ROOT / "extraction_contracts/vehicle_full_native_execution.yaml"
    )


def _lower(source, entry, values):
    from src.common.tensors.abstraction import AbstractTensor
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    return lower_ast_source_to_ssa(
        source, entry, extraction_contract=_contract(values),
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        runtime_closure_only=True, progress=lambda _: None,
        progress_bars=False,
    )


def _span(name, dtype, shape):
    return {"function": "*", "parameter": name, "storage": "span",
            "dtype": dtype, "rank": len(shape), "shape": list(shape),
            "python_type": "src.common.tensors.abstraction.AbstractTensor"}


def test_texture_atomic_and_fetch_through_a_helper_run_on_llvm():
    from src.compiler.glsl_ast_ingestion import lower_glsl_source_to_ast
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm,
        prepare_artifact_execution,
    )

    read = lower_glsl_source_to_ast(
        _SOURCE, texture_fields={"uFwd": {"dtype": "uint32", "rank": 3}})
    assert read.complete, read.shortfall_report()
    values = [_span(name, dtype, (2,)) for name, dtype, _e, _s in read.storage_buffers]
    values += [_span(name, dtype, (3, 1, 2)) for name, dtype, _d in read.textures]
    module, _outputs, exports = _lower(ast.unparse(read.module), "main", values)
    entry = exports[0]
    names = dict(module.functions[entry].metadata["parameter_names"])
    emitted = emit_ssa_function_to_llvm(module, entry)
    assert not emitted.shortfalls, emitted.shortfalls
    artifact = compile_artifact(emitted, directory=tempfile.mkdtemp())
    layer = np.zeros((3, 1, 2), dtype=np.uint32)
    layer[2, 0, 1] = 7
    fwd = np.zeros((3, 1, 2), dtype=np.uint32)
    fwd[2, 0, 1] = 42
    outv = np.zeros(2, dtype=np.uint32)
    prepare_artifact_execution(artifact, {
        names["outv"]: outv, names["uLayer"]: layer, names["uFwd"]: fwd,
    }).run()
    assert outv.tolist() == [7, 42]
    assert int(layer[2, 0, 1]) == 12


def test_record_fields_passed_before_a_later_tuple_read_of_them():
    # Plain Python: the constructor literals are read again after the call.
    source = '''from dataclasses import dataclass
from src.common.tensors.abstraction import AbstractTensor

@dataclass
class V:
    x: 'int32'
    y: 'int32'
    z: 'int32'

def use(q: V, out: AbstractTensor, layer: AbstractTensor):
    AbstractTensor.atomic_add(layer, (q.z, q.y, q.x), 5)

def main(out: AbstractTensor, layer: AbstractTensor, fwd: AbstractTensor):
    a: 'int32' = 1
    b: 'int32' = 0
    c: 'int32' = 2
    q: V = V(a, b, c)
    use(q, out, layer)
    out[1] = fwd[q.z, q.y, q.x]
'''
    _lower(source, "main", [
        _span("out", "int32", (2,)), _span("layer", "int32", (3, 1, 2)),
        _span("fwd", "int32", (3, 1, 2)),
    ])
