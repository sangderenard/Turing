"""GLSL storage buffers are caller-owned AbstractTensor parameters.

A ``buffer`` block member is a rank-1 span the caller allocates and hands
in by reference.  It is a parameter of ``main`` and of every helper that
touches it (directly or through a call); ``atomicAdd``/``atomicOr`` on one
of its elements is ``AbstractTensor.atomic_add``/``atomic_or``, and a
compute entry's outputs are its writes into those spans.
"""

import ast
from pathlib import Path
import tempfile

import numpy as np

from src.compiler.glsl_ast_ingestion import lower_glsl_source_to_ast

ROOT = Path(__file__).resolve().parents[1]

SHADER = '''#version 430
layout(local_size_x=1) in;
layout(std430, binding=0) buffer Words { uint words[]; };
layout(std430, binding=1) buffer Data { float data[]; };
uint claim(){ uint row=atomicAdd(words[0],1u); if(row>=4u){atomicOr(words[1],1u);} return row; }
void main(){ uint row=claim(); if(row<4u){ data[row]=2.5; } }
'''


def test_buffers_become_tensor_parameters_threaded_through_helpers():
    result = lower_glsl_source_to_ast(SHADER)
    assert result.complete, result.shortfall_report()
    assert result.storage_buffers == (
        ("words", "uint32", None, None), ("data", "float32", None, None),
    )
    functions = {
        node.name: node for node in result.module.body
        if isinstance(node, ast.FunctionDef)
    }
    # claim touches only words; main touches data and, through claim, words.
    assert [arg.arg for arg in functions["claim"].args.args] == ["words"]
    assert [arg.arg for arg in functions["main"].args.args] == ["words", "data"]
    text = ast.unparse(result.module)
    assert "from src.common.tensors.abstraction import AbstractTensor" in text
    assert "AbstractTensor.atomic_add(words," in text
    assert "AbstractTensor.atomic_or(words," in text
    assert "claim(words)" in text


def test_buffer_length_and_unsupported_members_are_named():
    result = lower_glsl_source_to_ast('''
        layout(std430, binding=0) buffer B { uint words[]; };
        layout(std430, binding=1) buffer S { bvec4 rows[]; };
        void main(){ words[0] = uint(words.length()); }''')
    text = ast.unparse(result.module)
    assert "len(words)" in text
    assert any(item.code == "GLSL_STORAGE" and "rows" in item.message
               for item in result.shortfalls)


def test_object_like_defines_expand_before_reading():
    # The Studio shader names its scalar-visit buffer through a macro
    # (``#define studio_scalar_columns studio_scalar_visits``) and its flags
    # as macros with trailing comments.
    result = lower_glsl_source_to_ast('''#version 430
#define FLAG 4u   // a flag
#define columns visits
layout(std430, binding=0) buffer V { uint visits[]; };
void main(){ atomicOr(columns[1], FLAG); }
''')
    assert result.complete, result.shortfall_report()
    text = ast.unparse(result.module)
    assert "AbstractTensor.atomic_or(visits," in text
    refused = lower_glsl_source_to_ast(
        "#define SQ(x) ((x)*(x))\nout float y; void main(){ y = 1.0; }")
    assert any(item.code == "GLSL_PREPROCESSOR" for item in refused.shortfalls)


def _compile(result, shapes):
    from src.common.tensors.abstraction import AbstractTensor
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm,
    )

    assert result.complete, result.shortfall_report()
    contract = ExtractionContract(
        ROOT / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({"records": {}, "bindings": [], "values": [
        # Every function that names the buffer takes the caller's span.
        # A buffer of vectors or structs is its std430 region: [rows, stride].
        {"function": "*", "parameter": name, "storage": "span",
         "dtype": dtype, "rank": 1 if stride is None else 2,
         "shape": [extent if extent is not None else shapes[name],
                   *(() if stride is None else (stride,))],
         "python_type": "src.common.tensors.abstraction.AbstractTensor"}
        for name, dtype, extent, stride in result.storage_buffers
    ]}).with_execution_file(
        ROOT / "extraction_contracts/vehicle_full_native_execution.yaml"
    )
    module, _outputs, exports = lower_ast_source_to_ssa(
        ast.unparse(result.module), "main", extraction_contract=contract,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        runtime_closure_only=True, progress=lambda _: None,
        progress_bars=False,
    )
    entry = exports[0]
    names = dict(module.functions[entry].metadata["parameter_names"])
    emitted = emit_ssa_function_to_llvm(module, entry)
    assert emitted.shortfalls == ()
    return compile_artifact(emitted, directory=tempfile.mkdtemp()), names


def test_scalar_buffer_member_is_a_one_element_span():
    from src.compiler.ssa_llvm_backend import prepare_artifact_execution

    result = lower_glsl_source_to_ast('''#version 430
layout(local_size_x=1) in;
layout(std430, binding=0) buffer CounterBuf { uint seg_count; uint ray_count; };
void main(){ uint slot=atomicAdd(seg_count,1u); ray_count=ray_count+slot; }
''')
    assert result.storage_buffers == (
        ("seg_count", "uint32", 1, None), ("ray_count", "uint32", 1, None),
    )
    artifact, names = _compile(result, {})
    seg = np.zeros(1, dtype=np.uint32)
    ray = np.zeros(1, dtype=np.uint32)
    for _ in range(4):
        prepare_artifact_execution(
            artifact, {names["seg_count"]: seg, names["ray_count"]: ray},
        ).run()
    # slots 0, 1, 2, 3 are claimed; ray_count sums them.
    assert seg.tolist() == [4]
    assert ray.tolist() == [6]


def test_std430_layout_of_vector_and_struct_elements():
    from src.compiler.glsl_ast_ingestion import _GLSLTypes, _std430_element

    types = _GLSLTypes({
        "Ray": [("vec4", "origin"), ("vec4", "dir")],
        "Packed": [("float", "a"), ("vec3", "b"), ("vec2", "c")],
        "Mixed": [("vec4", "p"), ("uvec4", "q")],
    })
    assert _std430_element("vec4", types) == (4, None, "float")
    # A vec3 array element is aligned (and so strided) as a vec4.
    assert _std430_element("vec3", types) == (4, None, "float")
    assert _std430_element("Ray", types) == (
        8, {"origin": (0, "vec4"), "dir": (4, "vec4")}, "float")
    # float at 0, vec3 aligned to 4, vec2 at 7 rounded to 8; stride 12.
    assert _std430_element("Packed", types) == (
        12, {"a": (0, "float"), "b": (4, "vec3"), "c": (8, "vec2")}, "float")
    # Lanes of two scalar types are not one span's elements.
    assert _std430_element("Mixed", types) is None


def test_struct_buffer_fields_read_and_write_their_lanes():
    from src.compiler.ssa_llvm_backend import prepare_artifact_execution

    result = lower_glsl_source_to_ast('''#version 430
layout(local_size_x=1) in;
struct Ray { vec4 origin_flags; vec4 dir; };
layout(std430, binding=0) buffer Rays { Ray rays[]; };
layout(std430, binding=1) buffer Segs { vec4 segs[]; };
layout(std430, binding=2) buffer Out { float outv[]; };
void main(){
  uint i = 1u;
  Ray r = rays[i];
  outv[0] = rays[i].dir.y + r.origin_flags.w;
  segs[2] = vec4(rays[0].origin_flags.xyz, 7.0);
  rays[i].dir.x += 1.0;
}
''')
    assert result.storage_buffers == (
        ("rays", "float32", None, 8), ("segs", "float32", None, 4),
        ("outv", "float32", None, None),
    )
    artifact, names = _compile(result, {"rays": 2, "segs": 3, "outv": 2})
    rays = np.arange(16, dtype=np.float32).reshape(2, 8)
    segs = np.zeros((3, 4), dtype=np.float32)
    outv = np.zeros(2, dtype=np.float32)
    prepare_artifact_execution(artifact, {
        names["rays"]: rays, names["segs"]: segs, names["outv"]: outv,
    }).run()
    # rays[1].dir.y is lane 5 of row 1 (13); rays[1].origin_flags.w is 11.
    assert outv.tolist() == [24.0, 0.0]
    assert segs[2].tolist() == [0.0, 1.0, 2.0, 7.0]
    # rays[1].dir.x (lane 4 of row 1) was 12 and is incremented in place.
    assert rays[1].tolist() == [8.0, 9.0, 10.0, 11.0, 13.0, 13.0, 14.0, 15.0]


def test_bit_reinterpretation_keeps_every_bit():
    from src.common.tensors.abstraction_methods.bit_reinterpretation import (
        BIT_REINTERPRETATION_INVERSES, float_bits_to_uint, uint_bits_to_float,
    )
    from src.compiler.ssa_llvm_backend import prepare_artifact_execution

    # Eager: exact inverses on a value and on its bits.
    assert int(float_bits_to_uint(1.0)) == 0x3F800000
    assert float(uint_bits_to_float(float_bits_to_uint(3.25))) == 3.25
    assert BIT_REINTERPRETATION_INVERSES["float_bits_to_uint"] == "uint_bits_to_float"

    result = lower_glsl_source_to_ast('''#version 430
layout(local_size_x=1) in;
layout(std430, binding=0) buffer F { float f[]; };
layout(std430, binding=1) buffer U { uint u[]; };
void main(){ u[0] = floatBitsToUint(f[0]); f[1] = uintBitsToFloat(u[1]); }
''')
    text = ast.unparse(result.module)
    assert "float_bits_to_uint(f[" in text and "uint_bits_to_float(u[" in text
    artifact, names = _compile(result, {"f": 2, "u": 2})
    f = np.array([1.0, 0.0], dtype=np.float32)
    u = np.array([0, 0x40490FDB], dtype=np.uint32)
    prepare_artifact_execution(artifact, {names["f"]: f, names["u"]: u}).run()
    # Each is the same 32 bits under the other type, no value conversion.
    assert int(u[0]) == 0x3F800000
    assert f[1] == np.uint32(0x40490FDB).view(np.float32)


def test_vector_record_crosses_a_helper_call():
    from src.compiler.ssa_llvm_backend import prepare_artifact_execution

    result = lower_glsl_source_to_ast('''#version 430
layout(local_size_x=1) in;
layout(std430, binding=0) buffer Out { float outv[]; };
float weigh(vec3 d, float s){ return dot(d, vec3(1.0, 2.0, 3.0)) * s; }
void main(){ vec3 d = vec3(outv[0], outv[1], 0.5); outv[1] = weigh(d, 2.0); }
''')
    artifact, names = _compile(result, {"outv": 2})
    outv = np.array([1.0, 2.0], dtype=np.float32)
    prepare_artifact_execution(artifact, {names["outv"]: outv}).run()
    assert outv.tolist() == [1.0, (1.0 + 2.0 * 2.0 + 0.5 * 3.0) * 2.0]


def test_integer_vector_literal_record_keeps_its_declared_field_width():
    # ``ivec3(3, 0, 2)``: each field literal is materialized at the record
    # field's declared int32, which the final audit now accepts from the
    # field declaration; the record is then passed to a helper and read again.
    from src.compiler.ssa_llvm_backend import prepare_artifact_execution

    result = lower_glsl_source_to_ast('''#version 430
layout(local_size_x=1) in;
layout(std430, binding=0) buffer Out { float outv[]; };
void put(ivec3 q){ outv[0] = float(q.x * 10 + q.z); }
void main(){ ivec3 q = ivec3(3, 0, 2); put(q); outv[1] = float(q.z * 100 + q.y * 10 + q.x); }
''')
    artifact, names = _compile(result, {"outv": 2})
    outv = np.zeros(2, dtype=np.float32)
    prepare_artifact_execution(artifact, {names["outv"]: outv}).run()
    assert outv.tolist() == [32.0, 203.0]


def test_compute_shader_with_buffer_atomics_runs_on_caller_storage():
    from src.compiler.ssa_llvm_backend import prepare_artifact_execution

    result = lower_glsl_source_to_ast(SHADER)
    artifact, names = _compile(result, {"words": 2, "data": 4})
    words = np.zeros(2, dtype=np.uint32)
    data = np.zeros(4, dtype=np.float32)
    # Six invocations run one after another: rows 0..3 are written, rows 4
    # and 5 overflow the capacity of 4 and set the overflow bit.
    for _ in range(6):
        prepare_artifact_execution(
            artifact, {names["words"]: words, names["data"]: data},
        ).run()
    assert words.tolist() == [6, 1]
    assert data.tolist() == [2.5, 2.5, 2.5, 2.5]
