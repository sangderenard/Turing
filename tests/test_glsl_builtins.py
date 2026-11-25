"""GLSL built-ins the reader lowers to existing compiler operations."""

import ast

from src.compiler.glsl_ast_ingestion import lower_glsl_source_to_ast


def test_exp2_is_two_raised_to_x_per_component():
    result = lower_glsl_source_to_ast(
        "uniform float x; out float y; "
        "void main(){ vec2 v = exp2(vec2(x, 1.0)); y = exp2(x) + v.y; }")
    assert result.complete, result.shortfall_report()
    assert any(isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow)
               for node in ast.walk(result.module))
    namespace = {"__name__": __name__}
    exec(compile(result.module, "<glsl>", "exec"), namespace)
    assert namespace["main"](3.0) == 10.0


def _run(source, *arguments):
    result = lower_glsl_source_to_ast(source)
    assert result.complete, result.shortfall_report()
    namespace = {"__name__": "__main__"}
    exec(compile(result.module, "<glsl>", "exec"), namespace)
    return namespace["main"](*arguments)


def test_global_and_local_scalar_constants_are_constant_expressions():
    # A global const scalar is its literal; a local ``const int`` may size
    # a local array.
    assert _run('''const int N = 3;
const float HALF = 0.5;
out float y; void main(){ const int K = 4; float a[K];
  for(int i=0;i<N;++i){ a[i] = HALF*float(i); } y = a[2] + float(N); }''') == 4.0


def test_constant_array_of_vectors_reads_its_lanes():
    source = '''const int N = 2;
const vec3 B[N] = vec3[](vec3(1.0, 2.0, 3.0), vec3(-4.0, 5.5, 6.0));
uniform int i; out float y; float pick(int k){ return B[k].y; }
void main(){ y = pick(i) + B[1].x; }'''
    assert _run(source, 0) == -2.0
    assert _run(source, 1) == 1.5


_SAMPLER = '''layout(local_size_x=1) in;
layout(binding=0) uniform usampler2D t;
layout(std430, binding=1) buffer O { uint o[]; };
void main(){ o[0] = texelFetch(t, ivec2(1, 2), 0).%s; }'''


def test_sampler_format_is_the_callers_declaration_not_an_assumption():
    # GLSL states no texel format for a sampler: without the caller's
    # declared field it is a shortfall, never a guessed channel count.
    undeclared = lower_glsl_source_to_ast(_SAMPLER % "r")
    assert any("has no texel format" in s.message for s in undeclared.shortfalls)
    wider = lower_glsl_source_to_ast(
        _SAMPLER % "r", texture_fields={"t": {"dtype": "uint32", "rank": 3}})
    assert any("multi-channel" in s.message for s in wider.shortfalls)
    declared = lower_glsl_source_to_ast(
        _SAMPLER % "r", texture_fields={"t": {"dtype": "uint32", "rank": 2}})
    assert declared.complete, declared.shortfall_report()
    assert declared.textures == (("t", "uint32", 2),)
    # The read is the stored lane itself: no constructed gvec4 lanes.
    text = ast.unparse(declared.module)
    read = next(node for node in ast.walk(declared.module)
                if isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name) and node.value.id == "t")
    assert [ast.unparse(axis) for axis in read.slice.elts] == [
        "__glsl_value_0.y", "__glsl_value_0.x"]
    assert "uvec4" not in text


def test_texel_lanes_the_format_does_not_store_are_refused():
    result = lower_glsl_source_to_ast(
        _SAMPLER % "g", texture_fields={"t": {"dtype": "uint32", "rank": 2}})
    assert any("does not store" in s.message for s in result.shortfalls)


def test_glsl_names_that_are_python_keywords_are_renamed():
    # The Studio tracer declares ``uint from``: valid GLSL, a Python keyword.
    source = '''uniform int lambda; out int y; int pass(int from){ return from + 1; }
void main(){ int from = lambda; y = pass(from); }'''
    result = lower_glsl_source_to_ast(source)
    assert result.complete, result.shortfall_report()
    text = ast.unparse(result.module)
    ast.parse(text)
    assert "from__glsl_keyword" in text and "pass__glsl_keyword" in text
    namespace = {"__name__": "__main__"}
    exec(compile(ast.parse(text), "<glsl>", "exec"), namespace)
    assert namespace["main"](4) == 5


def test_mutable_globals_are_main_owned_state_threaded_through_helpers():
    # ``vec3 g;`` is per-invocation state: written in one function, read in
    # another.  Read as a Python local, the write was silently dropped.
    # ``main`` now owns it (initializer or zero); a helper that writes a
    # value global hands it back, an array global is written in place.
    source = '''vec3 g; uint flags[4]; int k = 2, m = -1; out float y;
void put(vec3 d){ g = d; flags[1] = 7u; k += 1; }
float get(){ return g.x + float(flags[1]) + float(k) + float(m); }
void main(){ put(vec3(1.5)); y = get(); }'''
    assert _run(source) == 1.5 + 7 + 3 - 1


def test_workgroup_shared_and_unhanded_global_writes_are_shortfalls():
    shared = lower_glsl_source_to_ast(
        "shared float tile[8]; out float y; void main(){ y = tile[0]; }")
    assert any("workgroup-shared variable tile" in s.message for s in shared.shortfalls)


def test_store_through_a_runtime_vector_index_writes_that_component():
    source = '''uniform int k; out int y; void main(){
  ivec3 q = ivec3(1, 2, 3); q[k] = 10; q[k] += 5; y = q.x*100 + q.y*10 + q.z; }'''
    assert [_run(source, k) for k in (0, 1, 2)] == [1523, 253, 135]
