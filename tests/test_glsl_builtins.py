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


def test_store_through_a_runtime_vector_index_writes_that_component():
    source = '''uniform int k; out int y; void main(){
  ivec3 q = ivec3(1, 2, 3); q[k] = 10; q[k] += 5; y = q.x*100 + q.y*10 + q.z; }'''
    assert [_run(source, k) for k in (0, 1, 2)] == [1523, 253, 135]
