import ast
import inspect
import textwrap

from src.compiler.shader_extractor import extract_shader_compile_calls
from src.rendering.opengl_render.fluxspring_shader import (
    load_fluxspring_graph_shaders,
)
from src.rendering.symbolic_spring_image import run_symbolic_spring_image


def test_extractor_recognizes_actual_fluxspring_compile_calls():
    from src.common.tensors.autoautograd import spring_async_toy

    shaders = extract_shader_compile_calls(inspect.getsourcefile(spring_async_toy))
    by_stage = {shader.stage: shader for shader in shaders}

    assert {"vertex", "fragment"} <= by_stage.keys()
    assert by_stage["vertex"].source_name == "vsrc"
    assert "gl_Position" in by_stage["vertex"].source
    assert by_stage["fragment"].source_name == "fsrc"
    assert "FragColor" in by_stage["fragment"].source
    assert all(len(shader.sha256) == 64 for shader in shaders)


def test_whole_program_function_contains_every_requested_stage():
    source = textwrap.dedent(inspect.getsource(run_symbolic_spring_image))
    tree = ast.parse(source)
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert {
        "ProcessGraph",
        "symbolically_reduce_process_graph",
        "load_fluxspring_graph_shaders",
        "run_precompiled_graph",
    } <= calls
    vertex, fragment = load_fluxspring_graph_shaders()
    assert "gl_Position" in vertex
    assert "FragColor" in fragment
