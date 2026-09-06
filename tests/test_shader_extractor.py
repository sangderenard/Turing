import ast
import inspect
from pathlib import Path
import textwrap

from src.compiler.shader_extractor import (
    discover_shader_compile_calls,
    extract_shader_compile_calls,
)
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


def test_extractor_follows_raw_opengl_shader_handle_flow():
    shaders = extract_shader_compile_calls(textwrap.dedent("""
        VERTEX = "#version 430\\nvoid main() { gl_Position = vec4(0.0); }"

        def compile_shader(source, stage):
            handle = glCreateShader(stage)
            glShaderSource(handle, source)
            glCompileShader(handle)

        def build_program():
            compile_shader(VERTEX, GL_VERTEX_SHADER)
    """))

    assert len(shaders) == 1
    assert shaders[0].stage == "vertex"
    assert shaders[0].source_name == "VERTEX"
    assert shaders[0].compiler == "glShaderSource/glCompileShader"
    assert "gl_Position" in shaders[0].source


def test_extractor_follows_nested_program_helpers_without_guessing_dynamic_source():
    shaders = extract_shader_compile_calls(textwrap.dedent("""
        FRAGMENT = "#version 430\\nout vec4 color; void main() { color = vec4(1.0); }"

        def compile_shader(source, stage):
            handle = glCreateShader(stage)
            glShaderSource(handle, source)
            glCompileShader(handle)
            return handle

        def link_program(vertex_source, fragment_source):
            compile_shader(vertex_source, GL_VERTEX_SHADER)
            compile_shader(fragment_source, GL_FRAGMENT_SHADER)

        def build_program(runtime_vertex_source):
            link_program(runtime_vertex_source, FRAGMENT)
    """))

    assert [(shader.stage, shader.source_name) for shader in shaders] == [
        ("fragment", "FRAGMENT"),
    ]
    assert shaders[0].compiler == "glShaderSource/glCompileShader"


def test_extractor_recognizes_actual_raw_opengl_renderer_compile_sites():
    renderer_path = (
        Path(inspect.getsourcefile(run_symbolic_spring_image)).parent
        / "opengl_render"
        / "renderer.py"
    )
    shaders = extract_shader_compile_calls(renderer_path)
    by_name = {shader.source_name: shader for shader in shaders}

    assert {
        "MESH_VS", "MESH_FS", "LINE_VS", "LINE_FS", "POINT_VS", "POINT_FS",
    } == by_name.keys()
    assert {shader.stage for shader in shaders} == {"vertex", "fragment"}
    assert {
        shader.compiler for shader in shaders
    } == {"glShaderSource/glCompileShader"}


def test_discovery_packages_real_renderer_sources_with_deterministic_origins():
    rendering_root = Path(inspect.getsourcefile(run_symbolic_spring_image)).parent
    bundle = discover_shader_compile_calls(rendering_root)
    manifest = bundle.to_mapping()

    assert manifest["schema"] == "turing.extracted-shaders.v1"
    assert bundle.select(stage="vertex", language="glsl")
    assert bundle.select(stage="fragment", language="glsl")
    assert all(shader.filename.startswith(str(rendering_root)) for shader in bundle.shaders)
    assert all(len(shader.sha256) == 64 for shader in bundle.shaders)
    assert [shader.filename for shader in bundle.shaders] == sorted(
        shader.filename for shader in bundle.shaders
    )


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
