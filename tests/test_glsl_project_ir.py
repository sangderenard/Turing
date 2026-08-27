from src.compiler.glsl_project_ir import (
    compile_glsl_project_to_webgl,
    emit_webgl_from_glsl_project,
    ingest_glsl_project,
    plan_glsl_project_execution,
)


SOURCE = """#version 430 core
layout(std430, binding = 7) readonly buffer MaterialChunk { float material[]; };
in vec3 normal;
out vec4 FragColor;
uniform float gain = 1.0;

float highlight(float facing) {
    return pow(max(facing, 0.0), 16.0);
}

void main() {
    float value = material[3 + int(normal.x)] * gain;
    if (!gl_FrontFacing) { discard; }
    FragColor = vec4(value + highlight(normal.z));
}
"""


def test_raw_glsl_is_interpreted_line_by_line_into_project_ir_before_targeting():
    project = ingest_glsl_project(SOURCE, source_name="material")

    assert project.complete, [item.format() for item in project.diagnostics]
    assert len(project.instructions) == len(SOURCE.split("\n"))
    assert [item.name for item in project.functions] == ["highlight", "main"]
    assert project.storage_buffers[0].member_name == "material"
    assert project.storage_buffers[0].binding == 7
    assert project.instructions[1].kind == "storage-buffer"
    discard_line = next(
        item for item in project.instructions if "discard" in item.text
    )
    assert "discard" in discard_line.attributes["controls"]
    assert project.source_sha256


def test_execution_plan_keeps_transitively_fragment_dependent_helpers_fragment():
    project = ingest_glsl_project(SOURCE, source_name="material")
    plan = plan_glsl_project_execution(project)
    sections = {item.function: item for item in plan.sections}

    assert plan.selected_graphics_backend == "webgl2-fragment"
    assert plan.compute_sections == ()
    assert sections["main"].stage == "fragment"
    assert "uses fragment discard" in sections["main"].reasons
    assert sections["highlight"].stage == "fragment"
    assert "called from a fragment-bound function" in sections["highlight"].reasons


def test_webgl_lowering_consumes_project_ir_and_specializes_readonly_ssbo():
    project = ingest_glsl_project(SOURCE, source_name="material")
    emitted = emit_webgl_from_glsl_project(project)

    assert emitted.complete, [item.format() for item in emitted.diagnostics]
    assert emitted.project_ir is project
    assert emitted.source.startswith("#version 300 es")
    assert "layout(std430" not in emitted.source
    assert "uniform highp sampler2D turing_ssbo_material;" in emitted.source
    assert "turing_ssbo_load_material(int(3 + int(normal.x)))" in emitted.source
    assert "uniform float gain;" in emitted.source
    assert "if (!gl_FrontFacing) { discard; }" in emitted.source
    assert "FragColor = vec4(value + highlight(normal.z));" in emitted.source
    assert emitted.manifest()["bindings"][0]["source_binding"] == 7


def test_convenience_compiler_has_no_source_to_target_bypass():
    emitted = compile_glsl_project_to_webgl(SOURCE, source_name="material")

    assert emitted.project_ir.schema == "turing.glsl-project-ir.v1"
    assert emitted.manifest()["project_ir"]["line_count"] == len(SOURCE.split("\n"))


def test_writable_storage_is_a_visible_webgl_shortfall():
    source = """#version 430 core
layout(std430, binding = 0) buffer State { float state[]; };
out vec4 color;
void main() { state[0] = 1.0; color = vec4(1.0); }
"""
    emitted = compile_glsl_project_to_webgl(source)

    assert not emitted.complete
    assert any(item.code == "WEBGL_STORAGE_WRITE" for item in emitted.diagnostics)


def test_vertex_stage_uses_the_same_project_ir_route():
    source = """#version 430 core
layout(location = 0) in vec3 position;
uniform mat4 transform;
out vec3 location;
void main() { location = position; gl_Position = transform * vec4(position, 1.0); }
"""
    emitted = compile_glsl_project_to_webgl(
        source, source_name="mesh", stage_hint="vertex"
    )

    assert emitted.complete
    assert emitted.manifest()["stage"] == "vertex"
    assert emitted.execution_plan.selected_graphics_backend == "webgl2-vertex"
    assert emitted.execution_plan.sections[0].stage == "vertex"
    assert "gl_Position = transform * vec4(position, 1.0);" in emitted.source
