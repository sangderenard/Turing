"""The component ABI of a host-written shader comes from its declarations.

``glslangValidator``'s reflection database is the oracle where it is
installed: every uniform, block, buffer variable, pipeline input and output
it reports for a real host shader must appear in the published record with
the same type, array size and binding.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest
import yaml

from src.compiler.glsl_interface import GLSLInterfaceRefusal, read_glsl_interface
from src.compiler.shader_deployment import (
    extracted_shader_component,
    publish_extracted_shader,
)
from src.compiler.shader_extractor import ExtractedShader, discover_shader_compile_calls


ROOT = Path(__file__).resolve().parents[1]
GLSLANG = shutil.which("glslangValidator")

# GL type enums glslang's reflection prints, as the record's dtype spelling.
_GL_TYPES = {
    "1404": "i32", "1405": "u32", "1406": "f32", "8b56": "bool",
    "8b50": "f32x2", "8b51": "f32x3", "8b52": "f32x4",
    "8b53": "i32x2", "8b54": "i32x3", "8b55": "i32x4",
    "8dc6": "u32x2", "8dc7": "u32x3", "8dc8": "u32x4",
    "8b5a": "f32x2x2", "8b5b": "f32x3x3", "8b5c": "f32x4x4",
    "8b5e": "sampler2D", "8dc2": "samplerBuffer", "904d": "image2D",
}
_SUFFIX = {"vertex": "vert", "fragment": "frag", "compute": "comp"}


def _shader(stage, source, name="S"):
    return ExtractedShader(stage, source, "compileShader", "host.py", 7, name)


def _ports(abi):
    return {port.name: port for port in abi.ports}


def _resources(deployment):
    return {resource.name: resource for resource in deployment.resources}


def _reflection(tmp_path, shader):
    path = tmp_path / f"{shader.source_name or 'shader'}.{_SUFFIX[shader.stage]}"
    path.write_text(shader.source.lstrip(), encoding="utf-8")
    result = subprocess.run(
        [GLSLANG, "-l", "-q", "--reflect-separate-buffers", str(path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None
    sections: dict[str, list[dict[str, str]]] = {}
    current = None
    for line in result.stdout.splitlines():
        if line.startswith("Local size "):
            sections.setdefault("Local size", []).append(
                {"name": line[len("Local size ")], "size": line.split(":")[1].strip()}
            )
            current = None
        elif line.endswith("reflection:"):
            current = line[: -len(" reflection:")]
            sections[current] = []
        elif current and ":" in line:
            name, fields = line.split(":", 1)
            entry = {"name": name.strip()}
            for item in fields.split(","):
                key, _, value = item.strip().partition(" ")
                entry[key] = value
            sections[current].append(entry)
    return sections


def _assert_matches_reflection(shader, reflection):
    abi, deployment = extracted_shader_component(shader)
    assert abi is not None, deployment.shortfalls
    ports, resources = _ports(abi), _resources(deployment)
    members = {
        member["name"]: (resource, member)
        for resource in deployment.resources for member in resource.members
    }
    for entry in reflection.get("Uniform", ()):
        name = entry["name"].split(".")[-1].split("[")[0]
        if name.startswith("gl_"):
            continue
        if entry["index"] != "-1":
            # A member of the uniform block at that index.
            resource, member = members[name]
            assert resource.kind == "uniform-buffer"
            assert member["dtype"] == _GL_TYPES[entry["type"]], entry
            continue
        assert ports[name].dtype == _GL_TYPES[entry["type"]], entry
        if entry["binding"] != "-1":
            assert resources[name].binding == int(entry["binding"]), entry
    for section in ("Uniform block", "Buffer block"):
        for entry in reflection.get(section, ()):
            resource = resources[entry["name"]]
            assert resource.kind == (
                "uniform-buffer" if section == "Uniform block" else "storage-buffer"
            )
            assert len(resource.members) == int(entry["numMembers"]), entry
            if entry["binding"] != "-1":
                assert resource.binding == int(entry["binding"]), entry
    for entry in reflection.get("Buffer variable", ()):
        name = entry["name"].split(".")[-1].split("[")[0]
        resource, member = members[name]
        assert member["dtype"] == _GL_TYPES[entry["type"]], entry
        runtime = entry["size"] == "0"
        assert (member["array"] == [None]) == runtime, entry
    for entry in reflection.get("Local size", ()):
        axis = "XYZ".index(entry["name"])
        assert deployment.launch.local_size[axis] == int(entry["size"]), entry
    for section, roles in (("Pipeline input", {"feed"}), ("Pipeline output", {"output"})):
        for entry in reflection.get(section, ()):
            if entry["name"].startswith("gl_"):
                continue
            port = ports[entry["name"]]
            assert port.role in roles and port.dtype == _GL_TYPES[entry["type"]], entry


def test_every_host_shader_in_the_repository_matches_glslang_reflection(tmp_path):
    if GLSLANG is None:
        pytest.skip("glslangValidator is not installed")
    shaders = discover_shader_compile_calls(ROOT / "src", strict=False).shaders
    checked = 0
    for shader in shaders:
        reflection = _reflection(tmp_path, shader)
        if reflection is None:
            continue
        _assert_matches_reflection(shader, reflection)
        checked += 1
    # Every one the reader is given today compiles under glslang.
    assert checked == len(shaders) and checked >= 20


COMPUTE = """#version 430
layout(local_size_x = 64, local_size_y = 2) in;
layout(std430, binding = 0) readonly buffer Feed { float values[]; };
layout(std430, binding = 1) writeonly buffer Out { vec4 colors[]; } result;
layout(std430, binding = 2) buffer State { uint count; float history[8]; };
layout(std140, binding = 3) uniform Params { mat4 view; vec2 scale; };
layout(rgba32f, binding = 4) uniform readonly image2D source_image;
layout(binding = 5) uniform sampler2D lut;
uniform float gain;
uniform int taps[4];
shared float scratch[64];
const float PI = 3.14159;
float helper(float x) { return x * gain; }
void main() {
    uint i = gl_GlobalInvocationID.x;
    scratch[gl_LocalInvocationID.x] = helper(values[i]) + taps[0] + history[0];
    result.colors[i] = texture(lut, scale) + imageLoad(source_image, ivec2(i, 0)) + view[0];
    count = i;
}
"""


def test_compute_shader_interface_is_complete_but_its_launch_is_the_hosts(tmp_path):
    shader = _shader("compute", COMPUTE, "kernel")
    abi, deployment = extracted_shader_component(shader)
    ports, resources = _ports(abi), _resources(deployment)

    assert [(port.slot, port.name, port.role) for port in abi.ports] == [
        (0, "Feed", "feed"), (1, "source_image", "feed"), (2, "lut", "feed"),
        (3, "Out", "output"), (4, "State", "output"),
        (5, "Params", "uniform"), (6, "gain", "uniform"), (7, "taps", "uniform"),
    ]
    assert (ports["Feed"].dtype, ports["Feed"].transport, ports["Feed"].backend_binding) == (
        "f32", "storage-buffer", 0,
    )
    assert ports["Out"].dtype == "f32x4" and resources["Out"].access == "write"
    assert ports["State"].dtype == "block" and resources["State"].access == "read_write"
    assert [member["name"] for member in resources["State"].members] == ["count", "history"]
    assert resources["State"].members[1]["array"] == [8]
    assert resources["Feed"].realizes[0].shape is None
    assert resources["Feed"].realizes[0].extents_from == "fed"
    assert (resources["Params"].kind, resources["Params"].layout) == ("uniform-buffer", "std140")
    assert (resources["source_image"].kind, resources["source_image"].access) == (
        "storage-image", "read",
    )
    assert resources["lut"].kind == "sampled-texture" and resources["lut"].binding == 5
    assert resources["taps"].realizes[0].shape == (4,)
    assert resources["gain"].uniform == "gain" and resources["gain"].binding is None
    assert deployment.launch.local_size == (64, 2, 1)
    assert [item.code for item in deployment.shortfalls] == ["host-launch"]

    if GLSLANG is not None:
        _assert_matches_reflection(shader, _reflection(tmp_path, shader))


def test_published_extracted_product_carries_the_component_abi(tmp_path):
    product = publish_extracted_shader(tmp_path, _shader("compute", COMPUTE, "kernel"))
    manifest = json.loads((tmp_path / "kernel.shader.json").read_text(encoding="utf-8"))
    api = yaml.safe_load((tmp_path / manifest["api"]).read_text(encoding="utf-8"))
    record = api["metadata"]["shader_deployment"]
    abi = api["metadata"]["component_abi"]

    assert product.source_path.read_text(encoding="utf-8") == COMPUTE
    assert record["component_abi"] == {
        "schema": "turing.shader-component.v1",
        "component_id": "kernel",
        "record": "metadata.component_abi",
    }
    assert abi["component_id"] == "kernel"
    assert [port["slot"] for port in abi["ports"]] == list(range(8))
    assert [resource["realizes"][0]["slot"] for resource in record["resources"]] == list(range(8))
    assert record["resources"][4]["members"][0] == {
        "name": "count", "glsl_type": "uint", "dtype": "u32", "array": None,
    }


def test_bindings_left_to_the_host_are_named(tmp_path):
    source = """#version 330 core
in vec3 position;
uniform sampler2D albedo;
out vec4 color;
out vec4 normal;
void main() { color = texture(albedo, position.xy); normal = vec4(position, 1.0); }
"""
    _abi, deployment = extracted_shader_component(_shader("fragment", source))
    codes = {item.code: item.detail for item in deployment.shortfalls}

    assert set(codes) == {"graphics-stage", "link-assigned"}
    assert "albedo (sampled-texture, binding)" in codes["link-assigned"]
    assert "color (draw-buffer, location)" in codes["link-assigned"]
    assert "normal (draw-buffer, location)" in codes["link-assigned"]
    assert "position" not in codes["link-assigned"]


def test_varying_and_attribute_follow_the_stage():
    vertex = "#version 120\nattribute vec2 corner;\nvarying vec2 uv;\nvoid main() { uv = corner; gl_Position = vec4(corner, 0.0, 1.0); }\n"
    fragment = "#version 120\nvarying vec2 uv;\nvoid main() { gl_FragColor = vec4(uv, 0.0, 1.0); }\n"
    vertex_ports = _ports(extracted_shader_component(_shader("vertex", vertex))[0])
    fragment_ports = _ports(extracted_shader_component(_shader("fragment", fragment))[0])

    assert (vertex_ports["corner"].role, vertex_ports["corner"].transport) == (
        "feed", "vertex-attribute",
    )
    assert (vertex_ports["uv"].role, vertex_ports["uv"].transport) == ("output", "stage-output")
    assert (fragment_ports["uv"].role, fragment_ports["uv"].transport) == ("feed", "stage-input")


@pytest.mark.parametrize("source, reason", [
    ("#version 330\n#ifdef HDR\nuniform float exposure;\n#endif\nvoid main() {}\n", "#ifdef"),
    ("#version 330\n#define N 4\nuniform float w[N];\nvoid main() {}\n", "macro 'N'"),
    ("#version 330\nstruct Light { vec3 p; };\nuniform Light light;\nvoid main() {}\n", "struct type 'Light'"),
    ("#version 330\nuniform float w[2 * 2];\nvoid main() {}\n", "non-literal size"),
])
def test_an_interface_the_reader_cannot_be_sure_of_is_refused_by_name(source, reason):
    with pytest.raises(GLSLInterfaceRefusal, match=reason.replace("*", r"\*")):
        read_glsl_interface(source)
    abi, deployment = extracted_shader_component(_shader("fragment", source))

    assert abi is None and deployment.resources == ()
    ((code, detail),) = [(item.code, item.detail) for item in deployment.shortfalls]
    assert code == "no-component-abi" and reason in detail


def test_comments_and_function_bodies_are_not_interface():
    source = """#version 330
// uniform float commented_out;
/* in vec3 also_not; */
uniform float kept;   // trailing comment
vec3 shade(vec3 n) { float local_value = 1.0; return n * local_value; }
void main() { gl_FragColor = vec4(shade(vec3(kept)), 1.0); }
"""
    names = [item.name for item in read_glsl_interface(source).declarations]

    assert names == ["kept"]


DRIVEN = """#version 430
layout(local_size_x = 32) in;
layout(std430, binding = 3) readonly buffer Left { float left[]; };
layout(std430, binding = 5) readonly buffer Right { float right[]; };
layout(std430, binding = 1) writeonly buffer Result { float result[]; };
uniform float gain;
uniform uint count;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < count) { result[i] = gain * left[i] - right[i]; }
}
"""


def test_extracted_compute_shader_runs_on_gl_from_its_record_alone():
    import ctypes

    import numpy as np

    from src.common.tensors.accelerator_backends.glsl_backend import (
        GLContextUnavailable,
        compile_glsl_source,
        require_gl_context,
    )

    try:
        require_gl_context()
    except GLContextUnavailable as exc:
        pytest.skip(f"no OpenGL 4.3+ compute context: {exc}")
    from OpenGL import GL

    abi, deployment = extracted_shader_component(_shader("compute", DRIVEN, "driven"))
    assert [item.code for item in deployment.shortfalls] == ["host-launch"]
    count = 1000
    rng = np.random.default_rng(5)
    feeds = {
        "Left": rng.standard_normal(count).astype(np.float32),
        "Right": rng.standard_normal(count).astype(np.float32),
    }
    uniforms = {"gain": 1.75, "count": count}
    program = compile_glsl_source(DRIVEN)
    GL.glUseProgram(program)
    buffers = {}
    try:
        for resource in deployment.resources:
            if resource.kind == "storage-buffer":
                data = feeds.get(resource.name)
                if data is None:
                    data = np.full(count + 7, -1.0, dtype=np.float32)
                buffer = int(GL.glGenBuffers(1))
                GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, buffer)
                GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, data.nbytes, data, GL.GL_DYNAMIC_COPY)
                GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, resource.binding, buffer)
                buffers[resource.name] = (buffer, data.nbytes)
            elif resource.kind == "default-uniform":
                location = GL.glGetUniformLocation(program, resource.uniform)
                setter = {"f32": GL.glUniform1f, "u32": GL.glUniform1ui}[resource.dtype]
                setter(location, uniforms[resource.uniform])
        groups = -(-count // deployment.launch.local_size[0])
        GL.glDispatchCompute(groups + 1, 1, 1)
        GL.glMemoryBarrier(GL.GL_BUFFER_UPDATE_BARRIER_BIT)
        (output,) = [
            resource for resource in deployment.resources if resource.access == "write"
        ]
        buffer, nbytes = buffers[output.name]
        result = np.empty(nbytes // 4, dtype=np.float32)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, buffer)
        GL.glGetBufferSubData(
            GL.GL_SHADER_STORAGE_BUFFER, 0, nbytes, result.ctypes.data_as(ctypes.c_void_p),
        )
    finally:
        GL.glUseProgram(0)
        for buffer, _nbytes in buffers.values():
            GL.glDeleteBuffers(1, [buffer])

    expected = np.float32(1.75) * feeds["Left"] - feeds["Right"]
    # A driver may contract the multiply-subtract into one fma.
    np.testing.assert_allclose(result[:count], expected, rtol=1e-6, atol=1e-6)
    assert (result[count:] == -1.0).all()
