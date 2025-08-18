from src.common.dt_system.engine_api import ComputeShaderSpec
from src.transmogrifier.glsl import MonolithicShader


def test_monolithic_shader_concatenates_sources_and_buffers():
    a_buf, b_buf = object(), object()
    specs = [
        ComputeShaderSpec(name="a", source="void main(){}", buffers={"in": a_buf}),
        ComputeShaderSpec(name="b", source="void main2(){}", buffers={"out": b_buf}),
    ]
    monolith = MonolithicShader.from_specs(specs)
    assert monolith.buffers == {"in": a_buf, "out": b_buf}
    assert "#version" in monolith.source
    assert "void main(){}" in monolith.source
    assert "void main2(){}" in monolith.source


def test_buffer_name_collision_raises():
    shared = object()
    specs = [
        ComputeShaderSpec(name="a", source="void main(){}", buffers={"buf": shared}),
        ComputeShaderSpec(name="b", source="void main2(){}", buffers={"buf": shared}),
    ]
    try:
        MonolithicShader.from_specs(specs)
    except ValueError as exc:
        assert "buffer 'buf'" in str(exc)
    else:  # pragma: no cover
        assert False, "expected collision to raise"
