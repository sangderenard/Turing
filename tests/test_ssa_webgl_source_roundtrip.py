from src.compiler.glsl_source_ingestion import lower_glsl_source_to_ssa
from src.compiler.ssa_webgl_backend import emit_ssa_webgl_fragment_module
from src.compiler.evolution_metagraph import record_evolution
from src.rendering.precompiled_graph import EvolutionVisualProjector


def test_glsl_source_round_trips_through_ssa_into_webgl_emitter():
    lowered = lower_glsl_source_to_ssa(
        """
        uniform float left;
        uniform float right;
        out float color;
        void main() {
            float sum = left + right;
            color = sin(sum) * 0.5;
        }
        """
    )
    function = lowered.module.functions["main"]

    webgl = emit_ssa_webgl_fragment_module(function, name="from_source")

    assert lowered.complete, lowered.shortfall_report()
    assert webgl.complete, [item.format() for item in webgl.shortfalls]
    assert webgl.source.startswith("#version 300 es")
    assert "float v_2 = v_0 + v_1;" in webgl.source
    assert "float v_3 = sin(v_2);" in webgl.source
    assert "float v_5 = v_3 * v_4;" in webgl.source
    assert "turing_output_0 = vec4(v_5, 0.0, 0.0, 1.0);" in webgl.source


def test_ssa_to_webgl_does_not_reintroduce_unprocessed_source_calls():
    lowered = lower_glsl_source_to_ssa(
        """
        uniform float source;
        out float color;
        void main() { color = texture(source, 0.0); }
        """
    )
    webgl = emit_ssa_webgl_fragment_module(lowered.module.functions["main"])

    assert not lowered.complete
    assert not webgl.complete
    assert "texture(" not in webgl.source


def test_finalized_webgl_surface_is_indicated_without_an_invented_schedule():
    with record_evolution() as metagraph:
        lowered = lower_glsl_source_to_ssa(
            """
            uniform float left;
            uniform float right;
            out float color;
            void main() { color = left + right; }
            """
        )
        emit_ssa_webgl_fragment_module(
            lowered.module.functions["main"], name="indicated"
        )

    projector = EvolutionVisualProjector()
    for event in metagraph.snapshot().events:
        projector.apply(event, materialize=False)
    backend_nodes = [
        node for node in projector.graph().nodes
        if node.group == "backend:webgl"
    ]

    assert backend_nodes
    assert all(node.state == "finalized" for node in backend_nodes)
    assert all(node.schedule_group is None for node in backend_nodes)
