import numpy as np

from src.compiler.autogenesis import compile_source_autogenesis


def test_real_compiler_run_records_ingestion_precompile_ssa_and_package_handoffs():
    result = compile_source_autogenesis(
        """
def kernel(x, gain):
    scaled = x * gain
    shifted = scaled + 1.0
    return shifted.sin()
""",
        "kernel",
        {
            "x": np.ones(4),
            "gain": np.full(4, 2.0),
        },
    )

    snapshot = result.metagraph.snapshot()
    stages = {graph.stage for graph in snapshot.graphs}
    transformations = {
        event.detail.get("transformation")
        for event in snapshot.events
        if event.kind == "component-handoff"
    }
    assert result.ssa.complete
    assert {
        "process-graph",
        "precompile",
        "ssa",
        "ir-package",
        "backend-adapter:webgl",
        "backend:webgl",
    } <= stages
    assert "process-graph-to-precompile" in transformations
    assert "precompile-to-ssa" in transformations
    assert "ssa-to-package" in transformations
    assert "ssa-to-webgl-adapter" in transformations
    assert "webgl-adapter-to-glsl-es" in transformations
    webgl_handoffs = [
        event
        for event in snapshot.events
        if event.kind == "component-handoff"
        and event.detail.get("transformation")
        == "webgl-adapter-to-glsl-es"
    ]
    assert webgl_handoffs
    assert all(
        event.detail.get("granularity") == "exact-value"
        for event in webgl_handoffs
    )
    assert result.final_artifact.complete
