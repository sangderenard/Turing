"""Load FluxSpring's actual live-visualizer shader without importing its toy."""

from __future__ import annotations

from pathlib import Path

from ...compiler.shader_extractor import extract_shader_compile_calls


def load_fluxspring_graph_shaders() -> tuple[str, str]:
    """Return the shader literals from FluxSpring's ``LiveVizGLPoints``.

    The renderer lives in ``spring_async_toy.py`` and pulls in the full tensor
    experiment when imported. AST extraction keeps its ``vsrc``/``fsrc``
    literals as the source of truth without importing that runtime.
    """

    path = (
        Path(__file__).parents[2]
        / "common"
        / "tensors"
        / "autoautograd"
        / "spring_async_toy.py"
    )
    shaders = extract_shader_compile_calls(path)
    by_stage = {shader.stage: shader.source for shader in shaders}
    missing = {"vertex", "fragment"} - by_stage.keys()
    if missing:
        raise RuntimeError(
            f"FluxSpring live shader compiler calls are missing {sorted(missing)}"
        )
    return by_stage["vertex"], by_stage["fragment"]


__all__ = ["load_fluxspring_graph_shaders"]
