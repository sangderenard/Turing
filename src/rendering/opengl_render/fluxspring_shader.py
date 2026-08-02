"""Load FluxSpring's actual live-visualizer shader without importing its toy."""

from __future__ import annotations

import ast
from pathlib import Path


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
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: dict[str, str] = {}
    for statement in ast.walk(tree):
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        value = statement.value
        for target in targets:
            if isinstance(target, ast.Name) and target.id in {"vsrc", "fsrc"}:
                literal = ast.literal_eval(value)
                if not isinstance(literal, str):
                    raise TypeError(f"{target.id} in {path} is not literal shader text")
                values[target.id] = literal
    missing = {"vsrc", "fsrc"} - values.keys()
    if missing:
        raise RuntimeError(f"FluxSpring live shader is missing {sorted(missing)}")
    return values["vsrc"], values["fsrc"]


__all__ = ["load_fluxspring_graph_shaders"]
