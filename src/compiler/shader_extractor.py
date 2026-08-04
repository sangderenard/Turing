"""Extract shader source from the Python calls that actually compile it.

This is intentionally a source front end, not a shader registry or a second
shader IR.  It follows literal source values into recognised compiler calls so
the enclosing Python function remains the program and shader deployment can be
separated before that function is lowered to a native host.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
from pathlib import Path


_STAGES = {
    "GL_VERTEX_SHADER": "vertex",
    "GL_FRAGMENT_SHADER": "fragment",
    "GL_COMPUTE_SHADER": "compute",
    "GL_GEOMETRY_SHADER": "geometry",
    "GL_TESS_CONTROL_SHADER": "tess-control",
    "GL_TESS_EVALUATION_SHADER": "tess-evaluation",
}


@dataclass(frozen=True)
class ExtractedShader:
    stage: str
    source: str
    compiler: str
    filename: str
    line: int
    source_name: str | None = None

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.source.encode("utf-8")).hexdigest()


def _call_name(call: ast.Call) -> str:
    function = call.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return ""


def _name(value: ast.AST) -> str | None:
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Attribute):
        return value.attr
    return None


def extract_shader_compile_calls(
    source_or_path: str | Path,
    *,
    filename: str | None = None,
) -> tuple[ExtractedShader, ...]:
    """Return literal shaders passed to recognised compilation calls.

    A path is read as Python source. A string is treated as Python source
    unless it names an existing file. The first tranche recognises PyOpenGL's
    ``compileShader(source, GL_*_SHADER)`` form used by the live FluxSpring
    renderer. Unknown or dynamic source expressions remain unextracted rather
    than being guessed.
    """

    candidate = Path(source_or_path) if isinstance(source_or_path, (str, Path)) else None
    if candidate is not None and candidate.is_file():
        resolved = candidate.resolve()
        text = resolved.read_text(encoding="utf-8")
        source_filename = str(resolved)
    else:
        text = str(source_or_path)
        source_filename = filename or "<shader-host>"
    tree = ast.parse(text, filename=source_filename)

    literals: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
        if not isinstance(value, str):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                literals[target.id] = value

    extracted: list[ExtractedShader] = []
    seen: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _call_name(node) != "compileShader":
            continue
        if len(node.args) < 2:
            continue
        source_name = _name(node.args[0])
        if source_name is not None:
            shader_source = literals.get(source_name)
        else:
            try:
                shader_source = ast.literal_eval(node.args[0])
            except (ValueError, TypeError):
                shader_source = None
        stage_name = _name(node.args[1])
        stage = _STAGES.get(str(stage_name))
        if not isinstance(shader_source, str) or stage is None:
            continue
        digest = hashlib.sha256(shader_source.encode("utf-8")).hexdigest()
        key = (stage, digest)
        if key in seen:
            continue
        seen.add(key)
        extracted.append(ExtractedShader(
            stage=stage,
            source=shader_source,
            compiler="compileShader",
            filename=source_filename,
            line=int(getattr(node, "lineno", 0)),
            source_name=source_name,
        ))
    return tuple(sorted(extracted, key=lambda item: (item.line, item.stage)))


__all__ = ["ExtractedShader", "extract_shader_compile_calls"]
