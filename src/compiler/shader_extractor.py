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
import itertools
from pathlib import Path
from typing import Mapping


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

    @property
    def language(self) -> str:
        # Every recognized host API in this front end is an OpenGL API. Source
        # may target desktop or ES profiles, but both are GLSL and retain their
        # exact ``#version`` directive for a later profile-aware translator.
        return "glsl"

    def to_mapping(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "language": self.language,
            "source": self.source,
            "sha256": self.sha256,
            "compiler": self.compiler,
            "origin": {
                "filename": self.filename,
                "line": self.line,
                "source_name": self.source_name,
            },
        }


@dataclass(frozen=True)
class ExtractedShaderBundle:
    """Deterministic source package produced from real host compile sites."""

    root: str
    shaders: tuple[ExtractedShader, ...]
    schema: str = "turing.extracted-shaders.v1"

    def select(
        self,
        *,
        stage: str | None = None,
        language: str | None = None,
    ) -> tuple[ExtractedShader, ...]:
        return tuple(
            shader
            for shader in self.shaders
            if (stage is None or shader.stage == stage)
            and (language is None or shader.language == language)
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "root": self.root,
            "shaders": [shader.to_mapping() for shader in self.shaders],
        }


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


@dataclass(frozen=True)
class _Unknown:
    pass


@dataclass(frozen=True)
class _Parameter:
    name: str


@dataclass(frozen=True)
class _StringValue:
    value: str
    name: str | None = None


@dataclass(frozen=True)
class _StageValue:
    stage: str


@dataclass(frozen=True)
class _TupleValue:
    items: tuple[object, ...]


@dataclass(frozen=True)
class _Alternatives:
    values: tuple[object, ...]


@dataclass
class _ShaderHandle:
    stage: object
    source: object = _Unknown()


@dataclass(frozen=True)
class _CompileTemplate:
    source: object
    stage: object
    compiler: str
    line: int


@dataclass(frozen=True)
class _FunctionSummary:
    parameters: tuple[str, ...]
    compiles: tuple[_CompileTemplate, ...]


def _alternatives(*values: object) -> object:
    flattened = []
    for value in values:
        if isinstance(value, _Alternatives):
            flattened.extend(value.values)
        else:
            flattened.append(value)
    unique = tuple(dict.fromkeys(flattened))
    if not unique:
        return _Unknown()
    return unique[0] if len(unique) == 1 else _Alternatives(unique)


def _substitute(value: object, bindings: Mapping[str, object]) -> object:
    if isinstance(value, _Parameter):
        return bindings.get(value.name, value)
    if isinstance(value, _TupleValue):
        return _TupleValue(tuple(_substitute(item, bindings) for item in value.items))
    if isinstance(value, _Alternatives):
        return _alternatives(*(_substitute(item, bindings) for item in value.values))
    return value


def _strings(value: object) -> tuple[_StringValue, ...]:
    if isinstance(value, _StringValue):
        return (value,)
    if isinstance(value, _Alternatives):
        return tuple(
            item
            for alternative in value.values
            for item in _strings(alternative)
        )
    return ()


def _stages(value: object) -> tuple[str, ...]:
    if isinstance(value, _StageValue):
        return (value.stage,)
    if isinstance(value, _Alternatives):
        return tuple(
            stage
            for alternative in value.values
            for stage in _stages(alternative)
        )
    return ()


def _positional_or_keyword(
    call: ast.Call,
    position: int,
    *names: str,
) -> ast.AST | None:
    if len(call.args) > position:
        return call.args[position]
    wanted = set(names)
    return next(
        (keyword.value for keyword in call.keywords if keyword.arg in wanted),
        None,
    )


class _ShaderFlow:
    """Summarize only dataflow into recognized shader compiler APIs."""

    def __init__(
        self,
        module_values: Mapping[str, object],
        summaries: Mapping[str, _FunctionSummary],
        *,
        current_function: str | None = None,
    ) -> None:
        self.module_values = dict(module_values)
        self.summaries = summaries
        self.current_function = current_function
        self.compiles: list[_CompileTemplate] = []

    def expression(self, node: ast.AST | None, values: dict[str, object]) -> object:
        if node is None:
            return _Unknown()
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return _StringValue(node.value)
        if isinstance(node, ast.Name):
            stage = _STAGES.get(node.id)
            if stage is not None:
                return _StageValue(stage)
            if node.id in values:
                return values[node.id]
            if node.id in self.module_values:
                return self.module_values[node.id]
            return _Parameter(node.id)
        if isinstance(node, ast.Attribute):
            stage = _STAGES.get(node.attr)
            if stage is not None:
                return _StageValue(stage)
            return values.get(
                node.attr,
                self.module_values.get(node.attr, _Parameter(node.attr)),
            )
        if isinstance(node, (ast.Tuple, ast.List)):
            return _TupleValue(tuple(self.expression(item, values) for item in node.elts))
        if isinstance(node, ast.BoolOp):
            return _alternatives(*(self.expression(item, values) for item in node.values))
        if isinstance(node, ast.IfExp):
            return _alternatives(
                self.expression(node.body, values),
                self.expression(node.orelse, values),
            )
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = _strings(self.expression(node.left, values))
            right = _strings(self.expression(node.right, values))
            if left and right:
                return _alternatives(*(
                    _StringValue(a.value + b.value)
                    for a, b in itertools.product(left, right)
                ))
            return _Unknown()
        if isinstance(node, ast.Subscript):
            owner = self.expression(node.value, values)
            try:
                index = ast.literal_eval(node.slice)
            except (ValueError, TypeError):
                return _Unknown()
            candidates = owner.values if isinstance(owner, _Alternatives) else (owner,)
            selected = []
            for candidate in candidates:
                if (
                    isinstance(candidate, _TupleValue)
                    and isinstance(index, int)
                    and -len(candidate.items) <= index < len(candidate.items)
                ):
                    selected.append(candidate.items[index])
            return _alternatives(*selected)
        if isinstance(node, ast.Call):
            return self.call(node, values)
        return _Unknown()

    def call(self, call: ast.Call, values: dict[str, object]) -> object:
        name = _call_name(call)
        if name == "compileShader":
            source = _positional_or_keyword(call, 0, "source", "shaderSource")
            stage = _positional_or_keyword(call, 1, "shaderType", "shader_type", "stage")
            self.compiles.append(_CompileTemplate(
                self.expression(source, values),
                self.expression(stage, values),
                "compileShader",
                int(getattr(call, "lineno", 0)),
            ))
            return _Unknown()
        if name == "glCreateShader":
            stage = _positional_or_keyword(call, 0, "shaderType", "shader_type", "stage")
            return _ShaderHandle(self.expression(stage, values))
        if name == "glShaderSource":
            handle_node = _positional_or_keyword(call, 0, "shader")
            source_node = _positional_or_keyword(call, 1, "string", "source")
            handle = self.expression(handle_node, values)
            if isinstance(handle, _ShaderHandle):
                handle.source = self.expression(source_node, values)
            return _Unknown()
        if name == "glCompileShader":
            handle_node = _positional_or_keyword(call, 0, "shader")
            handle = self.expression(handle_node, values)
            if isinstance(handle, _ShaderHandle):
                self.compiles.append(_CompileTemplate(
                    handle.source,
                    handle.stage,
                    "glShaderSource/glCompileShader",
                    int(getattr(call, "lineno", 0)),
                ))
            return _Unknown()

        summary = self.summaries.get(name)
        if summary is None or name == self.current_function:
            # A recognized shader call is often nested in an unrelated host
            # call, for example ``compileProgram(compileShader(...), ...)``.
            # Traverse argument expressions for their compile-site effects
            # without assigning semantics to the enclosing API.
            for argument in call.args:
                self.expression(argument, values)
            for keyword in call.keywords:
                self.expression(keyword.value, values)
            return _Unknown()
        bindings: dict[str, object] = {}
        for index, parameter in enumerate(summary.parameters):
            argument = _positional_or_keyword(call, index, parameter)
            if argument is not None:
                bindings[parameter] = self.expression(argument, values)
        self.compiles.extend(
            _CompileTemplate(
                _substitute(item.source, bindings),
                _substitute(item.stage, bindings),
                item.compiler,
                item.line,
            )
            for item in summary.compiles
        )
        return _Unknown()

    def bind(self, target: ast.AST, value: object, values: dict[str, object]) -> None:
        if isinstance(target, ast.Name):
            if isinstance(value, _StringValue) and value.name is None:
                value = _StringValue(value.value, target.id)
            values[target.id] = value
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            alternatives = value.values if isinstance(value, _Alternatives) else (value,)
            for index, child in enumerate(target.elts):
                projected = []
                for alternative in alternatives:
                    if isinstance(alternative, _TupleValue) and index < len(alternative.items):
                        projected.append(alternative.items[index])
                    else:
                        projected.append(_Unknown())
                self.bind(child, _alternatives(*projected), values)

    def statements(self, statements: list[ast.stmt], values: dict[str, object]) -> None:
        for statement in statements:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(statement, ast.Assign):
                value = self.expression(statement.value, values)
                for target in statement.targets:
                    self.bind(target, value, values)
            elif isinstance(statement, ast.AnnAssign):
                self.bind(statement.target, self.expression(statement.value, values), values)
            elif isinstance(statement, ast.Expr):
                self.expression(statement.value, values)
            elif isinstance(statement, ast.Return):
                self.expression(statement.value, values)
            elif isinstance(statement, ast.If):
                self.statements(statement.body, dict(values))
                self.statements(statement.orelse, dict(values))
            elif isinstance(statement, (ast.For, ast.While, ast.With, ast.Try)):
                for field in ("body", "orelse", "finalbody"):
                    nested = getattr(statement, field, ())
                    if nested:
                        self.statements(list(nested), dict(values))
                for handler in getattr(statement, "handlers", ()):
                    self.statements(list(handler.body), dict(values))


def _module_values(tree: ast.Module) -> dict[str, object]:
    flow = _ShaderFlow({}, {})
    values: dict[str, object] = {}
    flow.statements(tree.body, values)
    return values


def _function_summaries(
    tree: ast.Module,
    module_values: Mapping[str, object],
) -> tuple[dict[str, _FunctionSummary], tuple[_CompileTemplate, ...]]:
    definitions = tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    name_counts = {
        name: sum(definition.name == name for definition in definitions)
        for name in {definition.name for definition in definitions}
    }
    summary_definitions = tuple(
        definition
        for definition in definitions
        if name_counts[definition.name] == 1
    )
    summaries: dict[str, _FunctionSummary] = {}
    for _round in range(max(1, len(summary_definitions) + 1)):
        changed = False
        for definition in summary_definitions:
            name = definition.name
            parameters = tuple(
                argument.arg
                for argument in (
                    *definition.args.posonlyargs,
                    *definition.args.args,
                    *definition.args.kwonlyargs,
                )
                if argument.arg not in {"self", "cls"}
            )
            values = {
                parameter: _Parameter(parameter)
                for parameter in parameters
            }
            flow = _ShaderFlow(
                module_values,
                summaries,
                current_function=name,
            )
            flow.statements(definition.body, values)
            compiles = tuple(dict.fromkeys(flow.compiles))
            summary = _FunctionSummary(parameters, compiles)
            previous = summaries.get(name)
            if previous != summary:
                summaries[name] = summary
                changed = True
        if not changed:
            break
    # Function names are only a call-resolution convenience. Large Python
    # modules legitimately contain many methods named ``__init__`` (and often
    # several ``compile`` helpers), so direct sites must be collected from
    # every definition rather than from the name-indexed summary table.
    all_compiles = []
    for definition in definitions:
        parameters = tuple(
            argument.arg
            for argument in (
                *definition.args.posonlyargs,
                *definition.args.args,
                *definition.args.kwonlyargs,
            )
            if argument.arg not in {"self", "cls"}
        )
        values = {parameter: _Parameter(parameter) for parameter in parameters}
        flow = _ShaderFlow(
            module_values,
            summaries,
            current_function=definition.name,
        )
        flow.statements(definition.body, values)
        all_compiles.extend(flow.compiles)
    return summaries, tuple(dict.fromkeys(all_compiles))


def extract_shader_compile_calls(
    source_or_path: str | Path,
    *,
    filename: str | None = None,
) -> tuple[ExtractedShader, ...]:
    """Return literal shaders passed to recognised compilation calls.

    A path is read as Python source. A string is treated as Python source
    unless it names an existing file. Recognised forms include PyOpenGL's
    ``compileShader(source, GL_*_SHADER)`` helper and the raw
    ``glCreateShader`` / ``glShaderSource`` / ``glCompileShader`` sequence.
    Ordinary helper functions are summarized by parameter flow, so shader
    compilation can remain factored exactly as it is in the application.
    Unknown or dynamic source expressions remain unextracted rather than being
    guessed.
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

    module_values = _module_values(tree)
    _summaries, compile_templates = _function_summaries(tree, module_values)
    module_flow = _ShaderFlow(module_values, _summaries)
    module_flow.statements(tree.body, dict(module_values))
    compile_templates = (*compile_templates, *module_flow.compiles)
    extracted: list[ExtractedShader] = []
    seen: set[tuple[str, str]] = set()
    for template in compile_templates:
        for source, stage in itertools.product(
            _strings(template.source),
            _stages(template.stage),
        ):
            digest = hashlib.sha256(source.value.encode("utf-8")).hexdigest()
            key = (stage, digest)
            if key in seen:
                continue
            seen.add(key)
            extracted.append(ExtractedShader(
                stage=stage,
                source=source.value,
                compiler=template.compiler,
                filename=source_filename,
                line=template.line,
                source_name=source.name,
            ))
    return tuple(sorted(extracted, key=lambda item: (item.line, item.stage)))


def discover_shader_compile_calls(
    root: str | Path,
    *,
    strict: bool = True,
) -> ExtractedShaderBundle:
    """Extract every provable shader compile site below a Python source root.

    Discovery preserves duplicate source text when it originates at different
    host sites; those origins are relevant when the whole-program compiler
    replaces or packages the host calls. ``strict=False`` is intended only for
    heterogeneous source archives where unrelated, unparsable Python files
    should not prevent discovery of valid modules.
    """

    root_path = Path(root).resolve()
    paths = (
        (root_path,)
        if root_path.is_file()
        else tuple(sorted(root_path.rglob("*.py")))
    )
    shaders: list[ExtractedShader] = []
    for path in paths:
        try:
            shaders.extend(extract_shader_compile_calls(path))
        except (OSError, SyntaxError, UnicodeError):
            if strict:
                raise
    return ExtractedShaderBundle(
        str(root_path),
        tuple(sorted(
            shaders,
            key=lambda item: (
                item.filename,
                item.line,
                item.stage,
                item.sha256,
            ),
        )),
    )


__all__ = [
    "ExtractedShader",
    "ExtractedShaderBundle",
    "discover_shader_compile_calls",
    "extract_shader_compile_calls",
]
