"""Whole-shader GLSL ingestion, deployment planning, and WebGL lowering.

The numerical GLSL reader in :mod:`glsl_source_ingestion` deliberately lowers
the scalar expression subset straight to repository SSA.  A real graphics
shader has another, larger surface: stage inputs, uniforms, resource blocks,
helper functions, control flow, texture operations, and fragment builtins.
Those constructs must not disappear into an unprocessed source string merely
because numerical SSA cannot represent them yet.

This module is the project-IR route for that surface.  Ingestion creates one
typed instruction for every authored line and records resource/function
structure before any target is selected.  The deployment planner consumes that
IR and decides whether a function is fragment-bound or compute-eligible.  The
WebGL backend then consumes the same IR, specializing desktop ``std430``
read-only scalar buffers into texture-backed reads because WebGL 2 has no
SSBOs.  No source-to-source entry point bypasses the project IR.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import re
from typing import Any, Iterable, Mapping


PROJECT_SHADER_SCHEMA = "turing.glsl-project-ir.v1"
WEBGL_TRANSLATION_SCHEMA = "turing.webgl-project-shader.v1"


@dataclass(frozen=True)
class GLSLProjectDiagnostic:
    code: str
    line: int
    message: str

    def format(self) -> str:
        return f"{self.code} at {self.line}: {self.message}"

    def to_mapping(self) -> dict[str, Any]:
        return {"code": self.code, "line": self.line, "message": self.message}


@dataclass(frozen=True)
class GLSLStorageBuffer:
    block_name: str
    member_name: str
    scalar_type: str
    binding: int
    readonly: bool
    start_line: int
    end_line: int

    @property
    def sampler_name(self) -> str:
        return f"turing_ssbo_{self.member_name}"

    @property
    def loader_name(self) -> str:
        return f"turing_ssbo_load_{self.member_name}"

    @property
    def sampler_type(self) -> str:
        return {
            "float": "sampler2D",
            "int": "isampler2D",
            "uint": "usampler2D",
        }[self.scalar_type]

    def to_mapping(self) -> dict[str, Any]:
        texture_format = {
            "float": ("R32F", "RED", "FLOAT"),
            "int": ("R32I", "RED_INTEGER", "INT"),
            "uint": ("R32UI", "RED_INTEGER", "UNSIGNED_INT"),
        }[self.scalar_type]
        return {
            "block_name": self.block_name,
            "member_name": self.member_name,
            "scalar_type": self.scalar_type,
            "source_binding": self.binding,
            "recommended_texture_unit": self.binding,
            "sampler": self.sampler_name,
            "loader": self.loader_name,
            "readonly": self.readonly,
            "source_lines": [self.start_line, self.end_line],
            "transport": "webgl2-texture-r32",
            "texture": {
                "internal_format": texture_format[0],
                "format": texture_format[1],
                "type": texture_format[2],
                "min_filter": "NEAREST",
                "mag_filter": "NEAREST",
                "packing": "row-major scalar; x=index%width, y=index/width",
            },
        }


@dataclass(frozen=True)
class GLSLProjectInstruction:
    """One source line after lexical interpretation, before target lowering."""

    line: int
    kind: str
    text: str
    function: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "line": self.line,
            "kind": self.kind,
            "text": self.text,
            "function": self.function,
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True)
class GLSLFunctionRegion:
    name: str
    return_type: str
    start_line: int
    end_line: int
    calls: tuple[str, ...]
    identifiers: tuple[str, ...]
    direct_graphics_reasons: tuple[str, ...]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "return_type": self.return_type,
            "source_lines": [self.start_line, self.end_line],
            "calls": list(self.calls),
            "direct_graphics_reasons": list(self.direct_graphics_reasons),
        }


@dataclass(frozen=True)
class GLSLProjectIR:
    """Target-neutral, line-addressable IR for one complete GLSL stage."""

    source_name: str
    stage_hint: str
    source_sha256: str
    instructions: tuple[GLSLProjectInstruction, ...]
    storage_buffers: tuple[GLSLStorageBuffer, ...]
    functions: tuple[GLSLFunctionRegion, ...]
    stage_inputs: tuple[str, ...]
    stage_outputs: tuple[str, ...]
    uniforms: tuple[str, ...]
    diagnostics: tuple[GLSLProjectDiagnostic, ...] = ()
    schema: str = PROJECT_SHADER_SCHEMA

    @property
    def complete(self) -> bool:
        return not self.diagnostics

    def to_mapping(self, *, include_lines: bool = False) -> dict[str, Any]:
        result = {
            "schema": self.schema,
            "source_name": self.source_name,
            "stage_hint": self.stage_hint,
            "source_sha256": self.source_sha256,
            "storage_buffers": [item.to_mapping() for item in self.storage_buffers],
            "functions": [item.to_mapping() for item in self.functions],
            "stage_inputs": list(self.stage_inputs),
            "stage_outputs": list(self.stage_outputs),
            "uniforms": list(self.uniforms),
            "diagnostics": [item.to_mapping() for item in self.diagnostics],
            "line_count": len(self.instructions),
        }
        if include_lines:
            result["instructions"] = [
                item.to_mapping() for item in self.instructions
            ]
        return result


@dataclass(frozen=True)
class GLSLExecutionSection:
    function: str
    stage: str
    source_lines: tuple[int, int]
    reasons: tuple[str, ...]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "function": self.function,
            "stage": self.stage,
            "source_lines": list(self.source_lines),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class GLSLProjectExecutionPlan:
    sections: tuple[GLSLExecutionSection, ...]
    selected_graphics_backend: str
    compute_sections: tuple[str, ...]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "selected_graphics_backend": self.selected_graphics_backend,
            "compute_sections": list(self.compute_sections),
            "sections": [item.to_mapping() for item in self.sections],
        }


@dataclass(frozen=True)
class WebGLProjectShader:
    name: str
    source: str
    project_ir: GLSLProjectIR
    execution_plan: GLSLProjectExecutionPlan
    diagnostics: tuple[GLSLProjectDiagnostic, ...]

    @property
    def complete(self) -> bool:
        return not self.diagnostics

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": WEBGL_TRANSLATION_SCHEMA,
            "name": self.name,
            "language": "webgl2-glsl-es",
            "stage": self.project_ir.stage_hint,
            "complete": self.complete,
            "project_ir": self.project_ir.to_mapping(),
            "execution_plan": self.execution_plan.to_mapping(),
            "bindings": [
                item.to_mapping() for item in self.project_ir.storage_buffers
            ],
            "diagnostics": [item.to_mapping() for item in self.diagnostics],
        }


_BUFFER_BLOCK = re.compile(
    r"layout\s*\((?P<layout>[^)]*)\)\s*"
    r"(?P<access>(?:(?:readonly|writeonly|coherent|volatile|restrict)\s+)*)"
    r"buffer\s+(?P<block>[A-Za-z_]\w*)\s*\{\s*"
    r"(?P<type>float|int|uint)\s+(?P<member>[A-Za-z_]\w*)\s*"
    r"\[\s*\]\s*;\s*\}\s*;",
    re.DOTALL,
)
_FUNCTION_START = re.compile(
    r"^\s*(?P<return>[A-Za-z_]\w*)\s+(?P<name>[A-Za-z_]\w*)\s*"
    r"\([^;]*\)\s*\{"
)
_INTERFACE = re.compile(
    r"^\s*(?:layout\s*\([^)]*\)\s*)?"
    r"(?:(?:flat|smooth|noperspective|centroid|highp|mediump|lowp)\s+)*"
    r"(?P<storage>uniform|in|out)\s+"
    r"(?:(?:highp|mediump|lowp)\s+)?[A-Za-z_]\w*\s+"
    r"(?P<name>[A-Za-z_]\w*)"
)
_CALL = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
_IDENTIFIER = re.compile(r"\b[A-Za-z_]\w*\b")
_CONTROL_WORDS = frozenset({"if", "for", "while", "switch", "return"})
_CONSTRUCTORS = frozenset(
    {"bool", "float", "int", "uint"}
    | {f"{prefix}{kind}{width}" for prefix in ("", "i", "u", "b")
       for kind in ("vec",) for width in range(2, 5)}
    | {f"mat{width}" for width in range(2, 5)}
)


def _line_at(source: str, position: int) -> int:
    return source.count("\n", 0, position) + 1


def _code_for_analysis(line: str, in_block_comment: bool) -> tuple[str, bool]:
    """Remove comments while retaining brace/token structure for one line."""

    result: list[str] = []
    index = 0
    while index < len(line):
        if in_block_comment:
            end = line.find("*/", index)
            if end < 0:
                return "".join(result), True
            index = end + 2
            in_block_comment = False
            continue
        block = line.find("/*", index)
        single = line.find("//", index)
        if single >= 0 and (block < 0 or single < block):
            result.append(line[index:single])
            break
        if block < 0:
            result.append(line[index:])
            break
        result.append(line[index:block])
        index = block + 2
        in_block_comment = True
    return "".join(result), in_block_comment


def _storage_blocks(
    source: str,
) -> tuple[tuple[GLSLStorageBuffer, ...], tuple[GLSLProjectDiagnostic, ...]]:
    buffers: list[GLSLStorageBuffer] = []
    diagnostics: list[GLSLProjectDiagnostic] = []
    for match in _BUFFER_BLOCK.finditer(source):
        layout = match.group("layout")
        binding_match = re.search(r"\bbinding\s*=\s*(\d+)", layout)
        start_line = _line_at(source, match.start())
        if "std430" not in layout or binding_match is None:
            diagnostics.append(GLSLProjectDiagnostic(
                "GLSL_STORAGE_LAYOUT",
                start_line,
                "storage buffer requires std430 and a literal binding",
            ))
            continue
        access = set(match.group("access").split())
        buffers.append(GLSLStorageBuffer(
            block_name=match.group("block"),
            member_name=match.group("member"),
            scalar_type=match.group("type"),
            binding=int(binding_match.group(1)),
            readonly="readonly" in access,
            start_line=start_line,
            end_line=_line_at(source, match.end()),
        ))
        if "readonly" not in access:
            diagnostics.append(GLSLProjectDiagnostic(
                "WEBGL_STORAGE_WRITE",
                start_line,
                f"buffer {match.group('block')} is writable; WebGL texture transport is read-only",
            ))
    # Any std430 buffer not matched above has a shape this first project-IR
    # storage specialization cannot honestly represent.
    matched_starts = {item.start_line for item in buffers}
    for candidate in re.finditer(r"layout\s*\([^)]*std430[^)]*\).*?\bbuffer\b", source, re.DOTALL):
        line = _line_at(source, candidate.start())
        if line not in matched_starts:
            diagnostics.append(GLSLProjectDiagnostic(
                "GLSL_STORAGE_SHAPE",
                line,
                "only one unsized float/int/uint array per storage block is supported",
            ))
    return tuple(buffers), tuple(diagnostics)


def ingest_glsl_project(
    source: str,
    *,
    source_name: str = "shader",
    stage_hint: str = "fragment",
) -> GLSLProjectIR:
    """Interpret every GLSL line into target-neutral project shader IR."""

    source = str(source).replace("\r\n", "\n").replace("\r", "\n")
    lines = source.split("\n")
    buffers, storage_diagnostics = _storage_blocks(source)
    buffer_by_line: dict[int, GLSLStorageBuffer] = {}
    for buffer in buffers:
        for line_number in range(buffer.start_line, buffer.end_line + 1):
            buffer_by_line[line_number] = buffer

    stage_inputs: list[str] = []
    stage_outputs: list[str] = []
    uniforms: list[str] = []
    instructions: list[GLSLProjectInstruction] = []
    functions: list[GLSLFunctionRegion] = []
    function_name: str | None = None
    function_return = "void"
    function_start = 0
    function_braces = 0
    function_calls: set[str] = set()
    function_identifiers: set[str] = set()
    function_graphics_reasons: set[str] = set()
    in_block_comment = False

    for line_number, text in enumerate(lines, 1):
        code, in_block_comment = _code_for_analysis(text, in_block_comment)
        stripped = code.strip()
        buffer = buffer_by_line.get(line_number)
        if buffer is not None:
            kind = (
                "storage-buffer"
                if line_number == buffer.start_line
                else "storage-buffer-continuation"
            )
            attributes: Mapping[str, Any] = buffer.to_mapping()
        elif stripped.startswith("#"):
            kind, attributes = "directive", {}
        elif not stripped:
            kind = "blank" if not text.strip() else "comment"
            attributes = {}
        else:
            interface = _INTERFACE.match(code)
            if interface is not None and function_name is None:
                storage, name = interface.group("storage"), interface.group("name")
                {"uniform": uniforms, "in": stage_inputs, "out": stage_outputs}[storage].append(name)
                kind, attributes = f"stage-{storage}", {"name": name}
            else:
                identifiers = tuple(_IDENTIFIER.findall(code))
                calls = tuple(
                    name for name in _CALL.findall(code)
                    if name not in _CONTROL_WORDS and name not in _CONSTRUCTORS
                )
                controls = tuple(
                    word
                    for word in (
                        "if", "else", "for", "while", "switch",
                        "discard", "return",
                    )
                    if re.search(rf"\b{word}\b", code)
                )
                kind, attributes = "source", {
                    "identifiers": identifiers,
                    "calls": calls,
                    "controls": controls,
                    "brace_delta": code.count("{") - code.count("}"),
                }

        start = _FUNCTION_START.match(code) if function_name is None else None
        if start is not None:
            function_name = start.group("name")
            function_return = start.group("return")
            function_start = line_number
            function_calls = set()
            function_identifiers = set()
            function_graphics_reasons = set()
            kind = "function-start"

        owner = function_name
        if owner is not None:
            identifiers = set(_IDENTIFIER.findall(code))
            calls = {
                name for name in _CALL.findall(code)
                if name not in _CONTROL_WORDS and name not in _CONSTRUCTORS
            }
            calls.discard(owner)
            function_calls.update(calls)
            function_identifiers.update(identifiers)
            if "discard" in identifiers:
                function_graphics_reasons.add("uses fragment discard")
            builtins = sorted(name for name in identifiers if name.startswith("gl_"))
            if builtins:
                function_graphics_reasons.add(
                    "uses stage builtin(s) " + ", ".join(builtins)
                )
            read_inputs = sorted(set(stage_inputs) & identifiers)
            if read_inputs:
                function_graphics_reasons.add(
                    "reads interpolated stage input(s) " + ", ".join(read_inputs)
                )
            written_outputs = sorted(set(stage_outputs) & identifiers)
            if written_outputs:
                function_graphics_reasons.add(
                    "accesses graphics output(s) " + ", ".join(written_outputs)
                )

        instructions.append(GLSLProjectInstruction(
            line=line_number,
            kind=kind,
            text=text,
            function=owner,
            attributes=attributes,
        ))

        if owner is not None:
            function_braces += code.count("{") - code.count("}")
            if function_braces == 0:
                functions.append(GLSLFunctionRegion(
                    name=owner,
                    return_type=function_return,
                    start_line=function_start,
                    end_line=line_number,
                    calls=tuple(sorted(function_calls)),
                    identifiers=tuple(sorted(function_identifiers)),
                    direct_graphics_reasons=tuple(sorted(function_graphics_reasons)),
                ))
                function_name = None

    diagnostics = list(storage_diagnostics)
    if function_name is not None:
        diagnostics.append(GLSLProjectDiagnostic(
            "GLSL_FUNCTION_BALANCE",
            function_start,
            f"function {function_name!r} has unbalanced braces",
        ))
    if not any(item.name == "main" for item in functions):
        diagnostics.append(GLSLProjectDiagnostic(
            "GLSL_ENTRY", 1, "shader has no interpreted main function"
        ))
    return GLSLProjectIR(
        source_name=str(source_name),
        stage_hint=str(stage_hint),
        source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        instructions=tuple(instructions),
        storage_buffers=buffers,
        functions=tuple(functions),
        stage_inputs=tuple(dict.fromkeys(stage_inputs)),
        stage_outputs=tuple(dict.fromkeys(stage_outputs)),
        uniforms=tuple(dict.fromkeys(uniforms)),
        diagnostics=tuple(diagnostics),
    )


def plan_glsl_project_execution(project: GLSLProjectIR) -> GLSLProjectExecutionPlan:
    """Classify shader functions after ingestion, conservatively and visibly."""

    functions = {item.name: item for item in project.functions}
    callers: dict[str, set[str]] = {name: set() for name in functions}
    for function in project.functions:
        for callee in function.calls:
            if callee in callers:
                callers[callee].add(function.name)

    graphics_stage = project.stage_hint if project.stage_hint in {"vertex", "fragment"} else None
    graphics_bound = {
        function.name for function in project.functions
        if (
            (graphics_stage is not None and function.name == "main")
            or function.direct_graphics_reasons
        )
    }
    changed = True
    while changed:
        changed = False
        for function in project.functions:
            if function.name in graphics_bound:
                continue
            if callers[function.name] & graphics_bound:
                graphics_bound.add(function.name)
                changed = True

    sections: list[GLSLExecutionSection] = []
    compute_sections: list[str] = []
    for function in project.functions:
        reasons = list(function.direct_graphics_reasons)
        if function.name == "main" and graphics_stage is not None:
            reasons.append(f"{graphics_stage} graphics entry point")
        elif function.name in graphics_bound and not reasons:
            reasons.append(f"called from a {graphics_stage or 'graphics'}-bound function")
        if function.name in graphics_bound:
            stage = graphics_stage or "fragment"
        elif callers[function.name]:
            # This is deliberately only eligibility. Extracting a compute pass
            # also requires a cross-stage value transport and reuse proof.
            stage = "compute-candidate"
            reasons.append("has no per-fragment dependency in project IR")
            compute_sections.append(function.name)
        else:
            stage = "inactive"
            reasons.append("unreachable helper; no dispatch is warranted")
        sections.append(GLSLExecutionSection(
            function=function.name,
            stage=stage,
            source_lines=(function.start_line, function.end_line),
            reasons=tuple(dict.fromkeys(reasons)),
        ))
    return GLSLProjectExecutionPlan(
        sections=tuple(sections),
        selected_graphics_backend=f"webgl2-{graphics_stage or 'fragment'}",
        compute_sections=tuple(compute_sections),
    )


def _replace_buffer_reads(text: str, buffers: Iterable[GLSLStorageBuffer]) -> str:
    """Replace named SSBO indexing using balanced-bracket source scanning."""

    for buffer in buffers:
        pattern = re.compile(rf"\b{re.escape(buffer.member_name)}\s*\[")
        search_from = 0
        while True:
            match = pattern.search(text, search_from)
            if match is None:
                break
            bracket = text.find("[", match.start())
            depth = 1
            cursor = bracket + 1
            while cursor < len(text) and depth:
                if text[cursor] == "[":
                    depth += 1
                elif text[cursor] == "]":
                    depth -= 1
                cursor += 1
            if depth:
                break
            expression = text[bracket + 1:cursor - 1]
            replacement = f"{buffer.loader_name}(int({expression}))"
            text = text[:match.start()] + replacement + text[cursor:]
            search_from = match.start() + len(replacement)
    return text


_UNIFORM_INITIALIZER = re.compile(
    r"^(?P<prefix>\s*uniform\s+(?:(?:highp|mediump|lowp)\s+)?"
    r"[A-Za-z_]\w*\s+[A-Za-z_]\w*(?:\s*\[[^]]+\])?)"
    r"\s*=\s*[^;]+;(?P<comment>\s*(?://.*)?)$"
)


def _webgl_storage_preamble(buffers: Iterable[GLSLStorageBuffer]) -> list[str]:
    lines = [
        "precision highp float;",
        "precision highp int;",
        "precision highp sampler2D;",
        "precision highp sampler2DArray;",
        "precision highp sampler3D;",
        "precision highp isampler2D;",
        "precision highp usampler2D;",
        "",
        "// Turing WebGL storage transport generated from project IR.",
    ]
    for buffer in buffers:
        lines.extend((
            f"uniform highp {buffer.sampler_type} {buffer.sampler_name};",
            f"{buffer.scalar_type} {buffer.loader_name}(int index) {{",
            f"    ivec2 extent = textureSize({buffer.sampler_name}, 0);",
            "    int safe_index = max(index, 0);",
            "    ivec2 coordinate = ivec2(safe_index % extent.x, safe_index / extent.x);",
            f"    return texelFetch({buffer.sampler_name}, coordinate, 0).r;",
            "}",
            "",
        ))
    return lines


def emit_webgl_from_glsl_project(
    project: GLSLProjectIR,
    *,
    name: str | None = None,
) -> WebGLProjectShader:
    """Lower already-ingested project shader IR to WebGL 2 fragment GLSL."""

    plan = plan_glsl_project_execution(project)
    diagnostics = list(project.diagnostics)
    if project.stage_hint not in {"vertex", "fragment"}:
        diagnostics.append(GLSLProjectDiagnostic(
            "WEBGL_STAGE", 1,
            "WebGL graphics output requires a vertex or fragment project, "
            f"got {project.stage_hint!r}",
        ))
    emitted: list[str] = []
    inserted_preamble = False
    continuation_lines: set[int] = set()
    for buffer in project.storage_buffers:
        continuation_lines.update(range(buffer.start_line + 1, buffer.end_line + 1))

    for instruction in project.instructions:
        text = instruction.text
        if instruction.kind == "directive" and text.lstrip().startswith("#version"):
            emitted.append("#version 300 es")
            emitted.extend(_webgl_storage_preamble(project.storage_buffers))
            emitted.append(f"#line {instruction.line + 1}")
            inserted_preamble = True
            continue
        if instruction.kind == "storage-buffer":
            buffer = next(
                item for item in project.storage_buffers
                if item.start_line == instruction.line
            )
            emitted.append(
                f"// project IR lowered std430 binding {buffer.binding} "
                f"({buffer.block_name}.{buffer.member_name}) to {buffer.sampler_name}"
            )
            continue
        if instruction.line in continuation_lines:
            emitted.append("// project IR storage declaration continuation")
            continue
        match = _UNIFORM_INITIALIZER.match(text)
        if match is not None:
            text = match.group("prefix") + ";" + match.group("comment")
        emitted.append(_replace_buffer_reads(text, project.storage_buffers))

    if not inserted_preamble:
        diagnostics.append(GLSLProjectDiagnostic(
            "GLSL_VERSION", 1, "shader has no #version directive"
        ))
        emitted = ["#version 300 es", *_webgl_storage_preamble(project.storage_buffers), "#line 1", *emitted]
    source = "\n".join(emitted)
    if not source.endswith("\n"):
        source += "\n"
    return WebGLProjectShader(
        name=name or project.source_name,
        source=source,
        project_ir=project,
        execution_plan=plan,
        diagnostics=tuple(diagnostics),
    )


def compile_glsl_project_to_webgl(
    source: str,
    *,
    source_name: str = "shader",
    stage_hint: str = "fragment",
) -> WebGLProjectShader:
    """Canonical convenience route: raw source -> project IR -> WebGL."""

    project = ingest_glsl_project(
        source, source_name=source_name, stage_hint=stage_hint
    )
    return emit_webgl_from_glsl_project(project, name=source_name)


__all__ = [
    "GLSLExecutionSection",
    "GLSLFunctionRegion",
    "GLSLProjectDiagnostic",
    "GLSLProjectExecutionPlan",
    "GLSLProjectIR",
    "GLSLProjectInstruction",
    "GLSLStorageBuffer",
    "PROJECT_SHADER_SCHEMA",
    "WEBGL_TRANSLATION_SCHEMA",
    "WebGLProjectShader",
    "compile_glsl_project_to_webgl",
    "emit_webgl_from_glsl_project",
    "ingest_glsl_project",
    "plan_glsl_project_execution",
]
