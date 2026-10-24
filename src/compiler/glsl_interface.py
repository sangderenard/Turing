"""Read the declared interface of a host-written GLSL shader.

An extracted host shader (``shader_extractor``) arrives as exact source with
no Turing program behind it, so its component ABI can only come from what
the source itself declares at global scope: stage inputs and outputs,
default-block uniforms, opaque uniforms, and uniform/buffer/IO blocks, with
their ``layout`` qualifiers. This module reads exactly that and nothing
else. A source this reader cannot read with certainty (conditional
compilation, macros in declarations, struct-typed interface variables, an
unknown qualifier) is refused by name, never guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


class GLSLInterfaceRefusal(ValueError):
    """The declared interface cannot be read with certainty."""


@dataclass(frozen=True)
class GLSLMember:
    name: str
    glsl_type: str
    #: ``None`` for a non-array, ``-1`` for a runtime-sized array.
    array: tuple[int, ...] | None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "glsl_type": self.glsl_type,
            "dtype": glsl_dtype(self.glsl_type),
            "array": None if self.array is None else [
                None if size < 0 else size for size in self.array
            ],
        }


@dataclass(frozen=True)
class GLSLDeclaration:
    """One global interface declaration."""

    #: in | out | uniform | buffer
    storage: str
    name: str
    #: The declared type, or the block name for a block.
    glsl_type: str
    array: tuple[int, ...] | None
    layout: tuple[tuple[str, str | None], ...]
    qualifiers: tuple[str, ...]
    #: Block members; empty for a plain variable.
    members: tuple[GLSLMember, ...] = ()
    #: The instance name of a block, when it has one.
    instance: str | None = None
    line: int = 0

    @property
    def is_block(self) -> bool:
        return bool(self.members)

    def layout_value(self, key: str) -> str | None:
        for name, value in self.layout:
            if name == key:
                return value
        return None

    def layout_int(self, key: str) -> int | None:
        value = self.layout_value(key)
        return None if value is None else int(value, 0)

    def has_layout(self, key: str) -> bool:
        return any(name == key for name, _value in self.layout)


@dataclass(frozen=True)
class GLSLInterface:
    version: str | None
    profile: str | None
    declarations: tuple[GLSLDeclaration, ...]
    #: ``layout(local_size_x = ...) in;`` for a compute shader.
    local_size: tuple[int, int, int] | None = None


_OPAQUE = re.compile(r"^[iu]?(sampler|image|texture|subpassInput)\w*$|^atomic_uint$")
_SCALARS = {"float": "f32", "int": "i32", "uint": "u32", "bool": "bool", "double": "f64"}
_VECTOR_PREFIX = {"": "f32", "i": "i32", "u": "u32", "b": "bool", "d": "f64"}
_PASSIVE_QUALIFIERS = {
    "flat", "smooth", "noperspective", "centroid", "sample", "patch",
    "highp", "mediump", "lowp", "invariant", "precise",
    "readonly", "writeonly", "coherent", "volatile", "restrict",
}
_STORAGE = {"in", "out", "uniform", "buffer", "attribute", "varying"}
_SKIPPED_STORAGE = {"const", "shared"}
_BUILTIN_BLOCKS = {"gl_PerVertex"}


def glsl_dtype(glsl_type: str) -> str:
    """The logical dtype spelling of a GLSL type.

    Scalars use the component ABI's ``f32``/``i32``/``u32``/``bool``;
    vectors are ``<scalar>x<N>`` and matrices ``f32x<C>x<R>`` (column count
    first, as GLSL spells ``matCxR``). Opaque types keep their GLSL name.
    """

    if glsl_type in _SCALARS:
        return _SCALARS[glsl_type]
    match = re.fullmatch(r"([iubd]?)vec([234])", glsl_type)
    if match:
        return f"{_VECTOR_PREFIX[match.group(1)]}x{match.group(2)}"
    match = re.fullmatch(r"(d?)mat([234])(?:x([234]))?", glsl_type)
    if match:
        columns = match.group(2)
        rows = match.group(3) or columns
        return f"{'f64' if match.group(1) else 'f32'}x{columns}x{rows}"
    return glsl_type


def is_opaque(glsl_type: str) -> bool:
    return bool(_OPAQUE.match(glsl_type))


def _strip_comments(source: str) -> str:
    """Remove comments, keeping every newline so line numbers hold."""

    def blank(match: re.Match[str]) -> str:
        return re.sub(r"[^\n]", " ", match.group(0))

    return re.sub(r"/\*.*?\*/|//[^\n]*", blank, source, flags=re.S)


def _preprocessor(source: str) -> tuple[str, set[str]]:
    """Blank the preprocessor lines this reader accepts; refuse the rest."""

    defined: set[str] = set()
    lines = source.split("\n")
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        directive = stripped[1:].strip().split(None, 1)
        word = directive[0] if directive else ""
        if word in {"version", "extension", "line", "pragma", ""}:
            lines[index] = ""
            continue
        if word == "define":
            name = re.match(r"\w+", directive[1] if len(directive) > 1 else "")
            if name is None:
                raise GLSLInterfaceRefusal(f"line {index + 1}: unreadable #define")
            defined.add(name.group(0))
            lines[index] = ""
            continue
        raise GLSLInterfaceRefusal(
            f"line {index + 1}: #{word} makes the declared interface depend on "
            "preprocessor state this reader does not evaluate"
        )
    return "\n".join(lines), defined


def _global_statements(source: str) -> list[tuple[int, str]]:
    """Split global scope into statements; function bodies are skipped."""

    statements: list[tuple[int, str]] = []
    start = 0
    index = 0
    length = len(source)
    while index < length:
        char = source[index]
        if char == "{":
            head = source[start:index]
            close = _matching_brace(source, index)
            if "(" in re.sub(r"\blayout\s*\([^)]*\)", "", head):
                # A function definition: its body is not interface.
                index = close + 1
                start = index
                continue
            index = close + 1
            continue
        if char == ";":
            text = source[start:index]
            if text.strip():
                line = source.count("\n", 0, start + len(text) - len(text.lstrip())) + 1
                statements.append((line, " ".join(text.split())))
            start = index + 1
        index += 1
    if source[start:].strip():
        raise GLSLInterfaceRefusal("source ends inside a global declaration")
    return statements


def _matching_brace(source: str, open_index: int) -> int:
    depth = 0
    for index in range(open_index, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    raise GLSLInterfaceRefusal("unbalanced braces in the shader source")


def _layout(text: str) -> tuple[tuple[tuple[str, str | None], ...], str]:
    entries: list[tuple[str, str | None]] = []
    while True:
        match = re.match(r"\s*layout\s*\(([^)]*)\)\s*", text)
        if match is None:
            return tuple(entries), text
        for item in match.group(1).split(","):
            item = item.strip()
            if not item:
                continue
            if "=" in item:
                key, value = (part.strip() for part in item.split("=", 1))
                if not re.fullmatch(r"(0[xX][0-9a-fA-F]+|\d+)[uU]?", value):
                    raise GLSLInterfaceRefusal(
                        f"layout value {item!r} is not an integer literal"
                    )
                entries.append((key, value.rstrip("uU")))
            else:
                entries.append((item, None))
        text = text[match.end():]


def _array(text: str, defined: set[str], line: int) -> tuple[tuple[int, ...] | None, str]:
    sizes: list[int] = []
    while True:
        match = re.search(r"\[\s*([^\]]*)\]\s*$", text)
        if match is None:
            break
        size = match.group(1).strip()
        if not size:
            sizes.append(-1)
        elif re.fullmatch(r"\d+[uU]?", size):
            sizes.append(int(size.rstrip("uU")))
        else:
            reason = "a macro" if size in defined else "a non-literal size"
            raise GLSLInterfaceRefusal(f"line {line}: array size {size!r} is {reason}")
        text = text[: match.start()].rstrip()
    return (tuple(reversed(sizes)) if sizes else None), text


def _members(body: str, defined: set[str], line: int, structs: set[str]) -> tuple[GLSLMember, ...]:
    members: list[GLSLMember] = []
    for statement in body.split(";"):
        statement = " ".join(statement.split())
        if not statement:
            continue
        _layout_entries, statement = _layout(statement)
        words = statement.split()
        while words and words[0] in _PASSIVE_QUALIFIERS:
            words.pop(0)
        if len(words) < 2:
            raise GLSLInterfaceRefusal(f"line {line}: unreadable block member {statement!r}")
        glsl_type = words[0]
        if glsl_type in structs:
            raise GLSLInterfaceRefusal(
                f"line {line}: block member of struct type {glsl_type!r}"
            )
        for declarator in " ".join(words[1:]).split(","):
            array, name = _array(declarator.strip(), defined, line)
            type_array, _ = _array(glsl_type, defined, line)
            if not re.fullmatch(r"[A-Za-z_]\w*", name):
                raise GLSLInterfaceRefusal(f"line {line}: unreadable block member {declarator!r}")
            members.append(GLSLMember(name, re.sub(r"\[.*", "", glsl_type), array or type_array))
    return tuple(members)


def read_glsl_interface(source: str) -> GLSLInterface:
    """Return every global interface declaration ``source`` makes."""

    from .shader_deployment import glsl_version_directive

    version, profile = glsl_version_directive(source)
    text, defined = _preprocessor(_strip_comments(source))
    declarations: list[GLSLDeclaration] = []
    structs: set[str] = set()
    local_size: tuple[int, int, int] | None = None
    for line, statement in _global_statements(text):
        if statement.startswith("precision "):
            continue
        struct = re.match(r"struct\s+(\w+)\s*\{", statement)
        if struct:
            structs.add(struct.group(1))
            continue
        macro = next(
            (word for word in re.findall(r"[A-Za-z_]\w*", statement) if word in defined),
            None,
        )
        if macro is not None:
            raise GLSLInterfaceRefusal(
                f"line {line}: macro {macro!r} in a global declaration"
            )
        layout, rest = _layout(statement)
        words = rest.split()
        if words == ["in"] and layout:
            keys = dict(layout)
            if any(key.startswith("local_size") for key in keys):
                local_size = tuple(  # type: ignore[assignment]
                    int(keys.get(f"local_size_{axis}") or "1", 0) for axis in "xyz"
                )
            continue
        if words and words[0] == "out" and len(words) == 1:
            continue  # e.g. a geometry ``layout(...) out;``
        qualifiers: list[str] = []
        storage: str | None = None
        while words and (words[0] in _PASSIVE_QUALIFIERS or words[0] in _STORAGE or words[0] in _SKIPPED_STORAGE):
            word = words.pop(0)
            if word in _STORAGE:
                if storage is not None:
                    raise GLSLInterfaceRefusal(f"line {line}: two storage qualifiers in {statement!r}")
                storage = {"attribute": "in", "varying": "varying"}.get(word, word)
            elif word in _SKIPPED_STORAGE:
                storage = word
            else:
                qualifiers.append(word)
        if storage is None or storage in _SKIPPED_STORAGE:
            if "(" in rest:
                continue  # a function prototype
            if storage in _SKIPPED_STORAGE:
                continue  # a global constant or workgroup-shared value
            raise GLSLInterfaceRefusal(f"line {line}: unread global declaration {statement!r}")
        remainder = " ".join(words)
        block = re.match(r"(\w+)\s*\{(.*)\}\s*(.*)$", remainder)
        if block:
            name, body, instance_text = block.groups()
            if name in _BUILTIN_BLOCKS:
                continue
            members = _members(body, defined, line, structs)
            array, instance = _array(instance_text.strip(), defined, line)
            declarations.append(GLSLDeclaration(
                storage=storage, name=name, glsl_type=name, array=array,
                layout=layout, qualifiers=tuple(qualifiers), members=members,
                instance=instance or None, line=line,
            ))
            continue
        if not words:
            raise GLSLInterfaceRefusal(f"line {line}: unread global declaration {statement!r}")
        glsl_type = words[0]
        if glsl_type in structs:
            raise GLSLInterfaceRefusal(
                f"line {line}: interface variable of struct type {glsl_type!r}"
            )
        type_array, glsl_type = _array(glsl_type, defined, line)
        for declarator in " ".join(words[1:]).split(","):
            declarator = declarator.strip()
            if "=" in declarator:
                if storage != "uniform":
                    raise GLSLInterfaceRefusal(f"line {line}: initializer on {storage} {declarator!r}")
                declarator = declarator.split("=", 1)[0].strip()
            array, name = _array(declarator, defined, line)
            if not re.fullmatch(r"[A-Za-z_]\w*", name) or not re.fullmatch(r"[A-Za-z_]\w*", glsl_type):
                raise GLSLInterfaceRefusal(f"line {line}: unread global declaration {statement!r}")
            if name.startswith("gl_"):
                continue  # a builtin redeclaration
            declarations.append(GLSLDeclaration(
                storage=storage, name=name, glsl_type=glsl_type,
                array=array or type_array, layout=layout,
                qualifiers=tuple(qualifiers), line=line,
            ))
    return GLSLInterface(version, profile, tuple(declarations), local_size)


__all__ = [
    "GLSLDeclaration",
    "GLSLInterface",
    "GLSLInterfaceRefusal",
    "GLSLMember",
    "glsl_dtype",
    "is_opaque",
    "read_glsl_interface",
]
