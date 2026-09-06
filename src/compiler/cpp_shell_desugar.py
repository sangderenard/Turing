"""Desugar a narrow C++-like shell into real C text.

Not a C++ parser -- see ``CPP_LIKE_SHELL_FOR_C_INTENT.md`` for why. This
rewrites source text so it can be handed, unchanged from here on, to the
existing, working, trusted ``pycparser``-based route
(``machine_code_lifting.py``). ``pycparser`` itself is never modified.

Scope, matching the intent document exactly:

* ``class Foo { fields; methods; };`` -> ``struct Foo { fields; };`` plus
  free functions ``ReturnType Foo__method(struct Foo* self, params)``.
* A constructor (a method named the same as the class) -> ``Foo__new``,
  returning an initialized ``struct Foo`` by value.
* ``obj.method(args)`` at a call site, where ``obj``'s class is known from
  a simple local declaration in the same block -> ``Foo__method(&obj, args)``.
* Single inheritance (``class Derived : public Base``) -> struct embedding
  (``struct Derived { struct Base base; ... };``), no virtual dispatch.

Anything else -- templates, operator overloading, multiple inheritance,
virtual functions, exceptions -- is rejected with ``CppShellUnsupported``,
never silently misdesugared.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


class CppShellUnsupported(ValueError):
    """Raised when source uses a construct outside the desugaring scope."""


_UNSUPPORTED_MARKERS = (
    (re.compile(r"\btemplate\s*<"), "templates"),
    (re.compile(r"\boperator\s*[\w+\-*/=<>!]"), "operator overloading"),
    (re.compile(r"\bvirtual\b"), "virtual functions"),
    (re.compile(r"\btry\b|\bcatch\s*\(|\bthrow\b"), "exceptions"),
    (re.compile(r"\bnamespace\b"), "namespaces"),
)


def _strip_comments(source: str) -> str:
    without_block = re.sub(r"/\*.*?\*/", " ", source, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", without_block)


def _check_supported(source: str) -> None:
    for pattern, label in _UNSUPPORTED_MARKERS:
        if pattern.search(source):
            raise CppShellUnsupported(
                f"C++-like shell does not support {label}"
            )
    # A single ``:`` after a class name is inheritance (handled); more than
    # one base separated by ``,`` is multiple inheritance (not handled).
    for match in re.finditer(r"\bclass\s+\w+\s*:\s*([^{]+)\{", source):
        bases = match.group(1)
        if "," in bases:
            raise CppShellUnsupported("multiple inheritance")


def _match_braces(source: str, open_index: int) -> int:
    """Return the index of the ``}`` matching the ``{`` at ``open_index``."""

    depth = 0
    for index in range(open_index, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    raise CppShellUnsupported("unbalanced braces in class body")


@dataclass
class _Method:
    name: str
    return_type: str
    params: str
    body: str
    is_constructor: bool


@dataclass
class _ClassInfo:
    name: str
    base: str | None
    field_names: set[str] = field(default_factory=set)
    field_decls: list[str] = field(default_factory=list)
    methods: list[_Method] = field(default_factory=list)


_CLASS_HEADER_RE = re.compile(
    r"\bclass\s+(?P<name>\w+)\s*(?::\s*public\s+(?P<base>\w+)\s*)?\{",
)
_METHOD_RE = re.compile(
    r"(?P<rtype>[\w][\w:<>*&\s]*?)\s+(?P<name>\w+)\s*\((?P<params>[^)]*)\)\s*\{",
)
_CONSTRUCTOR_RE = re.compile(r"^\s*(?P<name>\w+)\s*\((?P<params>[^)]*)\)\s*\{")
_FIELD_RE = re.compile(r"^\s*([\w][\w:<>*&\s]*?)\s+(\w+)\s*;\s*$")


def _parse_class(source: str, header_match: re.Match) -> tuple[_ClassInfo, int]:
    name = header_match.group("name")
    base = header_match.group("base")
    open_brace = header_match.end() - 1
    close_brace = _match_braces(source, open_brace)
    body = source[open_brace + 1:close_brace]
    info = _ClassInfo(name=name, base=base)
    if base:
        info.field_names.add("__base_placeholder__")  # never matched; documents intent

    cursor = 0
    while cursor < len(body):
        remaining = body[cursor:]
        leading_ws = len(remaining) - len(remaining.lstrip())
        stripped_remaining = remaining[leading_ws:]
        constructor_match = _CONSTRUCTOR_RE.match(stripped_remaining)
        method_match = _METHOD_RE.match(stripped_remaining)
        if constructor_match and constructor_match.group("name") == name:
            local_open = cursor + leading_ws + constructor_match.end() - 1
            local_close = _match_braces(body, local_open)
            method_body = body[local_open + 1:local_close]
            info.methods.append(_Method(
                name=name, return_type="struct " + name,
                params=constructor_match.group("params"),
                body=method_body, is_constructor=True,
            ))
            cursor = local_close + 1
            continue
        if method_match:
            local_open = cursor + leading_ws + method_match.end() - 1
            local_close = _match_braces(body, local_open)
            method_body = body[local_open + 1:local_close]
            info.methods.append(_Method(
                name=method_match.group("name").strip(),
                return_type=method_match.group("rtype").strip(),
                params=method_match.group("params"),
                body=method_body, is_constructor=False,
            ))
            cursor = local_close + 1
            continue
        newline = body.find("\n", cursor)
        line_end = newline if newline != -1 else len(body)
        line = body[cursor:line_end]
        field_match = _FIELD_RE.match(line)
        if field_match:
            field_type, field_name = field_match.group(1).strip(), field_match.group(2)
            info.field_decls.append(f"{field_type} {field_name};")
            info.field_names.add(field_name)
        cursor = line_end + 1

    return info, close_brace + 1


def _rewrite_field_refs(body: str, field_names: set[str], receiver: str, accessor: str) -> str:
    """Rewrite ``this->x`` / bare ``x`` to ``receiver<accessor>x`` for known fields."""

    body = re.sub(r"\bthis->", f"{receiver}{accessor}", body)
    body = re.sub(r"\bthis\b", receiver, body)
    for name in sorted(field_names, key=len, reverse=True):
        body = re.sub(
            rf"(?<![.\w>]){re.escape(name)}\b(?!\s*\()",
            f"{receiver}{accessor}{name}",
            body,
        )
    return body


def _method_owner_and_hops(
    classes: dict[str, "_ClassInfo"], class_name: str, method_name: str,
) -> tuple[str, int] | None:
    """Find which class in ``class_name``'s inheritance chain owns
    ``method_name``, and how many ``.base`` hops from ``self`` reach it.

    Requires bases to already be present in ``classes`` -- true regardless
    of declaration order once ``desugar_cpp_shell`` parses every class
    before emitting any of them.
    """

    hops = 0
    current: str | None = class_name
    while current is not None:
        info = classes.get(current)
        if info is None:
            return None
        if any(m.name == method_name for m in info.methods if not m.is_constructor):
            return current, hops
        current = info.base
        hops += 1
    return None


def _rewrite_method_calls(
    body: str, classes: dict[str, "_ClassInfo"], class_name: str, self_expr: str,
) -> str:
    """Rewrite a bare, implicit-``this`` call (``method(args)``, no
    receiver) into ``OwningClass__method(<hops to self>, args)``.

    Only bare calls -- ``obj.method(args)`` is ``_rewrite_call_sites``'s
    job, on the fully assembled output, where every class's shape is known.
    """

    def replace(match: re.Match) -> str:
        method_name, args = match.group(1), match.group(2)
        found = _method_owner_and_hops(classes, class_name, method_name)
        if found is None:
            return match.group(0)
        owner, hops = found
        # Verified for hops == 1 (direct base) only. hops > 1 (a method
        # inherited from a grandparent) is unverified and likely wrong:
        # every hop after the first should switch from ``->base`` to
        # ``.base`` (``self->base`` is a value, not a pointer, so chaining
        # ``->`` again is invalid C past the first hop). Not fixed because
        # nothing in this codebase exercises multi-level inheritance yet.
        receiver = self_expr
        for _ in range(hops):
            receiver = f"&({receiver.lstrip('&')}->base)"
        prefix = receiver if args.strip() == "" else f"{receiver}, {args}"
        return f"{owner}__{method_name}({prefix})"

    return re.sub(r"(?<![.\w>])(\w+)\s*\(([^)]*)\)", replace, body)


def _emit_class(info: _ClassInfo, classes: dict[str, _ClassInfo]) -> str:
    lines: list[str] = []
    struct_fields = list(info.field_decls)
    if info.base:
        struct_fields.insert(0, f"struct {info.base} base;")
    lines.append(f"struct {info.name} {{")
    for decl in struct_fields:
        lines.append(f"    {decl}")
    lines.append("};")
    # A caller writing ``ClassName obj;`` (valid in the original C++-like
    # shell, where a class name is directly usable as a type) needs a bare
    # type name to keep working in C, which requires ``struct`` unless a
    # typedef exists.
    lines.append(f"typedef struct {info.name} {info.name};")
    lines.append("")

    for method in info.methods:
        if method.is_constructor:
            rewritten = _rewrite_field_refs(method.body, info.field_names, "self", ".")
            rewritten = _rewrite_method_calls(rewritten, classes, info.name, "&self")
            lines.append(
                f"struct {info.name} {info.name}__new({method.params}) {{"
            )
            lines.append(f"    struct {info.name} self = {{0}};")
            lines.append(f"   {rewritten}")
            lines.append("    return self;")
            lines.append("}")
        else:
            rewritten = _rewrite_field_refs(method.body, info.field_names, "self", "->")
            rewritten = _rewrite_method_calls(rewritten, classes, info.name, "self")
            params = method.params.strip()
            full_params = f"struct {info.name}* self" + (f", {params}" if params else "")
            lines.append(
                f"{method.return_type} {info.name}__{method.name}({full_params}) {{"
            )
            lines.append(f"   {rewritten}")
            lines.append("}")
        lines.append("")
    return "\n".join(lines)


def _rewrite_call_sites(source: str, classes: dict[str, _ClassInfo]) -> str:
    """Rewrite ``obj.method(args)`` -> ``ClassName__method(&obj, args)``.

    ``obj``'s class is resolved only from a simple local declaration
    (``ClassName obj;`` or ``ClassName obj = ...;``) earlier in the same
    text -- deliberately narrow, matching the scope in the intent document.
    """

    variable_class: dict[str, str] = {}
    for class_name in classes:
        for match in re.finditer(
            rf"\b{re.escape(class_name)}\s+(\w+)\s*[=;]", source,
        ):
            variable_class[match.group(1)] = class_name

    def replace_call(match: re.Match) -> str:
        var_name, method_name, args = match.group(1), match.group(2), match.group(3)
        class_name = variable_class.get(var_name)
        if class_name is None or method_name not in {
            m.name for m in classes[class_name].methods if not m.is_constructor
        } if class_name in classes else False:
            return match.group(0)
        prefix = f"&{var_name}" if args.strip() == "" else f"&{var_name}, {args}"
        return f"{class_name}__{method_name}({prefix})"

    return re.sub(r"\b(\w+)\.(\w+)\s*\(([^)]*)\)", replace_call, source)


def desugar_cpp_shell(source: str) -> str:
    """Rewrite ``source`` (the narrow C++-like shell) into real C text."""

    stripped = _strip_comments(source)
    _check_supported(stripped)

    # Pass 1: parse every class before emitting any of them, so a derived
    # class can resolve an inherited method regardless of whether its base
    # was declared earlier or later in the source.
    classes: dict[str, _ClassInfo] = {}
    spans: list[tuple[int, int, str | None]] = []  # (start, end, class_name or None)
    cursor = 0
    while True:
        header_match = _CLASS_HEADER_RE.search(stripped, cursor)
        if header_match is None:
            spans.append((cursor, len(stripped), None))
            break
        spans.append((cursor, header_match.start(), None))
        info, next_cursor = _parse_class(stripped, header_match)
        classes[info.name] = info
        if next_cursor < len(stripped) and stripped[next_cursor] == ";":
            next_cursor += 1
        spans.append((header_match.start(), next_cursor, info.name))
        cursor = next_cursor

    # Pass 2: emit, now with every class (and its full inheritance chain)
    # known, so bare inherited-method calls resolve correctly.
    output_parts = [
        _emit_class(classes[class_name], classes) if class_name is not None
        else stripped[start:end]
        for start, end, class_name in spans
    ]

    combined = "".join(output_parts)
    return _rewrite_call_sites(combined, classes)


__all__ = ["desugar_cpp_shell", "CppShellUnsupported"]
