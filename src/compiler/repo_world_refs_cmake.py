"""CMake reference extractor plugin for the repository world scan.

Contract (see ``repo_world_scan``): ``LANGUAGE``, ``SUFFIXES``, ``FILENAMES``
and the pure ``references(path, text, tracked)`` returning
``(line, kind, specifier, target, reason)`` tuples.

Kinds: ``cmake-subdirectory``, ``cmake-include``, ``cmake-source``,
``cmake-configure`` and ``cmake-glob`` (an unresolved entry; globs are never
expanded).  Directories (include dirs) are regions, not rooms, so they are not
emitted.

Assumption stated once: ``${CMAKE_SOURCE_DIR}`` and ``${PROJECT_SOURCE_DIR}``
are taken to be the repository root, i.e. the scanned repository is the
top-level CMake project.  ``${CMAKE_CURRENT_SOURCE_DIR}`` and
``${CMAKE_CURRENT_LIST_DIR}`` are the directory of the file being read.  Any
other ``${...}``, ``@VAR@`` or generator expression ``$<...>`` makes an
argument unresolvable and it is not emitted.
"""

from __future__ import annotations

import re
from typing import NamedTuple

LANGUAGE = "cmake"
SUFFIXES: tuple[str, ...] = (".cmake",)
FILENAMES: tuple[str, ...] = ("CMakeLists.txt",)

_SOURCE_SUFFIXES = frozenset({
    ".c", ".cc", ".cpp", ".cxx", ".c++", ".cu", ".cuh", ".h", ".hh", ".hpp",
    ".hxx", ".inl", ".ipp", ".m", ".mm", ".f", ".f90", ".for", ".s", ".asm",
    ".rc", ".def", ".in", ".ui", ".qrc", ".glsl", ".vert", ".frag", ".comp",
    ".py", ".cs", ".swift", ".rs", ".ixx", ".cppm",
})
_KEYWORDS = frozenset({
    "STATIC", "SHARED", "MODULE", "OBJECT", "INTERFACE", "PUBLIC", "PRIVATE",
    "EXCLUDE_FROM_ALL", "WIN32", "MACOSX_BUNDLE", "IMPORTED", "ALIAS", "UNKNOWN",
    "GLOBAL", "FILE_SET", "FILES", "BASE_DIRS", "TYPE", "HEADERS", "CXX_MODULES",
    "CXX_MODULE_SETS", "CXX_MODULE_HEADER_UNITS", "SOURCES", "APPEND", "PREPEND",
})
_LIST_VAR = re.compile(r"(?:^|_)(SOURCES|SRCS|HEADERS)$", re.IGNORECASE)
_DIR_VARS = ("${CMAKE_CURRENT_SOURCE_DIR}", "${CMAKE_CURRENT_LIST_DIR}")
_ROOT_VARS = ("${CMAKE_SOURCE_DIR}", "${PROJECT_SOURCE_DIR}")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class _Arg(NamedTuple):
    line: int
    text: str
    quoted: bool


class _Command(NamedTuple):
    line: int
    name: str
    args: tuple[_Arg, ...]


def _bracket_open(text: str, i: int) -> int | None:
    """Return the '=' count if ``text[i:]`` opens a bracket ``[=*[``, else None."""
    if i >= len(text) or text[i] != "[":
        return None
    j = i + 1
    while j < len(text) and text[j] == "=":
        j += 1
    if j < len(text) and text[j] == "[":
        return j - i - 1
    return None


def _read_bracket(text: str, i: int, eq: int) -> tuple[str, int]:
    """Read a bracket body starting at the opening ``[``; returns (body, next index)."""
    start = i + eq + 2
    close = "]" + "=" * eq + "]"
    end = text.find(close, start)
    if end < 0:
        return text[start:], len(text)
    return text[start:end], end + len(close)


def _skip_comment(text: str, i: int) -> int:
    """``text[i]`` is '#'; return the index after the comment."""
    eq = _bracket_open(text, i + 1)
    if eq is not None:
        return _read_bracket(text, i + 1, eq)[1]
    end = text.find("\n", i)
    return len(text) if end < 0 else end


def _tokenize(text: str) -> list[_Command]:
    commands: list[_Command] = []
    n = len(text)
    i = 0
    line = 1
    while i < n:
        ch = text[i]
        if ch == "\n":
            line += 1
            i += 1
        elif ch == "#":
            j = _skip_comment(text, i)
            line += text.count("\n", i, j)
            i = j
        elif ch.isalpha() or ch == "_":
            m = _IDENT.match(text, i)
            if m is None:
                i += 1
                continue
            j = m.end()
            while j < n and text[j] in " \t":
                j += 1
            if j < n and text[j] == "(":
                start_line = line
                args, j, line = _read_args(text, j + 1, line)
                commands.append(_Command(start_line, m.group(0).lower(), tuple(args)))
                i = j
            else:
                i = m.end()
        else:
            i += 1
    return commands


def _read_args(text: str, i: int, line: int) -> tuple[list[_Arg], int, int]:
    """Read arguments after the opening paren; stops after the matching ``)``."""
    args: list[_Arg] = []
    depth = 1
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\n":
            line += 1
            i += 1
        elif ch in " \t\r":
            i += 1
        elif ch == "#":
            j = _skip_comment(text, i)
            line += text.count("\n", i, j)
            i = j
        elif ch == "(":
            depth += 1
            i += 1
        elif ch == ")":
            depth -= 1
            i += 1
            if depth == 0:
                return args, i, line
        elif ch == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            j = min(j, n)
            args.append(_Arg(line, text[i + 1:j], True))
            line += text.count("\n", i, j)
            i = min(j + 1, n)
        else:
            eq = _bracket_open(text, i)
            if eq is not None:
                body, j = _read_bracket(text, i, eq)
                args.append(_Arg(line, body, True))
                line += text.count("\n", i, j)
                i = j
                continue
            j = i
            while j < n and text[j] not in " \t\r\n()#":
                j += 1
            if j == i:  # defensive: always advance
                j = i + 1
            args.append(_Arg(line, text[i:j], False))
            i = j
    return args, i, line


# --- resolution ----------------------------------------------------------------

def _directory(path: str) -> str:
    return path.rpartition("/")[0]


def _substitute(arg: str, here: str) -> str | None:
    """Apply the four supported variables; None when anything else remains."""
    for var in _DIR_VARS:
        arg = arg.replace(var, here if here else ".")
    for var in _ROOT_VARS:
        arg = arg.replace(var, ".")
    if "${" in arg or "$<" in arg or "$ENV{" in arg or re.search(r"@\w+@", arg):
        return None
    return arg


def _normalise(here: str, arg: str) -> tuple[str | None, str]:
    """Join ``arg`` to the file's directory; (path, "") or (None, reason)."""
    arg = arg.replace("\\", "/")
    if arg.startswith("/") or re.match(r"^[A-Za-z]:/", arg):
        return None, "absolute path outside the repository"
    parts: list[str] = [p for p in here.split("/") if p] if here else []
    for part in arg.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if not parts:
                return None, "path escapes the repository root"
            parts.pop()
        else:
            parts.append(part)
    if not parts:
        return None, "empty path"
    return "/".join(parts), ""


def _looks_like_source(arg: str) -> bool:
    name = arg.replace("\\", "/").rsplit("/", 1)[-1]
    dot = name.rfind(".")
    return dot > 0 and name[dot:].lower() in _SOURCE_SUFFIXES


Ref = tuple[int, str, str, "str | None", str]


def _emit(out: list[Ref], line: int, kind: str, spec: str, here: str,
          tracked: frozenset[str], suffix: str = "", *, require_source: bool = False) -> None:
    sub = _substitute(spec, here)
    if sub is None or not sub.strip():
        return
    # A substituted variable already carries its own base (file dir or root).
    base = "" if sub != spec else here
    path, reason = _normalise(base, sub)
    if path is None:
        if not require_source or _looks_like_source(spec):
            out.append((line, kind, spec, None, reason))
        return
    candidate = path + suffix
    if candidate in tracked:
        out.append((line, kind, spec, candidate, ""))
    elif not require_source or _looks_like_source(spec):
        out.append((line, kind, spec, None, "not tracked"))


def _items(arg: _Arg) -> list[str]:
    """Split a ';'-list argument into its items."""
    return [p for p in arg.text.split(";") if p.strip()] if ";" in arg.text else [arg.text]


def _source_args(here: str, tracked: frozenset[str], out: list[Ref],
                 args: tuple[_Arg, ...]) -> None:
    for arg in args:
        if not arg.quoted and arg.text in _KEYWORDS:
            continue
        for item in _items(arg):
            if item in _KEYWORDS:
                continue
            _emit(out, arg.line, "cmake-source", item, here, tracked, require_source=True)


def _glob_entries(cmd: _Command, out: list[Ref]) -> None:
    args = cmd.args
    if len(args) < 2 or args[0].text.upper() not in ("GLOB", "GLOB_RECURSE"):
        return
    i = 2  # args[1] is the output variable
    while i < len(args):
        text = args[i].text
        upper = text.upper()
        if not args[i].quoted and upper in ("RELATIVE", "LIST_DIRECTORIES"):
            i += 2
            continue
        if not args[i].quoted and upper in ("CONFIGURE_DEPENDS", "FOLLOW_SYMLINKS"):
            i += 1
            continue
        out.append((args[i].line, "cmake-glob", text, None, "glob not expanded"))
        i += 1


def references(path: str, text: str, tracked: frozenset[str]) -> list[Ref]:
    try:
        return _references(path, text, tracked)
    except Exception:  # contract: never raises
        return []


def _references(path: str, text: str, tracked: frozenset[str]) -> list[Ref]:
    here = _directory(path)
    out: list[Ref] = []
    for cmd in _tokenize(text):
        args = cmd.args
        name = cmd.name
        if not args:
            continue
        if name == "add_subdirectory":
            _emit(out, args[0].line, "cmake-subdirectory", args[0].text, here, tracked,
                  "/CMakeLists.txt")
        elif name == "include":
            spec = args[0].text
            if "/" in spec or "\\" in spec or spec.lower().endswith(".cmake"):
                _emit(out, args[0].line, "cmake-include", spec, here, tracked)
        elif name in ("add_library", "add_executable"):
            if any(a.text == "ALIAS" and not a.quoted for a in args[1:]):
                continue
            _source_args(here, tracked, out, args[1:])
        elif name == "target_sources":
            _source_args(here, tracked, out, args[1:])
        elif name == "set":
            if _LIST_VAR.search(args[0].text):
                _source_args(here, tracked, out, args[1:])
        elif name == "list":
            if len(args) >= 2 and args[0].text.upper() == "APPEND" and _LIST_VAR.search(args[1].text):
                _source_args(here, tracked, out, args[2:])
        elif name == "configure_file":
            _emit(out, args[0].line, "cmake-configure", args[0].text, here, tracked)
        elif name == "file":
            _glob_entries(cmd, out)
    return out
