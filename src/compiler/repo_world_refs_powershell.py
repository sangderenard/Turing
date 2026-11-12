"""PowerShell reference plugin for the repository world scan.

Contract: ``references(path, text, tracked)`` returns
``(line, kind, specifier, target, reason)`` tuples.  Pure and deterministic;
never raises.  A path is emitted only when it is a literal (optionally
anchored at ``$PSScriptRoot``); any other variable or subexpression in the
path is not emitted.  A literal that leaves the repository root is reported
unresolved ("outside repository") rather than clamped.
"""

from __future__ import annotations

import re

LANGUAGE = "powershell"
SUFFIXES = (".ps1", ".psm1", ".psd1")

_SCRIPT_SUFFIXES = (".ps1", ".psm1", ".psd1", ".py", ".sh", ".bash", ".bat", ".cmd", ".js")

_JOIN = re.compile(
    r"""Join-Path\s+(?:-Path\s+)?(\$(?:\{\w+\}|\w+))\s+(?:-ChildPath\s+)?"""
    r"""(?:'([^']*)'|"([^"$`]*)")""", re.IGNORECASE)
_DOT_CTX = re.compile(r"(?:^|[\s;{(])\.\s*\(*\s*$")
_AMP_CTX = re.compile(r"(?<![&\w])&\s*\(*\s*$")
_IMPORT_CTX = re.compile(r"Import-Module\s+(?:-Name\s+)?\(*\s*$", re.IGNORECASE)
_STMT = r"(?:^\s*|[;{]\s*)"
_DOT = re.compile(_STMT + r"""\.\s+(?:'([^']*)'|"([^"]*)"|(\S+))""")
_AMP = re.compile(r"""(?<![&\w])&\s+(?:'([^']*)'|"([^"]*)"|(\.\.?[\\/]\S+|\$PSScriptRoot[\\/]\S+))""",
                  re.IGNORECASE)
_BARE = re.compile(_STMT + r"""(\.\.?[\\/][^\s;|&()'"$`]+|\$PSScriptRoot[\\/][^\s;|&()'"`]+)""",
                   re.IGNORECASE)
_IMPORT = re.compile(r"""Import-Module\s+(?:-Name\s+)?(?:'([^']*)'|"([^"]*)"|([^\s;|)]+))""",
                     re.IGNORECASE)
_RUN = re.compile(
    r"""(?<![\w.-])(?:python3?|py|pwsh|powershell)(?:\.exe)?['"]?\s+(?:-\w+\s+)*?(?:-File\s+)?(?!-)"""
    r"""(?:'([^'\s$`]+)'|"([^"\s$`]+)"|([^\s'"$`;|&()]+))""", re.IGNORECASE)
_QUOTED = re.compile(r"""'([^']*)'|"([^"$`]*)\"""")


def _strip_comments(text: str) -> str:
    out: list[str] = []
    n = len(text)
    i = 0
    quote = ""
    in_block = False
    while i < n:
        c = text[i]
        if c == "\n":
            out.append(c)
            quote = ""
            i += 1
        elif in_block:
            if text.startswith("#>", i):
                in_block = False
                i += 2
            else:
                i += 1
        elif quote:
            out.append(c)
            if quote == '"' and c == "`" and i + 1 < n and text[i + 1] != "\n":
                out.append(text[i + 1])
                i += 2
                continue
            if c == quote:
                quote = ""
            i += 1
        elif c in "'\"":
            quote = c
            out.append(c)
            i += 1
        elif text.startswith("<#", i):
            in_block = True
            i += 2
        elif c == "#" and (not out or out[-1] in " \t;({\n"):
            while i < n and text[i] != "\n":
                i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _expand(raw: str):
    """Return (relative POSIX text, 'dir'|'rel') or None when not a literal."""
    s = raw.strip()
    if not s or "`" in s:
        return None
    s = re.sub(r"\$\{PSScriptRoot\}", "$PSScriptRoot", s, flags=re.IGNORECASE)
    mode = "rel"
    if s.lower().startswith("$psscriptroot"):
        s = s[len("$PSScriptRoot"):]
        if s and s[0] not in "\\/":
            return None
        mode = "dir"
    if any(ch in s for ch in "$()@*?~%;|<>"):
        return None
    if re.match(r"^[A-Za-z]:", s) or s.startswith(("\\\\", "//")):
        return None
    s = s.replace("\\", "/")
    if mode == "rel" and s.startswith("/"):
        return None
    return s, mode


def _join(base: str, rel: str):
    parts = [p for p in base.split("/") if p]
    for seg in rel.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if not parts:
                return None
            parts.pop()
        else:
            parts.append(seg)
    return "/".join(parts)


def _looks_like_file(rel: str) -> bool:
    return rel.lower().endswith(_SCRIPT_SUFFIXES)


def _resolve(rel: str, mode: str, here: str, tracked, strict: bool = False):
    """Return (emit, target, reason)."""
    bases = [here] if mode == "dir" else [here, ""]
    ok = [c for c in (_join(b, rel) for b in bases) if c is not None]
    if not ok:
        return _looks_like_file(rel), None, "outside repository"
    hits: list[str] = []
    for c in ok:
        if c in tracked and c not in hits:
            hits.append(c)
    if strict and len(hits) > 1:
        return True, None, "ambiguous: tracked relative to both script directory and repository root"
    if hits:
        return True, hits[0], ""
    return _looks_like_file(rel), None, "not tracked"


def references(path: str, text: str, tracked: frozenset[str]) -> list[tuple[int, str, str, str | None, str]]:
    try:
        return _references(path, text, tracked)
    except Exception:
        return []


def _references(path, text, tracked):
    here = "/".join(path.split("/")[:-1])
    clean = _strip_comments(text.replace("\r\n", "\n").replace("\r", "\n"))
    found: list[tuple[int, str, str, str | None, str]] = []
    seen: set = set()

    def add(line, kind, spec, literal, strict=False):
        exp = _expand(literal)
        if exp is None:
            return
        emit, target, reason = _resolve(exp[0], exp[1], here, tracked, strict)
        if not emit:
            return
        key = (line, kind, spec, target)
        if key in seen:
            return
        seen.add(key)
        found.append((line, kind, spec, target, reason))

    is_psd1 = path.lower().endswith(".psd1")
    for lineno, line in enumerate(clean.split("\n"), 1):
        if not line.strip():
            continue
        if is_psd1:
            for m in _QUOTED.finditer(line):
                lit = m.group(1) if m.group(1) is not None else m.group(2)
                if lit and lit.lower().endswith((".psm1", ".ps1")):
                    add(lineno, "powershell-import-module", lit, lit)
            continue
        join_spans: list[tuple[int, int]] = []
        for m in _JOIN.finditer(line):
            join_spans.append(m.span())
            child = m.group(2) if m.group(2) is not None else m.group(3)
            if not child or not child.strip("\\/"):
                continue
            var = m.group(1).replace("{", "").replace("}", "").lower()
            child = child.lstrip("\\/")
            prefix = line[:m.start()]
            if _IMPORT_CTX.search(prefix):
                kind = "powershell-import-module"
            elif _DOT_CTX.search(prefix):
                kind = "powershell-dot-source"
            elif _AMP_CTX.search(prefix):
                kind = "powershell-call"
            else:
                kind = "powershell-path"
            if var == "$psscriptroot":
                add(lineno, kind, child, "$PSScriptRoot/" + child)
            else:
                add(lineno, kind, child, child, strict=True)

        def outside(m):
            return not any(a <= m.start() < b for a, b in join_spans)

        for m in _DOT.finditer(line):
            lit = next(g for g in m.groups() if g is not None)
            if lit and outside(m):
                add(lineno, "powershell-dot-source", lit, lit)
        for m in _AMP.finditer(line):
            lit = next(g for g in m.groups() if g is not None)
            if lit and outside(m):
                add(lineno, "powershell-call", lit, lit)
        for m in _BARE.finditer(line):
            lit = m.group(1)
            if lit and outside(m):
                add(lineno, "powershell-call", lit, lit)
        for m in _IMPORT.finditer(line):
            lit = next(g for g in m.groups() if g is not None)
            if lit and outside(m) and (lit.startswith(".") or lit.lower().startswith("$psscriptroot")
                                       or "\\" in lit or "/" in lit
                                       or lit.lower().endswith((".psm1", ".psd1", ".ps1"))):
                add(lineno, "powershell-import-module", lit, lit)
        for m in _RUN.finditer(line):
            lit = next(g for g in m.groups() if g is not None)
            if lit and outside(m) and lit.lower().endswith((".py", ".ps1")):
                add(lineno, "powershell-call", lit, lit)
    return found
