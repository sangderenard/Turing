"""Shell and batch reference plugin for the repository world scan.

Contract: ``references(path, text, tracked)`` returns
``(line, kind, specifier, target, reason)`` tuples.  Pure and deterministic;
never raises.  Only literal relative paths are emitted (``%~dp0`` in batch
files is the file's directory).  Paths containing other variables or
substitutions (``$X``, ``$(..)``, backticks, other ``%x%``) are not emitted.
"""

from __future__ import annotations

import re

LANGUAGE = "shell"
SUFFIXES = (".sh", ".bash", ".bat", ".cmd")

_SCRIPT_SUFFIXES = (".ps1", ".psm1", ".py", ".sh", ".bash", ".bat", ".cmd", ".js")
_TOKEN = r"""(?:"([^"\s$`]+)"|'([^'\s$`]+)'|([^\s"'$`;&|<>()]+))"""
_SH_LEAD = r"(?:^|[;&|(]\s*|\b(?:then|do|else)\s+)(?:\w+=\S*\s+)*(?:(?:exec|sudo|time|nohup)\s+)?"
_SH_SOURCE = re.compile(_SH_LEAD + r"(?:source|\.)\s+" + _TOKEN)
_SH_RUN = re.compile(
    _SH_LEAD + r"(?:(?P<sh>(?:ba|z|da)?sh)(?:\s+-[A-Za-z]+)*|(?P<py>python3?(?:\.\d+)?|py))\s+"
    r"(?:-(?!m\b)\w+\s+)*" + _TOKEN)
_SH_DIRECT = re.compile(_SH_LEAD + r"""(\.\.?/[^\s"'$`;&|<>()]+)""")
_BAT_LEAD = r"(?:^|[&|(]\s*)@?"
_BAT_CALL = re.compile(_BAT_LEAD + r"call\s+" + _TOKEN, re.IGNORECASE)
_BAT_PY = re.compile(_BAT_LEAD + r"(?:python3?|py)(?:\.exe)?\s+(?:-(?!m\b)\w+\s+)*" + _TOKEN, re.IGNORECASE)
_BAT_BARE = re.compile(_BAT_LEAD + r"""(\.\.?[\\/][^\s"'&|<>()]+)""", re.IGNORECASE)
_BAT_DP0 = re.compile(r"""%~dp0([^\s"'&|<>()%]+)""", re.IGNORECASE)
_BAT_PS = re.compile(r"(?<![\w.-])(?:pwsh|powershell)(?:\.exe)?\s+(?:-\w+\s+)*?-File\s+" + _TOKEN,
                     re.IGNORECASE)


def _strip_comment(line: str) -> str:
    quote = ""
    for i, c in enumerate(line):
        if quote:
            if c == quote:
                quote = ""
        elif c in "'\"":
            quote = c
        elif c == "#" and (i == 0 or line[i - 1] in " \t;&|("):
            return line[:i]
    return line


def _expand(raw: str, batch: bool):
    s = raw.strip()
    if not s or "`" in s:
        return None
    mode = "rel"
    if batch:
        if s.lower().startswith("%~dp0"):
            s = s[5:]
            mode = "dir"
        s = s.replace("\\", "/")
    if any(ch in s for ch in "$()@*?~%;|<>&"):
        return None
    if re.match(r"^[A-Za-z]:", s) or s.startswith("/"):
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


def references(path: str, text: str, tracked: frozenset[str]) -> list[tuple[int, str, str, str | None, str]]:
    try:
        return _references(path, text, tracked)
    except Exception:
        return []


def _tok(m):
    return next((m.group(i) for i in range(1, (m.re.groups or 0) + 1)
                 if m.re.groupindex.get("sh") != i and m.re.groupindex.get("py") != i
                 and m.group(i) is not None), None)


def _references(path, text, tracked):
    here = "/".join(path.split("/")[:-1])
    batch = path.lower().endswith((".bat", ".cmd"))
    found: list[tuple[int, str, str, str | None, str]] = []
    seen: set = set()

    def add(line, kind, lit):
        if lit is None:
            return
        exp = _expand(lit, batch)
        if exp is None:
            return
        rel, mode = exp
        bases = [here] if mode == "dir" else [here, ""]
        cands = [c for c in (_join(b, rel) for b in bases) if c is not None]
        target, reason = None, ""
        if not cands:
            reason = "outside repository"
        else:
            target = next((c for c in cands if c in tracked), None)
            if target is None:
                reason = "not tracked"
        if target is None and not _looks_like_file(rel):
            return
        key = (line, kind, lit, target)
        if key not in seen:
            seen.add(key)
            found.append((line, kind, lit, target, reason))

    for lineno, raw in enumerate(text.replace("\r\n", "\n").replace("\r", "\n").split("\n"), 1):
        if batch:
            low = raw.strip().lower()
            if low.startswith(("rem ", "@rem ", "::")) or low in ("rem", "@rem"):
                continue
            for m in _BAT_CALL.finditer(raw):
                add(lineno, "batch-call", _tok(m))
            for m in _BAT_PY.finditer(raw):
                add(lineno, "batch-call", _tok(m))
            for m in _BAT_PS.finditer(raw):
                add(lineno, "batch-call", _tok(m))
            for m in _BAT_BARE.finditer(raw):
                add(lineno, "batch-call", m.group(1))
            for m in _BAT_DP0.finditer(raw):
                if _looks_like_file(m.group(1)):
                    add(lineno, "batch-call", "%~dp0" + m.group(1))
        else:
            line = _strip_comment(raw)
            if not line.strip():
                continue
            for m in _SH_SOURCE.finditer(line):
                add(lineno, "shell-source", _tok(m))
            for m in _SH_RUN.finditer(line):
                lit = _tok(m)
                if lit is None or (m.group("py") and not lit.lower().endswith(".py")):
                    continue
                add(lineno, "shell-call", lit)
            for m in _SH_DIRECT.finditer(line):
                add(lineno, "shell-call", m.group(1))
    return found
