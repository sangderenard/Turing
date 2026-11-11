"""C-family `#include` / `#import` extractor plugin for repo_world_scan.

Pure: no filesystem, no subprocess. Resolution never guesses include
directories: relative to the including file, then repository-root exact, then
the unique tracked path ending in "/" + specifier.
"""
from __future__ import annotations

import re

LANGUAGE = "c"
SUFFIXES = (".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".inl", ".ipp", ".cu")

_DIRECTIVE = re.compile(r'^[ \t]*#[ \t]*(?:include|import)[ \t]*(?:"([^"\n]*)"|<([^>\n]*)>)')


def _mask_comments(text: str) -> str:
    """Blank comments (keeping newlines); string/char literals are kept verbatim."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "*":
            j = text.find("*/", i + 2)
            end = n if j < 0 else j + 2
            out.append("".join(ch if ch == "\n" else " " for ch in text[i:end]))
            i = end
        elif c == "/" and nxt == "/":
            j = i
            while j < n and text[j] != "\n":
                j += 1
            out.append(" " * (j - i))
            i = j
        elif c in "\"'":
            j = i + 1
            while j < n and text[j] != c and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            j = min(j + 1, n) if j < n and text[j] == c else min(j, n)
            out.append(text[i:j])
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _normalise(base: str, spec: str) -> str | None:
    parts = [p for p in base.split("/") if p] if base else []
    for part in spec.split("/"):
        if part == "..":
            if not parts:
                return None
            parts.pop()
        elif part not in ("", "."):
            parts.append(part)
    return "/".join(parts)


def _resolve(path: str, spec: str, tracked: frozenset[str]) -> tuple[str | None, str]:
    spec = spec.strip().replace("\\", "/")
    if not spec:
        return None, "empty include"
    if not spec.startswith("/"):
        base = path.rsplit("/", 1)[0] if "/" in path else ""
        rel = _normalise(base, spec)
        if rel is not None and rel in tracked:
            return rel, ""
        if spec in tracked:
            return spec, ""
        suffix = "/" + spec.lstrip("./") if not spec.startswith("../") else "/" + spec
        matches = sorted(p for p in tracked if p.endswith(suffix))
        if len(matches) == 1:
            return matches[0], ""
        if matches:
            return None, f"ambiguous: {len(matches)} candidates: " + ", ".join(matches[:3])
    return None, "not tracked"


def references(path: str, text: str, tracked: frozenset[str]) -> list[tuple[int, str, str, str | None, str]]:
    refs: list[tuple[int, str, str, str | None, str]] = []
    try:
        masked = _mask_comments(text if isinstance(text, str) else str(text))
        for number, line in enumerate(masked.splitlines(), start=1):
            match = _DIRECTIVE.match(line)
            if not match:
                continue
            spec = match.group(1) if match.group(1) is not None else match.group(2)
            target, reason = _resolve(path, spec, tracked)
            refs.append((number, "c-include", spec, target, reason))
    except Exception:  # malformed text must never raise
        return refs
    return refs
