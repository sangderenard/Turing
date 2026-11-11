"""JavaScript dynamic-reference plugin for the repo world scan (pure, deterministic).

Only what the static-import extractor misses, from string-literal arguments.
"""

from __future__ import annotations

import posixpath
import re

LANGUAGE = "javascript"
SUFFIXES = (".js", ".mjs", ".cjs")

_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*:")
_JS_SUFFIXES = ("", ".js", ".mjs", "/index.js")
_ASSET_EXT = (".wasm", ".json", ".wgsl", ".glsl", ".bin")


def _normalise(directory, rel):
    parts = []
    for part in (directory + "/" + rel if directory else rel).split("/"):
        if part == "..":
            if not parts:
                return None
            parts.pop()
        elif part not in ("", "."):
            parts.append(part)
    return "/".join(parts)


def _resolve(path, spec, tracked, allow_plain):
    if _SCHEME.match(spec) or spec.startswith("//"):
        return None, "bare specifier"
    relative = spec.startswith(("./", "../"))
    served = spec.startswith("/")
    if not (relative or served or allow_plain):
        return None, "bare specifier"
    stem = _normalise(posixpath.dirname(path), spec.lstrip("/") if served else spec)
    if stem is None:
        return None, "escapes the repository root"
    for suffix in _JS_SUFFIXES:
        if stem + suffix in tracked:
            return stem + suffix, ""
    if served:
        return None, "served-root path; not beside the page"
    return None, "not tracked"


def _tokens(text):
    out = []  # (kind, value, line); kinds: s string, t template, i ident, p punct
    i, n, line, prev = 0, len(text), 1, ""
    while i < n:
        c = text[i]
        if c == "\n":
            line += 1; i += 1; continue
        if c in " \t\r":
            i += 1; continue
        if text.startswith("//", i):
            while i < n and text[i] != "\n":
                i += 1
            continue
        if text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            line += text.count("\n", i, j); i = j; continue
        if c in "\"'":
            j, buf = i + 1, []
            while j < n and text[j] != c and text[j] != "\n":
                if text[j] == "\\" and j + 1 < n:
                    j += 1
                buf.append(text[j]); j += 1
            out.append(("s", "".join(buf), line)); i = j + 1; prev = "s"; continue
        if c == "`":
            j, depth, start = i + 1, 0, line
            while j < n:
                if text[j] == "\\":
                    j += 2; continue
                if text[j] == "\n":
                    line += 1
                if depth == 0 and text[j] == "`":
                    break
                if text.startswith("${", j):
                    depth += 1; j += 2; continue
                if depth and text[j] == "}":
                    depth -= 1
                j += 1
            out.append(("t", "", start)); i = j + 1; prev = "s"; continue
        if c == "/" and prev not in ("i", "s", ")", "]", "}"):
            j, cls = i + 1, False
            while j < n and text[j] != "\n":
                if text[j] == "\\":
                    j += 2; continue
                if text[j] == "[":
                    cls = True
                elif text[j] == "]":
                    cls = False
                elif text[j] == "/" and not cls:
                    break
                j += 1
            i = j + 1; prev = "s"; continue
        if c.isalnum() or c in "_$":
            j = i
            while j < n and (text[j].isalnum() or text[j] in "_$"):
                j += 1
            out.append(("i", text[i:j], line)); i = j; prev = "i"; continue
        out.append(("p", c, line)); i += 1; prev = c
    return out


def _is(tok, kind, val=None):
    return tok is not None and tok[0] == kind and (val is None or tok[1] == val)


def _scan(toks):
    """Yield (line, kind, specifier, allow_plain)."""
    n = len(toks)
    at = lambda k: toks[k] if 0 <= k < n else None
    consumed = set()

    def url_call(k):
        # toks[k] is `URL`; matches URL ( "x" , import . meta . url )
        if (_is(at(k + 1), "p", "(") and _is(at(k + 2), "s") and _is(at(k + 3), "p", ",")
                and _is(at(k + 4), "i", "import") and _is(at(k + 5), "p", ".")
                and _is(at(k + 6), "i", "meta") and _is(at(k + 7), "p", ".")
                and _is(at(k + 8), "i", "url")):
            return at(k + 2)
        return None

    for k, tok in enumerate(toks):
        if tok[0] != "i" or _is(at(k - 1), "p", "."):
            continue
        v, line = tok[1], tok[2]
        if v == "new" and _is(at(k + 1), "i") and at(k + 1)[1] in ("Worker", "SharedWorker") \
                and _is(at(k + 2), "p", "("):
            if _is(at(k + 3), "s"):
                yield (at(k + 3)[2], "js-worker", at(k + 3)[1], True)
            elif _is(at(k + 3), "i", "new") and _is(at(k + 4), "i", "URL"):
                lit = url_call(k + 4)
                if lit:
                    consumed.add(k + 4)
                    yield (lit[2], "js-worker", lit[1], True)
        elif v == "new" and _is(at(k + 1), "i", "URL") and (k + 1) not in consumed:
            lit = url_call(k + 1)
            if lit:
                consumed.add(k + 1)
                yield (lit[2], "js-url", lit[1], True)
        elif v == "import" and _is(at(k + 1), "p", "(") and _is(at(k + 2), "s"):
            yield (at(k + 2)[2], "js-dynamic-import", at(k + 2)[1], False)
        elif v == "importScripts" and _is(at(k + 1), "p", "("):
            j = k + 2
            while _is(at(j), "s"):
                yield (at(j)[2], "js-import-scripts", at(j)[1], True)
                j += 1
                if _is(at(j), "p", ","):
                    j += 1
                else:
                    break
        elif v == "require" and _is(at(k + 1), "p", "(") and _is(at(k + 2), "s"):
            yield (at(k + 2)[2], "js-require", at(k + 2)[1], False)
        elif v == "fetch" and _is(at(k + 1), "p", "(") and _is(at(k + 2), "s"):
            spec = at(k + 2)[1]
            bare = spec.split("#", 1)[0].split("?", 1)[0]
            if bare.lower().endswith(_ASSET_EXT) and not _SCHEME.match(bare) \
                    and not bare.startswith("//"):
                yield (at(k + 2)[2], "js-fetch-asset", spec, True)


def references(path: str, text: str, tracked: frozenset[str]):
    out = []
    try:
        for line, kind, spec, plain in _scan(_tokens(text)):
            clean = spec
            if kind in ("js-worker", "js-url", "js-fetch-asset", "js-import-scripts"):
                clean = spec.split("#", 1)[0].split("?", 1)[0]
            if not clean:
                continue
            target, reason = _resolve(path, clean, tracked, plain)
            out.append((line, kind, spec, target, reason))
    except Exception:
        pass
    return sorted(out, key=lambda r: r[0])
