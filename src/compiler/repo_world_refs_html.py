"""HTML reference plugin for the repo world scan (pure, deterministic)."""

from __future__ import annotations

from html.parser import HTMLParser
import posixpath
import re

LANGUAGE = "html"
SUFFIXES = (".html", ".htm")

_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*:")
_LINK_RELS = {"stylesheet", "modulepreload", "icon", "manifest", "apple-touch-icon"}
_MEDIA = {"img", "audio", "video", "source"}
_JS_SUFFIXES = ("", ".js", ".mjs", "/index.js")


def _normalise(directory: str, rel: str) -> str | None:
    parts: list[str] = []
    for part in (directory + "/" + rel if directory else rel).split("/"):
        if part == "..":
            if not parts:
                return None
            parts.pop()
        elif part not in ("", "."):
            parts.append(part)
    return "/".join(parts)


def _resolve(path, spec, tracked, suffixes=("",)):
    directory = posixpath.dirname(path)
    served = spec.startswith("/")
    rel = spec.lstrip("/") if served else spec
    stem = _normalise(directory, rel)
    if stem is None:
        return None, "escapes the repository root"
    for suffix in suffixes:
        if stem + suffix in tracked:
            return stem + suffix, ""
    if served:
        return None, "served-root path; not beside the page"
    return None, "not tracked"


def _clean(value):
    if value is None:
        return None
    value = value.strip().split("#", 1)[0].split("?", 1)[0]
    if not value or _SCHEME.match(value) or value.startswith("//"):
        return None
    return value


# --- minimal JS tokenizer for inline module bodies ---------------------------

def _tokens(text):
    out = []  # (kind, value, line)
    i, n, line = 0, len(text), 1
    prev = ""
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


def _inline_specs(body):
    toks = _tokens(body)
    found = []
    for k, (kind, val, line) in enumerate(toks):
        nxt = toks[k + 1] if k + 1 < len(toks) else None
        nxt2 = toks[k + 2] if k + 2 < len(toks) else None
        before = toks[k - 1] if k else None
        if kind != "i" or nxt is None:
            continue
        if before and before[1] == ".":
            continue
        if val == "import":
            if nxt[0] == "s":
                found.append((nxt[2], nxt[1]))
            elif nxt[1] == "(" and nxt2 and nxt2[0] == "s":
                found.append((nxt2[2], nxt2[1]))
        elif val == "from" and nxt[0] == "s":
            found.append((nxt[2], nxt[1]))
    return found


class _Collector(HTMLParser):
    def __init__(self, path, tracked):
        super().__init__(convert_charrefs=True)
        self.path, self.tracked = path, tracked
        self.refs = []
        self._inline = None  # [start_line, [chunks]]

    def _add(self, line, kind, spec, suffixes=("",), keep_unresolved=True):
        cleaned = _clean(spec)
        if cleaned is None:
            return
        target, reason = _resolve(self.path, cleaned, self.tracked, suffixes)
        if target is None and not keep_unresolved:
            return
        self.refs.append((line, kind, cleaned, target, reason))

    def handle_starttag(self, tag, attrs):
        a = {k.lower(): v for k, v in attrs}
        line = self.getpos()[0]
        if tag == "script":
            if a.get("src"):
                self._add(line, "html-script", a["src"])
            elif (a.get("type") or "").strip().lower() == "module":
                self._inline = [line, []]
        elif tag == "link":
            rels = set((a.get("rel") or "").lower().split())
            if rels & _LINK_RELS:
                self._add(line, "html-link", a.get("href"))
        elif tag in _MEDIA:
            self._add(line, "html-media", a.get("src"))
        elif tag == "a":
            self._add(line, "html-anchor", a.get("href"), keep_unresolved=False)
        elif tag == "iframe":
            self._add(line, "html-frame", a.get("src"))

    def handle_data(self, data):
        if self._inline is not None:
            if not self._inline[1]:
                self._inline[0] = self.getpos()[0]
            self._inline[1].append(data)

    def handle_endtag(self, tag):
        if tag == "script" and self._inline is not None:
            start, chunks = self._inline
            self._inline = None
            for line, spec in _inline_specs("".join(chunks)):
                at = start + line - 1
                if spec.startswith(("./", "../", "/")):
                    self._add(at, "html-inline-import", spec, _JS_SUFFIXES)
                elif _clean(spec) is not None:
                    self.refs.append((at, "html-inline-import", spec, None, "bare specifier"))


def references(path: str, text: str, tracked: frozenset[str]):
    collector = _Collector(path, tracked)
    try:
        collector.feed(text.replace("\r\n", "\n").replace("\r", "\n"))
        collector.close()
    except Exception:
        pass
    return sorted(collector.refs, key=lambda r: r[0])
