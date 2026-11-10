"""Interactive directory listing: ``GET /browse/<root>/<path...>`` (server.py [LISTING]; the page is web/browse.html + browse.js).

Replaces the stock directory index with three fixed, capability-gated roots and three renderings of one listing:

    site   web_dir                                       dev_ui          the published assets
    math   the caller's own math-job directories         compile_submit  owner rule of math_jobs.py; native files / *.piece hidden
    docs   examples/meowdoku/*.md and deploy/*.md        dev_ui          read-only documentation

``?format=json`` (schema ``turing.listing.v1``), ``text`` (line-oriented digest for agents / narration), ``html`` (the page);
default from Accept.  A file under a root is served through the same rule.  Nothing outside a root is ever listed or served: a
path segment may not be ``..``/empty/contain ``:`` ``\\`` ``/`` NUL, and no component may be a symlink or reparse point.
``describe`` comes from declared self-descriptions only (api.yaml, abi.json, manifest.json, products.json, lowering_library rows).
The module does no I/O on the socket: ``browse`` returns ``(status, headers, body)`` for the handler to write.
"""
from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import stat
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, quote, unquote

SCHEMA = "turing.listing.v1"
HERE = Path(__file__).resolve().parent
MAX_ENTRIES = 2000
HASH_FILE_LIMIT = 8 * 1024 * 1024
HASH_BUDGET = 64 * 1024 * 1024
HASH_CACHE_SIZE = 4096
NARRATION_LIMIT = 64 * 1024
NARRATION_FILES = ("_about.md", "README.md", "ABOUT.txt")
DESCRIBE_PARSE_LIMIT = 1024 * 1024
RATE_PER_SECOND = 10.0
ROOTS = {"site": "dev_ui", "math": "compile_submit", "docs": "dev_ui"}
FORMATS = ("json", "text", "html")


class ListingError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def capability_for(root: str) -> str:
    """The capability the request must hold before anything about ``root`` is said (unknown roots: the weakest gate, then 404)."""
    return ROOTS.get(root, "dev_ui")


# ---------------------------------------------------------------- rate limit and hash cache

class RateLimiter:
    """A small in-memory token bucket per client key: RATE_PER_SECOND tokens/s, burst RATE_PER_SECOND."""

    def __init__(self, rate: float = RATE_PER_SECOND, burst: float | None = None, clock: Callable[[], float] = time.monotonic):
        self.rate, self.burst, self.clock = rate, burst or rate, clock
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = self.clock()
        with self._lock:
            tokens, then = self._buckets.get(key, (self.burst, now))
            tokens = min(self.burst, tokens + (now - then) * self.rate)
            ok = tokens >= 1.0
            self._buckets[key] = (tokens - 1.0 if ok else tokens, now)
            if len(self._buckets) > 4096:
                for stale in [k for k, (_, t) in self._buckets.items() if now - t > 60][:2048]:
                    self._buckets.pop(stale, None)
            return ok


class HashCache:
    """sha256 by (path, size, mtime_ns); LRU."""

    def __init__(self, size: int = HASH_CACHE_SIZE):
        self.size, self._data, self._lock = size, OrderedDict(), threading.Lock()

    def get(self, key):
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                return self._data[key]
        return None

    def put(self, key, value: str) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.size:
                self._data.popitem(last=False)


HASHES = HashCache()
LIMITER = RateLimiter()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------- the request's world

@dataclass
class Context:
    web_dir: Path
    docs_dir: Path = HERE
    owned_jobs: Callable[[], list] = lambda: []          # [{job_id, state}] of the caller's own math jobs
    owns_job: Callable[[str], bool] = lambda job_id: False
    job_dir: Callable[[str], Path] = lambda job_id: Path(job_id)
    served_names: Callable[[Path], set] = lambda directory: set()   # the file endpoint's rule (server._mathlab_served_names)
    lowering_rows: Callable[[], list] = lambda: []       # lowering_library.lowerings(db)
    html_page: Callable[[], bytes | None] = lambda: None
    hasher: HashCache = field(default_factory=lambda: HASHES)


# ---------------------------------------------------------------- path rules

def parse_path(rest: str) -> tuple[str, list[str]]:
    """``<root>/<path...>`` (still percent-encoded) -> (root, segments).  Raises 403 for anything that is not a plain name."""
    parts = rest.split("/")
    if parts and parts[-1] == "":
        parts = parts[:-1]                                  # one trailing slash is the directory form
    names = []
    for raw in parts:
        name = unquote(raw)
        if (not name or name in (".", "..") or any(c in name for c in ':\\/\0') or name != name.strip()
                or name.endswith(".") or any(ord(c) < 32 for c in name)):
            raise ListingError(403, "path refused")
        names.append(name)
    if not names:
        return "", []
    return names[0], names[1:]


def _is_link(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400)   # FILE_ATTRIBUTE_REPARSE_POINT


def _lstat_chain(base: Path, segs: list[str]) -> Path:
    """base/segs with every component lstat'd: a link anywhere (or a missing component) is a 404; the result stays under base."""
    current = base
    for name in segs:
        current = current / name
        try:
            info = os.lstat(current)
        except OSError:
            raise ListingError(404, "not found")
        if _is_link(info):
            raise ListingError(404, "not found")
    try:
        resolved, root = current.resolve(), base.resolve()
    except OSError:
        raise ListingError(404, "not found")
    if resolved != root and root not in resolved.parents:
        raise ListingError(404, "not found")
    return current


def _scan(directory: Path) -> list[tuple[str, str, Path]]:
    """(name, kind, path) of the plain files and directories of a real directory, names sorted; links and ``:`` names skipped."""
    out = []
    try:
        with os.scandir(directory) as scan:
            for item in scan:
                if ":" in item.name:
                    continue
                try:
                    info = item.stat(follow_symlinks=False)
                except OSError:
                    continue
                if _is_link(info):
                    continue
                if stat.S_ISDIR(info.st_mode):
                    out.append((item.name, "dir", Path(item.path)))
                elif stat.S_ISREG(info.st_mode):
                    out.append((item.name, "file", Path(item.path)))
    except OSError:
        raise ListingError(404, "not found")
    out.sort(key=lambda e: (e[1] != "dir", e[0].lower(), e[0]))
    return out


@dataclass
class Node:
    kind: str                                   # dir | file
    path: Path | None = None                    # the real directory / file (None for the virtual roots index)
    items: list = field(default_factory=list)   # dirs: [(name, kind, Path)]
    job: str | None = None                      # a math job directory
    docs: bool = False


def _docs_items(directory: Path) -> list:
    return [e for e in _scan(directory) if e[1] == "file" and e[0].lower().endswith(".md")]


def resolve(root: str, segs: list[str], ctx: Context) -> Node:
    if root not in ROOTS:
        raise ListingError(404, "no such root")
    if root == "site":
        target = _lstat_chain(ctx.web_dir, segs)
        if target.is_dir():
            return Node("dir", target, _scan(target))
        return Node("file", target)
    if root == "docs":
        if not segs:
            items = _docs_items(ctx.docs_dir)
            deploy = ctx.docs_dir / "deploy"
            if deploy.is_dir() and not _is_link(os.lstat(deploy)) and _docs_items(deploy):
                items = [("deploy", "dir", deploy)] + items
            return Node("dir", ctx.docs_dir, items, docs=True)
        base = ctx.docs_dir / "deploy" if segs[0] == "deploy" else ctx.docs_dir
        rest = segs[1:] if segs[0] == "deploy" else segs
        if base != ctx.docs_dir and not rest:
            if not base.is_dir() or _is_link(os.lstat(base)):
                raise ListingError(404, "not found")
            return Node("dir", base, _docs_items(base), docs=True)
        if len(rest) != 1 or not rest[0].lower().endswith(".md"):
            raise ListingError(404, "not found")
        return Node("file", _lstat_chain(base, rest), docs=True)
    # math: only the caller's own jobs
    if not segs:
        items = []
        for job in ctx.owned_jobs():
            directory = ctx.job_dir(job["job_id"])
            if directory.is_dir() and not _is_link(os.lstat(directory)):
                items.append((job["job_id"], "dir", directory))
        return Node("dir", None, items)
    job_id = segs[0]
    if not re.fullmatch(r"[0-9a-f]{16}", job_id) or not ctx.owns_job(job_id):
        raise ListingError(404, "not found")           # another owner's job is indistinguishable from a missing one
    directory = ctx.job_dir(job_id)
    if not directory.is_dir() or _is_link(os.lstat(directory)):
        raise ListingError(404, "not found")
    served = ctx.served_names(directory)
    if len(segs) == 1:
        return Node("dir", directory, [e for e in _scan(directory) if e[1] == "file" and e[0] in served], job=job_id)
    if len(segs) == 2 and segs[1] in served:
        return Node("file", _lstat_chain(directory, [segs[1]]), job=job_id)
    raise ListingError(404, "not found")


# ---------------------------------------------------------------- content types

_TYPES = {".wasm": "application/wasm", ".json": "application/json", ".md": "text/markdown; charset=utf-8",
          ".yaml": "text/yaml; charset=utf-8", ".yml": "text/yaml; charset=utf-8", ".wgsl": "text/plain; charset=utf-8",
          ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8", ".py": "text/plain; charset=utf-8",
          ".glsl": "text/plain; charset=utf-8", ".html": "text/html", ".htm": "text/html", ".txt": "text/plain; charset=utf-8",
          ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml", ".png": "image/png", ".jpg": "image/jpeg",
          ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp"}


def content_type(name: str) -> str:
    ext = os.path.splitext(name)[1].lower()
    if ext in _TYPES:
        return _TYPES[ext]
    guess = mimetypes.guess_type(name)[0]
    return guess or "application/octet-stream"


def _served_as(name: str) -> tuple[str, str]:
    """(content type, disposition): text and json inline, images (not svg) and wasm inline, everything else an attachment.
    html and svg are never rendered from here: html is shown as text, svg is an attachment."""
    ctype = content_type(name)
    base = ctype.split(";")[0]
    if base == "text/html":
        return "text/plain; charset=utf-8", "inline"
    if base == "image/svg+xml":
        return ctype, "attachment"
    if base.startswith("text/") or base in ("application/json", "application/wasm") or (base.startswith("image/")):
        return ctype, "inline"
    return ctype, "attachment"


# ---------------------------------------------------------------- narration

def _read_text(path: Path, limit: int) -> str | None:
    try:
        if os.path.getsize(path) > limit:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


_FENCE = re.compile(r"```.*?```", re.S)


def speak_text(markdown: str) -> str:
    """Markdown stripped to plain sentences a voice can read: no fences, tables, markers, links or emphasis."""
    text = _FENCE.sub(" ", markdown)
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("|") or re.fullmatch(r"[-=_*\s]{3,}", line):
            continue
        line = re.sub(r"^#{1,6}\s*", "", line)
        line = re.sub(r"^([-*+]|\d+[.)])\s+", "", line)
        line = re.sub(r"^>\s?", "", line)
        line = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", line)
        line = re.sub(r"[`*_~]+", "", line)
        lines.append(line.rstrip(":") + ("" if line[-1:] in ".!?:" else "."))
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def _narrate_dir(directory: Path) -> dict | None:
    for name in NARRATION_FILES:
        path = directory / name
        try:
            info = os.lstat(path)
        except OSError:
            continue
        if _is_link(info) or not stat.S_ISREG(info.st_mode):
            continue
        text = _read_text(path, NARRATION_LIMIT)
        if text is not None:
            return {"source": name, "text": text, "speak": speak_text(text)}
    return None


def _narrate_job(directory: Path) -> dict | None:
    products = _json_file(directory / "products.json")
    law = _read_text(directory / "law.math.txt", NARRATION_LIMIT)
    if not isinstance(products, dict) and law is None:
        return None
    lines = []
    products = products if isinstance(products, dict) else {}
    lines.append(f"# {products.get('name') or 'Math job'}")
    equations = products.get("equations") or ([ln for ln in (law or "").splitlines() if ln.strip()] if law else [])
    if equations:
        lines += ["", "Equations:"] + [f"- `{eq}`" for eq in equations]
    if products.get("inputs") or products.get("outputs"):
        lines += ["", f"Inputs: {', '.join(map(str, products.get('inputs') or [])) or 'none'}. "
                      f"Outputs: {', '.join(map(str, products.get('outputs') or [])) or 'none'}."]
    targets = products.get("targets") or {}
    symbolic = products.get("symbolic")
    states = ([("symbolic", symbolic)] if isinstance(symbolic, dict) else []) + [(n, t) for n, t in targets.items()]
    if states:
        lines += ["", "Targets:"]
        for name, record in states:
            record = record if isinstance(record, dict) else {}
            seconds = record.get("seconds")
            lines.append(f"- {name}: {record.get('state', 'unknown')}" + (f" in {seconds} seconds" if isinstance(seconds, (int, float)) else "")
                         + (f" ({record['error']})" if record.get("error") else ""))
    text = "\n".join(lines)
    return {"source": "generated:products.json+law.math.txt", "text": text, "speak": speak_text(text)}


# ---------------------------------------------------------------- describe (declared self-descriptions only)

def _json_file(path: Path):
    text = _read_text(path, DESCRIBE_PARSE_LIMIT)
    if text is None:
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


def _names(value, limit: int = 64):
    if isinstance(value, dict):
        value = list(value)
    if not isinstance(value, list):
        return None
    out = []
    for item in value[:limit]:
        out.append(item.get("name") if isinstance(item, dict) and "name" in item else item if isinstance(item, (str, int, float)) else str(item))
    return out


def _describe_file(name: str, path: Path) -> dict | None:
    low = name.lower()
    if low.endswith(".api.yaml"):
        text = _read_text(path, DESCRIBE_PARSE_LIMIT)
        try:
            import yaml
            doc = yaml.safe_load(text) if text is not None else None
        except Exception:
            doc = None
        if not isinstance(doc, dict):
            return None
        points = doc.get("entry_points")
        return {"kind": "api", "schema": doc.get("schema"), "module": doc.get("module"), "language": doc.get("language"),
                "entry": doc.get("entry"),
                "entry_points": [p.get("name") for p in points if isinstance(p, dict)] if isinstance(points, list) else None}
    if low.endswith(".abi.json"):
        doc = _json_file(path)
        if not isinstance(doc, dict):
            return None
        return {"kind": "abi", "schema": doc.get("schema"),
                "inputs": _names(doc.get("inputs", doc.get("arguments"))), "outputs": _names(doc.get("outputs"))}
    if low.endswith(".manifest.json"):
        doc = _json_file(path)
        if not isinstance(doc, dict):
            return None
        sizes = doc.get("sizes")
        return {"kind": "manifest", "schema": doc.get("schema"), "sizes": sorted(map(str, sizes))[:64] if isinstance(sizes, dict) else None}
    if low == "products.json":
        doc = _json_file(path)
        if not isinstance(doc, dict):
            return None
        targets = doc.get("targets") if isinstance(doc.get("targets"), dict) else {}
        return {"kind": "products", "schema": doc.get("schema"), "inputs": _names(doc.get("inputs")), "outputs": _names(doc.get("outputs")),
                "targets": {n: (t.get("state") if isinstance(t, dict) else None) for n, t in targets.items()}}
    if low.endswith(".md"):
        text = _read_text(path, DESCRIBE_PARSE_LIMIT)
        return _describe_markdown(text) if text is not None else None
    return None


def _describe_markdown(text: str) -> dict:
    heading, paragraph, para = None, [], []
    for line in text.splitlines():
        s = line.strip()
        if heading is None and s.startswith("#"):
            heading = s.lstrip("#").strip()
            continue
        if heading is None and s:
            heading = ""                                   # text before any heading: no heading
        if s.startswith("#") or s.startswith("```"):
            if para:
                break
            continue
        if s:
            para.append(s)
        elif para:
            break
    paragraph = re.sub(r"[`*_]+", "", " ".join(para))
    return {"kind": "document", "heading": heading or None, "summary": paragraph[:300] or None}


def _lowering_map(ctx: Context) -> dict:
    try:
        rows = ctx.lowering_rows()
    except Exception:
        return {}
    out: dict = {}
    for row in rows:
        sha = row.get("product_sha256")
        if sha:
            out.setdefault(sha, row)
    return out


# ---------------------------------------------------------------- the listing

def _iso(timestamp: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp))


def _href(root: str, segs: list[str], trailing: bool = False) -> str:
    return "/browse/" + "/".join(quote(s, safe="") for s in ([root] + segs if root else [])) + ("/" if trailing and (root or segs) else "")


def _caps_required(root: str, segs: list[str]) -> list[str]:
    return [capability_for(root)] if root else ["dev_ui"]


def build_listing(root: str, segs: list[str], node: Node, ctx: Context) -> dict:
    entries, hashed, deferred_any, lowerings = [], 0, False, None
    items = node.items[:MAX_ENTRIES]
    for name, kind, path in items:
        child = segs + [name]
        entry = {"name": name, "kind": kind, "bytes": None, "mtime": None, "content_type": None, "sha256": None,
                 "href": _href(root, child, kind == "dir"), "describe": None}
        try:
            info = os.stat(path)
        except OSError:
            continue
        entry["mtime"] = _iso(info.st_mtime)
        if kind == "file":
            entry["bytes"] = info.st_size
            entry["content_type"] = content_type(name).split(";")[0]
            if info.st_size <= HASH_FILE_LIMIT:
                key = (str(path), info.st_size, info.st_mtime_ns)
                sha = ctx.hasher.get(key)
                if sha is None:
                    if hashed + info.st_size > HASH_BUDGET:
                        entry["hash_deferred"] = True
                        deferred_any = True
                    else:
                        try:
                            sha = _sha256_file(path)
                        except OSError:
                            sha = None
                        else:
                            hashed += info.st_size
                            ctx.hasher.put(key, sha)
                entry["sha256"] = sha
            describe = _describe_file(name, path)
            if entry["sha256"]:
                if lowerings is None:
                    lowerings = _lowering_map(ctx)
                row = lowerings.get(entry["sha256"])
                if row:
                    describe = dict(describe or {})
                    describe["lowering"] = {k: row.get(k) for k in ("core_id", "backend", "entry", "verdict", "abi_schema", "api_path")}
            entry["describe"] = describe
        entries.append(entry)
    if node.job:
        narration = _narrate_job(node.path)
    elif node.path is not None and root != "math":
        narration = _narrate_dir(node.path)
    else:
        narration = None
    parents = [{"name": "browse", "href": "/browse/"}]
    if root and segs:                                       # the current directory is the page's last crumb, never its own parent
        parents.append({"name": root, "href": _href(root, [], True)})
        for i, name in enumerate(segs[:-1] if segs else []):
            parents.append({"name": name, "href": _href(root, segs[: i + 1], True)})
    return {"schema": SCHEMA, "root": root, "path": "/".join(segs), "narration": narration, "entries": entries,
            "parents": parents, "capabilities_required": _caps_required(root, segs), "generated": _iso(time.time()),
            "truncated": len(node.items) > MAX_ENTRIES, "hash_deferred": deferred_any}


def _roots_index(caps: frozenset) -> dict:
    entries = [{"name": r, "kind": "dir", "bytes": None, "mtime": None, "content_type": None, "sha256": None,
                "href": _href(r, [], True), "describe": {"kind": "root", "capability": cap}}
               for r, cap in ROOTS.items() if cap in caps]
    return {"schema": SCHEMA, "root": "", "path": "", "narration": None, "entries": entries, "parents": [{"name": "browse", "href": "/browse/"}],
            "capabilities_required": ["dev_ui"], "generated": _iso(time.time()), "truncated": False, "hash_deferred": False}


def _oneline(describe) -> str:
    if not describe:
        return "-"
    text = json.dumps({k: v for k, v in describe.items() if v is not None}, separators=(",", ":"), default=str, ensure_ascii=False)
    return text[:240]


def _clean(value) -> str:
    return re.sub(r"[\t\r\n]+", " ", str(value))


def render_text(doc: dict) -> str:
    """Stable, line-oriented: ``# <root>/<path>``, narration, one tab-separated line per entry, then ``next:`` URLs of subdirectories."""
    lines = ["# " + _clean((doc["root"] + "/" + doc["path"]).strip("/") or "browse")]
    narration = doc.get("narration")
    if narration:
        lines += ["", narration["text"].rstrip(), ""]
    for e in doc["entries"]:
        sha = (e["sha256"] or "-")[:12] if e["kind"] == "file" else "-"
        lines.append("\t".join([e["kind"], "-" if e["bytes"] is None else str(e["bytes"]), sha, _clean(e["name"]), _clean(_oneline(e["describe"]))]))
    if doc.get("truncated"):
        lines.append(f"truncated\t{MAX_ENTRIES}")
    lines += ["", "next:"]
    for e in doc["entries"]:
        if e["kind"] == "dir":
            lines.append(f"json\t{e['href']}?format=json")
            lines.append(f"text\t{e['href']}?format=text")
    return "\n".join(lines) + "\n"


def negotiate(query: dict, accept: str) -> str:
    wanted = (query.get("format") or [""])[0].lower()
    if wanted:
        if wanted not in FORMATS:
            raise ListingError(400, f"format must be one of {', '.join(FORMATS)}")
        return wanted
    best, where = "html", len(accept) + 1
    for fmt, media in (("json", "application/json"), ("text", "text/plain"), ("html", "text/html")):
        at = accept.find(media)
        if 0 <= at < where:
            best, where = fmt, at
    return best


# ---------------------------------------------------------------- the response

SECURITY = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}
HTML_CSP = ("default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
            "connect-src 'self'; media-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")


def _respond(status: int, ctype: str, body: bytes, extra: dict | None = None) -> tuple[int, dict, bytes]:
    headers = {"Content-Type": ctype, "Content-Length": str(len(body)), **SECURITY, **(extra or {})}
    return status, headers, body


def _error(error: ListingError) -> tuple[int, dict, bytes]:
    return _respond(error.status, "application/json", json.dumps({"error": error.message}).encode("utf-8"))


def browse(path_and_query: str, accept: str, client: str, caps: frozenset, ctx: Context, limiter: RateLimiter | None = None):
    """The whole GET /browse/... request: (status, headers, body) or, for a file, (200, headers, Path) to be streamed."""
    try:
        path, _, query = path_and_query.partition("?")
        rest = path[len("/browse/"):] if path.startswith("/browse/") else ""
        root, segs = parse_path(rest)
        params = parse_qs(query)
        if root and capability_for(root) not in caps:
            raise ListingError(403, f"requires {capability_for(root)}")
        if not root:
            node = None
        else:
            node = resolve(root, segs, ctx)
            if node.kind == "file":
                ctype, disposition = _served_as(segs[-1])
                headers = {"Content-Type": ctype, "Content-Length": str(os.path.getsize(node.path)), **SECURITY,
                           "Content-Security-Policy": "sandbox; default-src 'none'; style-src 'unsafe-inline'; img-src 'self'",
                           "Content-Disposition": f'{disposition}; filename="{segs[-1]}"' if disposition == "attachment" else "inline"}
                return 200, headers, node.path
        if not (limiter or LIMITER).allow(client):
            return _respond(429, "application/json", b'{"error": "listing rate limit: 10 per second"}', {"Retry-After": "1"})
        fmt = negotiate(params, accept or "")
        if fmt == "html":
            page = ctx.html_page()
            if page is None:
                raise ListingError(404, "browse page not installed")
            return _respond(200, "text/html; charset=utf-8", page, {"Content-Security-Policy": HTML_CSP})
        doc = _roots_index(caps) if node is None else build_listing(root, segs, node, ctx)
        if fmt == "json":
            return _respond(200, "application/json", json.dumps(doc).encode("utf-8"))
        return _respond(200, "text/plain; charset=utf-8", render_text(doc).encode("utf-8"))
    except ListingError as error:
        return _error(error)
