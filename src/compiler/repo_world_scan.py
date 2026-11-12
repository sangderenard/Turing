"""Scan a clean git checkout into neutral repository world-map records.

This is the producer for the repository world map described in
``docs/REPO_WORLD_MAP_DESIGN.md``.  It knows nothing about HTML, geometry,
or the Living Data Map page.  Given the path of a clone it emits a plain
record set:

    regions   one per tracked directory (the repository root included);
              a directory's parent region is its parent directory
    rooms     one per tracked file
    portals   one PAIR per resolved cross-file reference: an ``out`` end on
              the referencing file's room and an ``in`` end on the referenced
              file's room, with the source line that made the reference

Two reference extractors are built in: Python imports read with :mod:`ast`,
and JavaScript ES-module static imports (``import … from``, bare
``import "…"``, ``export … from``) read line by line.  More are plugins: a
module-like object with ``LANGUAGE: str``, ``SUFFIXES: tuple[str, ...]``,
optional ``FILENAMES: tuple[str, ...]`` and
``references(path, text, tracked) -> [(line, kind, specifier, target|None,
reason)]``.  The plugins are named in ``EXTRACTOR_MODULES`` (an entry that
does not import is a :class:`RepoScanError`) or passed as ``extractors=``.
Every extractor matching a file runs, in addition to the built-ins.  A plugin
that raises, or returns a target that is not a tracked file, is a scan error.
A reference whose target is not a tracked file is kept as an *unresolved
reference* so nothing disappears silently; it is not a portal because a
portal needs two rooms.

Portal pair identity keys (source, line, target).  Two references of different
kinds on one line to one target are therefore ONE pair; the first in sorted
order (source, line, kind, specifier) is kept.

Import roots: Python absolute names resolve against the repository root and
each declared import root (a tracked directory, repo-relative POSIX).  Roots
are an input, never guessed from directory names: the ``import_roots``
argument, CLI ``--import-root DIR`` (repeatable), and
:func:`declared_import_roots`, which reads only declarations inside the tree:
pyproject.toml ``[tool.pytest.ini_options] pythonpath``,
``[tool.setuptools.packages.find] where``, ``[tool.setuptools] package-dir``
values; ``pythonpath`` in pytest.ini / setup.cfg (``[tool:pytest]``) /
tox.ini; and a repo-owned ``repo_world.toml`` with ``import_roots = ["dir"]``.
The earliest root wins when a name resolves under several; a scan note counts
those ambiguities.  Relative imports resolve from the file's own directory.
The stream lanes read the declarations from the stream itself
(``import_roots=None`` means "the declared roots").

A tracked ``.dream`` document is a room like any file, and every prose
stand-in inside it (``language = prose``, see ``DreamDocument.work_table``)
mints one more room: a *claimed room*, owned by the stand-in's author before
any content exists, ``open`` until a block in the document satisfies it.
That is the room system's side of the owner's work table.

Every identity is keyed by the commit hash, so two scans of the same commit
produce identical records and a scan of a different commit never collides.
The scanner refuses a tree with uncommitted or untracked changes: the record
set describes a commit, never a working state.

The sanctioned input is a GitHub repository, not the owner's local tree: the
local scratch checkout is messy and holds things that must not be exposed.
The source is a transient intermediary that passes through one operation
(this scan) and is never kept, so it never touches the disk:
:func:`scan_github_repository` streams the commit's tarball from the GitHub
API through :mod:`tarfile`, keeping in memory only the paths and the text of
the files a reader links (Python, JavaScript, dream documents);
:func:`scan_remote_repository` does the same for any other git URL through
``git archive --remote``.  Only what the remote serves can reach the record
set.  The record set itself is returned to the caller; what the caller does
with it is the call's own declaration (``--json`` to stdout, ``--out`` to a
path the call names, or a user-space store the service owns).

    python -m src.compiler.repo_world_scan owner/repo [--ref main] [--import-root DIR ...] [--json | --out <file>]
    python -m src.compiler.repo_world_scan --path <checkout> [--import-root DIR ...] [--json | --out <file>]
"""

from __future__ import annotations

import ast
import configparser
from dataclasses import dataclass, field
import importlib
import json
from pathlib import Path, PurePosixPath
import posixpath
import re
import subprocess
import sys
import tarfile
import tomllib
from typing import Any, Callable, Iterable, Mapping, Sequence
import urllib.error
import urllib.request

from .dream_document import (
    DREAM_LANGUAGE_TRANSLATIONS, DreamDocumentError, parse_dream_document,
)

REPO_WORLD_SCAN_VERSION = "repo-world-scan-v1"

# Fully qualified names of extractor plugin modules (contract in the module
# docstring).  An entry that fails to import is a RepoScanError, never skipped.
EXTRACTOR_MODULES: tuple[str, ...] = (
    "src.compiler.repo_world_refs_c",
    "src.compiler.repo_world_refs_cmake",
    "src.compiler.repo_world_refs_html",
    "src.compiler.repo_world_refs_js_dynamic",
    "src.compiler.repo_world_refs_powershell",
    "src.compiler.repo_world_refs_shell",
)

# Files whose declarations name Python import roots; read from the tree
# (on disk or in the stream) and never from anywhere else.
DECLARATION_FILES: tuple[str, ...] = (
    "pyproject.toml", "repo_world.toml", "pytest.ini", "setup.cfg", "tox.ini",
)
GITHUB_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
GITHUB_CLONE_URL = re.compile(
    r"^https://github\.com/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+?)(?:\.git)?/?$"
)
GITHUB_API = "https://api.github.com"
_USER_AGENT = "turing-repo-world-scan"

# The known languages are the compiler's own table, DREAM_LANGUAGE_TRANSLATIONS
# (each with its translation route).  This module adds two things that table
# does not carry: which file suffixes name each language, and which languages
# have a cross-file reference reader ("linking") here.  A language in the table
# without a reader is known and translatable but its rooms get no portals yet;
# that is declared per language in RepoWorldScan.languages, never implied.
LANGUAGE_SUFFIXES: Mapping[str, tuple[str, ...]] = {
    "python": (".py",),
    "javascript": (".js", ".mjs"),
    "cpp": (".c", ".h", ".cc", ".cpp", ".cxx", ".hh", ".hpp", ".hxx"),
    "glsl": (".glsl", ".vert", ".frag", ".geom", ".comp", ".tesc", ".tese"),
    "wgsl": (".wgsl",),
    # "sympy" is a Python-hosted language with no file suffix of its own.
}
# A dream document is a container of language blocks, not a language; its
# room keeps language "dream-document" and its stand-ins become claimed rooms.
DOCUMENT_SUFFIXES: Mapping[str, tuple[str, ...]] = {
    "dream-document": (".dream",),
}
LINKING_READERS: Mapping[str, str] = {
    "python": "python-import",        # ast.Import / ast.ImportFrom
    "javascript": "es-module-import",  # static ES-module import/export-from
}
# Plugin extractors add readers for more languages; language_table() reports
# them per scan from the extractors actually in use.
_SUFFIX_LANGUAGE = {
    suffix: language for language, suffixes in LANGUAGE_SUFFIXES.items() for suffix in suffixes
}
_SUFFIX_LANGUAGE.update({
    suffix: name for name, suffixes in DOCUMENT_SUFFIXES.items() for suffix in suffixes
})
_unknown_to_table = set(LANGUAGE_SUFFIXES) - set(DREAM_LANGUAGE_TRANSLATIONS)
if _unknown_to_table:
    raise RuntimeError(
        f"LANGUAGE_SUFFIXES names languages the compiler table does not: {sorted(_unknown_to_table)}"
    )

_JS_RESOLUTION_SUFFIXES = ("", ".js", ".mjs", "/index.js", "/index.mjs")

# ES-module static import forms.  The specifier is always the last string
# literal on the statement line, so one pattern per form is enough.
_JS_IMPORT_PATTERNS = (
    # import x from "spec" / import {a} from "spec" / import * as ns from "spec"
    re.compile(r"""^\s*import\s+(?:[^'";]*?)\s+from\s+(['"])([^'"]+)\1"""),
    # import "spec"
    re.compile(r"""^\s*import\s+(['"])([^'"]+)\1"""),
    # export {a} from "spec" / export * from "spec"
    re.compile(r"""^\s*export\s+(?:[^'";]*?)\s+from\s+(['"])([^'"]+)\1"""),
)


class RepoScanError(RuntimeError):
    """The path is not a scannable repository clone."""


class DirtyCloneError(RepoScanError):
    """The checkout has uncommitted or untracked changes."""


@dataclass(frozen=True, slots=True)
class RepoRegion:
    """A directory.  ``parent`` is None only for the repository root."""

    identity: str
    path: str
    name: str
    parent: str | None
    depth: int

    def to_data(self) -> dict[str, Any]:
        return {
            "identity": self.identity, "kind": "region", "path": self.path,
            "name": self.name, "parent": self.parent, "depth": self.depth,
            "dependencies": ([] if self.parent is None else [
                {"relationship": "contained-by", "target": self.parent},
            ]),
        }


@dataclass(frozen=True, slots=True)
class RepoRoom:
    """A tracked file inside exactly one region."""

    identity: str
    path: str
    name: str
    region: str
    language: str

    def to_data(self) -> dict[str, Any]:
        return {
            "identity": self.identity, "kind": "room", "path": self.path,
            "name": self.name, "region": self.region, "language": self.language,
            "dependencies": [{"relationship": "contained-by", "target": self.region}],
        }


@dataclass(frozen=True, slots=True)
class RepoPortalEnd:
    """One end of a portal pair, owned by one room's boundary.

    ``transmission`` is the declared, empty slot that a later consumer (the
    particle simulation, the studio's shaders or ray tracers) fills with
    optical or transport properties.  The scanner never writes it.
    """

    identity: str
    room: str
    role: str  # "out" on the referencing room, "in" on the referenced room
    pair: str
    transmission: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.role not in {"in", "out"}:
            raise ValueError(f"portal end role must be 'in' or 'out', not {self.role!r}")

    def to_data(self) -> dict[str, Any]:
        return {
            "identity": self.identity, "kind": "portal-end", "room": self.room,
            "role": self.role, "pair": self.pair,
            "transmission": None if self.transmission is None else dict(self.transmission),
            "dependencies": [
                {"relationship": "on-boundary-of", "target": self.room},
                {"relationship": "end-of", "target": self.pair},
            ],
        }


@dataclass(frozen=True, slots=True)
class RepoPortalPair:
    """A directed cross-file reference, realized as two portal ends."""

    identity: str
    source_room: str
    target_room: str
    source_line: int
    reference_kind: str  # "python-import" | "es-module-import"
    specifier: str       # the text the source used to name the target
    out_end: RepoPortalEnd
    in_end: RepoPortalEnd

    def to_data(self) -> dict[str, Any]:
        return {
            "identity": self.identity, "kind": "portal-pair",
            "source_room": self.source_room, "target_room": self.target_room,
            "direction": "source-to-target", "source_line": self.source_line,
            "reference_kind": self.reference_kind, "specifier": self.specifier,
            "ends": [self.out_end.to_data(), self.in_end.to_data()],
        }


@dataclass(frozen=True, slots=True)
class RepoRoomClaim:
    """A room claimed by a stand-in before its content exists.

    Minted from one prose block of a tracked ``.dream`` document.  The claim
    is contained by the document's region beside the document's own room and
    says who owns it (``owner``: the first code contributor once satisfied,
    else the describer) and whether the stand-in is still ``open``.  The room
    is unaltered: the scanner writes no content, geometry or portals for it.
    """

    identity: str
    document_room: str
    region: str
    stand_in: str          # the prose block's id inside the document
    content_hash: str      # the description's sha256, the card cache_key
    state: str             # "open" | "satisfied"
    owner: str | None
    described_by: str | None
    contributions: tuple[Mapping[str, Any], ...] = ()

    def to_data(self) -> dict[str, Any]:
        return {
            "identity": self.identity, "kind": "claimed-room",
            "document_room": self.document_room, "region": self.region,
            "stand_in": self.stand_in, "content_hash": self.content_hash,
            "state": self.state, "owner": self.owner, "described_by": self.described_by,
            "contributions": [dict(item) for item in self.contributions],
            "dependencies": [
                {"relationship": "contained-by", "target": self.region},
                {"relationship": "claimed-from", "target": self.document_room},
            ],
        }


@dataclass(frozen=True, slots=True)
class RepoUnresolvedReference:
    """A reference whose target is not a tracked file (stdlib, package, typo)."""

    source_room: str
    source_line: int
    reference_kind: str
    specifier: str
    reason: str

    def to_data(self) -> dict[str, Any]:
        return {
            "kind": "unresolved-reference", "source_room": self.source_room,
            "source_line": self.source_line, "reference_kind": self.reference_kind,
            "specifier": self.specifier, "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class RepoWorldScan:
    commit: str
    origin: str | None
    regions: tuple[RepoRegion, ...]
    rooms: tuple[RepoRoom, ...]
    portals: tuple[RepoPortalPair, ...]
    unresolved: tuple[RepoUnresolvedReference, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)
    claims: tuple[RepoRoomClaim, ...] = ()
    extractor_languages: tuple[str, ...] = ("javascript", "python")
    import_roots: tuple[str, ...] = ("",)
    languages: tuple[Mapping[str, Any], ...] = ()

    @property
    def root_region(self) -> RepoRegion:
        return self.regions[0]

    def to_data(self) -> dict[str, Any]:
        return {
            "schema": REPO_WORLD_SCAN_VERSION,
            "identity": f"repo:{self.commit}",
            "commit": self.commit,
            "origin": self.origin,
            "languages": [dict(row) for row in self.languages] or language_table(),
            "extractor_languages": list(self.extractor_languages),
            "import_roots": list(self.import_roots),
            "regions": [item.to_data() for item in self.regions],
            "rooms": [item.to_data() for item in self.rooms],
            "portals": [item.to_data() for item in self.portals],
            "unresolved_references": [item.to_data() for item in self.unresolved],
            "claims": [item.to_data() for item in self.claims],
            "notes": list(self.notes),
        }


# --- git -----------------------------------------------------------------------

def _git(clone: Path, *args: str) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(clone), *args],
            capture_output=True, text=True, check=True, encoding="utf-8",
        )
    except FileNotFoundError as exc:
        raise RepoScanError("git is not on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise RepoScanError(
            f"git {' '.join(args)} failed in {clone}: {exc.stderr.strip()}"
        ) from exc
    return out.stdout


def _require_clean_clone(clone: Path) -> None:
    toplevel = _git(clone, "rev-parse", "--show-toplevel").strip()
    if Path(toplevel).resolve() != clone.resolve():
        raise RepoScanError(
            f"{clone} is not the top level of its repository ({toplevel})"
        )
    status = _git(clone, "status", "--porcelain", "--untracked-files=all")
    if status.strip():
        lines = status.rstrip("\n").splitlines()
        shown = "\n".join(lines[:10])
        more = "" if len(lines) <= 10 else f"\n... {len(lines) - 10} more"
        raise DirtyCloneError(
            f"{clone} has uncommitted or untracked changes; the scan only "
            f"describes a clean checkout:\n{shown}{more}"
        )


def _tracked_files(clone: Path) -> tuple[str, ...]:
    raw = _git(clone, "ls-files", "-z")
    return tuple(sorted(part for part in raw.split("\0") if part))


# --- identities ----------------------------------------------------------------

def region_identity(commit: str, directory: str) -> str:
    return f"repo:{commit}/dir:{directory}"


def room_identity(commit: str, path: str) -> str:
    return f"repo:{commit}/file:{path}"


def _pair_identity(commit: str, source: str, line: int, target: str) -> str:
    return f"repo:{commit}/portal:{source}:{line}->{target}"


def claim_identity(commit: str, document: str, stand_in: str) -> str:
    return f"repo:{commit}/claim:{document}#{stand_in}"


# --- tree ----------------------------------------------------------------------

def _extractor_matches(extractor: Any, path: str) -> bool:
    pure = PurePosixPath(path)
    return (pure.suffix in tuple(getattr(extractor, "SUFFIXES", ()))
            or pure.name in tuple(getattr(extractor, "FILENAMES", ())))


def _extractor_name(extractor: Any) -> str:
    return str(getattr(extractor, "__name__", None) or getattr(extractor, "LANGUAGE", repr(extractor)))


def _language(path: str, extractors: Sequence[Any] = ()) -> str:
    """The compiler's language name for a file when a known language claims
    its suffix; else the first matching extractor's ``LANGUAGE``; else "other"."""

    known = _SUFFIX_LANGUAGE.get(PurePosixPath(path).suffix.lower())
    if known is not None:
        return known
    for extractor in extractors:
        if _extractor_matches(extractor, path):
            return str(extractor.LANGUAGE)
    return "other"


def language_table(extractors: Sequence[Any] = ()) -> list[dict[str, Any]]:
    """Known languages with their compiler route and their linking state.

    A compiler language is linked by a built-in reader (``LINKING_READERS``)
    or by any extractor whose suffixes overlap the language's.  Languages
    that only an extractor knows (html, cmake, …) are listed after them with
    no route.
    """

    def readers_for(suffixes: Iterable[str]) -> list[str]:
        wanted = set(suffixes)
        return sorted(
            _extractor_name(extractor) for extractor in extractors
            if wanted & set(getattr(extractor, "SUFFIXES", ()))
        )

    rows = []
    for name, translation in DREAM_LANGUAGE_TRANSLATIONS.items():
        suffixes = LANGUAGE_SUFFIXES.get(name, ())
        readers = ([LINKING_READERS[name]] if name in LINKING_READERS else []) + readers_for(suffixes)
        rows.append({
            "language": name,
            "route": translation.route,
            "translation_table": translation.translation_table,
            "execution_target": translation.execution_target,
            "suffixes": list(suffixes),
            "linking": readers[0] if readers else None,
            "linking_readers": readers,
            "linking_state": (
                "cross-file references read" if readers
                else "known language, no cross-file reference reader yet: rooms get no portals"
            ),
        })
    known = set(DREAM_LANGUAGE_TRANSLATIONS)
    for extractor in extractors:
        name = str(getattr(extractor, "LANGUAGE", "")) or _extractor_name(extractor)
        if name in known:
            continue
        known.add(name)
        rows.append({
            "language": name, "route": None, "translation_table": None,
            "execution_target": None,
            "suffixes": list(getattr(extractor, "SUFFIXES", ())),
            "linking": _extractor_name(extractor),
            "linking_readers": [_extractor_name(extractor)],
            "linking_state": "extractor-only language: cross-file references read, no compiler route",
        })
    return rows


def _build_tree(commit: str, files: Iterable[str], extractors: Sequence[Any] = ()) -> tuple[tuple[RepoRegion, ...], tuple[RepoRoom, ...]]:
    directories: dict[str, RepoRegion] = {
        ".": RepoRegion(region_identity(commit, "."), ".", ".", None, 0),
    }
    rooms: list[RepoRoom] = []
    for path in files:
        parts = PurePosixPath(path).parts
        parent = "."
        for depth, part in enumerate(parts[:-1], start=1):
            directory = part if parent == "." else f"{parent}/{part}"
            if directory not in directories:
                directories[directory] = RepoRegion(
                    region_identity(commit, directory), directory, part,
                    directories[parent].identity, depth,
                )
            parent = directory
        rooms.append(RepoRoom(
            room_identity(commit, path), path, parts[-1],
            directories[parent].identity, _language(path, extractors),
        ))
    regions = tuple(sorted(directories.values(), key=lambda r: (r.depth, r.path)))
    return regions, tuple(rooms)


# --- references ----------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class _Reference:
    source: str
    line: int
    kind: str
    specifier: str
    target: str | None
    reason: str = ""


def _python_module_candidates(module: str, root: str = "") -> tuple[str, ...]:
    base = module.replace(".", "/")
    if root:
        base = f"{root}/{base}"
    return (f"{base}.py", f"{base}/__init__.py")


def _resolve_python(
    module: str, tracked: frozenset[str], roots: Sequence[str] = ("",),
    ambiguities: list[str] | None = None,
) -> str | None:
    """First hit in root order; a name found under several roots is counted."""

    hits: list[str] = []
    for root in roots:
        for candidate in _python_module_candidates(module, root):
            if candidate in tracked:
                hits.append(candidate)
                break
    if len(hits) > 1 and ambiguities is not None:
        ambiguities.append(module)
    return hits[0] if hits else None


def _python_references(
    path: str, text: str, tracked: frozenset[str],
    roots: Sequence[str] = ("",), ambiguities: list[str] | None = None,
) -> tuple[list[_Reference], str | None]:
    try:
        tree = ast.parse(text, filename=path)
    except SyntaxError as exc:
        return [], f"{path}: not parsed as Python ({exc.msg} at line {exc.lineno})"
    package_parts = PurePosixPath(path).parts[:-1]
    refs: list[_Reference] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                target = _resolve_python(alias.name, tracked, roots, ambiguities)
                refs.append(_Reference(
                    path, node.lineno, "python-import", alias.name, target,
                    "" if target else "no tracked module at that absolute name",
                ))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                up = node.level - 1
                if up > len(package_parts):
                    refs.append(_Reference(
                        path, node.lineno, "python-import",
                        "." * node.level + (node.module or ""), None,
                        "relative import escapes the repository root",
                    ))
                    continue
                base_parts = package_parts[: len(package_parts) - up]
                base = ".".join(base_parts)
                module = ".".join(part for part in (base, node.module or "") if part)
                spelled = "." * node.level + (node.module or "")
            else:
                module = node.module or ""
                spelled = module
            # ``from pkg import name`` names a submodule when pkg/name.py is
            # tracked; otherwise the reference is to the module ``pkg`` itself.
            # Relative imports resolve from the file's own directory only.
            import_roots = ("",) if node.level else roots
            module_target = (
                _resolve_python(module, tracked, import_roots, ambiguities) if module else None
            )
            for alias in node.names:
                if alias.name == "*":
                    target, specifier = module_target, spelled
                else:
                    submodule = f"{module}.{alias.name}" if module else alias.name
                    target = _resolve_python(submodule, tracked, import_roots, ambiguities) or module_target
                    specifier = f"{spelled}:{alias.name}"
                refs.append(_Reference(
                    path, node.lineno, "python-import", specifier, target,
                    "" if target else "no tracked module at that name",
                ))
    return refs, None


def _resolve_javascript(source: str, specifier: str, tracked: frozenset[str]) -> tuple[str | None, str]:
    if not specifier.startswith(("./", "../", "/")):
        return None, "bare specifier (package or URL), not a repository path"
    if specifier.startswith("/"):
        return None, "absolute URL path has no repository anchor"
    base = PurePosixPath(source).parent
    joined = PurePosixPath(*(base / specifier).parts)
    parts: list[str] = []
    for part in joined.parts:
        if part == "..":
            if not parts:
                return None, "relative specifier escapes the repository root"
            parts.pop()
        elif part not in {".", ""}:
            parts.append(part)
    stem = "/".join(parts)
    for suffix in _JS_RESOLUTION_SUFFIXES:
        candidate = f"{stem}{suffix}"
        if candidate in tracked:
            return candidate, ""
    return None, "no tracked file at that relative path"


def _javascript_references(path: str, text: str, tracked: frozenset[str]) -> list[_Reference]:
    refs: list[_Reference] = []
    for number, line in enumerate(text.splitlines(), start=1):
        for pattern in _JS_IMPORT_PATTERNS:
            match = pattern.match(line)
            if match:
                specifier = match.group(2)
                target, reason = _resolve_javascript(path, specifier, tracked)
                refs.append(_Reference(path, number, "es-module-import", specifier, target, reason))
                break
    return refs


def _dream_claims(
    commit: str, path: str, raw: bytes, room: RepoRoom,
) -> tuple[list[RepoRoomClaim], str | None]:
    """Claimed rooms for every stand-in of one tracked dream document."""

    try:
        document = parse_dream_document(raw)
    except DreamDocumentError as error:
        return [], f"{path}: dream document not parsed ({error})"
    claims = []
    for entry in document.work_table().stand_ins:
        claims.append(RepoRoomClaim(
            claim_identity(commit, path, entry.block), room.identity, room.region,
            entry.block, entry.content_hash, entry.state, entry.owner,
            entry.described_by,
            tuple(contribution.to_data() for contribution in entry.contributions),
        ))
    return claims, None


# --- import roots --------------------------------------------------------------

def _normalize_root(raw: str) -> str:
    text = str(raw).strip().replace("\\", "/")
    if text.startswith("/") or (len(text) > 1 and text[1] == ":"):
        raise RepoScanError(f"import root {raw!r} is not repository-relative")
    norm = posixpath.normpath(text) if text else "."
    if norm == "." or norm == "":
        return ""
    if norm == ".." or norm.startswith("../"):
        raise RepoScanError(f"import root {raw!r} escapes the repository")
    return norm


def _tracked_directories(tracked: Iterable[str]) -> frozenset[str]:
    directories: set[str] = set()
    for path in tracked:
        parent = PurePosixPath(path).parent
        while str(parent) != ".":
            directories.add(str(parent))
            parent = parent.parent
    return frozenset(directories)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part for part in re.split(r"[\s,]+", value) if part]
    return [str(item) for item in value]


def _read_declared_roots(tracked: frozenset[str], read: Callable[[str], bytes]) -> list[str]:
    """Raw root declarations from the tracked declaration files, via ``read``."""

    raw: list[str] = []

    def toml_of(name: str) -> dict[str, Any]:
        if name not in tracked:
            return {}
        try:
            return tomllib.loads(_text(read(name)))
        except tomllib.TOMLDecodeError as exc:
            raise RepoScanError(f"{name}: not valid TOML ({exc})") from exc

    tool = toml_of("pyproject.toml").get("tool", {})
    raw += _as_list(tool.get("pytest", {}).get("ini_options", {}).get("pythonpath"))
    setuptools = tool.get("setuptools", {})
    packages = setuptools.get("packages", {})
    if isinstance(packages, dict):
        raw += _as_list(packages.get("find", {}).get("where"))
    package_dir = setuptools.get("package-dir", {})
    if isinstance(package_dir, dict):
        raw += _as_list(list(package_dir.values()))
    raw += _as_list(toml_of("repo_world.toml").get("import_roots"))
    for name, section in (("pytest.ini", "pytest"), ("setup.cfg", "tool:pytest"), ("tox.ini", "pytest")):
        if name not in tracked:
            continue
        parser = configparser.ConfigParser(interpolation=None)
        try:
            parser.read_string(_text(read(name)))
        except configparser.Error as exc:
            raise RepoScanError(f"{name}: not valid INI ({exc})") from exc
        if parser.has_option(section, "pythonpath"):
            raw += _as_list(parser.get(section, "pythonpath"))
    return raw


def _declared_roots(tracked: frozenset[str], read: Callable[[str], bytes]) -> tuple[str, ...]:
    directories = _tracked_directories(tracked)
    roots: list[str] = []
    for raw in _read_declared_roots(tracked, read):
        root = _normalize_root(raw)
        if root and root in directories and root not in roots:
            roots.append(root)
    return tuple(roots)


def declared_import_roots(clone_path: str | Path) -> tuple[str, ...]:
    """Import roots declared inside a checkout, in declaration order.

    Only tracked directories are returned; the repository root ("") is not
    included here (the scan always puts it first).
    """

    clone = Path(clone_path)
    tracked = frozenset(_tracked_files(clone))
    return _declared_roots(tracked, lambda path: (clone / path).read_bytes())


def _validated_roots(import_roots: Sequence[str], tracked: frozenset[str]) -> tuple[str, ...]:
    directories = _tracked_directories(tracked)
    roots = [""]
    for raw in import_roots:
        root = _normalize_root(raw)
        if root and root not in directories:
            raise RepoScanError(f"import root {raw!r} is not a tracked directory")
        if root not in roots:
            roots.append(root)
    return tuple(roots)


# --- extractors ----------------------------------------------------------------

def default_extractors() -> tuple[Any, ...]:
    modules = []
    for name in EXTRACTOR_MODULES:
        try:
            modules.append(importlib.import_module(name))
        except Exception as exc:
            raise RepoScanError(f"extractor module {name!r} failed to import: {exc!r}") from exc
    return tuple(modules)


def _plugin_references(
    extractor: Any, path: str, text: str, tracked: frozenset[str],
) -> list[_Reference]:
    name = _extractor_name(extractor)
    try:
        rows = list(extractor.references(path, text, tracked))
    except Exception as exc:
        raise RepoScanError(f"extractor {name} failed on {path}: {exc!r}") from exc
    refs: list[_Reference] = []
    for row in rows:
        try:
            line, kind, specifier, target, reason = row
        except (TypeError, ValueError) as exc:
            raise RepoScanError(f"extractor {name} returned a malformed row for {path}: {row!r}") from exc
        if target is not None and target not in tracked:
            raise RepoScanError(
                f"extractor {name} returned untracked target {target!r} for {path}:{line}"
            )
        refs.append(_Reference(path, int(line), str(kind), str(specifier), target, str(reason or "")))
    return refs


# Languages whose file bytes the built-in readers need; every other tracked
# file is a room by name only unless an extractor matches it or it declares
# import roots (DECLARATION_FILES).
_READ_LANGUAGES = frozenset({"python", "javascript", "dream-document"})


def _reads(path: str, extractors: Sequence[Any]) -> bool:
    """Whether the scan needs this file's bytes at all."""

    if _language(path) in _READ_LANGUAGES or PurePosixPath(path).name in DECLARATION_FILES:
        return True
    return any(_extractor_matches(extractor, path) for extractor in extractors)


def _text(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace")


# --- entry ---------------------------------------------------------------------

def scan_files(
    commit: str, origin: str | None, files: Iterable[str],
    read: Callable[[str], bytes], notes: Iterable[str] = (), *,
    extractors: Sequence[Any] | None = None,
    import_roots: Sequence[str] | None = (),
) -> RepoWorldScan:
    """The scan itself, over a tree given as tracked paths and a byte reader.

    Every entry point ends here.  ``read`` is called only for files
    :func:`_reads` says the scan needs; a streaming reader may hold just
    those bytes.  ``extractors`` are plugin modules (None: the default
    registry).  ``import_roots`` are extra Python import roots: a sequence is
    validated against the tracked directories; ``None`` means the roots the
    tree itself declares (:data:`DECLARATION_FILES`).  The repository root is
    always first.
    """

    files = tuple(sorted(files))
    tracked = frozenset(files)
    plugins = tuple(default_extractors() if extractors is None else extractors)
    if import_roots is None:
        import_roots = _declared_roots(tracked, read)
    roots = _validated_roots(import_roots, tracked)
    regions, rooms = _build_tree(commit, files, plugins)

    references: list[_Reference] = []
    notes = list(notes)
    ambiguities: list[str] = []
    languages_used: set[str] = {"python", "javascript"}
    claims: list[RepoRoomClaim] = []
    room_by_path = {room.path: room for room in rooms}
    for path in files:
        if not _reads(path, plugins):
            continue
        language = _language(path)
        raw = read(path)
        text: str | None = None
        if language == "dream-document":
            found, note = _dream_claims(commit, path, raw, room_by_path[path])
            claims.extend(found)
            if note:
                notes.append(note)
        elif language == "python":
            text = _text(raw)
            refs, note = _python_references(path, text, tracked, roots, ambiguities)
            references.extend(refs)
            if note:
                notes.append(note)
        elif language == "javascript":
            text = _text(raw)
            references.extend(_javascript_references(path, text, tracked))
        for extractor in plugins:
            if _extractor_matches(extractor, path):
                if text is None:
                    text = _text(raw)
                references.extend(_plugin_references(extractor, path, text, tracked))
                languages_used.add(str(extractor.LANGUAGE))
    if ambiguities:
        notes.append(
            f"{len(ambiguities)} python import name(s) resolved under more than one "
            f"import root; the earliest root was used"
        )

    portals: list[RepoPortalPair] = []
    unresolved: list[RepoUnresolvedReference] = []
    seen_pairs: set[str] = set()
    for ref in sorted(references, key=lambda r: (r.source, r.line, r.kind, r.specifier)):
        source_room = room_identity(commit, ref.source)
        if ref.target is None:
            unresolved.append(RepoUnresolvedReference(
                source_room, ref.line, ref.kind, ref.specifier, ref.reason,
            ))
            continue
        if ref.target == ref.source:
            unresolved.append(RepoUnresolvedReference(
                source_room, ref.line, ref.kind, ref.specifier,
                "self-reference; a portal needs two rooms",
            ))
            continue
        pair_id = _pair_identity(commit, ref.source, ref.line, ref.target)
        if pair_id in seen_pairs:
            # Two names (or two kinds) on one line to one target are one portal.
            continue
        seen_pairs.add(pair_id)
        target_room = room_identity(commit, ref.target)
        portals.append(RepoPortalPair(
            pair_id, source_room, target_room, ref.line, ref.kind, ref.specifier,
            RepoPortalEnd(f"{pair_id}/end:out", source_room, "out", pair_id),
            RepoPortalEnd(f"{pair_id}/end:in", target_room, "in", pair_id),
        ))

    return RepoWorldScan(
        commit, origin, regions, rooms, tuple(portals), tuple(unresolved), tuple(notes),
        tuple(claims), tuple(sorted(languages_used)), roots,
        tuple(language_table(plugins)),
    )


def scan_repository(
    clone_path: str | Path, *, extractors: Sequence[Any] | None = None,
    import_roots: Sequence[str] | None = (),
) -> RepoWorldScan:
    """Scan one clean checkout on disk (fixtures and explicit clones).

    ``extractors`` and ``import_roots`` as in :func:`scan_files`.  Raises
    :class:`DirtyCloneError` when the tree has uncommitted or untracked
    changes and :class:`RepoScanError` when the path is not a repository root.
    """

    clone = Path(clone_path)
    if not clone.is_dir():
        raise RepoScanError(f"{clone} is not a directory")
    _require_clean_clone(clone)
    commit = _git(clone, "rev-parse", "HEAD").strip()
    origin_raw = subprocess.run(
        ["git", "-C", str(clone), "config", "--get", "remote.origin.url"],
        capture_output=True, text=True, encoding="utf-8",
    )
    origin = origin_raw.stdout.strip() or None
    return scan_files(
        commit, origin, _tracked_files(clone), lambda path: (clone / path).read_bytes(),
        extractors=extractors, import_roots=import_roots,
    )


def scan_tar_stream(
    commit: str, origin: str | None, stream: Any, *, mode: str = "r|",
    strip_components: int = 0, extractors: Sequence[Any] | None = None,
    import_roots: Sequence[str] | None = None,
) -> RepoWorldScan:
    """Scan a tar archive of one commit as it streams past, keeping no file.

    One pass: every regular file becomes a room by name; only the bytes of
    files a reader links are held, and only until the scan returns.  Nothing
    is written anywhere.  ``strip_components`` drops the leading directory a
    GitHub tarball wraps the tree in.  Symbolic links and other non-regular
    members are noted, not scanned.
    """

    plugins = tuple(default_extractors() if extractors is None else extractors)
    paths: list[str] = []
    contents: dict[str, bytes] = {}
    notes: list[str] = []
    with tarfile.open(fileobj=stream, mode=mode) as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts[strip_components:]
            if not parts:
                continue
            path = "/".join(parts)
            if member.isdir():
                continue
            if not member.isfile():
                notes.append(f"{path}: not a regular file in the archive, not scanned")
                continue
            paths.append(path)
            if _reads(path, plugins):
                handle = archive.extractfile(member)
                contents[path] = handle.read() if handle is not None else b""
    return scan_files(
        commit, origin, paths, contents.__getitem__, notes,
        extractors=plugins, import_roots=import_roots,
    )


def github_clone_url(repository: str) -> str:
    """``owner/repo`` -> the public HTTPS clone URL.  Nothing else is accepted."""

    if not GITHUB_REPOSITORY.match(repository):
        raise RepoScanError(
            f"{repository!r} is not a GitHub repository spelled owner/repo"
        )
    return f"https://github.com/{repository}.git"


def _github_json(path: str) -> Any:
    request = urllib.request.Request(
        f"{GITHUB_API}/{path}",
        headers={"Accept": "application/vnd.github+json", "User-Agent": _USER_AGENT},
    )
    try:
        with urllib.request.urlopen(request) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RepoScanError(f"GitHub API {path}: HTTP {exc.code} {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RepoScanError(f"GitHub API {path}: {exc.reason}") from exc


def scan_github_repository(
    repository: str, *, ref: str | None = None,
    extractors: Sequence[Any] | None = None, import_roots: Sequence[str] | None = None,
) -> RepoWorldScan:
    """Scan ``owner/repo`` as GitHub serves it, streamed, never on disk.

    The commit is resolved first (``ref`` or the default branch), then that
    exact commit's tarball is streamed from the API and scanned as it passes.
    What the API excludes (``export-ignore`` paths) the scan never sees.
    """

    if not GITHUB_REPOSITORY.match(repository):
        raise RepoScanError(f"{repository!r} is not a GitHub repository spelled owner/repo")
    if ref is None:
        ref = _github_json(f"repos/{repository}")["default_branch"]
    commit = _github_json(f"repos/{repository}/commits/{ref}")["sha"]
    request = urllib.request.Request(
        f"{GITHUB_API}/repos/{repository}/tarball/{commit}",
        headers={"User-Agent": _USER_AGENT},
    )
    try:
        with urllib.request.urlopen(request) as response:
            return scan_tar_stream(
                commit, github_clone_url(repository), response, mode="r|gz",
                strip_components=1, extractors=extractors, import_roots=import_roots,
            )
    except urllib.error.HTTPError as exc:
        raise RepoScanError(f"GitHub tarball of {repository}@{commit[:12]}: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RepoScanError(f"GitHub tarball of {repository}: {exc.reason}") from exc


def scan_remote_repository(
    url: str, *, ref: str | None = None,
    extractors: Sequence[Any] | None = None, import_roots: Sequence[str] | None = None,
) -> RepoWorldScan:
    """Scan any git URL as the remote serves it, streamed, never on disk.

    A GitHub URL goes through :func:`scan_github_repository`.  Any other URL
    (a ``file://`` fixture, another host) is resolved with ``git ls-remote``
    and streamed with ``git archive --remote``; GitHub does not serve
    ``upload-archive``, which is why it has its own lane.
    """

    github = GITHUB_CLONE_URL.match(url.strip())
    if github:
        return scan_github_repository(
            f"{github['owner']}/{github['repo']}", ref=ref,
            extractors=extractors, import_roots=import_roots,
        )
    name = ref or "HEAD"
    try:
        listing = subprocess.run(
            ["git", "ls-remote", "--", url, name], capture_output=True, text=True,
            encoding="utf-8", check=True,
        ).stdout
    except FileNotFoundError as exc:
        raise RepoScanError("git is not on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise RepoScanError(f"git ls-remote {url}: {exc.stderr.strip()}") from exc
    if not listing.strip():
        raise RepoScanError(f"{url} serves no ref named {name!r}")
    commit = listing.split()[0]
    process = subprocess.Popen(
        ["git", "archive", f"--remote={url}", "--format=tar", name],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    try:
        scan = scan_tar_stream(
            commit, url, process.stdout, mode="r|",
            extractors=extractors, import_roots=import_roots,
        )
    finally:
        process.stdout.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        returncode = process.wait()
    if returncode != 0:
        raise RepoScanError(f"git archive --remote={url} {name}: {stderr.strip()}")
    return scan


def _github_with_flag_roots(repository: str, ref: str | None, flag_roots: list[str]) -> RepoWorldScan:
    """Declared roots from the stream plus the flags: scan once with the union."""

    declared = scan_github_repository(repository, ref=ref, import_roots=None).import_roots
    roots = [root for root in declared if root] + flag_roots
    return scan_github_repository(repository, ref=ref, import_roots=roots)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in args
    args = [item for item in args if item != "--json"]
    out_path = None
    if "--out" in args:
        index = args.index("--out")
        if index + 1 >= len(args):
            print("--out needs a file path", file=sys.stderr)
            return 2
        out_path = args[index + 1]
        del args[index:index + 2]
    flag_roots: list[str] = []
    while "--import-root" in args:
        index = args.index("--import-root")
        if index + 1 >= len(args):
            print("--import-root needs a directory", file=sys.stderr)
            return 2
        flag_roots.append(args[index + 1])
        del args[index:index + 2]
    ref = None
    if "--ref" in args:
        index = args.index("--ref")
        if index + 1 >= len(args):
            print("--ref needs a branch or tag", file=sys.stderr)
            return 2
        ref = args[index + 1]
        del args[index:index + 2]
    local_path = None
    if "--path" in args:
        index = args.index("--path")
        if index + 1 >= len(args):
            print("--path needs a clone directory", file=sys.stderr)
            return 2
        local_path = args[index + 1]
        del args[index:index + 2]
    usage = "\n".join(__doc__.strip().splitlines()[-2:])
    try:
        if local_path is not None:
            if args or ref:
                print(usage, file=sys.stderr)
                return 2
            scan = scan_repository(
                local_path, import_roots=(*declared_import_roots(local_path), *flag_roots),
            )
        elif len(args) == 1:
            # The stream lane reads the declared roots from the stream; flags add to them.
            scan = (scan_github_repository(args[0], ref=ref) if not flag_roots
                    else _github_with_flag_roots(args[0], ref, flag_roots))
        else:
            print(usage, file=sys.stderr)
            return 2
    except RepoScanError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    if out_path is not None:
        Path(out_path).write_text(json.dumps(scan.to_data(), indent=2) + "\n", encoding="utf-8")
        print(f"commit {scan.commit} -> {out_path}")
        return 0
    if as_json:
        json.dump(scan.to_data(), sys.stdout, indent=2, sort_keys=False)
        print()
        return 0
    print(f"commit {scan.commit}")
    print(f"regions {len(scan.regions)}  rooms {len(scan.rooms)}  portal pairs {len(scan.portals)}  "
          f"unresolved {len(scan.unresolved)}  claimed rooms {len(scan.claims)}")
    for claim in scan.claims:
        print(f"  claim {claim.document_room.split('/file:', 1)[1]}#{claim.stand_in} "
              f"{claim.state} owner={claim.owner}")
    for portal in scan.portals:
        print(f"  {portal.source_room.split('/file:', 1)[1]}:{portal.source_line} -> "
              f"{portal.target_room.split('/file:', 1)[1]}  [{portal.reference_kind} {portal.specifier}]")
    for note in scan.notes:
        print(f"  note: {note}")
    return 0


__all__ = [
    "DECLARATION_FILES", "EXTRACTOR_MODULES", "declared_import_roots", "default_extractors",
    "LANGUAGE_SUFFIXES", "LINKING_READERS", "REPO_WORLD_SCAN_VERSION",
    "DirtyCloneError", "RepoPortalEnd", "RepoPortalPair", "language_table",
    "RepoRegion", "RepoRoom", "RepoScanError", "RepoUnresolvedReference",
    "RepoWorldScan", "RepoRoomClaim", "claim_identity",
    "github_clone_url", "region_identity", "room_identity", "scan_files",
    "scan_github_repository", "scan_remote_repository", "scan_repository", "scan_tar_stream",
]


if __name__ == "__main__":
    sys.exit(main())
