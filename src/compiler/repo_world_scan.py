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

Only two reference extractors exist: Python imports read with :mod:`ast`,
and JavaScript ES-module static imports (``import … from``, bare
``import "…"``, ``export … from``) read line by line.  A reference whose
target is not a tracked file is kept as an *unresolved reference* so nothing
disappears silently; it is not a portal because a portal needs two rooms.

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

    python -m src.compiler.repo_world_scan owner/repo [--ref main] [--json | --out <file>]
    python -m src.compiler.repo_world_scan --path <checkout> [--json | --out <file>]   # fixtures
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile
from typing import Any, Callable, Iterable, Mapping
import urllib.error
import urllib.request

from .dream_document import (
    DREAM_LANGUAGE_TRANSLATIONS, DreamDocumentError, parse_dream_document,
)

REPO_WORLD_SCAN_VERSION = "repo-world-scan-v0"
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

    @property
    def root_region(self) -> RepoRegion:
        return self.regions[0]

    def to_data(self) -> dict[str, Any]:
        return {
            "schema": REPO_WORLD_SCAN_VERSION,
            "identity": f"repo:{self.commit}",
            "commit": self.commit,
            "origin": self.origin,
            "languages": language_table(),
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

def _language(path: str) -> str:
    """The compiler's language name for a file, or "other" when no known
    language claims its suffix."""

    return _SUFFIX_LANGUAGE.get(PurePosixPath(path).suffix.lower(), "other")


def language_table() -> list[dict[str, Any]]:
    """Known languages with their compiler route and this scan's linking state."""

    rows = []
    for name, translation in DREAM_LANGUAGE_TRANSLATIONS.items():
        reader = LINKING_READERS.get(name)
        rows.append({
            "language": name,
            "route": translation.route,
            "translation_table": translation.translation_table,
            "execution_target": translation.execution_target,
            "suffixes": list(LANGUAGE_SUFFIXES.get(name, ())),
            "linking": reader,
            "linking_state": (
                "cross-file references read" if reader
                else "known language, no cross-file reference reader yet: rooms get no portals"
            ),
        })
    return rows


def _build_tree(commit: str, files: Iterable[str]) -> tuple[tuple[RepoRegion, ...], tuple[RepoRoom, ...]]:
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
            directories[parent].identity, _language(path),
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


def _python_module_candidates(module: str) -> tuple[str, ...]:
    base = module.replace(".", "/")
    return (f"{base}.py", f"{base}/__init__.py")


def _resolve_python(module: str, tracked: frozenset[str]) -> str | None:
    for candidate in _python_module_candidates(module):
        if candidate in tracked:
            return candidate
    return None


def _python_references(path: str, text: str, tracked: frozenset[str]) -> tuple[list[_Reference], str | None]:
    try:
        tree = ast.parse(text, filename=path)
    except SyntaxError as exc:
        return [], f"{path}: not parsed as Python ({exc.msg} at line {exc.lineno})"
    package_parts = PurePosixPath(path).parts[:-1]
    refs: list[_Reference] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                target = _resolve_python(alias.name, tracked)
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
            module_target = _resolve_python(module, tracked) if module else None
            for alias in node.names:
                if alias.name == "*":
                    target, specifier = module_target, spelled
                else:
                    submodule = f"{module}.{alias.name}" if module else alias.name
                    target = _resolve_python(submodule, tracked) or module_target
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


# Languages whose file bytes the scan reads at all; every other tracked file
# is a room by name only and its bytes are never loaded.
_READ_LANGUAGES = frozenset({"python", "javascript", "dream-document"})


def _text(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace")


# --- entry ---------------------------------------------------------------------

def scan_files(
    commit: str, origin: str | None, files: Iterable[str],
    read: Callable[[str], bytes], notes: Iterable[str] = (),
) -> RepoWorldScan:
    """The scan itself, over a tree given as tracked paths and a byte reader.

    Every entry point ends here.  ``read`` is called only for files in
    :data:`_READ_LANGUAGES`; the reader may therefore hold just those bytes.
    """

    files = tuple(sorted(files))
    tracked = frozenset(files)
    regions, rooms = _build_tree(commit, files)

    references: list[_Reference] = []
    notes = list(notes)
    claims: list[RepoRoomClaim] = []
    room_by_path = {room.path: room for room in rooms}
    for path in files:
        language = _language(path)
        if language not in _READ_LANGUAGES:
            continue
        raw = read(path)
        if language == "dream-document":
            found, note = _dream_claims(commit, path, raw, room_by_path[path])
            claims.extend(found)
            if note:
                notes.append(note)
        elif language == "python":
            refs, note = _python_references(path, _text(raw), tracked)
            references.extend(refs)
            if note:
                notes.append(note)
        elif language == "javascript":
            references.extend(_javascript_references(path, _text(raw), tracked))

    portals: list[RepoPortalPair] = []
    unresolved: list[RepoUnresolvedReference] = []
    seen_pairs: set[str] = set()
    for ref in sorted(references, key=lambda r: (r.source, r.line, r.specifier)):
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
            # Two names imported on one line from one module are one portal.
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
        tuple(claims),
    )


def scan_repository(clone_path: str | Path) -> RepoWorldScan:
    """Scan one clean checkout on disk (fixtures and explicit clones).

    Raises :class:`DirtyCloneError` when the tree has uncommitted or untracked
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
    )


def scan_tar_stream(
    commit: str, origin: str | None, stream: Any, *, mode: str = "r|",
    strip_components: int = 0,
) -> RepoWorldScan:
    """Scan a tar archive of one commit as it streams past, keeping no file.

    One pass: every regular file becomes a room by name; only the bytes of
    files a reader links are held, and only until the scan returns.  Nothing
    is written anywhere.  ``strip_components`` drops the leading directory a
    GitHub tarball wraps the tree in.  Symbolic links and other non-regular
    members are noted, not scanned.
    """

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
            if _language(path) in _READ_LANGUAGES:
                handle = archive.extractfile(member)
                contents[path] = handle.read() if handle is not None else b""
    return scan_files(commit, origin, paths, contents.__getitem__, notes)


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


def scan_github_repository(repository: str, *, ref: str | None = None) -> RepoWorldScan:
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
                strip_components=1,
            )
    except urllib.error.HTTPError as exc:
        raise RepoScanError(f"GitHub tarball of {repository}@{commit[:12]}: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RepoScanError(f"GitHub tarball of {repository}: {exc.reason}") from exc


def scan_remote_repository(url: str, *, ref: str | None = None) -> RepoWorldScan:
    """Scan any git URL as the remote serves it, streamed, never on disk.

    A GitHub URL goes through :func:`scan_github_repository`.  Any other URL
    (a ``file://`` fixture, another host) is resolved with ``git ls-remote``
    and streamed with ``git archive --remote``; GitHub does not serve
    ``upload-archive``, which is why it has its own lane.
    """

    github = GITHUB_CLONE_URL.match(url.strip())
    if github:
        return scan_github_repository(f"{github['owner']}/{github['repo']}", ref=ref)
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
        scan = scan_tar_stream(commit, url, process.stdout, mode="r|")
    finally:
        process.stdout.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        returncode = process.wait()
    if returncode != 0:
        raise RepoScanError(f"git archive --remote={url} {name}: {stderr.strip()}")
    return scan


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
            scan = scan_repository(local_path)
        elif len(args) == 1:
            scan = scan_github_repository(args[0], ref=ref)
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
    "LANGUAGE_SUFFIXES", "LINKING_READERS", "REPO_WORLD_SCAN_VERSION",
    "DirtyCloneError", "RepoPortalEnd", "RepoPortalPair", "language_table",
    "RepoRegion", "RepoRoom", "RepoScanError", "RepoUnresolvedReference",
    "RepoWorldScan", "RepoRoomClaim", "claim_identity",
    "github_clone_url", "region_identity", "room_identity", "scan_files",
    "scan_github_repository", "scan_remote_repository", "scan_repository", "scan_tar_stream",
]


if __name__ == "__main__":
    sys.exit(main())
