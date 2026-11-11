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
:func:`declared_import_roots`, which reads only declarations inside the clone:
pyproject.toml ``[tool.pytest.ini_options] pythonpath``,
``[tool.setuptools.packages.find] where``, ``[tool.setuptools] package-dir``
values; ``pythonpath`` in pytest.ini / setup.cfg (``[tool:pytest]``) /
tox.ini; and a repo-owned ``repo_world.toml`` with ``import_roots = ["dir"]``.
The earliest root wins when a name resolves under several; a scan note counts
those ambiguities.  Relative imports resolve from the file's own directory.

Every identity is keyed by the commit hash, so two scans of the same commit
produce identical records and a scan of a different commit never collides.
The scanner refuses a tree with uncommitted or untracked changes: the record
set describes a commit, never a working state.

    python -m src.compiler.repo_world_scan <clone-path> [--json] [--import-root DIR ...]
"""

from __future__ import annotations

import ast
import configparser
from dataclasses import dataclass, field
import importlib
import json
import posixpath
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tomllib
from typing import Any, Iterable, Mapping, Sequence

REPO_WORLD_SCAN_VERSION = "repo-world-scan-v1"

# Fully qualified names of extractor plugin modules (contract in the module
# docstring).  Filled in once the plugins exist; an entry that fails to import
# is a RepoScanError, never skipped.
EXTRACTOR_MODULES: tuple[str, ...] = (
    "src.compiler.repo_world_refs_c",
    "src.compiler.repo_world_refs_cmake",
    "src.compiler.repo_world_refs_html",
    "src.compiler.repo_world_refs_js_dynamic",
    "src.compiler.repo_world_refs_powershell",
    "src.compiler.repo_world_refs_shell",
)

PYTHON_SUFFIXES = (".py",)
JAVASCRIPT_SUFFIXES = (".js", ".mjs")
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
    extractor_languages: tuple[str, ...] = ()
    import_roots: tuple[str, ...] = ("",)

    @property
    def root_region(self) -> RepoRegion:
        return self.regions[0]

    def to_data(self) -> dict[str, Any]:
        return {
            "schema": REPO_WORLD_SCAN_VERSION,
            "identity": f"repo:{self.commit}",
            "commit": self.commit,
            "origin": self.origin,
            "regions": [item.to_data() for item in self.regions],
            "rooms": [item.to_data() for item in self.rooms],
            "portals": [item.to_data() for item in self.portals],
            "unresolved_references": [item.to_data() for item in self.unresolved],
            "notes": list(self.notes),
            "extractor_languages": list(self.extractor_languages),
            "import_roots": list(self.import_roots),
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


# --- tree ----------------------------------------------------------------------

def _extractor_matches(extractor: Any, path: str) -> bool:
    pure = PurePosixPath(path)
    return (pure.suffix in tuple(getattr(extractor, "SUFFIXES", ()))
            or pure.name in tuple(getattr(extractor, "FILENAMES", ())))


def _language(path: str, extractors: Sequence[Any] = ()) -> str:
    suffix = PurePosixPath(path).suffix
    if suffix in PYTHON_SUFFIXES:
        return "python"
    if suffix in JAVASCRIPT_SUFFIXES:
        return "javascript"
    for extractor in extractors:
        if _extractor_matches(extractor, path):
            return str(extractor.LANGUAGE)
    return "other"


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


def _read(clone: Path, path: str) -> str:
    return (clone / path).read_text(encoding="utf-8", errors="replace")


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


def _read_declared_roots(clone: Path, tracked: frozenset[str]) -> list[str]:
    raw: list[str] = []

    def toml_of(name: str) -> dict[str, Any]:
        if name not in tracked:
            return {}
        try:
            return tomllib.loads(_read(clone, name))
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
            parser.read_string(_read(clone, name))
        except configparser.Error as exc:
            raise RepoScanError(f"{name}: not valid INI ({exc})") from exc
        if parser.has_option(section, "pythonpath"):
            raw += _as_list(parser.get(section, "pythonpath"))
    return raw


def declared_import_roots(clone_path: str | Path) -> tuple[str, ...]:
    """Import roots declared inside the clone, in declaration order.

    Only tracked directories are returned; the repository root ("") is not
    included here (``scan_repository`` always puts it first).
    """

    clone = Path(clone_path)
    tracked = frozenset(_tracked_files(clone))
    directories = _tracked_directories(tracked)
    roots: list[str] = []
    for raw in _read_declared_roots(clone, tracked):
        root = _normalize_root(raw)
        if root and root in directories and root not in roots:
            roots.append(root)
    return tuple(roots)


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


def _extractor_name(extractor: Any) -> str:
    return str(getattr(extractor, "__name__", None) or getattr(extractor, "LANGUAGE", repr(extractor)))


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


# --- entry ---------------------------------------------------------------------

def scan_repository(
    clone_path: str | Path, *, extractors: Sequence[Any] | None = None,
    import_roots: Sequence[str] = (),
) -> RepoWorldScan:
    """Scan one clean clone into regions, rooms and portal pairs.

    ``extractors`` are plugin modules (None: :func:`default_extractors`);
    ``import_roots`` are extra repo-relative Python import roots (validated
    tracked directories; the repository root is always first).

    Raises :class:`DirtyCloneError` when the tree has uncommitted or untracked
    changes and :class:`RepoScanError` when the path is not a repository root.
    """

    clone = Path(clone_path)
    if not clone.is_dir():
        raise RepoScanError(f"{clone} is not a directory")
    _require_clean_clone(clone)
    plugins = tuple(default_extractors() if extractors is None else extractors)
    commit = _git(clone, "rev-parse", "HEAD").strip()
    origin_raw = subprocess.run(
        ["git", "-C", str(clone), "config", "--get", "remote.origin.url"],
        capture_output=True, text=True, encoding="utf-8",
    )
    origin = origin_raw.stdout.strip() or None

    files = _tracked_files(clone)
    tracked = frozenset(files)
    roots = _validated_roots(import_roots, tracked)
    regions, rooms = _build_tree(commit, files, plugins)

    references: list[_Reference] = []
    notes: list[str] = []
    ambiguities: list[str] = []
    languages_used: set[str] = {"python", "javascript"}
    for path in files:
        language = _language(path)
        text: str | None = None
        if language == "python":
            text = _read(clone, path)
            refs, note = _python_references(path, text, tracked, roots, ambiguities)
            references.extend(refs)
            if note:
                notes.append(note)
        elif language == "javascript":
            text = _read(clone, path)
            references.extend(_javascript_references(path, text, tracked))
        for extractor in plugins:
            if _extractor_matches(extractor, path):
                if text is None:
                    text = _read(clone, path)
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
        tuple(sorted(languages_used)), roots,
    )


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in args
    args = [item for item in args if item != "--json"]
    paths: list[str] = []
    flag_roots: list[str] = []
    index = 0
    while index < len(args):
        if args[index] == "--import-root":
            if index + 1 >= len(args):
                print("--import-root needs a directory", file=sys.stderr)
                return 2
            flag_roots.append(args[index + 1])
            index += 2
        else:
            paths.append(args[index])
            index += 1
    if len(paths) != 1:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    try:
        scan = scan_repository(
            paths[0],
            import_roots=(*declared_import_roots(paths[0]), *flag_roots),
        )
    except RepoScanError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    if as_json:
        json.dump(scan.to_data(), sys.stdout, indent=2, sort_keys=False)
        print()
        return 0
    print(f"commit {scan.commit}")
    print(f"regions {len(scan.regions)}  rooms {len(scan.rooms)}  portal pairs {len(scan.portals)}  unresolved {len(scan.unresolved)}")
    for portal in scan.portals:
        print(f"  {portal.source_room.split('/file:', 1)[1]}:{portal.source_line} -> "
              f"{portal.target_room.split('/file:', 1)[1]}  [{portal.reference_kind} {portal.specifier}]")
    for note in scan.notes:
        print(f"  note: {note}")
    return 0


__all__ = [
    "REPO_WORLD_SCAN_VERSION", "DirtyCloneError", "RepoPortalEnd", "RepoPortalPair",
    "RepoRegion", "RepoRoom", "RepoScanError", "RepoUnresolvedReference",
    "RepoWorldScan", "EXTRACTOR_MODULES", "declared_import_roots", "default_extractors", "region_identity", "room_identity", "scan_repository",
]


if __name__ == "__main__":
    sys.exit(main())
