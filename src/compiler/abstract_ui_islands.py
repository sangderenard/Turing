"""Islands: an item's interior as an independent world, carried in a private file beside the item.

Every item of a tree (a file, a directory) may have an island: its own world, with its own geometry, reached only
through its door (``abstract_ui_doorways``). The island is not laid out in, sized by, or stored with the world
outside it. Its document is a small JSON file that sits BESIDE the item, so it travels with the item:

    a/b/x.py         ->  a/b/.x.py.island.json
    a/b/   (a dir)   ->  a/b/.island.json

``root`` in the functions below is whatever tree holds those files: the working tree itself (the document is
carried by the repository) or a private store that mirrors the tree's paths (the document is carried privately).
The relative sidecar path is the same in both.

What a document holds:

- ``identity``: stable across commits (``island:<repository>/<path>``). The repository scan keys its records by
  commit; an island must outlive commits, so it is keyed by what the item IS.
- ``subject``: the item (kind, repository, path).
- ``producer``: how the island's RAW world is produced (a name and parameters). The raw state is someone else's
  structure and problems, the same for everybody. ``None`` means not chosen yet.
- ``edits``: what the occupant changed, as the living map's own edit payload, opaque here. ``None`` means the
  island is still raw.
- ``occupant``: who lives there, if anyone.
- ``doors``: the doors this island takes part in (door identity, the lot and the world outside).

This module reads and writes documents. It builds no world and runs no producer.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

ISLAND_SCHEMA = "abstract-ui-island.v1"
ISLAND_SUFFIX = ".island.json"
SUBJECT_KINDS = ("file", "directory")


class IslandError(ValueError):
    """An island document or path that cannot mean what it says."""


def _clean(path: str) -> str:
    parts = PurePosixPath(str(path).replace("\\", "/")).parts
    if any(part in ("..", "/") or ":" in part for part in parts):
        raise IslandError(f"{path!r} is not a path inside the tree")
    return "/".join(part for part in parts if part != ".")


def island_identity(repository: str, path: str) -> str:
    """The identity of the island of ``path`` in ``repository``; the same at every commit."""

    return f"island:{repository}/{_clean(path)}"


def island_sidecar(path: str, *, is_directory: bool) -> str:
    """The tree-relative path of the document carried beside ``path``."""

    clean = _clean(path)
    if is_directory:
        return f"{clean}/{ISLAND_SUFFIX}" if clean else ISLAND_SUFFIX
    if not clean:
        raise IslandError("a file's island needs the file's path")
    parent, _, name = clean.rpartition("/")
    return f"{parent}/.{name}{ISLAND_SUFFIX}" if parent else f".{name}{ISLAND_SUFFIX}"


def is_island_sidecar(path: str) -> bool:
    """True for a path that is an island document, so a tree walk can skip it as an item of its own."""

    name = PurePosixPath(str(path).replace("\\", "/")).name
    return name == ISLAND_SUFFIX or (name.startswith(".") and name.endswith(ISLAND_SUFFIX))


@dataclass(frozen=True, slots=True)
class IslandDocument:
    identity: str
    subject_kind: str
    repository: str
    path: str
    producer: Mapping[str, Any] | None = None
    edits: Mapping[str, Any] | None = None
    occupant: str | None = None
    doors: tuple[Mapping[str, Any], ...] = ()
    previous_identities: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.subject_kind not in SUBJECT_KINDS:
            raise IslandError(f"subject kind must be one of {SUBJECT_KINDS}, not {self.subject_kind!r}")
        if self.identity != island_identity(self.repository, self.path):
            raise IslandError(f"identity {self.identity!r} is not the island of {self.repository}/{self.path}")

    @property
    def raw(self) -> bool:
        """Nobody has changed anything: the island is as its producer makes it."""

        return self.edits is None

    @property
    def sidecar(self) -> str:
        return island_sidecar(self.path, is_directory=self.subject_kind == "directory")

    def to_data(self) -> dict[str, Any]:
        return {
            "schema": ISLAND_SCHEMA, "identity": self.identity,
            "subject": {"kind": self.subject_kind, "repository": self.repository, "path": self.path},
            "producer": None if self.producer is None else dict(self.producer),
            "edits": None if self.edits is None else dict(self.edits),
            "occupant": self.occupant,
            "doors": [dict(door) for door in self.doors],
            "previous_identities": list(self.previous_identities),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "IslandDocument":
        if data.get("schema") != ISLAND_SCHEMA:
            raise IslandError(f"not an island document: schema {data.get('schema')!r}")
        subject = data.get("subject") or {}
        return cls(
            identity=str(data.get("identity")), subject_kind=str(subject.get("kind")),
            repository=str(subject.get("repository")), path=_clean(str(subject.get("path", ""))),
            producer=data.get("producer"), edits=data.get("edits"), occupant=data.get("occupant"),
            doors=tuple(dict(door) for door in data.get("doors") or ()),
            previous_identities=tuple(str(item) for item in data.get("previous_identities") or ()),
        )


def raw_island(repository: str, path: str, *, is_directory: bool) -> IslandDocument:
    """The island every item starts with: nothing chosen, nothing changed, nobody home."""

    clean = _clean(path)
    return IslandDocument(island_identity(repository, clean), "directory" if is_directory else "file",
                          repository, clean)


def relocated(document: IslandDocument, repository: str, path: str) -> IslandDocument:
    """The same island after its item moved: a new identity, with the old one remembered."""

    clean = _clean(path)
    identity = island_identity(repository, clean)
    if identity == document.identity:
        return document
    return replace(document, identity=identity, repository=repository, path=clean,
                   previous_identities=(*document.previous_identities, document.identity))


def _inside(root: Path, relative: str) -> Path:
    root = Path(root).resolve()
    target = (root / relative).resolve()
    if root != target and root not in target.parents:
        raise IslandError(f"{relative!r} leaves the tree {str(root)!r}")
    return target


def load_island(root: str | Path, path: str, *, is_directory: bool) -> IslandDocument | None:
    """The document carried beside ``path`` under ``root``, or ``None`` when the item has none (it is raw)."""

    target = _inside(Path(root), island_sidecar(path, is_directory=is_directory))
    if not target.is_file():
        return None
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except ValueError as error:
        raise IslandError(f"{target} is not readable as an island document: {error}") from None
    return IslandDocument.from_data(data)


def save_island(root: str | Path, document: IslandDocument) -> Path:
    """Write ``document`` beside its item under ``root`` (whole file replaced at once)."""

    target = _inside(Path(root), document.sidecar)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".writing")
    temporary.write_text(json.dumps(document.to_data(), indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return target
