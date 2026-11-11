"""Clone the whole workspace from GitHub in its on-disk layout, then scan it.

The owner's workspace is one root repository (``nogodsnomasters``) whose
child directories are independent repositories with their own GitHub
remotes.  The world map must come from GitHub, never from the local disk, so
this module has three steps:

1. ``discover``  reads ONLY git metadata of the local tree (which directories
   are repositories, each one's remote URL and current branch) and writes a
   manifest.  No file contents are read.  Git worktrees (a ``.git`` file, not
   a directory) are skipped: they are views of another entry.  A repository
   without a remote is skipped with a note: GitHub cannot serve it.
2. ``clone``     recreates the layout under a fresh destination directory
   from the manifest with shallow clones (one commit each, Git LFS smudge
   disabled), root first, children inside it.  Nothing is copied from disk.
3. ``scan``      runs :func:`repo_world_scan.scan_repository` on every clone
   and returns one workspace record set: each repository's scan plus a
   ``mounts`` table saying which root-repository region each child
   repository's root region sits inside.

    python -m src.compiler.repo_world_workspace discover C:/dev/Powershell docs/repo_world_workspace.json
    python -m src.compiler.repo_world_workspace clone docs/repo_world_workspace.json <dest>
    python -m src.compiler.repo_world_workspace scan docs/repo_world_workspace.json [--json]
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
from typing import Any, Iterable

from .repo_world_scan import (
    RepoScanError, RepoWorldScan, region_identity, scan_repository,
)

REPO_WORLD_WORKSPACE_VERSION = "repo-world-workspace-v0"
MANIFEST_VERSION = "repo-world-workspace-manifest-v0"

_GITHUB_FORMS = (
    re.compile(r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?/?$"),
    re.compile(r"^(?:ssh://)?git@github\.com[:/](?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?/?$"),
)


@dataclass(frozen=True, slots=True)
class WorkspaceRepository:
    """One repository in the layout.  ``path`` is '.' for the root."""

    path: str
    clone_url: str
    ref: str | None

    def to_data(self) -> dict[str, Any]:
        return {"path": self.path, "clone_url": self.clone_url, "ref": self.ref}


@dataclass(frozen=True, slots=True)
class WorkspaceManifest:
    repositories: tuple[WorkspaceRepository, ...]
    skipped: tuple[str, ...] = ()

    def to_data(self) -> dict[str, Any]:
        return {
            "schema": MANIFEST_VERSION,
            "repositories": [item.to_data() for item in self.repositories],
            "skipped": list(self.skipped),
        }

    @classmethod
    def from_data(cls, data: dict[str, Any]) -> "WorkspaceManifest":
        if data.get("schema") != MANIFEST_VERSION:
            raise RepoScanError(f"manifest schema {data.get('schema')!r} is not {MANIFEST_VERSION}")
        return cls(
            tuple(WorkspaceRepository(str(item["path"]), str(item["clone_url"]),
                                      item.get("ref")) for item in data["repositories"]),
            tuple(str(item) for item in data.get("skipped", ())),
        )

    def write(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_data(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: str | Path) -> "WorkspaceManifest":
        return cls.from_data(json.loads(Path(path).read_text(encoding="utf-8")))


def github_https_url(url: str) -> str:
    """Normalize any GitHub spelling to the public HTTPS clone URL.

    A URL that is not GitHub is returned unchanged (a fixture ``file://`` clone
    in tests, say); GitHub is the only remote the owner publishes to.
    """

    for form in _GITHUB_FORMS:
        match = form.match(url.strip())
        if match:
            return f"https://github.com/{match['owner']}/{match['repo']}.git"
    return url.strip()


def _git(repo: Path, *args: str, allow_failure: bool = False) -> str | None:
    out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                         text=True, encoding="utf-8")
    if out.returncode != 0:
        if allow_failure:
            return None
        raise RepoScanError(f"git {' '.join(args)} failed in {repo}: {out.stderr.strip()}")
    return out.stdout.strip()


def _is_repository_root(directory: Path) -> bool | None:
    """True for a repository, False for a worktree, None for neither."""

    marker = directory / ".git"
    if marker.is_dir():
        return True
    if marker.is_file():
        return False
    return None


def _remote_url(repo: Path) -> str | None:
    names = (_git(repo, "remote", allow_failure=True) or "").split()
    if not names:
        return None
    name = "origin" if "origin" in names else names[0]
    return _git(repo, "remote", "get-url", name, allow_failure=True)


def _current_ref(repo: Path) -> str | None:
    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD", allow_failure=True)
    return None if branch in (None, "HEAD") else branch


def discover(workspace_root: str | Path, *, exclude: Iterable[str] = ()) -> WorkspaceManifest:
    """Read the local layout's git metadata into a manifest.  Reads no file contents."""

    root = Path(workspace_root)
    if _is_repository_root(root) is not True:
        raise RepoScanError(f"{root} is not a repository root")
    excluded = set(exclude)
    repositories: list[WorkspaceRepository] = []
    skipped: list[str] = []

    def add(path: str, repo: Path) -> None:
        url = _remote_url(repo)
        if url is None:
            skipped.append(f"{path}: no remote, GitHub cannot serve it")
            return
        repositories.append(WorkspaceRepository(path, github_https_url(url), _current_ref(repo)))

    add(".", root)
    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if not child.is_dir() or child.name in excluded or child.name.startswith("."):
            continue
        kind = _is_repository_root(child)
        if kind is None:
            continue
        if kind is False:
            skipped.append(f"{child.name}: git worktree of another repository")
            continue
        add(child.name, child)
    return WorkspaceManifest(tuple(repositories), tuple(skipped))


def clone(manifest: WorkspaceManifest, destination: str | Path, *, strict: bool = False) -> tuple[str, ...]:
    """Recreate the layout under ``destination`` with fresh shallow clones.

    Returns the notes for clones that failed (a private or renamed repository,
    a missing ref).  With ``strict`` the first failure raises instead.
    """

    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        raise RepoScanError(f"{dest} is not empty; the workspace clone must start fresh")
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1", GIT_TERMINAL_PROMPT="0")
    notes: list[str] = []
    for item in sorted(manifest.repositories, key=lambda r: (r.path != ".", r.path)):
        target = dest if item.path == "." else dest / PurePosixPath(item.path)
        command = ["git", "clone", "--quiet", "--depth", "1", "--no-tags"]
        if item.ref:
            command += ["--branch", item.ref]
        command += [item.clone_url, str(target)]
        out = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", env=env)
        if out.returncode != 0:
            note = f"{item.path}: clone of {item.clone_url}{' at ' + item.ref if item.ref else ''} failed: {out.stderr.strip()}"
            if strict:
                raise RepoScanError(note)
            notes.append(note)
    return tuple(notes)


@dataclass(frozen=True, slots=True)
class WorkspaceScan:
    scans: tuple[tuple[str, RepoWorldScan], ...]
    notes: tuple[str, ...]

    def to_data(self) -> dict[str, Any]:
        root = next((scan for path, scan in self.scans if path == "."), None)
        mounts = []
        for path, scan in self.scans:
            if path == "." or root is None:
                continue
            parent = str(PurePosixPath(path).parent)
            mounts.append({
                "path": path,
                "region": scan.root_region.identity,
                "parent_region": region_identity(root.commit, parent),
                "relationship": "contained-by",
            })
        return {
            "schema": REPO_WORLD_WORKSPACE_VERSION,
            "repositories": [{"path": path, **scan.to_data()} for path, scan in self.scans],
            "mounts": mounts,
            "notes": list(self.notes),
        }


def scan_workspace_clone(destination: str | Path, manifest: WorkspaceManifest, notes: Iterable[str] = ()) -> WorkspaceScan:
    dest = Path(destination)
    scans = []
    collected = list(notes)
    for item in manifest.repositories:
        target = dest if item.path == "." else dest / PurePosixPath(item.path)
        if not (target / ".git").exists():
            collected.append(f"{item.path}: not cloned, not scanned")
            continue
        scans.append((item.path, scan_repository(target)))
    return WorkspaceScan(tuple(scans), tuple(collected))


def scan_workspace(manifest: WorkspaceManifest) -> WorkspaceScan:
    """Clone the manifest into a temporary directory, scan everything, delete it."""

    with tempfile.TemporaryDirectory(prefix="repo-world-workspace-") as tmp:
        dest = Path(tmp) / "workspace"
        notes = clone(manifest, dest)
        return scan_workspace_clone(dest, manifest, notes)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in args
    args = [a for a in args if a != "--json"]
    usage = "\n".join(__doc__.strip().splitlines()[-3:])
    try:
        if len(args) == 3 and args[0] == "discover":
            manifest = discover(args[1])
            manifest.write(args[2])
            print(f"{len(manifest.repositories)} repositories -> {args[2]}")
            for note in manifest.skipped:
                print(f"  skipped {note}")
            return 0
        if len(args) == 3 and args[0] == "clone":
            notes = clone(WorkspaceManifest.read(args[1]), args[2])
            for note in notes:
                print(f"  failed {note}")
            return 1 if notes else 0
        if len(args) == 2 and args[0] == "scan":
            result = scan_workspace(WorkspaceManifest.read(args[1]))
            if as_json:
                json.dump(result.to_data(), sys.stdout, indent=2)
                print()
                return 0
            for path, scan in result.scans:
                print(f"{path:30s} {scan.commit[:12]}  regions {len(scan.regions):5d}  rooms {len(scan.rooms):6d}  portals {len(scan.portals):6d}")
            for note in result.notes:
                print(f"  note: {note}")
            return 0
    except RepoScanError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    print(usage, file=sys.stderr)
    return 2


__all__ = [
    "MANIFEST_VERSION", "REPO_WORLD_WORKSPACE_VERSION", "WorkspaceManifest",
    "WorkspaceRepository", "WorkspaceScan", "clone", "discover", "github_https_url",
    "scan_workspace", "scan_workspace_clone",
]


if __name__ == "__main__":
    sys.exit(main())
