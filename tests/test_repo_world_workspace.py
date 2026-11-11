"""Targeted test: a fixture workspace (root repo + child repos + a worktree) is
discovered from git metadata only, cloned fresh in its layout, and scanned."""

import subprocess
from pathlib import Path

import pytest

from src.compiler.repo_world_workspace import (
    MANIFEST_VERSION,
    WorkspaceManifest,
    clone,
    discover,
    github_https_url,
    scan_workspace,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
         "-c", "commit.gpgsign=false", *args],
        capture_output=True, text=True, check=True, encoding="utf-8",
    ).stdout.strip()


def _repo(path: Path, files: dict[str, str], *, remote: tuple[str, str] | None, branch: str = "main") -> Path:
    path.mkdir(parents=True)
    _git(path, "init", "-q", "-b", branch)
    for name, text in files.items():
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", "fixture")
    if remote:
        _git(path, "remote", "add", *remote)
    return path


def test_workspace_is_discovered_cloned_in_layout_and_scanned(tmp_path):
    # "GitHub": bare repositories the clones come from.
    origins = tmp_path / "origins"
    ws = tmp_path / "workspace"
    root_src = _repo(tmp_path / "src_root", {"AGENTS.md": "map\n", ".gitignore": "/*/\n"}, remote=None, branch="nogodsnomasters")
    child_src = _repo(tmp_path / "src_child", {"pkg/a.py": "from . import b\n", "pkg/b.py": "x = 1\n"}, remote=None)
    for name, src in (("root", root_src), ("child", child_src)):
        subprocess.run(["git", "clone", "-q", "--bare", str(src), str(origins / f"{name}.git")], check=True)

    # Local layout: root repo with a child repo, a remote-less repo and a worktree inside it.
    root = _repo(ws, {"AGENTS.md": "map\n", ".gitignore": "/*/\n"}, remote=("origin", (origins / "root.git").as_uri()), branch="nogodsnomasters")
    _repo(ws / "child", {"pkg/a.py": "from . import b\n", "pkg/b.py": "x = 1\n"}, remote=("main", (origins / "child.git").as_uri()))
    _repo(ws / "orphan", {"o.py": "\n"}, remote=None)
    _git(ws / "child", "worktree", "add", "-q", str(ws / "child-wt"), "-b", "wt")
    (ws / "child" / "pkg" / "secret.py").write_text("never\n", encoding="utf-8")  # uncommitted mess
    (ws / "plain").mkdir()

    manifest = discover(ws)
    assert manifest.to_data()["schema"] == MANIFEST_VERSION
    assert [(r.path, r.ref) for r in manifest.repositories] == [(".", "nogodsnomasters"), ("child", "main")]
    assert any(s.startswith("orphan: no remote") for s in manifest.skipped)
    assert any(s.startswith("child-wt: git worktree") for s in manifest.skipped)

    manifest_path = tmp_path / "manifest.json"
    manifest.write(manifest_path)
    assert WorkspaceManifest.read(manifest_path) == manifest

    dest = tmp_path / "fresh"
    assert clone(manifest, dest) == ()
    assert (dest / "AGENTS.md").is_file() and (dest / "child" / "pkg" / "a.py").is_file()
    assert not (dest / "child" / "pkg" / "secret.py").exists()
    assert _git(dest, "rev-parse", "--abbrev-ref", "HEAD") == "nogodsnomasters"
    with pytest.raises(Exception, match="not empty"):
        clone(manifest, dest)

    result = scan_workspace(manifest)
    data = result.to_data()
    paths = [entry["path"] for entry in data["repositories"]]
    assert paths == [".", "child"]
    child = data["repositories"][1]
    assert child["commit"] == _git(child_src, "rev-parse", "HEAD")
    assert len(child["portals"]) == 1 and child["portals"][0]["source_line"] == 1
    assert "pkg/secret.py" not in {room["path"] for room in child["rooms"]}
    root_commit = data["repositories"][0]["commit"]
    assert data["mounts"] == [{
        "path": "child", "region": f"repo:{child['commit']}/dir:.",
        "parent_region": f"repo:{root_commit}/dir:.", "relationship": "contained-by",
    }]
    assert result.notes == ()


def test_github_urls_normalize_to_public_https():
    expected = "https://github.com/sangderenard/geometry.git"
    assert github_https_url("ssh://git@github.com/sangderenard/geometry") == expected
    assert github_https_url("git@github.com:sangderenard/geometry.git") == expected
    assert github_https_url("https://github.com/sangderenard/geometry") == expected
    assert github_https_url("file:///C:/origins/child.git") == "file:///C:/origins/child.git"
