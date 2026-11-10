"""Targeted test: a small fixture repository scanned into neutral world records."""

import subprocess
from pathlib import Path

import pytest

from src.compiler.repo_world_scan import (
    REPO_WORLD_SCAN_VERSION,
    DirtyCloneError,
    RepoScanError,
    scan_repository,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
         "-c", "commit.gpgsign=false", *args],
        capture_output=True, text=True, check=True, encoding="utf-8",
    ).stdout


def _write(repo: Path, relative: str, text: str) -> None:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _fixture_repo(root: Path) -> Path:
    repo = root / "fixture"
    repo.mkdir()
    _git(repo, "init", "-q")
    _write(repo, "app.py", "import os\nfrom pkg.core import run\nimport pkg.util\n")
    _write(repo, "pkg/__init__.py", "")
    _write(repo, "pkg/core.py", "from . import util\nfrom .util import helper\nimport json\n")
    _write(repo, "pkg/util.py", "def helper():\n    return 1\n")
    _write(repo, "pkg/broken.py", "def (:\n")
    _write(repo, "web/main.js",
           'import {draw} from "./lib/helper.js";\nimport "./lib/side.js";\n'
           "import three from 'three';\nexport {x} from './lib/helper.js';\n")
    _write(repo, "web/lib/helper.js", "export const draw = 1;\nexport const x = 2;\n")
    _write(repo, "web/lib/side.js", "import '../main.js';\n")
    _write(repo, "README.md", "fixture\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "fixture")
    return repo


def _portal_table(scan):
    def short(identity: str) -> str:
        return identity.split("/file:", 1)[1]
    return sorted(
        (short(p.source_room), p.source_line, short(p.target_room), p.reference_kind)
        for p in scan.portals
    )


def test_fixture_repository_scans_into_regions_rooms_and_portal_pairs(tmp_path):
    repo = _fixture_repo(tmp_path)
    commit = _git(repo, "rev-parse", "HEAD").strip()

    scan = scan_repository(repo)

    assert scan.commit == commit
    assert scan.to_data()["schema"] == REPO_WORLD_SCAN_VERSION
    assert scan.to_data()["identity"] == f"repo:{commit}"

    # Folder -> region / subregion, parented through the directory tree.
    regions = {r.path: r for r in scan.regions}
    assert set(regions) == {".", "pkg", "web", "web/lib"}
    assert regions["."].parent is None
    assert regions["pkg"].parent == regions["."].identity
    assert regions["web/lib"].parent == regions["web"].identity
    assert regions["web/lib"].depth == 2
    assert all(r.identity.startswith(f"repo:{commit}/dir:") for r in scan.regions)

    # File -> room, one per tracked file, inside its directory's region.
    rooms = {r.path: r for r in scan.rooms}
    assert set(rooms) == {
        "README.md", "app.py", "pkg/__init__.py", "pkg/broken.py", "pkg/core.py",
        "pkg/util.py", "web/lib/helper.js", "web/lib/side.js", "web/main.js",
    }
    assert rooms["pkg/core.py"].region == regions["pkg"].identity
    assert rooms["web/lib/side.js"].region == regions["web/lib"].identity
    assert rooms["README.md"].language == "other"

    # Cross-file reference -> one portal pair with the source line.
    assert _portal_table(scan) == [
        ("app.py", 2, "pkg/core.py", "python-import"),
        ("app.py", 3, "pkg/util.py", "python-import"),
        ("pkg/core.py", 1, "pkg/util.py", "python-import"),
        ("pkg/core.py", 2, "pkg/util.py", "python-import"),
        ("web/lib/side.js", 1, "web/main.js", "es-module-import"),
        ("web/main.js", 1, "web/lib/helper.js", "es-module-import"),
        ("web/main.js", 2, "web/lib/side.js", "es-module-import"),
        ("web/main.js", 4, "web/lib/helper.js", "es-module-import"),
    ]
    pair = next(p for p in scan.portals if p.source_line == 2 and p.source_room.endswith("app.py"))
    assert pair.out_end.room == pair.source_room and pair.out_end.role == "out"
    assert pair.in_end.room == pair.target_room and pair.in_end.role == "in"
    assert pair.out_end.pair == pair.in_end.pair == pair.identity
    assert pair.out_end.transmission is None and pair.in_end.transmission is None
    assert pair.specifier == "pkg.core:run"

    # References that are not tracked files stay visible as unresolved, never portals.
    unresolved = {(u.source_room.split("/file:", 1)[1], u.source_line, u.specifier) for u in scan.unresolved}
    assert ("app.py", 1, "os") in unresolved
    assert ("pkg/core.py", 3, "json") in unresolved
    assert ("web/main.js", 3, "three") in unresolved
    assert any("pkg/broken.py" in note for note in scan.notes)

    # Deterministic for the commit.
    assert scan_repository(repo).to_data() == scan.to_data()


def test_scan_refuses_uncommitted_and_untracked_changes(tmp_path):
    repo = _fixture_repo(tmp_path)

    (repo / "app.py").write_text("import os\n", encoding="utf-8")
    with pytest.raises(DirtyCloneError, match="app.py"):
        scan_repository(repo)
    _git(repo, "checkout", "--", "app.py")

    (repo / "stray.txt").write_text("x\n", encoding="utf-8")
    with pytest.raises(DirtyCloneError, match="stray.txt"):
        scan_repository(repo)
    (repo / "stray.txt").unlink()

    assert scan_repository(repo).commit == _git(repo, "rev-parse", "HEAD").strip()

    with pytest.raises(RepoScanError):
        scan_repository(repo / "pkg")
