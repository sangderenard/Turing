"""Targeted test: a small fixture repository scanned into neutral world records."""

import io
import subprocess
import tarfile
from pathlib import Path

import pytest

from src.compiler import repo_world_scan
from src.compiler.repo_world_scan import (
    REPO_WORLD_SCAN_VERSION,
    DirtyCloneError,
    RepoScanError,
    github_clone_url,
    scan_remote_repository,
    declared_import_roots,
    scan_repository,
    scan_tar_stream,
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
    _write(repo, "native/geom.cpp", '#include "geom.hpp"\n')
    _write(repo, "native/geom.hpp", "struct G {};\n")
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
    assert set(regions) == {".", "native", "pkg", "web", "web/lib"}
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
        "native/geom.cpp", "native/geom.hpp",
    }
    assert rooms["pkg/core.py"].region == regions["pkg"].identity
    assert rooms["web/lib/side.js"].region == regions["web/lib"].identity
    assert rooms["README.md"].language == "other"

    # Languages come from the compiler's own table; C++ is known but unlinked.
    assert rooms["native/geom.cpp"].language == "cpp"
    languages = {row["language"]: row for row in scan.to_data()["languages"]}
    # Compiler languages first (with a route), then extractor-only languages (none).
    assert set(languages) == {
        "python", "sympy", "javascript", "glsl", "wgsl", "cpp", "prose",
        "c", "cmake", "html", "powershell", "shell",
    }
    assert languages["prose"]["route"] == "stand-in" and languages["prose"]["suffixes"] == []
    # C++ is a compiler language, linked by the C include extractor plugin.
    assert languages["cpp"]["route"] == "cpp-shell"
    assert languages["cpp"]["linking"] == "src.compiler.repo_world_refs_c"
    assert languages["html"]["route"] is None
    assert languages["html"]["linking"] == "src.compiler.repo_world_refs_html"
    assert languages["python"]["linking"] == "python-import"
    assert languages["javascript"]["linking_readers"] == [
        "es-module-import", "src.compiler.repo_world_refs_js_dynamic",
    ]
    assert scan.to_data()["extractor_languages"] == ["c", "javascript", "python"]
    assert scan.to_data()["import_roots"] == [""]

    # Cross-file reference -> one portal pair with the source line.
    assert _portal_table(scan) == [
        ("app.py", 2, "pkg/core.py", "python-import"),
        ("app.py", 3, "pkg/util.py", "python-import"),
        ("native/geom.cpp", 1, "native/geom.hpp", "c-include"),
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


def test_remote_stream_scan_sees_only_what_the_remote_serves_and_touches_no_disk(tmp_path, monkeypatch):
    repo = _fixture_repo(tmp_path)
    committed = _git(repo, "rev-parse", "HEAD").strip()
    # Local, uncommitted mess that must never reach the map.
    (repo / "secret.txt").write_text("do not expose\n", encoding="utf-8")
    (repo / "app.py").write_text("import pkg.util  # local edit\n", encoding="utf-8")
    # The source is a transient intermediary: nothing may be written anywhere.
    import tempfile
    monkeypatch.setattr(tempfile, "mkdtemp", lambda *a, **k: pytest.fail("a temp dir was made"))
    monkeypatch.setattr(tempfile, "mkstemp", lambda *a, **k: pytest.fail("a temp file was made"))
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))

    scan = scan_remote_repository(repo.as_uri())

    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before
    assert scan.commit == committed
    assert scan.origin == repo.as_uri()
    assert "secret.txt" not in {r.path for r in scan.rooms}
    assert _portal_table(scan)[:2] == [
        ("app.py", 2, "pkg/core.py", "python-import"),
        ("app.py", 3, "pkg/util.py", "python-import"),
    ]
    # The stream lane and the on-disk lane describe the same commit identically;
    # only `origin` differs (the checkout has no remote, the stream names its URL).
    _git(repo, "checkout", "--", "app.py")
    (repo / "secret.txt").unlink()
    streamed, on_disk = scan.to_data(), scan_repository(repo).to_data()
    assert streamed.pop("origin") == repo.as_uri() and on_disk.pop("origin") is None
    assert streamed == on_disk

    assert github_clone_url("sangderenard/turing") == "https://github.com/sangderenard/turing.git"
    with pytest.raises(RepoScanError):
        github_clone_url("C:/dev/Powershell/turing")
    with pytest.raises(RepoScanError, match="no ref named"):
        scan_remote_repository(repo.as_uri(), ref="no-such-branch")


def test_github_shaped_tarball_streams_through_the_same_scan():
    # The GitHub API tarball wraps the tree in one directory and may carry a
    # pax header and symlinks; only regular files become rooms and only
    # linkable files are ever read.
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
        def add(name, data=None, kind=tarfile.REGTYPE):
            info = tarfile.TarInfo(name)
            info.type = kind
            if kind == tarfile.SYMTYPE:
                info.linkname = "a.py"
                archive.addfile(info)
                return
            if kind == tarfile.DIRTYPE:
                archive.addfile(info)
                return
            raw = data.encode("utf-8")
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
        add("owner-repo-abc1234", kind=tarfile.DIRTYPE)
        add("owner-repo-abc1234/a.py", "import b\n")
        add("owner-repo-abc1234/b.py", "x = 1\n")
        add("owner-repo-abc1234/link.py", kind=tarfile.SYMTYPE)
        add("owner-repo-abc1234/blob.bin", "\x00binary never read")
        add("owner-repo-abc1234/sub/wish.dream",
            "/*@turing.segment.v1\nid = w\nlanguage = prose\nauthor = a\n@end*/\nA thing.\n/*@turing.end*/\n")

    buffer.seek(0)
    scan = scan_tar_stream("c" * 40, "https://github.com/owner/repo.git", buffer,
                           mode="r|gz", strip_components=1)
    assert {r.path for r in scan.rooms} == {"a.py", "b.py", "blob.bin", "sub/wish.dream"}
    assert {r.path for r in scan.regions} == {".", "sub"}
    assert _portal_table(scan) == [("a.py", 1, "b.py", "python-import")]
    assert [c.stand_in for c in scan.claims] == ["w"]
    assert any("link.py" in note for note in scan.notes)

class _Fake:
    LANGUAGE = "fakelang"
    SUFFIXES = (".fk",)
    FILENAMES = ("Fakefile",)

    def __init__(self, fn):
        self._fn = fn

    def references(self, path, text, tracked):
        return self._fn(path, text, tracked)


def _plugin_repo(root: Path) -> Path:
    repo = root / "plug"
    repo.mkdir()
    _git(repo, "init", "-q")
    _write(repo, "a.fk", "use b\nuse missing\n")
    _write(repo, "b.fk", "x\n")
    _write(repo, "Fakefile", "y\n")
    _write(repo, "m.js", "import './n.js';\n")
    _write(repo, "n.js", "export const n = 1;\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "plug")
    return repo


def test_extractor_plugins_make_portals_and_unresolved_rows(tmp_path):
    repo = _plugin_repo(tmp_path)

    def refs(path, text, tracked):
        if path == "a.fk":
            return [(1, "fake-use", "b", "b.fk", ""), (2, "fake-use", "missing", None, "not tracked")]
        return []

    scan = scan_repository(repo, extractors=[_Fake(refs)])
    assert _portal_table(scan) == [
        ("a.fk", 1, "b.fk", "fake-use"), ("m.js", 1, "n.js", "es-module-import"),
    ]
    assert {(u.specifier, u.reason) for u in scan.unresolved} == {("missing", "not tracked")}
    assert {r.path: r.language for r in scan.rooms}["a.fk"] == "fakelang"
    assert {r.path: r.language for r in scan.rooms}["Fakefile"] == "fakelang"
    data = scan.to_data()
    assert data["schema"] == "repo-world-scan-v1"
    assert data["extractor_languages"] == ["fakelang", "javascript", "python"]
    assert data["import_roots"] == [""]


def test_extractor_runs_in_addition_to_builtin_on_one_file(tmp_path):
    repo = _plugin_repo(tmp_path)

    class JsToo:
        LANGUAGE = "jstoo"
        SUFFIXES = (".js",)

        def references(self, path, text, tracked):
            return [(1, "other-kind", "n", "n.js", "")] if path == "m.js" else []

    scan = scan_repository(repo, extractors=[JsToo()])
    # Two kinds on one line to one target are ONE pair; the first in sorted order is kept.
    assert _portal_table(scan) == [("m.js", 1, "n.js", "other-kind")] or \
        _portal_table(scan) == [("m.js", 1, "n.js", "es-module-import")]
    assert len(scan.portals) == 1


def test_extractor_failures_are_scan_errors(tmp_path):
    repo = _plugin_repo(tmp_path)

    def boom(path, text, tracked):
        raise ValueError("kaput")

    class Named(_Fake):
        __name__ = "plugin_named"

    with pytest.raises(RepoScanError, match=r"plugin_named failed on Fakefile"):
        scan_repository(repo, extractors=[Named(boom)])
    with pytest.raises(RepoScanError, match="untracked target"):
        scan_repository(repo, extractors=[_Fake(lambda p, t, k: [(1, "k", "z", "nope.fk", "")])])


def test_bad_extractor_module_name_is_a_scan_error(tmp_path, monkeypatch):
    repo = _plugin_repo(tmp_path)
    monkeypatch.setattr(repo_world_scan, "EXTRACTOR_MODULES", ("src.compiler.no_such_extractor_module",))
    with pytest.raises(RepoScanError, match="no_such_extractor_module"):
        scan_repository(repo)


def _roots_repo(root: Path, name: str, files: dict) -> Path:
    repo = root / name
    repo.mkdir()
    _git(repo, "init", "-q")
    base = {
        "app.py": "import meowdoku.core\n",
        "examples/meowdoku/__init__.py": "",
        "examples/meowdoku/core.py": "from . import util\n",
        "examples/meowdoku/util.py": "x = 1\n",
    }
    base.update(files)
    for relative, text in base.items():
        _write(repo, relative, text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "roots")
    return repo


def _app_targets(scan):
    return [t for (s, _l, t, _k) in _portal_table(scan) if s == "app.py"]


def test_import_roots_resolve_names_only_when_declared(tmp_path):
    repo = _roots_repo(tmp_path, "undeclared", {})
    scan = scan_repository(repo)
    assert _app_targets(scan) == []
    assert any(u.specifier == "meowdoku.core" for u in scan.unresolved)
    assert declared_import_roots(repo) == ()

    via_param = scan_repository(repo, import_roots=["examples"])
    assert _app_targets(via_param) == ["examples/meowdoku/core.py"]
    assert via_param.to_data()["import_roots"] == ["", "examples"]
    # Relative imports keep working under a root.
    assert ("examples/meowdoku/core.py", 1, "examples/meowdoku/util.py", "python-import") in _portal_table(via_param)

    with pytest.raises(RepoScanError, match="not a tracked directory"):
        scan_repository(repo, import_roots=["nowhere"])
    with pytest.raises(RepoScanError, match="escapes"):
        scan_repository(repo, import_roots=["../x"])


def test_import_roots_declared_in_pyproject_and_repo_world_toml(tmp_path):
    pyproject = _roots_repo(tmp_path, "pyp", {
        "pyproject.toml": '[tool.pytest.ini_options]\npythonpath = ["examples"]\n',
    })
    assert declared_import_roots(pyproject) == ("examples",)
    assert _app_targets(scan_repository(pyproject, import_roots=declared_import_roots(pyproject))) == [
        "examples/meowdoku/core.py"]

    toml = _roots_repo(tmp_path, "rwt", {"repo_world.toml": 'import_roots = ["examples"]\n'})
    assert declared_import_roots(toml) == ("examples",)
    assert _app_targets(scan_repository(toml, import_roots=declared_import_roots(toml))) == [
        "examples/meowdoku/core.py"]

    ini = _roots_repo(tmp_path, "ini", {"pytest.ini": "[pytest]\npythonpath = . examples\n"})
    assert declared_import_roots(ini) == ("examples",)


def test_import_root_ambiguity_prefers_earliest_root_and_is_noted(tmp_path):
    repo = _roots_repo(tmp_path, "amb", {"meowdoku/__init__.py": "", "meowdoku/core.py": ""})
    scan = scan_repository(repo, import_roots=["examples"])
    assert _app_targets(scan) == ["meowdoku/core.py"]
    assert any("more than one import root" in note for note in scan.notes)
    assert scan_repository(repo, import_roots=["examples"]).to_data() == scan.to_data()
