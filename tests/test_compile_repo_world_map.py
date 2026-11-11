import gzip
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import compile_repo_world_map as cm  # noqa: E402


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                   check=True, capture_output=True)


def make_repo(path: Path, extra=False):
    (path / "pkg" / "sub").mkdir(parents=True)
    (path / "web").mkdir()
    (path / "main.py").write_text("import pkg.a\nimport os\n")
    (path / "pkg" / "__init__.py").write_text("")
    (path / "pkg" / "a.py").write_text("from pkg.sub import b\n")
    (path / "pkg" / "sub" / "b.py").write_text("x = 1\n")
    (path / "web" / "app.js").write_text("import './util.js';\nimport 'react';\n")
    (path / "web" / "util.js").write_text("export const u = 1;\n")
    (path / "blob.bin").write_bytes(b"\0\1\2\n\n\n")
    if extra:
        (path / "more.py").write_text("y = 2\n")
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    git(path, "add", "-A")
    git(path, "commit", "-qm", "init")
    return path


def run(args):
    return cm.main([str(a) for a in args])


def load(out, repo="local__src"):
    latest = json.loads((out / repo / "latest.json").read_text())
    return latest, json.loads((out / repo / latest["map"]).read_text())


@pytest.fixture
def src(tmp_path):
    return make_repo(tmp_path / "src")


def test_schema_and_counts(tmp_path, src):
    out = tmp_path / "out"
    assert run(["--out", out, "--repo", src, "--allow-local", "--work", tmp_path / "w"]) == 0
    latest, m = load(out)
    assert m["schema"] == "repo-world-map.v1" and m["repo"] == "local/src"
    assert len(m["commit"]) == 40 and latest["commit"] == m["commit"]
    paths = [r["path"] for r in m["regions"]]
    assert paths == ["", "pkg", "pkg/sub", "web"]
    for i, r in enumerate(m["regions"]):
        assert r["id"] == i and (r["parent"] is None or r["parent"] < i)
    assert [r["parent"] for r in m["regions"]] == [None, 0, 1, 0]
    assert [(r["files"], r["total_files"]) for r in m["regions"]] == [(2, 7), (2, 3), (1, 1), (2, 2)]
    full = ["/".join(filter(None, (m["regions"][r["region"]]["path"], r["name"]))) for r in m["rooms"]]
    assert full == sorted(full) and [r["id"] for r in m["rooms"]] == list(range(7))
    room = dict(zip(full, m["rooms"]))
    assert room["blob.bin"]["lines"] == 0 and room["blob.bin"]["bytes"] == 6
    assert room["main.py"]["lines"] == 2 and room["main.py"]["language"] == "python"
    assert m["regions"][0]["bytes"] == sum(r["bytes"] for r in m["rooms"])
    idx = {f: r["id"] for f, r in room.items()}
    assert m["kinds"] == sorted(m["kinds"]) and len(m["kinds"]) == 2
    got = {(s, t, line) for s, t, line, _ in m["portals"]}
    assert got == {(idx["main.py"], idx["pkg/a.py"], 1), (idx["pkg/a.py"], idx["pkg/sub/b.py"], 1),
                   (idx["web/app.js"], idx["web/util.js"], 1)}
    assert m["portals"] == sorted(m["portals"])
    assert room["web/app.js"]["out"] == 1 and room["web/util.js"]["in"] == 1
    assert m["region_links"] == [[0, 1, 1], [1, 2, 1]]
    assert m["unresolved_count"] >= 2
    assert ["react", 1] in m["unresolved_top"] and ["os", 1] in m["unresolved_top"]
    d = out / "local__src" / m["commit"]
    assert json.loads(gzip.decompress((d / "world.json.gz").read_bytes()))["commit"] == m["commit"]
    s = json.loads((d / "summary.json").read_text())
    assert s["counts"]["rooms"] == 7 and s["languages"]["python"] == 4
    raw = (d / "map.json").read_text()
    assert ", " not in raw and '": ' not in raw


def test_deterministic(tmp_path, src):
    a, b = tmp_path / "a", tmp_path / "b"
    run(["--out", a, "--repo", src, "--allow-local", "--work", tmp_path / "w"])
    run(["--out", b, "--repo", src, "--allow-local", "--work", tmp_path / "w"])
    ma, mb = load(a)[1], load(b)[1]
    ma.pop("generated"), mb.pop("generated")
    assert ma == mb


def test_two_repos_latest_index_and_skip_force(tmp_path, src, capsys):
    other = make_repo(tmp_path / "other", extra=True)
    out, w = tmp_path / "out", tmp_path / "w"
    assert run(["--out", out, "--repo", src, "--repo", other, "--allow-local", "--work", w]) == 0
    index = json.loads((out / "index.json").read_text())
    assert [e["repo"] for e in index] == ["local/other", "local/src"]
    assert index[0]["latest"]["summary"]["counts"]["rooms"] == 8
    assert not list(out.rglob("*.tmp*"))
    commit = index[1]["latest"]["commit"]
    mp = out / "local__src" / commit / "map.json"
    mp.write_text(mp.read_text() + " ")
    size = mp.stat().st_size
    capsys.readouterr()
    assert run(["--out", out, "--repo", src, "--allow-local", "--work", w]) == 0
    assert "(reused)" in capsys.readouterr().out and mp.stat().st_size == size
    assert run(["--out", out, "--repo", src, "--allow-local", "--work", w, "--force"]) == 0
    assert not mp.read_text().endswith(" ")
    assert len(json.loads((out / "index.json").read_text())) == 2


def test_non_github_refused(tmp_path, capsys):
    out = tmp_path / "out"
    assert run(["--out", out, "--repo", "https://example.com/a/b", "--work", tmp_path / "w"]) == 1
    assert "refused" in capsys.readouterr().err
    assert run(["--out", out, "--repo", str(tmp_path), "--work", tmp_path / "w"]) == 1
    assert cm.parse_repo_spec("https://github.com/o/n.git@v1")[1:] == ("v1", "o/n")


def test_dirty_source_and_cleanup(tmp_path, src):
    (src / "main.py").write_text("dirty = True\n")
    (src / "untracked.py").write_text("")
    out, w = tmp_path / "out", tmp_path / "w"
    assert run(["--out", out, "--repo", src, "--allow-local", "--work", w]) == 0
    assert len(load(out)[1]["rooms"]) == 7
    assert list(w.iterdir()) == []
    run(["--out", out, "--repo", src, "--allow-local", "--work", w, "--force", "--keep"])
    assert len(list(w.iterdir())) == 1
