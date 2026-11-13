"""Agent orientation: markdown of real temporary repositories in causal order, filters and position in real SQLite."""
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from meowdoku import core_git, orientation  # noqa: E402
from meowdoku.living_data_query import GraphView  # noqa: E402

WHO = {"user_id": 2, "username": "ann"}


def git(path, *args, when=None):
    env = dict(os.environ, **({"GIT_AUTHOR_DATE": f"{when} +0000", "GIT_COMMITTER_DATE": f"{when} +0000"} if when else {}))
    done = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, env=env)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def repo(path):
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True)
    for key, value in (("user.name", "t"), ("user.email", "t@example.invalid"), ("commit.gpgsign", "false")):
        git(path, "config", key, value)


def commit(path, files, when):
    for name, text in files.items():
        (path / name).parent.mkdir(parents=True, exist_ok=True)
        (path / name).write_text(text, encoding="utf-8")
    git(path, "add", ".")
    git(path, "commit", "-q", "-m", "add", when=when)


@pytest.fixture
def world(tmp_path):
    outer, inner = tmp_path / "core", tmp_path / "core" / "inner"
    repo(outer)
    (outer / ".gitignore").write_text("inner/\n", encoding="utf-8")
    commit(outer, {"AGENTS.md": "Map: [inner](inner/AGENTS.md) then [b](docs/b.md).\n", "docs/b.md": "[a](../AGENTS.md)\n",
                   "notes/late.md": "late\n", "code.py": "x = '[no](y.md)'\n"}, 1_700_000_000)
    commit(outer, {"notes/early.md": "see [late](late.md)\n"}, 1_600_000_000)   # committed later, dated earlier
    repo(inner)
    commit(inner, {"AGENTS.md": "[guide](docs/guide.md) and [web](https://example.invalid/x.md)\n",
                   "docs/guide.md": "back to [the map](../../AGENTS.md)\n", "docs/hazards/one.md": "h\n"}, 1_650_000_000)
    policy = {"repositories": {"outer": {"path": str(outer), "default": "main"},
                               "inner": {"path": str(inner), "default": "main"}}}
    orientation._cache.clear()
    return {"db": sqlite3.connect(tmp_path / "accounts.db", isolation_level=None), "policy": policy}


def paths(docs):
    return [f"{d['repository']}/{d['path']}" for d in docs]


def test_links_in_order_then_history(world):
    docs, notes = orientation.documents(world["db"], world["policy"], **WHO)
    assert notes == {"truncated": [], "unavailable": []}
    # the walk follows the outer map into the inner repository, then its own next link; unreached files come oldest
    # first by the commit that added them, each starting a walk of its own (early leads to late).
    assert paths(docs) == ["outer/AGENTS.md", "inner/AGENTS.md", "inner/docs/guide.md", "outer/docs/b.md",
                           "outer/notes/early.md", "outer/notes/late.md", "inner/docs/hazards/one.md"]
    by = {d["identity"]: d for d in docs}
    assert by["doc:inner/AGENTS.md"]["links"] == ["doc:inner/docs/guide.md"], "a web address is not a document"
    assert by["doc:inner/docs/guide.md"]["links"] == ["doc:outer/AGENTS.md"], "a link climbs back out to the outer map"
    assert [d["kind"] for d in docs] == ["agents", "agents", "doc", "doc", "doc", "doc", "report"]
    calls = world["db"].execute("SELECT COUNT(*) FROM git_calls").fetchone()[0]
    orientation.documents(world["db"], world["policy"], **WHO)
    assert world["db"].execute("SELECT COUNT(*) FROM git_calls").fetchone()[0] == calls + 2, "unchanged heads: one log each"


def test_filters_keep_the_order_and_the_position_is_kept(world):
    db, policy = world["db"], world["policy"]
    assert orientation.state(db, policy, **WHO)["position"] is None
    first = orientation.update(db, policy, {"step": "next"}, **WHO)
    assert first["at"] == "doc:outer/AGENTS.md" and first["position"] == 1 and first["next"] == "doc:inner/AGENTS.md"
    moved = orientation.update(db, policy, {"filters": {"repository": "outer"}}, **WHO)
    assert moved["at"] == "doc:outer/AGENTS.md" and moved["matching"] == 4
    assert orientation.update(db, policy, {"step": "next"}, **WHO)["at"] == "doc:outer/docs/b.md"
    assert orientation.update(db, policy, {"filters": {"prefix": "notes/", "keyword": "LATE"}}, **WHO)["position"] is None
    assert orientation.update(db, policy, {"step": "first"}, **WHO)["at"] == "doc:outer/notes/late.md"
    again = orientation.state(db, policy, **WHO)
    assert again["filters"] == {"keyword": "LATE", "prefix": "notes/"} and again["position"] == 1, "kept in SQL"
    cleared = orientation.update(db, policy, {"clear": True}, **WHO)
    assert cleared["matching"] == 7 and cleared["position"] == 6
    assert orientation.update(db, policy, {"at": "doc:inner/docs/guide.md"}, **WHO)["position"] == 3
    for body in ({"filters": {"kind": "poem"}}, {"filters": {"owner": "x"}}, {"filters": {"repository": "nope"}},
                 {"step": "sideways"}, {"at": "doc:outer/none.md"}, {"at": "x", "step": "next"},
                 {"filters": {}, "clear": True}, {"extra": 1}):
        with pytest.raises(orientation.OrientationError):
            orientation.update(db, policy, body, **WHO)


def test_the_page_is_a_map_the_cursor_walks(world):
    db, policy = world["db"], world["policy"]
    orientation.update(db, policy, {"at": "doc:outer/docs/b.md"}, **WHO)
    found = orientation.page(db, policy, **WHO)
    assert found["at"] == "doc:outer/docs/b.md" and found["next_offset"] is None and found["matching"] == 7
    view = GraphView(found["map"])
    assert "doc:inner/docs/guide.md" in view.nodes
    tracks = found["map"]["tracks"]
    nexts = [(t["source"], t["target"]) for t in tracks if t["relationship"] == "next"]
    assert nexts[0] == ("doc:outer/AGENTS.md", "doc:inner/AGENTS.md") and len(nexts) == 6
    assert {"source": "orientation", "target": "doc:outer/AGENTS.md", "relationship": "begins"} in tracks
    assert {"source": "doc:inner/docs/guide.md", "target": "doc:outer/AGENTS.md", "relationship": "links-to"} in tracks
    objects = {o["identity"]: o for o in found["map"]["world"]["objects"]}
    assert objects["doc:inner/docs/guide.md"]["position"] == 3 and len(objects["doc:inner/docs/guide.md"]["ref"]) == 40
    with pytest.raises(orientation.OrientationError):
        orientation.page(db, policy, offset=50, **WHO)


def test_markdown_operations_declare_their_own_pathspec(world):
    run = lambda operation, **a: core_git.run(world["db"], world["policy"], repository="outer", operation=operation,
                                              arguments=a, **WHO)
    assert "notes/late.md" not in run("markdown_links")["output"], "a file with no links has no lines"
    assert "code.py" not in run("markdown_links")["output"], "only markdown files are searched"
    assert run("markdown_added")["output"].count("\x00") == 2
    with pytest.raises(core_git.GitError):
        run("markdown_links", path="docs")
    empty = core_git.run(world["db"], world["policy"], repository="inner", operation="markdown_links",
                         arguments={"ref": "main"}, **WHO)
    assert "docs/guide.md" in empty["output"]
