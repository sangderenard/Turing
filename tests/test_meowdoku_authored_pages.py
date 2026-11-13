"""Publication checks at the shared account boundary."""
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from meowdoku import authored_pages as pages
from meowdoku.game_pages import GAMES, export_game_pages
from meowdoku import workspace_accounting
from meowdoku.register_site_bundle import main as register_site_bundle


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "accounts.db"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE users(id INTEGER PRIMARY KEY,username TEXT NOT NULL)")
    connection.executemany("INSERT INTO users VALUES(?,?)", ((1, "alice"), (2, "bob")))
    connection.commit()
    connection.close()
    result = pages.connect(path)
    yield result
    result.close()


def test_dev_page_is_author_only_and_release_is_public(db):
    page = pages.upload(db, 1, title="Alice's page", description="First", content=b"<h1>Hello</h1>",
                        content_type="text/html")
    key = page["id"]
    assert pages.listing(db, site="release") == []
    assert pages.listing(db, site="dev", actor=2) == []
    assert [p["id"] for p in pages.listing(db, site="dev", actor=1)] == [key]
    with pytest.raises(pages.PageError) as hidden:
        pages.read(db, key, site="release")
    assert hidden.value.status == 404
    with pytest.raises(pages.PageError) as hidden:
        pages.read(db, key, site="dev", actor=2)
    assert hidden.value.status == 404
    assert pages.read(db, key, site="dev", actor=1)[0] == b"<h1>Hello</h1>"
    with pytest.raises(pages.PageError):
        pages.stage(db, 2, key, "release")
    pages.stage(db, 1, key, "release")
    assert pages.listing(db, site="release", author="alice")[0]["author"] == "alice"
    assert pages.read(db, key, site="release")[0] == b"<h1>Hello</h1>"
    pages.stage(db, 1, key, "dev")
    with pytest.raises(pages.PageError) as hidden:
        pages.read(db, key, site="release")
    assert hidden.value.status == 404


def test_upload_requires_separate_bounded_utf8_public_copy(db):
    for data in (b"", b"\xff", b"x" * (pages.MAX_UPLOAD + 1)):
        with pytest.raises(pages.PageError):
            pages.upload(db, 1, title="Page", description="", content=data, content_type="text/html")
    assert pages.listing(db, site="dev", actor=1) == []


def test_compiled_bundle_registration_requires_owned_measured_workspace(db, tmp_path):
    workspace = tmp_path / "workspaces"
    bundle = workspace / "ws1" / "site" / "programs" / "x" / "versions" / "v1"
    bundle.mkdir(parents=True)
    html = b"<h1>Compiled</h1>"
    (bundle / "index.html").write_bytes(html)
    manifest = {"schema": "turing-program-bundle-v1", "page": {"path": "index.html", "url": "/site/programs/x/versions/v1/index.html"},
                "program": {"title": "Compiled"}, "artifacts": [{"path": "index.html", "bytes": len(html),
                "sha256": hashlib.sha256(html).hexdigest()}]}
    (bundle / "bundle.json").write_text(json.dumps(manifest), encoding="utf-8")
    db.execute("CREATE TABLE workspace_volumes(volume_id TEXT PRIMARY KEY,root TEXT NOT NULL)")
    db.execute("CREATE TABLE workspace_leases(workspace_id TEXT PRIMARY KEY,volume_id TEXT,owner_id INTEGER,state TEXT,measured_bytes INTEGER)")
    db.execute("INSERT INTO workspace_volumes VALUES('v',?)", (str(workspace),))
    db.execute("INSERT INTO workspace_leases VALUES('ws1','v',1,'active',?)",
               (len(html) + (bundle / "bundle.json").stat().st_size,))
    db.commit()
    with pytest.raises(pages.PageError):
        pages.register_bundle(db, 2, "ws1", bundle)
    created = pages.register_bundle(db, 1, "ws1", bundle)
    key = created["id"]
    assert pages.register_bundle(db, 1, "ws1", bundle)["id"] == key
    assert pages.read(db, key, site="dev", actor=1)[0] == html
    assert pages.read_bundle_url(db, "/site/programs/x/versions/v1/index.html", site="dev", actor=1)[0] == html
    with pytest.raises(pages.PageError):
        pages.read_bundle_url(db, "/site/programs/x/versions/v1/index.html", site="release")
    (bundle / "index.html").write_bytes(b"changed before release")
    with pytest.raises(pages.PageError) as changed:
        pages.stage(db, 1, key, "release")
    assert changed.value.status == 409
    assert pages.listing(db, site="release") == []
    (bundle / "index.html").write_bytes(html)
    pages.stage(db, 1, key, "release")
    assert pages.read(db, key, site="release")[0] == html
    (bundle / "index.html").write_bytes(b"changed")
    with pytest.raises(pages.PageError) as changed:
        pages.read(db, key, site="release")
    assert changed.value.status == 503


def test_each_game_exports_a_direct_page(tmp_path):
    source = Path(__file__).resolve().parents[1] / "examples" / "meowdoku" / "web" / "play.html"
    (tmp_path / "play.html").write_bytes(source.read_bytes())
    export_game_pages(tmp_path)
    assert len(list(tmp_path.glob("play-*.html"))) == len(GAMES)
    assert 'data-page-game="chess"' in (tmp_path / "play-chess.html").read_text(encoding="utf-8")
    assert '/play-chess.html' in (tmp_path / "games.html").read_text(encoding="utf-8")


def test_trusted_builder_handoff_measures_account_workspace(db, tmp_path, capsys):
    root = tmp_path / "workspaces"
    root.mkdir()
    store = workspace_accounting.Store(tmp_path / "accounts.db", root,
                 capacity_bytes=1024 * 1024, user_quota_bytes=1024 * 1024, free_floor_bytes=0)
    workspace_id = store.reserve(1, 4096, base_revision="a" * 64)
    bundle = root / workspace_id / "site" / "programs" / "cat" / "versions" / "v1"
    bundle.mkdir(parents=True)
    html = b"<h1>Cat bundle</h1>"
    (bundle / "index.html").write_bytes(html)
    (bundle / "bundle.json").write_text(json.dumps({
        "schema": "turing-program-bundle-v1", "page": {"path": "index.html", "url": "/site/programs/cat/versions/v1/index.html"},
        "program": {"title": "Cat bundle"}, "artifacts": [{"path": "index.html", "bytes": len(html),
        "sha256": hashlib.sha256(html).hexdigest()}]}), encoding="utf-8")
    assert register_site_bundle(["--accounts-db", str(tmp_path / "accounts.db"),
                                 "--workspace-id", workspace_id, "--bundle-dir", str(bundle)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["stage"] == "dev"
    assert pages.read(db, result["id"], site="dev", actor=1)[0] == html
    with pytest.raises(workspace_accounting.Refused):
        store.begin_release(1, workspace_id)
