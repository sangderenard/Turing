"""Publication checks at the shared account boundary."""
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from meowdoku import authored_pages as pages, page_abi
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


def test_bake_retains_private_dev_source_and_freezes_public_copy(db):
    draft = pages.upload(db, 1, title="Draft", description="", content=b"<h1>Page</h1>",
                         content_type="text/html")
    page_id = draft["id"]
    baked = pages.bake(db, 1, page_id)
    assert baked["id"] != page_id
    assert pages.bake(db, 1, page_id)["id"] == baked["id"]
    assert [p["id"] for p in pages.listing(db, site="release")] == [baked["id"]]
    assert pages.descriptor(db, baked["id"], site="release")["page_contract"] is None
    assert pages.read(db, baked["id"], site="release")[0] == b"<h1>Page</h1>"
    with pytest.raises(pages.PageError) as hidden:
        pages.descriptor(db, page_id, site="release")
    assert hidden.value.status == 404
    with pytest.raises(pages.PageError, match="immutable"):
        pages.stage(db, 1, baked["id"], "dev")
    with pytest.raises(pages.PageError):
        pages.bake(db, 2, page_id)
    assert pages.read(db, page_id, site="dev", actor=1)[0] == b"<h1>Page</h1>"


def test_uploaded_page_can_declare_same_contract(db):
    contract = {"schema": page_abi.SCHEMA,
                "presentation": {"title": "Upload", "summary": "", "poster": None, "surface": "dom"},
                "execution": {"standalone": "unknown", "network": "unknown", "compute": [], "site_endpoints": []},
                "access": {"required_capabilities": []}, "controls": {"local": None, "server": None}}
    draft = pages.upload(db, 1, title="Upload", description="", content=b"hello",
                         content_type="text/plain", page_contract=contract)
    release = pages.bake(db, 1, draft["id"])
    assert pages.descriptor(db, release["id"], site="release")["page_contract"] == contract


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


def test_page_contract_is_sealed_for_feed_and_release(db, tmp_path):
    workspace = tmp_path / "workspaces"
    bundle = workspace / "ws2" / "site" / "programs" / "game" / "versions" / "v1"
    bundle.mkdir(parents=True)
    html, poster = b"<h1>Game</h1>", b"static-poster"
    (bundle / "index.html").write_bytes(html)
    (bundle / "poster.webp").write_bytes(poster)
    contract = {"schema": page_abi.SCHEMA,
                "presentation": {"title": "Game", "summary": "A game", "poster": "poster.webp", "surface": "canvas"},
                "execution": {"standalone": "unknown", "network": "optional", "compute": ["browser.webgpu"],
                              "site_endpoints": ["photo_picture_ask"]},
                "access": {"required_capabilities": ["photo_submit"]},
                "controls": {"local": "site-nav.registerControlSuite", "server": None}}
    manifest = {"schema": "turing-program-bundle-v1",
                "page": {"path": "index.html", "url": "/site/programs/game/versions/v1/index.html"},
                "program": {"title": "Game"}, "page_contract": contract,
                "artifacts": [{"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                              for name, data in (("index.html", html), ("poster.webp", poster))]}
    manifest_path = bundle / "bundle.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    db.execute("CREATE TABLE workspace_volumes(volume_id TEXT PRIMARY KEY,root TEXT NOT NULL)")
    db.execute("CREATE TABLE workspace_leases(workspace_id TEXT PRIMARY KEY,volume_id TEXT,owner_id INTEGER,state TEXT,measured_bytes INTEGER)")
    db.execute("INSERT INTO workspace_volumes VALUES('v',?)", (str(workspace),))
    db.execute("INSERT INTO workspace_leases VALUES('ws2','v',1,'active',?)",
               (len(html) + len(poster) + manifest_path.stat().st_size + 1000,))
    db.commit()
    page_id = pages.register_bundle(db, 1, "ws2", bundle)["id"]
    assert pages.listing(db, site="dev", actor=1)[0]["page_contract"] == contract
    assert pages.read(db, page_id, site="dev", actor=1, path="poster.webp")[0] == poster
    manifest["page_contract"]["execution"]["network"] = "required"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(pages.PageError, match="bundle changed since registration"):
        pages.register_bundle(db, 1, "ws2", bundle)
    with pytest.raises(pages.PageError, match="bundle changed since registration"):
        pages.stage(db, 1, page_id, "release")
    manifest["page_contract"]["execution"]["network"] = "optional"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    baked = pages.bake(db, 1, page_id)
    assert pages.read_bundle_url(db, "/site/programs/game/versions/v1/poster.webp", site="release")[0] == poster
    assert pages.descriptor(db, baked["id"], site="release")["page_contract"] == contract
    pages.stage(db, 1, page_id, "release")
    manifest["page_contract"]["execution"]["network"] = "required"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert pages.listing(db, site="release")[0]["page_contract"]["execution"]["network"] == "optional"


def test_page_contract_refuses_unsupported_claims():
    contract = {"schema": page_abi.SCHEMA,
                "presentation": {"title": "Game", "summary": "", "poster": "missing.webp", "surface": "canvas"},
                "execution": {"standalone": "unknown", "network": "unknown", "compute": [], "site_endpoints": []},
                "access": {"required_capabilities": []}, "controls": {"local": None, "server": None}}
    with pytest.raises(page_abi.PageABIError, match="poster"):
        page_abi.validate(contract, {"index.html": (1, "0" * 64)})
    contract["presentation"]["poster"] = None
    contract["execution"]["standalone"] = "verified"
    with pytest.raises(page_abi.PageABIError, match="execution"):
        page_abi.validate(contract, {})


def test_existing_pages_gain_nullable_contract_column(tmp_path):
    path = tmp_path / "old-accounts.db"
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE users(id INTEGER PRIMARY KEY,username TEXT NOT NULL)")
    old.execute("INSERT INTO users VALUES(1,'alice')")
    old.execute("""CREATE TABLE authored_pages (
        id TEXT PRIMARY KEY, owner_id INTEGER NOT NULL, title TEXT NOT NULL, description TEXT NOT NULL,
        kind TEXT NOT NULL, stage TEXT NOT NULL, content BLOB, content_type TEXT,
        workspace_id TEXT, bundle_root TEXT, bundle_route TEXT, bundle_inventory TEXT,
        created_at REAL NOT NULL, updated_at REAL NOT NULL)""")
    old.execute("INSERT INTO authored_pages VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                ("a" * 32, 1, "Old page", "", "upload", "release", b"hello", "text/plain",
                 None, None, None, None, 1.0, 1.0))
    old.execute("CREATE UNIQUE INDEX authored_pages_bundle ON authored_pages(bundle_root) WHERE bundle_root IS NOT NULL")
    old.execute("CREATE UNIQUE INDEX authored_pages_bundle_route ON authored_pages(bundle_route) WHERE bundle_route IS NOT NULL")
    old.commit()
    old.close()
    current = pages.connect(path)
    try:
        assert current.execute("SELECT version FROM schema_version WHERE module='authored_pages'").fetchone()[0] == 3
        assert not any(row[2] for row in current.execute("PRAGMA index_list(authored_pages)")
                       if row[1] in ("authored_pages_bundle", "authored_pages_bundle_route"))
        assert pages.listing(current, site="release")[0]["page_contract"] is None
        assert pages.read(current, "a" * 32, site="release")[0] == b"hello"
    finally:
        current.close()


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
