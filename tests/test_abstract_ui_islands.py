"""Island documents: an item's interior, carried in a private file beside the item."""

import json
import shutil

import pytest

from src.compiler.abstract_ui_islands import (
    ISLAND_SCHEMA, IslandDocument, IslandError, is_island_sidecar, island_identity, island_sidecar,
    load_island, raw_island, relocated, save_island,
)


def test_the_document_sits_beside_its_item():
    assert island_sidecar("a/b/x.py", is_directory=False) == "a/b/.x.py.island.json"
    assert island_sidecar("x.py", is_directory=False) == ".x.py.island.json"
    assert island_sidecar("a/b", is_directory=True) == "a/b/.island.json"
    assert island_sidecar("", is_directory=True) == ".island.json"
    assert island_sidecar(r"a\b\x.py", is_directory=False) == "a/b/.x.py.island.json"
    for path in ("a/b/.x.py.island.json", ".island.json", "deep/.island.json"):
        assert is_island_sidecar(path)
    for path in ("a/b/x.py", "island.json", "a/x.island.json", ".gitignore"):
        assert not is_island_sidecar(path)
    for bad in ("../x.py", "a/../../x.py", "C:/x.py", "/etc/x"):
        with pytest.raises(IslandError):
            island_sidecar(bad, is_directory=False)
    with pytest.raises(IslandError):
        island_sidecar("", is_directory=False)


def test_identity_is_what_the_item_is_not_which_commit():
    assert island_identity("owner/name", "src/x.py") == "island:owner/name/src/x.py"
    assert island_identity("owner/name", "./src//x.py") == island_identity("owner/name", "src/x.py")
    assert "repo:" not in island_identity("owner/name", "src/x.py")


def test_every_item_starts_raw_and_absent(tmp_path):
    assert load_island(tmp_path, "src/x.py", is_directory=False) is None
    island = raw_island("owner/name", "src/x.py", is_directory=False)
    assert island.raw and island.occupant is None and island.producer is None and island.doors == ()
    assert island.subject_kind == "file" and island.sidecar == "src/.x.py.island.json"
    assert raw_island("owner/name", "src", is_directory=True).sidecar == "src/.island.json"


def test_round_trip_and_whole_file_replacement(tmp_path):
    island = raw_island("owner/name", "src/x.py", is_directory=False)
    lived_in = IslandDocument(
        island.identity, "file", "owner/name", "src/x.py",
        producer={"name": "class-world", "parameters": {"depth_down": 1}},
        edits={"objects": {"wall": {"height": 2.5}}, "portal_splats": []}, occupant="albert",
        doors=({"door": "lot/door:island:owner/name/src/x.py", "lot": "lot", "outside": "world:outside"},))
    written = save_island(tmp_path, lived_in)
    assert written == tmp_path / "src" / ".x.py.island.json"
    assert not list((tmp_path / "src").glob("*.writing"))
    data = json.loads(written.read_text(encoding="utf-8"))
    assert data["schema"] == ISLAND_SCHEMA and data["subject"] == {"kind": "file", "repository": "owner/name", "path": "src/x.py"}
    loaded = load_island(tmp_path, "src/x.py", is_directory=False)
    assert loaded == lived_in and not loaded.raw
    save_island(tmp_path, island)                      # back to raw, replacing the file
    assert load_island(tmp_path, "src/x.py", is_directory=False).raw


def test_the_document_travels_with_its_item(tmp_path):
    island = IslandDocument(island_identity("owner/name", "old/x.py"), "file", "owner/name", "old/x.py",
                            edits={"objects": {}}, occupant="albert")
    save_island(tmp_path, island)
    (tmp_path / "new").mkdir()
    for name in ("x.py", ".x.py.island.json"):
        source = tmp_path / "old" / name
        if name == "x.py":
            source.write_text("pass\n", encoding="utf-8")
        shutil.move(str(source), str(tmp_path / "new" / name))
    carried = load_island(tmp_path, "new/x.py", is_directory=False)
    assert carried.identity == "island:owner/name/old/x.py"        # it still says where it came from
    moved = relocated(carried, "owner/name", "new/x.py")
    assert moved.identity == "island:owner/name/new/x.py" and moved.previous_identities == (island.identity,)
    assert moved.edits == island.edits and moved.occupant == "albert"
    assert relocated(moved, "owner/name", "new/x.py") is moved


def test_documents_that_cannot_mean_what_they_say_are_refused(tmp_path):
    with pytest.raises(IslandError):
        IslandDocument("island:owner/name/other.py", "file", "owner/name", "src/x.py")
    with pytest.raises(IslandError):
        IslandDocument(island_identity("o/n", "x"), "room", "o/n", "x")
    with pytest.raises(IslandError):
        IslandDocument.from_data({"schema": "something-else"})
    target = tmp_path / ".x.py.island.json"
    target.write_text("{not json", encoding="utf-8")
    with pytest.raises(IslandError):
        load_island(tmp_path, "x.py", is_directory=False)
    with pytest.raises(IslandError):
        load_island(tmp_path, "../outside.py", is_directory=False)
