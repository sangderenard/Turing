"""Pin the natural-language stand-in: prose blocks, the work table, the tree,
and the room claim the world scan mints for them (2026-11-11)."""

import subprocess
from pathlib import Path

import pytest

from src.compiler.dream_document import (
    DREAM_LANGUAGE_TRANSLATIONS, DreamDocumentError, parse_dream_document,
)
from src.compiler.repo_world_scan import claim_identity, scan_repository


def _block(identity, language, body, **keys):
    header = "".join(f"{key} = {value}\n" for key, value in keys.items())
    return (
        f"/*@turing.segment.v1\nid = {identity}\nlanguage = {language}\n{header}@end*/\n"
        f"{body}\n/*@turing.end*/\n"
    )


WANT = _block("want", "prose", "A function returning the larger of two numbers.", author="albert")
NARROW = _block("narrow", "prose", "Ties return the first argument.", satisfies="want", author="albert")
GOT = _block(
    "got", "python", "def larger(a, b):\n    return a if a >= b else b", satisfies="want", author="claude",
)


def test_prose_is_a_declared_language_routed_to_the_work_table():
    translation = DREAM_LANGUAGE_TRANSLATIONS["prose"]
    assert translation.route == "stand-in"
    assert translation.execution_target == "work-table"
    assert translation.executable is False


def test_open_stand_in_is_irreducible_and_visibly_open():
    document = parse_dream_document(WANT)
    table = document.work_table()
    (entry,) = table.stand_ins
    assert entry.state == "open" and entry.owner == "albert" and entry.contributions == ()
    assert table.open == (entry,)

    (compilation,) = document.compile_sections()
    assert compilation.route == "stand-in"
    assert [node["type"] for node in compilation.graph["nodes"]] == ["SourceSection", "SourceLine"]
    assert any("open on the work table" in item for item in compilation.shortfalls)


def test_code_contribution_satisfies_and_hands_ownership_to_its_author():
    document = parse_dream_document(WANT + NARROW + GOT)
    table = document.work_table().to_data()
    by_block = {item["block"]: item for item in table["stand_ins"]}

    # The description became a tree: refinement and code are its children.
    assert by_block["want"]["state"] == "satisfied"
    assert by_block["want"]["owner"] == "claude"
    assert by_block["want"]["described_by"] == "albert"
    assert [(c["block"], c["kind"]) for c in by_block["want"]["contributions"]] == [
        ("narrow", "refinement"), ("got", "code"),
    ]
    # The refinement is itself an open stand-in with no code under it yet.
    assert by_block["narrow"]["state"] == "open" and by_block["narrow"]["owner"] == "albert"
    assert sorted((e["from"], e["to"], e["role"]) for e in table["edges"]) == [
        ("want", "got", "satisfied-by"), ("want", "narrow", "refined-by"),
    ]

    compilations = {c.block: c for c in document.compile_sections()}
    assert not any("open on the work table" in s for s in compilations["want"].shortfalls)
    assert any("open on the work table" in s for s in compilations["narrow"].shortfalls)
    assert compilations["got"].route == "ast"


def test_malformed_satisfaction_bindings_fail_at_parse_time():
    with pytest.raises(DreamDocumentError, match="not a prose block"):
        parse_dream_document(WANT + _block("x", "python", "pass", satisfies="missing", author="a"))
    with pytest.raises(DreamDocumentError, match="not a prose block"):
        # Satisfying a code block is not a contribution; only prose is a stand-in.
        parse_dream_document(WANT + GOT + _block("y", "python", "pass", satisfies="got", author="a"))
    with pytest.raises(DreamDocumentError, match="names no author"):
        parse_dream_document(WANT + _block("x", "python", "pass", satisfies="want"))
    with pytest.raises(DreamDocumentError, match="cannot satisfy itself"):
        parse_dream_document(_block("loop", "prose", "me", satisfies="loop", author="a"))
    with pytest.raises(DreamDocumentError, match="cycle"):
        parse_dream_document(
            _block("a", "prose", "a", satisfies="b", author="x")
            + _block("b", "prose", "b", satisfies="a", author="x")
        )


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
         "-c", "commit.gpgsign=false", *args],
        capture_output=True, text=True, check=True, encoding="utf-8",
    ).stdout


def test_scan_mints_a_claimed_room_per_stand_in(tmp_path):
    repo = tmp_path / "fixture"
    (repo / "plans").mkdir(parents=True)
    _git(repo, "init", "-q")
    (repo / "plans" / "larger.dream").write_bytes((WANT + GOT).encode("utf-8"))
    (repo / "plans" / "wish.dream").write_bytes(
        _block("wish", "prose", "A room that fabricates gears.").encode("utf-8")
    )
    (repo / "plans" / "broken.dream").write_bytes(b"not framed\n")
    (repo / "main.py").write_text("x = 1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "fixture")
    commit = _git(repo, "rev-parse", "HEAD").strip()

    scan = scan_repository(repo)
    rooms = {room.path: room for room in scan.rooms}
    assert rooms["plans/larger.dream"].language == "dream-document"

    claims = {claim.stand_in: claim for claim in scan.claims}
    assert set(claims) == {"want", "wish"}
    want = claims["want"]
    assert want.identity == claim_identity(commit, "plans/larger.dream", "want")
    assert want.document_room == rooms["plans/larger.dream"].identity
    assert want.region == rooms["plans/larger.dream"].region
    assert want.state == "satisfied" and want.owner == "claude" and want.described_by == "albert"
    assert [c["block"] for c in want.contributions] == ["got"]

    # Claimed before any content exists, with no author yet: open and unowned.
    wish = claims["wish"]
    assert wish.state == "open" and wish.owner is None and wish.described_by is None
    data = wish.to_data()
    assert data["kind"] == "claimed-room"
    assert {d["relationship"] for d in data["dependencies"]} == {"contained-by", "claimed-from"}

    assert any("plans/broken.dream" in note for note in scan.notes)
    assert scan.to_data()["claims"] == [c.to_data() for c in scan.claims]
    assert scan_repository(repo).to_data() == scan.to_data()
