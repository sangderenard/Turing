"""Streamed, compressed, levelled identity-book logs (no compiler run)."""

from __future__ import annotations

import glob
import lzma
import os

from src.compiler.identity_concordance import (
    IdentityBook,
    IdentityLogLevel,
    Registry,
    RowField,
    RowFieldKind,
    group_by_prefix,
    iter_identity_book_lines,
    render_identity_book,
    render_row,
    resolve_identity_log_level,
    row_value_id,
    write_identity_log,
)

_LONG = "x" * 1000


def _old_render(book) -> str:
    """The pre-streaming ``render_identity_book``, verbatim."""
    lines = [f"identity book: {len(book.pages)} page(s)"]
    for page_name in sorted(book.pages):
        page = book.pages[page_name]
        rows = page.rows()
        lines.append(f"[{page_name}] {len(rows)} row(s), {len(page.cells)} cell(s)")
        by_group: dict = {}
        for row in rows:
            value_id = row_value_id(row)
            group = (
                "unkeyed" if value_id is None
                else (group_by_prefix([value_id])[0].label)
            )
            by_group.setdefault(group, []).append(row)
        for group in sorted(by_group):
            group_rows = by_group[group]
            if len(by_group) > 1:
                lines.append(f"  ({group}) {len(group_rows)} row(s)")
            for row in group_rows:
                spans = page.spans(row)
                trail = " -> ".join(
                    f"{start}..{end}={fact}" for start, end, fact in spans
                )
                lines.append(f"  {render_row(row)}: {trail}")
    return "\n".join(lines)


def _book() -> IdentityBook:
    registry = Registry()
    registry.declare_page(
        "light", (RowField("scope", RowFieldKind.SCOPE),
                  RowField("id", RowFieldKind.VALUE_ID)), str)
    registry.declare_page(
        "heavy", (RowField("scope", RowFieldKind.SCOPE),
                  RowField("id", RowFieldKind.VALUE_ID)), str,
        rows_level=IdentityLogLevel.FULL)
    book = IdentityBook(registry=registry)
    light = book.page("light")
    light.set(("f", 1), 0, "first")
    light.set(("f", 1), 1, "second")
    light.set(("f", 1), 2, _LONG)
    light.set(("f", 2), 0, "only")
    heavy = book.page("heavy")
    heavy.set(("f", 7), 0, "heavy-fact")
    book.page("undeclared").set(("g", 3), 0, "raw")
    return book


def test_full_is_byte_identical_to_the_old_render():
    book = _book()
    assert render_identity_book(book) == _old_render(book)
    assert "\n".join(iter_identity_book_lines(book)) == _old_render(book)


def test_off_prints_nothing():
    assert list(iter_identity_book_lines(_book(), IdentityLogLevel.OFF)) == []


def test_summary_has_page_lines_and_no_rows():
    text = "\n".join(iter_identity_book_lines(
        _book(), IdentityLogLevel.SUMMARY, extra_lines=("detector findings: 0",)))
    assert "log level: summary" in text
    assert "[light] 2 row(s)" in text and "[heavy] 1 row(s)" in text
    assert "second" not in text and "heavy-fact" not in text
    assert "unsourced: latch" in text
    assert text.endswith("detector findings: 0")


def test_facts_keeps_last_span_truncated_and_withholds_declared_full_pages():
    text = "\n".join(iter_identity_book_lines(_book(), IdentityLogLevel.FACTS))
    assert "first" not in text            # history dropped, last span only
    assert "only" in text
    assert "...[1000 chars]" in text and _LONG not in text
    assert "[3 spans]" in text
    assert "heavy-fact" not in text       # page declared FULL
    assert "[heavy] 1 row(s)" in text and "rows withheld below full" in text
    assert "raw" in text                  # undeclared page defaults to FACTS


def test_xz_round_trip_and_final_name(tmp_path):
    book = _book()
    path = write_identity_log(
        book, str(tmp_path / "stem"), level="full", kind="ok")
    assert path == str(tmp_path / "stem.ok.log.xz")
    with lzma.open(path, "rt", encoding="utf-8", newline="\n") as handle:
        assert handle.read() == render_identity_book(book) + "\n"
    assert os.listdir(tmp_path) == ["stem.ok.log.xz"]


def test_atomic_write_leaves_no_partial_file(tmp_path):
    book = _book()
    original = book.page("light").rows

    def explode():
        raise RuntimeError("boom mid-stream")

    book.pages["light"].rows = explode
    assert write_identity_log(
        book, str(tmp_path / "stem"), level=IdentityLogLevel.FULL, kind="failed"
    ) is None
    assert os.listdir(tmp_path) == []
    book.pages["light"].rows = original


def test_off_writes_nothing(tmp_path):
    assert write_identity_log(_book(), str(tmp_path / "s"), level="off") is None
    assert os.listdir(tmp_path) == []


def test_preset_override(tmp_path, monkeypatch):
    from src.compiler.identity_concordance import identity_log_lzma_preset

    assert identity_log_lzma_preset() == 3
    monkeypatch.setenv("TURING_IDENTITY_LOG_LZMA_PRESET", "6e")
    assert identity_log_lzma_preset() == (6 | lzma.PRESET_EXTREME)
    monkeypatch.setenv("TURING_IDENTITY_LOG_LZMA_PRESET", "bogus")
    assert identity_log_lzma_preset() == 3


def test_env_override_and_defaults(monkeypatch):
    monkeypatch.delenv("TURING_IDENTITY_LOG_LEVEL", raising=False)
    assert resolve_identity_log_level(None, ok=True) is IdentityLogLevel.SUMMARY
    assert resolve_identity_log_level(None, ok=False) is IdentityLogLevel.FULL
    monkeypatch.setenv("TURING_IDENTITY_LOG_LEVEL", "Facts")
    assert resolve_identity_log_level(None, ok=True) is IdentityLogLevel.FACTS
    assert resolve_identity_log_level(None, ok=False) is IdentityLogLevel.FACTS
    assert resolve_identity_log_level("off", ok=False) is IdentityLogLevel.OFF
    monkeypatch.setenv("TURING_IDENTITY_LOG_LEVEL", "nonsense")
    assert resolve_identity_log_level(None, ok=True) is IdentityLogLevel.SUMMARY


def test_dump_ok_is_summary_and_failed_is_full(tmp_path, monkeypatch):
    from src.compiler.fortran_c_shell import _dump_identity_book_log

    monkeypatch.delenv("TURING_IDENTITY_LOG_LEVEL", raising=False)
    monkeypatch.chdir(tmp_path)
    book = _book()
    _dump_identity_book_log(book, name="p", entrypoint=None, ok=True,
                            extra_lines=("detector findings: 0",))
    _dump_identity_book_log(book, name="p", entrypoint=None, ok=False)
    directory = tmp_path / "artifacts" / "identity_logs"
    (ok,) = glob.glob(str(directory / "p.*.ok.log.xz"))
    (failed,) = glob.glob(str(directory / "p.*.failed.log.xz"))
    with lzma.open(ok, "rt", encoding="utf-8") as handle:
        ok_text = handle.read()
    with lzma.open(failed, "rt", encoding="utf-8") as handle:
        failed_text = handle.read()
    assert "log level: summary" in ok_text and "heavy-fact" not in ok_text
    assert "detector findings: 0" in ok_text
    assert "heavy-fact" in failed_text and "log level" not in failed_text
