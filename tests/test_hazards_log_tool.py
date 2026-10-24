"""tools/hazards_log.py: entry files sort newest first and never collide."""
import datetime as dt
import importlib.util
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "hazards_log.py"
_spec = importlib.util.spec_from_file_location("hazards_log", _TOOL)
hazards_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hazards_log)


def test_entry_name_sorts_by_time_and_carries_branch():
    t = dt.datetime(2026, 10, 24, 9, 5, 7, tzinfo=dt.timezone.utc)
    assert hazards_log.entry_name("claude/project-thread-x", t) == (
        "20261024T090507Z_claude-project-thread-x.md"
    )


def test_entry_files_newest_first_and_reject_bad_names(tmp_path):
    for name in ("20261024T090000Z_a.md", "20261024T100000Z_b.md",
                 "20261023T235959Z_c.md", "README.md"):
        (tmp_path / name).write_text("x\n")
    assert [p.name for p in hazards_log.entry_files(tmp_path)] == [
        "20261024T100000Z_b.md", "20261024T090000Z_a.md", "20261023T235959Z_c.md",
    ]
    (tmp_path / "notes.md").write_text("x\n")
    with pytest.raises(SystemExit):
        hazards_log.entry_files(tmp_path)


def test_repository_entries_are_well_named():
    hazards_log.entry_files()


def test_legacy_log_header_is_split_from_entries():
    text = hazards_log.LEGACY_LOG.read_text(encoding="utf-8")
    assert hazards_log.LEGACY_HEADER_END in text
    assert "docs/hazards/" not in hazards_log.legacy_entries(text)[:200]
