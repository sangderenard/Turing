"""[HERMES] SOUL.md from the account's own orientation over REST (an isolated account cannot read accounts.db)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "examples" / "meowdoku" / "deploy")]

import hermes_soul as S  # noqa: E402


class FakeRest:
    def __init__(self, answer):
        self.answer, self.calls = answer, []

    def call(self, path, body=None, **kw):
        self.calls.append((path, kw.get("headers")))
        return self.answer


def test_soul_is_built_from_the_rest_orientation_with_same_header_and_mcp_line(tmp_path):
    rest = FakeRest({"preamble": "You are {agent}. Be kind.\n"})
    code = S.main(["--account", "gemma", "--hermes-home", str(tmp_path), "--accounts-db", str(tmp_path / "unreadable.db")], rest=rest)
    assert code == 0
    assert rest.calls == [("/api/agent-workspace/orientation?full=1", {"X-Meowdoku-Workspace": "1"})]
    soul = (tmp_path / "SOUL.md").read_text(encoding="utf-8")
    assert soul.startswith(S.HEADER + "\n\nYou are gemma. Be kind.\n\n") and soul.endswith(S.MCP_LINE + "\n")


def test_empty_answer_writes_nothing_and_rest_failure_prints_no_secret(tmp_path, capsys):
    assert S.main(["--account", "gemma", "--hermes-home", str(tmp_path)], rest=FakeRest({})) == 2
    assert not (tmp_path / "SOUL.md").exists()

    class Broken:
        def call(self, *a, **k):
            raise RuntimeError("cannot reach the server")

    assert S.main(["--account", "gemma", "--hermes-home", str(tmp_path), "--rest"], rest=Broken()) == 1
    assert "cannot reach the server" in capsys.readouterr().err


def test_rest_without_secret_or_base_is_refused(tmp_path, capsys):
    assert S.main(["--account", "gemma", "--hermes-home", str(tmp_path), "--rest"]) == 1
    assert "--secret-file" in capsys.readouterr().err
