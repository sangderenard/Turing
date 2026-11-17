"""[GUIDE] Llama's chat carries its own orientation and memory, fitted to the context, and no reaction bait unless training."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "examples" / "meowdoku" / "deploy" / "hosted_llm_runner.py"
sys.path[:0] = [str(ROOT), str(ROOT / "examples"), r"C:\dev\Powershell\speaktome"]


def _runner():
    spec = importlib.util.spec_from_file_location("hosted_llm_runner_guide_t", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Rest:
    def __init__(self, fail=False):
        self.fail, self.calls = fail, []

    def call(self, path, body=None, **kw):
        self.calls.append(path)
        if self.fail:
            raise OSError("site away")
        if path.startswith("/api/agent-workspace/orientation"):
            return {"preamble": "# Llama brief\nYou are Llama, a helpful assistant."}
        if path == "/api/memory":
            return {"files": [{"id": "a" * 32, "updated": 1}, {"id": "b" * 32, "updated": 9}]}
        if path == "/api/memory/file/" + "b" * 32:
            return {"body": "Bob asked about tickets yesterday."}
        raise AssertionError(path)


def test_the_guide_reads_orientation_and_the_newest_memory_and_keeps_them_when_the_site_fails():
    runner = _runner()
    rest = Rest()
    guide = runner.Guide(rest)
    guide.refresh(now=1000.0)
    assert "a helpful assistant" in guide.setting() and runner.RUNNER_RULES in guide.setting()
    assert guide.memory == "Bob asked about tickets yesterday.", "the most recently updated memory file"
    assert rest.calls[0] == "/api/agent-workspace/orientation?full=1"
    guide.refresh(now=1000.0 + runner.GUIDE_SECONDS - 1)
    assert len(rest.calls) == 3, "read again only every GUIDE_SECONDS (orientation, memory list, memory file)"
    rest.fail = True
    guide.refresh(now=1000.0 + runner.GUIDE_SECONDS + 1)
    assert "a helpful assistant" in guide.setting() and guide.memory, "a failed read keeps the last values"
    empty = runner.Guide(Rest(fail=True))
    empty.refresh(now=1.0)
    assert empty.setting().startswith(runner.FALLBACK_SETTING) and "coach" not in empty.setting().lower()


def test_chat_keeps_the_whole_setting_and_trims_chat_lines_before_giving_it_up():
    runner = _runner()
    model = runner.LocalModel(Path("x.exe"), Path("m.gguf"), None, 1, Path("."))
    lines = [{"id": i, "username": "bob", "body": "word " * 40} for i in range(30)]
    count = lambda text: len(text.split())                      # one token per word
    messages = model.chat_messages("SETTING " * 50, "memory note", "room agents", lines, count_tokens=count)
    assert messages[0]["content"].startswith("SETTING") and "memory note" in messages[1]["content"]
    used = messages[1]["content"].count("bob:")
    assert used < 30 and used in runner.CHAT_LINES, "fewer recent lines, never a shorter setting"


def test_no_reaction_bait_unless_training():
    text = RUNNER.read_text(encoding="utf-8")
    assert "target = random.choice(kinds) if invite else None" in text
    assert "visit_room(rest, model, db, username, room, kinds, guide, args.train)" in text
    assert "People may coach you" not in text
