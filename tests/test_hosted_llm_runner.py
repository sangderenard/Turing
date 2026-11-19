"""The hosted Llama: chat logs, its orientation and its memory in; no Nodus control, tools, training or emoji rewards."""
import importlib.util
from pathlib import Path
import sqlite3
import threading

RUNNER = Path(__file__).resolve().parents[1] / "examples" / "meowdoku" / "deploy" / "hosted_llm_runner.py"
spec = importlib.util.spec_from_file_location("hosted_llm_runner", RUNNER)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class Model:
    def __init__(self):
        self.calls = []

    def answer(self, room, messages, peer=None, guide=None):
        self.calls.append((room, messages, peer))
        return "A short reply."


class Rest:
    def __init__(self):
        self.messages = [{"id": 1, "username": "alice", "body": "hello", "created": 1}]
        self.calls = []

    def call(self, path, body=None, **kw):
        self.calls.append((path, body))
        if path.startswith("/api/chat/rooms/agents?"):
            return {"messages": self.messages}
        if path.startswith("/api/hosted-agents/"):
            return {"effective": False}
        assert not path.endswith("/react"), "Llama no longer reacts: the emoji bookmark is removed"
        if path == "/api/chat/rooms/agents":
            made = {"id": 100, "username": "llama", "body": body["body"], "created": 1}
            self.messages.append(made)
            return {"message": made}
        if path == "/api/messages/contacts":
            return {"requests": []}
        if path == "/api/messages/inbox":
            return {"messages": [
                {"id": 1, "from": "svc-nodus-messenger", "body": "hosted-dispatch: visit agents",
                 "subject": {"kind": "control", "ref": "agents"}},
                {"id": 2, "from": "alice", "body": "Please answer.", "created": 1}]}
        if path.startswith("/api/messages/thread?"):
            return {"messages": [{"id": 2, "from": "alice", "body": "Please answer."}]}
        if path == "/api/messages/conversations":
            return {"contacts": [{"username": "alice", "conversation_handle": "x" * 40}]}
        if path == "/api/messages/send":
            assert body.get("conversation_handle") == "x" * 40 and "to" not in body
            return {"ok": True}
        if path == "/api/messages/ack":
            return {"cursor": body["up_to"]}
        if path == "/api/memory":
            return {"ok": True}
        raise AssertionError(path)


def database():
    db = sqlite3.connect(":memory:", isolation_level=None)
    db.executescript(runner.SCHEMA)
    return db


def test_operator_goal_enters_chat_setting_and_controls_refresh():
    class ControlRest:
        def __init__(self):
            self.calls = 0

        def call(self, path):
            assert path == "/api/hosted-agents/llama/controls"
            self.calls += 1
            return {"enabled": False, "goal": "Answer questions about the site"}

    runner._CONTROLS.clear()
    rest = ControlRest()
    value = runner.controls(rest, "llama", now=1)
    assert value["enabled"] is False
    assert runner.controls(rest, "llama", now=2) == value and rest.calls == 1
    guide = runner.Guide(rest)
    guide.goal = value["goal"]
    assert "Current operator goal:\nAnswer questions about the site" in guide.setting()


def test_room_bootstrap_then_replies_once_without_self_loop_and_remembers_it():
    runner._MODES.clear()
    db, rest, model = database(), Rest(), Model()
    guide = runner.Guide(rest)
    runner.visit_room(rest, model, db, "llama", "agents", guide)
    assert not model.calls, "the first visit only sets the cursor"
    rest.messages.append({"id": 2, "username": "bob", "body": "What do you think?", "created": 1})
    runner.visit_room(rest, model, db, "llama", "agents", guide)
    assert len(model.calls) == 1
    runner.visit_room(rest, model, db, "llama", "agents", guide)
    assert len(model.calls) == 1, "its own reply is never answered"
    assert guide.pending and "room agents | bob: What do you think? | me: A short reply." in guide.pending[0]


def test_last_word_mode_answers_current_final_message():
    class LastWordRest(Rest):
        def call(self, path, body=None, **kw):
            if path.startswith("/api/hosted-agents/"):
                return {"effective": True}
            return super().call(path, body, **kw)

    runner._MODES.clear()
    db, rest, model = database(), LastWordRest(), Model()
    runner.visit_room(rest, model, db, "llama", "agents")
    runner.visit_room(rest, model, db, "llama", "agents")
    assert len(model.calls) == 1
    assert runner.cursor(db, "room:agents") == 1


def test_nodus_dispatch_stays_out_of_model_context():
    runner._MODES.clear()
    db, rest, model = database(), Rest(), Model()
    runner.handle_direct(rest, model, db, "llama")
    assert runner.current(db, "dispatch_room") == "agents"
    assert len(model.calls) == 1
    assert all(m.get("from") != "svc-nodus-messenger" for m in model.calls[0][1])
    assert ("/api/messages/ack", {"up_to": 2}) in rest.calls


def test_memory_is_created_written_at_most_a_minute_apart_and_kept_under_its_size(monkeypatch):
    rest = Rest()
    guide = runner.Guide(rest)
    guide.remember("room agents", "bob", "hi", "hello", now=0)
    assert guide.flush(now=1000) and guide.memory_id and guide.memory.startswith(runner.MEMORY_HEADER)
    write = [body for path, body in rest.calls if path == "/api/memory"][-1]
    assert write["create"] is True and write["action"] == "write" and "bob: hi | me: hello" in write["body"]
    guide.remember("room agents", "bob", "again", "yes", now=1010)
    assert not guide.flush(now=1010), "at most one write a minute"
    assert guide.flush(now=1061)
    assert [body for path, body in rest.calls if path == "/api/memory"][-1]["create"] is False, "the same file afterwards"
    monkeypatch.setattr(runner, "MEMORY_BYTES", 600)
    for i in range(40):
        guide.remember("room agents", "bob", f"line {i}", "ok", now=2000 + i)
    guide.flush(now=3000)
    assert len(guide.memory.encode("utf-8")) <= 600 and "line 39" in guide.memory and "line 0 " not in guide.memory


def test_no_training_lora_or_emoji_machinery_remains():
    text = RUNNER.read_text(encoding="utf-8")
    for gone in ("--lora", "train_due", "settle_rewards", "choose_reaction", "react_to_others", "target_kind",
                 "--train", "--adapter", "trainer", "People may coach you", "emoji reaction"):
        assert gone not in text, gone
    assert not (RUNNER.parent / "hosted_llm_train.py").exists()
    assert not (RUNNER.parent / "Provision-HostedLlmTrain.ps1").exists()
    launcher = (RUNNER.parent / "Run-HostedLlm.ps1").read_text(encoding="utf-8")
    for gone in ("--trainer", "--adapter", "trainingPython", "convert_lora"):
        assert gone not in launcher, gone


def test_profile_tick_submits_two_individuals_to_one_model_concurrently():
    class BatchRest:
        def __init__(self):
            self.completed = []
            self.claimed = False

        def call(self, path, body=None, *, headers=None):
            assert headers == {"X-Meowdoku-Workspace": "1"}
            if path.endswith("/claim"):
                assert body["limit"] == 2
                if self.claimed:
                    return {"runs": []}
                self.claimed = True
                return {"runs": [{"id": "a", "claim_token": "one", "preamble": "north", "memory": "",
                                  "prompt": "draw north"},
                                 {"id": "b", "claim_token": "two", "preamble": "south", "memory": "",
                                  "prompt": "draw south"}]}
            assert path.endswith("/complete")
            self.completed.append(body)
            return {"status": "done"}

    class BatchModel:
        def __init__(self):
            self.barrier = threading.Barrier(2)

        def answer_profile(self, work):
            self.barrier.wait(timeout=3)
            return work["preamble"] + " result"

    rest = BatchRest()
    assert runner.run_profile_batch(rest, BatchModel(), 2) == 2
    assert {item["response"] for item in rest.completed} == {"north result", "south result"}
    assert {item["claim_token"] for item in rest.completed} == {"one", "two"}


def test_profile_subbatches_isolate_generation_and_completion_errors(capsys):
    class BatchRest:
        def __init__(self):
            self.todo = [{"id": str(i), "claim_token": str(i), "preamble": str(i)} for i in range(5)]
            self.claim_limits = []
            self.completed = []

        def call(self, path, body=None, *, headers=None):
            assert headers == {"X-Meowdoku-Workspace": "1"}
            if path.endswith("/claim"):
                self.claim_limits.append(body["limit"])
                chunk, self.todo = self.todo[:body["limit"]], self.todo[body["limit"]:]
                return {"runs": chunk}
            assert path.endswith("/complete")
            if body["id"] == "2":
                raise runner.RestError("temporary completion failure", 503)
            self.completed.append(body)
            return {"status": "done"}

    class BatchModel:
        def answer_profile(self, work):
            if work["id"] == "1":
                raise RuntimeError("model rejected this run")
            return work["id"] + " result"

    rest = BatchRest()
    assert runner.run_profile_batch(rest, BatchModel(), parallel=2, limit=5) == 5
    assert rest.claim_limits == [2, 2, 1]
    assert {row["id"] for row in rest.completed} == {"0", "1", "3", "4"}
    assert next(row for row in rest.completed if row["id"] == "1")["error"].endswith(
        "model rejected this run")
    assert "completion for 2 failed" in capsys.readouterr().err


def test_profile_context_keeps_contract_and_job_while_excerpting_memory():
    work = {"preamble": "character contract", "prompt": "current task",
            "memory": "old facts " * 30 + "LATEST MEMORY"}
    messages = runner.profile_messages(work, len, input_budget=220)
    assert messages[0]["content"] == "character contract"
    assert messages[1]["content"].endswith("Current job:\ncurrent task")
    assert "LATEST MEMORY" in messages[1]["content"]
    assert "old facts " * 30 not in messages[1]["content"]
    assert sum(map(lambda message: len(message["content"]), messages)) <= 220


def test_profile_context_rejects_contract_that_cannot_fit():
    work = {"preamble": "x" * 200, "prompt": "job", "memory": "memory"}
    try:
        runner.profile_messages(work, len, input_budget=160)
    except RuntimeError as error:
        assert "contract and job" in str(error)
    else:
        raise AssertionError("oversized character contract was silently truncated")


def test_both_model_callers_send_the_repetition_penalty(monkeypatch):
    """Ticket 5166f79d: the runner, not only the server, must emit the sampler fields. A recording stub stands in
    for llama-server's /v1/chat/completions; the chat reply and the profile run both go through LocalModel._complete."""
    import io
    import json

    sent = []

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def urlopen(request, timeout=None):
        assert request.full_url.endswith("/v1/chat/completions")
        sent.append(json.loads(request.data.decode("utf-8")))
        return Response(json.dumps({"choices": [{"message": {"content": " An answer. "}}]}).encode("utf-8"))

    monkeypatch.setattr(runner.urllib.request, "urlopen", urlopen)
    model = runner.LocalModel(Path("llama-server"), Path("base-model.gguf"), 11435, Path("logs"))
    model.count_tokens = len
    assert model.answer("agents", [{"id": 1, "username": "bob", "body": "hi"}]) == "An answer."
    assert model.answer_profile({"preamble": "contract", "prompt": "job", "memory": ""}) == "An answer."
    assert len(sent) == 2
    for body, max_tokens in zip(sent, (160, 256)):
        assert body["repeat_penalty"] == runner.REPEAT_PENALTY == 1.15
        assert body["repeat_last_n"] == runner.REPEAT_LAST_N == 256
        assert body["max_tokens"] == max_tokens and body["model"] == "base-model"


def test_a_one_second_cycle_spaces_out_the_slow_checks():
    assert runner.POLL_SECONDS == 1
    assert runner.PROFILE_SECONDS >= 5 and runner.ROOM_LIST_SECONDS >= 30 and runner.MODE_SECONDS >= 10
    runner._MODES.clear()
    rest = Rest()
    asked = lambda: sum(1 for path, _ in rest.calls if path.startswith("/api/hosted-agents/"))
    for second in range(10):
        runner.mode(rest, "llama", "room", "agents", now=1000.0 + second)
    assert asked() == 1, "a last-word setting is asked once per MODE_SECONDS, not every pass"
    runner.mode(rest, "llama", "room", "agents", now=1000.0 + runner.MODE_SECONDS + 1)
    assert asked() == 2
