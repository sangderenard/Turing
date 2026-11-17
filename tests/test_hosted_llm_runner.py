"""The hosted LLM receives chat logs, not Nodus control messages or tools."""
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

    def choose_reaction(self, messages, kinds):
        return kinds[0]

    def answer(self, room, messages, target=None, peer=None):
        self.calls.append((room, messages, target, peer))
        return "A short reply."


class Rest:
    def __init__(self):
        self.messages = [{"id": 1, "username": "alice", "body": "hello", "created": 1}]
        self.calls = []

    def call(self, path, body=None):
        self.calls.append((path, body))
        if path.startswith("/api/chat/rooms/agents?"):
            return {"messages": self.messages}
        if path.startswith("/api/hosted-agents/"):
            return {"effective": False}
        if path.endswith("/react"):
            return {"id": body["id"]}
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
        raise AssertionError(path)


def database():
    db = sqlite3.connect(":memory:", isolation_level=None)
    db.executescript(runner.SCHEMA)
    return db


def test_room_bootstrap_then_reacts_and_replies_without_self_loop(monkeypatch):
    monkeypatch.setattr(runner.random, "choice", lambda kinds: kinds[0])
    db, rest, model = database(), Rest(), Model()
    runner.visit_room(rest, model, db, "llama", "agents", ["frog"])
    assert not model.calls
    rest.messages.append({"id": 2, "username": "bob", "body": "What do you think?", "created": 1})
    runner.visit_room(rest, model, db, "llama", "agents", ["frog"])
    assert len(model.calls) == 1
    assert db.execute("SELECT target_kind FROM attempts").fetchone()[0] == "frog"
    assert any(path.endswith("/react") and body == {"id": 2, "kind": "frog"} for path, body in rest.calls)
    runner.visit_room(rest, model, db, "llama", "agents", ["frog"])
    assert len(model.calls) == 1


def test_last_word_mode_answers_current_final_message(monkeypatch):
    class LastWordRest(Rest):
        def call(self, path, body=None):
            if path.startswith("/api/hosted-agents/"):
                return {"effective": True}
            return super().call(path, body)

    monkeypatch.setattr(runner.random, "choice", lambda kinds: kinds[0])
    db, rest, model = database(), LastWordRest(), Model()
    runner.visit_room(rest, model, db, "llama", "agents", ["frog"])
    runner.visit_room(rest, model, db, "llama", "agents", ["frog"])
    assert len(model.calls) == 1
    assert runner.cursor(db, "room:agents") == 1


def test_nodus_dispatch_stays_out_of_model_context():
    db, rest, model = database(), Rest(), Model()
    runner.handle_direct(rest, model, db, "llama")
    assert runner.current(db, "dispatch_room") == "agents"
    assert len(model.calls) == 1
    assert all(m.get("from") != "svc-nodus-messenger" for m in model.calls[0][1])
    assert ("/api/messages/ack", {"up_to": 2}) in rest.calls


def test_reward_requires_requested_emoji_within_ten_minutes(monkeypatch):
    db = database()
    db.execute("INSERT INTO attempts(room,message_id,target_kind,prompt,response,posted_at) "
               "VALUES('agents',100,'frog','hello','reply',100)")
    rest = Rest()
    rest.messages = [{"id": 100, "username": "llama", "body": "reply",
                      "reaction_events": [
                          {"kind": "frog", "username": "llama", "at": 200},
                          {"kind": "frog", "username": "alice", "at": 701},
                          {"kind": "heart", "username": "bob", "at": 300}]}]
    monkeypatch.setattr(runner.time, "time", lambda: 800)
    runner.settle_rewards(rest, db, "llama")
    assert db.execute("SELECT reward FROM attempts").fetchone()[0] == 0
    db.execute("UPDATE attempts SET reward=NULL")
    rest.messages[0]["reaction_events"].append({"kind": "frog", "username": "alice", "at": 699})
    runner.settle_rewards(rest, db, "llama")
    assert db.execute("SELECT reward FROM attempts").fetchone()[0] == 1


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
