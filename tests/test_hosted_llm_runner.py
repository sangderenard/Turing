"""The hosted LLM receives chat logs, not Nodus control messages or tools."""
import importlib.util
from pathlib import Path
import sqlite3

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
