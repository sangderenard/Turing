"""roundtable_seat: an agent kept running between turns, given the room's last messages, its answer posted as itself."""
import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "meowdoku" / "deploy"))

import gemini_agent  # noqa: E402
import roundtable_seat as seat  # noqa: E402


class Process:
    """A child process whose output is scripted; what the seat writes to it is kept."""

    def __init__(self, *events):
        self.stdin = io.BytesIO()
        self.stdin.close = lambda: None
        self.stdout = io.BytesIO(b"".join((e if isinstance(e, bytes) else json.dumps(e).encode()) + b"\n" for e in events))
        self.args = None

    def sent(self):
        return [json.loads(line) for line in self.stdin.getvalue().decode("ascii").splitlines()]

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


def popen_of(process):
    def popen(args, **kw):
        process.args = args
        return process
    return popen


def test_claude_keeps_one_session_and_drops_a_late_turn():
    process = Process({"type": "system", "subtype": "init"}, {"type": "assistant", "message": {}},
                      {"type": "result", "subtype": "success", "result": "Hello, table."},
                      b"not json",
                      {"type": "result", "subtype": "success", "result": "late"},
                      {"type": "result", "subtype": "success", "result": "(pass)"})
    claude = seat.Claude("claude.exe", popen=popen_of(process))
    assert "--input-format" in process.args and "mcp__speaktome__chat_post" in process.args, "the seat posts, not claude"
    assert claude.ask("first", 5) == "Hello, table."
    claude.owed = 1                                      # a turn that ran out of time: its result is dropped
    assert claude.ask("second", 5) == "(pass)"
    sent = process.sent()
    assert [m["message"]["content"][0]["text"] for m in sent] == ["first", "second"]
    assert sent[0]["type"] == "user" and sent[0]["message"]["role"] == "user"


def test_claude_reports_a_failed_turn():
    process = Process({"type": "result", "subtype": "error_max_turns", "is_error": True})
    with pytest.raises(seat.SeatError, match="error_max_turns"):
        seat.Claude("claude.exe", popen=popen_of(process)).ask("x", 5)


def test_codex_app_server_one_thread_one_turn_approvals_declined():
    process = Process({"id": 1, "result": {"userAgent": "codex"}},
                      {"method": "thread/started", "params": {"thread": {"id": "th1"}}},
                      {"id": 2, "result": {"thread": {"id": "th1"}}},
                      {"id": 3, "result": {"turn": {"id": "t1", "status": "inProgress", "items": []}}},
                      {"id": 99, "method": "item/commandExecution/requestApproval", "params": {}},
                      {"id": 100, "method": "item/tool/requestUserInput", "params": {}},
                      {"method": "item/completed", "params": {"turnId": "t1", "item": {"type": "agentMessage", "id": "a", "text": "Let me look."}}},
                      {"method": "item/completed", "params": {"turnId": "t1", "item": {"type": "agentMessage", "id": "b", "text": "Final words."}}},
                      {"method": "turn/completed", "params": {"threadId": "th1", "turn": {"id": "t1", "status": "completed", "items": []}}})
    codex = seat.Codex("codex.exe", popen=popen_of(process), seconds=5)
    assert process.args == ["codex.exe", "app-server"] and codex.thread == "th1"
    assert codex.ask("your turn", 5) == "Final words.", "the turn's last agent message is the answer"
    sent = process.sent()
    assert [m.get("method") for m in sent[:4]] == ["initialize", "initialized", "thread/start", "turn/start"]
    assert sent[2]["params"]["sandbox"] == "read-only" and sent[2]["params"]["approvalPolicy"] == "never"
    assert sent[3]["params"] == {"threadId": "th1", "input": [{"type": "text", "text": "your turn"}]}
    assert {"id": 99, "result": {"decision": "decline"}} in sent
    assert any(m.get("id") == 100 and "error" in m for m in sent), "what the seat does not answer is refused"


def test_codex_a_failed_turn_is_an_error():
    process = Process({"id": 1, "result": {}}, {"id": 2, "result": {"thread": {"id": "th"}}}, {"id": 3, "result": {"turn": {"id": "t"}}},
                      {"method": "turn/completed", "params": {"turn": {"id": "t", "status": "failed", "error": {"message": "usage limit"}}}})
    codex = seat.Codex("codex.exe", popen=popen_of(process), seconds=5)
    with pytest.raises(seat.SeatError, match="usage limit"):
        codex.ask("x", 5)


class World:
    def __init__(self, messages):
        self.messages, self.posted, self.calls = list(messages), [], []
        self.tools, self.instructions = {}, ""

    def call(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        if name == "chat_room":
            if "after" in arguments:
                return {"data": {"messages": [m for m in self.messages if m["id"] > int(arguments["after"])]}}
            return {"data": {"messages": self.messages[-int(arguments["tail"]):]}}
        if name == "chat_post":
            made = {"id": 100 + len(self.posted), "username": "claude", "body": arguments["body"]}
            self.posted.append(arguments["body"])
            return {"data": {"message": made}}
        return {"data": {}}


class Agent:
    def __init__(self, *replies):
        self.replies, self.asked = list(replies), []

    def ask(self, text, seconds):
        self.asked.append(text)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


MESSAGES = [{"id": i, "username": who, "body": f"m{i}"} for i, who in ((1, "admin"), (2, "codex"), (3, "gemini"), (4, "admin"))]


def test_a_turn_gives_the_last_messages_and_posts_the_answer_as_the_agent():
    world, agent = World(MESSAGES), Agent("my answer", "(pass)", seat.SeatError("the agent did not answer in time"), "x" * 5000)
    s = seat.Seat(world, agent, name="claude", login="claude", room="agents", order=["admin", "claude", "codex", "gemini"], owner="admin")
    assert s.turn({"op": "turn", "turn": 2, "round": 1}) == {"op": "posted", "turn": 2, "id": 100}
    first = agent.asked[0]
    assert ("chat_room", {"room": "agents", "tail": "3"}) in world.calls, "n-1 messages: four seats, three messages"
    assert "[2] codex: m2" in first and "[4] admin: m4" in first and "m1" not in first.split("follow")[-1]
    assert "round table" in first and "data from other accounts" in first and seat.PASS in first, "the first turn says the rules"
    assert world.posted == ["my answer"]
    assert s.turn({"op": "turn", "turn": 6, "round": 2, "tail": 2}) == {"op": "passed", "turn": 6}
    assert "round table" not in agent.asked[1], "later turns carry only the messages"
    assert s.turn({"op": "turn", "turn": 10})["op"] == "error"
    s.turn({"op": "turn", "turn": 14})
    assert len(world.posted[-1]) <= seat.MAX_POST and world.posted[-1].endswith("length limit]")


def test_waiting_for_the_owner():
    world = World(MESSAGES)
    clock = [0.0]

    def sleep(seconds):
        clock[0] += seconds
        if clock[0] == 10:
            world.messages.append({"id": 5, "username": "codex", "body": "not the owner"})
        if clock[0] == 20:
            world.messages.append({"id": 6, "username": "admin", "body": "my turn"})

    s = seat.Seat(world, Agent(), name="claude", login="claude", room="agents", order=["admin", "claude"], owner="admin",
                  sleep=sleep, clock=lambda: clock[0])
    assert s.wait({"op": "wait", "turn": 1, "for": "admin", "seconds": 60}) == {"op": "posted", "turn": 1, "id": 6}
    assert s.wait({"op": "wait", "turn": 3, "for": "admin", "after": 6, "seconds": 30}) == {"op": "timeout", "turn": 3}


def test_the_seat_serves_lines_until_stop():
    world, agent = World(MESSAGES), Agent("hi")
    s = seat.Seat(world, agent, name="gemini", login="gemini", room="agents", order=["admin", "gemini"], owner="admin")
    out = []
    s.serve(["﻿" + json.dumps({"op": "turn", "turn": 2}), "garbage", json.dumps({"op": "dance"}),
             json.dumps({"op": "stop"}), json.dumps({"op": "turn", "turn": 9})], out.append)
    assert out[0] == {"op": "posted", "turn": 2, "id": 100}
    assert [o["op"] for o in out] == ["posted", "error", "error"], "nothing after stop"


def test_gemini_keeps_its_history_between_turns():
    sent = []
    answers = iter([[{"text": "first answer"}], [], [{"text": "second answer"}]])

    def call(method, path, body=None):
        sent.append(json.loads(json.dumps(body)))
        return {"candidates": [{"content": {"role": "model", "parts": next(answers)}, "finishReason": "STOP"}]}

    gemini = seat.Gemini(World([]), "m", call)
    assert gemini.ask("turn one", 60) == "first answer"
    with pytest.raises(gemini_agent.AgentError):
        gemini.ask("turn two", 60)
    assert len(gemini.history) == 2, "a failed turn leaves the history as it was"
    assert gemini.ask("turn three", 60) == "second answer"
    texts = [c["parts"][0]["text"] for c in sent[-1]["contents"]]
    assert texts == ["turn one", "first answer", "turn three"]
