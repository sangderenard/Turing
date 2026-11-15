"""Long-running, account-isolated hosted LLM intern.

Only SpeakToMe chat, messages, reactions and mode settings enter the model context.
No model output is interpreted as a tool call. The runner owns posting and reacting.
Set SPEAKTOME_BASE, SPEAKTOME_USER, SPEAKTOME_SECRET_FILE and PYTHONPATH for the
separate guest account before starting.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sqlite3
import subprocess
import sys
import time
import urllib.parse
import urllib.request

from speaktome_mcp.rest import Rest, RestError

POLL_SECONDS = 8
QUIET_SECONDS = 12
ROOM_SECONDS = 180
REWARD_SECONDS = 600
TRAIN_SECONDS = 1800
MAX_ROOM_REPLIES = 3
SYSTEM_SENDERS = {"svc-nodus-messenger", "nodus"}
SCHEMA = """
CREATE TABLE IF NOT EXISTS cursors (place TEXT PRIMARY KEY, message_id INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS attempts (
    id INTEGER PRIMARY KEY, room TEXT NOT NULL, message_id INTEGER NOT NULL UNIQUE,
    target_kind TEXT NOT NULL, prompt TEXT NOT NULL, response TEXT NOT NULL,
    posted_at REAL NOT NULL, reward INTEGER, trained_at REAL
);
CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS dm_replies (message_id INTEGER PRIMARY KEY, peer TEXT NOT NULL, body TEXT NOT NULL);
"""


def key(value):
    return urllib.parse.quote(str(value), safe="")


def cursor(db, place):
    row = db.execute("SELECT message_id FROM cursors WHERE place=?", (place,)).fetchone()
    return None if row is None else row[0]


def advance(db, place, message_id):
    db.execute("INSERT INTO cursors(place,message_id) VALUES(?,?) "
               "ON CONFLICT(place) DO UPDATE SET message_id=max(message_id,excluded.message_id)",
               (place, int(message_id)))


def current(db, name, default=""):
    row = db.execute("SELECT value FROM state WHERE key=?", (name,)).fetchone()
    return row[0] if row else default


def save(db, name, value):
    db.execute("INSERT INTO state(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
               (name, str(value)))


def eligible(message, username):
    return (isinstance(message, dict) and isinstance(message.get("id"), int)
            and message.get("username") != username and message.get("from") != username
            and message.get("username") not in SYSTEM_SENDERS and message.get("from") not in SYSTEM_SENDERS
            and not message.get("removed"))


def context(messages, username, limit=24):
    return "\n".join(f"{m.get('username', m.get('from', '?'))}: {str(m.get('body') or ('[image attachment]' if m.get('image') else '[empty message]'))[:1000]}"
                     for m in messages[-limit:] if isinstance(m, dict)
                     and m.get("username", m.get("from")) not in SYSTEM_SENDERS
                     and not m.get("removed"))


class LocalModel:
    def __init__(self, executable, model, adapter, port, log_dir):
        self.executable, self.model, self.adapter, self.port = executable, model, adapter, port
        self.log_dir = log_dir
        self.process = None
        self.log = None

    def start(self):
        if self.process and self.process.poll() is None:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log = open(self.log_dir / "llama-server.log", "ab", buffering=0)
        args = [str(self.executable), "-m", str(self.model), "--host", "127.0.0.1",
                "--port", str(self.port), "-ngl", "99", "-c", "4096"]
        if self.adapter.is_file():
            args += ["--lora", str(self.adapter)]
        self.process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=self.log, stderr=subprocess.STDOUT,
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        for _ in range(90):
            if self.process.poll() is not None:
                raise RuntimeError("llama-server exited; inspect its log")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as response:
                    if response.status == 200:
                        return
            except Exception:
                pass
            time.sleep(1)
        raise RuntimeError("llama-server did not become ready")

    def stop(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
            self.process = None
        if self.log:
            self.log.close()
            self.log = None

    def choose_reaction(self, messages, kinds):
        payload = {"model": self.model.stem, "messages": [
            {"role": "system", "content": "Choose one fitting emoji kind for the latest other person's chat message. "
             "Answer with only one name from: " + ", ".join(kinds)},
            {"role": "user", "content": context(messages, "")}],
            "temperature": 0.4, "max_tokens": 8}
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions",
                                         data=json.dumps(payload).encode("utf-8"),
                                         headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.load(response)
        choice = result["choices"][0]["message"]["content"].strip().lower().strip(" .!\n")
        return choice if choice in kinds else random.choice(kinds)

    def answer(self, room, messages, target=None, peer=None):
        setting = ("You are Llama, an intern account in SpeakToMe. Participate plainly and courteously. "
                   "People may coach you; take specific corrections seriously. You can only speak or react through "
                   "your runner. Never claim to have used a tool or seen information beyond these chat logs. "
                   "Reply to the latest other person's message. Keep the reply under 500 characters.")
        if target:
            setting += f" In this turn, naturally invite a {target} emoji reaction. Do not ask for reactions directly."
        place = f"direct conversation with {peer}" if peer else f"room {room}"
        payload = {"model": self.model.stem, "messages": [
            {"role": "system", "content": setting},
            {"role": "user", "content": f"Recent chat in {place}:\n{context(messages, '')}\nYour reply:"}],
            "temperature": 0.8, "max_tokens": 160}
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions",
                                         data=json.dumps(payload).encode("utf-8"),
                                         headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.load(response)
        answer = result["choices"][0]["message"]["content"].strip()
        return answer[:500]


def mode(rest, username, scope, place):
    query = urllib.parse.urlencode({"scope": scope, "key": place})
    response = rest.call(f"/api/hosted-agents/{key(username)}/last-word?{query}")
    return bool(response["effective"])


def room_list(rest, username):
    listed = rest.call("/api/chat/rooms")["rooms"]
    return [row["room"] for row in listed if username in row.get("hosted_agents", [])]


def room_history(rest, room, tail=100):
    return rest.call(f"/api/chat/rooms/{key(room)}?tail={tail}")["messages"]


def atlas_kinds(rest):
    kinds = [kind for kind in rest.call("/api/chat/rooms").get("emoji_kinds", []) if isinstance(kind, str)]
    if not kinds:
        raise RuntimeError("the site published no emoji kinds")
    return kinds


def react_to_others(rest, model, room, messages, username, kinds, seen):
    for message in messages[-8:]:
        if not eligible(message, username) or message["id"] <= seen:
            continue
        kind = model.choose_reaction([m for m in messages if m["id"] <= message["id"]], kinds)
        rest.call(f"/api/chat/rooms/{key(room)}/react", {"id": message["id"], "kind": kind})
        seen = message["id"]
    return seen


def visit_room(rest, model, db, username, room, kinds):
    history = room_history(rest, room)
    place = "room:" + room
    seen = cursor(db, place)
    if seen is None:
        newest = max((m["id"] for m in history), default=0)
        # An enabled last-word room needs an answer to its current final message.
        latest = max(history, key=lambda m: m["id"], default=None)
        pending_last_word = latest is not None and eligible(latest, username) and mode(rest, username, "room", room)
        advance(db, place, newest - 1 if pending_last_word else newest)
        save(db, "react:" + room, newest)
        return
    react_seen = int(current(db, "react:" + room, "0"))
    react_seen = react_to_others(rest, model, room, history, username, kinds, react_seen)
    save(db, "react:" + room, react_seen)
    incoming = [m for m in history if eligible(m, username) and m["id"] > seen]
    if not incoming:
        return
    if mode(rest, username, "room", room):
        incoming = [incoming[-1]]
        if time.time() - incoming[-1].get("created", 0) < QUIET_SECONDS:
            return
    else:
        incoming = incoming[:MAX_ROOM_REPLIES]
    for message in incoming:
        target = random.choice(kinds)
        prior = [m for m in history if m["id"] <= message["id"]]
        answer = model.answer(room, prior, target=target)
        if answer:
            posted = rest.call(f"/api/chat/rooms/{key(room)}", {"body": answer})["message"]
            db.execute("INSERT OR IGNORE INTO attempts(room,message_id,target_kind,prompt,response,posted_at) "
                       "VALUES(?,?,?,?,?,?)", (room, posted["id"], target, context(prior, username), answer, time.time()))
        advance(db, place, message["id"])


def handle_direct(rest, model, db, username):
    contacts = rest.call("/api/messages/contacts")
    for request in contacts.get("requests", []):
        peer = request.get("username")
        if peer and peer not in SYSTEM_SENDERS:
            rest.call("/api/messages/contacts", {"action": "accept", "username": peer})
    inbox = rest.call("/api/messages/inbox")
    highest = 0
    for message in inbox.get("messages", []):
        message_id = message["id"]
        peer = message.get("from")
        if peer in SYSTEM_SENDERS:
            # Nodus dispatches a room as control data, never as a model prompt.
            if message.get("subject", {}).get("kind") == "control" and str(message.get("body", "")).startswith("hosted-dispatch:"):
                save(db, "dispatch_room", str(message["subject"].get("ref", "")))
        elif peer and peer != username:
            thread = rest.call("/api/messages/thread?peer=" + key(peer))
            lines = sorted(thread.get("messages", []), key=lambda row: row.get("id", 0))
            if not lines:
                lines = [message]
            if mode(rest, username, "conversation", peer) and time.time() - message.get("created", 0) < QUIET_SECONDS:
                break
            prior = db.execute("SELECT body FROM dm_replies WHERE message_id=?", (message_id,)).fetchone()
            answer = prior[0] if prior else model.answer(None, lines, peer=peer)
            if answer:
                if prior is None:
                    db.execute("INSERT INTO dm_replies(message_id,peer,body) VALUES(?,?,?)", (message_id, peer, answer))
                contacts_now = rest.call("/api/messages/conversations")["contacts"]
                match = next((row for row in contacts_now if row["username"] == peer), None)
                if match is None:
                    raise RuntimeError("conversation ended before reply")
                rest.call("/api/messages/send", {"conversation_handle": match["conversation_handle"], "body": answer,
                                                 "idempotency_key": f"llama-reply-{message_id:020d}"})
        highest = max(highest, message_id)
    if highest:
        rest.call("/api/messages/ack", {"up_to": highest})


def settle_rewards(rest, db, username):
    rows = db.execute("SELECT id,room,message_id,target_kind,posted_at FROM attempts WHERE reward IS NULL AND posted_at<=?",
                      (time.time() - REWARD_SECONDS,)).fetchall()
    for attempt_id, room, message_id, target, posted_at in rows:
        history = room_history(rest, room, 1000)
        matched = next((m for m in history if m["id"] == message_id), None)
        if matched is None:
            continue
        reward = int(any(event["kind"] == target and event["username"] != username
                         and posted_at <= event["at"] <= posted_at + REWARD_SECONDS
                         for event in matched.get("reaction_events", [])))
        db.execute("UPDATE attempts SET reward=? WHERE id=?", (reward, attempt_id))


def train_due(args, db, model):
    if time.time() - float(current(db, "train_at", "0")) < TRAIN_SECONDS:
        return
    found = db.execute("SELECT 1 FROM attempts WHERE reward=1 AND trained_at IS NULL LIMIT 1").fetchone()
    if not found:
        return
    save(db, "train_at", time.time())
    model.stop()
    try:
        subprocess.run([args.trainer_python, args.trainer, "--db", str(Path(args.state_dir) / "hosted-llm.sqlite"),
                        "--base", args.base, "--adapter-root", args.adapter_root,
                        "--converter", args.converter, "--gguf-out", args.adapter],
                       check=True, timeout=1200)
    finally:
        model.start()


def run(args):
    username = os.environ.get("SPEAKTOME_USER")
    if not username:
        raise RuntimeError("SPEAKTOME_USER is required")
    rest = Rest.from_env()
    db_path = Path(args.state_dir) / "hosted-llm.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(db_path, isolation_level=None)
    db.executescript(SCHEMA)
    model = LocalModel(Path(args.llama_server), Path(args.model), Path(args.adapter),
                       args.port, Path(args.state_dir) / "logs")
    try:
        model.start()
        kinds = atlas_kinds(rest)
        room = ""
        until = 0
        while True:
            try:
                handle_direct(rest, model, db, username)
                rooms = room_list(rest, username)
                dispatch = current(db, "dispatch_room")
                if dispatch in rooms:
                    room = dispatch
                    save(db, "dispatch_room", "")
                    until = time.time() + ROOM_SECONDS
                if room not in rooms or time.time() >= until:
                    room = rooms[(rooms.index(room) + 1) % len(rooms)] if room in rooms else (rooms[0] if rooms else "")
                    until = time.time() + ROOM_SECONDS
                if room:
                    visit_room(rest, model, db, username, room, kinds)
                for watched in rooms:
                    if watched != room and mode(rest, username, "room", watched):
                        visit_room(rest, model, db, username, watched, kinds)
                settle_rewards(rest, db, username)
                train_due(args, db, model)
            except (RestError, OSError, ValueError, KeyError, RuntimeError) as error:
                print(f"[hosted-llm] {type(error).__name__}: {error}", file=sys.stderr, flush=True)
            time.sleep(POLL_SECONDS)
    finally:
        model.stop()
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--llama-server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--port", type=int, default=11435)
    parser.add_argument("--trainer-python", required=True)
    parser.add_argument("--trainer", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--converter", required=True)
    parser.add_argument("--adapter-root", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
