"""Nodus as everyone's contact, a counted-down restart that can be interrupted, and its broadcast to every admin.

The first tests are pure SQL. The last (opt-in, NODUS_HEADLESS_TEST_DLL) runs a real supervisor with a real Nodus
canvas host and the real task worker: a restart with a countdown is announced to the admins through the Nodus task
table, interrupted, and never performed; a second one runs out and is performed.
"""
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import accounts as accounts_mod  # noqa: E402
from meowdoku import authz as authz_mod  # noqa: E402
from meowdoku import instance_control as control  # noqa: E402
from meowdoku import messaging, scheduled_tasks  # noqa: E402

NODUS = ROOT.parent / "nodus"
SUP = ROOT / "examples" / "meowdoku" / "deploy" / "apache_supervisor.py"


def make_accounts(path):
    """ann and bo are admins, cy is a member, the system and a worker are service accounts."""
    accounts = accounts_mod.Accounts(path)
    ids = {}
    db = sqlite3.connect(path)
    for name in ("ann", "bo", "cy", "svc-nodus", "svc-worker"):
        db.execute("INSERT INTO users (username, password_hash, display_name, created_at) VALUES (?,?,?,?)",
                   (name, "x", name, time.time()))
        ids[name] = db.execute("SELECT id FROM users WHERE username = ?", (name,)).fetchone()[0]
    db.commit()
    db.close()
    accounts._conn.close()
    conn = authz_mod._connect(path)
    for name, role in (("ann", "admin"), ("bo", "admin"), ("svc-nodus", "service"), ("svc-worker", "service")):
        conn.execute("INSERT OR REPLACE INTO account_roles (user_id, role, granted_by, created_at) VALUES (?,?,?,?)",
                     (ids[name], role, "test", time.time()))
    conn.commit()
    conn.close()
    return ids


def messages_to(path, user_id):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in db.execute("SELECT * FROM msg_messages WHERE recipient_id = ? ORDER BY id", (user_id,))]
    finally:
        db.close()


def test_the_system_becomes_a_contact_of_every_person_and_a_block_is_respected(tmp_path):
    path = tmp_path / "accounts.db"
    ids = make_accounts(path)
    db = messaging.connect(path)
    messaging.block(db, ids["cy"], "svc-nodus")
    db.close()
    sentence = scheduled_tasks.befriend_everyone(path, "svc-nodus", messaging=messaging)
    assert sentence == "Nodus is a contact of 2 of 3 accounts", "service accounts are not people; cy blocked the system"
    db = messaging.connect(path)
    assert messaging._mutual(db, ids["svc-nodus"], ids["ann"]) and messaging._mutual(db, ids["svc-nodus"], ids["bo"])
    assert not messaging._mutual(db, ids["svc-nodus"], ids["cy"]) and messaging._state(db, ids["cy"], ids["svc-nodus"]) == "blocked"
    assert messaging._state(db, ids["svc-nodus"], ids["svc-worker"]) is None
    shown = [c["display_name"] for c in messaging.contacts(db, ids["ann"])["contacts"]]
    db.close()
    assert shown == ["Nodus"], "people see the system as Nodus"
    assert scheduled_tasks.befriend_everyone(path, "svc-nodus", messaging=messaging) == sentence, "doing it again changes nothing"


def test_a_broadcast_reaches_every_admin_and_nobody_else(tmp_path):
    path = tmp_path / "accounts.db"
    ids = make_accounts(path)
    request = {"text": "There will be a restart of web in 30 seconds (request 4).", "audience": "admin",
               "msg_type": "approval_request", "subject": {"kind": "control", "ref": "4"}}
    assert scheduled_tasks.broadcast(path, "svc-nodus", request, messaging=messaging) \
        == "told 2 admin account(s); audience found directly"
    for name in ("ann", "bo"):
        (got,) = messages_to(path, ids[name])
        assert (got["sender_id"], got["msg_type"], got["subject_kind"], got["subject_ref"]) == (ids["svc-nodus"], "approval_request", "control", "4")
        assert got["body"] == request["text"]
    assert messages_to(path, ids["cy"]) == [] and messages_to(path, ids["svc-worker"]) == []
    db = messaging.connect(path)
    waiting = messaging.inbox(db, ids["ann"])["messages"]
    db.close()
    assert waiting[0]["from"] == "svc-nodus" and waiting[0]["subject"] == {"kind": "control", "ref": "4"}
    with pytest.raises(ValueError):
        scheduled_tasks.broadcast(path, "svc-nodus", {"text": "x", "audience": "the world"}, messaging=messaging)
    with pytest.raises(ValueError):
        scheduled_tasks.broadcast(path, "svc-nodus", {"text": " "}, messaging=messaging)
    assert scheduled_tasks.broadcast(path, "svc-nodus", {"text": "all hands", "audience": "everyone"}, messaging=messaging) \
        == "told 3 everyone account(s); audience found directly"
    assert scheduled_tasks.broadcast(path, "svc-nodus", {"text": "just you", "terms": ["role:member", "cy"]}, messaging=messaging) \
        == "told 1 matching account(s); audience found directly"


def test_the_user_list_takes_search_terms_and_gives_every_identity_that_matched(tmp_path):
    from meowdoku import identity_tools
    path = tmp_path / "accounts.db"
    ids = make_accounts(path)
    db = messaging.connect(path)
    try:
        everyone = identity_tools.search(db)
        assert [i["username"] for i in everyone] == ["ann", "bo", "cy", "svc-nodus", "svc-worker"]
        assert everyone[0] == {"identity": f"account:{ids['ann']}", "username": "ann", "display_name": "ann", "role": "admin",
                               "kind": "person"}
        assert [i["username"] for i in identity_tools.search(db, ["role:admin"])] == ["ann", "bo"]
        assert [i["username"] for i in identity_tools.search(db, ["kind:person"])] == ["ann", "bo", "cy"]
        assert [i["username"] for i in identity_tools.search(db, ["kind:service", "WORK"])] == ["svc-worker"]
        assert [i["role"] for i in identity_tools.search(db, ["cy"])] == ["member"], "an account with no role row is a member"
        assert identity_tools.search(db, ["role:admin", "cy"]) == [], "every term must match"
        assert identity_tools.represent(db, "ANN")["identity"] == f"account:{ids['ann']}"
        assert identity_tools.represent(db, f"account:{ids['bo']}")["username"] == "bo"
        assert identity_tools.represent(db, "nobody") is None
        for bad in (["kind:robot"], ["role:"], [""], "admin", ["x"] * 9, [7]):
            with pytest.raises(identity_tools.IdentityError):
                identity_tools.search(db, bad)
    finally:
        db.close()


def test_the_tables_tasks_are_split_by_the_table_that_performs_them():
    class Statuses:
        DONE, UNCHANGED = 1, 2
    offered = scheduled_tasks.handlers(Statuses, system={"accounts_db": "x", "username": "svc-nodus", "module": messaging})
    assert sorted(offered) == ["identity.represent", "identity.search", "system.broadcast"]
    assert scheduled_tasks.PREFIX_TASKS["userlist"] == ("identity.search", "identity.represent")
    assert "system.broadcast" in scheduled_tasks.PREFIX_TASKS["cron"]


def test_a_countdown_is_due_only_when_it_runs_out_and_can_be_interrupted_once(tmp_path):
    db = sqlite3.connect(tmp_path / "a.db", timeout=10, isolation_level=None)
    made = control.request(db, instance="web", action="restart", user_id=3, username="ann", delay_s=30, known=["web"])
    assert made["due_at"] - made["requested_at"] == pytest.approx(30) and made["announced"] == 0
    assert control.due(db) == [] and [r["id"] for r in control.to_announce(db)] == [made["id"]]
    control.mark_announced(db, made["id"], 1)
    assert control.to_announce(db) == []
    assert [r["id"] for r in control.due(db, now=made["due_at"] + 1)] == [made["id"]]
    stopped = control.cancel(db, made["id"], user_id=4, username="bo")
    assert stopped["state"] == "cancelled" and stopped["detail"] == "interrupted by bo"
    assert control.due(db, now=made["due_at"] + 1) == [], "an interrupted request is never performed"
    assert [r["id"] for r in control.to_announce(db)] == [made["id"]], "the interruption is announced too, once"
    control.mark_announced(db, made["id"], 2)
    assert control.to_announce(db) == []
    with pytest.raises(control.ControlError) as again:
        control.cancel(db, made["id"], user_id=4, username="bo")
    assert again.value.status == 409
    with pytest.raises(control.ControlError) as unknown:
        control.cancel(db, 999, user_id=4, username="bo")
    assert unknown.value.status == 404
    now = control.request(db, instance="web", action="restart", user_id=3, username="ann", known=["web"])
    assert control.to_announce(db) == [] and [r["id"] for r in control.due(db)] == [now["id"]], "no countdown: nothing to announce"
    for bad in (-1, 4000, "soon", True):
        with pytest.raises(control.ControlError):
            control.request(db, instance="web", action="shutdown", user_id=3, username="ann", delay_s=bad, known=["web"])


def test_a_table_made_before_the_countdown_is_brought_up_to_date_with_its_rows(tmp_path):
    db = sqlite3.connect(tmp_path / "a.db", timeout=10, isolation_level=None)
    db.execute("CREATE TABLE instance_requests (id INTEGER PRIMARY KEY AUTOINCREMENT, instance TEXT NOT NULL, "
               "action TEXT NOT NULL CHECK(action IN ('restart','shutdown','start')), requested_by INTEGER NOT NULL, "
               "requested_name TEXT, reason TEXT, requested_at REAL NOT NULL, "
               "state TEXT NOT NULL DEFAULT 'pending' CHECK(state IN ('pending','done','refused')), handled_at REAL, detail TEXT)")
    db.execute("INSERT INTO instance_requests (instance, action, requested_by, requested_name, requested_at, state, detail) "
               "VALUES ('task-worker', 'restart', 2, 'claude', 100.0, 'done', 'started pid 1')")
    old = control.recent(db)
    assert [(r["id"], r["state"], r["detail"], r["due_at"], r["announced"]) for r in old] == [(1, "done", "started pid 1", 100.0, 0)]
    made = control.request(db, instance="web", action="restart", user_id=3, username="ann", delay_s=5, known=["web"])
    assert control.cancel(db, made["id"], user_id=3, username="ann")["state"] == "cancelled"


def _rows(db, sql):
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql)]
    finally:
        conn.close()


def _wait(pred, timeout=60.0):
    end = time.time() + timeout
    while time.time() < end:
        try:
            got = pred()
        except sqlite3.Error:
            got = None
        if got:
            return got
        time.sleep(0.25)
    raise AssertionError("timed out waiting")


def test_a_real_countdown_is_broadcast_through_nodus_interrupted_and_then_one_runs_out(tmp_path):
    if not os.environ.get("NODUS_HEADLESS_TEST_DLL"):
        pytest.skip("Set NODUS_HEADLESS_TEST_DLL (and PATH for its dependencies) to run the real supervisor, canvas host and worker")
    db, state, inst = tmp_path / "accounts.db", tmp_path / "state", tmp_path / "instances.json"
    jobs, web, canvas, blank = tmp_path / "jobs", tmp_path / "web", tmp_path / "both.canvas", tmp_path / "blank.canvas"
    users, one = tmp_path / "userlist", tmp_path / "cron.canvas"
    jobs.mkdir()
    web.mkdir()
    (jobs / "cron.table").write_text("# nothing scheduled in this test\nnever 0 0 1 1 * missed=skip\n", encoding="utf-8")
    blank.write_text("CANVAS V4\n1 1 56\nOFFSET 0 0\n\n", encoding="utf-8")
    port, users_port = 47699, 47701
    subprocess.run([sys.executable, str(NODUS / "tools" / "add_contract_port.py"), "--canvas", str(blank), "--out", str(one),
                    "--prefix", "cron", "--port", str(port), "--loan", "3600"], check=True, capture_output=True, timeout=120)
    subprocess.run([sys.executable, str(NODUS / "tools" / "add_contract_port.py"), "--canvas", str(one), "--out", str(canvas),
                    "--prefix", "userlist", "--port", str(users_port), "--loan", "3600", "--keep", "cron:4:3600"],
                   check=True, capture_output=True, timeout=120)
    ids = make_accounts(db)
    worker = [sys.executable, str(ROOT / "examples" / "meowdoku" / "deploy" / "task_worker.py"), "--state-dir", str(state),
              "--web-dir", str(web), "--cron-jobs-dir", str(jobs), "--accounts-db", str(db), "--system-account", "svc-nodus",
              "--userlist-jobs-dir", str(users)]
    config = {"system_events": {"from": "nodus", "prefix": "cron", "jobs_dir": str(jobs), "engine_id": 7202}, "instances": [
        {"name": "nodus", "kind": "nodus", "port": None, "route": "", "bind": "127.0.0.1", "account": "svc-nodus", "leases": [],
         "args": ["--cron", f"cron:7201:{jobs}"],
         "nodus": {"tool": str(NODUS / "tools" / "run_canvas.py"), "canvas": str(canvas),
                   "acquire": ["cron:4:3600", "userlist:4:3600"], "rate": 60,
                   "dll": os.environ["NODUS_HEADLESS_TEST_DLL"]}},
        {"name": "task-worker", "kind": "command", "port": None, "route": "", "bind": "127.0.0.1", "account": "svc-worker",
         "federation_from": "nodus", "cron_prefix": "cron", "userlist_prefix": "userlist", "leases": [], "args": worker},
        {"name": "web", "kind": "command", "port": None, "route": "", "bind": "127.0.0.1", "account": "svc-web", "leases": [],
         "args": [sys.executable, "-c", "import time\nprint('up', flush=True)\ntime.sleep(600)"]}]}
    inst.write_text(json.dumps(config), encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(SUP), "--db", str(db), "--instances", str(inst), "--state-dir", str(state)],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def instance(name):
        return next((r for r in _rows(db, "SELECT * FROM service_instances") if r["name"] == name), None)

    def request_row(request_id):
        return next(r for r in _rows(db, "SELECT * FROM instance_requests") if r["id"] == request_id)

    def ask(**fields):
        conn = sqlite3.connect(db, timeout=10, isolation_level=None)
        try:
            return control.request(conn, user_id=ids["ann"], username="ann", known=["web", "nodus", "task-worker"], **fields)["id"]
        finally:
            conn.close()
    try:
        for name in ("nodus", "task-worker", "web"):
            _wait(lambda: (instance(name) or {}).get("state") == "running", timeout=90)
        first = instance("web")["pid"]
        # a countdown is announced to both admins, with the control request as its subject
        request_id = ask(instance="web", action="restart", delay_s=20, reason="maintenance")
        told = _wait(lambda: messages_to(db, ids["bo"]), timeout=30)
        assert len(messages_to(db, ids["ann"])) == 1 and messages_to(db, ids["cy"]) == []
        assert told[0]["msg_type"] == "approval_request" and (told[0]["subject_kind"], told[0]["subject_ref"]) == ("control", str(request_id))
        assert "restart of web in" in told[0]["body"] and "asked by ann: maintenance" in told[0]["body"]
        assert told[0]["sender_id"] == ids["svc-nodus"]
        assert request_row(request_id)["announced"] == 1 and instance("web")["pid"] == first
        # the audience came from one tick of the userlist table: its request carries the terms, its output the identities
        searches = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(users.glob("*.result.json"))]
        asked = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(users.glob("*.request.json"))]
        assert asked[0]["task"] == "identity.search" and asked[0]["terms"] == ["role:admin"]
        assert [i["username"] for i in searches[0]["output"]] == ["ann", "bo"] and searches[0]["detail"] == "2 identities matched"
        told_by = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(jobs.glob("*.result.json"))]
        assert "audience found userlist table, contract" in told_by[0]["detail"]
        # the interrupt: nothing happens, and the admins are told
        conn = sqlite3.connect(db, timeout=10, isolation_level=None)
        control.cancel(conn, request_id, user_id=ids["bo"], username="bo")
        conn.close()
        after = _wait(lambda: messages_to(db, ids["ann"]) if len(messages_to(db, ids["ann"])) == 2 else None, timeout=30)
        assert "was interrupted: interrupted by bo" in after[1]["body"] and after[1]["msg_type"] == "status"
        time.sleep(3)
        assert request_row(request_id)["state"] == "cancelled" and instance("web")["pid"] == first, "never performed"
        # a short countdown that runs out is performed
        second = ask(instance="web", action="restart", delay_s=4)
        _wait(lambda: request_row(second)["state"] == "done", timeout=60)
        assert "started pid" in request_row(second)["detail"]
        _wait(lambda: (instance("web") or {}).get("state") == "running" and instance("web")["pid"] != first)
        assert len(messages_to(db, ids["ann"])) == 3
        proc.stdin.close()
        assert proc.wait(timeout=90) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.stdout.close()
        proc.stderr.close()
