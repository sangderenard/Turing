"""SpeakToMe mail (examples/meowdoku/mail.py) with real SQLite databases and a fake provider.

The SMTP handler is driven the way aiosmtpd drives it (handle_MAIL / handle_RCPT / handle_DATA with a session and an
envelope). The last test runs the real aiosmtpd server on loopback when aiosmtpd is installed, and is skipped otherwise.
"""
import asyncio
import email
import email.policy
import json
import os
import smtplib
import subprocess
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import mail, scheduled_tasks, studio_product_nodus  # noqa: E402

NODUS = ROOT.parent / "nodus"

SECRET = "pw-never-shown-0123"


@pytest.fixture
def world(tmp_path):
    accounts_db = tmp_path / "accounts.db"
    db = sqlite3.connect(accounts_db)
    db.executescript("""
        CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL,
                            display_name TEXT, created_at REAL NOT NULL);
        CREATE TABLE account_roles (user_id INTEGER PRIMARY KEY, role TEXT NOT NULL, granted_by TEXT NOT NULL,
                                    created_at REAL NOT NULL);
        INSERT INTO users VALUES (1, 'svc-meowdoku-a', 'x', NULL, 0), (2, 'Ann', 'x', 'Ann A', 0), (3, 'bob', 'x', 'Bob', 0);
        INSERT INTO account_roles VALUES (1, 'service', 'operator', 0);
    """)
    db.commit()
    db.close()
    credential = tmp_path / "provider.json"
    credential.write_text(json.dumps({"username": "relay-user", "password": SECRET}), encoding="utf-8")
    config = {**mail.DEFAULTS, "provider_host": "smtp.example.net", "from_address": "noreply@example.net",
              "credential_file": str(credential), "per_ip_per_minute": 3, "max_recipients": 3}
    return {"accounts_db": accounts_db, "mail_db": tmp_path / "mail.db", "config": config}


def dbs(world):
    return mail.connect(world["mail_db"]), mail.connect_accounts(world["accounts_db"])


def session(ip="198.51.100.7", helo="peer.example", tls=True):
    return SimpleNamespace(peer=(ip, 40000), host_name=helo, ssl={"cipher": "x"} if tls else None)


def envelope():
    return SimpleNamespace(mail_from=None, mail_options=[], rcpt_tos=[], rcpt_options=[], content=None,
                           original_content=None)


def smtp_conversation(handler, sess, env, rcpts, data):
    async def run():
        replies = [await handler.handle_MAIL(None, sess, env, "someone@peer.example", [])]
        if not replies[0].startswith("250"):
            return replies                             # as the server does: no RCPT without an accepted MAIL
        for rcpt in rcpts:
            replies.append(await handler.handle_RCPT(None, sess, env, rcpt, []))
        env.original_content = data
        replies.append(await handler.handle_DATA(None, sess, env))
        return replies
    return asyncio.run(run())


MESSAGE = (b"From: Someone <someone@peer.example>\r\nTo: ann@stichka.duckdns.org\r\nSubject: Hello there\r\n"
           b"Date: Fri, 13 Nov 2026 12:00:00 +0000\r\n\r\nA line of text.\r\n")


def test_inbound_accepts_only_person_mailboxes_of_the_domain_and_files_them(world):
    asked = []
    handler = mail.Inbound(world["mail_db"], world["accounts_db"], world["config"], on_received=asked.append)
    replies = smtp_conversation(handler, session(), envelope(),
                                ["ANN@Stichka.DuckDNS.org", "someone@gmail.com", "svc-meowdoku-a@stichka.duckdns.org",
                                 "nobody@stichka.duckdns.org", "not an address"], MESSAGE)
    assert replies[0] == "250 OK"
    assert replies[1] == "250 OK"
    assert replies[2].startswith("550 5.7.1")          # never a relay
    assert replies[3].startswith("550 5.1.1")          # a service account has no mailbox
    assert replies[4].startswith("550 5.1.1")
    assert replies[5].startswith("501")
    assert replies[6].startswith("250 2.0.0 OK queued as ")
    message_id = int(replies[6].rsplit(" ", 1)[1])
    assert asked == [message_id]
    db, accounts = dbs(world)
    assert db.execute("SELECT state FROM mail_recipients").fetchall()[0][0] == "queued"
    assert mail.inbox(db, 2) == []                      # stored, not yet filed: filing is the mail table's task
    assert mail.file_message(db, accounts, message_id, config=world["config"]) == 1
    listed = mail.inbox(db, 2)
    assert [m["subject"] for m in listed] == ["Hello there"] and listed[0]["read"] is False
    one = mail.read(db, 2, listed[0]["id"])
    assert one["body"].strip() == "A line of text." and one["from"] == "Someone <someone@peer.example>"
    assert mail.inbox(db, 2)[0]["read"] is True
    raw = db.execute("SELECT raw FROM mail_messages WHERE id = ?", (message_id,)).fetchone()[0]
    assert raw.startswith(b"Received: from peer.example ([198.51.100.7])") and b"ESMTPS" in raw
    with pytest.raises(mail.MailError):
        mail.read(db, 3, listed[0]["id"])              # bob cannot read ann's mail
    assert mail.remove(db, 2, listed[0]["id"])["removed"] and mail.inbox(db, 2) == []


def test_inbound_limits_rate_recipients_size_and_full_mailboxes(world):
    config = {**world["config"], "max_message_bytes": 200}
    clock = [1000.0]
    handler = mail.Inbound(world["mail_db"], world["accounts_db"], config, clock=lambda: clock[0])
    many = ["ann@stichka.duckdns.org", "bob@stichka.duckdns.org", "ann@stichka.duckdns.org"]
    env = envelope()
    replies = smtp_conversation(handler, session(), env, many, b"x" * 300)
    assert replies[1:4] == ["250 OK", "250 OK", "250 OK"] and env.rcpt_tos == many[:2]     # the same address once
    assert replies[4].startswith("552")                                                    # too big
    for _ in range(2):
        assert smtp_conversation(handler, session(), envelope(), many[:1], MESSAGE)[-1].startswith("250 2.0.0")
    assert smtp_conversation(handler, session(), envelope(), many[:1], MESSAGE)[0].startswith("451")  # 4th in a minute
    assert smtp_conversation(handler, session("203.0.113.99"), envelope(), many[:1], MESSAGE)[0] == "250 OK"
    clock[0] += 61
    db, accounts = dbs(world)
    sizes = [r[0] for r in db.execute("SELECT size FROM mail_messages ORDER BY id")]
    assert len(sizes) == 3 and len(set(sizes)) == 1
    config["mailbox_bytes"] = 2 * sizes[0]                                                 # room for two of them
    mail.sweep(db, accounts, config=config)
    assert len(mail.inbox(db, 2)) == 2
    assert db.execute("SELECT detail FROM mail_recipients WHERE state = 'failed'").fetchone()[0] == "mailbox full"
    replies = smtp_conversation(handler, session("203.0.113.10"), envelope(), many[:1], MESSAGE)
    assert replies[1].startswith("452 4.2.2")                                              # ann's mailbox is full

    four = mail.Inbound(world["mail_db"], world["accounts_db"], {**world["config"], "max_recipients": 1})
    env = envelope()
    replies = smtp_conversation(four, session("203.0.113.11"), env, many[:2], MESSAGE)
    assert replies[2].startswith("452 4.5.3")


class FakeProvider:
    """The SMTP client calls send_message makes, answered from a script."""
    def __init__(self, rcpt=None, data=(250, b"2.0.0 Ok: queued as ABC123"), login_error=None):
        self.rcpt_codes, self.data_reply, self.login_error = dict(rcpt or {}), data, login_error
        self.calls, self.wire = [], None

    def __call__(self, config):
        self.calls.append(("connect", config["provider_host"]))
        return self

    def login(self, user, password):
        self.calls.append(("login", user, password == SECRET))
        if self.login_error:
            raise self.login_error

    def mail(self, sender):
        self.calls.append(("mail", sender))
        return 250, b"ok"

    def rcpt(self, address):
        self.calls.append(("rcpt", address))
        return self.rcpt_codes.get(address, (250, b"ok"))

    def data(self, wire):
        self.wire = wire
        return self.data_reply

    def quit(self):
        self.calls.append(("quit",))


def queue(world, to, key="k" * 16, subject="Your invitation", text="Come and see.\n", now=None):
    db, accounts = dbs(world)
    return mail.queue_outgoing(db, accounts, user={"id": 2, "username": "Ann", "display_name": "Ann A"}, to=to,
                               subject=subject, text=text, idempotency_key=key, config=world["config"], now=now)


def test_outgoing_goes_from_the_verified_address_with_no_reply_to_and_local_copies_are_filed(world):
    queued = queue(world, ["friend@example.org", "bob@stichka.duckdns.org"])
    assert [r["route"] for r in queued["recipients"]] == ["outside", "local"]
    asked = []
    assert mail.route_after_queue(queued, lambda task, mid: asked.append((task, mid))).startswith("the mail table")
    assert asked == [("mail.file", queued["id"]), ("mail.send", queued["id"])]
    db, accounts = dbs(world)
    assert mail.file_message(db, accounts, queued["id"], config=world["config"]) == 1
    provider = FakeProvider()
    counts = mail.send_message(db, queued["id"], config=world["config"], smtp_factory=provider)
    assert counts["sent"] == 1
    assert ("login", "relay-user", True) in provider.calls and ("mail", "noreply@example.net") in provider.calls
    wire = provider.wire.decode("utf-8")
    assert 'From: "Ann A via stichka.duckdns.org" <noreply@example.net>' in wire
    parsed = email.message_from_bytes(provider.wire, policy=email.policy.default)
    assert parsed["Reply-To"] is None and "Replies to this address are not read" in parsed.get_content()
    assert "To: friend@example.org, bob@stichka.duckdns.org" in wire
    sent = mail.read(db, 2, mail.inbox(db, 2, folder="sent")[0]["id"])
    assert {r["address"]: r["state"] for r in sent["recipients"]} == {"friend@example.org": "sent",
                                                                       "bob@stichka.duckdns.org": "delivered"}
    assert "ABC123" in [r for r in sent["recipients"] if r["route"] == "outside"][0]["detail"]
    assert sent["from"] == "Ann A <ann@stichka.duckdns.org>"          # the stored copy keeps its author
    assert db.execute("SELECT state FROM mail_messages WHERE id = ?", (queued["id"],)).fetchone()[0] == "done"
    assert mail.inbox(db, 3)[0]["subject"] == "Your invitation"
    everything = json.dumps([list(r) for r in db.execute("SELECT * FROM mail_receipts")])
    assert SECRET not in everything


def test_temporary_failures_are_retried_refusals_fail_and_nothing_tells_the_secret(world):
    queued = queue(world, ["a@example.org", "b@example.org", "c@example.org"], now=1000.0)
    db, _accounts = dbs(world)
    provider = FakeProvider(rcpt={"b@example.org": (450, b"try later"), "c@example.org": (550, b"no such user")})
    counts = mail.send_message(db, queued["id"], config=world["config"], smtp_factory=provider, now=1000.0)
    assert counts == {"sent": 1, "deferred": 1, "failed": 1, "waiting": 0}
    row = db.execute("SELECT state, attempts, next_attempt FROM mail_recipients WHERE address = 'b@example.org'").fetchone()
    assert tuple(row) == ("deferred", 1, 1000.0 + 5 * 60)
    assert mail.send_message(db, queued["id"], config=world["config"], smtp_factory=provider, now=1100.0)["sent"] == 0
    login_fails = FakeProvider(login_error=smtplib.SMTPAuthenticationError(535, f"bad {SECRET}".encode()))
    counts = mail.send_message(db, queued["id"], config=world["config"], smtp_factory=login_fails, now=1400.0)
    assert counts["deferred"] == 1
    detail = db.execute("SELECT detail FROM mail_recipients WHERE address = 'b@example.org'").fetchone()[0]
    assert detail.startswith("SMTPAuthenticationError: 535") and SECRET not in detail.replace(f"bad {SECRET}", "")
    late = 1000.0 + 73 * 3600
    counts = mail.send_message(db, queued["id"], config=world["config"], smtp_factory=FakeProvider(rcpt={"b@example.org": (451, b"busy")}), now=late)
    assert counts["failed"] == 1
    assert db.execute("SELECT detail FROM mail_recipients WHERE address = 'b@example.org'").fetchone()[0].startswith("gave up")


def test_the_provider_reply_detail_never_carries_the_password_from_our_own_text(world):
    """Our own receipts never contain the credential: only what the provider said is recorded."""
    queued = queue(world, ["a@example.org"])
    db, _accounts = dbs(world)
    mail.send_message(db, queued["id"], config=world["config"], smtp_factory=FakeProvider())
    rows = json.dumps([list(r) for r in db.execute("SELECT * FROM mail_receipts")] +
                      [list(r) for r in db.execute("SELECT id, address, detail FROM mail_recipients")])
    assert SECRET not in rows and "relay-user" not in rows


def test_without_a_provider_outgoing_mail_waits(world):
    world["config"]["credential_file"] = str(Path(world["config"]["credential_file"]).with_name("absent.json"))
    queued = queue(world, ["a@example.org"])
    db, accounts = dbs(world)
    provider = FakeProvider()
    assert mail.send_message(db, queued["id"], config=world["config"], smtp_factory=provider)["waiting"] == 1
    assert provider.calls == []
    assert mail.sweep(db, accounts, config=world["config"], smtp_factory=provider)["sent"] == 0
    assert db.execute("SELECT state FROM mail_recipients").fetchone()[0] == "queued"


def test_idempotency_limits_and_who_may_send(world):
    first = queue(world, ["a@example.org"])
    assert queue(world, ["a@example.org"])["id"] == first["id"]
    with pytest.raises(mail.MailError) as refused:
        queue(world, ["a@example.org"], text="different")
    assert refused.value.status == 409
    with pytest.raises(mail.MailError):
        queue(world, ["nobody@stichka.duckdns.org"], key="n" * 16)
    with pytest.raises(mail.MailError):
        queue(world, ["a@example.org", "b@example.org", "c@example.org", "d@example.org"], key="m" * 16)
    with pytest.raises(mail.MailError):
        queue(world, ["a@example.org"], key="s" * 16, subject="two\r\nBcc: x@example.org")
    world["config"]["outgoing_per_day"] = 2
    queue(world, ["a@example.org"], key="2" * 16)
    with pytest.raises(mail.MailError) as limited:
        queue(world, ["a@example.org"], key="3" * 16)
    assert limited.value.status == 429
    queue(world, ["bob@stichka.duckdns.org"], key="4" * 16)          # local mail is not counted against the limit
    db, accounts = dbs(world)
    with pytest.raises(mail.MailError) as service:
        mail.queue_outgoing(db, accounts, user={"id": 1, "username": "svc-meowdoku-a"}, to="a@example.org",
                            subject="", text="x", idempotency_key="z" * 16, config=world["config"])
    assert service.value.status == 403


def test_send_to_outside_is_the_call_the_invitation_backend_uses(world):
    asked = []
    queued = mail.send_to_outside(world["mail_db"], world["accounts_db"], account="bob", to="new@example.org",
                                  subject="You are invited", text="Open the link.", config=world["config"],
                                  idempotency_key="invite-" + "1" * 12, submit=lambda t, m: asked.append(t))
    assert asked == ["mail.send"] and queued["recipients"][0]["route"] == "outside"


def test_the_mail_table_tasks(world):
    host = SimpleNamespace(DONE=1, UNCHANGED=2)
    tasks = mail.handlers(host, mail_db=world["mail_db"], accounts_db=world["accounts_db"], config=world["config"],
                          smtp_factory=FakeProvider())
    assert set(tasks) == set(mail.TASKS)
    queued = queue(world, ["a@example.org", "bob@stichka.duckdns.org"])
    assert tasks["mail.file"]({"task": "mail.file", "message": queued["id"]}) == (1, "1 delivered")
    assert tasks["mail.file"]({"task": "mail.file", "message": queued["id"]})[0] == 2
    assert tasks["mail.send"]({"task": "mail.send", "message": queued["id"]}) == (1, "1 sent")
    assert tasks["mail.sweep"]({"task": "mail.sweep"})[0] == 2
    with pytest.raises(ValueError):
        tasks["mail.send"]({"task": "mail.send", "message": "1; DROP"})


def test_mail_over_rest_is_about_the_session_account_only(world):
    caps = {2: {"message", "mail_send"}, 3: {"message"}}
    asked = []

    def ask(method, path, user=2, username="Ann", payload=None):
        sent = []

        def require(cap):
            if cap in caps.get(user, ()):
                return True
            sent.append((403, {"error": f"this needs {cap}"}))
            return False
        handler = SimpleNamespace(path=path, headers={"Content-Type": "application/json"}, _require=require,
                                  _principal=lambda: SimpleNamespace(user_id=user, username=username),
                                  _body_json=lambda limit: payload,
                                  _json=lambda document, status=200: sent.append((int(status), json.loads(json.dumps(document)))))
        mail.http_route(handler, path.split("?")[0], method, accounts_db=world["accounts_db"], mail_db=world["mail_db"],
                        config=world["config"], submit=lambda task, mid: asked.append(task))
        return sent[0]

    status, inbox = ask("GET", "/api/mail/inbox")
    assert status == 200 and inbox["address"] == "ann@stichka.duckdns.org" and inbox["messages"] == []
    payload = {"to": ["bob@stichka.duckdns.org"], "subject": "Hi", "text": "Hello Bob", "idempotency_key": "r" * 20}
    outside = {**payload, "to": ["friend@example.org"], "idempotency_key": "o" * 20}
    assert ask("POST", "/api/mail/send", user=3, username="bob", payload=outside)[0] == 403      # no mail_send
    assert ask("POST", "/api/mail/send", user=3, username="bob", payload={**payload, "to": "Ann@stichka.duckdns.org"})[0] == 200
    status, sent = ask("POST", "/api/mail/send", payload=payload)
    assert status == 200 and asked == ["mail.file", "mail.file"]
    db, accounts = dbs(world)
    mail.file_message(db, accounts, sent["id"], config=world["config"])
    bobs = ask("GET", "/api/mail/inbox", user=3, username="bob")[1]["messages"]
    assert [m["subject"] for m in bobs] == ["Hi"]
    assert ask("GET", f"/api/mail/message?id={bobs[0]['id']}")[0] == 404                       # not ann's
    assert ask("GET", f"/api/mail/message?id={bobs[0]['id']}", user=3, username="bob")[1]["body"].strip() == "Hello Bob"
    assert ask("GET", "/api/mail/message?id=x", user=3, username="bob")[0] == 400
    assert ask("GET", "/api/mail/inbox?folder=spam")[0] == 400
    assert ask("POST", "/api/mail/remove", user=3, username="bob", payload={"id": bobs[0]["id"]})[1]["removed"]
    assert ask("GET", "/api/mail/inbox", user=None, username=None)[0] == 403


def test_the_real_receiver_over_smtp_when_aiosmtpd_is_installed(world, tmp_path):
    controller_module = pytest.importorskip("aiosmtpd.controller")
    import socket
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    asked = []
    handler = mail.Inbound(world["mail_db"], world["accounts_db"], world["config"], on_received=asked.append)
    controller = controller_module.Controller(handler, hostname="127.0.0.1", port=port,
                                              data_size_limit=world["config"]["max_message_bytes"])
    controller.start()
    try:
        with smtplib.SMTP("127.0.0.1", port, timeout=10) as client:
            refused = client.sendmail("someone@peer.example", ["ann@stichka.duckdns.org", "x@gmail.com"], MESSAGE)
        assert list(refused) == ["x@gmail.com"]
    finally:
        controller.stop()
    db, accounts = dbs(world)
    mail.file_message(db, accounts, asked[0], config=world["config"])
    assert mail.inbox(db, 2)[0]["subject"] == "Hello there"


def test_the_mail_table_routes_a_message_through_a_real_nodus_canvas(world, tmp_path):
    """One tick of the email manager table: a real canvas host with a `mail` contract port, the real mail service as
    its host, and an engine asking for mail.file. Opt-in: NODUS_HEADLESS_TEST_DLL names the built headless library."""
    if not os.environ.get("NODUS_HEADLESS_TEST_DLL"):
        pytest.skip("Set NODUS_HEADLESS_TEST_DLL (and PATH for its dependencies) to run a real canvas host")
    import socket
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    blank, canvas, jobs = tmp_path / "blank.canvas", tmp_path / "mail.canvas", tmp_path / "jobs"
    blank.write_text("CANVAS V4\n1 1 56\nOFFSET 0 0\n\n", encoding="utf-8")
    subprocess.run([sys.executable, str(NODUS / "tools" / "add_contract_port.py"), "--canvas", str(blank), "--out", str(canvas),
                    "--prefix", "mail", "--port", str(port), "--loan", "3600"], check=True, capture_output=True, timeout=120)
    config_file = tmp_path / "mail.json"
    config_file.write_text(json.dumps(world["config"]), encoding="utf-8")
    canvas_host = subprocess.Popen([sys.executable, str(NODUS / "tools" / "run_canvas.py"), "--canvas", str(canvas),
                                    "--acquire", "mail:4:3600", "--rate", "60"],
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    service = None
    try:
        first = canvas_host.stdout.readline()
        ports = {p["prefix"]: p["listen_port"] for p in json.loads(first)["contract_ports"]}
        assert ports == {"mail": port}
        service = subprocess.Popen([sys.executable, str(ROOT / "examples" / "meowdoku" / "deploy" / "mail_service.py"),
                                    "--config", str(config_file), "--accounts-db", str(world["accounts_db"]),
                                    "--mail-db", str(world["mail_db"]), "--mail-jobs-dir", str(jobs),
                                    "--mail-port", str(port), "--no-receiver"],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        said = []
        while not any("the mail table performs" in line for line in said):
            line = service.stdout.readline()
            assert line, "the mail service ended: " + "".join(said)
            said.append(line)
        queued = queue(world, ["bob@stichka.duckdns.org"])
        tools = studio_product_nodus.DEFAULT_NODUS_TOOLS
        task_request = scheduled_tasks.load_task_request(tools)
        outcome = task_request.submit(studio_product_nodus.nodus_modules(tools)[0], port=port,
                                      engine_id=mail.ENGINE_IDS["server"], jobs_dir=jobs, task="mail.file",
                                      fields={"message": queued["id"]}, timeout=30.0)
        assert outcome["outcome"] == "done" and outcome["status"] == 1, outcome
        assert task_request.result(jobs, outcome["contract"])["detail"] == "1 delivered"
        db, _accounts = dbs(world)
        assert [m["subject"] for m in mail.inbox(db, 3)] == ["Your invitation"]
    finally:
        for process in (service, canvas_host):
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
