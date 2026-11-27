"""SpeakToMe mail: mailboxes for the service's accounts, mail in on an odd port, mail out through a provider.

The domain is ``stichka.duckdns.org`` (``deploy/mail.json``). Every PERSON account (not a service account) has the
address ``<username>@<domain>``. Mail comes in on an odd high port (47625), never port 25: the owner's decision
(2026-11-13) is that only our own servers, or someone who configured the port by hand, know there is mail to
connect to, and that is itself a spam filter. Ordinary internet senders only ever try port 25 of the MX, so their
mail does not arrive here; outgoing mail therefore carries no Reply-To at this domain, and its text says that
answers go through the site.

THE ROUTE. Every message and every recipient is a row in ``<state>/sql/mail.db`` (this module is its one writer;
account ids join to ``accounts.db`` ``users.id``). The routing between the rows is the Nodus EMAIL MANAGER TABLE:
the contract prefix ``mail`` of the service canvas, like the ``userlist`` and ``cron`` tables. Whoever stores a
row asks that table for one tick (nodus ``tools/task_request.py``); the host of the table (``deploy/mail_service.py``)
performs the task (``handlers`` below):

    mail.file    file a message's local recipients into their mailboxes          (asked for by the receiver and
                                                                                  by a send to a local address)
    mail.send    submit a message's outside recipients to the provider           (asked for by a send)
    mail.sweep   both, for every row that is still waiting                       (the table's own cron tool, every
                                                                                  few minutes; catches anything a
                                                                                  lost contract left behind)

A request names a message id and nothing else: no body, no address, no credential.

    receive(db, accounts, peer=, helo=, tls=, mail_from=, recipients=, data=, config=)   -> message id (in)
    queue_outgoing(db, accounts, user=, to=, subject=, text=, idempotency_key=, config=)   -> {"id", ...} (out)
    file_message(db, accounts, message_id, config=)        local recipients -> mailboxes
    send_message(db, message_id, config=, smtp_factory=)   outside recipients -> the provider
    sweep(db, accounts, config=, smtp_factory=)            every waiting row
    inbox(db, user_id, folder=, after=, limit=) / read(db, user_id, item) / remove(db, user_id, item)
    send_to_outside(mail_db, accounts_db, account=, to=, subject=, text=, config=, submit=)   for in-process callers
                                                           (the invitation backend)

OUTGOING. The provider sends From its verified address (``from_address``), shown as "<display name> via <domain>";
there is no Reply-To. The provider credential is a JSON file ``{"username": ..., "password": ...}`` at
``credential_file``; it is read when a message is submitted and is never logged, returned or put in a row.

STATES. A recipient is ``queued`` -> ``delivered`` (local) | ``sent`` (accepted by the provider) | ``deferred``
(a temporary failure; tried again by the retry schedule until ``give_up_hours``) | ``failed`` (refused, or given
up). A message is ``queued`` while any recipient waits and ``done`` after. Every step is a ``mail_receipts`` row:
the SMTP peer, HELO and TLS of mail that came in; the provider's reply line for mail that went out.

Every REST question is about the session's own account. Nothing here produces HTML: a body is returned as text.
"""
from __future__ import annotations

import email
import email.policy
import hashlib
import json
import re
import smtplib
import sqlite3
import ssl
import sys
import threading
import time
from email.message import EmailMessage
from email.parser import BytesHeaderParser, BytesParser
from email.utils import formataddr, formatdate, getaddresses, make_msgid, parseaddr
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parents[1]), str(HERE.parents[1] / "examples")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from meowdoku import identity_tools  # noqa: E402
from meowdoku.accounts import USERNAME as LOCAL_PART  # noqa: E402  every username is a valid local part
from meowdoku import schema_version as _schema_version  # noqa: E402

DEFAULTS = {
    "domain": "stichka.duckdns.org",
    "listen_host": "0.0.0.0",
    "listen_port": 47625,
    "tls_cert": None,
    "tls_key": None,
    "max_message_bytes": 10 * 1024 * 1024,
    "max_recipients": 20,
    "per_ip_per_minute": 20,
    "mailbox_bytes": 200 * 1024 * 1024,
    "outgoing_per_day": 50,
    "max_subject": 200,
    "max_text": 100_000,
    "provider_host": None,
    "provider_port": 587,
    "provider_security": "starttls",
    "provider_timeout": 30,
    "from_address": None,
    "credential_file": None,
    "site_url": "https://stichka.duckdns.org:8766/",
    "retry_minutes": [5, 15, 60, 240, 720],
    "give_up_hours": 72,
}
SECURITY = ("starttls", "tls")
FOLDERS = ("inbox", "sent")
MAX_ADDRESS = 254
KEY_MIN, KEY_MAX = 16, 128
MAX_LIST = 200
TASKS = ("mail.file", "mail.send", "mail.sweep")

DDL = (
    "CREATE TABLE IF NOT EXISTS mail_messages (id INTEGER PRIMARY KEY AUTOINCREMENT, "
    "direction TEXT NOT NULL CHECK(direction IN ('in','out')), user_id INTEGER, "
    "state TEXT NOT NULL CHECK(state IN ('queued','done')), envelope_from TEXT, peer TEXT, helo TEXT, tls INTEGER, "
    "from_header TEXT, to_header TEXT, subject TEXT, date_header TEXT, message_id TEXT, size INTEGER NOT NULL, "
    "digest TEXT NOT NULL, raw BLOB NOT NULL, idempotency_key TEXT, request_digest TEXT, created REAL NOT NULL, "
    "finished REAL)",
    "CREATE UNIQUE INDEX IF NOT EXISTS mail_messages_key ON mail_messages(user_id, idempotency_key) "
    "WHERE idempotency_key IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS mail_messages_out ON mail_messages(direction, user_id, created)",
    "CREATE TABLE IF NOT EXISTS mail_recipients (id INTEGER PRIMARY KEY AUTOINCREMENT, message_id INTEGER NOT NULL, "
    "address TEXT NOT NULL, route TEXT NOT NULL CHECK(route IN ('local','outside')), user_id INTEGER, "
    "state TEXT NOT NULL CHECK(state IN ('queued','deferred','delivered','sent','failed')), "
    "attempts INTEGER NOT NULL DEFAULT 0, next_attempt REAL NOT NULL, updated REAL NOT NULL, detail TEXT)",
    "CREATE INDEX IF NOT EXISTS mail_recipients_waiting ON mail_recipients(state, route, next_attempt)",
    "CREATE INDEX IF NOT EXISTS mail_recipients_message ON mail_recipients(message_id)",
    "CREATE TABLE IF NOT EXISTS mail_receipts (id INTEGER PRIMARY KEY AUTOINCREMENT, message_id INTEGER NOT NULL, "
    "recipient_id INTEGER, step TEXT NOT NULL, status TEXT NOT NULL, detail TEXT, at REAL NOT NULL)",
    "CREATE INDEX IF NOT EXISTS mail_receipts_message ON mail_receipts(message_id, id)",
    "CREATE TABLE IF NOT EXISTS mail_mailbox (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, "
    "message_id INTEGER NOT NULL, folder TEXT NOT NULL CHECK(folder IN ('inbox','sent')), size INTEGER NOT NULL, "
    "created REAL NOT NULL, read_at REAL, removed_at REAL, UNIQUE(user_id, message_id, folder))",
    "CREATE INDEX IF NOT EXISTS mail_mailbox_owner ON mail_mailbox(user_id, folder, id)",
)


class MailError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message, self.status = message, status


# ---- configuration and connections -----------------------------------------------------------------------------------
def load_config(path=None) -> dict:
    given = json.loads(Path(path).read_text(encoding="utf-8")) if path else {}
    config = {**DEFAULTS, **{k: v for k, v in given.items() if k in DEFAULTS}}
    if config["provider_security"] not in SECURITY:
        raise ValueError(f"provider_security must be one of {', '.join(SECURITY)}")
    config["domain"] = str(config["domain"]).strip().lower()
    return config


def connect(db_path) -> sqlite3.Connection:
    """Own connection to mail.db (autocommit, WAL). Declares schema module ``mail`` 1."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(db_path), timeout=10, isolation_level=None, check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    try:
        _schema_version.declare(db, "mail", 1)
    except _schema_version.SchemaVersionError:
        db.close()
        raise
    for ddl in DDL:
        db.execute(ddl)
    return db


def connect_accounts(db_path) -> sqlite3.Connection:
    """A read connection to accounts.db: this module only looks accounts up there, it never writes."""
    db = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True, timeout=10, check_same_thread=False)
    db.row_factory = sqlite3.Row
    return db


def mail_db_beside(accounts_db) -> Path:
    """Where mail.db lives: the service's sql folder, beside accounts.db."""
    return Path(accounts_db).with_name("mail.db")


# ---- addresses --------------------------------------------------------------------------------------------------------
def _clean_address(text) -> str:
    if not isinstance(text, str):
        raise MailError("an address is text")
    name, address = parseaddr(text.strip())
    address = address.strip()
    if (not address or len(address) > MAX_ADDRESS or address.count("@") != 1 or any(c in address for c in "\r\n <>,;")
            or address.startswith("@") or address.endswith("@")):
        raise MailError(f"{text.strip()[:80]!r} is not an address")
    local, _, domain = address.partition("@")
    if "." not in domain:
        raise MailError(f"{address!r} is not an address")
    return f"{local}@{domain.lower()}"


def owner_of(accounts, address, config) -> dict | None:
    """The person account whose mailbox ``address`` is, or None. Service accounts and other domains have none."""
    local, _, domain = address.rpartition("@")
    if domain.lower() != config["domain"] or not LOCAL_PART.match(local):
        return None
    one = identity_tools.represent(accounts, local)
    if one is None or one["kind"] != "person":
        return None
    return {"id": int(one["identity"].split(":", 1)[1]), "username": one["username"],
            "display_name": one["display_name"]}


def address_of(accounts, user_id, config) -> str | None:
    """The account's own address, or None when it has none (a service account)."""
    one = identity_tools.represent(accounts, f"account:{int(user_id)}")
    if one is None or one["kind"] != "person" or not LOCAL_PART.match(one["username"]):
        return None
    return f"{one['username'].lower()}@{config['domain']}"


def mailbox_bytes(db, user_id) -> int:
    row = db.execute("SELECT COALESCE(SUM(size), 0) FROM mail_mailbox WHERE user_id = ? AND removed_at IS NULL",
                     (int(user_id),)).fetchone()
    return int(row[0])


def _receipt(db, message_id, step, status, detail=None, recipient_id=None, now=None) -> None:
    db.execute("INSERT INTO mail_receipts (message_id, recipient_id, step, status, detail, at) VALUES (?,?,?,?,?,?)",
               (message_id, recipient_id, step, status, None if detail is None else str(detail)[:500],
                time.time() if now is None else now))


def _headers(raw: bytes) -> dict:
    parsed = BytesHeaderParser(policy=email.policy.default).parsebytes(raw)

    def one(name):
        try:
            value = parsed.get(name)
        except Exception:                                  # noqa: BLE001  a malformed header from outside is data
            value = None
        return None if value is None else str(value)[:500]
    return {"from_header": one("From"), "to_header": one("To"), "subject": one("Subject"),
            "date_header": one("Date"), "message_id": one("Message-ID")}


def _store(db, *, direction, user_id, raw, envelope_from=None, peer=None, helo=None, tls=None,
           idempotency_key=None, request_digest=None, now) -> int:
    fields = _headers(raw)
    cursor = db.execute(
        "INSERT INTO mail_messages (direction, user_id, state, envelope_from, peer, helo, tls, from_header, to_header, "
        "subject, date_header, message_id, size, digest, raw, idempotency_key, request_digest, created) "
        "VALUES (?,?,'queued',?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (direction, user_id, envelope_from, peer, helo, None if tls is None else int(bool(tls)), fields["from_header"],
         fields["to_header"], fields["subject"], fields["date_header"], fields["message_id"], len(raw),
         hashlib.sha256(raw).hexdigest(), sqlite3.Binary(raw), idempotency_key, request_digest, now))
    return int(cursor.lastrowid)


def _recipient(db, message_id, address, route, user_id, now) -> int:
    cursor = db.execute("INSERT INTO mail_recipients (message_id, address, route, user_id, state, next_attempt, updated) "
                        "VALUES (?,?,?,?,'queued',?,?)", (message_id, address, route, user_id, now, now))
    return int(cursor.lastrowid)


def _settle(db, message_id, now) -> None:
    waiting = db.execute("SELECT COUNT(*) FROM mail_recipients WHERE message_id = ? AND state IN ('queued','deferred')",
                         (message_id,)).fetchone()[0]
    if not waiting:
        db.execute("UPDATE mail_messages SET state = 'done', finished = ? WHERE id = ? AND state = 'queued'",
                   (now, message_id))


# ---- mail that comes in -----------------------------------------------------------------------------------------------
def receive(db, accounts, *, peer, helo, tls, mail_from, recipients, data, config, now=None) -> int:
    """Store one message that came in over SMTP, for recipients the receiver already accepted. A Received header is
    put on top. Returns the message id; filing it into the mailboxes is ``file_message`` (the mail.file task)."""
    now = time.time() if now is None else now
    if len(data) > config["max_message_bytes"]:
        raise MailError("the message is too large", 413)
    owners = []
    for address in recipients:
        owner = owner_of(accounts, address, config)
        if owner is None:
            raise MailError(f"no mailbox {address!r}", 404)
        owners.append((address.lower(), owner))
    stamp = formatdate(now, usegmt=True)
    received = (f"Received: from {_header_safe(helo) or 'unknown'} ([{_header_safe(peer)}])\r\n"
                f"\tby {config['domain']} with {'ESMTPS' if tls else 'ESMTP'}; {stamp}\r\n").encode("ascii", "replace")
    raw = received + data
    db.execute("BEGIN IMMEDIATE")
    try:
        message_id = _store(db, direction="in", user_id=None, raw=raw, envelope_from=_header_safe(mail_from),
                            peer=_header_safe(peer), helo=_header_safe(helo), tls=tls, now=now)
        for address, owner in owners:
            _recipient(db, message_id, address, "local", owner["id"], now)
        _receipt(db, message_id, "received", "ok", f"from {peer} helo {helo!r} {'tls' if tls else 'plain'}; "
                 f"{len(raw)} bytes for {len(owners)}", now=now)
        db.execute("COMMIT")
    except BaseException:
        db.execute("ROLLBACK")
        raise
    return message_id


def _header_safe(text) -> str:
    return "" if text is None else re.sub(r"[\x00-\x1f\x7f]", " ", str(text))[:200]


class Inbound:
    """The SMTP handler (aiosmtpd's handler interface: ``handle_MAIL``, ``handle_RCPT``, ``handle_DATA``).

    Only ``<account>@<domain>`` of a person account is accepted (no relaying, ever), at most ``max_recipients`` a
    message, at most ``per_ip_per_minute`` messages from one address, a mailbox over ``mailbox_bytes`` refuses with a
    temporary failure. A stored message is handed to ``on_received(message_id)``, which asks the mail table to file
    it. It imports nothing of aiosmtpd, so it is tested without it."""

    def __init__(self, mail_db, accounts_db, config, *, on_received=None, clock=time.time, log=print):
        self.mail_db, self.accounts_db, self.config = mail_db, accounts_db, config
        self.on_received, self.clock, self.log = on_received, clock, log
        self._lock = threading.Lock()
        self._seen = {}

    def _rate_ok(self, ip) -> bool:
        now = self.clock()
        with self._lock:
            recent = [t for t in self._seen.get(ip, ()) if now - t < 60.0]
            if len(recent) >= self.config["per_ip_per_minute"]:
                self._seen[ip] = recent
                return False
            recent.append(now)
            self._seen[ip] = recent
            if len(self._seen) > 4096:                     # forget quiet addresses
                self._seen = {k: v for k, v in self._seen.items() if v and now - v[-1] < 60.0}
            return True

    async def handle_MAIL(self, server, session, envelope, address, mail_options):
        ip = (session.peer or ("?",))[0]
        if not self._rate_ok(ip):
            return "451 4.7.1 too many messages from your address; try again later"
        envelope.mail_from = address
        envelope.mail_options.extend(mail_options)
        return "250 OK"

    async def handle_RCPT(self, server, session, envelope, address, rcpt_options):
        if len(envelope.rcpt_tos) >= self.config["max_recipients"]:
            return "452 4.5.3 too many recipients"
        try:
            clean = _clean_address(address)
        except MailError:
            return "501 5.1.3 bad recipient address"
        if clean.rpartition("@")[2] != self.config["domain"]:
            return "550 5.7.1 relaying denied"
        accounts, db = connect_accounts(self.accounts_db), connect(self.mail_db)
        try:
            owner = owner_of(accounts, clean, self.config)
            if owner is None:
                return "550 5.1.1 no such mailbox"
            if mailbox_bytes(db, owner["id"]) >= self.config["mailbox_bytes"]:
                return "452 4.2.2 mailbox full"
        finally:
            accounts.close()
            db.close()
        if clean not in envelope.rcpt_tos:
            envelope.rcpt_tos.append(clean)
            envelope.rcpt_options.extend(rcpt_options)
        return "250 OK"

    async def handle_DATA(self, server, session, envelope):
        data = envelope.original_content if envelope.original_content is not None else envelope.content
        if isinstance(data, str):
            data = data.encode("utf-8", "surrogateescape")
        if not envelope.rcpt_tos:
            return "554 5.5.1 no valid recipients"
        if len(data) > self.config["max_message_bytes"]:
            return "552 5.3.4 message too big"
        accounts, db = connect_accounts(self.accounts_db), connect(self.mail_db)
        try:
            message_id = receive(db, accounts, peer=(session.peer or ("?",))[0], helo=session.host_name,
                                 tls=session.ssl is not None, mail_from=envelope.mail_from,
                                 recipients=list(envelope.rcpt_tos), data=data, config=self.config, now=self.clock())
        except MailError as error:
            return f"550 5.1.1 {error.message}" if error.status == 404 else f"552 5.3.4 {error.message}"
        except sqlite3.Error as error:
            self.log(f"[mail] storing a message failed: {type(error).__name__}")
            return "451 4.3.0 temporary failure; try again later"
        finally:
            accounts.close()
            db.close()
        if self.on_received is not None:
            try:
                self.on_received(message_id)
            except Exception as error:                     # noqa: BLE001  the sweep files it later
                self.log(f"[mail] message {message_id}: the mail table was not asked ({type(error).__name__})")
        return f"250 2.0.0 OK queued as {message_id}"


# ---- mail that goes out -----------------------------------------------------------------------------------------------
def _recipients_list(to) -> list:
    if isinstance(to, str):
        to = [to]
    if not isinstance(to, (list, tuple)) or not to:
        raise MailError("name at least one recipient")
    found = []
    for item in to:
        if not isinstance(item, str):
            raise MailError("each recipient is text")
        pairs = getaddresses([item])
        if not pairs or any(not address for _name, address in pairs):
            raise MailError(f"{item.strip()[:80]!r} is not an address")
        found.extend(_clean_address(address) for _name, address in pairs)
    unique = []
    for address in found:
        if address.lower() not in [a.lower() for a in unique]:
            unique.append(address)
    return unique


def queue_outgoing(db, accounts, *, user, to, subject, text, idempotency_key, config, now=None) -> dict:
    """Queue one plain-text message from ``user`` ({id, username, display_name}). Local recipients are filed by
    mail.file, outside ones submitted by mail.send. A replay with the same idempotency key returns the first
    message; the same key with different content is refused (409)."""
    now = time.time() if now is None else now
    if not isinstance(idempotency_key, str) or not KEY_MIN <= len(idempotency_key) <= KEY_MAX:
        raise MailError(f"idempotency_key is text of {KEY_MIN} to {KEY_MAX} characters")
    if not isinstance(subject, str) or len(subject) > config["max_subject"] or any(c in subject for c in "\r\n"):
        raise MailError(f"subject is one line of at most {config['max_subject']} characters")
    if not isinstance(text, str) or not text.strip() or len(text) > config["max_text"]:
        raise MailError(f"text is 1 to {config['max_text']} characters")
    addresses = _recipients_list(to)
    if len(addresses) > config["max_recipients"]:
        raise MailError(f"at most {config['max_recipients']} recipients")
    request_digest = hashlib.sha256(json.dumps([addresses, subject, text]).encode("utf-8")).hexdigest()
    held = db.execute("SELECT id, request_digest FROM mail_messages WHERE user_id = ? AND idempotency_key = ?",
                      (user["id"], idempotency_key)).fetchone()
    if held is not None:
        if held["request_digest"] != request_digest:
            raise MailError("this idempotency_key was used for a different message", 409)
        return {"id": int(held["id"]), "replayed": True, "recipients": _recipient_rows(db, int(held["id"]))}
    sender = address_of(accounts, user["id"], config)
    if sender is None:
        raise MailError("this account has no address", 403)
    routes = []
    for address in addresses:
        if address.rpartition("@")[2] == config["domain"]:
            owner = owner_of(accounts, address, config)
            if owner is None:
                raise MailError(f"no mailbox {address!r}", 404)
            routes.append((address.lower(), "local", owner["id"]))
        else:
            routes.append((address, "outside", None))
    if any(route == "outside" for _a, route, _u in routes):
        sent_today = db.execute("SELECT COUNT(*) FROM mail_messages m WHERE m.direction = 'out' AND m.user_id = ? "
                                "AND m.created > ? AND EXISTS (SELECT 1 FROM mail_recipients r WHERE r.message_id = m.id "
                                "AND r.route = 'outside')", (user["id"], now - 86400.0)).fetchone()[0]
        if sent_today >= config["outgoing_per_day"]:
            raise MailError(f"at most {config['outgoing_per_day']} outgoing messages a day", 429)
    message = EmailMessage()
    message["From"] = formataddr((user.get("display_name") or user["username"], sender))
    message["To"] = ", ".join(address for address, _r, _u in routes)
    message["Subject"] = subject
    message["Date"] = formatdate(now, usegmt=True)
    message["Message-ID"] = make_msgid(domain=config["domain"])
    message.set_content(text)
    raw = message.as_bytes(policy=email.policy.SMTP)
    db.execute("BEGIN IMMEDIATE")
    try:
        message_id = _store(db, direction="out", user_id=user["id"], raw=raw, envelope_from=sender,
                            idempotency_key=idempotency_key, request_digest=request_digest, now=now)
        for address, route, user_id in routes:
            _recipient(db, message_id, address, route, user_id, now)
        db.execute("INSERT INTO mail_mailbox (user_id, message_id, folder, size, created) VALUES (?,?,'sent',?,?)",
                   (user["id"], message_id, len(raw), now))
        _receipt(db, message_id, "queued", "ok", f"{len(routes)} recipient(s)", now=now)
        db.execute("COMMIT")
    except BaseException:
        db.execute("ROLLBACK")
        raise
    return {"id": message_id, "replayed": False, "recipients": _recipient_rows(db, message_id)}


def _recipient_rows(db, message_id) -> list:
    return [{"address": row["address"], "route": row["route"], "state": row["state"], "detail": row["detail"]}
            for row in db.execute("SELECT address, route, state, detail FROM mail_recipients WHERE message_id = ? "
                                  "ORDER BY id", (message_id,))]


def file_message(db, accounts, message_id, *, config, now=None) -> int:
    """File the waiting local recipients of one message into their inboxes. Returns how many were delivered."""
    now = time.time() if now is None else now
    message = db.execute("SELECT id, size FROM mail_messages WHERE id = ?", (int(message_id),)).fetchone()
    if message is None:
        raise MailError("no such message", 404)
    delivered = 0
    for row in db.execute("SELECT id, address, user_id FROM mail_recipients WHERE message_id = ? AND route = 'local' "
                          "AND state IN ('queued','deferred')", (message["id"],)).fetchall():
        if row["user_id"] is None or owner_of(accounts, row["address"], config) is None:
            db.execute("UPDATE mail_recipients SET state = 'failed', updated = ?, detail = 'no such mailbox' WHERE id = ?",
                       (now, row["id"]))
            _receipt(db, message["id"], "file", "failed", "no such mailbox", row["id"], now)
            continue
        if mailbox_bytes(db, row["user_id"]) + message["size"] > config["mailbox_bytes"]:
            db.execute("UPDATE mail_recipients SET state = 'failed', updated = ?, detail = 'mailbox full' WHERE id = ?",
                       (now, row["id"]))
            _receipt(db, message["id"], "file", "failed", "mailbox full", row["id"], now)
            continue
        db.execute("BEGIN IMMEDIATE")
        try:
            db.execute("INSERT OR IGNORE INTO mail_mailbox (user_id, message_id, folder, size, created) "
                       "VALUES (?,?,'inbox',?,?)", (row["user_id"], message["id"], message["size"], now))
            db.execute("UPDATE mail_recipients SET state = 'delivered', updated = ?, detail = NULL WHERE id = ?",
                       (now, row["id"]))
            _receipt(db, message["id"], "file", "delivered", None, row["id"], now)
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise
        delivered += 1
    _settle(db, message["id"], now)
    return delivered


def provider_ready(config) -> str | None:
    """None when outgoing mail can be submitted; otherwise why not (no secret is ever part of the answer)."""
    for key in ("provider_host", "from_address", "credential_file"):
        if not config.get(key):
            return f"no {key} is configured"
    if not Path(config["credential_file"]).is_file():
        return "the credential file is not there"
    return None


def _credential(config) -> tuple:
    """Server-side only: never log, return or store this value."""
    try:
        data = json.loads(Path(config["credential_file"]).read_text(encoding="utf-8"))
        return str(data["username"]), str(data["password"])
    except (OSError, ValueError, KeyError, TypeError):
        raise MailError("the provider credential is unreadable", 503) from None


def wire_copy(raw: bytes, *, config, sender_name) -> bytes:
    """The copy that goes to the provider: From the verified address, no Reply-To, and a line saying how to answer."""
    message = BytesParser(policy=email.policy.default).parsebytes(raw)
    for name in ("From", "Reply-To", "Sender"):
        del message[name]
    message["From"] = formataddr((f"{sender_name} via {config['domain']}", config["from_address"]))
    body = message.get_content()
    footer = (f"\n\n--\nSent by {sender_name} through {config['domain']}. Replies to this address are not read: "
              f"to answer, sign in at {config['site_url']} and use Messages.\n")
    message.set_content(body.rstrip("\n") + footer)
    return message.as_bytes(policy=email.policy.SMTP)


def _retry_at(config, attempts, now) -> float:
    steps = config["retry_minutes"] or [15]
    return now + 60.0 * float(steps[min(attempts, len(steps)) - 1])


def send_message(db, message_id, *, config, smtp_factory=None, now=None) -> dict:
    """Submit the due outside recipients of one message to the provider. Returns {sent, deferred, failed, waiting}.

    A refused recipient (5xx) or a refused message fails; anything temporary (4xx, the connection, the login, the
    provider refusing our sender) defers it on the retry schedule, until ``give_up_hours`` after it was queued."""
    now = time.time() if now is None else now
    message = db.execute("SELECT m.id, m.raw, m.created, m.user_id FROM mail_messages m WHERE m.id = ? AND "
                         "m.direction = 'out'", (int(message_id),)).fetchone()
    if message is None:
        raise MailError("no such outgoing message", 404)
    due = db.execute("SELECT id, address, attempts FROM mail_recipients WHERE message_id = ? AND route = 'outside' AND "
                     "state IN ('queued','deferred') AND next_attempt <= ? ORDER BY id", (message["id"], now)).fetchall()
    counts = {"sent": 0, "deferred": 0, "failed": 0, "waiting": 0}
    if not due:
        _settle(db, message["id"], now)
        return counts
    why = provider_ready(config)
    if why is not None:
        counts["waiting"] = len(due)
        _receipt(db, message["id"], "send", "waiting", why, now=now)
        return counts
    sender = BytesHeaderParser(policy=email.policy.default).parsebytes(message["raw"])
    sender_name = (parseaddr(str(sender.get("From") or ""))[0] or "someone").strip() or "someone"
    wire = wire_copy(message["raw"], config=config, sender_name=sender_name)

    marked = set()

    def mark(row, state, detail):
        marked.add(row["id"])
        attempts = int(row["attempts"]) + 1
        if state == "deferred" and now - float(message["created"]) > 3600.0 * float(config["give_up_hours"]):
            state, detail = "failed", f"gave up after {attempts} attempts: {detail}"
        db.execute("UPDATE mail_recipients SET state = ?, attempts = ?, next_attempt = ?, updated = ?, detail = ? "
                   "WHERE id = ?", (state, attempts, _retry_at(config, attempts, now) if state == "deferred" else now,
                                    now, str(detail)[:500], row["id"]))
        _receipt(db, message["id"], "send", state, detail, row["id"], now)
        counts[state] += 1

    factory = smtp_factory or _smtp
    try:
        username, password = _credential(config)
        client = factory(config)
    except (MailError, OSError, smtplib.SMTPException) as error:
        for row in due:
            mark(row, "deferred", _describe(error))
        _settle(db, message["id"], now)
        return counts
    try:
        try:
            client.login(username, password)
            code, reply = client.mail(config["from_address"])
            if code != 250:
                raise smtplib.SMTPSenderRefused(code, reply, config["from_address"])
            accepted = []
            for row in due:
                code, reply = client.rcpt(row["address"])
                if code in (250, 251):
                    accepted.append(row)
                else:
                    mark(row, "failed" if code >= 500 else "deferred", _reply(code, reply))
            if accepted:
                code, reply = client.data(wire)
                for row in accepted:
                    if code == 250:
                        mark(row, "sent", _reply(code, reply))
                    else:
                        mark(row, "failed" if code >= 500 else "deferred", _reply(code, reply))
        finally:
            try:
                client.quit()
            except (OSError, smtplib.SMTPException):
                pass
    except (OSError, smtplib.SMTPException) as error:
        for row in due:
            if row["id"] not in marked:
                mark(row, "deferred", _describe(error))
    _settle(db, message["id"], now)
    return counts


def _reply(code, reply) -> str:
    text = reply.decode("utf-8", "replace") if isinstance(reply, bytes) else str(reply)
    return f"{code} {text}".strip()[:500]


def _describe(error) -> str:
    """A failure in words, without anything the error might carry about the credential."""
    if isinstance(error, MailError):
        return error.message
    if isinstance(error, smtplib.SMTPResponseException):
        return f"{type(error).__name__}: {_reply(error.smtp_code, error.smtp_error)}"
    return type(error).__name__


def _smtp(config):
    """A connection to the provider's submission port, TLS verified against the system's trust store."""
    context = ssl.create_default_context()
    host, port, timeout = config["provider_host"], int(config["provider_port"]), float(config["provider_timeout"])
    if config["provider_security"] == "tls":
        client = smtplib.SMTP_SSL(host, port, timeout=timeout, context=context)
        client.ehlo(config["domain"])
        return client
    client = smtplib.SMTP(host, port, timeout=timeout, local_hostname=config["domain"])
    client.ehlo()
    client.starttls(context=context)
    client.ehlo()
    return client


def sweep(db, accounts, *, config, smtp_factory=None, now=None) -> dict:
    """Every waiting row: local recipients filed, due outside recipients submitted."""
    now = time.time() if now is None else now
    filed = sent = 0
    local = [r[0] for r in db.execute("SELECT DISTINCT message_id FROM mail_recipients WHERE route = 'local' AND "
                                      "state IN ('queued','deferred')")]
    for message_id in local:
        filed += file_message(db, accounts, message_id, config=config, now=now)
    outside = [r[0] for r in db.execute("SELECT DISTINCT message_id FROM mail_recipients WHERE route = 'outside' AND "
                                        "state IN ('queued','deferred') AND next_attempt <= ?", (now,))]
    for message_id in outside:
        sent += send_message(db, message_id, config=config, smtp_factory=smtp_factory, now=now)["sent"]
    return {"messages": len(set(local) | set(outside)), "filed": filed, "sent": sent}


# ---- mailboxes --------------------------------------------------------------------------------------------------------
def inbox(db, user_id, *, folder="inbox", after=0, limit=50) -> list:
    if folder not in FOLDERS:
        raise MailError(f"folder is one of {', '.join(FOLDERS)}")
    limit = max(1, min(int(limit), MAX_LIST))
    rows = db.execute("SELECT b.id, b.folder, b.size, b.created, b.read_at, m.from_header, m.to_header, m.subject, "
                      "m.date_header, m.state FROM mail_mailbox b JOIN mail_messages m ON m.id = b.message_id "
                      "WHERE b.user_id = ? AND b.folder = ? AND b.removed_at IS NULL AND b.id > ? ORDER BY b.id DESC "
                      "LIMIT ?", (int(user_id), folder, int(after), limit)).fetchall()
    return [{"id": row["id"], "folder": row["folder"], "from": row["from_header"], "to": row["to_header"],
             "subject": row["subject"], "date": row["date_header"], "size": row["size"], "received": row["created"],
             "read": row["read_at"] is not None, "state": row["state"]} for row in rows]


def read(db, user_id, item, *, now=None) -> dict:
    """One message of the caller's own mailbox, as text. Marks it read."""
    now = time.time() if now is None else now
    if isinstance(item, bool) or not isinstance(item, int):
        raise MailError("id must be a whole number")
    row = db.execute("SELECT b.id, b.folder, b.message_id, b.read_at, m.raw, m.direction FROM mail_mailbox b "
                     "JOIN mail_messages m ON m.id = b.message_id WHERE b.id = ? AND b.user_id = ? AND "
                     "b.removed_at IS NULL", (item, int(user_id))).fetchone()
    if row is None:
        raise MailError("no such message", 404)
    message = BytesParser(policy=email.policy.default).parsebytes(row["raw"])

    def header(name):
        try:
            value = message.get(name)
        except Exception:                                  # noqa: BLE001
            return None
        return None if value is None else str(value)
    body, kind = "", None
    try:
        part = message.get_body(preferencelist=("plain", "html"))
        if part is not None:
            body, kind = part.get_content(), part.get_content_subtype()
    except (LookupError, KeyError, ValueError):
        body, kind = "", None
    attachments = []
    for part in message.iter_attachments():
        payload = part.get_payload(decode=True) or b""
        attachments.append({"filename": part.get_filename(), "content_type": part.get_content_type(),
                            "size": len(payload)})
    if row["read_at"] is None:
        db.execute("UPDATE mail_mailbox SET read_at = ? WHERE id = ?", (now, row["id"]))
    answer = {"id": row["id"], "folder": row["folder"], "from": header("From"), "to": header("To"), "cc": header("Cc"),
              "subject": header("Subject"), "date": header("Date"), "message_id": header("Message-ID"),
              "body": body, "body_kind": kind, "attachments": attachments}
    if row["direction"] == "out":
        answer["recipients"] = _recipient_rows(db, row["message_id"])
    return answer


def remove(db, user_id, item, *, now=None) -> dict:
    if isinstance(item, bool) or not isinstance(item, int):
        raise MailError("id must be a whole number")
    changed = db.execute("UPDATE mail_mailbox SET removed_at = ? WHERE id = ? AND user_id = ? AND removed_at IS NULL",
                         (time.time() if now is None else now, item, int(user_id))).rowcount
    if not changed:
        raise MailError("no such message", 404)
    return {"id": item, "removed": True}


# ---- callers ----------------------------------------------------------------------------------------------------------
def route_after_queue(queued, submit) -> str:
    """Ask the mail table to route a freshly queued message: mail.file for local recipients, mail.send for outside
    ones. ``submit(task, message_id)`` is the caller's way to reach the table; without one the sweep routes it."""
    if submit is None:
        return "the mail table's sweep will route it"
    routes = {r["route"] for r in queued["recipients"] if r["state"] in ("queued", "deferred")}
    for task, route in (("mail.file", "local"), ("mail.send", "outside")):
        if route in routes:
            submit(task, queued["id"])
    return "the mail table was asked to route it"


ENGINE_IDS = {"receiver": 7205, "server": 7206}      # the cron tool of the mail table is 7204 (instances.json)
HOST_ID = 7303                                       # beside the cron host 7301 and the userlist host 7302
SUBMIT_TIMEOUT = 180.0                               # a send may wait on the provider; the engine stays registered


def table_submitter(*, port, jobs_dir, engine_id, nodus_tools=None, log=print):
    """``submit(task, message_id)`` that asks the mail table (contract port ``port``) for one tick, on a thread of its
    own so neither a request nor the SMTP conversation waits for it. A lost ask is harmless: the sweep routes the
    row later. None when no port was given."""
    if not port or not jobs_dir:
        return None
    from meowdoku import scheduled_tasks, studio_product_nodus
    tools = nodus_tools or studio_product_nodus.DEFAULT_NODUS_TOOLS
    native, task_request = studio_product_nodus.nodus_modules(tools)[0], scheduled_tasks.load_task_request(tools)

    def ask(task, message_id):
        try:
            outcome = task_request.submit(native, port=int(port), engine_id=int(engine_id), jobs_dir=jobs_dir,
                                          task=task, fields={"message": int(message_id)}, timeout=SUBMIT_TIMEOUT)
            if outcome["outcome"] != "done":
                log(f"[mail] {task} {message_id}: the mail table answered {outcome['outcome']}; the sweep will route it")
        except Exception as error:                     # noqa: BLE001
            log(f"[mail] {task} {message_id}: the mail table was not reached ({type(error).__name__})")

    def submit(task, message_id):
        threading.Thread(target=ask, args=(task, message_id), name=f"mail-{task}", daemon=True).start()
    return submit


def send_to_outside(mail_db, accounts_db, *, account, to, subject, text, config, idempotency_key, submit=None) -> dict:
    """Queue a message from ``account`` (a username) for in-process callers, such as the invitation backend.
    Whether the account may send is the caller's decision; the limits of ``queue_outgoing`` still apply."""
    accounts = connect_accounts(accounts_db)
    db = connect(mail_db)
    try:
        one = identity_tools.represent(accounts, account)
        if one is None:
            raise MailError("no such account", 404)
        user = {"id": int(one["identity"].split(":", 1)[1]), "username": one["username"],
                "display_name": one["display_name"]}
        queued = queue_outgoing(db, accounts, user=user, to=to, subject=subject, text=text,
                                idempotency_key=idempotency_key, config=config)
    finally:
        accounts.close()
        db.close()
    queued["routed"] = route_after_queue(queued, submit) if not queued["replayed"] else "already queued"
    return queued


def handlers(cron_host, *, mail_db, accounts_db, config, smtp_factory=None) -> dict:
    """The tasks of the mail table, for its host (``scheduled_tasks.start``-style; nodus ``tools/cron_host.py``)."""
    def with_dbs(work):
        def task(request):
            accounts, db = connect_accounts(accounts_db), connect(mail_db)
            try:
                return work(request, db, accounts)
            finally:
                accounts.close()
                db.close()
        return task

    def message_of(request) -> int:
        value = request.get("message")
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("the request names no message")
        return value

    def file_task(request, db, accounts):
        delivered = file_message(db, accounts, message_of(request), config=config)
        return (cron_host.DONE if delivered else cron_host.UNCHANGED), f"{delivered} delivered"

    def send_task(request, db, accounts):
        counts = send_message(db, message_of(request), config=config, smtp_factory=smtp_factory)
        said = ", ".join(f"{n} {state}" for state, n in counts.items() if n) or "nothing due"
        return (cron_host.DONE if counts["sent"] or counts["failed"] else cron_host.UNCHANGED), said

    def sweep_task(request, db, accounts):
        done = sweep(db, accounts, config=config, smtp_factory=smtp_factory)
        said = f"{done['messages']} waiting message(s): {done['filed']} filed, {done['sent']} sent"
        return (cron_host.DONE if done["filed"] or done["sent"] else cron_host.UNCHANGED), said

    return {"mail.file": with_dbs(file_task), "mail.send": with_dbs(send_task), "mail.sweep": with_dbs(sweep_task)}


def http_route(self, path, method, *, accounts_db, mail_db, config, submit=None):
    """Mailboxes over REST for the session's own account.

        GET  /api/mail/inbox?folder=inbox|sent&after=ID&limit=N     needs message; also says your address
        GET  /api/mail/message?id=N                                  needs message; marks it read
        POST /api/mail/send     {to, subject, text, idempotency_key} needs message; mail_send when any recipient
                                                                     is outside the domain
        POST /api/mail/remove   {id}                                 needs message
    """
    from urllib.parse import parse_qs
    self._set_cookie = []
    if method == "POST":
        self.close_connection = True
    if not self._require("message"):
        return None
    principal = self._principal()
    if principal.user_id is None:
        return self._json({"error": "sign in first"}, 401)
    accounts, db = connect_accounts(accounts_db), connect(mail_db)
    try:
        if method == "GET":
            query = parse_qs(self.path.partition("?")[2])

            def number(key, default):
                raw = (query.get(key) or [default])[0]
                if raw is None or not str(raw).isdigit():
                    raise MailError(f"{key} must be a whole number")
                return int(raw)
            if path == "/api/mail/inbox":
                folder = (query.get("folder") or ["inbox"])[0]
                return self._json({"address": address_of(accounts, principal.user_id, config),
                                   "folder": folder, "used_bytes": mailbox_bytes(db, principal.user_id),
                                   "quota_bytes": config["mailbox_bytes"],
                                   "messages": inbox(db, principal.user_id, folder=folder, after=number("after", 0),
                                                     limit=number("limit", 50))})
            if path == "/api/mail/message":
                return self._json(read(db, principal.user_id, number("id", None)))
            return self._json({"error": "not found"}, 404)
        if not (self.headers.get("Content-Type") or "").lower().startswith("application/json"):
            return self._json({"error": "JSON only"}, 415)
        body = self._body_json(config["max_text"] * 2 + 16384)
        if body is None:
            return None
        if path == "/api/mail/remove":
            return self._json(remove(db, principal.user_id, body.get("id")))
        if path == "/api/mail/send":
            outside = any(a.rpartition("@")[2] != config["domain"] for a in _recipients_list(body.get("to")))
            if outside and not self._require("mail_send"):
                return None
            one = identity_tools.represent(accounts, f"account:{int(principal.user_id)}")
            user = {"id": principal.user_id, "username": principal.username,
                    "display_name": (one or {}).get("display_name") or principal.username}
            queued = queue_outgoing(db, accounts, user=user, to=body.get("to"), subject=body.get("subject", ""),
                                    text=body.get("text"), idempotency_key=body.get("idempotency_key"), config=config)
            queued["routed"] = route_after_queue(queued, submit) if not queued["replayed"] else "already queued"
            return self._json(queued)
        return self._json({"error": "not found"}, 404)
    except MailError as error:
        return self._json({"error": error.message}, error.status)
    finally:
        accounts.close()
        db.close()
