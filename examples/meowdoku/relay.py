"""Datagrams and packages, address to address, in and out: the service's relay.

An account opens an address. A UDP address is a real UDP port of this host: anyone, anywhere, may send a datagram
to it, and its owner may send from it to anywhere, so a peer's reply comes back to the same address. That is all a
bidirectional UDP connection is. The same addresses carry packages between themselves without a socket: one
account's address to another's, any kind, any bytes.

    open_address(db, user_id=, username=, config=)      -> {"address": 47701, ...}
    close_address(db, address, user_id=)
    send(db, address, user_id=, to=, body=, kind=, config=)   to "198.51.100.7:9000" (a datagram, leaves from the address)
                                                              or to "47702" (a package, straight to that address)
    receive(db, address, user_id=, after=, limit=)      what arrived, oldest first, with who sent it

    udp_router <bind-host> <local-port>                 the native router (run by the supervisor)
    python relay.py --db <accounts.db> --config deploy/relay.json      this service, connected locally to the router

Every UDP socket belongs to the native router, ``udp_router`` (nodus ``tools/udp_router``), which the supervisor
runs. This module is one of the server's services and reaches the router over a local connection: it says which
addresses are open and what to send. The router is greedy: each round it drains everything that has arrived and
hands over ONE conglomerate record per destination, and that record is stored as one row. ``receive`` unpacks the
records, so a caller still reads datagrams, in arrival order, each with who sent it.

What keeps it from being a weapon: a datagram leaves only from an address its sender opened while signed in, one
datagram out for one request, at most ``rate_per_minute`` per address. A stranger can send TO an address, never
THROUGH the relay to a third party, so it cannot be used to reflect or amplify traffic at someone else.

Who a caller is comes from the authenticated host, never from a request. Other accounts' addresses are numbers;
nothing about their owner is told.
"""
from __future__ import annotations

import argparse
import base64
import json
import queue
import socket
import sqlite3
import struct
import sys
import threading
import time
from pathlib import Path

DEFAULTS = {"host": "0.0.0.0", "router": "127.0.0.1:47699", "first_port": 47700, "last_port": 47763, "addresses_per_account": 8, "max_body": 60000,
            "rate_per_minute": 600, "kept_per_address": 1000}
KIND_MAX = 64
DDL = (
    "CREATE TABLE IF NOT EXISTS relay_addresses (address INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, username TEXT, "
    "opened_at REAL NOT NULL, closed_at REAL)",
    "CREATE TABLE IF NOT EXISTS relay_packages (id INTEGER PRIMARY KEY AUTOINCREMENT, address INTEGER NOT NULL, "
    "direction TEXT NOT NULL CHECK(direction IN ('in','out')), peer TEXT NOT NULL, kind TEXT NOT NULL, body BLOB NOT NULL, "
    "created_at REAL NOT NULL, sent_at REAL)",
    "CREATE INDEX IF NOT EXISTS relay_packages_in ON relay_packages(address, direction, id)",
    "CREATE INDEX IF NOT EXISTS relay_packages_out ON relay_packages(direction, sent_at)",
)


class RelayError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message, self.status = message, status


def load_config(path=None) -> dict:
    given = json.loads(Path(path).read_text(encoding="utf-8")) if path else {}
    return {**DEFAULTS, **{k: v for k, v in given.items() if k in DEFAULTS}}


def ensure(db) -> None:
    for statement in DDL:
        db.execute(statement)


def _account(user_id) -> int:
    if isinstance(user_id, bool) or not isinstance(user_id, int):
        raise RelayError("a signed-in account is required", 401)
    return user_id


def _owned(db, address, user_id) -> int:
    _account(user_id)
    if isinstance(address, bool) or not isinstance(address, int):
        raise RelayError("address must be an address's number")
    row = db.execute("SELECT user_id FROM relay_addresses WHERE address = ? AND closed_at IS NULL", (address,)).fetchone()
    if row is None or row[0] != user_id:
        raise RelayError("that is not an open address of yours", 404)
    return address


def addresses(db, user_id) -> list:
    ensure(db)
    return [{"address": r[0], "opened_at": r[1]} for r in db.execute(
        "SELECT address, opened_at FROM relay_addresses WHERE user_id = ? AND closed_at IS NULL ORDER BY address", (_account(user_id),))]


def open_address(db, *, user_id, username, config, now=None) -> dict:
    ensure(db)
    _account(user_id)
    now = time.time() if now is None else now
    db.execute("BEGIN IMMEDIATE")
    try:
        if db.execute("SELECT COUNT(*) FROM relay_addresses WHERE user_id = ? AND closed_at IS NULL", (user_id,)).fetchone()[0] \
                >= config["addresses_per_account"]:
            raise RelayError(f"an account holds at most {config['addresses_per_account']} addresses; close one first", 429)
        taken = {r[0] for r in db.execute("SELECT address FROM relay_addresses WHERE closed_at IS NULL")}
        free = next((p for p in range(config["first_port"], config["last_port"] + 1) if p not in taken), None)
        if free is None:
            raise RelayError("every address is in use", 503)
        db.execute("INSERT INTO relay_addresses VALUES (?,?,?,?,NULL) ON CONFLICT(address) DO UPDATE SET user_id = excluded.user_id, "
                   "username = excluded.username, opened_at = excluded.opened_at, closed_at = NULL", (free, user_id, username, now))
        db.execute("DELETE FROM relay_packages WHERE address = ?", (free,))      # an address starts empty for its new owner
        db.execute("COMMIT")
    except Exception:
        db.execute("ROLLBACK")
        raise
    return {"address": free, "opened_at": now, "udp_port": free}


def close_address(db, address, *, user_id, now=None) -> dict:
    ensure(db)
    _owned(db, address, user_id)
    db.execute("UPDATE relay_addresses SET closed_at = ? WHERE address = ?", (time.time() if now is None else now, address))
    return {"address": address, "closed": True}


def _trim(db, address, config) -> None:
    db.execute("DELETE FROM relay_packages WHERE address = ? AND direction = 'in' AND id <= (SELECT id FROM relay_packages "
               "WHERE address = ? AND direction = 'in' ORDER BY id DESC LIMIT 1 OFFSET ?)", (address, address, config["kept_per_address"]))


def send(db, address, *, user_id, to, body, kind="datagram", config, now=None) -> dict:
    """One package out of one of your addresses: to ``host:port`` as a datagram, or to another address's number."""
    ensure(db)
    _owned(db, address, user_id)
    if not isinstance(body, (bytes, bytearray)) or len(body) > config["max_body"]:
        raise RelayError(f"body must be bytes, at most {config['max_body']} of them", 413)
    if not isinstance(kind, str) or not 0 < len(kind) <= KIND_MAX or not kind.isprintable():
        raise RelayError("kind must be a short word")
    if not isinstance(to, str) or not to:
        raise RelayError("to must be host:port, or an address's number")
    now = time.time() if now is None else now
    recent = db.execute("SELECT COUNT(*) FROM relay_packages WHERE address = ? AND direction = 'out' AND created_at > ?",
                        (address, now - 60.0)).fetchone()[0]
    if recent >= config["rate_per_minute"]:
        raise RelayError(f"this address has sent {config['rate_per_minute']} packages in the last minute; wait", 429)
    if to.isdigit():                             # address to address: no socket, any kind, delivered at once
        target = int(to)
        if db.execute("SELECT 1 FROM relay_addresses WHERE address = ? AND closed_at IS NULL", (target,)).fetchone() is None:
            raise RelayError("no such open address", 404)
        db.execute("INSERT INTO relay_packages (address, direction, peer, kind, body, created_at, sent_at) VALUES (?,?,?,?,?,?,?)",
                   (address, "out", str(target), kind, bytes(body), now, now))
        made = db.execute("INSERT INTO relay_packages (address, direction, peer, kind, body, created_at) VALUES (?,?,?,?,?,?)",
                          (target, "in", str(address), kind, bytes(body), now))
        _trim(db, target, config)
        return {"id": made.lastrowid, "from": address, "to": to, "delivered": True}
    host, _, port = to.rpartition(":")
    if not host or not port.isdigit() or not 0 < int(port) < 65536:
        raise RelayError("to must be host:port, or an address's number")
    made = db.execute("INSERT INTO relay_packages (address, direction, peer, kind, body, created_at) VALUES (?,?,?,?,?,?)",
                      (address, "out", f"{host.strip('[]')}:{int(port)}", "datagram", bytes(body), now))
    return {"id": made.lastrowid, "from": address, "to": to, "delivered": False}


def receive(db, address, *, user_id, after=0, limit=100) -> dict:
    ensure(db)
    _owned(db, address, user_id)
    if isinstance(after, bool) or not isinstance(after, int) or isinstance(limit, bool) or not isinstance(limit, int):
        raise RelayError("after and limit must be whole numbers")
    rows = db.execute("SELECT id, peer, kind, body, created_at FROM relay_packages WHERE address = ? AND direction = 'in' AND id > ? "
                      "ORDER BY id LIMIT ?", (address, after, max(1, min(limit, 500)))).fetchall()
    packages = []
    for row_id, peer, kind, data, at in rows:    # a conglomerate record is read back as the datagrams it holds
        for sender, payload in (unpack_record(data) if kind == CONGLOMERATE else [(peer, data)]):
            packages.append({"id": row_id, "from": sender, "kind": "datagram" if kind == CONGLOMERATE else kind,
                             "body_base64": base64.b64encode(payload).decode("ascii"), "bytes": len(payload), "at": at})
    return {"address": address, "after": rows[-1][0] if rows else after, "records": len(rows), "packages": packages}


FRAME = struct.Struct("<BH4sHI")       # op, port, ipv4 (network order), peer port, payload length
OPEN, CLOSE, SEND, NOTE, READY, OPENED, BATCH = 1, 2, 3, 5, 6, 7, 8
ENTRY = struct.Struct("<4sHH")          # inside a conglomerate record: ipv4, peer port, length; the payload follows
CONGLOMERATE = "datagrams"              # the kind of a row that holds one conglomerate record


def unpack_record(record: bytes) -> list:
    """The datagrams of one conglomerate record, in arrival order: (peer "ip:port", payload)."""
    count = struct.unpack_from("<I", record, 0)[0]
    out, at = [], 4
    for _ in range(count):
        ip, peer_port, length = ENTRY.unpack_from(record, at)
        at += ENTRY.size
        out.append((f"{socket.inet_ntoa(ip)}:{peer_port}", record[at:at + length]))
        at += length
    return out


class Relay:
    """The service side of the UDP router. The native ``udp_router`` (nodus tools/udp_router) holds every UDP socket;
    this class is one of the server's services and reaches it over a local connection. It says which addresses are
    open and what to send, and stores what the router hands back. The router is greedy: everything that arrived for
    one destination comes as ONE conglomerate record, and that record is one row."""

    def __init__(self, db_path, config, log=print):
        self.db_path, self.config, self.log = str(db_path), config, log
        self.sockets = set()                     # the addresses the router has confirmed open for this service
        self.failed = set()
        self.frames = queue.Queue()
        host, _, port = config["router"].rpartition(":")
        self.router = socket.create_connection((host, int(port)), timeout=10)        # a local connection, never a listener
        self.router.settimeout(None)
        threading.Thread(target=self._read, args=(self.router.makefile("rb"),), daemon=True).start()

    def _read(self, stream) -> None:
        try:
            while True:
                head = stream.read(FRAME.size)
                if len(head) < FRAME.size:
                    break
                op, port, ip, peer_port, length = FRAME.unpack(head)
                self.frames.put((op, port, stream.read(length) if length else b""))
        except OSError:
            pass
        self.frames.put(None)

    def _tell(self, op, port, ip=b"\0\0\0\0", peer_port=0, payload=b"") -> None:
        self.router.sendall(FRAME.pack(op, port, ip, peer_port, len(payload)) + payload)

    def _db(self):
        db = sqlite3.connect(self.db_path, timeout=10, isolation_level=None)
        ensure(db)
        return db

    def _take(self, db, wait) -> int:
        """Handle what the router has reported, waiting up to ``wait`` for the first frame. Returns datagrams taken in."""
        taken, timeout = 0, wait
        while True:
            try:
                frame = self.frames.get(timeout=timeout) if timeout else self.frames.get_nowait()
            except queue.Empty:
                return taken
            timeout = 0
            if frame is None:
                raise RelayError("the UDP router closed the connection", 503)
            op, port, payload = frame
            if op == BATCH:                      # one row for everything that arrived for this destination
                db.execute("INSERT INTO relay_packages (address, direction, peer, kind, body, created_at) VALUES (?,?,?,?,?,?)",
                           (port, "in", "*", CONGLOMERATE, payload, time.time()))
                _trim(db, port, self.config)
                taken += struct.unpack_from("<I", payload, 0)[0]
            elif op == OPENED:
                self.sockets.add(port)
            elif op == NOTE:
                self.log(f"[relay] address {port}: {payload.decode('utf-8', 'replace')}")
                if port not in self.sockets:
                    self.failed.add(port)

    def _sync(self, db) -> int:
        wanted = {r[0] for r in db.execute("SELECT address FROM relay_addresses WHERE closed_at IS NULL")}
        for address in self.sockets - wanted:
            self._tell(CLOSE, address)
            self.sockets.discard(address)
        self.failed &= wanted
        asked = wanted - self.sockets - self.failed
        for address in asked:
            self._tell(OPEN, address)
        taken, deadline = 0, time.time() + 2.0
        while asked - self.sockets - self.failed and time.time() < deadline:     # an address is used only once it listens
            taken += self._take(db, 0.05)
        return taken

    def step(self, wait=0.05) -> tuple:
        """(datagrams taken in, datagrams sent out) in one round."""
        db = self._db()
        try:
            taken = self._sync(db)
            sent = 0
            for row in db.execute("SELECT id, address, peer, body FROM relay_packages WHERE direction = 'out' AND sent_at IS NULL "
                                  "ORDER BY id LIMIT 256").fetchall():
                host, _, port = row[2].rpartition(":")
                if row[1] in self.sockets:
                    try:
                        target = socket.getaddrinfo(host, int(port), socket.AF_INET, socket.SOCK_DGRAM)[0][4][0]
                        self._tell(SEND, row[1], socket.inet_aton(target), int(port), row[3])
                        sent += 1
                    except OSError as error:
                        self.log(f"[relay] {row[1]} -> {row[2]}: {error}")
                db.execute("UPDATE relay_packages SET sent_at = ? WHERE id = ?", (time.time(), row[0]))     # tried once, never again
            db.execute("DELETE FROM relay_packages WHERE direction = 'out' AND sent_at IS NOT NULL AND sent_at < ?", (time.time() - 3600.0,))
            taken += self._take(db, wait)
            return taken, sent
        finally:
            db.close()

    def close(self) -> None:
        try:
            self.router.close()                  # the router gives up this service's ports when the connection closes
        except OSError:
            pass
        self.sockets.clear()


def http_route(self, path, method, *, accounts_db, config):
    """The relay over REST, for any signed-in account.

        GET  /api/relay/addresses
        POST /api/relay/open      {}
        POST /api/relay/close     {address}
        POST /api/relay/send      {address, to, text | body_base64, kind?}
        GET  /api/relay/receive?address=N&after=ID&limit=N
    """
    from urllib.parse import parse_qs
    self._set_cookie = []
    if method == "POST":
        self.close_connection = True
    principal = self._principal()
    if principal.user_id is None:
        return self._json({"error": "sign in first"}, 401)
    db = sqlite3.connect(accounts_db, timeout=10, isolation_level=None)
    try:
        if method == "GET":
            if path == "/api/relay/addresses":
                return self._json({"addresses": addresses(db, principal.user_id), "host": config["host"],
                                   "ports": [config["first_port"], config["last_port"]]})
            if path != "/api/relay/receive":
                return self._json({"error": "not found"}, 404)
            query = parse_qs(self.path.partition("?")[2])
            numbers = {}
            for key, default in (("address", None), ("after", 0), ("limit", 100)):
                raw = (query.get(key) or [default])[0]
                if raw is None or not str(raw).lstrip("-").isdigit():
                    raise RelayError(f"{key} must be a whole number")
                numbers[key] = int(raw)
            return self._json(receive(db, numbers["address"], user_id=principal.user_id, after=numbers["after"], limit=numbers["limit"]))
        if not (self.headers.get("Content-Type") or "").lower().startswith("application/json"):
            return self._json({"error": "JSON only"}, 415)
        body = self._body_json(config["max_body"] * 2 + 4096)
        if body is None:
            return
        if not isinstance(body, dict):
            raise RelayError("a JSON object is required")
        if path == "/api/relay/open":
            return self._json(open_address(db, user_id=principal.user_id, username=principal.username, config=config))
        if path == "/api/relay/close":
            return self._json(close_address(db, body.get("address"), user_id=principal.user_id))
        if path == "/api/relay/send":
            if ("text" in body) == ("body_base64" in body):
                raise RelayError("give the body as text, or as body_base64, not both")
            try:
                data = body["text"].encode("utf-8") if "text" in body else base64.b64decode(body["body_base64"], validate=True)
            except (AttributeError, ValueError, TypeError):
                raise RelayError("the body is not text or valid base64") from None
            return self._json(send(db, body.get("address"), user_id=principal.user_id, to=body.get("to"), body=data,
                                   kind=body.get("kind", "datagram"), config=config))
        return self._json({"error": "not found"}, 404)
    except RelayError as error:
        return self._json({"error": error.message}, error.status)
    finally:
        db.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", required=True, help="the accounts database")
    parser.add_argument("--config", default=None)
    args, _rest = parser.parse_known_args(argv)          # the supervisor may append arguments meant for other instances
    config = load_config(args.config)
    relay = None
    while relay is None:                                 # the router may still be starting
        try:
            relay = Relay(args.db, config)
        except OSError as error:
            print(f"[relay] the UDP router at {config['router']} is not reachable yet: {error}", flush=True)
            time.sleep(2.0)
    print(f"[relay] addresses {config['first_port']}-{config['last_port']}; UDP sockets held by the router at {config['router']}", flush=True)
    try:
        while True:
            relay.step()
    except KeyboardInterrupt:
        return 0
    finally:
        relay.close()


if __name__ == "__main__":
    sys.exit(main())
