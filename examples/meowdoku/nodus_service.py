"""Read-only web view of a running Nodus canvas host (server.py [NODUS], web/nodus.html).

The host (nodus/tools/run_canvas.py, run by deploy/apache_supervisor.py as an instance of kind "nodus") keeps a READ-ONLY console on
loopback: ``HELLO nodus-console/1 <token>`` -> ``OK console <name> read-only``; then the verbs DIGEST, GRAPH, TOOLS and ACTIVITY, each answered
``JSON <verb> <n>`` + n bytes of UTF-8 JSON.  A host that predates a verb answers ``REFUSED read-only``: that is NotSupported (HTTP 501).

Where the console is found: the supervisor's own records.  ``service_instances`` (accounts.db) gives state and ``console_port``; the per-run
token file is ``apache_supervisor.console_secret_path(paths, name)`` (imported, not re-derived).  The token is read, used for one HELLO and
dropped: it is never logged, returned, or placed in an exception message.  Nothing here writes to a canvas; there is no action verb.

``as_generic_graph`` is a SECOND PRODUCER of the generic identity graph (a list of nodes, each with ``identity``, ``kind``, ``name`` and
``dependencies: [{"relationship", "target"}]``), the same shape the AbstractUI world model and its other producers emit (the class-mode and
file-mode producers, abstract_ui_div_map.py world_div_map_graph, abstract_ui_filesystem.py, repo_world_scan.py).  It carries structure only:
nothing about pixels or layout.
"""
from __future__ import annotations

import importlib.util
import json
import socket
import sqlite3
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROTOCOL = "nodus-console/1"
HOST = "127.0.0.1"
TIMEOUT = 3.0
MAX_REPLY = 32 * 1024 * 1024
NOTE_MAX = 500
VERBS = ("DIGEST", "GRAPH", "TOOLS", "ACTIVITY")


class NodusError(Exception):
    """A service failure with the HTTP status it maps to; the message never contains the token."""

    def __init__(self, message: str, status: int = 500):
        super().__init__(message)
        self.message, self.status = message, status


class NotSupported(NodusError):
    def __init__(self, what: str = "this verb"):
        super().__init__(f"not supported by this host ({what})", 501)


# ---- console client -------------------------------------------------------------------------------------------------------------
class ConsoleClient:
    """One connection to a host console.  ``with ConsoleClient(host, port, token, timeout) as c: c.json_verb("DIGEST")``."""

    def __init__(self, host: str, port: int, token: str, timeout: float = TIMEOUT):
        self._host, self._port, self._timeout = host, int(port), float(timeout)
        self._token = str(token)
        self._sock = None
        self._file = None
        self.name = None

    def __repr__(self) -> str:
        return f"ConsoleClient({self._host}:{self._port})"

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()

    def open(self) -> None:
        try:
            self._sock = socket.create_connection((self._host, self._port), timeout=self._timeout)
            self._sock.settimeout(self._timeout)
            self._file = self._sock.makefile("rb")
            self._sock.sendall(f"HELLO {PROTOCOL} {self._token}\n".encode("utf-8"))
            line = self._line()
        except (OSError, EOFError):
            self.close()
            raise NodusError("the console is unreachable", 503) from None
        if not line.startswith("OK console"):
            self.close()
            raise NodusError("the console refused the connection", 503)
        parts = line.split(" ", 3)
        self.name = parts[2] if len(parts) > 2 else None
        self._token = ""

    def close(self) -> None:
        for item in (self._file, self._sock):
            try:
                if item is not None:
                    item.close()
            except OSError:
                pass
        self._file = self._sock = None

    def _line(self) -> str:
        raw = self._file.readline(65536)
        if not raw.endswith(b"\n"):
            raise EOFError("console closed")
        return raw[:-1].decode("utf-8", "replace").rstrip("\r")

    def json_verb(self, verb: str):
        if verb not in VERBS:
            raise ValueError("unknown verb")
        try:
            self._sock.sendall(verb.encode("ascii") + b"\n")
            line = self._line()
            if line.startswith("REFUSED"):
                raise NotSupported(verb.lower())
            parts = line.split(" ")
            if len(parts) != 3 or parts[0] != "JSON" or parts[1] != verb.lower() and parts[1] != verb:
                raise NodusError("the console sent an unexpected reply", 502)
            size = int(parts[2])
            if not 0 <= size <= MAX_REPLY:
                raise NodusError("the console reply is too large", 502)
            data = self._file.read(size)
            if len(data) != size:
                raise EOFError("short reply")
            return json.loads(data.decode("utf-8"))
        except (OSError, EOFError):
            raise NodusError("the console is unreachable", 503) from None
        except ValueError:
            raise NodusError("the console sent an unexpected reply", 502) from None


# ---- locating the running instance -----------------------------------------------------------------------------------------------
_SUPERVISOR = None


def _supervisor():
    """deploy/apache_supervisor.py imported by path (its own helpers, not copies)."""
    global _SUPERVISOR
    if _SUPERVISOR is None:
        spec = importlib.util.spec_from_file_location("meowdoku_deploy_apache_supervisor", HERE / "deploy" / "apache_supervisor.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _SUPERVISOR = module
    return _SUPERVISOR


def _rows(db):
    """The nodus rows of service_instances (SELECT only); a database without the table has none."""
    old = getattr(db, "row_factory", None)
    db.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in db.execute("SELECT name, kind, state, console_port FROM service_instances WHERE kind = 'nodus' ORDER BY name")]
    except sqlite3.Error:
        return []
    finally:
        db.row_factory = old


def instances(db, paths) -> list:
    """[{"name", "state", "console": bool}] for the nodus instances; the token and its path are never included."""
    out = []
    for row in _rows(db):
        out.append({"name": row["name"], "state": row["state"], "console": _console_ready(row, paths)})
    return out


def _console_ready(row, paths) -> bool:
    return row["state"] == "running" and bool(row["console_port"]) and _supervisor().console_secret_path(paths, row["name"]).is_file()


def locate(db, paths, name=None) -> dict:
    """{"name", "host", "port", "token_path"} of the running nodus instance's console, from the supervisor's records."""
    rows = _rows(db)
    if name is None:
        ready = [r for r in rows if _console_ready(r, paths)]
        if len(ready) != 1:
            raise NodusError("name the instance" if ready else "no nodus instance with a console is running", 400 if ready else 503)
        row = ready[0]
    else:
        row = next((r for r in rows if r["name"] == name), None)
        if row is None:
            raise NodusError("no such nodus instance", 404)
        if not _console_ready(row, paths):
            raise NodusError("that instance has no console available", 503)
    return {"name": row["name"], "host": HOST, "port": int(row["console_port"]),
            "token_path": _supervisor().console_secret_path(paths, row["name"])}


def _read_token(path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        raise NodusError("that instance has no console available", 503) from None


# ---- cache and the four verbs ----------------------------------------------------------------------------------------------------
class Cache:
    """Short per-process cache: digest and activity 1 s per instance; graph and tools 5 s per (instance, revision)."""

    def __init__(self, clock=time.monotonic):
        self._clock, self._lock, self._items = clock, threading.Lock(), {}

    def get(self, key, ttl, fetch):
        with self._lock:
            hit = self._items.get(key)
            if hit is not None and self._clock() - hit[0] < ttl:
                return hit[1]
        value = fetch()
        with self._lock:
            self._items[key] = (self._clock(), value)
            if len(self._items) > 256:
                for k in sorted(self._items, key=lambda k: self._items[k][0])[:128]:
                    del self._items[k]
        return value


CACHE = Cache()


def _fetch(db, paths, name, verb, timeout):
    where = locate(db, paths, name)
    with ConsoleClient(where["host"], where["port"], _read_token(where["token_path"]), timeout) as client:
        return where["name"], client.json_verb(verb)


def _result(name, payload):
    return {**payload, "instance": name} if isinstance(payload, dict) else {"value": payload, "instance": name}


def digest(db, paths, name=None, *, cache=CACHE, timeout=TIMEOUT) -> dict:
    where = locate(db, paths, name)
    key = ("digest", where["name"])
    inst, payload = cache.get(key, 1.0, lambda: _fetch(db, paths, where["name"], "DIGEST", timeout))
    return _result(inst, payload)


def _by_revision(verb, db, paths, name, cache, timeout):
    head = digest(db, paths, name, cache=cache, timeout=timeout)
    inst = head["instance"]
    key = (verb, inst, head.get("revision"))
    got, payload = cache.get(key, 5.0, lambda: _fetch(db, paths, inst, verb, timeout))
    return _result(got, payload)


def graph(db, paths, name=None, *, cache=CACHE, timeout=TIMEOUT) -> dict:
    return _by_revision("GRAPH", db, paths, name, cache, timeout)


def tools(db, paths, name=None, *, cache=CACHE, timeout=TIMEOUT) -> dict:
    return _by_revision("TOOLS", db, paths, name, cache, timeout)


def activity(db, paths, name=None, *, cache=CACHE, timeout=TIMEOUT) -> dict:
    where = locate(db, paths, name)
    inst, payload = cache.get(("activity", where["name"]), 1.0, lambda: _fetch(db, paths, where["name"], "ACTIVITY", timeout))
    return _result(inst, payload)


# ---- the generic identity graph ---------------------------------------------------------------------------------------------------
_SCALAR = (str, int, float, bool)


def _scalars(item, skip=()):
    return {k: v for k, v in item.items() if k not in skip and isinstance(v, _SCALAR)} if isinstance(item, dict) else {}


def _idx(item, position):
    value = item.get("index") if isinstance(item, dict) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else position


def as_generic_graph(graph_json, instance) -> list:
    """The GRAPH reply as a list of identity-graph nodes (second producer of the generic graph; see the module docstring).

    Kinds: canvas, module, row, contact, edge, contract-port.  Identities are ``nodus:<instance>/<canvas>`` plus ``/m<i>``, ``/m<i>/r<j>``,
    ``/m<i>/c<j>``, ``/e<i>``, ``/cp<i>`` from the host's own indices, so they are stable while the structure is.  Relationships reuse the
    world model's ``contained-by`` (every non-canvas node has exactly one); an edge adds ``from-contact`` and ``to-contact``; a contract
    port adds ``bound-to`` its edge when it names one that exists.
    """
    g = graph_json if isinstance(graph_json, dict) else {}
    canvas = str(g.get("canvas") or instance)
    root = f"nodus:{instance}/{canvas}"
    nodes = [{"identity": root, "kind": "canvas", "name": canvas, "dependencies": [],
              "attributes": {"instance": str(instance), **({"revision": g["revision"]} if isinstance(g.get("revision"), _SCALAR) else {})}}]
    have = {root}

    def add(identity, kind, name, deps, attributes=None):
        if identity in have:
            return
        have.add(identity)
        node = {"identity": identity, "kind": kind, "name": str(name), "dependencies": deps}
        if attributes:
            node["attributes"] = attributes
        nodes.append(node)

    def contained(target):
        return [{"relationship": "contained-by", "target": target}]

    contact_id = {}
    for mpos, module in enumerate(g.get("modules") or []):
        if not isinstance(module, dict):
            continue
        mi = _idx(module, mpos)
        mid = f"{root}/m{mi}"
        add(mid, "module", module.get("name") or f"module {mi}", contained(root),
            _scalars(module, ("index", "name")))
        for rpos, row in enumerate(module.get("rows") or []):
            ri = _idx(row, rpos)
            add(f"{mid}/r{ri}", "row", (row.get("tool") if isinstance(row, dict) else None) or f"row {ri}", contained(mid),
                _scalars(row, ("index",)))
        for cpos, contact in enumerate(module.get("contacts") or []):
            ci = _idx(contact, cpos)
            cid = f"{mid}/c{ci}"
            name = (contact.get("name") if isinstance(contact, dict) else contact) or f"contact {ci}"
            add(cid, "contact", name, contained(mid), _scalars(contact, ("index", "name")))
            contact_id[(mi, ci)] = cid
    modules_present = {k[0] for k in contact_id} | {_idx(m, p) for p, m in enumerate(g.get("modules") or []) if isinstance(m, dict)}

    def end(spec):
        """The contact node an edge end names; a contact the module did not list is declared here under its module."""
        spec = spec if isinstance(spec, dict) else {}
        mi, ci = spec.get("module"), spec.get("contact")
        if not isinstance(mi, int) or not isinstance(ci, int):
            return None
        if (mi, ci) not in contact_id:
            if mi not in modules_present:
                return None
            cid = f"{root}/m{mi}/c{ci}"
            add(cid, "contact", f"contact {ci}", contained(f"{root}/m{mi}"))
            contact_id[(mi, ci)] = cid
        return contact_id[(mi, ci)]

    edge_id = {}
    for epos, edge in enumerate(g.get("edges") or []):
        if not isinstance(edge, dict):
            continue
        ei = _idx(edge, epos)
        eid = f"{root}/e{ei}"
        deps = contained(root)
        for relationship, key in (("from-contact", "from"), ("to-contact", "to")):
            target = end(edge.get(key))
            if target is not None:
                deps.append({"relationship": relationship, "target": target})
        add(eid, "edge", f"edge {ei}", deps, _scalars(edge, ("index",)))
        edge_id[ei] = eid

    for ppos, port in enumerate(g.get("contract_ports") or []):
        if not isinstance(port, dict):
            continue
        pi = ppos
        deps = contained(root)
        if port.get("edge") in edge_id:
            deps.append({"relationship": "bound-to", "target": edge_id[port["edge"]]})
        add(f"{root}/cp{pi}", "contract-port", port.get("prefix") or f"contract port {pi}", deps, _scalars(port, ("prefix",)))
    return nodes


# ---- sharing --------------------------------------------------------------------------------------------------------------------
def summary_line(d) -> str:
    d = d if isinstance(d, dict) else {}
    sha = str(d.get("sha256") or "")[:12] or "none"
    return (f"nodus {d.get('instance')} rev {d.get('revision')} sha {sha} modules {d.get('modules')} rows {d.get('rows')} "
            f"edges {d.get('edges')} tick {d.get('tick')} rate {d.get('rate')}")


def share(db, paths, accounts_db, me: int, instance, to, note, *, cache=CACHE, timeout=TIMEOUT) -> dict:
    """ONE note to a mutual contact through messaging.send (its rules and limits apply): the digest summary line plus the note."""
    from meowdoku import messaging
    if not isinstance(note, str) or len(note) > NOTE_MAX or "\x00" in note:
        raise NodusError(f"note must be text of at most {NOTE_MAX} characters", 400)
    if not isinstance(to, str) or not to.strip():
        raise NodusError("name the contact", 400)
    d = digest(db, paths, instance, cache=cache, timeout=timeout)
    ref = f"nodus:{d['instance']}@{d.get('revision')}:{str(d.get('sha256') or 'none')[:12]}"
    text = note.strip()
    body = (text + "\n\n" if text else "") + summary_line(d)
    conn = messaging.connect(accounts_db)
    try:
        sent = messaging.send(conn, me, to, body, subject={"kind": "topic", "ref": ref}, msg_type="note")
    except messaging.MessagingError as error:
        raise NodusError(error.message, error.status) from None
    finally:
        conn.close()
    return {"ok": True, "id": sent["id"], "subject": {"kind": "topic", "ref": ref}}
