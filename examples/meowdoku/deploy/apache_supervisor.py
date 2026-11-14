"""Apache-owned lifecycle for the Meowdoku and Nodus instances (stdlib + the meowdoku modules only).

Apache starts this as the 443 vhost's piped CustomLog program (``CustomLog "|python apache_supervisor.py" combined``): httpd
starts it at startup, restarts it if it exits, and closes its stdin at shutdown.  So this process
  * ALWAYS keeps reading stdin (the access-log lines) into ``<state>/logs/apache-access.log`` -- if it stopped reading, Apache would block;
  * on start reads ``instances.json`` (beside this file), ensures each instance's service account (role ``service``), checks out
    the instance's declared lowering leases for ``account:<service_user_id>``, launches the child, and supervises it
    (1 s poll; crash -> exponential backoff, at most 5 restarts per 10 min, then ``failed``; leases renewed before expiry;
    a scheduled lifecycle exit (75) relaunches at once);
  * on stdin EOF / SIGTERM / CTRL_BREAK: graceful stop (Meowdoku: the session-preserving lifecycle restart request, exit-only here
    because this process does not relaunch it; Nodus: CTRL_BREAK), wait 20 s, terminate, check every lease in, mark ``stopped``, exit 0.
NOT GATED: unlike deploy/*.ps1 (operator gate, authz.py script-auth; see LAUNCH_2026-11-10.md "Who may run the scripts"), this process is
launched by httpd under Apache's identity, not by a user, so it carries no -User/-PasswordFile check.
Only one supervisor manages instances (an OS lock on ``<state>/supervisor.lock``); a second one (httpd on Windows can open the pipe
from more than one process) stays a log sink.  A port that already has a listener is not launched over (state ``external``).

  python apache_supervisor.py --status                 # service_instances as JSON
  python apache_supervisor.py --once --dry-run         # print what would be done, change nothing
Tables in accounts.db: service_instances, instance_leases, account_instances (users under an instance; DDL owned by authz.py, written by server.py --instance).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import secrets
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [p for p in (str(ROOT), str(ROOT / "examples")) if p not in sys.path]

from meowdoku import accounts as accounts_mod        # noqa: E402  (its own scrypt hashing; not duplicated here)
from meowdoku import authz as authz_mod              # noqa: E402
from meowdoku import schema_version as _schema_version   # noqa: E402
from meowdoku import lifecycle as lifecycle_mod      # noqa: E402
from meowdoku import lowering_library as lib         # noqa: E402
from meowdoku import service_jobs                     # noqa: E402  (the server-wide job table a restart waits for)
from meowdoku.state_paths import StatePaths          # noqa: E402

STATE = StatePaths.legacy(ROOT).root                # the old layout: no --state-dir, no "state_dir" in instances.json
LEASE_SECONDS = 86400
RENEW_MARGIN = 3600
STOP_WAIT = 20.0
MAX_RESTARTS, RESTART_WINDOW = 5, 600.0
START_TIMEOUT = {"meowdoku": 300.0, "nodus": 60.0}
EXIT_RESTART = lifecycle_mod.EXIT_RESTART

DDL = """
CREATE TABLE IF NOT EXISTS service_instances (name TEXT PRIMARY KEY, kind TEXT NOT NULL, port INTEGER, route TEXT,
    account_user_id INTEGER NOT NULL, pid INTEGER, state TEXT NOT NULL, started REAL, stopped REAL,
    restarts INTEGER NOT NULL DEFAULT 0, supervisor_pid INTEGER, console_port INTEGER,
    contract_prefix TEXT, contract_port INTEGER);
CREATE TABLE IF NOT EXISTS instance_leases (name TEXT NOT NULL, lease_id INTEGER NOT NULL, product_sha256 TEXT NOT NULL,
    generation INTEGER NOT NULL, PRIMARY KEY(name, lease_id));
"""


@contextmanager
def connect(db_path):
    """accounts.db with users/sessions (accounts.py), account_roles (authz.py) and this module's three tables."""
    db_path = Path(db_path)
    accounts_mod.Accounts(db_path)._conn.close()
    authz_mod._connect(db_path).close()
    db = sqlite3.connect(db_path, timeout=10, check_same_thread=False)
    db.row_factory = sqlite3.Row
    try:
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript(DDL)
        have = {r[1] for r in db.execute("PRAGMA table_info(service_instances)")}
        for column, kind in (("console_port", "INTEGER"), ("contract_prefix", "TEXT"), ("contract_port", "INTEGER")):
            if column not in have:
                db.execute(f"ALTER TABLE service_instances ADD COLUMN {column} {kind}")   # tables made before the column existed
        _schema_version.declare(db, "supervisor", 1)
        with db:
            yield db
    finally:
        db.close()


def ensure_service_user(username, db_path=None) -> int:
    """The service account's user id; created with a random password that is stored nowhere (only its scrypt hash), role 'service'."""
    db_path = Path(db_path) if db_path else StatePaths.legacy_at(STATE).accounts_db
    username = accounts_mod.Accounts._clean_username(username)
    with connect(db_path) as db:
        row = db.execute("SELECT id FROM users WHERE username = ? COLLATE NOCASE", (username,)).fetchone()
        if row is None:
            cur = db.execute("INSERT INTO users (username, password_hash, display_name, created_at) VALUES (?, ?, ?, ?)",
                             (username, accounts_mod._hash_password(secrets.token_urlsafe(32)), username, time.time()))
            uid = cur.lastrowid
        else:
            uid = row["id"]
        db.execute("INSERT OR REPLACE INTO account_roles (user_id, role, granted_by, created_at) VALUES (?, 'service', 'supervisor', ?)",
                   (uid, time.time()))
    return uid


def _read_instances(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SystemExit(f"cannot read {path}: {error}")


def load_instances(path):
    # instances.json: a list, or {"supervisor_python": <documented, read by nothing yet>, "state_dir": <state root>, "instances": [...]}
    data = _read_instances(path)
    if isinstance(data, dict):
        data = data.get("instances", [])
    return data if isinstance(data, list) else []


def instances_system_events(path):
    """[CONTROL] The top-level ``system_events`` of the object form of instances.json: where system events are asked
    for ({"from": <nodus instance>, "prefix": <its task prefix>, "jobs_dir": <that prefix's directory>, "engine_id"})."""
    data = _read_instances(path)
    value = data.get("system_events") if isinstance(data, dict) else None
    return dict(value) if isinstance(value, dict) and value.get("from") and value.get("prefix") and value.get("jobs_dir") else None


def instances_restart_account(path):
    """[CONTROL] The top-level ``restart_account`` / ``restart_account_display_name`` of the object form of
    instances.json: the service account that authorizes a waiting restart or shutdown once the job table shows no
    running job. (None, None) when it is not declared: requests still wait, but nobody signs the authorization."""
    data = _read_instances(path)
    if not isinstance(data, dict) or not isinstance(data.get("restart_account"), str) or not data["restart_account"]:
        return None, None
    return data["restart_account"], str(data.get("restart_account_display_name") or data["restart_account"])


def instances_state_dir(path):
    """The top-level ``state_dir`` of the object form of instances.json (a path, absolute or relative to the repo root), or None."""
    data = _read_instances(path)
    value = data.get("state_dir") if isinstance(data, dict) else None
    return _abs(value) if value else None


def listening(port, host="127.0.0.1") -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=0.5):
            return True
    except OSError:
        return False


_JOB = None      # [JOB] the kill-on-close job this supervisor put itself in; held for the life of the process


def tie_children_to_this_process() -> str:
    """Put this process in a Windows job object that ends every process in it when its last handle closes.

    The supervisor holds the only handle, so when the supervisor ends, however it ends, every instance it started
    ends with it. Children join the job when they are created, so none can start outside it. Without this, a service
    stop ended the supervisor without its shutdown and left the instances running on their ports (2026-11-13: the next
    supervisor found 8766, 47611-47617 and 47699 taken and could start nothing). Returns a sentence for the log."""
    global _JOB
    if os.name != "nt":
        return "children follow the supervisor by process group only (not Windows)"
    if _JOB is not None:
        return "children are already tied to this supervisor"
    import ctypes
    from ctypes import wintypes

    class BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

    class IO(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                                                         "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class EXTENDED(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IO), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
    kernel32.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
    kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return f"children are NOT tied to the supervisor: CreateJobObject failed ({ctypes.get_last_error()})"
    info = EXTENDED()
    info.BasicLimitInformation.LimitFlags = 0x2000                # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):   # JobObjectExtendedLimitInformation
        error = ctypes.get_last_error()
        kernel32.CloseHandle(job)
        return f"children are NOT tied to the supervisor: SetInformationJobObject failed ({error})"
    if not kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess()):
        error = ctypes.get_last_error()
        kernel32.CloseHandle(job)
        return f"children are NOT tied to the supervisor: AssignProcessToJobObject failed ({error})"
    _JOB = job
    return "children are tied to this supervisor: a kill-on-close job ends them when it ends"


def console_secret_path(state_dir, name) -> Path:
    """Per-run read-only-console token file: <run>/console/<instance>.console.secret. Not under secrets/operator_credentials:
    the supervisor runs as the service account, which writes this file at every launch and must not hold write access to
    the operators' credentials (2026-11-13: that write was refused and the Nodus host did not start)."""
    return StatePaths.coerce(state_dir).console_dir / f"{name}.console.secret"


def _abs(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


class Instance:
    def __init__(self, spec, sup):
        self.spec, self.sup = spec, sup
        self.name, self.kind = spec["name"], spec["kind"]
        self.port, self.route = spec.get("port"), spec.get("route") or ""
        self.proc = None
        self.state = "new"
        self.uid = None
        self.leases = []          # [(want, lowering_library.Lease)]
        self.starts = []          # monotonic times of crash restarts, for the 10-minute window
        self.backoff, self.next_start, self.deadline = 1.0, 0.0, 0.0
        self.logfile = None
        self.skip_reason = None
        self.console_port = None  # the port the host reported on its first JSON line
        self.contract_prefix = None   # first entry of that line's contract_ports: its prefix and listen_port
        self.contract_port = None
        # [ROUTE] per-instance "state_dir" (a path, absolute or relative to the repo root): that child's --state-dir and nothing else
        self.paths = StatePaths(_abs(spec["state_dir"])) if spec.get("state_dir") else sup.paths
        self.federation_args = []     # [FEDERATE] --federation-port/--federation-canvas, set by begin() from spec["federation_from"]

    def console_requested(self) -> bool:
        n = self.spec.get("nodus") or {}
        return self.kind == "nodus" and (n.get("console_port", self.spec.get("console_port")) is not None)

    def console_token_path(self) -> Path:
        return console_secret_path(self.sup.paths, self.name)

    def write_console_token(self):
        """A fresh token for this run (overwritten on every launch), in the restricted console dir: the account the stack runs
        as has full control, Administrators may read (an elevated Open-Console.ps1)."""
        path = self.console_token_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.sup.secrets_restricted:
            self.sup.secrets_restricted = True
            sys.path[:0] = [p for p in (str(HERE),) if p not in sys.path]
            import operators as operators_mod            # noqa: E402  (the same icacls restriction operators.py applies)
            self.sup.log(operators_mod._restrict(path.parent, readers=("*S-1-5-32-544",)))
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(secrets.token_urlsafe(32))

    def remove_console_token(self):
        if self.console_requested():
            try:
                self.console_token_path().unlink()
            except OSError:
                pass
        self.console_port = None
        self.contract_prefix = self.contract_port = None

    # ---- command
    def command(self):
        py = self.spec.get("python") or sys.executable   # optional per-instance interpreter (the private service python)
        if self.kind == "meowdoku":
            argv = [py, "examples/meowdoku/server.py", "--port", str(self.port), "--bind", self.spec.get("bind", "127.0.0.1"),
                    *self.spec.get("args", [])]
            if not self.paths.is_legacy:                # an explicit state root is handed to the child; the old layout is its own default
                argv += ["--state-dir", str(self.paths.root)]
            if self.paths is not self.sup.paths:        # [ROUTE] a per-instance root shares the supervisor's accounts.db (users, roles, sessions, preferences)
                argv += ["--accounts-db", str(self.sup.db_path)]
            if self.route:
                argv += ["--route", self.route]
            argv += self.sup.route_args()               # [ROUTE] --routes / --stable-route / --dev-route from the enabled instances
            return argv + ["--instance", self.name, *self.federation_args]
        if self.kind == "nodus":
            n = self.spec["nodus"]
            argv = [py, n["tool"], "--canvas", str(_abs(n["canvas"])), "--rate", str(n.get("rate", 60))]
            for a in n.get("acquire", []):
                argv += ["--acquire", a]
            if self.console_requested():
                cp = n.get("console_port", self.spec.get("console_port"))
                argv += ["--console-port", str(int(cp)), "--console-token-file", str(self.console_token_path())]
            return argv + list(self.spec.get("args", []))
        # "python -c" style stand-ins (tests): kind 'command' is not shipped; args is the full argv
        return list(self.spec.get("args", [])) + self.federation_args

    def env(self):
        env = dict(os.environ, PYTHONUTF8="1")
        n = self.spec.get("nodus") or {}
        if n.get("dll"):
            env["NODUS_HEADLESS_TEST_DLL"] = n["dll"]
        if n.get("path_prepend"):
            env["PATH"] = os.pathsep.join([str(Path(p)) for p in n["path_prepend"]] + [env.get("PATH", "")])
        # Optional "env" of the instance: variables its child is started with (e.g. the Nodus exec rows' declarations).
        for key, value in (self.spec.get("env") or {}).items():
            env[str(key)] = str(value)
        return env

    def check(self):
        """A reason this instance cannot run, or None."""
        if self.spec.get("enabled", True) is False:       # [ROUTE] rollback switch: "enabled": false
            return "disabled (enabled: false)"
        if self.kind not in ("meowdoku", "nodus", "command"):
            return f"unknown kind {self.kind!r}"
        if self.kind == "nodus":
            n = self.spec.get("nodus") or {}
            if not n.get("canvas") or (not n.get("create_canvas", False) and not _abs(n["canvas"]).exists()):
                return f"canvas not found: {n.get('canvas')}"
            if not Path(n.get("tool", "")).exists():
                return f"run_canvas tool not found: {n.get('tool')}"
        return None

    # ---- leases
    def take_leases(self):
        key = f"account:{self.uid}"
        for want in self.spec.get("leases", []):
            lease = lib.checkout(self.sup.db_path, tuple(want["purpose"]), want.get("capabilities", []), key, LEASE_SECONDS,
                                 prefer=tuple(want.get("prefer", ())))
            if lease is None:
                self.sup.log(f"{self.name}: no lowering for purpose {want['purpose']} with {want.get('capabilities')}")
                continue
            self.leases.append((want, lease))
            with connect(self.sup.db_path) as db:
                db.execute("INSERT OR REPLACE INTO instance_leases VALUES (?, ?, ?, ?)",
                           (self.name, lease.lease_id, lease.product_sha256, lease.generation))
            self.sup.log(f"{self.name}: leased {lease.backend} {lease.product_sha256[:12]} gen {lease.generation} (lease {lease.lease_id})")

    def renew_leases(self):
        now, key = time.time(), f"account:{self.uid}"
        for i, (want, lease) in enumerate(self.leases):
            if lease.deadline - now > RENEW_MARGIN:
                continue
            new = lib.checkout(self.sup.db_path, tuple(want["purpose"]), want.get("capabilities", []), key, LEASE_SECONDS,
                               prefer=tuple(want.get("prefer", ())))
            if new is None:
                continue
            lib.checkin(self.sup.db_path, lease.lease_id, key, lease.generation)
            with connect(self.sup.db_path) as db:
                db.execute("DELETE FROM instance_leases WHERE name = ? AND lease_id = ?", (self.name, lease.lease_id))
                db.execute("INSERT OR REPLACE INTO instance_leases VALUES (?, ?, ?, ?)",
                           (self.name, new.lease_id, new.product_sha256, new.generation))
            self.leases[i] = (want, new)
            self.sup.log(f"{self.name}: renewed lease {lease.lease_id} -> {new.lease_id}")

    def return_leases(self):
        key = f"account:{self.uid}"
        for _, lease in self.leases:
            lib.checkin(self.sup.db_path, lease.lease_id, key, lease.generation)
        self.leases = []

    # ---- process
    def launch(self):
        self.sup.log_dir.mkdir(parents=True, exist_ok=True)
        if self.console_requested():
            self.write_console_token()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.logfile = self.sup.log_dir / f"{self.name}-{stamp}.log"
        out = open(self.logfile, "ab")
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        try:
            self.proc = subprocess.Popen(self.command(), cwd=str(ROOT), env=self.env(), stdin=subprocess.DEVNULL, stdout=out,
                                         stderr=subprocess.STDOUT, creationflags=flags)
        finally:
            out.close()
        self.deadline = time.monotonic() + START_TIMEOUT.get(self.kind, 30.0)
        self.state = "starting"
        self.sup.log(f"{self.name}: started pid {self.proc.pid} -> {self.logfile.name}")
        self.record(started=time.time(), stopped=None)

    def ready(self):
        if self.kind == "meowdoku":
            return listening(self.port)
        try:
            for line in self.logfile.read_text(encoding="utf-8", errors="replace").splitlines():
                if self.kind == "nodus" and line.startswith("{") and "contract_ports" in line:
                    try:
                        first = json.loads(line)
                    except ValueError:
                        first = {}
                    if self.console_requested():
                        self.console_port = first.get("console_port")
                    ports = first.get("contract_ports") or []
                    self.contract_ports = {p.get("prefix"): int(p["listen_port"]) for p in ports
                                           if isinstance(p, dict) and p.get("prefix") and p.get("listen_port")}
                    if ports and isinstance(ports[0], dict) and ports[0].get("listen_port"):
                        self.contract_prefix, self.contract_port = ports[0].get("prefix"), int(ports[0]["listen_port"])
                    return True
                if self.kind == "command" and line.strip():
                    return True
        except OSError:
            pass
        return False

    def record(self, **fields):
        alive = bool(self.proc and self.proc.poll() is None)
        cols = {"state": self.state, "pid": self.proc.pid if alive else None,
                "restarts": len(self.starts), "supervisor_pid": os.getpid(),
                "console_port": self.console_port if alive or self.state == "starting" else None,
                "contract_prefix": self.contract_prefix if alive or self.state == "starting" else None,
                "contract_port": self.contract_port if alive or self.state == "starting" else None, **fields}
        sets = ", ".join(f"{k} = ?" for k in cols)
        with connect(self.sup.db_path) as db:
            db.execute("INSERT OR IGNORE INTO service_instances (name, kind, port, route, account_user_id, state) VALUES (?,?,?,?,?,?)",
                       (self.name, self.kind, self.port, self.route, self.uid, "new"))
            db.execute(f"UPDATE service_instances SET kind=?, port=?, route=?, account_user_id=?, {sets} WHERE name = ?",
                       (self.kind, self.port, self.route, self.uid, *cols.values(), self.name))

    def begin(self):
        self.skip_reason = self.check()
        if self.skip_reason:
            self.state = "skipped"
            self.sup.log(f"{self.name}: skipped: {self.skip_reason}")
            return
        if self.spec.get("state_dir"):
            self.paths.ensure(restrict=False)               # [ROUTE] the instance's own root, created on a real run only (never plan/status)
        self.uid = ensure_service_user(self.spec["account"], self.sup.db_path)
        with connect(self.sup.db_path) as db:
            db.execute("INSERT OR REPLACE INTO account_instances VALUES (?, ?)", (self.uid, self.name))
        if self.kind == "meowdoku" and listening(self.port):
            self.state = "external"
            self.sup.log(f"{self.name}: port {self.port} already has a listener; not launching over it (state external)")
            self.record(started=None)
            return
        self.federation_args = self.resolve_federation()
        self.take_leases()
        self.launch()

    def resolve_federation(self):
        """[FEDERATE] ``federation_from: <nodus instance name>`` (meowdoku, or the command stand-in): wait for that host's first JSON line,
        then hand the child its contract port and canvas.  Not running / no contract port / unknown name: no flags, and the reason is logged."""
        source_name = self.spec.get("federation_from")
        if not source_name or self.kind not in ("meowdoku", "command"):
            return []
        source = next((i for i in self.sup.instances if i.name == source_name and i.kind == "nodus"), None)
        if source is None:
            self.sup.log(f"{self.name}: federation_from {source_name!r}: no such nodus instance; launching without federation flags")
            return []
        end = time.monotonic() + START_TIMEOUT["nodus"]
        while source.state == "starting" and time.monotonic() < end:
            source.poll()
            time.sleep(0.2)
        if source.state != "running":
            self.sup.log(f"{self.name}: federation_from {source_name!r}: instance is {source.state}, not running; launching without federation flags")
            return []
        if not source.contract_port:
            self.sup.log(f"{self.name}: federation_from {source_name!r}: the host reported no contract port; launching without federation flags")
            return []
        canvas = str(_abs(source.spec["nodus"]["canvas"]))
        self.sup.log(f"{self.name}: federating {source_name!r} contract port {source.contract_port} ({source.contract_prefix}) canvas {canvas}")
        flags = ["--federation-port", str(source.contract_port), "--federation-canvas", canvas]
        studio = self.spec.get("studio_prefix")     # [SHOW] the declared Studio prefix on the same host, when this instance names one
        if studio:
            port = getattr(source, "contract_ports", {}).get(studio)
            if port:
                self.sup.log(f"{self.name}: Studio pictures use {source_name!r} contract port {port} ({studio})")
                flags += ["--studio-nodus-port", str(port), "--studio-nodus-prefix", studio]
            else:
                self.sup.log(f"{self.name}: studio_prefix {studio!r}: the host declares no such contract port; pictures stay off")
        cron = self.spec.get("cron_prefix")         # [CRON] the declared cron prefix whose tasks this instance performs
        if cron:
            port = getattr(source, "contract_ports", {}).get(cron)
            if port:
                self.sup.log(f"{self.name}: scheduled tasks come from {source_name!r} contract port {port} ({cron})")
                flags += ["--cron-port", str(port)]
            else:
                self.sup.log(f"{self.name}: cron_prefix {cron!r}: the host declares no such contract port; no scheduled tasks")
        userlist = self.spec.get("userlist_prefix")  # [CONTROL] the user list as a table: this instance performs its searches
        if userlist:
            port = getattr(source, "contract_ports", {}).get(userlist)
            if port:
                self.sup.log(f"{self.name}: the user list table is {source_name!r} contract port {port} ({userlist})")
                flags += ["--userlist-port", str(port)]
            else:
                self.sup.log(f"{self.name}: userlist_prefix {userlist!r}: the host declares no such contract port; searched directly")
        mail = self.spec.get("mail_prefix")          # [MAIL] the email manager table: its host, or an engine that queues mail
        if mail:
            port = getattr(source, "contract_ports", {}).get(mail)
            if port:
                self.sup.log(f"{self.name}: the mail table is {source_name!r} contract port {port} ({mail})")
                flags += ["--mail-port", str(port)]
            else:
                self.sup.log(f"{self.name}: mail_prefix {mail!r}: the host declares no such contract port; mail waits for the sweep")
        return flags

    def poll(self):
        if self.state in ("skipped", "external", "failed", "stopped", "new"):
            return
        now = time.monotonic()
        if self.state == "backoff":
            if now >= self.next_start:
                self.launch()
            return
        code = self.proc.poll()
        if code is not None:
            if code == EXIT_RESTART and self.kind == "meowdoku":
                self.sup.log(f"{self.name}: scheduled exit ({code}); relaunching")
                self.launch()
                return
            self.remove_console_token()
            self.starts = [t for t in self.starts if now - t < RESTART_WINDOW]
            if len(self.starts) >= MAX_RESTARTS:
                self.state = "failed"
                self.sup.log(f"{self.name}: exit {code}; {MAX_RESTARTS} restarts in {RESTART_WINDOW:.0f}s: failed")
                self.record(stopped=time.time())
                self.return_leases()
                return
            self.starts.append(now)
            self.state, self.next_start = "backoff", now + self.backoff
            self.sup.log(f"{self.name}: exit {code}; restart {len(self.starts)} in {self.backoff:.0f}s")
            self.backoff = min(self.backoff * 2, 60.0)
            self.record()
            return
        if self.state == "starting":
            if self.ready():
                self.state, self.backoff = "running", 1.0
                self.sup.log(f"{self.name}: running pid {self.proc.pid}")
                self.record()
            elif now > self.deadline:
                self.sup.log(f"{self.name}: not ready after {START_TIMEOUT.get(self.kind, 30.0):.0f}s (still supervised)")
                self.deadline = now + START_TIMEOUT.get(self.kind, 30.0)
        elif self.leases:
            self.renew_leases()

    # ---- stop
    def request_stop(self):
        if self.proc is None or self.proc.poll() is not None:
            return
        try:
            if self.kind == "meowdoku":
                self.post_lifecycle()
            elif os.name == "nt":
                os.kill(self.proc.pid, signal.CTRL_BREAK_EVENT)
            else:
                self.proc.terminate()
        except Exception as error:                      # the wait + terminate in finish_stop is the fallback
            self.sup.log(f"{self.name}: graceful stop request failed: {error}")

    def post_lifecycle(self):
        """The Stop-Meowdoku.ps1 request: POST {op:restart,in:N} with the admin token; the server saves sessions and exits 75."""
        self.lifecycle_op({"op": "restart", "in": 1})

    def lifecycle_op(self, body):
        """POST one op to this server's /api/admin/lifecycle with its admin token (loopback); its JSON answer."""
        import ssl
        import urllib.request
        paths = lifecycle_mod.state_paths(self.paths, int(self.port))
        token = paths["token"].read_text(encoding="utf-8").strip()
        tls = "--tls-cert" in self.spec.get("args", [])
        ctx = None
        if tls:
            ca = self.sup.paths.tls_dir / "ca.crt"
            ctx = ssl.create_default_context(cafile=str(ca)) if ca.exists() else ssl.create_default_context()
            ctx.check_hostname = False                  # the cert names the LAN address, not 127.0.0.1; trust rests on the CA
            if not ca.exists():
                ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(f"{'https' if tls else 'http'}://127.0.0.1:{self.port}/api/admin/lifecycle",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "X-Meowdoku-Admin": token})
        with urllib.request.urlopen(req, context=ctx, timeout=10) as r:
            data = r.read()
        return json.loads(data or b"{}")

    # ---- [CONTROL] forced by an admin's request (instance_control.py); the supervisor does not ask the instance's leave
    def force_stop(self, wait=15.0) -> str:
        """Ask the process to stop, wait, then kill it. Leaves the instance ``stopped``; returns what happened."""
        if self.state in ("skipped", "external"):
            return f"not managed here ({self.state})"
        if self.proc is None or self.proc.poll() is not None:
            self.state = "stopped"
            self.record(stopped=time.time())
            return "was not running"
        pid = self.proc.pid
        self.state = "stopping"
        self.request_stop()
        end = time.monotonic() + wait
        while time.monotonic() < end and self.proc.poll() is None:
            time.sleep(0.2)
        killed = self.proc.poll() is None
        if killed:
            self.proc.kill()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                pass
        self.remove_console_token()
        self.return_leases()
        self.state = "stopped"
        self.record(stopped=time.time())
        return f"pid {pid} " + ("did not stop when asked and was killed" if killed else "stopped")

    def force_start(self) -> str:
        if self.state in ("skipped", "external"):
            return f"not managed here ({self.state})"
        if self.proc is not None and self.proc.poll() is None:
            return f"already running (pid {self.proc.pid})"
        self.starts, self.backoff = [], 1.0              # an admin's start is not one of the crash restarts
        self.federation_args = self.resolve_federation()
        self.take_leases()
        self.launch()
        return f"started pid {self.proc.pid}"

    def finish_stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.sup.log(f"{self.name}: did not exit in {STOP_WAIT:.0f}s; terminating")
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.remove_console_token()
        if self.uid is not None and self.state not in ("skipped", "external"):
            self.return_leases()
            self.state = "stopped"
            self.record(stopped=time.time())


class Supervisor:
    def __init__(self, db_path, instances_path, state_dir):
        """state_dir: a StatePaths, or a plain directory (the old meaning: the legacy layout rooted there)."""
        self.paths = StatePaths.coerce(state_dir)
        self.db_path, self.instances_path, self.state_dir = Path(db_path), Path(instances_path), self.paths.run_dir
        self.log_dir = self.paths.logs_dir
        self.stop = threading.Event()
        self.instances = []
        self.system_events = None       # [CONTROL] {"from", "prefix", "jobs_dir", "engine_id"}: where system events are asked for
        self.restart_account_id = None  # [CONTROL] the service account that authorizes a waiting restart (instances.json)
        self.restart_display = "the restart account"
        self.secrets_restricted = False
        self._lock_file = None

    def log(self, text):
        line = time.strftime("%Y-%m-%d %H:%M:%S ") + f"[apache_supervisor] {text}"
        print(line, file=sys.stderr, flush=True)
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            with open(self.log_dir / "apache-supervisor.log", "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def acquire_lock(self) -> bool:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._lock_file = open(self.state_dir / "supervisor.lock", "a+b")       # <state>/run/supervisor.lock in the new layout
        try:
            if os.name == "nt":
                import msvcrt
                self._lock_file.seek(0)
                msvcrt.locking(self._lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def tail_stdin(self):
        """Apache's access-log lines -> apache-access.log.  Returns on EOF (= httpd is shutting down)."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with open(self.log_dir / "apache-access.log", "ab") as out:
            for line in iter(sys.stdin.buffer.readline, b""):
                out.write(line)
                out.flush()
        self.stop.set()

    def route_args(self):
        """[ROUTE] ``--routes a,b --stable-route a --dev-route b`` for every meowdoku child: built from the ENABLED meowdoku instances
        (a disabled dev instance contributes no route, so the page reports no Development server).  Optional per-instance ``"site"``:
        "stable" (default) | "dev"."""
        on = [s for s in load_instances(self.instances_path)
              if s.get("kind") == "meowdoku" and s.get("route") and s.get("enabled", True) is not False]
        out = []
        if on:
            out += ["--routes", ",".join(s["route"] for s in on)]
            for flag, site in (("--stable-route", "stable"), ("--dev-route", "dev")):
                first = next((s["route"] for s in on if (s.get("site") or "stable") == site), None)
                if first:
                    out += [flag, first]
        return out

    def plan(self):
        out = []
        for spec in load_instances(self.instances_path):
            inst = Instance(spec, self)
            reason = inst.check()
            out.append({"name": inst.name, "kind": inst.kind, "account": spec.get("account"), "port": inst.port,
                        "route": inst.route, "console_port": (spec.get("nodus") or {}).get("console_port", spec.get("console_port")),
                        "leases": spec.get("leases", []), "federation_from": spec.get("federation_from"),
                        "command": inst.command() if not reason else None, "skipped": reason,
                        "state_dir": str(inst.paths.root) if spec.get("state_dir") else None})
        return out

    def run(self, manage=True):
        reader = threading.Thread(target=self.tail_stdin, daemon=True)
        reader.start()
        for sig in ("SIGTERM", "SIGBREAK", "SIGINT"):
            if hasattr(signal, sig):
                signal.signal(getattr(signal, sig), lambda *_: self.stop.set())
        if manage:
            self.log(tie_children_to_this_process())       # [JOB] before the first instance starts
            specs = load_instances(self.instances_path)
            self.system_events = instances_system_events(self.instances_path)
            account, display = instances_restart_account(self.instances_path)
            if account:
                try:
                    self.restart_account_id = ensure_service_user(account, self.db_path)
                    self.restart_display = display
                except Exception as error:              # the requests still wait for jobs; only the signature is missing
                    self.log(f"restart account {account!r} could not be made: {error}")
            sources = {s.get("federation_from") for s in specs}
            for spec in sorted(specs, key=lambda s: s.get("name") not in sources):      # a federation source starts before its dependents
                inst = Instance(spec, self)
                self.instances.append(inst)
                try:
                    inst.begin()
                except Exception as error:
                    inst.state = "failed"
                    self.log(f"{inst.name}: start failed: {error!r}")
            while not self.stop.wait(1.0):
                try:
                    self.handle_requests()
                except Exception as error:
                    self.log(f"request handling error: {error!r}")
                for inst in self.instances:
                    try:
                        inst.poll()
                    except Exception as error:
                        self.log(f"{inst.name}: poll error: {error!r}")
            self.shutdown()
        else:
            self.log("another supervisor owns the instances; this process only logs")
            self.stop.wait()
        return 0

    def handle_requests(self):
        """[CONTROL] Perform the pending rows of ``instance_requests``. ``*`` means every instance: sources of a
        federation are started first and stopped last, as at the start and stop of the whole stack."""
        from meowdoku import instance_control
        light = sqlite3.connect(self.db_path, timeout=5, isolation_level=None)   # once a second: a plain connection
        try:
            news = instance_control.to_announce(light)
            todo = instance_control.due(light)
        finally:
            light.close()
        for item in news:
            self.announce(item)
        if not todo:
            return
        sources = {i.spec.get("federation_from") for i in self.instances}
        first_to_start = sorted(self.instances, key=lambda i: i.name not in sources)
        for item in todo:
            targets = [i for i in first_to_start if item["instance"] in (instance_control.ALL, i.name)]
            authorized = None
            if targets and item.get("after_jobs") and item["action"] in ("restart", "shutdown"):
                # The policy: the restart account reads the server-wide job table (service_jobs) and authorizes the
                # request only when no instance it names runs a job; until then the row waits and says why.
                with connect(self.db_path) as db:
                    jobs = service_jobs.running(db, [i.name for i in targets])
                if jobs:
                    why = f"{self.restart_display} waits for running jobs: " + ", ".join(
                        f"{j['instance']} {j['kind']} {str(j['job_id'])[:12]}" for j in jobs)
                    if item.get("detail") != why:
                        self.log(f"request {item['id']} by {item['requested_name']!r}: {item['action']} {item['instance']}: {why}")
                        with connect(self.db_path) as db:
                            instance_control.waiting(db, item["id"], why)
                    continue
                authorized = self.restart_account_id
            if not targets:
                state, detail = "refused", "no such instance under this supervisor"
            else:
                done = []
                try:
                    if item["action"] in ("restart", "shutdown"):
                        for inst in reversed(targets):
                            done.append(f"{inst.name}: {inst.force_stop()}")
                    if item["action"] in ("restart", "start"):
                        for inst in targets:
                            done.append(f"{inst.name}: {inst.force_start()}")
                    if authorized is not None:
                        done.insert(0, f"authorized by {self.restart_display}: no running jobs")
                    state, detail = "done", "; ".join(done)
                except Exception as error:
                    state, detail = "refused", "; ".join(done + [f"failed: {error!r}"])
            self.log(f"request {item['id']} by {item['requested_name']!r}: {item['action']} {item['instance']}: {detail}")
            with connect(self.db_path) as db:
                instance_control.finish(db, item["id"], state, detail, authorized_by=authorized)

    def announce(self, item):
        """[CONTROL] Tell every admin about a counted-down request, or its interruption: one task asked of the Nodus
        task table (nodus tools/task_request.py), performed by the task worker as a broadcast from the system account.
        The request is marked announced whatever happens: an announcement that cannot be made is logged, not retried,
        and never holds up the request itself."""
        from meowdoku import instance_control
        cancelled = item["state"] == "cancelled"
        seconds = max(0, int(round(item["due_at"] - time.time())))
        what = f"{item['action']} of {'every instance' if item['instance'] == instance_control.ALL else item['instance']}"
        if cancelled:
            text = f"The {what} (request {item['id']}) was interrupted: {item['detail']}. Nothing will happen."
        else:
            text = (f"There will be a {what} in {seconds} seconds (request {item['id']}), asked by {item['requested_name']}"
                    + (f": {item['reason']}" if item["reason"] else "") + ". Interrupt it to stop it.")
        outcome = {"outcome": "not configured"}
        events = self.system_events or {}
        source = next((i for i in self.instances if i.name == events.get("from") and i.kind == "nodus"), None)
        port = getattr(source, "contract_ports", {}).get(events.get("prefix")) if source is not None else None
        if port:
            try:
                from meowdoku import studio_product_nodus
                native = studio_product_nodus.nodus_modules()[0]
                spec = importlib.util.spec_from_file_location("_supervisor_task_request",
                                                              studio_product_nodus.DEFAULT_NODUS_TOOLS / "task_request.py")
                task_request = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(task_request)
                outcome = task_request.submit(
                    native, port=port, engine_id=int(events.get("engine_id", 7202)), jobs_dir=events["jobs_dir"],
                    task="system.broadcast", timeout=5.0,
                    fields={"text": text, "audience": "admin", "msg_type": "status" if cancelled else "approval_request",
                            "subject": {"kind": "control", "ref": str(item["id"])}})
            except Exception as error:
                outcome = {"outcome": "failed", "error": repr(error)}
        self.log(f"request {item['id']}: announced ({'interrupted' if cancelled else f'in {seconds} s'}) -> {outcome}")
        with connect(self.db_path) as db:
            instance_control.mark_announced(db, item["id"], 2 if cancelled else 1)

    def shutdown(self):
        self.log("stopping instances")
        for inst in self.instances:
            if inst.state in ("starting", "running", "backoff"):
                inst.state = "stopping"
                inst.request_stop()
        end = time.monotonic() + STOP_WAIT
        while time.monotonic() < end and any(i.proc and i.proc.poll() is None for i in self.instances):
            time.sleep(0.2)
        for inst in self.instances:
            try:
                inst.finish_stop()
            except Exception as error:
                self.log(f"{inst.name}: stop error: {error!r}")
        self.log("stopped")


def status(db_path):
    with connect(db_path) as db:
        return [dict(r) for r in db.execute("SELECT * FROM service_instances ORDER BY name")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default=None, help="default <state>/sql/accounts.db (build/meowdoku/accounts.db without a state root)")
    ap.add_argument("--instances", default=str(HERE / "instances.json"))
    ap.add_argument("--state-dir", default=None,
                    help="the state root (state_paths.py); overrides \"state_dir\" in instances.json; default: the old build/meowdoku layout")
    ap.add_argument("--status", action="store_true", help="print service_instances as JSON and exit")
    ap.add_argument("--once", action="store_true", help="with --dry-run: print the plan and exit")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    chosen = Path(args.state_dir) if args.state_dir else instances_state_dir(args.instances)
    paths = StatePaths(chosen) if chosen else StatePaths.legacy_at(STATE)
    sup = Supervisor(args.db or paths.accounts_db, args.instances, paths)
    args.db = str(sup.db_path)
    if args.status:
        print(json.dumps(status(args.db), indent=1))
        return 0
    if args.once and args.dry_run:
        print(json.dumps({"db": str(sup.db_path), "access_log": str(sup.log_dir / "apache-access.log"), "instances": sup.plan()}, indent=1))
        return 0
    if chosen:
        paths.ensure(restrict=False)            # the directories; the ACLs on secrets/ are the operator's step (operators.py)
    return sup.run(manage=sup.acquire_lock())


if __name__ == "__main__":
    raise SystemExit(main())
