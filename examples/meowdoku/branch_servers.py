"""[LOAN] Every central checkout of a server repository lends its owner a server (Stage 3 of Research case 8bbe0de7).

The loan record is one row in ``branch_servers`` (accounts DB), beside the checkout's own ``checkouts`` row:
repository, branch, owner, a port from 8769 to 8900 that is the loan's for its whole life, the commit whose pages were
last exported, and a state.

- **open** (``central_checkout`` open): a loan is recorded when the account has fewer than ``PER_ACCOUNT`` and the
  service fewer than ``TOTAL``. Otherwise the checkout still opens and says why it has no server.
- **close**: the loan ends with the checkout.
- **refresh** (the task worker's ``loans.refresh``):
  - exports the pages of a checkout whose commit changed since its last export into ``<loans state root>/<port>/web``;
  - marks the loan ``ready``, and asks for that loan's restart when it was already running;
  - ends loans whose checkout or folder is gone;
  - stops loans nobody has used through the checkout for ``IDLE_HOURS`` (state ``idle``). The next use brings them
    back.
- **The supervisor** turns every ``ready`` loan into an instance named ``loan-<port>``, using the ``"loans"`` block of
  instances.json:
  - for a turing branch, ``code_root`` is the checkout's folder; for a speaktome branch, ``speaktome_root`` is. The
    other repository comes from the dev trees;
  - its ``state_dir`` is ``<state root>/<port>``, and it shares the accounts database (one identity authority);
  - it binds 127.0.0.1 only;
  - it starts the loans that become ready and stops the ones that leave.

Restarting a loan is the existing ``instance_requests``. Its owner may ask for their own loan (``may_restart``);
operators and admins may ask for any instance, as before.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

SERVED = ("turing", "speaktome")
PORT_FIRST, PORT_LAST = 8769, 8900
PER_ACCOUNT, TOTAL = 2, 16
IDLE_HOURS = 12.0
STATES = ("pending", "ready", "idle")
PREFIX = "loan-"

DDL = ("CREATE TABLE IF NOT EXISTS branch_servers (repository TEXT NOT NULL, branch TEXT NOT NULL, user_id INTEGER NOT NULL, "
       "port INTEGER NOT NULL UNIQUE, created_at REAL NOT NULL, exported_head TEXT, state TEXT NOT NULL DEFAULT 'pending', "
       "detail TEXT, PRIMARY KEY (repository, branch))")


class LoanError(Exception):
    pass


def ensure(db) -> None:
    from meowdoku import central_checkout             # the loans are read beside the checkouts they belong to
    for ddl in central_checkout.DDL:
        db.execute(ddl)
    db.execute(DDL)


def name(port) -> str:
    return f"{PREFIX}{int(port)}"


def lend(db, *, repository, branch, user_id, now=None) -> dict:
    """The checkout's server: its existing loan, a new one, or why there is none. Never refuses the checkout itself."""
    ensure(db)
    if repository not in SERVED:
        return {"none": f"a {repository} checkout has no server (only {', '.join(SERVED)} do)"}
    row = db.execute("SELECT port, state FROM branch_servers WHERE repository = ? AND branch = ?", (repository, branch)).fetchone()
    if row is not None:
        return {"instance": name(row[0]), "port": row[0], "state": row[1]}
    if db.execute("SELECT COUNT(*) FROM branch_servers WHERE user_id = ?", (int(user_id),)).fetchone()[0] >= PER_ACCOUNT:
        return {"none": f"you already have {PER_ACCOUNT} branch servers; close a checkout to get one for this branch"}
    taken = {r[0] for r in db.execute("SELECT port FROM branch_servers")}
    if len(taken) >= TOTAL:
        return {"none": f"all {TOTAL} branch servers are lent; one frees when a checkout closes"}
    port = next((p for p in range(PORT_FIRST, PORT_LAST + 1) if p not in taken), None)
    if port is None:
        return {"none": "no free branch server port"}
    db.execute("INSERT INTO branch_servers (repository, branch, user_id, port, created_at, state) VALUES (?,?,?,?,?, 'pending')",
               (repository, branch, int(user_id), port, time.time() if now is None else now))
    return {"instance": name(port), "port": port, "state": "pending",
            "note": "its pages are exported within a few minutes; then it starts"}


def release(db, repository, branch) -> int | None:
    ensure(db)
    row = db.execute("SELECT port FROM branch_servers WHERE repository = ? AND branch = ?", (repository, branch)).fetchone()
    db.execute("DELETE FROM branch_servers WHERE repository = ? AND branch = ?", (repository, branch))
    return row[0] if row else None


def loans(db) -> list:
    """Every loan with its checkout's folder and owner name (None when the checkout row is gone)."""
    ensure(db)
    rows = db.execute(
        "SELECT s.repository, s.branch, s.user_id, s.port, s.created_at, s.exported_head, s.state, s.detail, c.path, c.username "
        "FROM branch_servers s LEFT JOIN checkouts c ON c.repository = s.repository AND c.branch = s.branch ORDER BY s.port").fetchall()
    keys = ("repository", "branch", "user_id", "port", "created_at", "exported_head", "state", "detail", "path", "username")
    return [dict(zip(keys, r), instance=name(r[3])) for r in rows]


def public(db) -> list:
    """What the dev site's server menu shows (GET /api/loans): no folder paths."""
    return [{k: loan[k] for k in ("instance", "port", "repository", "branch", "username", "state", "exported_head", "created_at")}
            for loan in loans(db)]


def owner_of(db, instance) -> int | None:
    if not isinstance(instance, str) or not instance.startswith(PREFIX) or not instance[len(PREFIX):].isdigit():
        return None
    ensure(db)
    row = db.execute("SELECT user_id FROM branch_servers WHERE port = ?", (int(instance[len(PREFIX):]),)).fetchone()
    return row[0] if row else None


def may_restart(db, instance, user_id, action) -> bool:
    """A loan's owner may restart or start their own loan; nothing else is widened."""
    return action in ("restart", "start") and user_id is not None and owner_of(db, instance) == user_id


def config(instances_path) -> dict | None:
    """The ``"loans"`` block of instances.json: {"state_root", "dev_turing", "dev_speaktome", "args", "python"?, "account"?,
    "boards_from"?}. Without it no loan server runs."""
    try:
        data = json.loads(Path(instances_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    block = data.get("loans") if isinstance(data, dict) else None
    return block if isinstance(block, dict) and block.get("state_root") and block.get("dev_turing") else None


def specs(db, cfg) -> list:
    """The supervisor's instance specs for the loans that are ready and whose folder still exists."""
    out = []
    for loan in loans(db):
        if loan["state"] != "ready" or not loan["path"] or not Path(loan["path"]).is_dir():
            continue
        root = Path(cfg["state_root"]) / str(loan["port"])
        spec = {"name": loan["instance"], "kind": "meowdoku", "port": loan["port"], "bind": "127.0.0.1", "route": "",
                "site": "loan", "account": cfg.get("account", "svc-meowdoku-loans"), "leases": [],
                "code_root": loan["path"] if loan["repository"] == "turing" else cfg["dev_turing"],
                "state_dir": str(root), "args": [*cfg.get("args", []), "--web-dir", str(root / "web")],
                "loan": {"repository": loan["repository"], "branch": loan["branch"], "user_id": loan["user_id"]}}
        speaktome = loan["path"] if loan["repository"] == "speaktome" else cfg.get("dev_speaktome")
        if speaktome:
            spec["speaktome_root"] = speaktome
        if cfg.get("python"):
            spec["python"] = cfg["python"]
        out.append(spec)
    return out


def _git(path, *args) -> str:
    done = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True)
    return done.stdout.strip() if done.returncode == 0 else ""


def _last_used(db, repository, branch) -> float | None:
    try:
        row = db.execute("SELECT MAX(at) FROM checkout_calls WHERE repository = ? AND branch = ?", (repository, branch)).fetchone()
    except Exception:
        return None
    return row[0] if row else None


def export_pages(code_root, web_dir, *, boards_from=None, python=sys.executable, timeout=1800) -> str:
    """The checkout's pages, by its own web_export.py. The boards database is copied from the dev pages, not generated."""
    web_dir = Path(web_dir)
    web_dir.mkdir(parents=True, exist_ok=True)
    if not (web_dir / "boards.json").exists() and boards_from and Path(boards_from).is_file():
        shutil.copyfile(boards_from, web_dir / "boards.json")
    script = Path(code_root) / "examples" / "meowdoku" / "web_export.py"
    if not script.is_file():
        raise LoanError(f"{script} is missing")
    done = subprocess.run([python, str(script), "--skip-boards", "--out", str(web_dir)], cwd=str(code_root),
                          capture_output=True, text=True, timeout=timeout)
    if done.returncode != 0:
        raise LoanError(f"web_export exited {done.returncode}: {(done.stderr or done.stdout).strip()[-200:]}")
    if not (web_dir / "boards.json").exists():
        raise LoanError("the export has no boards.json (give the loans block a boards_from)")
    return "pages exported"


def refresh(db, cfg, *, requester, known=(), now=None, python=sys.executable, log=print) -> list:
    """One pass over the loans; returns what was done to each. ``requester`` = (user_id, username) for restart requests."""
    from meowdoku import instance_control
    now = time.time() if now is None else now
    done = []
    for loan in loans(db):
        entry = {"instance": loan["instance"], "repository": loan["repository"], "branch": loan["branch"]}
        if not loan["path"] or not Path(loan["path"]).is_dir():
            release(db, loan["repository"], loan["branch"])
            done.append(dict(entry, action="ended", reason="its checkout is closed or its folder is gone"))
            continue
        used = _last_used(db, loan["repository"], loan["branch"]) or loan["created_at"]
        if now - used > IDLE_HOURS * 3600:
            if loan["state"] != "idle":
                db.execute("UPDATE branch_servers SET state = 'idle', detail = ? WHERE port = ?",
                           (f"unused for {IDLE_HOURS:g} h", loan["port"]))
                done.append(dict(entry, action="idle", reason=f"unused through its checkout for {IDLE_HOURS:g} h"))
            continue
        head = _git(loan["path"], "rev-parse", "HEAD")
        if head and head == loan["exported_head"]:
            if loan["state"] == "idle":                  # used again: back to ready, and the supervisor starts it
                db.execute("UPDATE branch_servers SET state = 'ready', detail = NULL WHERE port = ?", (loan["port"],))
                done.append(dict(entry, action="woken", reason="its checkout was used again"))
            continue
        code_root = loan["path"] if loan["repository"] == "turing" else cfg["dev_turing"]
        web = Path(cfg["state_root"]) / str(loan["port"]) / "web"
        try:
            said = export_pages(code_root, web, boards_from=cfg.get("boards_from"), python=python)
        except (LoanError, OSError, subprocess.TimeoutExpired) as error:
            db.execute("UPDATE branch_servers SET detail = ? WHERE port = ?", (f"export failed: {error}"[:300], loan["port"]))
            done.append(dict(entry, action="held", reason=f"export failed: {error}"))
            continue
        was_running = loan["state"] == "ready"
        db.execute("UPDATE branch_servers SET exported_head = ?, state = 'ready', detail = NULL WHERE port = ?", (head, loan["port"]))
        entry.update(action="exported", reason=f"{said} at {head[:12]}")
        if was_running:
            try:
                instance_control.request(db, instance=loan["instance"], action="restart", user_id=requester[0],
                                         username=requester[1], reason=f"new commit {head[:12]} in {loan['branch']}",
                                         known=[*known, loan["instance"]])
                entry["reason"] += "; restart requested"
            except instance_control.ControlError as error:
                entry["reason"] += f"; restart not requested: {error}"
        done.append(entry)
    write_proxy_map(db, cfg)
    for entry in done:
        log(f"[loans] {entry['instance']} {entry['repository']} {entry['branch']}: {entry['action']} ({entry['reason']})")
    return done


def write_proxy_map(db, cfg) -> Path:
    """``<state root>/loans.map``: the dev Apache's RewriteMap (txt) of the ports it may proxy to, one ``<port> ready`` line per
    ready loan. Apache rereads a txt map when its time changes; a port missing here is refused (deploy/apache/dev-proxy.conf).
    Written whole and replaced in one rename, so Apache never reads half a file."""
    path = Path(cfg["state_root"]) / "loans.map"
    path.parent.mkdir(parents=True, exist_ok=True)
    ready = sorted(loan["port"] for loan in loans(db) if loan["state"] == "ready" and loan["path"] and Path(loan["path"]).is_dir())
    text = "# written by branch_servers.py (loans.refresh): the ports of the ready branch servers\n" + "".join(f"{p} ready\n" for p in ready)
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        tmp = path.with_suffix(".map.tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    return path
