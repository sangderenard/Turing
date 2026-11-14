"""The scheduled tasks of the site, and the worker-side host that performs them when the Nodus cron tool asks.

The schedule is not here. It is the cron tool's table (`<jobs_dir>/cron.table`, read by the Nodus canvas host), and
the cron tool decides what is due and what a gap made it miss. Whoever calls `start` becomes a HOST on the cron
prefix: it takes a contract, reads the request the tool wrote, performs the named task and answers with a status
(nodus `tools/cron_host.py`). A task name selects one of the handlers registered below, nothing else.

The web server does not call `start`. The tasks are performed by the task worker (`deploy/task_worker.py`), a process
of its own, so that baking and backing up are off the server's plate.

`manual.hardcopy` is a refresh check: the manual is a map, the hard copy is that map baked into static pages under
`<web_dir>/manual/`, and the folder records the hash of the map it was baked from. When the hash is unchanged
nothing is written. When it differs the whole folder is replaced, so pages of places that no longer exist go away.

`state.backup` is a snapshot of the state root through `state_backup.snapshot` (databases through the sqlite backup
API, files copied, a manifest with hashes). The owner's policy, 2026-11-11: "just the non binary files and the
database nightly no rolling window". So the deployment runs it text-only and keeps one snapshot: each night's copy
replaces the last. Binary products (photo jobs and the like) are a different policy and are not this task's.

`system.broadcast` is one system event told to an audience as messages from the system account (the Nodus host's
own service account, shown as "Nodus"). It is not on the schedule: an engine asks for it once, when the event
happens (nodus `tools/task_request.py`). The system account is made a contact of each recipient first.

The audience is not looked up here. It is a search of the USER LIST, which is its own table on the canvas (prefix
`userlist`): that table receives search terms and its finished output is every identity that matched
(`identity.search`); another tool of it represents one identity (`identity.represent`). The broadcast asks that
table for one tick and then tells each identity it returned. The tools are `identity_tools.py`; they only read.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

MANUAL_FOLDER = "manual"
STAMP = "hard-copy.json"
HOST_ID = 7301
AUDIENCE_TERMS = {"admin": ("role:admin",), "everyone": ("kind:person",)}     # an audience is a search of the user list
SYSTEM_DISPLAY_NAME = "Nodus"
USERLIST_ENGINE_ID = 7203
# Which table performs which task: scheduled and system tasks on the cron prefix, the user list on its own.
PREFIX_TASKS = {"cron": ("manual.hardcopy", "state.backup", "system.broadcast", "reviews.land", "branches.clean",
                         "core.index"),
                "userlist": ("identity.search", "identity.represent")}


def _system(db, username):
    """The system account's id; it is shown to people as Nodus."""
    row = db.execute("SELECT id, display_name FROM users WHERE username = ? COLLATE NOCASE", (username,)).fetchone()
    if row is None:
        raise LookupError(f"no system account {username!r}")
    if row["display_name"] != SYSTEM_DISPLAY_NAME:
        db.execute("UPDATE users SET display_name = ? WHERE id = ?", (SYSTEM_DISPLAY_NAME, row["id"]))
    return row["id"]


def find_identities(db, terms, userlist=None) -> tuple:
    """The identities a list of search terms matches, and how they were found.

    With a ``userlist`` table to ask ({"port", "jobs_dir", "native", "task_request"}) the search is ONE TICK OF THAT
    TABLE: a contract carrying the terms, performed by the table's host, whose finished output is the identities.
    Without one (no such port on this canvas) the same tool is called here directly."""
    from meowdoku import identity_tools
    if userlist is not None:
        asked = userlist["task_request"].submit(userlist["native"], port=userlist["port"], engine_id=USERLIST_ENGINE_ID,
                                                jobs_dir=userlist["jobs_dir"], task="identity.search",
                                                fields={"terms": list(terms)}, timeout=10.0)
        record = userlist["task_request"].result(userlist["jobs_dir"], asked["contract"]) if asked["outcome"] == "done" else None
        if record is not None and isinstance(record.get("output"), list):
            return record["output"], f"userlist table, contract {asked['contract']}"
        return identity_tools.search(db, terms), f"directly (the userlist table answered {asked['outcome']})"
    return identity_tools.search(db, terms), "directly"


def befriend_everyone(accounts_db, system_username, *, messaging) -> str:
    """Make the system account a mutual contact of every account that is not a service account."""
    db = messaging.connect(accounts_db)
    try:
        system_id = _system(db, system_username)
        from meowdoku import identity_tools
        people = [int(i["identity"].split(":", 1)[1]) for i in identity_tools.search(db, AUDIENCE_TERMS["everyone"])]
        people = [user_id for user_id in people if user_id != system_id]
        mutual = messaging.befriend_all(db, system_id, people)
    finally:
        db.close()
    return f"{SYSTEM_DISPLAY_NAME} is a contact of {mutual} of {len(people)} accounts"


def broadcast(accounts_db, system_username, request, *, messaging, userlist=None) -> str:
    """One system event to an audience, as messages from the system account.  The request carries ``text``, and
    either ``audience`` (admin | everyone) or ``terms`` (search terms of the user list), and optionally ``msg_type``
    and ``subject`` ({kind, ref}).  The audience is found by searching the user list (``find_identities``); each
    identity found is made a contact of the system (one that blocked it is skipped) and told.  Returns the sentence."""
    text, audience = request.get("text"), request.get("audience", "admin")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("a broadcast needs text")
    if "terms" in request:
        terms, audience = request["terms"], "matching"
    elif audience in AUDIENCE_TERMS:
        terms = AUDIENCE_TERMS[audience]
    else:
        raise ValueError(f"audience must be one of {', '.join(AUDIENCE_TERMS)}")
    db = messaging.connect(accounts_db)
    sent, skipped = 0, 0
    try:
        system_id = _system(db, system_username)
        identities, how = find_identities(db, terms, userlist)
        for identity in identities:
            user_id = int(identity["identity"].split(":", 1)[1])
            if user_id == system_id:
                continue
            if not messaging.system_contact(db, system_id, user_id):
                skipped += 1
                continue
            try:
                messaging.send(db, system_id, identity["username"], text, subject=request.get("subject"),
                               msg_type=request.get("msg_type"))
                sent += 1
            except messaging.MessagingError:
                skipped += 1
    finally:
        db.close()
    return f"told {sent} {audience} account(s)" + (f", {skipped} skipped" if skipped else "") + f"; audience found {how}"


def refresh_hard_copy(web_dir, *, name, document, site, graph_view) -> tuple:
    """Bake `document` into `<web_dir>/<name>/` unless that folder already holds this exact map. (changed, sentence)."""
    web_dir = Path(web_dir)
    target, stamp = web_dir / name, web_dir / name / STAMP
    digest = site.digest(document)
    try:
        held = json.loads(stamp.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        held = {}
    if held.get("digest") == digest and (target / "index.html").is_file():
        return False, f"the {name} hard copy already matches the map ({digest[:12]})"
    fresh, old = web_dir / f".{name}.new", web_dir / f".{name}.old"
    for leftover in (fresh, old):
        shutil.rmtree(leftover, ignore_errors=True)
    scratch = web_dir / f".{name}.cache"
    try:
        pages = site.Site(scratch, graph_view).open(name, document).bake(fresh)
        (fresh / STAMP).write_text(json.dumps({"digest": digest, "pages": pages, "baked": time.time()}), encoding="utf-8")
        if target.exists():
            target.rename(old)
        fresh.rename(target)
    finally:
        for leftover in (fresh, old, scratch):
            shutil.rmtree(leftover, ignore_errors=True)
    return True, f"the {name} hard copy was rebuilt: {pages} pages from map {digest[:12]}"


def backup_state(state_dir, backup_dir, *, keep, exclude, state_backup, text_only=False) -> str:
    """One snapshot of the state root; returns the sentence for the result."""
    manifest = state_backup.snapshot(state_dir, backup_dir, keep=keep, exclude=tuple(exclude), text_only=text_only)
    size = sum(entry["size"] for entry in manifest["files"])
    binary = manifest["binary_left_out"]
    return (f"snapshot {Path(manifest['snapshot']).name}: {len(manifest['files'])} files, {size} bytes, "
            f"{len(manifest['skipped'])} skipped, {len(manifest['removed'])} old removed, keeping {keep}; "
            f"binary left out: {binary['files']} files, {binary['bytes']} bytes; "
            f"folders left out: {', '.join(exclude) or 'none'}")


def load_cron_host(nodus_tools):
    """The actual nodus `tools/cron_host.py`, loaded by path as the other Nodus helpers are."""
    path = Path(nodus_tools).resolve() / "cron_host.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    name = "_meowdoku_nodus_cron_host"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def handlers(cron_host, *, web_dir=None, manual=None, site=None, graph_view=None, backup=None, system=None,
             branches=None) -> dict:
    """The tasks this process can perform, by name. A task appears only when what it needs was given.

    `system` is `{"accounts_db", "username", "module", "userlist"?}` (module = messaging): the account system events
    are sent as, and optionally the userlist table to ask for an audience ({"port", "jobs_dir", "native", "task_request"}).
    `backup` is `{"state_dir", "backup_dir", "keep", "exclude", "text_only", "module"}` (module = state_backup).
    `branches` is `{"policy", "accounts_db", "module"}` (module = branch_gate): reviewed changes are landed and merged
    branches cleaned by that module's rules; while the policy's `apply` is false both tasks only report."""
    table = {}
    if branches is not None:
        def branch_task(which):
            def task(request):
                import sqlite3
                gate = branches["module"]
                policy = gate.load_policy(branches["policy"])
                db = sqlite3.connect(branches["accounts_db"], timeout=10, isolation_level=None)
                try:
                    if which == "clean":         # the worker performs the cleaning as the policy's cleanup account, and
                        acting = {"apply": bool(policy["apply"]), "actor": policy["cleaner"]}    # as no other
                        done = gate.clean_worktrees(policy, db, **acting) + gate.clean(policy, db, **acting)
                    else:                # a one-off request may name the reviews to land (the landing queue's "land now")
                        only = request.get("reviews") if isinstance(request, dict) else None
                        only = [str(r) for r in only] if isinstance(only, list) and only else None
                        done = gate.land(policy, db, apply=bool(policy["apply"]), only=only)
                finally:
                    db.close()
                counts = {}
                for entry in done:
                    counts[entry["action"]] = counts.get(entry["action"], 0) + 1
                acted = any(entry["action"] in ("landed", "removed") for entry in done)
                said = ", ".join(f"{count} {action}" for action, count in sorted(counts.items())) or "nothing to do"
                if which == "clean" and any(entry["action"] == "removed" for entry in done):
                    try:                         # the record of what was removed goes into Git's history as well as SQL
                        db = sqlite3.connect(branches["accounts_db"], timeout=10, isolation_level=None)
                        try:
                            said += "; " + gate.publish_notices(policy, db, commit=True)
                        finally:
                            db.close()
                    except Exception as error:
                        said += f"; the notices were not published ({type(error).__name__})"
                skipped = sum(1 for entry in done if entry.get("remote_skipped"))
                if skipped:
                    said += f"; {skipped} remote copies left because a GitHub rule forbids deleting them"
                return (cron_host.DONE if acted else cron_host.UNCHANGED), said, [e for e in done if e["action"] not in ("keep", "wait")]
            return task
        def core_index(request):
            from pathlib import Path
            from meowdoku import file_index
            policy = branches["module"].load_policy(branches["policy"])
            made = file_index.build(policy, Path(branches["accounts_db"]).with_name("file_index.db"), actor=policy["cleaner"])
            return cron_host.DONE, f"{made['entries']} entries indexed in {made['seconds']} s", made
        table["core.index"] = core_index
        table["reviews.land"] = branch_task("land")
        table["branches.clean"] = branch_task("clean")
    if web_dir is not None and manual is not None:
        def manual_hardcopy(request):
            changed, sentence = refresh_hard_copy(web_dir, name=MANUAL_FOLDER, document=manual.as_map(), site=site,
                                                  graph_view=graph_view)
            return (cron_host.DONE if changed else cron_host.UNCHANGED), sentence
        table["manual.hardcopy"] = manual_hardcopy
    if backup is not None:
        def state_backup_task(request):
            return cron_host.DONE, backup_state(backup["state_dir"], backup["backup_dir"], keep=backup["keep"],
                                                exclude=backup["exclude"], state_backup=backup["module"],
                                                text_only=bool(backup.get("text_only")))
        table["state.backup"] = state_backup_task
    if system is not None:
        def system_broadcast(request):
            return cron_host.DONE, broadcast(system["accounts_db"], system["username"], request, messaging=system["module"],
                                             userlist=system.get("userlist"))
        table["system.broadcast"] = system_broadcast

        def identity_search(request):
            from meowdoku import identity_tools
            db = system["module"].connect(system["accounts_db"])
            try:
                found = identity_tools.search(db, request.get("terms"))
            finally:
                db.close()
            return cron_host.DONE, f"{len(found)} identities matched", found
        table["identity.search"] = identity_search

        def identity_represent(request):
            from meowdoku import identity_tools
            db = system["module"].connect(system["accounts_db"])
            try:
                one = identity_tools.represent(db, request.get("identity"))
            finally:
                db.close()
            if one is None:
                return cron_host.UNCHANGED, "no such identity"
            return cron_host.DONE, f"{one['identity']} is {one['username']}", one
        table["identity.represent"] = identity_represent
    return table


def load_task_request(nodus_tools):
    """The actual nodus `tools/task_request.py`, loaded by path."""
    path = Path(nodus_tools).resolve() / "task_request.py"
    name = "_meowdoku_nodus_task_request"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def start(*, port, jobs_dir, native, nodus_tools, log=print, prefix="cron", host_id=HOST_ID, **needs):
    """Join one task prefix as its host, for the tasks that prefix performs (PREFIX_TASKS) among those
    `handlers(**needs)` gives. Returns the CronHost; close() it."""
    cron_host = load_cron_host(nodus_tools)
    offered = handlers(cron_host, **needs)
    mine = {name: offered[name] for name in PREFIX_TASKS[prefix] if name in offered}
    return cron_host.CronHost(native, port=port, host_id=host_id, jobs_dir=jobs_dir, log=log, handlers=mine)
