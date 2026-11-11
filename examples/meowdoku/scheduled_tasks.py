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
API, files copied, a manifest with hashes), leaving out the folders the deployment names and keeping a small number
of snapshots. Large binary products (photo jobs) are a different policy and are not this task's.
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


def backup_state(state_dir, backup_dir, *, keep, exclude, state_backup) -> str:
    """One snapshot of the state root without the excluded folders; returns the sentence for the result."""
    manifest = state_backup.snapshot(state_dir, backup_dir, keep=keep, exclude=tuple(exclude))
    size = sum(entry["size"] for entry in manifest["files"])
    return (f"snapshot {Path(manifest['snapshot']).name}: {len(manifest['files'])} files, {size} bytes, "
            f"{len(manifest['skipped'])} skipped, {len(manifest['removed'])} old removed, keeping {keep}; "
            f"left out: {', '.join(exclude) or 'nothing'}")


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


def handlers(cron_host, *, web_dir=None, manual=None, site=None, graph_view=None, backup=None) -> dict:
    """The tasks this process can perform, by name. A task appears only when what it needs was given.

    `backup` is `{"state_dir", "backup_dir", "keep", "exclude", "module"}` (module = state_backup)."""
    table = {}
    if web_dir is not None and manual is not None:
        def manual_hardcopy(request):
            changed, sentence = refresh_hard_copy(web_dir, name=MANUAL_FOLDER, document=manual.as_map(), site=site,
                                                  graph_view=graph_view)
            return (cron_host.DONE if changed else cron_host.UNCHANGED), sentence
        table["manual.hardcopy"] = manual_hardcopy
    if backup is not None:
        def state_backup_task(request):
            return cron_host.DONE, backup_state(backup["state_dir"], backup["backup_dir"], keep=backup["keep"],
                                                exclude=backup["exclude"], state_backup=backup["module"])
        table["state.backup"] = state_backup_task
    return table


def start(*, port, jobs_dir, native, nodus_tools, log=print, **needs):
    """Join the cron prefix as a host for every task `handlers(**needs)` gives. Returns the CronHost; close() it."""
    cron_host = load_cron_host(nodus_tools)
    return cron_host.CronHost(native, port=port, host_id=HOST_ID, jobs_dir=jobs_dir, log=log,
                              handlers=handlers(cron_host, **needs))
