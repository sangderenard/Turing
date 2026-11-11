"""The tasks this server performs when the Nodus cron tool asks, and the manual's hard copy.

The schedule is not here. It is the cron tool's table (`<jobs_dir>/cron.table`, read by the Nodus canvas host), and
the cron tool decides what is due and what a gap made it miss. This server is a HOST on the cron prefix: it takes a
contract, reads the request the tool wrote, performs the named task and answers with a status (nodus
`tools/cron_host.py`). A task name selects one of the handlers registered below, nothing else.

`manual.hardcopy` is a refresh check: the manual is a map, the hard copy is that map baked into static pages under
`<web_dir>/manual/`, and the folder records the hash of the map it was baked from. When the hash is unchanged
nothing is written. When it differs the whole folder is replaced, so pages of places that no longer exist go away.
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


def start(*, port, jobs_dir, web_dir, native, nodus_tools, manual, site, graph_view, log=print):
    """Join the cron prefix as a host. Returns the running CronHost; close() it at shutdown."""
    cron_host = load_cron_host(nodus_tools)

    def manual_hardcopy(request):
        changed, sentence = refresh_hard_copy(web_dir, name=MANUAL_FOLDER, document=manual.as_map(), site=site,
                                              graph_view=graph_view)
        return (cron_host.DONE if changed else cron_host.UNCHANGED), sentence

    return cron_host.CronHost(native, port=port, host_id=HOST_ID, jobs_dir=jobs_dir, log=log,
                              handlers={"manual.hardcopy": manual_hardcopy})
