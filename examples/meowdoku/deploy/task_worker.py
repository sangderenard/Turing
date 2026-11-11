"""The site's task worker: it performs the scheduled tasks the Nodus cron tool asks for, so the web server does not.

    python examples/meowdoku/deploy/task_worker.py --state-dir STATE --web-dir WEB --cron-jobs-dir STATE/jobs/cron \
        --backup-dir BACKUPS --backup-keep 3 --backup-exclude content/marble_photos [--cron-port PORT]

A process of its own under the supervisor (an instance of kind `command` that names the Nodus host and the `cron`
prefix; the supervisor then appends `--cron-port`). It serves nothing and listens on nothing: it connects to the cron
prefix as a host and waits to be asked. What it can be asked:

  manual.hardcopy   check the published hard copy of the manual against the manual; rebuild it when they differ
  state.backup      one snapshot of the state root (when --backup-dir is given), without the --backup-exclude folders

At its own start it also makes sure the manual's hard copy exists, so the site has one before the first schedule.
Without `--cron-port` (the Nodus host declares no cron port) it does that one check and then idles, so the supervisor
sees a running instance rather than one that keeps exiting.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import signal
import sys
import threading

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import rest_api, scheduled_tasks, state_backup, studio_product_nodus  # noqa: E402
from meowdoku.living_data_query import GraphView  # noqa: E402


def say(text: str) -> None:
    print(text, flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--web-dir", required=True, help="the folder the site is served from; the manual goes in <web-dir>/manual")
    parser.add_argument("--cron-jobs-dir", required=True)
    parser.add_argument("--cron-port", type=int, default=None, help="appended by the supervisor from the declared cron prefix")
    parser.add_argument("--backup-dir", default=None, help="where state snapshots go; without it state.backup is not offered")
    parser.add_argument("--backup-keep", type=int, default=3, help="how many snapshots to keep")
    parser.add_argument("--backup-exclude", action="append", default=[], metavar="STATE-RELATIVE-FOLDER",
                        help="a folder of the state root a snapshot leaves out (repeatable)")
    parser.add_argument("--federation-port", default=None, help=argparse.SUPPRESS)      # the supervisor passes these to any
    parser.add_argument("--federation-canvas", default=None, help=argparse.SUPPRESS)    # instance that names a Nodus host
    args = parser.parse_args(argv)

    needs = {}
    if rest_api.manual is not None:
        needs.update(web_dir=args.web_dir, manual=rest_api.manual, site=rest_api.site, graph_view=GraphView)
        try:
            _changed, sentence = scheduled_tasks.refresh_hard_copy(args.web_dir, name=scheduled_tasks.MANUAL_FOLDER,
                                                                   document=rest_api.manual.as_map(), site=rest_api.site,
                                                                   graph_view=GraphView)
            say(f"[worker] at start: {sentence}")
        except Exception as error:
            say(f"[worker] the manual hard copy could not be written at start: {type(error).__name__}: {error}")
    if args.backup_dir:
        needs["backup"] = {"state_dir": args.state_dir, "backup_dir": args.backup_dir, "keep": max(1, args.backup_keep),
                           "exclude": tuple(args.backup_exclude), "module": state_backup}

    stop = threading.Event()
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), lambda *_: stop.set())
    host = None
    if args.cron_port:
        host = scheduled_tasks.start(port=args.cron_port, jobs_dir=args.cron_jobs_dir,
                                     native=studio_product_nodus.nodus_modules()[0],
                                     nodus_tools=studio_product_nodus.DEFAULT_NODUS_TOOLS, log=say, **needs)
        say(f"[worker] performing {', '.join(sorted(host.handlers)) or 'nothing'} from Nodus port {args.cron_port}")
    else:
        say("[worker] no cron port was given; nothing will be scheduled here")
    try:
        while not stop.wait(1.0):
            pass
    finally:
        if host is not None:
            host.close()
    say("[worker] stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
