"""The site's task worker: it performs the scheduled tasks the Nodus cron tool asks for, so the web server does not.

    python examples/meowdoku/deploy/task_worker.py --state-dir STATE --web-dir WEB --cron-jobs-dir STATE/jobs/cron \
        --backup-dir BACKUPS --backup-keep 1 --backup-text-only [--cron-port PORT]

A process of its own under the supervisor (an instance of kind `command` that names the Nodus host and the `cron`
prefix; the supervisor then appends `--cron-port`). It serves nothing and listens on nothing: it connects to the cron
prefix as a host and waits to be asked. What it can be asked:

  manual.hardcopy   check the published hard copy of the manual against the manual; rebuild it when they differ
  system.broadcast  tell an audience one system event, as messages from the system account (when --accounts-db and
                    --system-account are given); asked for once per event, never on the schedule
  state.backup      one snapshot of the state root (when --backup-dir is given): with --backup-text-only the
                    databases and the text files, no binary files; --backup-keep 1 means no rolling window

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

from meowdoku import messaging, rest_api, scheduled_tasks, state_backup, studio_product_nodus  # noqa: E402
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
    parser.add_argument("--backup-keep", type=int, default=1, help="how many snapshots to keep (1 = the last one only)")
    parser.add_argument("--backup-text-only", action="store_true", help="databases and text files only; no binary files")
    parser.add_argument("--backup-exclude", action="append", default=[], metavar="STATE-RELATIVE-FOLDER",
                        help="a folder of the state root a snapshot leaves out (repeatable)")
    parser.add_argument("--userlist-port", type=int, default=None, help="appended by the supervisor from the declared userlist prefix")
    parser.add_argument("--userlist-jobs-dir", default=None, help="the userlist table's directory (one request and result file per search)")
    parser.add_argument("--accounts-db", default=None, help="the accounts database; with --system-account, system.broadcast is offered")
    parser.add_argument("--system-account", default=None,
                        help="the account system events are sent as (the Nodus host's service account, shown as Nodus)")
    parser.add_argument("--branch-policy", default=None,
                        help="the branch policy file; with --accounts-db, reviews.land and branches.clean are offered")
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
                           "exclude": tuple(args.backup_exclude), "text_only": args.backup_text_only,
                           "module": state_backup}

    if args.branch_policy and args.accounts_db:
        from meowdoku import branch_gate
        needs["branches"] = {"policy": args.branch_policy, "accounts_db": args.accounts_db, "module": branch_gate}

    native, tools = studio_product_nodus.nodus_modules()[0], studio_product_nodus.DEFAULT_NODUS_TOOLS
    if args.accounts_db and args.system_account:
        needs["system"] = {"accounts_db": args.accounts_db, "username": args.system_account, "module": messaging}
        if args.userlist_port and args.userlist_jobs_dir:      # the broadcast finds its audience by asking this table
            needs["system"]["userlist"] = {"port": args.userlist_port, "jobs_dir": args.userlist_jobs_dir, "native": native,
                                           "task_request": scheduled_tasks.load_task_request(tools)}
        try:
            say("[worker] at start: " + scheduled_tasks.befriend_everyone(args.accounts_db, args.system_account, messaging=messaging))
        except Exception as error:
            say(f"[worker] the system account could not be made a contact at start: {type(error).__name__}: {error}")

    stop = threading.Event()
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), lambda *_: stop.set())
    hosts = []
    if args.cron_port:
        host = scheduled_tasks.start(port=args.cron_port, jobs_dir=args.cron_jobs_dir, native=native, nodus_tools=tools,
                                     log=say, **needs)
        hosts.append(host)
        say(f"[worker] performing {', '.join(sorted(host.handlers)) or 'nothing'} from Nodus port {args.cron_port}")
    else:
        say("[worker] no cron port was given; nothing will be scheduled here")
    if args.userlist_port and args.userlist_jobs_dir and "system" in needs:
        Path(args.userlist_jobs_dir).mkdir(parents=True, exist_ok=True)
        host = scheduled_tasks.start(port=args.userlist_port, jobs_dir=args.userlist_jobs_dir, native=native, nodus_tools=tools,
                                     log=say, prefix="userlist", host_id=scheduled_tasks.HOST_ID + 1, **needs)
        hosts.append(host)
        say(f"[worker] the userlist table performs {', '.join(sorted(host.handlers))} from Nodus port {args.userlist_port}")
    try:
        while not stop.wait(1.0):
            pass
    finally:
        for host in hosts:
            host.close()
    say("[worker] stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
