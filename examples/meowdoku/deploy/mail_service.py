"""The mail service: the SMTP receiver on an odd port and the host of the Nodus email manager table.

    python examples/meowdoku/deploy/mail_service.py --config examples/meowdoku/deploy/mail.json \
        --accounts-db STATE/sql/accounts.db --mail-jobs-dir STATE/jobs/mail [--mail-db STATE/sql/mail.db] [--mail-port PORT]

A process of its own under the supervisor (instance `mail`, kind command, account svc-mail; it names the Nodus host
and the `mail` prefix, so the supervisor appends `--mail-port`). Two jobs:

  receiver   aiosmtpd on <listen_host>:<listen_port> (47625, never 25), STARTTLS with the configured certificate,
             mail accepted only for person accounts of the domain (mail.Inbound). A stored message is handed to the
             mail table as a mail.file task.
  host       the host of the `mail` prefix: performs mail.file, mail.send and mail.sweep (mail.handlers). The table's
             own cron tool asks for mail.sweep on its schedule (deploy/mail.cron.table).

Without aiosmtpd in this interpreter the receiver stays off and says so; the host still runs, so outgoing mail
works. Without `--mail-port` (the canvas has no mail table yet) nothing is hosted and the receiver hands nothing
on: the sweep, once the table exists, files what was stored. Nothing here prints the provider credential or the
certificate's key.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import signal
import ssl
import sys
import threading

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import mail, scheduled_tasks, studio_product_nodus  # noqa: E402


def say(text: str) -> None:
    print(text, flush=True)


def tls_context(config):
    """The STARTTLS context, or None (and why) when no certificate is configured or it cannot be loaded."""
    if not config.get("tls_cert") or not config.get("tls_key"):
        return None, "no certificate is configured; STARTTLS is not offered"
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    try:
        context.load_cert_chain(config["tls_cert"], config["tls_key"])
    except (OSError, ssl.SSLError) as error:
        return None, f"the certificate could not be loaded ({type(error).__name__}); STARTTLS is not offered"
    return context, f"STARTTLS with {Path(config['tls_cert']).name}"


def start_receiver(handler, config):
    """The aiosmtpd controller, started; or (None, why)."""
    try:
        from aiosmtpd.controller import Controller
    except ImportError:
        return None, "aiosmtpd is not installed in this interpreter; the receiver is off"
    context, how = tls_context(config)
    controller = Controller(handler, hostname=config["listen_host"], port=int(config["listen_port"]),
                            server_hostname=config["domain"], ident=f"{config['domain']} ESMTP",
                            tls_context=context, require_starttls=False,
                            data_size_limit=int(config["max_message_bytes"]))
    controller.start()
    return controller, f"receiving for @{config['domain']} on {config['listen_host']}:{config['listen_port']}; {how}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="deploy/mail.json")
    parser.add_argument("--accounts-db", required=True, help="the accounts database (read only here)")
    parser.add_argument("--mail-db", default=None, help="the mail database; beside accounts.db when omitted")
    parser.add_argument("--mail-jobs-dir", required=True, help="the mail table's directory (request and result files)")
    parser.add_argument("--mail-port", type=int, default=None, help="appended by the supervisor from the declared mail prefix")
    parser.add_argument("--no-receiver", action="store_true", help="host the table only")
    parser.add_argument("--federation-port", default=None, help=argparse.SUPPRESS)      # the supervisor passes these to any
    parser.add_argument("--federation-canvas", default=None, help=argparse.SUPPRESS)    # instance that names a Nodus host
    args = parser.parse_args(argv)

    config = mail.load_config(args.config)
    mail_db = Path(args.mail_db) if args.mail_db else mail.mail_db_beside(args.accounts_db)
    mail.connect(mail_db).close()                                    # the tables exist before anyone asks
    Path(args.mail_jobs_dir).mkdir(parents=True, exist_ok=True)
    why = mail.provider_ready(config)
    say(f"[mail] outgoing: {'through ' + config['provider_host'] if why is None else 'waiting, ' + why}")

    stop = threading.Event()
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), lambda *_: stop.set())

    host = None
    if args.mail_port:
        tools = studio_product_nodus.DEFAULT_NODUS_TOOLS
        cron_host = scheduled_tasks.load_cron_host(tools)
        host = cron_host.CronHost(studio_product_nodus.nodus_modules(tools)[0], port=args.mail_port, host_id=mail.HOST_ID,
                                  jobs_dir=args.mail_jobs_dir, log=say,
                                  handlers=mail.handlers(cron_host, mail_db=mail_db, accounts_db=args.accounts_db,
                                                         config=config))
        say(f"[mail] the mail table performs {', '.join(sorted(host.handlers))} from Nodus port {args.mail_port}")
    else:
        say("[mail] no mail port was given: the canvas has no mail table, so nothing is routed from here")

    controller = None
    if not args.no_receiver:
        submit = mail.table_submitter(port=args.mail_port, jobs_dir=args.mail_jobs_dir,
                                      engine_id=mail.ENGINE_IDS["receiver"], log=say)
        handler = mail.Inbound(mail_db, args.accounts_db, config,
                               on_received=(lambda message_id: submit("mail.file", message_id)) if submit else None,
                               log=say)
        try:
            controller, said = start_receiver(handler, config)
        except OSError as error:
            controller, said = None, f"the receiver could not listen ({type(error).__name__}: {error})"
        say(f"[mail] {said}")
    try:
        while not stop.wait(1.0):
            pass
    finally:
        if controller is not None:
            controller.stop()
        if host is not None:
            host.close()
    say("[mail] stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
