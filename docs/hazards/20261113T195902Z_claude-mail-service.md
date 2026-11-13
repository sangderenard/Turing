2026-11-13 Mail service: mailboxes, an odd-port receiver, a provider gateway, the Nodus `mail` table (MEASURED, claude/mail-service from main fe797953)

Meowdoku only; no compiler code touched. Design and owner steps: examples/meowdoku/deploy/MAIL_2026-11-13.md.

Measured on this machine (system Python 3.11.7):
- tests/test_meowdoku_mail.py: 10 passed, 2 skipped (aiosmtpd is installed nowhere here; the real-canvas test is
  opt-in).
- The same file with NODUS_HEADLESS_TEST_DLL=nodus/build-bridging/Release/nodus_headless.dll and that folder on PATH,
  `-k real_nodus`: 1 passed (canvas host with a `mail` contract port, deploy/mail_service.py as host, an engine
  asking mail.file, message filed). No build was run.
- tests/test_meowdoku_authz.py + tests/test_meowdoku_server_authz.py: 26 passed (member LAN-edit expectation now
  excludes the new `mail_send`).
- tests/test_meowdoku_apache_supervisor.py -k "federation_from or shipped_instances or disabled_instance": 3 passed
  (shipped instance list now includes `mail`, which ships disabled).

Hazards:
- The `mail` instance ships `"enabled": false`. instances.json also gives nodus-messenger `--acquire mail:4:36000`
  and `--cron mail:7204:<state>/jobs/mail`; until Enable-MailTable.ps1 has added the port and installed the table,
  the canvas host logs that the mail cron tool did not attach (not fatal) and meowdoku-a queues sends for the sweep.
- Not measured: aiosmtpd's Controller arguments (server_hostname, ident, tls_context, require_starttls,
  data_size_limit) against aiosmtpd 1.4.6 itself, and a real provider.
