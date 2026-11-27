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

Later the same day (service interpreter C:/dev/Powershell/srv/python 3.11.18, aiosmtpd 1.4.6 installed there):
tests/test_meowdoku_mail.py 11 passed, 1 skipped; STARTTLS through mail_service.start_receiver negotiated TLSv1.3 on
loopback and stored the message with tls=1. Set-ServiceFirewall.ps1, non-elevated: reports the missing rule, exits 0.

Hazards:
- The `mail` instance is ENABLED. At the first restart after this lands, the supervisor starts it: the receiver binds
  0.0.0.0:47625 as srv/python/python.exe. Its firewall rule comes from Set-ServiceFirewall.ps1, which Start-Apache.ps1
  runs; a non-elevated bring-up only reports the missing rule. instances.json also gives nodus-messenger
  `--acquire mail:4:36000` and `--cron mail:7204:<state>/jobs/mail`; until Enable-MailTable.ps1 has added port 47619
  and installed the table, the canvas host logs that the mail cron tool did not attach (not fatal; not observed live)
  and sends wait for the sweep.
- Not measured: an elevated firewall run, a real provider, the live stop window.
