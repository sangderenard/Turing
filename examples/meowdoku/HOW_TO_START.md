# How to start Meowdoku on Windows

Run from `C:\dev\Powershell\turing`. These are the verified deployment settings
as of 2026-11-10. Before a later restart, inspect `build/meowdoku/launch.json`
for the current executable, argv, cwd, and environment. Preserve that record's
settings rather than replacing newer ones with this dated example.

## 1. Required inbound TCP rule

**The Windows inbound TCP 8766 allow rule must be in place for public access.**
If it is missing, an authorized operator must run this in **PowerShell as
Administrator**:

```powershell
New-NetFirewallRule -DisplayName "Web Server 8766" -Direction Inbound -Protocol TCP -LocalPort 8766 -Action Allow -Profile Any
```

This changes the firewall. Do not run it on every restart to create duplicate
rules. Check the existing rule when permitted; an access-denied check does not
prove the rule is absent. A normal non-admin shell can receive Access denied.
Do not disable the firewall or evade a denied operation. The verified rule is
port-based, not a `-Program` rule. No additional executable-specific command
has been established by the restart evidence.

The router forwarding must also reach this machine's current LAN address on
TCP 8766. The recorded installation uses two routers; both forwarding hops
must be correct. A local firewall rule does not configure either router.

## 2. Do not launch over an existing listener

```powershell
Get-NetTCPConnection -LocalPort 8766 -State Listen | Select-Object LocalAddress, LocalPort, OwningProcess
```

If the server is already running, use the restart procedure below instead.
Historical launch scripts also used `System.Net.Sockets.TcpClient.Connect`
to check whether localhost:8766 was occupied. That is a diagnostic probe, not
an enabling or port-forwarding command. Neither check requires a firewall edit.

## 3. Start with the recorded settings

After confirming no existing listener, this is the verified foreground launch:

```powershell
Set-Location -LiteralPath 'C:\dev\Powershell\turing'
$env:PYTHONUTF8='1'
& 'C:\Users\alber\AppData\Local\Programs\Python\Python311\python.exe' examples\meowdoku\server.py --port 8766 --bind 0.0.0.0 --tls-cert build\meowdoku\tls\cert.pem --tls-key build\meowdoku\tls\key.pem --retain-fountain-products C:\dev\Powershell\turing\build\dot-retained-fountain-20261109\fountain-receipt.json
```

Keep that terminal or managed execution session alive. Do not terminate it when
an unrelated task finishes. A short-lived tool's child process may not survive
its host. Historical hidden launches used `Start-Process -WindowStyle Hidden`
with stdout/stderr logs, but a returned PID alone does not prove survival.
Verify the listener and HTTPS after the launching call completes.

The retain receipt preserves 74 existing fountain products. Do not omit it or
rebuild those products as a side effect of a server restart. Do not print or
share TLS private keys or admin tokens.

## 4. Session-preserving restart

1. Coordinate with other workers. Check current still exposures and live jobs;
   avoid interrupting an active exposure. Confirm what backend edits will load.
2. Save a copy of the current `build/meowdoku/launch.json` for the relaunch.
3. Use the existing authenticated lifecycle endpoint. It announces the countdown,
   saves sessions, and exits with code 75. Do not use forced termination.
4. Wait for the old listener to close. Relaunch the same argv/cwd/env only after
   shutdown is confirmed. Keep `build/meowdoku/sessions` and browser cookies.
5. Verify the new boot and returning clients. Sessions restore lazily when the
   matching cookie reconnects, so zero restored sessions immediately after boot
   does not mean the saved sessions were lost.

For this HTTPS deployment, run this Python snippet from the repository root
using the same Python executable. It requests a 10-second controlled restart;
it does not itself relaunch the server:

```python
from pathlib import Path
import json
import ssl
import urllib.request

state = Path('build/meowdoku')
context = ssl.create_default_context(cafile=str(state / 'tls/ca.crt'))
request = urllib.request.Request(
    'https://127.0.0.1:8766/api/admin/lifecycle',
    data=json.dumps({'op': 'restart', 'in': 10}).encode(),
    headers={
        'Content-Type': 'application/json',
        'X-Meowdoku-Admin': (state / 'admin_token.txt').read_text().strip(),
    },
)
with urllib.request.urlopen(request, context=context, timeout=15) as response:
    print(response.read().decode())
```

The lifecycle module's current CLI posts to HTTP localhost; do not assume that
CLI handles this TLS deployment. The HTTPS request above uses the same existing
lifecycle API and validates against the deployment CA.

## 5. Verify local and public access separately

- Listener: `0.0.0.0:8766`, owned by the expected Python process.
- Local HTTPS: `https://127.0.0.1:8766/api/lifecycle` must return 200 with the
  deployment CA trusted. Compare the boot ID before and after a restart.
- Public route: test the actual public hostname or IP on 8766, preferably also
  from outside the LAN. Self-LAN success does not prove external ingress, and
  a public-IP test from inside the LAN depends on router NAT loopback support.
- TLS: use a certificate valid for the hostname/IP and a trusted issuer. A TCP
  connection followed by a certificate mismatch is not an HTTP success. Do not
  disable certificate verification to present it as healthy.
- Sessions and assets: verify reconnect restoration and retained artifact hashes.

On 2026-11-10, after the operator reported installing the rule and requested a
restart, the public IP 173.195.26.104 changed from a timeout to a TLS response.
The probe then reported an IP-name certificate mismatch; local HTTPS returned
200, an existing session restored, and all 74 retained fountain hashes matched.
This records the observed sequence, not proof of a particular firewall mechanism
or an unidentified extra TCP command. Public IPs, leases, certificates, and route
configuration can change; recheck current values instead of assuming this IP.
