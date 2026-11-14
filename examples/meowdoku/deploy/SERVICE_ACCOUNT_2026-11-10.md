# Service account for Apache + Meowdoku + Nodus (2026-11-10)

**Nothing in this document has been executed.** The scripts were only run with `-WhatIf` / `-Status`.

## Purpose

One local Windows user (default `svc-meowdoku`) under which the whole stack runs: Apache as the service `Apache2.4`, which starts
`apache_supervisor.py` (piped CustomLog program of the 443 vhost), which launches the Meowdoku server and the Nodus canvas host.
Because children inherit the token, the supervisor and both instances run as that user too.

## Rights it gets

- Local user, password generated in PowerShell (32 random bytes, url-safe base64), `PasswordNeverExpires`, `UserMayNotChangePassword`.
  Password is written only to `build/meowdoku/operator_credentials/<Name>.windows.secret` (never overwritten; inheritance cut so
  only Administrators, SYSTEM and alber can read it, not the service itself).
- `SeServiceLogonRight` (needed to run a service) and `SeDenyInteractiveLogonRight` (no console/RDP logon), via `secedit` export,
  edit, import of a temporary `.inf`; existing holders are preserved.
- ACLs (icacls):
  - RX, inherited: `C:/Apache24`, `C:/dev/Powershell/turing`, `C:/dev/Powershell/nodus`, `C:/dev/Powershell/srv` (code and DLLs must load).
  - No grant touches `C:/Users/alber`.
  - M, inherited: `C:/Apache24/logs`, `turing/build` (Meowdoku state, logs, accounts.db, tls), `nodus/out`.
  - R on `build/meowdoku/admin_token*.txt` and `tls/key.pem` (explicit; the service is added, alber's access is unchanged).
  - State layout: RX on `secrets/` itself, this folder only (2026-11-13). Apache resolves each folder of a path by name, so
    without listing `secrets/` it is refused `tls-public/fullchain.cer` although the file grants read.
- Service `ConfigArgs` = `-E C:/Apache24/logs/startup-errors.log` (2026-11-13): errors httpd hits before it opens `error.log`
  go there; `Apache-Service.ps1 -Start` prints the file when a start fails.

## First switch, 2026-11-13 (receipt)

The first service start failed three times with only "Incorrect function" in the System log and nothing in `error.log`:
1. The vhost loaded the certificate from the operator's profile (Posh-ACME). Fixed: `apache/meowdoku-proxy.conf` uses the
   `secrets/tls-public` copies.
2. With `-E` set, httpd reported `SSLCertificateFile: Invalid file path .../tls-public/fullchain.cer`: the service could not list
   `secrets/`. Fixed by the folder grant above (`(RD)` alone is not enough; it lacks Synchronize).
3. `httpd -k install` had exited non-zero while the console Apache held the ports; the script now judges by the service.
Still open after the switch: the supervisor writes Nodus console tokens into `secrets/operator_credentials`, which the service
cannot write, so `nodus-messenger`, `nodus-training` and `nodus-ui-mirror` do not start under the service.

## Private interpreter (no operator profile, no machine-wide resource)

The service must not run as `alber` and must not use anything machine-wide (no all-users Python in Program Files). It gets its own
CPython in `C:/dev/Powershell/srv/python` (relocatable python-build-standalone, release 20261026, asset
`cpython-3.11.18+20261026-x86_64-pc-windows-msvc-install_only.tar.gz`, sha256 `75cb501dc11c7b55c6b447c58e4910bce5b1f609f1017328b4f3a858dfaf5428`),
created by `Provision-ServicePython.ps1 -Archive <downloaded file>` (hash-verified, packages from `requirements-service.txt`, no PATH/registry
changes). Torch IS installed there, exactly the operator's build (2.5.1+cu124), because the Nodus service build (`nodus/build-service`, with Torch, against this interpreter) must match it; the service never uses the operator's torch under `C:/Users/alber`.
`instances.json` points each instance at that interpreter (`python` field); the service never reads `C:/Users/alber`.

## Consequences once Apache runs as the service

- `build/meowdoku` must be writable by the account (granted M) and `admin_token.txt` readable.
- Console-mode Apache (`Start-Apache.ps1`, run as alber) and the service both want port 443: stop one before starting the other.
- Files created by the service are owned by it; alber keeps access through its own inherited entries.
- Other operators' `.secret` files under `operator_credentials` inherit Modify from `build/`; consider cutting inheritance there.

## Running the setup

```powershell
Start-Process pwsh -Verb RunAs -ArgumentList '-NoExit','-File','C:\dev\Powershell\turing\examples\meowdoku\deploy\Setup-ServiceAccount.ps1'
```

Dry run first: `pwsh -File ...\Setup-ServiceAccount.ps1 -WhatIf`. Exits 2 if not elevated. Afterwards `Apache-Service.ps1 -Start | -Stop | -Restart | -Status`.

## Rollback

```powershell
sc.exe stop Apache2.4; sc.exe delete Apache2.4
icacls C:\Apache24 /remove svc-meowdoku /t      # likewise turing, nodus, srv
Remove-LocalUser svc-meowdoku                    # then delete the .windows.secret file
```

The two user rights are removed with the account; a stale SID entry can be dropped by re-running a secedit export/import without it.
