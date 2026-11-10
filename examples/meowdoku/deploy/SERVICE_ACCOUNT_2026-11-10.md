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
  - RX, inherited: `C:/Apache24`, `C:/dev/Powershell/turing`, `C:/dev/Powershell/nodus`, the Python 3.11 directory (code and DLLs must load).
  - RX, not inherited: `C:/Users/alber`, `AppData`, `Local`, `Programs` - traversal to the Python directory only.
  - M, inherited: `C:/Apache24/logs`, `turing/build` (Meowdoku state, logs, accounts.db, tls), `nodus/out`.
  - R on `build/meowdoku/admin_token*.txt` and `tls/key.pem` (explicit; the service is added, alber's access is unchanged).

## Weak point

Python lives under alber's profile (`C:/Users/alber/AppData/Local/Programs/Python/Python311`), including the torch DLLs
used by Nodus. The service needs traversal of four profile folders plus RX on the whole Python tree. A machine-wide Python
install (for example `C:/Python311`) would remove those profile ACLs; do that later and update the paths in the deploy scripts and conf.

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
icacls C:\Apache24 /remove svc-meowdoku /t      # likewise turing, nodus, the Python dir
icacls C:\Users\alber /remove svc-meowdoku      # and AppData, AppData\Local, ...\Programs
Remove-LocalUser svc-meowdoku                    # then delete the .windows.secret file
```

The two user rights are removed with the account; a stale SID entry can be dropped by re-running a secedit export/import without it.
