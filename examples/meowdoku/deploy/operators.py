"""Operator accounts for the deploy scripts: bootstrap and rotate.  Prints only usernames and file paths, never a secret.

  python operators.py bootstrap --db build/meowdoku/accounts.db --users admin,codex,claude,guest_agent \\
                                --secrets-dir build/meowdoku/operator_credentials
  python operators.py rotate <username> --db ... --secrets-dir ...
  python operators.py bootstrap --state-dir C:/dev/Powershell/srv/state     (db = <state>/sql/accounts.db, secrets = <state>/secrets/operator_credentials)

bootstrap: each missing account is created with a fresh ``secrets.token_urlsafe(32)`` password written ONLY to
``<secrets-dir>/<username>.secret`` (an existing secret file is never overwritten; the account is then left as it is).
``admin`` gets role admin (authz.grant_admin, granted_by ``operator:bootstrap``); every other name gets role ``operator``; all are
added to ``script_access``.  The secrets dir is restricted to the current user with icacls (best effort, reported).
"""
from __future__ import annotations

import argparse
import getpass
import os
import secrets
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [p for p in (str(ROOT), str(ROOT / "examples")) if p not in sys.path]

from meowdoku import accounts as accounts_mod        # noqa: E402
from meowdoku import authz as authz_mod              # noqa: E402
from meowdoku.state_paths import StatePaths          # noqa: E402

BOOTSTRAP = "operator:bootstrap"


def _restrict(directory: Path, readers: tuple = ()) -> str:
    """Cut inheritance and leave the current user full control; ``readers`` (account names or *SIDs) may also read."""
    if os.name != "nt":
        os.chmod(directory, 0o700)
        return f"{directory}: chmod 700"
    try:
        grants = [f"{getpass.getuser()}:(OI)(CI)F"] + [f"{reader}:(OI)(CI)R" for reader in readers]
        r = subprocess.run(["icacls", str(directory), "/inheritance:r", "/grant:r", *grants],
                           capture_output=True, text=True, timeout=30)
        return f"{directory}: icacls " + ("restricted to the current user" if r.returncode == 0 else f"failed (exit {r.returncode})")
    except (OSError, subprocess.SubprocessError) as error:
        return f"{directory}: icacls unavailable ({type(error).__name__})"


def _write_secret(path: Path, secret: str) -> None:
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(secret)


def bootstrap(db: Path, users: list, secrets_dir: Path) -> list:
    report = []
    secrets_dir.mkdir(parents=True, exist_ok=True)
    report.append(_restrict(secrets_dir))
    accounts = accounts_mod.Accounts(db)
    for name in users:
        file = secrets_dir / f"{name}.secret"
        exists = accounts._conn.execute("SELECT 1 FROM users WHERE username = ? COLLATE NOCASE", (name,)).fetchone()
        if file.exists():
            report.append(f"{name}: secret file exists, kept ({file})")
            if not exists:      # the file is the secret: an account missing for it is created with that password, never overwritten
                accounts.ensure_user(name, file.read_text(encoding="utf-8").strip())
        else:
            secret = secrets.token_urlsafe(32)
            _write_secret(file, secret)
            if exists:
                accounts.set_password(name, secret)     # an account with no secret file: its secret is defined now
            else:
                accounts.ensure_user(name, secret)
            report.append(f"{name}: secret written to {file}")
        if name == "admin":
            authz_mod.grant_admin(db, name, BOOTSTRAP)
        else:
            authz_mod.grant_role(db, name, "operator", BOOTSTRAP)
        authz_mod.script_grant(db, name, BOOTSTRAP)
        report.append(f"{name}: role {authz_mod.role_of(db, accounts._conn.execute('SELECT id FROM users WHERE username = ? COLLATE NOCASE', (name,)).fetchone()[0])}, script access granted")
    return report


def rotate(db: Path, username: str, secrets_dir: Path) -> list:
    accounts = accounts_mod.Accounts(db)
    secrets_dir.mkdir(parents=True, exist_ok=True)
    secret = secrets.token_urlsafe(32)
    accounts.set_password(username, secret)
    file = secrets_dir / f"{username}.secret"
    file.write_text(secret, encoding="utf-8")
    return [f"{username}: new secret written to {file}"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="operators.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("bootstrap")
    p.add_argument("--db")
    p.add_argument("--users", default="admin,codex,claude,guest_agent")
    p.add_argument("--secrets-dir")
    p.add_argument("--state-dir", help="the state root: replaces --db and --secrets-dir")
    p = sub.add_parser("rotate")
    p.add_argument("username")
    p.add_argument("--db")
    p.add_argument("--secrets-dir")
    p.add_argument("--state-dir", help="the state root: replaces --db and --secrets-dir")
    args = ap.parse_args(argv)
    if args.state_dir:
        if args.db or args.secrets_dir:
            ap.error("--state-dir replaces --db and --secrets-dir; give either, not both")
        paths = StatePaths(Path(args.state_dir))
        args.db, args.secrets_dir = str(paths.accounts_db), str(paths.operator_credentials_dir)
        paths.sql_dir.mkdir(parents=True, exist_ok=True)
    elif not (args.db and args.secrets_dir):
        ap.error("give --state-dir, or both --db and --secrets-dir")
    try:
        if args.cmd == "bootstrap":
            lines = bootstrap(Path(args.db), [u.strip() for u in args.users.split(",") if u.strip()], Path(args.secrets_dir))
        else:
            lines = rotate(Path(args.db), args.username, Path(args.secrets_dir))
    except (accounts_mod.AccountError, authz_mod.AuthzError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
