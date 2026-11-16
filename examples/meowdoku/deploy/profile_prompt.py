"""Resolve operator-owned individual guidance for the trusted agent-day launcher.

The launcher reads accounts.db before changing Windows account. Nonoperator
agents receive only the resolved text, never access to the operator workspace.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from meowdoku import agent_workspace  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--accounts-db", type=Path, required=True)
    parser.add_argument("--individual", required=True)
    parser.add_argument("--profile-id")
    parser.add_argument("--job")
    parser.add_argument("--ensure-memory")
    args = parser.parse_args(argv)
    if not args.accounts_db.is_file():
        print("accounts database missing", file=sys.stderr)
        return 3
    db = agent_workspace.connect(args.accounts_db)
    try:
        if args.ensure_memory:
            account = db.execute("SELECT u.id FROM users u LEFT JOIN temporary_accounts t ON t.user_id=u.id "
                                 "WHERE u.username=? COLLATE NOCASE AND t.user_id IS NULL",
                                 (args.individual,)).fetchone()
            if account is None:
                print("permanent individual account not found", file=sys.stderr)
                return 1
            try:
                agent_workspace.memory_read(db, account[0], args.ensure_memory)
            except agent_workspace.WorkspaceError as error:
                if error.status != 404:
                    print(str(error), file=sys.stderr)
                    return 1
                try:
                    agent_workspace.memory_write(db, account[0], {"id": args.ensure_memory, "create": True,
                                                                   "title": args.individual + " memory", "body": ""})
                except agent_workspace.WorkspaceError as problem:
                    print(str(problem), file=sys.stderr)
                    return 1
            return 0
        author = db.execute("SELECT user_id FROM account_roles WHERE role IN ('admin','operator') "
                            "ORDER BY CASE role WHEN 'admin' THEN 0 ELSE 1 END,user_id LIMIT 1").fetchone()
        if author:
            agent_workspace.seed_builtin_profiles(db, author[0])
        if args.profile_id:
            row = db.execute("SELECT p.id FROM agent_profiles p JOIN users u ON u.id=p.individual_id "
                             "WHERE p.id=? AND u.username=? COLLATE NOCASE",
                             (args.profile_id, args.individual)).fetchone()
        else:
            row = db.execute("SELECT p.id FROM agent_profiles p JOIN users u ON u.id=p.individual_id "
                             "WHERE u.username=? COLLATE NOCASE ORDER BY (p.seed_key IS NULL),p.updated LIMIT 1",
                             (args.individual,)).fetchone()
        if row is None:
            return 2
        try:
            print(agent_workspace.resolve_profile(db, row[0], args.job)["preamble"])
        except agent_workspace.WorkspaceError as error:
            print(str(error), file=sys.stderr)
            return 1
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
