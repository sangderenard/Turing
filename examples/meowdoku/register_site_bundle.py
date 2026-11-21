"""Trusted builder handoff for an account-owned Turing site bundle.

The builder first writes a complete turing-program-bundle-v1 beneath the
workspace_id directory of the configured external workspace volume and records
the measured bytes in workspace_accounting. This command binds that immutable
bundle to its existing account owner, initially as a dev page. It never accepts
an owner id from the caller or creates a new compilation route.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meowdoku import authored_pages, compile_artifacts, workspace_accounting


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--accounts-db", type=Path, required=True)
    parser.add_argument("--workspace-id", required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.accounts_db.is_file():
        parser.error("Accounts database does not exist")
    db = authored_pages.connect(args.accounts_db)
    try:
        row = db.execute("SELECT w.owner_id,v.root,v.capacity_bytes,v.user_quota_bytes,v.free_floor_bytes "
                         "FROM workspace_leases w JOIN workspace_volumes v USING(volume_id) "
                         "WHERE w.workspace_id=?", (args.workspace_id,)).fetchone()
        if row is None:
            parser.error("workspace does not exist")
        root = compile_artifacts.external_directory(row["root"])
        directory = compile_artifacts.external_directory(root / args.workspace_id)
        if not directory.is_dir():
            parser.error("workspace directory does not exist")
        store = workspace_accounting.Store(args.accounts_db, root,
                    capacity_bytes=row["capacity_bytes"], user_quota_bytes=row["user_quota_bytes"],
                    free_floor_bytes=row["free_floor_bytes"])
        with store._transaction() as ledger:
            measured = compile_artifacts._regular_size(directory) + store._retained_bytes(ledger, args.workspace_id)
            store._record_live_usage(ledger, row["owner_id"], args.workspace_id, measured)
        result = authored_pages.register_bundle(db, row["owner_id"], args.workspace_id, args.bundle_dir)
    except (authored_pages.PageError, workspace_accounting.WorkspaceError, sqlite3.Error, OSError, ValueError) as error:
        parser.error(str(error))
    finally:
        db.close()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
