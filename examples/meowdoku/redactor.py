"""[REDACT] The dev stack's accounts database is an export made by this redactor, never the live file.

Owner, 2026-11-17: "a redactor is of utmost importance, the exported database should be just as secure as on live,
and log data must be aliased." Ticket 1ffa389bd91b4288aecda3b8c168d2cf; Development case 8bbe0de76497481fbb0b86cc107b13f8.

Everything in the export is readable by the dev account, and therefore by any developer's branch code. So the rule is:
the export holds nothing that a developer could not read on live.

- Every table has a declared rule (``RULES``). A table without one stops the export, so new data never leaks by default.
- Log values are aliased with the DEV state's own key (``private_record`` tokens): addresses, emails and agents in log
  columns and free text. A value that is already a live token is aliased again, so a dev alias never matches a live one.
  Nothing is recorded that would let anyone resolve a dev alias back.
- Bearer secrets are dropped: sessions, OAuth codes and tokens, invitations, recovery codes, peer grants, shares.
- Content that is private on live is withheld. Its row stays, so features still work, but the text is replaced:
  direct messages, issue tickets, unpublished articles and posts, agent memories and workspace documents, user files
  and their keys.
- Password hashes are kept only for the accounts that use the dev side: admin, operator, service, and the dev
  permission group. Every other hash is replaced by a value that never verifies.
- The copy is a consistent snapshot (SQLite's backup API). It is redacted with ``secure_delete`` on, then VACUUMed, so
  that no freed page keeps an old value. Its integrity is checked, and it is renamed into place whole.

    python redactor.py audit  --from <db>                     the tables without a rule (exit 1 when any)
    python redactor.py export --from <live db> --to <dev db> --key <dev state>/secrets/private_hash.key
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meowdoku import private_record  # noqa: E402

WITHHELD = "[withheld in the dev export]"
NEVER_VERIFIES = "withheld-in-dev-export"              # no "$": accounts._verify_password answers False
HASH_ROLES = ("admin", "operator", "service")
HASH_GROUPS = ("dev",)
KEEP, DROP = "keep", "drop"

# Column rules: "ip" / "email" / "ua" alias the value; "text" aliases every address and token inside free text; "json"
# does the same inside every string of a JSON value; "withhold" replaces it with WITHHELD; "null" empties it.
RULES = {
    # ---- identity and roles: kept (the one identity authority), except what is secret
    "users": {"password_hash": "hash"},
    "temporary_accounts": KEEP, "account_roles": KEEP, "account_permission_grants": KEEP, "account_permission_audit": KEEP,
    "account_capability_modifiers": KEEP, "account_capability_modifier_audit": KEEP, "account_prefs": KEEP,
    "account_instances": KEEP, "account_nodus_grants": KEEP, "alpha_agreements": KEEP, "orientation_state": KEEP,
    "account_git_identity": {"name": "withhold", "email": "email"},
    "account_content_access": KEEP, "account_content_groups": KEEP, "account_content_invitations": {"request_message": "withhold"},
    "account_content_members": KEEP, "account_content_audit": {"detail": "text"},
    "script_access": KEEP, "script_audit": {"host": "ip"},
    "owned_assets": KEEP,
    # ---- bearer secrets and private files: dropped
    "sessions": DROP, "oauth_mcp_clients": DROP, "oauth_mcp_codes": DROP, "oauth_mcp_pending": DROP, "oauth_mcp_tokens": DROP,
    "preview_recovery": DROP, "site_invitations": DROP, "peer_grants": DROP, "msg_handles": DROP, "msg_send_keys": DROP,
    "connection_shares": DROP, "connection_share_calls": DROP, "chat_room_grants": DROP, "chat_grant_calls": DROP,
    "user_files": DROP, "user_file_keys": DROP, "msg_images": DROP, "relay_packages": DROP, "radio_plays": DROP,
    # ---- logs: aliased
    "login_attempts": {"client": "ip"},
    "checkout_calls": {"error": "text"}, "git_calls": {"arguments": "text"},
    "instance_requests": {"reason": "text", "detail": "text"},
    "branch_land_runs": {"summary": "text", "error": "text"},
    "service_jobs": {"label": "text"}, "service_instances": KEEP, "instance_leases": KEEP,
    "chat_audit": KEEP, "cm_audit": KEEP, "msg_audit": KEEP, "hosted_agent_mode_audit": KEEP,
    "research_events": {"before_json": "json", "after_json": "json"},
    # ---- messaging: the graph is kept, private bodies are withheld
    "msg_messages": {"body": "withhold"}, "msg_contacts": KEEP, "msg_cursor": KEEP, "msg_image_consent": KEEP,
    "msg_suspended": KEEP,
    # ---- rooms and comments: readable by every signed-in account on live, so kept and aliased
    "chat_messages": {"body": "text"}, "chat_reactions": KEEP, "chat_room_defs": KEEP, "chat_stickers": KEEP,
    "chat_table_marks": KEEP, "chat_tables": {"note": "text"}, "cm_comments": {"body": "text"},
    # ---- owner-private content: withheld
    "issue_tickets": {"title": "withhold", "description": "withhold", "diagnostics": "withhold"},
    "issue_ticket_events": {"note": "withhold"}, "issue_ticket_assignees": KEEP,
    "experience_reports": {"body": "withhold", "prompt_history": "withhold"},
    "experience_report_followups": {"text": "withhold"},
    "agent_documents": {"body": "withhold"}, "agent_memories": {"body": "withhold"}, "agent_scratch": {"body": "withhold"},
    "agent_profiles": KEEP, "agent_profile_jobs": KEEP, "agent_profile_imports": KEEP, "agent_instances": KEEP,
    "agent_runs": {"prompt": "withhold", "preamble": "withhold", "response": "withhold", "error": "text", "claim_token": "null"},
    "authored_pages": {"content": "withhold", "description": "withhold"},
    "guest_post_queue": {"body": "withhold"}, "guest_test_evidence": {"response": "withhold"},
    "guest_actor_kind": KEEP, "guest_qualification": KEEP,
    "nl_articles": {"key_hash": "never", "body": "unpublished"},
    "math_jobs": {"text": "text"},
    # ---- research: participants are every signed-in account on live
    "research_cases": {"goal": "text"}, "research_docs": {"body": "text"}, "research_affirmations": KEEP,
    "research_brief": KEEP, "research_surveys": KEEP, "research_proposals": KEEP, "research_ballots": KEEP,
    "research_eligible": KEEP,
    # ---- development record and system state: kept
    "branch_reviews": {"note": "text", "landed_detail": "text", "land_result": "text", "revoke_reason": "text"},
    "branch_review_remarks": {"note": "text"}, "branch_review_read": KEEP, "branch_owners": KEEP, "branch_removed": KEEP,
    "branch_merged_seen": KEEP, "checkouts": KEEP, "worktree_verdicts": KEEP,
    "divider_jobs": KEEP, "divider_pieces": KEEP, "divider_attempts": {"error": "text"},
    "lowering_blobs": KEEP, "lowering_cores": KEEP, "lowering_leases": KEEP, "lowerings": KEEP,
    "hosted_agent_last_word": KEEP, "hosted_dispatch_cursor": KEEP,
    "palette_feedback": KEEP, "palette_participants": KEEP,
    "radio_actions": KEEP, "radio_channel": KEEP, "radio_requests": KEEP, "radio_turns": KEEP, "radio_votes": KEEP,
    "suno_radio_assets": KEEP, "relay_addresses": KEEP,
    "schema_version": KEEP, "sqlite_sequence": KEEP,
}


class RedactError(Exception):
    pass


def _tables(db) -> list:
    return [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")]


def undeclared(db) -> list:
    return [t for t in _tables(db) if t not in RULES]


def realias(text, record):
    """Every live token becomes a dev alias of it; every raw address, email and agent inside becomes a dev alias."""
    if not isinstance(text, str) or not text:
        return text
    text = private_record.TOKEN_RE.sub(lambda m: record.token(m.group(0).split("~", 1)[0], m.group(0)), text)
    return record.redact(text)


def _realias_json(value, record):
    if isinstance(value, str):
        return realias(value, record)
    if isinstance(value, list):
        return [_realias_json(v, record) for v in value]
    if isinstance(value, dict):
        return {k: _realias_json(v, record) for k, v in value.items()}
    return value


def _value(kind, value, record, row):
    if value is None:
        return None
    if kind in private_record.KINDS:
        return record.token(kind, str(value))
    if kind == "text":
        return realias(str(value), record)
    if kind == "json":
        try:
            return json.dumps(_realias_json(json.loads(value), record))
        except (TypeError, ValueError):
            return realias(str(value), record)
    if kind == "withhold":
        return WITHHELD
    if kind == "never":
        return NEVER_VERIFIES
    if kind == "null":
        return None
    if kind == "unpublished":
        return value if row.get("status") == "published" else WITHHELD
    raise RedactError(f"unknown column rule {kind!r}")


def _hash_keepers(db) -> set:
    keep = set()
    tables = set(_tables(db))
    if "account_roles" in tables:
        keep |= {r[0] for r in db.execute(f"SELECT user_id FROM account_roles WHERE role IN ({','.join('?' * len(HASH_ROLES))})",
                                          HASH_ROLES)}
    if "account_permission_grants" in tables:
        keep |= {r[0] for r in db.execute(
            f"SELECT user_id FROM account_permission_grants WHERE group_name IN ({','.join('?' * len(HASH_GROUPS))})", HASH_GROUPS)}
    return keep


def redact(db, record, log=print) -> dict:
    """Apply RULES to an open copy. Refuses when a table has no rule."""
    missing = undeclared(db)
    if missing:
        raise RedactError("no redaction rule for: " + ", ".join(missing) + " (declare each in redactor.RULES)")
    keepers = _hash_keepers(db)
    counts = {}
    for table in _tables(db):
        rule = RULES[table]
        if rule == KEEP:
            continue
        if rule == DROP:
            counts[table] = db.execute(f'DELETE FROM "{table}"').rowcount
            continue
        cols = [r[1] for r in db.execute(f'PRAGMA table_info("{table}")')]
        wanted = {c: k for c, k in rule.items() if c in cols}
        if not wanted:
            continue
        changed = 0
        rows = db.execute(f'SELECT rowid, * FROM "{table}"').fetchall()
        for raw in rows:
            row = dict(zip(["rowid", *cols], raw))
            new = {}
            for col, kind in wanted.items():
                if kind == "hash":
                    value = row[col] if row.get("id") in keepers else NEVER_VERIFIES
                else:
                    value = _value(kind, row[col], record, row)
                if value != row[col]:
                    new[col] = value
            if new:
                db.execute(f'UPDATE "{table}" SET ' + ", ".join(f'"{c}" = ?' for c in new) + " WHERE rowid = ?",
                           (*new.values(), row["rowid"]))
                changed += 1
        counts[table] = changed
    return counts


def export(src, dst, *, key_path, log=print) -> dict:
    """Snapshot ``src`` (never written), redact the copy, VACUUM it, check it, and rename it into place at ``dst``."""
    src, dst = Path(src), Path(dst)
    if src.resolve() == dst.resolve():
        raise RedactError("the export must not replace its source")
    live_key = (src.resolve().parent.parent / "secrets" / "private_hash.key")       # <state>/sql/accounts.db -> <state>/secrets
    if Path(key_path).resolve() == live_key:
        raise RedactError("the alias key must be the dev state's own, never the source state's private_hash.key")
    record = private_record.Record(private_record.load_key(Path(key_path)))     # records nothing: no alias resolves back
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".redacting")
    for leftover in (tmp, Path(str(tmp) + "-journal"), Path(str(tmp) + "-wal"), Path(str(tmp) + "-shm")):
        leftover.unlink(missing_ok=True)
    source = sqlite3.connect(f"file:{src.as_posix()}?mode=ro", uri=True)
    copy = sqlite3.connect(tmp, isolation_level=None)
    try:
        source.backup(copy)
        copy.execute("PRAGMA journal_mode=DELETE")
        copy.execute("PRAGMA secure_delete=ON")
        copy.execute("BEGIN")
        counts = redact(copy, record, log=log)
        copy.execute("COMMIT")
        copy.execute("VACUUM")
        verdict = copy.execute("PRAGMA integrity_check").fetchone()[0]
        if verdict != "ok":
            raise RedactError(f"the redacted copy failed its integrity check: {verdict}")
    except BaseException:
        copy.close(); source.close()
        tmp.unlink(missing_ok=True)
        raise
    copy.close(); source.close()
    os.replace(tmp, dst)
    log(f"[redact] {dst}: " + ", ".join(f"{t} {n}" for t, n in sorted(counts.items()) if n) )
    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit")
    a.add_argument("--from", dest="src", required=True)
    e = sub.add_parser("export")
    e.add_argument("--from", dest="src", required=True)
    e.add_argument("--to", dest="dst", required=True)
    e.add_argument("--key", required=True, help="the DEV state's private_hash.key (made when missing; never the live one)")
    args = parser.parse_args(argv)
    if args.cmd == "audit":
        db = sqlite3.connect(f"file:{Path(args.src).as_posix()}?mode=ro", uri=True)
        try:
            missing = undeclared(db)
        finally:
            db.close()
        print("every table has a rule" if not missing else "no rule for: " + ", ".join(missing))
        return 1 if missing else 0
    export(args.src, args.dst, key_path=args.key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
