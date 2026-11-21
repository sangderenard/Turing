"""The admin's view of accounts: every account with its authority, and the SpeakToMe tools that authority gives it.

Read-only. Granting and removing a permission group stays with the existing routes, POST /api/account/authority/grant
(dev, operator, nodus, execution) and POST /api/account/admin/grant|revoke (admin), which check and audit it. Tools follow the
groups: an account's tools are the catalog entries its capabilities allow, the same rule /api/index answers its own
caller with and the speaktome MCP turns into tools (an entry declared as a tool, sent as JSON, allowed).

  GET /api/account/admin/people                       every account: role row, authority (permission_groups)
  GET /api/account/admin/tools?username=NAME          that account's capabilities and tools, from this request's origin

An account's capabilities depend on where it connects from (the origin tier), so the tools are worked out as if the
account connected from where this request comes from; the answer says which origin that was.
"""
from __future__ import annotations

import sqlite3
from http import HTTPStatus
from urllib.parse import parse_qs

from meowdoku import authz as authz_mod
from meowdoku import permission_groups


def people(db_path) -> list:
    conn = authz_mod._connect(db_path)                      # makes account_roles exist
    try:
        permission_groups.prepare(conn)
        rows = conn.execute("SELECT u.id, u.username, u.display_name, u.created_at, r.role, r.granted_by "
                            "FROM users u LEFT JOIN account_roles r ON r.user_id = u.id "
                            "LEFT JOIN temporary_accounts t ON t.user_id=u.id "
                            "WHERE t.user_id IS NULL ORDER BY u.username COLLATE NOCASE").fetchall()
        from meowdoku import agent_execution
        return [{"id": uid, "username": name, "display_name": shown, "created_at": created, "role_row": role,
                 "role_granted_by": granted, "server_agent": agent_execution.server_owned(db_path, name),
                 "authority": permission_groups.authority(conn, uid)}
                for uid, name, shown, created, role, granted in rows]
    finally:
        conn.close()


def needs(endpoint: dict) -> str:
    """What an entry needs, read as the server's own refusal reads it: capability is one name or a list, and
    capability_all says whether every one of them is needed (else any one)."""
    need = endpoint.get("capability")
    names = [need] if isinstance(need, str) else list(need or ())
    if not names:
        return "an account" if endpoint.get("account", True) else "nothing"
    return (" and " if endpoint.get("capability_all") else " or ").join(names)


def capabilities_for(db_path, account: dict, *, origin: str, origin_tier: str, policy: dict, lift: dict):
    """(principal, capabilities) of an account ({"id", "username"}) as if it connected from this origin: the shader
    lift as the server applies it, then authz's own rules. The People page and @mentions both ask this."""
    tier = origin_tier
    if origin_tier == "shader":                             # the server's own rule: a signed-in account lifts the shader lock
        tier = lift.get(authz_mod.role_of(db_path, account["id"]), "shader")
    principal = authz_mod.principal_from(account, origin, tier, db_path)
    return principal, frozenset(authz_mod.capabilities(principal, policy))


def tools_for(db_path, username: str, *, origin: str, origin_tier: str, policy: dict, catalog, lift: dict) -> dict:
    conn = sqlite3.connect(str(db_path), timeout=10)
    try:
        row = conn.execute("SELECT id, username FROM users WHERE username = ? COLLATE NOCASE", (username,)).fetchone()
    finally:
        conn.close()
    if row is None:
        raise LookupError("account not found")
    account = {"id": row[0], "username": row[1]}
    principal, caps = capabilities_for(db_path, account, origin=origin, origin_tier=origin_tier, policy=policy, lift=lift)
    tier = principal.tier
    kiosk = authz_mod.kiosk(principal)
    index = catalog.index(sorted(caps), True, None, kiosk, False)
    tools, not_offered = [], []
    for endpoint in index["endpoints"]:
        if not endpoint.get("tool") or endpoint.get("request", "json") != "json":
            continue
        entry = {"name": endpoint["name"], "group": endpoint["group"], "summary": endpoint["summary"],
                 "changes": bool(endpoint.get("changes")), "needs": needs(endpoint)}
        (tools if endpoint.get("allowed") else not_offered).append(entry)
    return {"username": account["username"], "role": principal.role, "groups": sorted(principal.groups),
            "origin": origin, "tier": tier, "kiosk": kiosk, "capabilities": sorted(caps),
            "tools": tools, "not_offered": not_offered,
            "note": "the speaktome MCP also gives every signed-in account its story verbs (look, enter, say, pending, ...)"}


def http_route(handler, path: str, *, accounts_db, policy: dict, catalog, lift: dict) -> None:
    """The caller has already passed account_manage."""
    handler._set_cookie = []
    if path == "/api/account/admin/people":
        return handler._json({"people": people(accounts_db)})
    query = parse_qs(handler.path.partition("?")[2])
    username = (query.get("username") or [""])[0]
    if not username:
        return handler._json({"error": "name an account with ?username="}, HTTPStatus.BAD_REQUEST)
    if catalog is None:
        return handler._json({"error": "this server has no SpeakToMe catalog"}, HTTPStatus.SERVICE_UNAVAILABLE)
    try:
        return handler._json(tools_for(accounts_db, username, origin=handler._origin, origin_tier=handler._origin_tier,
                                       policy=policy, catalog=catalog, lift=lift))
    except LookupError as error:
        return handler._json({"error": str(error)}, HTTPStatus.NOT_FOUND)
