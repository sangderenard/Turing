"""Very small tools over the one identity authority (the accounts database): search the user list, represent one.

These are the bodies of the tasks the ``userlist`` table of the Nodus canvas performs. The table receives search
terms and its finished output is every identity that matched; another tool represents one identity. Nothing here
sends, grants or changes anything: both tools only read.

An identity is the account as other parts of the system may name it:

    {"identity": "account:7", "username": "ann", "display_name": "Ann", "role": "admin", "kind": "person"}

``kind`` is ``service`` for the accounts programs run under and ``person`` for everyone else (people and agents).
A search term is either ``key:value`` with a declared key, or plain text:

    role:admin        the account's role (guest is never an account; an account with no role row is a member)
    kind:person       person | service
    ann               the text appears in the username or the display name, in any letter case

Every term must match. No terms matches everyone. The output is ordered by account id.
The result is for the system's own use: it lists accounts, so it is never handed to an ordinary user.
"""
from __future__ import annotations

KEYS = ("role", "kind")
KINDS = ("person", "service")
MAX_TERMS, MAX_TERM = 8, 64


class IdentityError(ValueError):
    pass


def _identity(row) -> dict:
    role = row[3] or "member"
    return {"identity": f"account:{row[0]}", "username": row[1], "display_name": row[2] or row[1], "role": role,
            "kind": "service" if role == "service" else "person"}


def _all(db) -> list:
    rows = db.execute("SELECT u.id, u.username, u.display_name, r.role FROM users u "
                      "LEFT JOIN account_roles r ON r.user_id = u.id ORDER BY u.id").fetchall()
    return [_identity(tuple(row)) for row in rows]


def _terms(terms) -> list:
    if terms is None:
        return []
    if not isinstance(terms, (list, tuple)) or len(terms) > MAX_TERMS:
        raise IdentityError(f"terms must be a list of at most {MAX_TERMS}")
    parsed = []
    for term in terms:
        if not isinstance(term, str) or not term.strip() or len(term) > MAX_TERM:
            raise IdentityError(f"a term is text of 1 to {MAX_TERM} characters")
        key, colon, value = term.strip().partition(":")
        if colon and key in KEYS:
            if not value or (key == "kind" and value not in KINDS):
                raise IdentityError(f"bad term {term!r}")
            parsed.append((key, value))
        else:
            parsed.append(("text", term.strip().casefold()))
    return parsed


def search(db, terms=None) -> list:
    """Every identity that matches all the terms."""
    wanted = _terms(terms)

    def matches(identity) -> bool:
        for key, value in wanted:
            if key == "text":
                if value not in identity["username"].casefold() and value not in identity["display_name"].casefold():
                    return False
            elif identity[key] != value:
                return False
        return True
    return [identity for identity in _all(db) if matches(identity)]


def represent(db, name) -> dict | None:
    """One identity, by ``account:<id>`` or by username (any letter case); None when there is no such account."""
    if not isinstance(name, str) or not name.strip() or len(name) > 200:
        raise IdentityError("name one identity: account:<id> or a username")
    name = name.strip()
    for identity in _all(db):
        if identity["identity"] == name or identity["username"].casefold() == name.casefold():
            return identity
    return None
