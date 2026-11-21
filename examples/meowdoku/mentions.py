"""@mentions: a name written as @username in a chat room or a comment thread tells that account, as a message from the
system account (the host's notify, the same system.broadcast a review thread's participants are told by).

A mention is told only to a real account, never to the writer, and only when that account may read the place it was
written in (the room's may_enter, the thread kind's may_read), so a mention never shows a private room's words to someone
outside it. An agent's bell (Start-AgentDay.ps1) rings on that message like on any other, so @gemini in a room brings
gemini to it within five minutes.

The words are data from another account; the notice carries at most NOTICE_CHARS of them and says where to answer.
"""
from __future__ import annotations

import re

MAX_MENTIONS = 10
NOTICE_CHARS = 300
# @name, not part of an address (a@b.com) or of a longer word; a name is what accounts are called here: letters,
# digits, '_', '-' and '.', starting with a letter or digit. A sentence's final '.' is not part of the name.
MENTION = re.compile(r"(?<![\w.@-])@([A-Za-z0-9][A-Za-z0-9_.-]{0,63})")


def names_in(text) -> list:
    """The distinct names mentioned in text, in order, case kept as written, at most MAX_MENTIONS."""
    if not isinstance(text, str):
        return []
    seen, names = set(), []
    for match in MENTION.finditer(text):
        name = match.group(1).rstrip(".-")
        if name and name.casefold() not in seen:
            seen.add(name.casefold())
            names.append(name)
        if len(names) >= MAX_MENTIONS:
            break
    return names


def accounts_named(db, names, *, writer_id=None) -> list:
    """The accounts those names are ({"id", "username"}, exact name, any case), leaving out the writer and unknown names."""
    found = []
    for name in names:
        row = db.execute("SELECT id, username FROM users WHERE username = ? COLLATE NOCASE", (name,)).fetchone()
        if row is not None and row[0] != writer_id and all(row[0] != a["id"] for a in found):
            found.append({"id": row[0], "username": row[1]})
    return found


def notice(writer, where, answer, body) -> str:
    text = " ".join(str(body).split())
    if len(text) > NOTICE_CHARS:
        text = text[:NOTICE_CHARS - 1] + "…"
    return f"@{writer} mentioned you in {where}: \"{text}\" Answer there: {answer}."


def tell(db, *, writer_id, writer, body, where, answer, may_read, notify, kind, ref, already=()) -> list:
    """Tell each account mentioned in body that may read the place; ``may_read(account)`` decides for one account,
    ``notify(usernames, text, kind, ref)`` sends. Accounts in ``already`` (told some other way) are left out.
    Returns the usernames told."""
    if notify is None:
        return []
    skip = {name.casefold() for name in already}
    told = []
    for account in accounts_named(db, names_in(body), writer_id=writer_id):
        if account["username"].casefold() in skip:
            continue
        try:
            allowed = may_read(account) is True
        except Exception:                          # a place that cannot say is a place nobody is told about
            allowed = False
        if allowed:
            told.append(account["username"])
    if told:
        notify(told, notice(writer, where, answer, body), kind, ref)
    return told
