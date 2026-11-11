"""Walk the live Nodus canvas as a graph cursor (stdlib only), over the REST the site already has.

  python agent_walker.py --base https://HOST:8766 --user claude enter [--instance NAME] [--at IDENTITY]
  ... where                      the cursor's text, then its exits:  N. <relationship> <direction> -> <name> [<identity>]
  ... go N                       take exit N of the list `where` prints (resolved against the exits fetched now)
  ... go --to IDENTITY [--relationship R --direction D]
  ... look [--radius K]          names around the cursor by graph distance (POST /api/living-data/query)
  ... leave                      close the cursor
  --json                         print the raw server response instead

Verbs and endpoints: enter = GET /api/nodus/map then POST /api/living-data/sessions; where = GET /api/living-data/sessions/{id};
go = POST .../{id}/move; look = POST /api/living-data/query; leave = POST .../{id}/close.  The cursor is graph position only: nothing on the
canvas is invoked, held or changed.  One cursor per account; it does not survive a server restart.

Signs in exactly as agent_messenger.py does (its Client and read_secret, imported); the secret is never printed.  The cursor id and the map
are kept in a small file under the user's temp directory keyed by base URL and user (--state-file overrides), so successive commands
continue one walk.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import ssl
import sys
import tempfile
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent_messenger import Client, ClientError, read_secret  # noqa: E402

SESSIONS = "/api/living-data/sessions"


def state_path(base: str, user: str, override: str | None = None) -> str:
    if override:
        return override
    key = hashlib.sha256(f"{base.rstrip('/')}\n{user}".encode()).hexdigest()[:16]
    return os.path.join(tempfile.gettempdir(), f"meowdoku-walker-{key}.json")


def load_state(path: str):
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) and data.get("session_id") else None
    except (OSError, ValueError):
        return None


def save_state(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle)


def drop_state(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def names_of(living_map: dict) -> dict:
    names = {}
    if isinstance(living_map, dict):
        names[living_map.get("identity")] = living_map.get("name") or living_map.get("identity")
        for region in living_map.get("regions") or []:
            names[region.get("identity")] = region.get("name") or region.get("identity")
    return names


def fetch_cursor(client: Client, session_id: str) -> dict:
    """The cursor with every exit page collected (64 per page)."""
    cursor = client.call(f"{SESSIONS}/{session_id}")
    while cursor.get("next_exit_offset") is not None:
        more = client.call(f"{SESSIONS}/{session_id}?offset={cursor['next_exit_offset']}")
        cursor["exits"] = cursor["exits"] + more["exits"]
        cursor["next_exit_offset"] = more.get("next_exit_offset")
    return cursor


def exit_line(number: int, item: dict, names: dict) -> str:
    return f"{number}. {item['relationship']} {item['direction']} -> {names.get(item['to'], item['to'])} [{item['to']}]"


def show_cursor(cursor: dict, names: dict) -> None:
    print(cursor.get("text", ""))
    for number, item in enumerate(cursor.get("exits", []), 1):
        print(exit_line(number, item, names))


def need_state(path: str) -> dict:
    state = load_state(path)
    if state is None:
        raise ClientError("no cursor of yours is recorded here; run `enter` first")
    return state


def run(client: Client, args, path: str) -> int:
    cmd = args.command
    if cmd == "enter":
        old = load_state(path)
        if old is not None:
            try:
                client.call(f"{SESSIONS}/{old['session_id']}")
            except ClientError as error:
                if error.status != 404:
                    raise
                drop_state(path)
            else:
                print("this account already has a cursor; run `leave` first (it was not closed)")
                return 1
        target = "/api/nodus/map" + ("?instance=" + urllib.parse.quote(args.instance) if args.instance else "")
        found = client.call(target)
        at = args.at or found["at"]
        try:
            cursor = client.call(SESSIONS, {"map": found["map"], "at": at, "request_id": "walk-" + secrets.token_urlsafe(16)})
        except ClientError as error:
            if error.status == 409 and "close your current graph session" in str(error):
                print("this account already has a cursor that is not recorded here; it cannot be closed from this state file", file=sys.stderr)
            raise
        save_state(path, {"session_id": cursor["session_id"], "map": found["map"], "instance": found.get("instance")})
        if args.json:
            print(json.dumps(cursor))
        else:
            show_cursor(cursor, names_of(found["map"]))
        return 0
    state = need_state(path)
    names, sid = names_of(state["map"]), state["session_id"]
    try:
        if cmd == "where":
            cursor = fetch_cursor(client, sid)
            print(json.dumps(cursor)) if args.json else show_cursor(cursor, names)
        elif cmd == "go":
            cursor = fetch_cursor(client, sid)
            exits = cursor["exits"]
            if args.number is not None:
                if not 1 <= args.number <= len(exits):
                    raise ClientError(f"there is no exit {args.number}; this position has {len(exits)}")
                chosen = exits[args.number - 1]
            elif args.to:
                match = [e for e in exits if e["to"] == args.to and args.relationship in (None, e["relationship"])
                         and args.direction in (None, e["direction"])]
                if len(match) != 1:
                    raise ClientError("no exit matches that identity" if not match else
                                      "several exits match; add --relationship and --direction")
                chosen = match[0]
            else:
                raise ClientError("give an exit number, or --to IDENTITY")
            move = {"to": chosen["to"], "relationship": chosen["relationship"], "direction": chosen["direction"],
                    "expected_revision": cursor["revision"], "request_id": "move-" + secrets.token_urlsafe(16)}
            try:
                after = client.call(f"{SESSIONS}/{sid}/move", move)
            except ClientError as error:
                if error.status == 409:
                    print(f"refused: {error}")
                    show_cursor(fetch_cursor(client, sid), names)
                    return 1
                raise
            print(json.dumps(after)) if args.json else show_cursor(after, names)
        elif cmd == "look":
            cursor = client.call(f"{SESSIONS}/{sid}")
            seen = client.call("/api/living-data/query", {"map": state["map"], "at": cursor["at"]["identity"], "radius": args.radius})
            if args.json:
                print(json.dumps(seen))
            else:
                here = seen["at"]
                print(f"0  {names.get(here['identity'], here['identity'])} [{here['identity']}]  (you are here)")
                for node in sorted(seen["nodes"], key=lambda n: n["distance"]):
                    print(f"{node['distance']}  {names.get(node['identity'], node['identity'])} [{node['identity']}]")
        elif cmd == "leave":
            cursor = client.call(f"{SESSIONS}/{sid}")
            done = client.call(f"{SESSIONS}/{sid}/close", {"expected_revision": cursor["revision"]})
            drop_state(path)
            print(json.dumps(done)) if args.json else print("left; the cursor is closed")
    except ClientError as error:
        if error.status == 404:
            drop_state(path)
            raise ClientError("the cursor is gone (closed, expired or the server restarted); run `enter`") from None
        raise
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Walk the live Nodus canvas as a graph cursor")
    p.add_argument("--base", required=True)
    p.add_argument("--user", required=True)
    p.add_argument("--password-file")
    p.add_argument("--state-dir")
    p.add_argument("--insecure-test-ca", metavar="CAFILE")
    p.add_argument("--state-file", help="where the walk is recorded (default: a file in the temp directory keyed by base and user)")
    p.add_argument("--json", action="store_true", help="print the raw response")
    sub = p.add_subparsers(dest="command", required=True)
    e = sub.add_parser("enter")
    e.add_argument("--instance")
    e.add_argument("--at", metavar="IDENTITY")
    sub.add_parser("where")
    g = sub.add_parser("go")
    g.add_argument("number", nargs="?", type=int)
    g.add_argument("--to", metavar="IDENTITY")
    g.add_argument("--relationship")
    g.add_argument("--direction", choices=("outgoing", "incoming"))
    lk = sub.add_parser("look")
    lk.add_argument("--radius", type=int, default=1)
    sub.add_parser("leave")
    return p


def main(argv=None, client: Client | None = None) -> int:
    """``client`` is injectable (a Client already signed in); otherwise sign in as agent_messenger.py does."""
    args = build_parser().parse_args(argv)
    try:
        if client is None:
            context = ssl.create_default_context(cafile=args.insecure_test_ca) if args.insecure_test_ca else None
            client = Client(args.base, args.user, read_secret(args.user, args.password_file, args.state_dir), context)
        return run(client, args, state_path(args.base, args.user, args.state_file))
    except ClientError as error:
        print(f"agent_walker: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
