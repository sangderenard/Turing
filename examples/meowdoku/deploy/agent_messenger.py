"""Command-line client for Meowdoku messaging (stdlib only): how the agent accounts (claude, codex) talk through the server.

  python agent_messenger.py --base https://stichka.duckdns.org:8766 --user claude contacts
  ... add <username> | accept <username> | send <username> <text | -> | read <username> [--limit N]
  ... send <username> <text | -> [--type note|status|question|handoff|approval_request|answer] [--subject KIND:REF]
                                           KIND is lease|job|ticket|file|topic
  ... wait [--after ID | --from-cursor] [--timeout S]
                                           long-polls; prints incoming messages as JSON lines {id, from, to, created, body, subject, type};
                                           exit 0 on a message, 3 on timeout; --from-cursor starts after the server-kept ack cursor
  ... pending                              messages to me past my server-kept cursor, as JSON lines (does not advance the cursor)
  ... ack <id>                             advance my server-kept cursor to <id> (the sender never sees it)
  ... inbox                                unread summary (per contact, plus pending requests)

The password is read from --password-file, else <state>/secrets/operator_credentials/<user>.secret (state = --state-dir, else
MEOWDOKU_STATE_DIR).  It is sent only to POST /api/account/login and never printed.  TLS is verified with the system store; the one
exception is --insecure-test-ca <cafile>, which trusts that single CA file (for a test CA) instead.  Messaging is server-mediated and
operator-readable. Sends resolve a relationship handle over HTTPS and never include the recipient's name in the send payload.
Use --idempotency-key with identical content to retry an uncertain send. Exit codes: 0 ok, 1 error, 3 wait timed out.
"""
from __future__ import annotations

import argparse
import http.cookiejar
import http.client
import json
import math
import os
import secrets
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


class ClientError(Exception):
    def __init__(self, message, status=None, uncertain=False):
        super().__init__(message)
        self.status, self.uncertain = status, uncertain


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ClientError("API redirect refused; use the verified HTTPS origin directly")


class Client:
    def __init__(self, base: str, user: str, password: str, context: ssl.SSLContext | None = None):
        self.base = base.rstrip("/")
        parsed = urllib.parse.urlsplit(self.base)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ClientError("--base must be an HTTPS origin without credentials, query or fragment")
        self.user = user
        handlers = [urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()), _NoRedirect()]
        if self.base.startswith("https://"):
            handlers.append(urllib.request.HTTPSHandler(context=context or ssl.create_default_context()))
        self.opener = urllib.request.build_opener(*handlers)
        self._login(password)

    def call(self, path: str, body=None, timeout: float = 40):
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(self.base + path, data=data, method="GET" if body is None else "POST",
                                         headers={"Content-Type": "application/json"} if body is not None else {})
        try:
            with self.opener.open(request, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            try:
                message = json.loads(error.read()).get("error", "")
            except (ValueError, AttributeError, http.client.HTTPException, OSError):
                message = ""
            raise ClientError(f"HTTP {error.code} {path.split('?')[0]}: {message}", status=error.code,
                              uncertain=error.code >= 500) from None
        except (ValueError, UnicodeError):
            raise ClientError("invalid JSON response; request outcome may be uncertain", uncertain=True) from None
        except http.client.HTTPException:
            raise ClientError("incomplete HTTP response; request outcome may be uncertain", uncertain=True) from None
        except (urllib.error.URLError, OSError) as error:
            raise ClientError(f"cannot reach {self.base}: {getattr(error, 'reason', error)}", uncertain=True) from None

    def _login(self, password: str) -> None:
        self.call("/api/account/login", {"username": self.user, "password": password})


def read_secret(user: str, password_file: str | None, state_dir: str | None) -> str:
    if password_file:
        path = password_file
    else:
        state = state_dir or os.environ.get("MEOWDOKU_STATE_DIR")
        if not state:
            raise ClientError("give --password-file, or --state-dir / MEOWDOKU_STATE_DIR")
        path = os.path.join(state, "secrets", "operator_credentials", f"{user}.secret")
    try:
        with open(path, encoding="utf-8") as handle:
            secret = handle.read().strip()
    except OSError as error:
        raise ClientError(f"cannot read the secret file {path}: {error.strerror}") from None
    if not secret:
        raise ClientError(f"the secret file {path} is empty")
    return secret


def _emit(message: dict) -> None:
    print(json.dumps({k: message[k] for k in ("id", "from", "to", "created", "body", "subject", "type")}, ensure_ascii=False), flush=True)


def run(client: Client, args) -> int:
    cmd = args.command
    q = urllib.parse.quote
    if cmd == "contacts":
        print(json.dumps(client.call("/api/messages/contacts"), ensure_ascii=False))
    elif cmd in ("add", "accept"):
        action = "request" if cmd == "add" else "accept"
        print(json.dumps(client.call("/api/messages/contacts", {"username": args.username, "action": action})))
    elif cmd == "send":
        text = sys.stdin.read() if args.text == "-" else args.text
        payload = {"body": text, "idempotency_key": getattr(args, "idempotency_key", None) or secrets.token_urlsafe(24)}
        if args.type:
            payload["type"] = args.type
        if args.subject:
            kind, sep, ref = args.subject.partition(":")
            if not sep:
                raise ClientError("--subject must be KIND:REF")
            payload["subject"] = {"kind": kind, "ref": ref}
        # Recipient selection stays local. A fresh short-lived handle carries the
        # relationship, and the same operation key survives a lost HTTP response.
        uncertain_send = False
        for attempt in range(2):
            sending_request = False
            try:
                response = client.call("/api/messages/conversations")
                if (not isinstance(response, dict) or not isinstance(response.get("contacts"), list)
                        or any(not isinstance(c, dict) or not isinstance(c.get("username"), str)
                               or not isinstance(c.get("conversation_handle"), str) or not c["conversation_handle"]
                               for c in response["contacts"])):
                    raise ClientError("invalid conversations response; request outcome may be uncertain", uncertain=True)
                conversations = response["contacts"]
                peer = next((c for c in conversations if c["username"].casefold() == args.username.strip().casefold()), None)
                if peer is None:
                    raise ClientError("you cannot message this user")
                payload["conversation_handle"] = peer["conversation_handle"]
                sending_request = True
                result = client.call("/api/messages/send", payload)
                if (not isinstance(result, dict) or result.get("ok") is not True
                        or type(result.get("id")) is not int or result["id"] <= 0
                        or type(result.get("created")) not in (int, float) or not math.isfinite(result["created"])):
                    raise ClientError("invalid send receipt; request outcome may be uncertain", uncertain=True)
                print(json.dumps(result))
                break
            except ClientError as error:
                uncertain_send = uncertain_send or (sending_request and error.uncertain)
                if attempt == 0 and (error.uncertain or error.status == 403):
                    continue
                if uncertain_send or error.uncertain:
                    raise ClientError(f"send outcome uncertain; retry with --idempotency-key {payload['idempotency_key']} and the identical recipient and text",
                                      uncertain=True) from None
                raise
    elif cmd == "pending":
        for message in client.call("/api/messages/inbox")["messages"]:
            _emit(message)
    elif cmd == "ack":
        print(json.dumps(client.call("/api/messages/ack", {"up_to": args.id})))
    elif cmd == "read":
        messages = client.call(f"/api/messages/thread?peer={q(args.username)}")["messages"][-args.limit:]
        for message in messages:
            _emit(message)
        if messages:
            client.call("/api/messages/read", {"peer": args.username, "up_to": messages[-1]["id"]})
    elif cmd == "inbox":
        data = client.call("/api/messages/contacts")
        print(json.dumps({"unread": {c["username"]: c["unread"] for c in data["contacts"] if c["unread"]},
                          "requests": [r["username"] for r in data["requests"]]}))
    elif cmd == "wait":
        if args.from_cursor:
            after = client.call("/api/messages/inbox")["cursor"]
        else:
            after = args.after if args.after is not None else client.call("/api/messages/since?after=-1")["latest"]
        deadline = time.monotonic() + args.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return 3
            data = client.call(f"/api/messages/since?after={after}&wait={min(25, max(1, int(remaining)))}", timeout=40)
            fresh = data["messages"]
            if fresh:
                after = max(m["id"] for m in fresh)
                incoming = [m for m in fresh if m["from"].lower() != client.user.lower()]
                for message in incoming:
                    _emit(message)
                if incoming:
                    return 0
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Meowdoku messaging client (server-mediated, operator-readable)")
    p.add_argument("--base", required=True)
    p.add_argument("--user", required=True)
    p.add_argument("--password-file")
    p.add_argument("--state-dir")
    p.add_argument("--insecure-test-ca", metavar="CAFILE", help="trust only this CA file (a test CA) instead of the system store")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("contacts")
    sub.add_parser("inbox")
    for name in ("add", "accept"):
        sub.add_parser(name).add_argument("username")
    s = sub.add_parser("send")
    s.add_argument("username")
    s.add_argument("text", help="the message, or - to read it from stdin")
    s.add_argument("--type", choices=("note", "status", "question", "handoff", "approval_request", "answer"))
    s.add_argument("--subject", metavar="KIND:REF")
    s.add_argument("--idempotency-key", help="reuse the reported operation key when retrying an uncertain send with identical content")
    sub.add_parser("pending")
    sub.add_parser("ack").add_argument("id", type=int)
    r = sub.add_parser("read")
    r.add_argument("username")
    r.add_argument("--limit", type=int, default=50)
    w = sub.add_parser("wait")
    w.add_argument("--after", type=int)
    w.add_argument("--from-cursor", action="store_true", help="start after the server-kept ack cursor instead of the latest message")
    w.add_argument("--timeout", type=float, default=60.0)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        context = ssl.create_default_context(cafile=args.insecure_test_ca) if args.insecure_test_ca else None
        client = Client(args.base, args.user, read_secret(args.user, args.password_file, args.state_dir), context)
        return run(client, args)
    except ClientError as error:
        print(f"agent_messenger: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
