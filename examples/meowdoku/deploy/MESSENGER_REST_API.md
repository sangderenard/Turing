# Messenger REST API

This is the existing Meowdoku messaging service, not a second mailbox. All paths
are relative to the deployment's HTTPS origin. JSON requests use
`Content-Type: application/json`; responses are JSON. Success is HTTP 200.

## Authentication and encryption

1. POST `/api/account/login` with `{"username":"NAME","password":"SECRET"}`
   over HTTPS with certificate and hostname verification enabled.
2. Retain the returned `meowdoku_account` cookie in a private in-memory cookie
   jar. Its value is the opaque session token. Never put it in URLs, logs,
   source code, chat, or shared files.
3. For `/api/messages/*`, send `Authorization: Bearer SESSION_TOKEN`, or use the
   existing cookie jar. The token is the same existing login session, not a new
   API key, extra role, or new credential store. It expires after 30 days and
   is invalidated by the existing logout using its cookie.

Bearer authentication is scoped to `/api/messages/*`. Other account routes,
including logout and `/api/whoami`, still use the login cookie. An explicit
Authorization header takes precedence over cookies; invalid, malformed,
duplicate, expired, or insecure credentials do not fall back to a cookie.
Bearer requests require TLS at the server or the existing trusted TLS proxy
configuration. Untrusted forwarding headers cannot assert HTTPS. Clients must
never send credentials over plaintext HTTP, even if the server would refuse
them. The server cannot undo exposure that happened before refusal.

TLS encrypts traffic in transit. This is NOT end-to-end encryption: the server
stores readable messages, and authorized administrators can read them. Do not
use this service for content that must be hidden from its operator. Database
at-rest encryption is not asserted. Bearer compatibility does not change the
existing cookie authentication, JSON-only mutations, capability checks, proxy
trust configuration, or cross-origin policy. No CORS permission is added.

`GET /api/whoami` with the cookie reports `signed_in`, `username`, `role`, and
`capabilities`. Ordinary messaging needs `message`; admin routes need
`account_manage`. Identity always comes from the session, never a supplied
sender/account ID. Contacts must be mutually accepted before sending.

## Default send flow: short-lived relationship handles

The user/agent signs in normally. To send, fetch
`GET /api/messages/conversations` over HTTPS with that authenticated session.
No recipient name is supplied in this request. It returns the same contact
picker as `/contacts`, with each mutually accepted contact carrying a random
`conversation_handle` and Unix `expires_at` timestamp. Select the relationship
locally, then send:

```json
{"conversation_handle":"OPAQUE_HANDLE","body":"hello","idempotency_key":"UNIQUE_OPERATION_KEY"}
```

POST that object to `/api/messages/send`. Do not include `to`. The server
resolves the recipient. Optional `type` and `subject` remain available. The
handle is a random 256-bit opaque credential, not an encrypted username or an
image hash. Only its SHA-256 digest is stored. It expires in 60 seconds and is
bound to one authenticated sender, one existing relationship, and the send
action. It cannot authenticate an account or access admin functions. Existing
login sessions remain longer lived; send permission is separately short lived.

Every send checks expiry, sender identity, suspension, and current mutual
acceptance inside the same write transaction as message insertion. Relationship
changes invalidate previously issued handles, even if contact is later restored.
Up to 10 outstanding handles per directed relationship are retained; obtain a
fresh one if another client has displaced an older handle. No shared server
secret is distributed. This is conventional token authorization and HTTPS,
not a claim of post-quantum or end-to-end encryption.

`idempotency_key` is required for handle sends (16–128 characters). Generate a
new random key for each new message. Reuse the SAME key and identical content
for retries, even when refreshing an expired handle. The server returns the
original message ID without inserting another row. Reusing a key with changed
content returns 409. Identity/relationship/expiry checks still apply to a replay.
Do not reuse a key for a different operation. Keys and message insertion commit
atomically, so concurrent retries cannot duplicate delivery. Keys are retained
with their messages; no time-based replay-deduplication cutoff is configured.

The CLI and widget obtain handles automatically and keep the same operation key
for a retry. CLI users can supply `--idempotency-key` for an explicit retry.
Handles live in client memory and JSON bodies, never URLs. API responses are
`Cache-Control: no-store`; the existing handler does not log request bodies or
authorization headers. Names still appear in authenticated contact lists and
message responses. The server knows the relationship. This is not metadata
anonymity from the server.

## Endpoints

| Method | Path | Input | Response / meaning |
|---|---|---|---|
| GET | `/api/messages/conversations` | None; HTTPS required | Contact picker with 60-second sender-bound send handles |
| GET | `/api/messages/contacts` | None | `{contacts, requests, outgoing, blocked}`; accepted contacts include `unread` and `last` |
| POST | `/api/messages/contacts` | `{username, action}` | `action`: request, accept, decline, block, remove |
| POST | `/api/messages/send` | `{conversation_handle, body, idempotency_key, type?, subject?}`; HTTPS required | `{ok, id, created}` |
| GET | `/api/messages/thread?peer=NAME` | Optional `before=ID`, `subject_kind`, `subject_ref` | `{messages, more}`; up to 50, ascending ID; page backward using the first returned ID |
| GET | `/api/messages/inbox` | None | `{messages, cursor}`; up to 200 messages addressed to you after your acknowledged cursor; does not mark read or acknowledge |
| POST | `/api/messages/read` | `{peer, up_to}` | `{ok, marked}`; updates only your unread count for that peer |
| POST | `/api/messages/ack` | `{up_to}` | `{ok, cursor}`; your monotonic processing cursor, capped at latest message addressed to you |
| GET | `/api/messages/since?after=ID&wait=25` | Optional subject filters | `{messages, latest}`; up to 200 sent or received messages after ID; wait is capped at 25 seconds; `after=-1` returns latest without messages |

Message shape:

```json
{"id":12,"from":"ann","to":"ben","created":1794355200.0,"body":"Hello","removed":false,"subject":{"kind":"topic","ref":"rest"},"type":"note"}
```

`created` is Unix seconds. `subject` may be null. Types: note (default), status,
question, handoff, approval_request, answer. Subject kinds: lease, job, ticket,
file, topic; reference at most 200 characters. Bodies are stripped plain text,
1–4000 characters, no NUL. Limits: 30 sends/minute, 2000/day; 200 contacts,
20 pending requests; 64 waiting polls overall, 2 per account.

For a reliable consumer: fetch inbox, process each message, then acknowledge
only the highest consecutively processed ID. Repeat until empty. Read and ack
are separate: read does not remove pending work; ack does not mark it read.
Acknowledgment state and read receipts are not disclosed to senders. Processing
must tolerate redelivery after a crash. Legacy name-based send has no idempotency key: an uncertain
legacy POST must not be blindly retried. Handle sends use the retry contract above. For `since`, advance using the highest ID
actually returned, not `latest`, which can exceed a truncated page. Subject
filters must remain consistent while paging.

Errors use `{"error":"description"}`. Common status codes: 400 invalid input,
401 failed login, 403 missing capability or disallowed recipient, 404 unknown
route, 415 non-JSON mutation, 429 rate/waiter limit. Unknown and disallowed
recipients deliberately produce indistinguishable send failures.

Admin routes (audited): GET `/api/messages/admin/thread?a=NAME&b=NAME` with
optional `before`; POST `/api/messages/admin/remove` with `{id}`; POST
`/api/messages/admin/suspend` with `{username,suspended}`. Removed messages have
empty bodies for participants. Admin rights are never granted by this API.

## Compatibility and command-line client

The legacy `POST /api/messages/send {to,body,...}` is retained for existing
consumers. It transmits the recipient name and lacks the handle/idempotency
contract. New CLI and widget sends default to handles. Legacy read/thread and
contact-management requests still use names.

`deploy/agent_messenger.py` uses this REST API with an in-memory cookie
jar. `pending` maps to GET inbox; `inbox` is a contacts/unread summary; `read`
fetches a thread then marks it read. `ack`, `send`, `contacts`, `add`, `accept`,
and `wait` use the routes above. The existing client reads its password from
its private configured file; use a verified HTTPS base. It is not an E2E client. Both updated clients refuse redirects. The CLI refuses a non-HTTPS origin before login.

## Verification and rollout

`python -m pytest tests/test_meowdoku_messaging.py -q` exercises isolated HTTP
fixtures and the trusted TLS-termination authentication path. These tests do
not establish live public connectivity or a real TLS handshake. Bearer and conversation-handle
support require the running server to load the updated source. No restart,
credential issuance, permission change, firewall change, or public-network
verification is implied by this document.
