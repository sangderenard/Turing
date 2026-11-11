# Account-owned SQL issue tickets

Implemented locally, not deployed. This is a separate incident system, not an
adaptation of the immutable workspace/snapshot review queue.

## Source and authority

- `examples/meowdoku/issue_tickets.py` owns `issue_tickets` and
  `issue_ticket_events`, schema module `issue_tickets` version 1 in the existing
  Accounts database. The first authenticated operation registers this schema.
- The service binds the actual open Accounts database. Existing account sessions
  are rechecked inside the same transaction as each operation. No request can
  provide its owner, account role, database path or server diagnostic sequence.
- Owners can submit, read, investigate, resolve and reopen their own issues.
  Actual SQL administrators can read/update any issue and list all issues.
  Operator status, LAN access and `dev_ui` alone do not confer that access.
- Status changes require a note and the exact current version. They append an
  audit event atomically. Resolving keeps the report and history; it is not
  deletion, a merge approval, or evidence that a problem was actually fixed.

## HTTP

`server.py` dispatches these routes after the existing routing/access gates:

- `GET /api/issues?scope=mine|all&status=all|open|investigating|resolved&offset=N`
  returns at most 50 summaries and a next offset. `all` requires SQL admin.
- `POST /api/issues` accepts exactly `title`, `description`, `diagnostics`
  (optional object), and `client_key`. An unchanged owner/key retry returns the
  same ticket. Reusing a key for changed content returns 409.
- `GET /api/issues/<id>` returns a visible ticket and its latest 20 audit events.
  `?before_version=N` retrieves an older page. Other owners and absent tickets
  produce the same 404.
- `POST /api/issues/<id>/status` accepts exactly `status`, `note`, `version`.

Writes require a real account cookie, JSON and `X-Meowdoku-Issue: 1`. Browser
Origin and Fetch-Site checks reject cross-origin writes. Viewer requests are
refused. POST connections close explicitly; existing bounded body draining is
preserved. No CORS or anonymous-write exception is added.

Body limit is 100,000 bytes; diagnostics 80,000 encoded bytes, title 200
characters, description 12,000 and update note 4,000. Creation is limited to
10 tickets per account per ten minutes and updates to 30 events per account
per minute. These are abuse bounds, not a physical disk quota or retention
policy. No destructive retention operation is provided.

Server instance and log sequence are server-owned metadata. General server logs
are not copied into owner-visible issues, because they may contain other users'
activity. User-submitted game snapshots and browser logs are explicitly
disclosed by the game report form. Basic site reports attach only the page path
and browser details when the user leaves that option selected.

## Browser interface

`web/issues.html` and `web/issues.js` provide account-owned reporting, list
filters, status notes and paginated history. Stored content is rendered only as
text. Failed submissions retain the draft and an unchanged retry key. Status
updates use optimistic versions rather than silently overwriting another edit.

The shared `site-nav.js` banner includes Tickets and Report issue for signed-in
visitors. As of 2026-11-11, the shared Report issue button always opens the
basic SQL issue form. A frozen deployed game may still contain the legacy report
handler, so the header does not delegate to that handler. The updated source
game form itself submits to the SQL API and its ticket link opens `issues.html`. Shared banner opt-ins
were added to Marble Studio, jobs, loading Studio and log pages. Generic control
suite registration is documented separately in `SITE_CONTROL_SUITES.md`.

Legacy `tickets.py`, its stored diagnostic files and old API are not deleted or
migrated. Existing clients can still use them during rollout; all newly changed
game/banner entry points use the new account-owned incident system. Workspace
review tables and semantics are unchanged.

## Verification

61 focused pytest tests passed, plus 22 subtests, across:

- `tests/test_issue_tickets.py` (14 tests)
- `tests/test_meowdoku_server_authz.py`
- `tests/test_workspace_accounting.py`
- `tests/test_meowdoku_schema_version.py`

Coverage includes isolated ownership/admin, role revocation, logout/expiry,
schema forward-version refusal, idempotent retry, concurrent update conflicts,
atomic audit rollback, bounded history, malformed/CSRF requests and rejected
POST connection handling. JavaScript syntax and the separate Node control-suite
lifecycle harness pass. This is not the entire repository test suite.

Real Chrome against a disposable synthetic-account server verified creating and
resolving a test incident with retained history, the shared Report issue entry,
signed-out denial, cancel/reopen preserving a draft, account menu/Escape, and
390px layout. The test server and tab were shut down. No real issue was resolved.
Initial standalone headless launches failed with crashpad errors; connected
Chrome provided the successful visual/interaction check instead.

Independent security review's findings were addressed and rechecked. General
game physics, native renderer and full Studio-state capture were not retested.

## Rollout gate

No production DB has been migrated and no live server/assets were changed.
Publication must use the current authorized instance workflow, account identity
and schema backup/verification procedure. Coordinate the backend restart with
the server/auth owner: the shared working tree contains unrelated changes that
must not be deployed or committed inadvertently. Static files alone do not
activate the Python endpoint.

Do not insert a production issue directly into SQLite to evade a denied network
request. Once an authorized live route is available, authenticate through the
existing account flow, create the requested real incident through this API,
read it back and report its ID. Keep any transport-policy incident open until
the formerly blocked action is actually verified. SQL admin promotion is not
needed for the submitting account to manage its own issue.
