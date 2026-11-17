# Private data: inventory and who reads it

Owner direction, 2026-11-17: private data is locked to operators on both servers (live and dev), and developers are
given no access to it on any server. A private value is hashed at the point where it is captured, before any display
can show it. The raw value is kept only in a record on the server.

## The rule in code

- **Tokens.** `private_record.py` replaces each IP address, email address and user agent with a token such as
  `ip~3f9a1c0b7d2e`. A token is HMAC-SHA256 under the state's own key, `<state>/secrets/private_hash.key`.
  - The same value gives the same token on one server, so developers can still see "the same visitor again".
  - The live and dev servers have different keys, so their tokens cannot be matched against each other.
  - A plain hash would not protect anything: every IPv4 address can be hashed in minutes.
- **Raw values.** The raw value goes into `<state>/sql/private_record.db` (`private_record` table) and nowhere else
  that is displayed.
- **The `private_read` capability.** Only `private_read` reads raw values back.
  - `authz.capabilities` gives it to the operator and admin roles, and removes it from everyone else.
  - Operator can come from an account role or a permission-group grant.
  - No origin tier, policy file, dev grant, modifier or guest ceiling confers it.
  - `tests/test_meowdoku_private_record.py` tries every tier against every role.
- **Resolving a token.** `GET /api/private/resolve?tokens=ip~...,email~...` requires `private_read` and HTTPS, and is
  never available to a view-only link. Every call prints a `[private]` line naming the account; that line holds no
  value.

## Inventory (2026-11-17)

| Data | Captured at | Kept | Shown by | After this change |
|---|---|---|---|---|
| Client IP on sign-in, sign-out and activity | server.py login, logout and the session lookup (`logring.note_login`, `note_activity`) | log ring and `log_ring.jsonl` | `GET /api/log` (dev_ui) | Ring lines hold tokens. Legacy persisted lines are redacted on load. |
| Anything the server prints (client errors, tickets, federation, library loads) | stdout tee | log ring, instance log files | `GET /api/log`, `/api/profile/diagnostics` | Every ring line, and stdout itself, passes through `redact`. |
| Request log lines (`log_message`) | stderr | instance log files | `/api/profile/diagnostics` (operator, admin) | The address and the line are tokens. |
| Client-error user agent | `POST /api/clientlog` | ring | `GET /api/log` | A `ua~` token. |
| Ticket submitter (`peer`), plus the ticket's 80 ring lines | `POST /api/tickets` | `content/tickets/*.json` | `GET /api/tickets` (dev_ui), web/tickets.js | `peer` is a token. Each string in a legacy ticket is redacted when read out. |
| Mathlab browser profile `user_agent_hint` | `POST /api/mathlab/.../profile` | job folder | job owner | A `ua~` token. |
| IP per account | the ring | the ring | `/api/account/admin/access`, `look-queue` (account_manage) | Tokens are resolved inline only for a `private_read` holder. |
| Failed and successful login attempts | `login_guard.record` | accounts.db `login_attempts`, pruned after 24 h | no endpoint | A server record only; unchanged. |
| Apache access lines | the supervisor's CustomLog | `logs/apache-access.log` | no endpoint on main | A server record. Codex's unlanded `codex/admin-encrypted-records-20261116` encrypts it at rest. |
| Invitee email | `POST /api/invitations` | accounts.db `site_invitations` | the inviter's own list | Owner only; unchanged. |
| Git author name and email | `/api/reviews/identity` | accounts.db `account_git_identity` | the account itself | Owner only; unchanged. |
| Message bodies | messaging | accounts.db | participants, and the admin thread (account_manage, audited) | Unchanged. |
| OAuth, session and peer secrets | oauth_mcp, accounts, authz | sha256 digests or session rows | never displayed | Unchanged. |

## Not covered yet

- The speaktome catalog does not offer `/api/private/resolve` as an MCP tool yet.
- Older instance log files, backups and the Apache access log keep the raw values they already hold. They are server
  records with no developer route to them.
- Text a person types into a free-form field (a ticket note, a message) is redacted when a ticket is read out. Message
  bodies are not redacted: they stay between their participants.
