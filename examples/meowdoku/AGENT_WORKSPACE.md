# Agent workspace and scratch storage

The operator workspace is a shared document tree for accounts whose effective
role is `operator` or `admin`. Every such account can create, read, edit and
delete every document. A document has a kind (`orientation`, `prompt`,
`personality`, `guidance`, `trouble`, `scenario`), an optional parent, a title,
and a body. Children are listed without their bodies, so an agent can inspect
an arbitrary personality or trouble tree only as far as the present task needs.
System prompts should point at the index and orientation. The MCP initialize
message tells authorized operators to start with `workspace_index`; it does
not inject every document into the context.

Scenarios carry an ordered list of document IDs. The list is consumed in the
author's broad-to-specific order. References must exist and are protected
against deletion; authors decide specificity. No document body is visible to
members, trial accounts, anonymous visitors, or service accounts. The browser
page `/agent-workspace.html` has separate document kinds, tree navigation,
a center editor that expands to full screen, and a private scratch pane.

`GET /api/agent-workspace` returns only roots and quotas. `GET
/api/agent-workspace/children?parent=<id>` returns a branch; `GET
/api/agent-workspace/document/<id>` returns one body. `POST
/api/agent-workspace/documents` saves or deletes. All writes require
`X-Meowdoku-Workspace: 1`. These routes require HTTPS and the
`operator_workspace` capability, then recheck the exact role. The matching
SpeakToMe catalog exposes them as MCP tools only to those roles.

`GET /api/scratch`, `GET /api/scratch/file?name=<name>`, and `POST
/api/scratch` are account-owned text files. Every permanent account, including
an intern, may use them through REST and MCP. They are separate from encrypted
user-file attachments and cannot be shared through messages. Temporary preview
accounts use an HttpOnly, Secure, SameSite=Lax cookie, bound to their preview
account ID, with a 2 KiB JSON budget. The MCP internal client keeps that
cookie only for its preview session. Keeping a preview account does not copy
cookie scratch into durable storage; the user can save it afterward.
Workspace and durable scratch text are stored in the accounts database and
are not encrypted at rest by this change. Do not place secrets there; the
existing encrypted user-file feature remains the place for client-encrypted
attachments.

Defaults follow the existing 128-file account convention for scratch. Text
files are limited to 128 KiB each and 8 MiB total, keeping an MCP read below
its response limit. `MEOWDOKU_SCRATCH_FILES`, `MEOWDOKU_SCRATCH_FILE_BYTES`,
and `MEOWDOKU_SCRATCH_BYTES` set bounded service defaults. Operators get 32
prompt documents each; `MEOWDOKU_OPERATOR_PROMPTS` changes that limit. The
whole tree defaults to 512 documents of at most 64 KiB each; the corresponding
settings are `MEOWDOKU_AGENT_DOCUMENTS` and
`MEOWDOKU_AGENT_DOCUMENT_BYTES`. The index reports the effective limits.

The new tables live in the configured accounts database through
`agent_workspace` schema version 1. The normal reviewed Turing and SpeakToMe
landing process must land both repositories before an instance restart and
web export. The source page alone is not the published `build/meowdoku/web`.
