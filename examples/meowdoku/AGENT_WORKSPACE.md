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
`agent_workspace` schema version 2. The normal reviewed Turing and SpeakToMe
landing process must land both repositories before an instance restart and
web export. The source page alone is not the published `build/meowdoku/web`.

## Enduring characters, instances, and jobs

An individual profile has a stable ID, title, assigned permanent account and
ordered document IDs. Several profiles may share one account; the profile ID
distinguishes the characters. A job has a stable key and title under one
profile. It replaces selected base document IDs and appends additional IDs in
order, leaving the base untouched. The operator-only resolve route returns the
exact ordered documents and a joined Markdown preamble. Any operator or admin
can edit profiles, jobs, or the shared source documents. Documents referenced
by profiles or jobs cannot be deleted until the reference is removed.

`GET /api/agent-workspace/profiles` imports the existing short runner briefs
once for present `claude`, `codex`, `gemini`, and `llama` accounts. Each imported
profile has a stable seed key, so changing its title or documents does not
create a duplicate. The agent-day preamble and coaching text come from the
existing `Start-AgentDay.ps1`; roundtable guidance is a job-specific replacement
for that preamble. The imported brief copies can then be edited in the page.

An instance catalog entry has its own ID, title, character profile ID, and
attached memory file ID. Several instances can use one enduring character, but
one memory file attaches to only one instance at a time. Removing an instance
leaves its memory file intact. An operator can transfer an instance to another
character in one transaction; the attached memory keeps its ID and moves to
the destination account. Queued work blocks a transfer. A character with
catalogued instances cannot be deleted.

The trusted `Start-AgentDay.ps1` launcher resolves profiles from the accounts
database before spawning the nonoperator Windows account. Use `-RunOne claude`
or `-RunOne codex`, optionally `-InstanceId <id> -JobKey <key>`, to spawn a
catalogued character instance with its attached memory. `-ProfileId <id>` and
`-NewPersistence` can create a new instance; `-PersistenceId <id>` attaches an
existing, unassigned memory file. Scheduled runs reuse their default memory
ID and register the corresponding instance. A job key resolves its document
replacements and additions on top of the character for that run.
The profile text is handed to the agent as its preamble; the agent cannot read
the operator workspace endpoint. The roundtable runner also passes its resolved
`roundtable` job text through the existing seat command pipe on the first turn.
The old hardcoded launcher preamble is retained as migration source; a missing
character profile now stops the spawn instead of silently using it.

`GET /api/memory` lists durable account-owned memory files by stable 32-hex
IDs. `GET /api/memory/file/<id>` reads one file. `POST /api/memory` writes,
deletes or, for an operator/admin, transfers ownership to another permanent
account. The ID remains unchanged when transferred. Nonoperators can read and
write only their own files. A run's `-PersistenceId <id>` selects an existing
memory ID and `-NewPersistence` generates a fresh one. Ordinary scheduled runs
reuse `<AgentDays>/<agent>/persistence.id`. The launcher creates the memory
record before spawning and the preamble directs the agent to read and update
it. Storage uses the same
configured 128-file, 128-KiB-per-file, 8-MiB-per-account limits as scratch.
The memory text is separate from the profile and may be handed to a different
account by an operator. The memory and profile text remain unencrypted at rest.

The operator can queue a prompt for a catalogued Llama instance through
`POST /api/agent-runs`; `GET /api/agent-runs` and `GET /api/agent-runs/<id>`
show status and result. The Llama account alone may claim batches and complete
them, with a token that rejects stale completions. The queue holds at most 256
unfinished runs; a claim leases up to 32 for 15 minutes and recovers an
unfinished lease after that. The hosted runner submits multiple requests to
its one local `llama-server` process using continuous batching. Defaults are
two server slots and 16 runs per polling tick, configurable with
`--batch-slots` (1–8) and `--batch-limit` (1–32). This allows many logical
Llama instances to share one model process; throughput, context and VRAM remain
finite. The model output is stored as a run result, never executed as a site
command. The attached memory is read into the run as context; this batch path
does not automatically rewrite it.

Nodus's tensor state library can retain addressed numeric tensors in GPU
banks; it does not currently own llama.cpp model weights or KV cache state.
