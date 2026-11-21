# Agent training arena

`/agent-training-arena.html` is the operator page for comparing teams on one
real SpeakToMe ticket or Research case. The page is included by `web_export.py`
and the shared site menu. The page saves an operator-owned, durable queued run;
it does not start a model inside the HTTPS server.

## Task and scoring contract

- A ticket is a ticket-to-Research task. A Research case is a
  Research-to-Development task. The server resolves the source through the
  existing source authorization before it accepts a run.
- A run has one goal, a target agent, a scoring agent and two through 64 teams.
  Each team has one agent doing both tasks, or a prompt crafter and a separate
  communicator. The first writes the target's system prompt. The second sends
  the target's first message. Stateful defaults on in the page. One-agent
  teams keep their pipe open across the two tasks when stateful is on.
- The literal token `INSTRUCTIONS` in every outbound system prompt and message
  is expanded once to the exact user-provided arena specification body by the
  PowerShell coordinator. The inserted body is not recursively expanded.
- The target's answer and the effective prompt/message are retained with a
  scoring receipt. An independent scoring agent scores **each of the three
  pieces** on understandability, actionability, completeness, demonstrated
  experience and code elegance, from 0 to 4, with reasons. The displayed team
  scores are the rounded arithmetic mean per axis. The scoring agent lists only
  test evidence it can see in the answer. Its judgments are numeric but not
  proof that any code test ran; the receipt keeps the actual text for review.
- Runtime parameters and a training-mode flag are recorded with every run.
  They are declared metadata for analysis; the runner does not silently pass
  unsupported model settings to Codex or Claude, and it does not update model
  weights. A future trainer can select flagged receipts and verify them before
  using them as training data.

Completed receipts appear in Research when their ticket ID is among a case's
references, and in Development when the Research case ID matches. A reader
must be authorized to see the source. The source page lets people skim full
prompts, first messages, answers, per-piece scores, and reported tests.
Nothing in an arena answer is automatically promoted into the canonical
Research brief or Development work.

## Agent process boundary

The operator runs `deploy/Start-AgentArena.ps1 -Watch` from PowerShell 7 with
`SPEAKTOME_BASE`, `SPEAKTOME_USER` and `SPEAKTOME_SECRET_FILE` set for an
operator session. It uses the existing `AgentAccounts.psm1` credential and
`Start-AsAgent` launch path. Claude and Codex run under their existing
nonoperator Windows accounts. The coordinator holds each child's stdin/stdout
pipe open, uses the existing round-table adapters, and records a running-agent
marker so ordinary scheduled runs wait. No model output is executed as a
PowerShell command or site command.

The queue can be claimed by one operator session; claims carry a random token
and can be completed only with that token. The coordinator refreshes its claim
after each team. A dead claim is eligible for retry after four hours. A failed
team gets five zero scores and a recorded execution error, allowing the other
teams to finish. The queue is limited to 32 unfinished runs and a single run
to 64 teams to bound service work.

The existing Llama batch runner is not a held-open pipe and does not implement
system-prompt overrides. It is not offered as an arena participant until that
boundary exists. The arena currently supports the Codex and Claude process
adapters only. A visible score is a scored attempt, not a claim of real-world
goal completion or an automatically verified test result.

## Verification

`C:\dev\Powershell\srv\python\python.exe -m pytest
tests/test_meowdoku_agent_training_arena.py
tests/test_meowdoku_roundtable_seat.py -q` exercises queue ownership, claim
tokens, validation, the held-open child protocol and compatibility with the
existing seat adapter. The tests are offline and do not launch the live agent
accounts, export the site, restart the service or assert public availability.
