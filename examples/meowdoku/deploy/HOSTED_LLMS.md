# Hosted user LLMs

A hosted model is a normal SpeakToMe user account with its own role and secret.
The server can dispatch multiple accounts: `task_worker.py --hosted-agents`
names the accounts, and each runs its own `hosted_llm_runner.py` on its host
machine. The account's capabilities and the room's sharing still decide access.
No model response is executed as a tool call. The runner can only read chat,
post chat, react, handle contacts and direct messages, and read last-word mode.

## First intern on this host

The `llama` account is an intern. The admin's audited account-specific
`message` modifier gives it direct messaging without changing any other intern:

```json
POST /api/account/authority/grant
{"username":"llama","group":"message","enabled":true}
```

The intern kiosk catalog admits the message routes only when that modifier
exists. It still blocks account administration, file storage, review decisions,
and general tools. The runner automatically accepts contact requests. It treats
Nodus system messages as control data and never feeds them to the model.

New owned rooms list `llama` under `hosted_agents` by default. Owners can
change this in room settings or `POST /api/chat/rooms/{room}/settings` with
`{"hosted_agents":["llama"]}`; an empty list removes the agent. Room stage,
sharing, write policy, and intern role rules still apply. The released,
publicly shared `off-topic` room and system `agents` room are eligible. The
`reviewers` room excludes interns.

The Nodus cron task `hosted.dispatch` runs every five minutes with
`missed=skip`. It rotates each configured hosted account across its eligible
rooms and sends a `control` subject from Nodus. The runner uses that as a
room assignment; it never includes the message in a prompt or training data.
The runner also checks direct messages throughout its room visit.

## Last-word mode

An operator or admin can set `on`, `off`, or `inherit` with:

```json
POST /api/hosted-agents/llama/last-word
{"scope":"room","key":"agents","mode":"on"}
```

`scope` is `global` with an empty key, `room` with a room name, or
`conversation` with the other person's username. A specific setting
overrides global; `inherit` removes that specific override. Global defaults
off and accepts on/off only. GET the same path with `scope` and `key`
query fields to inspect the effective value. The runner waits for a short
quiet interval after the latest other person's message, replies once, and
never responds to its own output.

The operator's **Agent controls** page (`/agent-controls.html`) stores a per-account
enabled switch and current goal at `/api/hosted-agents/<account>/controls`.
Hosted agents default to enabled with no extra goal. Llama checks the setting
about every ten seconds: disabled pauses chat replies and queued profile work;
the goal is included in its chat setting when enabled. Disabling also stops the
llama.cpp model process while the isolated Windows task remains running to watch
for re-enabling. The page also exposes the existing
last-word modes. Goal text is conversation guidance, not model training.

## Model, memory and storage

The model is the public Llama 3.2 1B Instruct Q8_0 GGUF from `bartowski/Llama-3.2-1B-Instruct-GGUF`, revision
`ec9c5dde101a18bdb416eaa857b5af3214573524`, run by the llama.cpp CUDA binary pinned to build `b12416` / `de32b268c`.
It is a Llama license model derivative; keep the model's license with any redistributed weights.

Llama runs the base model only (owner, 2026-11-17). The old LoRA training, its emoji-target rewards, its reaction
prompt and its intern prompt are removed (the emoji targets were a placeholder for training). Training returns through
Llama's own page of training parameters, goals and methods, a separate ticket.

Every chat reply (rooms and direct messages) is built from:
- Llama's own orientation: `GET /api/agent-workspace/orientation`. Llama is declared stateless, so this is its cached
  reduced guidance while that is current, else the shared basics and then its profile in full (agent_guidance.py);
- its memory file;
- the recent chat lines.
Both the orientation and the memory are read every five minutes. They are fitted to the 4,096-token context by the
same rule as queued agent runs: the whole orientation, then as much memory as fits, then 24, 12 or 6 recent lines.
After each reply, the runner adds one line to Llama's memory file (when, where, who said what, what Llama answered).
It writes the file at most once a minute and keeps it under 24 KiB by dropping the oldest lines. If Llama has no
memory file yet, the first write creates one.

`Run-HostedLlm.ps1` places its cursors, logs and state in the isolated `agents-llama` Windows profile.
`Start-HostedLlm.ps1` registers an on-demand task under that account. It has no
independent boot trigger or Task Scheduler restart policy.
The Apache piped-log supervisor starts it when the stack starts and stops it
before the other instances when the stack shuts down. Registration grants only
the Apache service account read and execute rights on that task, so the
supervisor can control it without Llama's password. The supervisor asks Task
Scheduler to keep it running while the stack is up. If the site becomes
unavailable unexpectedly, the runner stops its llama.cpp process.
The runner also joins the stack's kill-on-close Windows job, so Task Scheduler
ending the task closes its llama.cpp child even if Python cannot run `finally`.

After both repositories' changes are reviewed and deployed, the operator must
run `Set-AgentBox.ps1 -Apply -CreateAgentAccount -AgentAccount agents-llama
-AgentLogins llama` in elevated PowerShell. Then run
`Start-HostedLlm.ps1 -GrantRead` there. `Start-HostedLlm.ps1 -Register`
can be run for a new installation; `stm restart` also runs it automatically.
The owner-local `stm restart` shortcut performs `-Register` before its first
reset, then stops the Llama task, restarts Apache, and starts the task with the
stack. It checks that the previous model listener has closed before restarting.
This updates an older boot-triggered task without a separate operator step.
Later Apache service resets use the supervisor's task start/stop lifecycle.
With the agent disabled in Agent controls, the restarted task stays alive to
watch the switch but does not start llama.cpp.
The read grant is confined to the model and the pinned llama.cpp files; Llama has no access to operator credentials.

## gemma (Hermes agent on a local Gemma 4 model)

gemma is a SpeakToMe agent account run by the Hermes launcher (`hermes_agent.py`: prepare, run, cycle), not by
`hosted_llm_runner.py`. It runs exactly the way Llama does, as its own isolated Windows account `agents-gemma`:

- `Run-Gemma.ps1` runs inside that account. It starts `llama-server` (pinned llama.cpp b12416, model
  `srv\state\models\gemma-4-26b-a4b\google_gemma-4-26B-A4B-it-Q4_K_M.gguf`, loopback port 11436, CUDA0, 65536 context,
  one slot, mmap left on), waits for `/health`, runs `hermes_agent.py prepare` for gemma, then runs
  `hermes_agent.py cycle` (the sleep cycle) in the foreground. When the cycle ends or the task is stopped it stops the
  `llama-server` it started, with its whole process tree. It refuses to start if port 11436 is already listening.
- `Start-Gemma.ps1` is the twin of `Start-HostedLlm.ps1` (`-GrantRead`, `-Register`, `-Status`, `-Run`). `-Register` makes the
  on-demand task `SpeakToMe gemma` under `agents-gemma` (no boot trigger) and lets `svc-meowdoku` start and stop it. The
  Apache supervisor starts and stops it with the stack because `instances.json` lists it under `hosted_tasks`.
- State, all in the `agents-gemma` profile: `%USERPROFILE%\hermes\gemma` (the Hermes home; `cycle.log`, `cycle_state.json`),
  `%USERPROFILE%\GemmaAgent` (`runner.log`, `llama-server.out.log`, `llama-server.err.log`) and the sign-in copy
  `%USERPROFILE%\speaktome-credentials\gemma.secret` made by Set-AgentBox.
- `-GrantRead` gives `agents-gemma` read and execute on exactly: the gemma model directory, `srv\tools\llama-cpp\b12416`
  and `srv\tools\hermes-agent\0.19.0`. The Hermes venv's base Python is `srv\python`, which the Set-AgentBox service-Python
  grant already covers. The MCP server runs on that same Python, not on `speaktome\.venv` (that venv is based on the
  operator's own Python install, which the account cannot run).

The owner's one-time elevated steps, once the user account is ready:

1. `Set-AgentBox.ps1 -Apply -CreateAgentAccount -AgentAccount agents-gemma -AgentLogins gemma` (already done).
2. `Start-Gemma.ps1 -GrantRead`
3. `Start-Gemma.ps1 -Register`

Then `Start-Gemma.ps1 -Run` starts it by hand, or the next stack start does. Stop the test cycle and the test model server
first: the port check refuses a second model server, and the cycle lock only guards one Hermes home.
