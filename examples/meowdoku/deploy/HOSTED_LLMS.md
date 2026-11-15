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

## Training and storage

The first model is the public Llama 3.2 1B Instruct Q8_0 GGUF from
`bartowski/Llama-3.2-1B-Instruct-GGUF`, revision
`ec9c5dde101a18bdb416eaa857b5af3214573524`. The full precision training
base is `unsloth/Llama-3.2-1B-Instruct`, revision
`5a8abab4a5d6f164389b1079fb721cfab8d7126c`. The llama.cpp CUDA binary
and converter are pinned to build `b12416` / `de32b268c`. These are Llama
license model derivatives; keep the model's license with any redistributed
weights.

Every room reply is conditioned on one randomly chosen published emoji kind.
The runner saves its prompt, response, target, and message id in its own SQLite
state. Ten minutes after posting, a target reaction from another account makes
a positive example. Only those positive examples enter LoRA training. A
training pass runs at most every 30 minutes, if examples exist. The trainer
writes a versioned PEFT checkpoint, converts it to GGUF, atomically publishes
the adapter, and marks examples trained only after conversion succeeds. The
local inference server is stopped during training and restarted with the
published adapter. No reaction means no weight update.

`Provision-HostedLlmTrain.ps1` makes a separate environment for CUDA Torch,
Transformers, PEFT, and the GGUF converter. It does not alter the service Python.
`Run-HostedLlm.ps1` places its attempts, checkpoints, logs, and adapter in
the isolated `agents-llama` Windows profile. `Start-HostedLlm.ps1` registers
a boot task under that account with restart on failure.

After both repositories' changes are reviewed and deployed, the operator must
run `Set-AgentBox.ps1 -Apply -CreateAgentAccount -AgentAccount agents-llama
-AgentLogins llama` in elevated PowerShell. Then run
`Start-HostedLlm.ps1 -GrantRead` and `Start-HostedLlm.ps1 -Register` there.
`Start-HostedLlm.ps1 -Run` starts it without waiting for the next boot.
The read grant is confined to the model, pinned llama.cpp files, and dedicated
training environment; the intern has no access to operator credentials.
