# Owner-run operator control

The web page `/operator-control.html` is available to permanent admin and operator accounts with `deploy_scripts`. Apache only queues named actions. A separate worker, running as the owner's Windows account, executes them from `examples/meowdoku/deploy`.

## First installation

After this change lands on Turing `main` and the live checkout is moved to that commit, run elevated PowerShell **as the Windows account that owns deployment**:

```powershell
cd C:\dev\Powershell\turing\examples\meowdoku\deploy
pwsh -File .\Admin-Control-Service.ps1 -Register -StateDir C:\dev\Powershell\srv\state
pwsh -File .\Admin-Control-Service.ps1 -Run -StateDir C:\dev\Powershell\srv\state
pwsh -File .\Admin-Control-Service.ps1 -Status
```

Registration prompts for that same Windows account's password and gives it the batch logon right. Task Scheduler holds the logon credential; the script does not save it in a file. The task starts at boot, runs at highest privilege, and restarts after a worker failure. If the Windows password changes, register again. `-Unregister` removes the task.

The task action points to `C:\dev\Powershell\srv\python\python.exe` and the live Turing worker source. Do not register from a review checkout: its path is temporary. The installer checks the service Python and account database paths before registration.

## Daily use

An admin creates an action by choosing one of the fixed deploy script names, a label, exact PowerShell arguments, and a timeout. The admin may issue an operator account a grant for one action for 1 minute through 30 days, or revoke it. The action button asks for the signed-in account's password each time. The worker rechecks the account's role, grant expiry, preset state, and script name immediately before execution. Disabling a preset revokes its grants. New runs are bounded to 20 queued jobs, with one queued request per account and action.

Each script executes as the owner with `MEOWDOKU_OPERATOR=admin` and the specified state directory. Existing script-auth gates therefore authenticate the owner's admin credential; the broker's separate audit database records the requesting site account. All script output is local under `<state>\secrets\admin-control-logs\<job id>.log`, never returned to the browser. Review a preset's arguments and its script's effects before granting it. The bootstrap `Admin-Control-Service.ps1` itself is excluded from the action catalogue because registration needs an interactive Windows credential prompt.

If a worker dies during a script, the job stays `running` and queued jobs stay held. Inspect the process and local log before manually reconciling the database; the worker does not replay a possibly partial deployment.

A site restart does not restart the worker. After a code update, stop and start the scheduled task to load the new worker code.

## External API agents (case 989e507e)

The admin-only "External agents" section declares an API-backed model as a SpeakToMe agent: name, provider
(`openai` for any OpenAI-compatible Chat Completions server, `anthropic`, `gemini`), base URL, model, SpeakToMe tools,
the minute of its hourly work run and its brief. Saving a declaration (`agent_declare`, password-checked like every
action) writes only `external_agents` in `<state>\sql\admin_control.db`; nothing on the computer changes and no key is
asked for. The declaration is not an action of the worker either.

The install is the owner's, from an elevated shell (an SSH session as the owner is one), with the command the page shows:

```
pwsh -File 'C:\dev\Powershell\turing\examples\meowdoku\deploy\Install-ExternalAgent.ps1' -Name <name>
```

`Install-ExternalAgent.ps1` does what was done by hand for Gemini on 2026-11-14: the SpeakToMe login with a random
secret in `<state>\secrets\operator_credentials\<name>.secret` and role member; the Windows account `agents-<name>`
through `Set-AgentBox.ps1 -Apply -CreateAgentAccount` (generated password, DPAPI for the owner); the API key typed at its
prompt and written as that account to its own `~\<name>.key` (DPAPI for the agent, like `~\gemini.key`); a key check as
the agent (`api_agent.py --check`); the character profile; then `Start-AgentDay.ps1 -Register`. The key passes to the
agent's process on standard input only and is never logged. `Install-ExternalAgent.ps1 -Name <name> -RotateKey`
replaces the key, `Start-AgentDay.ps1 -RunOne <name>` starts one work run, and the page's Withdraw (`agent_withdraw`)
stops scheduling it (its accounts and key stay; remove them by hand if wanted). The owner's own `stm` shortcut
(`stm agent install|key|run|withdraw <name>`) calls these same scripts, but it is the owner's untracked `~\stm.ps1`,
not part of this repository, so the page shows the script itself (issue 3b8a5c6f). The script is not in the action
catalogue because it prompts for the key.
