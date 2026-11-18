"""A kept bell session: one claude or codex session that answers each new notice within seconds (case 46d6ad2d, stage 5).

A one-shot bell run starts its CLI cold and is handed the whole brief every time, so an answer takes minutes. For a
character whose session is declared resumable, Start-AgentDay.ps1 starts this instead, as the agent's own Windows account
(AgentAccounts.psm1 Start-AsAgent, hidden), when its bell first finds something new:

  python agent_session.py --agent claude --login claude --account agents --slot 00 --memory <bell.id> --why "..."

The agent is the round table's own adapter (roundtable_seat.Claude with its chat tools, roundtable_seat.Codex): one
process and one session. This process:
- briefs it in full once: its own orientation (GET /api/agent-workspace/orientation?full=1), these session rules and its
  memory file; again only on a new day (UTC) or when its documents change (the orientation's source_digest);
- otherwise hands it only what is new: it asks the bell's own question (pending, reviews_attention, the MCP's own rule)
  every POLL_SECONDS, and each new message or thread is one turn;
- keeps the character's session in SQL (POST /api/agent-workspace/session keep, every HEARTBEAT_SECONDS), so the
  character reads stateful while it lives, and drops it, with the error when one ended it;
- ends after IDLE_SECONDS without a notice; the next notice the tick finds starts a fresh session with a full brief.
A failed turn closes the session; the next notice starts a fresh one. Text from other accounts stays data.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from roundtable_seat import BASE, Claude, Codex, SeatError, log, sign_in  # noqa: E402

IDLE_SECONDS = 30 * 60
POLL_SECONDS = 10
HEARTBEAT_SECONDS = 60
RECHECK_SECONDS = 5 * 60                 # how often, at most, a notice first checks whether the documents changed
HEADER = {"X-Meowdoku-Workspace": "1"}
ORIENTATION = "/api/agent-workspace/orientation?full=1"
SESSION = "/api/agent-workspace/session"

RULES = (
    "This is your kept bell session (case 46d6ad2d, stage 5). You stay running, and each time something new waits for "
    "you (a message, an @mention of you, a review thread waiting for your answer) you are told so here, within seconds. "
    "Answer what woke you within {minutes} minutes, then end your turn with one line saying what you did. You are "
    "briefed in full only now, on a new day (UTC) or when your documents change; otherwise you are handed only what is "
    "new, so keep what matters in your memory file. A work run of yours may be going at the same time.")
MEMORY = ("Persistent memory ID: {memory}. At the start of this session, call memory_file with this ID. If it does not "
          "exist, create it with memory_change action write, id {memory}, create true and a title. Update that same "
          "memory file before each turn ends.")                     # Start-AgentDay.ps1 Invoke-AgentRun says the same
WORK = " Your work memory is {work}: read it with memory_file if you need it, but do not change it."


class ProbeError(Exception):
    pass


def probe(server):
    """What waits for the agent, asked as the agent: the bell's own rule (Start-AgentDay.ps1 $BellProbe)."""
    answer = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "pending", "arguments": {}}})
    result = answer.get("result") or {}
    try:
        pending = json.loads((result.get("content") or [{}])[0].get("text", "{}"))
    except ValueError:
        pending = {}
    if result.get("isError"):
        raise ProbeError(str(pending.get("error") or "pending refused"))
    messages = {f"m{m.get('id')}" for m in pending.get("untrusted", []) if isinstance(m, dict)}
    return messages | {"t%s:%s" % pair for pair in server._seen(server._attention())}


def notice(new) -> str:
    messages = sum(1 for item in new if item.startswith("m"))
    threads = len(new) - messages
    return (f"New since your last turn: {messages} message(s) and {threads} review thread(s) waiting for your answer. "
            "Handle them as your brief says (pending, reviews_attention), then end your turn with one line on what you did.")


class Session:
    def __init__(self, rest, look, start_agent, *, fill, memory, work_memory="", answer_seconds=240, idle=IDLE_SECONDS,
                 poll=POLL_SECONDS, heartbeat=HEARTBEAT_SECONDS, recheck=RECHECK_SECONDS, clock=time.monotonic,
                 sleep=time.sleep, today=lambda: time.strftime("%Y-%m-%d", time.gmtime())):
        self.rest, self.look, self.start_agent, self.fill = rest, look, start_agent, fill
        self.memory, self.work_memory, self.answer_seconds = memory, work_memory, answer_seconds
        self.idle, self.poll, self.heartbeat, self.recheck = idle, poll, heartbeat, recheck
        self.clock, self.sleep, self.today = clock, sleep, today
        self.digest = self.day = None
        self.checked = None
        self.turns = self.briefs = 0

    def _keep(self, action="keep", error=""):
        body = {"action": action, **({"error": error[:900]} if error else {})}
        return self.rest.call(SESSION, body, method="POST", headers=HEADER)

    def _orientation(self):
        got = self.rest.call(ORIENTATION)
        self.checked = self.clock()
        return self.fill(str(got.get("preamble") or "")), got.get("source_digest")

    def _brief(self, text=None, digest=None):
        if text is None:
            text, digest = self._orientation()
        self.digest, self.day, self.briefs = digest, self.today(), self.briefs + 1
        rules = RULES.format(minutes=max(1, int(self.answer_seconds // 60)))
        work = WORK.format(work=self.work_memory) if self.work_memory else ""
        return text.strip() + "\n\n" + rules + work + "\n\n" + MEMORY.format(memory=self.memory) + "\n\n"

    def _again(self):
        """The full brief again when the day turned or the documents changed, else nothing."""
        if self.today() != self.day:
            return "A new day began (UTC). Your brief again, in full:\n\n" + self._brief()
        if self.clock() - self.checked >= self.recheck:
            text, digest = self._orientation()
            if digest != self.digest:
                return "Your documents changed. Your brief again, in full:\n\n" + self._brief(text, digest)
        return ""

    def run(self, why: str) -> int:
        """Serve until IDLE_SECONDS pass without a notice. Returns the number of turns answered."""
        error = ""
        agent = None
        try:
            self._keep()
            seen = self.look()
            waiting = "You were woken because: " + why
            last = beat = self.clock()
            while True:
                if waiting:
                    try:
                        if agent is None:
                            agent = self.start_agent()
                            prefix = self._brief()
                        else:
                            prefix = self._again()
                        answer = agent.ask(prefix + waiting, self.answer_seconds)
                        self.turns += 1
                        log(f"turn {self.turns}: {answer.strip()[:300]}")
                    except SeatError as failed:                # a broken session is not kept; the next notice starts one
                        log(f"turn failed: {failed}")
                        if agent is not None:
                            agent.close()
                        agent = None
                    waiting, last = "", self.clock()
                if self.clock() - last >= self.idle:
                    log(f"idle {self.idle // 60} minutes: the session is dropped")
                    break
                self.sleep(self.poll)
                if self.clock() - beat >= self.heartbeat:
                    self._keep()
                    beat = self.clock()
                try:
                    current = self.look()
                except ProbeError as failed:
                    log(f"the bell's question failed: {failed}")
                    continue
                new, seen = current - seen, current
                if new:
                    waiting = notice(new)
        except Exception as failed:
            error = f"{type(failed).__name__}: {failed}"
            raise
        finally:
            if agent is not None:
                agent.close()
            try:
                self._keep("drop", error)
            except Exception as failed:                         # noqa: BLE001  reported, never hides the first error
                log(f"could not drop the session: {failed}")
        return self.turns


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--agent", required=True, choices=("claude", "codex"))
    parser.add_argument("--login", required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--slot", default="00")
    parser.add_argument("--memory", required=True)
    parser.add_argument("--work-memory", default="")
    parser.add_argument("--why", default="something new waits for you")
    parser.add_argument("--answer-minutes", type=float, default=4)
    parser.add_argument("--base", default=BASE)
    args = parser.parse_args(argv)
    sign_in(args.login, args.base)
    from speaktome_mcp.api import ApiServer
    from speaktome_mcp.rest import Rest
    rest = Rest.from_env()
    server = ApiServer(rest)
    start = (lambda: Claude(disallowed=())) if args.agent == "claude" else (lambda: Codex())

    def fill(text):                                       # the placeholders Start-AgentDay.ps1 fills for a run
        return (text.replace("{agent}", args.agent).replace("{login}", args.login).replace("{account}", args.account)
                .replace("{reason}", "bell session").replace("{now}", time.strftime("%Y-%m-%d %H:%M"))
                .replace("{slot}", args.slot))

    Session(rest, lambda: probe(server), start, fill=fill, memory=args.memory, work_memory=args.work_memory,
            answer_seconds=args.answer_minutes * 60).run(args.why)
    return 0


if __name__ == "__main__":
    sys.exit(main())
