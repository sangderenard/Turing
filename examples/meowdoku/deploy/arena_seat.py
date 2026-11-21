"""Held-open, nonoperator agent pipe for an arena instance.

The operator coordinator sends one JSON object per line. The child starts the
existing Claude/Codex round-table adapter under its own Windows account, keeps
that process for stateful turns, and returns only text or a bounded error.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from roundtable_seat import Claude, Codex  # noqa: E402


def serve(agent_name, lines, write, factory=None):
    factory = factory or {"claude": lambda system: Claude(system_prompt=system),
                          "codex": lambda system: Codex(base_instructions=system)}[agent_name]
    agent = None
    system = None
    try:
        write({"op": "ready"})
        for raw in lines:
            try:
                command = json.loads(raw)
                op = command.get("op")
                if op == "stop":
                    break
                if op == "start":
                    if agent is not None:
                        agent.close()
                    system = command["system"]
                    if not isinstance(system, str) or not system.strip() or len(system.encode("utf-8")) > 32000:
                        raise ValueError("invalid system prompt")
                    agent = factory(system)
                    write({"op": "started"})
                elif op == "ask":
                    if agent is None:
                        raise ValueError("start an instance first")
                    text = command["text"]
                    seconds = command.get("seconds", 600)
                    if not isinstance(text, str) or len(text.encode("utf-8")) > 32000 or \
                            type(seconds) is not int or not 30 <= seconds <= 1800:
                        raise ValueError("invalid turn")
                    write({"op": "answer", "text": agent.ask(text, seconds)[:32000]})
                    if not command.get("stateful", True):
                        agent.close()
                        agent = None
                else:
                    raise ValueError("unknown arena operation")
            except Exception as error:
                write({"op": "error", "error": f"{type(error).__name__}: {error}"[:1000]})
    finally:
        if agent is not None:
            agent.close()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True, choices=("claude", "codex"))
    parser.add_argument("--login", required=True)
    parser.add_argument("--base", default="https://stichka.duckdns.org:8766")
    args = parser.parse_args(argv)
    speaktome = r"C:\dev\Powershell\speaktome"
    os.environ.update(SPEAKTOME_USER=args.login, SPEAKTOME_BASE=args.base,
                      SPEAKTOME_SECRET_FILE=os.path.join(os.path.expanduser("~"), "speaktome-credentials",
                                                         f"{args.login}.secret"),
                      PYTHONPATH=os.pathsep.join(p for p in (speaktome, os.environ.get("PYTHONPATH", "")) if p))
    def write(value):
        sys.stdout.write(json.dumps(value, ensure_ascii=True) + "\n")
        sys.stdout.flush()
    serve(args.agent, (line.decode("utf-8", "replace") for line in sys.stdin.buffer), write)


if __name__ == "__main__":
    main()
