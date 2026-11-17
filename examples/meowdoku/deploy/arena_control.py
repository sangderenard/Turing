"""Small REST bridge for the operator's PowerShell arena coordinator."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "speaktome"))
from speaktome_mcp.rest import Rest  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("claim", "heartbeat", "complete", "source"))
    parser.add_argument("--kind", choices=("ticket", "research"))
    parser.add_argument("--id")
    args = parser.parse_args(argv)
    rest = Rest.from_env()
    headers = {"X-Meowdoku-Arena": "1"}
    if args.operation == "claim":
        value = rest.call("/api/agent-training-arena/claim", {}, headers=headers)
    elif args.operation == "heartbeat":
        value = rest.call("/api/agent-training-arena/heartbeat", json.load(sys.stdin), headers=headers)
    elif args.operation == "complete":
        value = rest.call("/api/agent-training-arena/complete", json.load(sys.stdin), headers=headers)
    else:
        if not args.id or not args.kind:
            parser.error("source requires --kind and --id")
        from urllib.parse import quote
        path = ("/api/issues/" if args.kind == "ticket" else "/api/research/cases/") + quote(args.id, safe="")
        value = rest.call(path)
    print(json.dumps(value, ensure_ascii=True))


if __name__ == "__main__":
    main()
