"""Read a native execution trace (emit_ssa_to_c(trace=True)) and summarize it.

    python tools/read_native_trace.py <trace file> [--function SUBSTRING]
        [--grep SUBSTRING ...] [--first N] [--entries]

``--entries`` prints how many times each function was entered.  With
``--function`` the lines of the first activation of a matching function are
printed (``--first N`` limits them).  ``--grep`` prints every line whose
instruction text contains the substring (any function), in execution order.
"""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--function", default=None)
    parser.add_argument("--grep", action="append", default=[])
    parser.add_argument("--first", type=int, default=400)
    parser.add_argument("--entries", action="store_true")
    parser.add_argument("--activation", type=int, default=1,
                        help="which activation of --function to print (1-based)")
    args = parser.parse_args()
    lines = args.trace.read_text(encoding="utf-8", errors="replace").splitlines()
    print(f"{len(lines)} trace lines")
    if args.entries:
        counts: Counter = Counter()
        for line in lines:
            parts = line.split(" | ", 2)
            if len(parts) == 3 and parts[2].startswith("ENTER "):
                counts[parts[0]] += 1
        for name, count in counts.most_common():
            print(f"  {count:6d}  {name[-90:]}")
    if args.function:
        activation = 0
        printing = False
        printed = 0
        depth_owner = None
        for line in lines:
            parts = line.split(" | ", 2)
            if len(parts) != 3:
                continue
            owner, block, text = parts
            if text.startswith("ENTER ") and args.function in owner and "planned_region" not in owner:
                activation += 1
                if activation == args.activation:
                    printing = True
                    depth_owner = owner
                    print(f"=== activation {activation} of {owner[-80:]}")
            if printing and owner == depth_owner:
                print(f"  {block:>22s} | {text[:190]}")
                printed += 1
                if text.startswith("RET ") or printed >= args.first:
                    printing = False
                    if printed >= args.first:
                        print("  ... (truncated, raise --first)")
                    break
    for needle in args.grep:
        print(f"=== lines containing {needle!r}")
        shown = 0
        for line in lines:
            if needle in line:
                print("  " + line[-230:])
                shown += 1
                if shown >= args.first:
                    print("  ... (truncated)")
                    break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
