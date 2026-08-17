"""Follow one buffer's identity through the whole call graph of emitted Fortran.

A compiled program's ABI is a chain of `call` statements, each binding actual
arguments to a callee's formal dummies by POSITION. Reading that chain by eye
across dozens of subroutines and hundreds of parameters is exactly the kind
of thing that should never be done twice by hand -- so this does it once,
mechanically, and prints the chain.

Given an entry point and a source name (``state.height``), it:

1. finds the entry's own parameter carrying that source name;
2. finds every `call` statement inside that entry's body;
3. for each call, checks whether the SAME actual token appears among the
   call's arguments, and at which position;
4. resolves that position against the CALLEE's own declared formal list to
   get the callee's name for it and its Fortran ``intent``;
5. records any array-element WRITE to that formal inside the callee's own
   body (not nested calls -- those are the next hop);
6. recurses into the callee, following the SAME actual identity onward.

The output is the whole hand-traceable chain in one table: which subroutine,
which formal, which intent, and whether that hop itself writes to it. A
buffer that is `input` at every hop and never locally written, yet still
changes value between load and store, is a defect this tool exists to catch
-- the write is happening somewhere this trace does NOT show, which narrows
the search to exactly the paths not covered (tensor-table copies, C-level
aliasing, or a call this script's regex missed).

    python tools/trace_fortran_alias.py build/fluid-trace/symbolic_fluid_frame_shell.f90 \
        --entry symbolic_fluid_control__symbolic_fluid_frame --source state.height
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HEADER = re.compile(
    r'^\s*subroutine\s+(\S+?)\((.*?)\)\s*bind\(C,\s*name="([^"]+)"\)\s*$'
)
CALL = re.compile(r'^\s*call\s+(\S+?)\((.*)\)\s*$')
DECL = re.compile(
    r'^\s*(real|integer|logical)\([^)]*\),\s*intent\((\w+)\)(?:,\s*value)?\s*'
    r':: (\w+)\b'
)
ARRAY_WRITE = re.compile(r'^\s*(\w+)\s*\(')


@dataclass(frozen=True)
class Subroutine:
    name: str
    formals: tuple[str, ...]
    body: tuple[str, ...]
    intents: dict  # formal name -> intent ('in' | 'out' | 'inout' | None)
    array_writes: frozenset  # formal names with a local element-write

    def find_call(self, callee_name_suffix: str) -> Optional[tuple]:
        for line in self.body:
            match = CALL.match(line)
            if match and match.group(1).endswith(callee_name_suffix):
                return match.group(1), [
                    argument.strip() for argument in match.group(2).split(",")
                ]
        return None


def _split_balanced(text: str) -> list[str]:
    """Split on top-level commas only, so nested ``foo(a, b)`` stays one item."""
    parts, depth, current = [], 0, []
    for character in text:
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        if character == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(character)
    if current:
        parts.append("".join(current))
    return [part.strip() for part in parts]


def parse_module(path: Path) -> dict[str, Subroutine]:
    lines = path.read_text(encoding="utf-8").splitlines()
    subs: dict[str, Subroutine] = {}
    index = 0
    while index < len(lines):
        match = HEADER.match(lines[index])
        if not match:
            index += 1
            continue
        native_name, formal_text, bind_name = match.groups()
        formals = tuple(_split_balanced(formal_text))
        start = index
        end = start + 1
        while end < len(lines) and not lines[end].strip().startswith(
            "end subroutine"
        ):
            end += 1
        body = tuple(lines[start:end])
        intents: dict[str, str | None] = {}
        for line in body:
            declared = DECL.match(line)
            if declared:
                intents[declared.group(3)] = declared.group(2)
        writes = frozenset(
            m.group(1) for line in body
            if (m := ARRAY_WRITE.match(line)) and "::" not in line
            and "=" in line and not line.strip().startswith(("if", "!"))
        )
        subs[bind_name] = Subroutine(bind_name, formals, body, intents, writes)
        subs.setdefault(native_name, subs[bind_name])
        index = end
    return subs


def trace(
    subs: dict[str, Subroutine],
    entry: str,
    token: str,
    *,
    depth: int = 0,
    seen: set | None = None,
) -> None:
    seen = seen if seen is not None else set()
    sub = subs.get(entry)
    if sub is None:
        print("  " * depth + f"(no subroutine named {entry!r} found)")
        return
    if entry in sub.formals:  # guard against pathological recursion
        pass
    if (entry, token) in seen:
        print("  " * depth + "(already visited this hop; stopping)")
        return
    seen.add((entry, token))

    if token in sub.formals:
        intent = sub.intents.get(token, "?")
        writes = "WRITES here" if token in sub.array_writes else "no local write"
        print("  " * depth + f"{sub.name}")
        print("  " * depth + f"    {token}: intent({intent}), {writes}")
    else:
        print(
            "  " * depth
            + f"{sub.name}: {token!r} is not a formal here (body-local)"
        )

    for line in sub.body:
        match = CALL.match(line)
        if not match:
            continue
        callee_bind_name, actuals = match.group(1), _split_balanced(
            match.group(2)
        )
        # The SAME actual can occupy MULTIPLE positions in one call --
        # nine "state.height" formals at a callee legitimately means the
        # identical caller buffer passed nine times, once per spatial
        # view. `.index()` finds only the first; using it here silently
        # dropped the other eight, which is exactly the kind of coverage
        # gap this tool exists to prevent.
        positions = [i for i, actual in enumerate(actuals) if actual == token]
        if not positions:
            continue
        callee = subs.get(callee_bind_name)
        if callee is None:
            print(
                "  " * (depth + 1)
                + f"-> call {callee_bind_name} (unresolved; not in this module)"
            )
            continue
        if len(positions) > 1:
            print(
                "  " * (depth + 1)
                + f"-> call {callee_bind_name}: {token!r} passed "
                f"{len(positions)} times, at positions {positions}"
            )
        for position in positions:
            if position >= len(callee.formals):
                print(
                    "  " * (depth + 1)
                    + f"-> call {callee_bind_name}: position {position} out "
                    f"of range ({len(callee.formals)} formals) -- "
                    "ARITY MISMATCH"
                )
                continue
            next_token = callee.formals[position]
            print(
                "  " * (depth + 1)
                + f"-> call {callee_bind_name} at argument {position} "
                f"(passes {token!r} as {next_token!r})"
            )
            trace(
                subs, callee_bind_name, next_token, depth=depth + 2,
                seen=seen,
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("fortran_file", type=Path)
    parser.add_argument("--entry", required=True)
    parser.add_argument(
        "--source",
        help="a dotted source name from the .api.yaml beside the .f90; "
        "resolved to the entry's own formal automatically",
    )
    parser.add_argument(
        "--token", help="skip source-name resolution; trace this exact "
        "formal/actual name from the entry directly",
    )
    arguments = parser.parse_args()

    subs = parse_module(arguments.fortran_file)
    token = arguments.token
    if token is None:
        if not arguments.source:
            raise SystemExit("pass --source or --token")
        import yaml

        api_path = arguments.fortran_file.with_suffix(".api.yaml")
        contract = yaml.safe_load(api_path.read_text(encoding="utf-8"))
        entry_record = next(
            (e for e in contract["entry_points"] if e["name"] == arguments.entry),
            None,
        )
        if entry_record is None:
            raise SystemExit(f"no entry point {arguments.entry!r} in the contract")
        matches = [
            p["name"] for p in entry_record["parameters"]
            if p.get("source_name") == arguments.source
        ]
        if not matches:
            raise SystemExit(
                f"no parameter of {arguments.entry!r} carries source "
                f"{arguments.source!r}"
            )
        if len(matches) > 1:
            print(f"note: {len(matches)} parameters carry this source name; "
                  f"tracing the first: {matches}")
        token = matches[0]

    print(f"tracing {token!r} from {arguments.entry}\n")
    trace(subs, arguments.entry, token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
