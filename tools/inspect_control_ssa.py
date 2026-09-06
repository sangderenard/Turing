"""Compact views of a captured control module: op histograms and op searches.

    python tools/inspect_control_ssa.py <dir> ops   [function-substring]
    python tools/inspect_control_ssa.py <dir> find  <op-substring> [function-substring]
    python tools/inspect_control_ssa.py <dir> abi   [function-substring]
"""

from __future__ import annotations

import pickle
import sys
from collections import Counter
from pathlib import Path


def _load(directory: Path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    with (directory / "control_repository_ssa.pkl").open("rb") as stream:
        return pickle.load(stream)


def _brief(value) -> str:
    accounting = dict(getattr(value, "accounting", None) or {})
    keys = {
        key: accounting[key]
        for key in accounting
        if key.startswith("program_abi") or key.startswith("iterable")
        or key in {"linked_call_frame_storage", "keyed_mapping"}
    }
    return (
        f"%{value.id}:{getattr(value, 'dtype', None)}"
        f"{tuple(getattr(value, 'shape', ()) or ())}"
        + (f" {keys}" if keys else "")
    )


def main() -> int:
    directory = Path(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "ops"
    module, _outputs, _exports = _load(directory)

    if mode == "ops":
        wanted = sys.argv[3] if len(sys.argv) > 3 else None
        for name, function in module.functions.items():
            if wanted is not None and wanted not in name:
                continue
            counts: Counter[str] = Counter()
            for block in function.blocks.values():
                for instruction in block.instrs:
                    counts[str(instruction.op)] += 1
            print(f"{name}  ({sum(counts.values())} instrs, "
                  f"{len(function.blocks)} blocks)")
            for op, count in counts.most_common():
                print(f"    {count:4d}  {op}")
        return 0

    if mode == "find":
        needle = sys.argv[3]
        wanted = sys.argv[4] if len(sys.argv) > 4 else None
        for name, function in module.functions.items():
            if wanted is not None and wanted not in name:
                continue
            for block_name, block in function.blocks.items():
                for instruction in block.instrs:
                    if needle.lower() not in str(instruction.op).lower():
                        continue
                    operands = " ".join(
                        _brief(argument) for argument in instruction.args
                    )
                    result = (
                        _brief(instruction.res)
                        if instruction.res is not None else "-"
                    )
                    print(
                        f"{name}/{block_name}: {instruction.op} "
                        f"[{operands}] -> {result} "
                        f"{dict(instruction.attributes or {})}"
                    )
        return 0

    if mode == "abi":
        wanted = sys.argv[3] if len(sys.argv) > 3 else None
        seen: set[tuple] = set()
        for name, function in module.functions.items():
            if wanted is not None and wanted not in name:
                continue
            values = [*function.args]
            for block in function.blocks.values():
                for instruction in block.instrs:
                    values.extend(instruction.args)
                    if instruction.res is not None:
                        values.append(instruction.res)
            for value in values:
                accounting = dict(getattr(value, "accounting", None) or {})
                keyed = {
                    key: accounting[key]
                    for key in accounting
                    if "keyed" in key
                }
                if not keyed:
                    continue
                record = (name, int(value.id), tuple(sorted(keyed.items())))
                if record in seen:
                    continue
                seen.add(record)
                print(f"{name}: %{value.id} {keyed}")
        return 0

    if mode == "body":
        wanted = sys.argv[3]
        for name, function in module.functions.items():
            if wanted != name:
                continue
            print(f"{name}({', '.join(_brief(a) for a in function.args)})")
            for block_name, block in function.blocks.items():
                print(f"  {block_name}:")
                for instruction in block.instrs:
                    operands = " ".join(
                        f"%{argument.id}" for argument in instruction.args
                    )
                    result = (
                        f"%{instruction.res.id} = "
                        if instruction.res is not None else ""
                    )
                    attributes = dict(instruction.attributes or {})
                    print(
                        f"    {result}{instruction.op} {operands}"
                        + (f"  {attributes}" if attributes else "")
                    )
        return 0

    raise SystemExit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    raise SystemExit(main())
