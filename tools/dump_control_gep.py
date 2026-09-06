"""Print every GetElementPtr in the captured control module, with metadata."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    directory = Path(sys.argv[1])
    wanted = sys.argv[2] if len(sys.argv) > 2 else None
    with (directory / "control_repository_ssa.pkl").open("rb") as stream:
        module, _outputs, _exports = pickle.load(stream)
    for name, function in module.functions.items():
        if wanted is not None and wanted not in name:
            continue
        printed_header = False
        for block_name, block in function.blocks.items():
            for instruction in block.instrs:
                if str(instruction.op) != "GetElementPtr":
                    continue
                if not printed_header:
                    print(f"function {name}")
                    printed_header = True
                result = instruction.res
                print(f"  block {block_name}")
                print(f"    {instruction}")
                print(f"    res    = {result!r}")
                for index, argument in enumerate(instruction.args):
                    print(
                        f"    arg[{index}] id={argument.id} "
                        f"dtype={getattr(argument, 'dtype', None)} "
                        f"shape={getattr(argument, 'shape', None)}"
                    )
                print(f"    attrs  = {getattr(instruction, 'attrs', None)!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
