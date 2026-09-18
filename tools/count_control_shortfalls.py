"""Emit every function of a captured control module and count LLVM shortfalls.

A shortfall count alone is not evidence a program is correct -- silent
miscompiles report success -- but it is the cheapest way to tell whether a
seam has closed.  Usage:

    python tools/count_control_shortfalls.py build/<dir>
"""

from __future__ import annotations

import pickle
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    directory = Path(sys.argv[1] if len(sys.argv) > 1 else "build/sfdc-get-baseline")
    from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm

    with (directory / "control_repository_ssa.pkl").open("rb") as stream:
        module, _outputs, _exports = pickle.load(stream)

    total = 0
    reasons: Counter[str] = Counter()
    failing = 0
    for name in module.functions:
        try:
            artifact = emit_ssa_function_to_llvm(module, name)
        except Exception as error:  # emission itself refusing is also a shortfall
            failing += 1
            total += 1
            reasons[f"{type(error).__name__}: {error}"[:120]] += 1
            print(f"{name}: RAISED {type(error).__name__}: {error}")
            continue
        if artifact.shortfalls:
            failing += 1
            total += len(artifact.shortfalls)
            print(f"{name}: {len(artifact.shortfalls)} shortfall(s)")
            for item in artifact.shortfalls:
                described = f"{item.operation}: {item.reason}"
                print(f"    {described}")
                reasons[described[:160]] += 1
    print()
    print(f"functions       {len(module.functions)}")
    print(f"functions short {failing}")
    print(f"shortfalls      {total}")
    for reason, count in reasons.most_common():
        print(f"  {count:3d}  {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
