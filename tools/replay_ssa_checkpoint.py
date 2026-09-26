"""Replay a local pre-frame-link checkpoint without source extraction or planning.

Input is the (args, kwargs) snapshot of _class_surface_ssa_program produced by
the patch-sequence diagnostic. This is lower-only inspection, not native parity.
Like other compiler pickle artifacts, checkpoints must come from trusted local
compiler runs. No execution deadline or iteration limit is imposed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.compiler.fortran_c_shell import _class_surface_ssa_program
from src.compiler.ssa_self_check import run_all


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    positional, keywords = pickle.loads(args.checkpoint.read_bytes())
    keywords = {**keywords, "progress": lambda message: print(message, flush=True)}
    module, outputs, exports = _class_surface_ssa_program(*positional, **keywords)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump((module, outputs, exports), stream, protocol=5)
    temporary.replace(args.output)
    print(f"SSA saved: {args.output}", flush=True)
    findings = run_all(module)
    for finding in findings:
        print(finding, flush=True)
    print(f"Structural check findings: {len(findings)}", flush=True)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
