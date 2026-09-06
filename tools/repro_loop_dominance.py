"""Isolates the ``value N was read before it was defined; does not dominate``
failure on a keyed-dict ``for ... in x.items():`` loop -- lowers cleanly,
fails only when actually EXECUTED via the reference evaluator.  Real,
minimal, authored-shape source (the same accumulate-over-items() pattern
used throughout coerce_metrics/_propose_dt_pen), not a hand-picked oddity.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa  # noqa: E402
from src.compiler.ssa_reference_evaluator import SSAReferenceEvaluator  # noqa: E402

CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "extraction_contracts" / "program_extraction.yaml"
)

CASES = {
    "iterate": (
        "def root(channels):\n"
        "    total = 0.0\n"
        "    for name, limit in channels.items():\n"
        "        total = total + limit\n"
        "    return total\n"
    ),
}


def main() -> int:
    wanted = sys.argv[1:] or list(CASES)
    failures = 0
    for case_name in wanted:
        source = CASES[case_name]
        print(f"===== {case_name} =====", flush=True)
        module, outputs, exports = lower_ast_source_to_ssa(
            source, "root", name=case_name, extraction_contract=CONTRACT,
        )
        entry = next(n for n in module.functions if n.endswith("__root"))
        function = module.functions[entry]
        length_value, keys_value, values_value = function.args
        channels = {"a": 3.0, "b": 4.0, "c": 5.0}
        arguments = {
            int(length_value.id): len(channels),
            int(keys_value.id): list(channels.keys()),
            int(values_value.id): list(channels.values()),
        }
        evaluator = SSAReferenceEvaluator(module)
        try:
            result = evaluator.run(entry, arguments)
        except Exception as error:
            print(f"  EXECUTION FAILED: {type(error).__name__}: {str(error)[:400]}",
                  flush=True)
            failures += 1
            continue
        got = result.returned[0] if result.returned else None
        expected = sum(channels.values())
        ok = got is not None and abs(float(got) - expected) < 1e-9
        print(f"  got={got} expected={expected} {'OK' if ok else 'MISMATCH'}",
              flush=True)
        if not ok:
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
