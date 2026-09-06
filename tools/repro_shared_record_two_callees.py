"""Does a shared record's field stay stable in identity across two callees
that each read a DIFFERENT subset of its attributes?

Mirrors the real conflict: ``coerce_metrics(metrics)`` and
``_propose_dt_pen(metrics, ...)`` both take the SAME ``metrics`` object,
each reading some of its fields but not others, and the compiler ends up
disagreeing about one field's dtype between the two callees.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa  # noqa: E402

CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "extraction_contracts"
    / "program_extraction.yaml"
)

CASES = {
    # Both callees read the SAME fields, same order: control case.
    "same_fields_same_order": (
        "def consume_a(record):\n"
        "    return record.speed + record.flag\n"
        "def consume_b(record):\n"
        "    return record.speed - record.flag\n"
        "def root(speed, flag):\n"
        "    record = {'speed': speed, 'flag': flag}\n"
        "    return consume_a(record) + consume_b(record)\n"
    ),
    # consume_a reads only 'speed'; consume_b reads only 'flag'.  If each
    # callee's member-formal index is assigned by ITS OWN usage order
    # instead of a stable field-name mapping, consume_a's index-0 formal
    # ('speed') and consume_b's index-0 formal ('flag') would collide as
    # "the same slot" even though they are different fields.
    "different_subset_each": (
        "def consume_a(record):\n"
        "    return record.speed\n"
        "def consume_b(record):\n"
        "    return record.flag\n"
        "def root(speed, flag):\n"
        "    record = {'speed': speed, 'flag': flag}\n"
        "    return consume_a(record) + consume_b(record)\n"
    ),
    # consume_a reads flag-then-speed; consume_b reads speed-then-flag --
    # opposite ORDER, same two fields.  Same collision risk if index is
    # usage-order-derived rather than name-derived.
    "same_fields_opposite_order": (
        "def consume_a(record):\n"
        "    return record.flag + record.speed\n"
        "def consume_b(record):\n"
        "    return record.speed + record.flag\n"
        "def root(speed, flag):\n"
        "    record = {'speed': speed, 'flag': flag}\n"
        "    return consume_a(record) + consume_b(record)\n"
    ),
    # Closer to the real topology: `metrics = coerce_metrics(metrics)` (a
    # call whose result is either the mutated INPUT or a freshly built
    # record, exactly ``coerce_metrics``'s own two-branch shape) is REBOUND
    # to the same name, and THAT rebound value -- a call RESULT, not a
    # plain dict literal -- is what flows into the second callee.
    "normalize_then_consume": (
        "def normalize(record):\n"
        "    fresh = {'speed': 0.0, 'flag': False}\n"
        "    if record.get('replace'):\n"
        "        return fresh\n"
        "    record['speed'] = record['speed']\n"
        "    return record\n"
        "def consume(record):\n"
        "    return record['flag']\n"
        "def root(speed, flag, replace):\n"
        "    record = {'speed': speed, 'flag': flag, 'replace': replace}\n"
        "    record = normalize(record)\n"
        "    return consume(record)\n"
    ),
}


def main() -> int:
    wanted = sys.argv[1:] or list(CASES)
    for case in wanted:
        source = CASES[case]
        print(f"===== {case} =====", flush=True)
        try:
            module, _outputs, _exports = lower_ast_source_to_ssa(
                source, "root", name=case, extraction_contract=CONTRACT,
            )
            print("  OK:", ", ".join(module.functions), flush=True)
        except Exception as error:
            print(f"  FAILED: {type(error).__name__}: {str(error)[:400]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
