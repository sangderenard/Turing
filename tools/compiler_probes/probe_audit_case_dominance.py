"""Every operand use in the identity-concordance audit's lowerings is defined.

The audit CASES (``tools/audit_identity_concordance.py``) are real,
seconds-long lowerings.  This probe lowers the named cases (default: all) and
runs ``module_dominance_violations`` -- the compiler's own undefined-operand
inventory plus the definition-dominance check -- over each module, so a
dangling operand shows up as the same defect the audit reports as
``operand-never-written``, with the function, block and value id.

    python tools/compiler_probes/probe_audit_case_dominance.py [case ...]

Exit status is the violation count (0 = clean).  Cause map:

  controller          the ``_restore_type`` IfExp guard
                      ``value.item() if isinstance(value, AbstractTensor)
                      else value`` over a callee formal bound to computed
                      tensors (``formal_tensor_class``): CLEAN.
  controller_untyped  the ``isinstance(ref, ...)`` guards of the generic
                      ``_restore_type``: ``ref`` is the root's undeclared
                      ``dx`` / ``dt_prev`` (PARAMETER_ABI_UNDECLARED), no fact
                      decides the guard.
  toplevel            ``distribution(...)`` is a call through an undeclared
                      callable parameter: no callee exists to lower.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for entry in (ROOT, ROOT / "tools", ROOT / "tools" / "compiler_probes"):
    sys.path.insert(0, str(entry))

import audit_identity_concordance as audit  # noqa: E402
from probe_module_dominance import module_dominance_violations  # noqa: E402


def main(argv: list[str]) -> int:
    total = 0
    for case in argv[1:] or list(audit.CASES):
        module = audit.CASES[case]()
        violations = module_dominance_violations(module)
        count = sum(len(items) for items in violations.values())
        total += count
        print(f"{case}: {count} violation(s)")
        for check, items in violations.items():
            for item in items:
                print(f"  {check}: {item}")
    print(f"{total} violation(s)")
    return total


if __name__ == "__main__":
    sys.exit(main(sys.argv))
