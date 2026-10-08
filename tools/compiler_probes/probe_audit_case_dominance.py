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


def declared_none_distribution() -> int:
    """``distribution`` declared ``builtins.NoneType`` on the root: the callee
    must keep its numeric path.  The declared type used to reach the callee as
    ``builtins.int`` (re-derived from the int64 storage), ``distribution is not
    None`` folded True and ``_propose_dt_pen`` was emptied to
    ``return distribution(...)`` -- two functions, 0 findings, no body.
    Returns the number of defects: dominance violations, plus one when the
    lowered module has no planned numeric regions for ``_propose_dt_pen``."""

    module = audit._lower_toplevel(distribution_declared=True)
    violations = module_dominance_violations(module)
    count = sum(len(items) for items in violations.values())
    regions = [
        name for name in module.functions
        if "_propose_dt_pen" in str(name) and "__planned_region_" in str(name)
    ]
    print(f"declared_none_distribution: {count} violation(s), "
          f"{len(module.functions)} functions, "
          f"{len(regions)} _propose_dt_pen region(s)")
    for check, items in violations.items():
        for item in items:
            print(f"  {check}: {item}")
    if len(regions) < 5:
        print("  _propose_dt_pen lost its numeric path")
        count += 1
    return count


def main(argv: list[str]) -> int:
    total = 0
    if argv[1:] == ["declared_none_distribution"]:
        return declared_none_distribution()
    for case in argv[1:] or list(audit.CASES):
        module = audit.CASES[case]()
        violations = module_dominance_violations(module)
        count = sum(len(items) for items in violations.values())
        total += count
        print(f"{case}: {count} violation(s)")
        for check, items in violations.items():
            for item in items:
                print(f"  {check}: {item}")
    if not argv[1:]:
        total += declared_none_distribution()
    print(f"{total} violation(s)")
    return total


if __name__ == "__main__":
    sys.exit(main(sys.argv))
