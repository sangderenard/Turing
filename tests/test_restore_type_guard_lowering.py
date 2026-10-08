"""``_restore_type``'s ``isinstance(value, AbstractTensor)`` IfExp guard.

``value`` is a callee formal every callsite binds to a computed AbstractTensor
(``update_dt_max``'s ``dx_t / maximum(...)``, ``pi_update``'s merged ``dt_new``).
No declaration states its Python type, so the structural fold used to leave
the ``schema_type_guard`` call undecided and the control program branched on a
value nothing produces (``operand-never-written``).  The callsite fixed point
now posts a ``formal_tensor_class`` row for such a formal and the fold decides
the guard from it.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "tools", ROOT / "tools" / "compiler_probes"):
    sys.path.insert(0, str(entry))


def test_controller_lowering_has_no_undefined_operand():
    import audit_identity_concordance as audit
    from probe_module_dominance import module_dominance_violations

    module = audit.CASES["controller"]()
    violations = module_dominance_violations(module)
    assert {check: items for check, items in violations.items() if items} == {}


def test_formal_tensor_class_row_is_posted_for_restore_type_value():
    import audit_identity_concordance as audit
    from src.compiler.concordance_declarations import FORMAL_TENSOR_CLASS
    from src.compiler.identity_concordance import identity_book

    module = audit.CASES["controller"]()
    page = identity_book(module).pages[FORMAL_TENSOR_CLASS.name]
    posted = {
        row[1] for scope in {row[0] for row in page.rows()}
        for row in page.scope_rows(scope) if page.latest(row) is True
    }
    assert "value" in posted


def test_declared_none_distribution_keeps_the_numeric_path_and_posts_its_type():
    """A root parameter declared ``builtins.NoneType`` (int64 storage) reaches
    the callee as NoneType, not as the int its dtype implies."""

    import audit_identity_concordance as audit
    from probe_audit_case_dominance import declared_none_distribution
    from src.compiler.concordance_declarations import PARAMETER_PYTHON_TYPE
    from src.compiler.identity_concordance import identity_book

    assert declared_none_distribution() == 0
    module = audit._lower_toplevel(distribution_declared=True)
    page = identity_book(module).pages[PARAMETER_PYTHON_TYPE.name]
    declared = {
        (row[1], page.latest(row))
        for scope in {row[0] for row in page.rows()}
        for row in page.scope_rows(scope)
    }
    assert ("distribution", "builtins.NoneType") in declared
