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
