"""The descriptor query's call-scoped reuse equals a fresh recompute.

``_tensor_descriptor`` hands one ``_DescriptorQuery`` down its recursion so a
node reached twice by one outermost query is derived once.  This test lowers
the audit programs and, at every outermost rule call, derives the same node
again with a plain ``_seen`` set (the path the rule took before the query
object existed, which holds nothing) and requires the two answers equal.  It
also requires the scoped ``formal_shape`` read to return exactly the rows the
full-page scan selected.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import audit_identity_concordance as audit  # noqa: E402
from src.compiler import glsl_deployment_strategy as strategy  # noqa: E402
from src.compiler.identity_concordance import (  # noqa: E402
    authored_function_name, current_identity_book,
)


@pytest.mark.parametrize("case", list(audit.CASES))
def test_query_scoped_answers_equal_fresh_recompute(case, monkeypatch):
    rule = strategy._tensor_descriptor_rule
    compared = {"roots": 0, "scoped_rows": 0}
    mismatches: list[tuple] = []

    def checked(graph, node_id, _seen=None):
        answer = rule(graph, node_id, _seen)
        if isinstance(_seen, strategy._DescriptorQuery) and not _seen:
            compared["roots"] += 1
            fresh = rule(graph, node_id, set())
            if answer != fresh:
                mismatches.append((
                    graph.G.graph.get("function_name"), int(node_id),
                    answer, fresh,
                ))
            owner = authored_function_name(
                graph.G.graph.get("function_name")
            )
            page = current_identity_book().page("formal_shape")
            scanned = [
                row for row in page.rows()
                if isinstance(row, tuple) and len(row) >= 2
                and authored_function_name(row[0]) == owner
            ]
            assert set(scanned) == set(page.scope_rows(owner))
            compared["scoped_rows"] += len(scanned)
        return answer

    monkeypatch.setattr(strategy, "_tensor_descriptor_rule", checked)
    audit.CASES[case]()
    assert not mismatches, mismatches[:3]
    print(f"{case}: roots compared={compared['roots']} "
          f"formal rows={compared['scoped_rows']}")
