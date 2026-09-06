"""Source-conditional membership must use authored syntax, not AST tokens."""

from __future__ import annotations

import ast
from types import SimpleNamespace

import networkx as nx

from src.compiler.glsl_deployment_strategy import _branch_compartments


def test_locationless_ast_helpers_do_not_claim_unrelated_branch_work():
    conditional = ast.parse("if ready:\n    value = source\n").body[0]
    guarded_name = conditional.body[0].value
    outside_name = ast.parse("other = source\n").body[0].value

    graph = nx.DiGraph()
    graph.add_node(10, expr_obj=conditional)
    graph.add_node(11, expr_obj=guarded_name)
    graph.add_node(12, expr_obj=outside_name)
    # Context/operator helpers have no source position.  In CPython these are
    # commonly shared singleton-shaped nodes; matching one to a branch used
    # to pollute arbitrary scheduled regions with that branch membership.
    graph.add_node(13, expr_obj=ast.Load())

    memberships = _branch_compartments(SimpleNamespace(G=graph))

    assert memberships[11] == frozenset({(10, "body")})
    assert (10, "body") not in memberships.get(12, frozenset())
    assert (10, "body") not in memberships.get(13, frozenset())

