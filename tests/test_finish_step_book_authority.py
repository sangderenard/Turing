"""``_finish_pages`` reads the posted rows as the authority.

Program shape: a function returning ``total`` whose ``function_output`` row
the builder posted, while the builder's live ``named_returns`` / alias dict
say something else.  The old view assembly computed the book's answer and
replaced it with the live dict; the rows are now the view and a
disagreement is a ``ConcordanceRefusal`` naming the page, the row and both
values.
"""

from types import SimpleNamespace

import pytest

from src.compiler.concordance_declarations import (
    CONTROL_VALUE_ALIAS,
    FUNCTION_OUTPUT,
    AliasFact,
    AliasKind,
)
from src.compiler.identity_concordance import ConcordanceRefusal
from src.compiler.precompile_to_ssa import _ControlSSABuilder


class _Page:
    def __init__(self):
        self.facts = {}

    def scope_rows(self, scope):
        return tuple(row for row in self.facts if row[0] == scope)

    def latest(self, row, default=None):
        return self.facts.get(row, default)


def _builder(posted_name, aliases=None, posted_aliases=None):
    scope = ("scope", 1)
    page = _Page()
    alias_page = _Page()
    for alias, source in (posted_aliases or {}).items():
        alias_page.facts[(scope, alias)] = AliasFact(source, AliasKind.PLANNING)
    book = SimpleNamespace(pages={
        FUNCTION_OUTPUT.name: page,
        CONTROL_VALUE_ALIAS.name: alias_page,
    })
    builder = object.__new__(_ControlSSABuilder)
    builder._book = lambda: book
    builder._scope = lambda: scope
    builder._binding_cell = lambda value_id: None

    def post(page_decl, row, fact, *cells, stage=None):
        # The row the builder posts for the slot: the name it is told to
        # post, which the test makes differ from the live named_returns.
        page.facts[row] = (posted_name, fact[1])

    builder._post_concord = post
    builder.value_aliases = dict(aliases or {})
    builder.ssa_value_objects = {}
    builder._carried_port_values = {}
    builder.function_name = "f"
    return builder


def _returned():
    return (SimpleNamespace(id=7),)


def test_finish_pages_reads_function_output_rows_when_they_agree():
    builder = _builder("total")
    view = builder._finish_pages((("total", 7),), (), _returned())
    assert view["named_outputs"] == (("total", 7),)


def test_finish_pages_refuses_function_output_rows_that_disagree_with_named_returns():
    builder = _builder("posted_name")
    with pytest.raises(ConcordanceRefusal) as refusal:
        builder._finish_pages((("total", 7),), (), _returned())
    message = str(refusal.value)
    assert "function_output" in message
    assert "posted_name" in message and "total" in message


def test_finish_pages_refuses_alias_rows_that_disagree_with_live_aliases():
    builder = _builder("total", aliases={5: 9}, posted_aliases={5: 8})
    with pytest.raises(ConcordanceRefusal) as refusal:
        builder._finish_pages((("total", 7),), (), _returned())
    message = str(refusal.value)
    assert "control_value_alias" in message
    assert "8" in message and "9" in message
