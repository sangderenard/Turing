"""Pursuit holds definitions in the concordance, not in a private table."""
from __future__ import annotations

import ast

import pytest

from src.compiler.concordance_declarations import (
    SOURCE_DEFINITION, SOURCE_SPAN,
)
from src.compiler.identity_concordance import (
    begin_identity_book, end_identity_book,
)
from src.transmogrifier.graph.graph_express2 import (
    _HeldDefinitions, _annotate_visual_source_owners,
)

SOURCE = '''
class Stage:
    def __init__(self, name):
        self.name = name

class Holder:
    @property
    def scope(self):
        return 1

    @scope.setter
    def scope(self, value):
        pass
'''


def _module():
    tree = ast.parse(SOURCE)
    tree._turing_source_module = "probe.module"
    _annotate_visual_source_owners(tree)
    return tree


def test_a_held_definition_is_a_book_row_derived_from_its_span():
    book, token = begin_identity_book()
    try:
        tree = _module()
        stage = tree.body[0]
        held = _HeldDefinitions()
        identity = ("probe.module", "Stage")
        assert identity not in held
        held[identity] = stage
        assert identity in held and held[identity] is stage
        row = ("probe.module", "Stage")
        assert book.page(SOURCE_DEFINITION).latest(row) == "ClassDef"
        cell = book.latest_ref(SOURCE_DEFINITION, row)
        span = book.latest_ref(SOURCE_SPAN, ("probe.module", "Stage", ()))
        assert span in {source for source, _stage in book.edges_into(cell)}
    finally:
        end_identity_book(token)


def test_the_book_decides_not_the_handle():
    book, token = begin_identity_book()
    try:
        tree = _module()
        held = _HeldDefinitions()
        held[("probe.module", "Stage")] = tree.body[0]
    finally:
        end_identity_book(token)
    other, other_token = begin_identity_book()
    try:
        # A different book holds nothing, so nothing is held, whatever this
        # table once saw.
        assert ("probe.module", "Stage") not in held
    finally:
        end_identity_book(other_token)


def test_a_second_node_for_one_identity_is_duplicate_material():
    book, token = begin_identity_book()
    try:
        first = _module().body[0]
        second = _module().body[0]
        held = _HeldDefinitions()
        held[("probe.module", "Stage")] = first
        held[("probe.module", "Stage")] = first
        with pytest.raises(ValueError, match="already held"):
            held[("probe.module", "Stage")] = second
    finally:
        end_identity_book(token)


def test_a_property_getter_and_setter_are_one_authored_identity():
    book, token = begin_identity_book()
    try:
        tree = _module()
        holder = tree.body[1]
        getter, setter = holder.body
        held = _HeldDefinitions()
        identity = ("probe.module", "Holder.scope")
        assert held.hold_authored(identity, getter) is True
        assert held.hold_authored(identity, setter) is False
        assert held[identity] is getter
    finally:
        end_identity_book(token)
