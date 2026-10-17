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


def test_the_last_authored_definition_of_a_name_is_the_held_one():
    book, token = begin_identity_book()
    try:
        tree = _module()
        holder = tree.body[1]
        getter, setter = holder.body
        held = _HeldDefinitions()
        identity = ("probe.module", "Holder.scope")
        assert held.hold_authored(identity, getter) is True
        assert held.hold_authored(identity, setter) is True
        assert held[identity] is setter
        # one book row for the one identity
        assert book.page(SOURCE_DEFINITION).latest(identity) == "FunctionDef"
    finally:
        end_identity_book(token)


def test_an_overload_stub_is_not_held():
    book, token = begin_identity_book()
    try:
        tree = ast.parse(
            "from typing import overload\n"
            "@overload\n"
            "def f(x: int) -> int: ...\n"
            "def f(x):\n"
            "    return x\n"
        )
        tree._turing_source_module = "probe.module"
        _annotate_visual_source_owners(tree)
        stub, real = tree.body[1], tree.body[2]
        held = _HeldDefinitions()
        identity = ("probe.module", "f")
        assert held.hold_authored(identity, stub) is False
        assert identity not in held
        assert held.hold_authored(identity, real) is True
        assert held[identity] is real
    finally:
        end_identity_book(token)


def test_a_discovered_incumbent_may_be_replaced_but_an_authored_one_may_not():
    book, token = begin_identity_book()
    try:
        authored = _module().body[0]
        copy = _module().body[0]
        held = _HeldDefinitions()
        identity = ("probe.module", "Stage")
        held.hold_authored(identity, authored)
        with pytest.raises(ValueError, match="already held"):
            held[identity] = copy
        discovered = _module().body[1]
        discovered._python_source_identity = ("probe.module", "Holder")
        replacement = _module().body[1]
        replacement._python_source_identity = ("probe.module", "Holder")
        held[("probe.module", "Holder")] = discovered
        held[("probe.module", "Holder")] = replacement
        assert held[("probe.module", "Holder")] is replacement
    finally:
        end_identity_book(token)
