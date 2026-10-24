"""A method's compile partition and its complete class declare one field.

``partition_authored_source`` keeps the class with the selected method
alone, so ``Rec.run``'s unit sees ``history`` only through run's unannotated
``self.history = ...``; the unit's declared complete source sees
``__init__``'s annotated declaration as well.  Both views post the same
``class_field_declaration`` row.  The smaller view never contradicts the
larger one, while two different annotations still disagree.
"""

from __future__ import annotations

import ast

import pytest

from src.compiler import identity_concordance as concordance
from src.compiler.concordance_declarations import CLASS_FIELD_DECLARATION
from src.compiler.fortran_c_shell import _authored_dataclass_record_views
from src.compiler.project_compilation_product import partition_authored_source


SOURCE = '''
class Rec:
    def __init__(self) -> None:
        self.history: dict[int, list[int]] = {}
        self.steps = 0

    def run(self, n: int) -> int:
        self.history = {n: []}
        self.steps = n
        return n + 1
'''

ROW = ("probe_partition_module", "Rec", "history")


def _views(source):
    _authored_dataclass_record_views(
        ast.parse(source), source_module="probe_partition_module",
    )


@pytest.fixture
def book():
    book, token = concordance.begin_identity_book()
    try:
        yield book
    finally:
        concordance.end_identity_book(token)


def test_partition_view_then_complete_class_reads_the_annotation(book):
    partition, _retained = partition_authored_source(SOURCE, "Rec.run")
    assert "self.history: dict" not in partition
    _views(partition)
    assert book.page(CLASS_FIELD_DECLARATION).latest(ROW).annotation == ""
    _views(SOURCE)
    assert (
        book.page(CLASS_FIELD_DECLARATION).latest(ROW).annotation
        == "dict[int, list[int]]"
    )
    # The partition seen again after the complete class reads the incumbent.
    _views(partition)
    assert (
        book.page(CLASS_FIELD_DECLARATION).latest(ROW).annotation
        == "dict[int, list[int]]"
    )


def test_two_annotations_for_one_field_still_disagree(book):
    _views(SOURCE)
    with pytest.raises(ValueError, match="class_field_declaration disagreement"):
        _views(SOURCE.replace("dict[int, list[int]]", "dict[int, list[str]]"))
