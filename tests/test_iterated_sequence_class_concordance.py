"""Loop bindings take their scalar class from source sequence declarations."""

from pathlib import Path

import pytest

from src.compiler.concordance_declarations import SOURCE_VALUE_CLASS
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import begin_identity_book, end_identity_book


CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "extraction_contracts" / "program_extraction.yaml"
)


@pytest.mark.parametrize("source, expected, annotated", [
    ("""\
def root(xs: list[int]) -> int:
    total = 0
    for index, value in enumerate(xs):
        total += index + value
    return total
""", ("builtins.int", "builtins.int"), (False, True)),
    ("""\
def root(xs: list[int], ys: list[float]) -> float:
    total = 0.0
    for left, right in zip(xs, ys):
        total += left + right
    return total
""", ("builtins.int", "builtins.float"), (True, True)),
])
def test_iterated_sequence_target_class_has_exact_source_edges(
    source, expected, annotated,
):
    book, token = begin_identity_book()
    try:
        lower_ast_source_to_ssa(
            source, "root", name="root", extraction_contract=CONTRACT,
            progress=lambda _message: None, progress_bars=False,
            identity_book=book,
        )
        page = book.page(SOURCE_VALUE_CLASS)
        receipts = tuple(
            next((row, page.latest(row)) for row in page.rows()
                 if isinstance(row[0], tuple) and len(row[0]) == 2
                 and row[0][1] == "root" and row[1] == position)
            for position in range(len(expected))
        )
        assert tuple(fact[0] for _row, fact in receipts) == expected
        for position, (row, _fact) in enumerate(receipts):
            cell = book.latest_ref(SOURCE_VALUE_CLASS, row)
            sources = {source.page.name for source, _stage in book.edges_into(cell)}
            if annotated[position]:
                assert "parameter_annotation" in sources
            else:
                assert "canonical_value" in sources
    finally:
        end_identity_book(token)
