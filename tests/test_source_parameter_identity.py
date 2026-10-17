"""A classmethod's receiver binding is a record with sources.

``_concorded_static_parameter_bindings`` posts the ``cls`` row of
``source_parameter_identity_concordance`` DERIVED from the parameter's
``source_span`` cell and the bound class's ``static_symbol_definition``
cell.  The whole-program bootstrap died on this row (``CompressedTile.encode``)
while it was still a raw ``page.set``.
"""
from __future__ import annotations

from pathlib import Path

from src.compiler.concordance_declarations import (
    SOURCE_PARAMETER_IDENTITY, SOURCE_SPAN, STATIC_SYMBOL_DEFINITION,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import identity_book

CONTRACTS = Path(__file__).resolve().parents[1] / "extraction_contracts"


class Codec:
    """A class reached only through the Python binding of the call site;
    source pursuit ingests ``encode`` and binds its ``cls`` to this class."""

    scale: float = 2.0

    @classmethod
    def encode(cls, x: float) -> float:
        return x * cls.scale


SOURCE = """
def root(x: float) -> float:
    return Codec.encode(x)
"""


def test_classmethod_receiver_binding_is_posted_with_sources():
    module, _outputs, _exports = lower_ast_source_to_ssa(
        SOURCE, "root", name="source_parameter_identity",
        python_bindings={"Codec": Codec},
        extraction_contract=ExtractionContract(
            CONTRACTS / "program_extraction.yaml"
        ),
        runtime_closure_only=True,
        progress=lambda _message: None,
    )
    book = identity_book(module)
    page = book.pages.get(SOURCE_PARAMETER_IDENTITY.name)
    assert page is not None, sorted(book.pages)
    rows = [
        row for row in page.rows()
        if row[1] == "cls" and str(row[0][-1]).endswith("Codec.encode")
    ]
    assert rows, list(page.rows())
    row = rows[0]
    assert page.latest(row) == ("static_class", __name__, "Codec")

    cell = book.latest_ref(SOURCE_PARAMETER_IDENTITY, row)
    assert cell is not None
    sources = book.edges_into(cell)
    assert sources, "the cls binding row was posted with no edge"
    pages = {source.page.name for source, _stage in sources}
    # The parameter's own span and the class it is bound to.
    assert SOURCE_SPAN.name in pages, sources
    assert STATIC_SYMBOL_DEFINITION.name in pages, sources
    span = next(
        source for source, _stage in sources
        if source.page.name == SOURCE_SPAN.name
    )
    assert book.page(SOURCE_SPAN).latest(span.row).kind == "arg", span
    symbol = next(
        source for source, _stage in sources
        if source.page.name == STATIC_SYMBOL_DEFINITION.name
    )
    assert symbol.row[:2] == (__name__, "Codec"), symbol
