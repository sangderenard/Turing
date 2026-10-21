"""The ordered SEMANTIC_OUTPUT rows are a function's semantic outputs.

Program shape: a function whose outputs are consumed by position.  Wrong
record: ``function.metadata["semantic_output_ids"]``, a tuple that structural
recovery used to write from its own list.  Rule: the metadata is a derived view
of the posted rows; the writer refuses when recovery and rows disagree, and a
reader given only the rows (no metadata) pairs from the rows.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.compiler.concordance_declarations import (  # noqa: E402
    CONTROL_SSA_FINISH, SEMANTIC_OUTPUT,
)
from src.compiler.extraction_contract import ExtractionContract  # noqa: E402
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa  # noqa: E402
from src.compiler.identity_concordance import (  # noqa: E402
    ConcordanceRefusal, IdentityBook, IdentityPage, Mode, Novel, Registry,
    REGISTRY, begin_identity_book, end_identity_book, identity_book,
)
from src.compiler.ssa_reachability import (  # noqa: E402
    function_output_rows, semantic_output_ids_of,
)

SOURCE = '''
def f(a: int, b: int) -> int:
    return a * b + 1
'''


def _lower():
    contract = ExtractionContract(ROOT / "extraction_contracts" / "program_extraction.yaml")
    module, _outputs, exports = lower_ast_source_to_ssa(
        SOURCE, "f", extraction_contract=contract,
        progress=lambda message: None, progress_bars=False,
    )
    return module, module.functions[exports[0]]


def test_writer_publishes_metadata_as_the_rows_in_row_order():
    module, function = _lower()
    rows = function_output_rows(function, book=identity_book(module))
    assert rows
    assert function.metadata["semantic_output_ids"] == tuple(
        value_id for _name, value_id in rows
    )


def test_writer_refuses_when_recovery_and_posted_row_disagree(monkeypatch):
    real_latest = IdentityPage.latest

    def latest(self, row, default=None):
        fact = real_latest(self, row, default)
        # Only the structural-recovery writer sees a differently grouped row.
        if (
            self.name == SEMANTIC_OUTPUT.name and fact is not None
            and sys._getframe(1).f_code.co_name
            == "recover_structural_source_outputs"
        ):
            return (fact[0], int(fact[1]) + 1000)
        return fact

    monkeypatch.setattr(IdentityPage, "latest", latest)
    with pytest.raises(ConcordanceRefusal) as raised:
        _lower()
    message = str(raised.value)
    assert "semantic_output page, row" in message
    assert "the reducer's row holds" in message
    assert "structural recovery names" in message


def _book_with_rows(scope, facts):
    book = IdentityBook()
    registry = Registry()
    for name in ("pages", "stages", "transforms", "reasons"):
        getattr(registry, name).update(getattr(REGISTRY, name))
    registry.private_pages.update(REGISTRY.private_pages)
    book.registry = registry
    origin = registry.declare_transform("test_function_output_rows", 0)
    for slot, fact in enumerate(facts):
        book.post(
            SEMANTIC_OUTPUT, (scope, slot), fact, stage=CONTROL_SSA_FINISH,
            provenance=Novel(origin, ()), mode=Mode.CONCORD,
        )
    return book


def _function(name, metadata):
    return SimpleNamespace(name=name, metadata=metadata)


def test_reader_pairs_from_rows_not_from_metadata():
    book = _book_with_rows("h", [("x", 7), (None, 3)])
    token = begin_identity_book(book)[1]
    try:
        # The metadata tuple is deliberately a different, wrongly ordered view.
        function = _function("h", {"semantic_output_ids": (3, 7, 99)})
        assert semantic_output_ids_of(function) == (7, 3)
        assert function_output_rows(function) == (("x", 7), (None, 3))
    finally:
        end_identity_book(token)


def test_accepted_component_reads_its_own_retained_book():
    owner = _book_with_rows("h", [("x", 5), ("y", 6)])
    function = _function("h", {
        "semantic_output_ids": (6, 5),
        "accepted_ssa_component": {
            "schema": "turing.accepted-ssa-component.v1",
            "artifact_sha256": "0" * 64,
            "identity_book": owner,
        },
    })
    # Nothing is posted in the active book; the retained book answers.
    assert semantic_output_ids_of(function) == (5, 6)


def test_metadata_without_rows_is_refused():
    function = _function("nobody", {"semantic_output_ids": (1, 2)})
    with pytest.raises(ConcordanceRefusal):
        semantic_output_ids_of(function, book=IdentityBook())
