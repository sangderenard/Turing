"""A folded static view keeps its own rank across storage metadata transport."""

from src.compiler.concordance_declarations import (
    AGGREGATE_ABI_LEGALIZATION, AGGREGATE_PASSTHROUGH_REBINDING,
    INGESTION, INGEST_SOURCE, SSA_VALUE, SSAValueFact, SSAValueOrigin,
)
from src.compiler.identity_concordance import Derived, IdentityBook, Mode, Novel
from src.compiler.ssa_record_return_state import function_scope_of
from src.compiler.tensor_ssa_lowering import _booked_static_view_rebinding
from src.transmogrifier.ssa import BasicBlock, Function, SSAValue


def _view_fixture(*, cite_source=True):
    source = SSAValue(0, "int64", (4,))
    view = SSAValue(64, "int64", (4, 1), accounting={
        "ssa_storage_view": {
            "storage_value_id": 6, "view_shape": (4, 1),
            "operation": "unsqueeze",
        },
    })
    function = Function("root", [source, view], {
        "entry": BasicBlock("entry", []),
    })
    scope = function_scope_of(function)
    book = IdentityBook()
    source_cell = book.post(
        SSA_VALUE, (scope, 0),
        SSAValueFact("int64", (4,), SSAValueOrigin.ADOPTED_GRAPH_ID,
                     physical_dtype="int64", storage="span", rank=1),
        stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()),
        mode=Mode.CONCORD,
    )
    view_cell = book.post(
        SSA_VALUE, (scope, 64),
        SSAValueFact("int64", (4, 1), SSAValueOrigin.ADOPTED_GRAPH_ID,
                     physical_dtype="int64", storage="span", rank=2),
        stage=INGESTION, provenance=Derived((source_cell,)),
        mode=Mode.CONCORD,
    )
    book.post(
        AGGREGATE_PASSTHROUGH_REBINDING, (scope, 64),
        (0, "root__planned_region_18", 64, (4, 1)),
        stage=AGGREGATE_ABI_LEGALIZATION,
        provenance=Derived((view_cell, source_cell) if cite_source
                           else (view_cell,)), mode=Mode.REVISE,
    )
    return book, function


def test_booked_passthrough_keeps_static_view_rank_from_exact_storage_edge():
    book, function = _view_fixture()

    assert _booked_static_view_rebinding(
        book, function, 64, function, 0, (4, 1),
    )
    assert not _booked_static_view_rebinding(
        book, function, 64, function, 6, (4, 1),
    )
    assert not _booked_static_view_rebinding(
        book, function, 64, function, 0, (1, 4),
    )

    foreign = Function("foreign", [SSAValue(0, "int64", (4,))], {
        "entry": BasicBlock("entry", []),
    })
    assert not _booked_static_view_rebinding(
        book, function, 64, foreign, 0, (4, 1),
    )
    missing_source, same_function = _view_fixture(cite_source=False)
    assert not _booked_static_view_rebinding(
        missing_source, same_function, 64, same_function, 0, (4, 1),
    )
