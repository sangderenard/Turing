"""A linked ABI may follow a newer shape cell for the same tensor value."""

from dataclasses import replace

import pytest

from src.compiler.concordance_declarations import (
    CANONICAL_VALUE,
    INGESTION,
    INGEST_SOURCE,
    ValueTypeABIAlternative,
)
from src.compiler.fortran_c_shell import _refresh_linked_value_type_abi_arm
from src.compiler.identity_concordance import (
    Mode,
    Novel,
    begin_identity_book,
    end_identity_book,
    invalidate_shape_transformation,
    record_shape_transformation,
    SHAPE_STATE_PAGE,
)


def _fixture():
    book, token = begin_identity_book()
    scope = ("linked_result|shape", 0)
    lexical = ("linked_result_reads", 0)
    source = book.post(
        CANONICAL_VALUE, (lexical, 11), ("source",), stage=INGESTION,
        provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD,
    )
    result = book.post(
        CANONICAL_VALUE, (lexical, 13), ("result",), stage=INGESTION,
        provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD,
    )
    dynamic = ((), "int64", 1, "dynamic", None)
    static = ((256,), "int64", 1, "static", None)
    record_shape_transformation(
        scope, 11, scope, 13, stage="graph_tensor_descriptor",
        operation="to_dtype", role="operand", source_state=dynamic,
        target_state=dynamic, source_cells=(source, result),
    )
    old = book.latest_ref(SHAPE_STATE_PAGE, (scope, 13))
    # The input's linked ABI settles first.  Its new shape cell is the
    # changed cause for the subsequent to_dtype edge revision.
    record_shape_transformation(
        scope, 10, scope, 11, stage="linked_value_abi_settlement",
        operation="source_shape", role="value", source_state=static,
        target_state=static, source_cells=(source,),
    )
    invalidate_shape_transformation(
        scope, 13, 11, "linked_value_abi_settlement",
        source_cells=(source,),
    )
    record_shape_transformation(
        scope, 11, scope, 13, stage="graph_tensor_descriptor",
        operation="to_dtype", role="operand", source_state=static,
        target_state=static, source_cells=(source, result),
    )
    new = book.latest_ref(SHAPE_STATE_PAGE, (scope, 13))
    common = dict(identity="AbstractTensor", storage="span", dtype="int64",
                  rank=1, storage_identity=(), row_shape=None)
    incumbent = ValueTypeABIAlternative(
        **common, shape=(), metadata_state="dynamic",
        source_cells=(result, old),
    )
    incoming = ValueTypeABIAlternative(
        **common, shape=(256,), metadata_state="static",
        source_cells=(result, new),
    )
    source_contract = {"shape": (256,), "rank": 1, "storage": "span", "dtype": "int64"}
    target_contract = dict(source_contract)
    return book, token, incumbent, incoming, source_contract, target_contract


@pytest.mark.parametrize("retained_history", [False, True])
def test_new_shape_cell_refreshes_one_canonical_linked_abi_arm(retained_history):
    book, token, old, new, source, target = _fixture()
    try:
        if retained_history:
            new = replace(new, source_cells=(
                new.source_cells[0], old.source_cells[1], new.source_cells[1],
            ))
        refreshed = _refresh_linked_value_type_abi_arm(
            old, new, source_contract=source, target_contract=target, book=book,
        )
        assert refreshed is not None
        assert refreshed.shape == (256,)
        assert refreshed.metadata_state == "static"
        assert set(refreshed.source_cells) == set(old.source_cells + new.source_cells)
    finally:
        end_identity_book(token)


@pytest.mark.parametrize("poison", [
    "foreign_canonical", "foreign_shape_scope", "stale_shape_ref",
    "wrong_source_shape", "different_dtype", "different_rank", "different_storage",
    "wrong_target_shape",
])
def test_shape_refresh_refuses_different_identity_or_contract(poison):
    book, token, old, new, source, target = _fixture()
    try:
        if poison == "foreign_canonical":
            foreign = book.post(
                CANONICAL_VALUE, (("foreign_reads", 0), 13), ("foreign",),
                stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()),
                mode=Mode.CONCORD,
            )
            new = replace(new, source_cells=(foreign, new.source_cells[1]))
        elif poison == "foreign_shape_scope":
            record_shape_transformation(
                ("foreign_shape", 0), 11, ("foreign_shape", 0), 13,
                stage="graph_tensor_descriptor", operation="to_dtype",
                role="operand", source_state=((256,), "int64", 1, "static", None),
                target_state=((256,), "int64", 1, "static", None),
                source_cells=(new.source_cells[0],),
            )
            new = replace(new, source_cells=(new.source_cells[0],
                book.latest_ref(SHAPE_STATE_PAGE, (("foreign_shape", 0), 13))))
        elif poison == "stale_shape_ref":
            new = replace(new, source_cells=old.source_cells)
        elif poison == "wrong_source_shape":
            source["shape"] = (255,)
        elif poison == "different_dtype":
            new = replace(new, dtype="float64")
        elif poison == "different_rank":
            new = replace(new, rank=2)
        elif poison == "different_storage":
            new = replace(new, storage="scalar")
        elif poison == "wrong_target_shape":
            target["shape"] = (255,)
        assert _refresh_linked_value_type_abi_arm(
            old, new, source_contract=source, target_contract=target, book=book,
        ) is None
    finally:
        end_identity_book(token)
