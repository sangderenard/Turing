"""A source Call result with a proved dynamic span is not a row payload."""

from dataclasses import replace

import pytest

from src.compiler.concordance_declarations import (
    CANONICAL_VALUE, INGESTION, INGEST_SOURCE, VALUE_TYPE_ABI,
    ValueTypeABIAlternative, ValueTypeABIFact,
)
from src.compiler.identity_concordance import (
    Derived, Mode, Novel, SHAPE_STATE_PAGE, begin_identity_book,
    end_identity_book, record_shape_transformation,
)
from src.compiler.precompile_to_ssa import _concordant_span_payload_exclusion


@pytest.mark.parametrize("poison", (
    None, "wrong_lexical", "wrong_shape", "arm_canonical", "arm_state",
    "direct_state", "record_arm", "unknown_dtype", "unknown_rank",
    "heterogeneous_arms",
))
def test_dynamic_call_span_uses_exact_selected_abi_and_shape_sources(poison):
    # This constructs the same authoritative row pattern observed for Run67
    # lz_encode Call93. The actual saved Run67 book was also checked against
    # this production reader; this test isolates each cause without copying
    # the 198 MB diagnostic artifact into the repository.
    book, token = begin_identity_book()
    try:
        lexical_scope = ("lz_reads", 5)
        shape_scope = ("lz_shape", 4)
        value_id = 93
        canonical = book.post(
            CANONICAL_VALUE, (lexical_scope, value_id), ("authored-call",),
            stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()),
            mode=Mode.CONCORD,
        )
        state = ((), "int64", 1, "dynamic", None)
        record_shape_transformation(
            shape_scope, value_id, shape_scope, value_id,
            stage="graph_tensor_descriptor", operation="Call", role="result",
            source_state=state, target_state=state,
            source_cells=(canonical,),
        )
        state_cell = book.latest_ref(
            SHAPE_STATE_PAGE, (shape_scope, value_id),
        )
        assert state_cell is not None
        arm_sources = tuple(
            cell for cell in (canonical, state_cell)
            if not (
                (poison == "arm_canonical" and cell == canonical)
                or (poison == "arm_state" and cell == state_cell)
            )
        )
        arm = ValueTypeABIAlternative(
            "AbstractTensor",
            "record" if poison == "record_arm" else "span",
            "unknown" if poison == "unknown_dtype" else "int64",
            (), -1 if poison == "unknown_rank" else 1, (), arm_sources,
            metadata_state="dynamic",
        )
        arms = (arm,)
        alternatives = (arm.identity,)
        tags = ()
        if poison == "heterogeneous_arms":
            absent = ValueTypeABIAlternative(
                "builtins.NoneType", "none", "", (), 0, (), (canonical,),
            )
            arms += (absent,)
            alternatives += (absent.identity,)
            tags = ((arm.identity, 0), (absent.identity, 1))
        book.post(
            VALUE_TYPE_ABI, (lexical_scope, value_id),
            ValueTypeABIFact(alternatives, tags, arms),
            stage=INGESTION,
            provenance=Derived((canonical,) if poison == "direct_state"
                               else (canonical, state_cell)),
            mode=Mode.CONCORD,
        )
        query_lexical = ("foreign_reads", 5) if poison == "wrong_lexical" else lexical_scope
        query_shape = ("foreign_shape", 4) if poison == "wrong_shape" else shape_scope
        assert _concordant_span_payload_exclusion(
            book, query_lexical, query_shape, "lz_plan", value_id,
        ) is (poison is None)
        assert not _concordant_span_payload_exclusion(
            book, lexical_scope, shape_scope, "lz_plan", value_id - 1,
        )
    finally:
        end_identity_book(token)
