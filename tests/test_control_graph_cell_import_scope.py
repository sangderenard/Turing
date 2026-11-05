"""A control-only lexical read can consult its graph identity before regions."""

from src.compiler.control_source import ControlProgram, ControlUniform, SequenceBlock
from src.compiler.concordance_declarations import (
    CANONICAL_VALUE, CONTROL_UNIFORM_DTYPE, INGESTION, INGEST_SOURCE,
)
from src.compiler.identity_concordance import (
    Mode, Novel, begin_identity_book, end_identity_book,
)
from src.compiler.precompile_to_ssa import lower_control_sections_to_ssa


def test_control_only_uniform_reads_graph_cell_without_prefix_region():
    # Run69's PackedBitstream.to_bytes failed in _graph_cell before the
    # conditional prefix-slice import had executed. This exercises that same
    # early helper through the production control lowering entry.
    control = ControlProgram(
        SequenceBlock(()), (), uniforms=(ControlUniform("count", 40, "int"),),
    )
    book, token = begin_identity_book()
    try:
        scope = ("run69-control-only",)
        canonical = book.post(
            CANONICAL_VALUE, (scope, 40), ("count",), stage=INGESTION,
            provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD,
        )
        module, shortfalls, _ = lower_control_sections_to_ssa(
            control, identity_table={"count": (40,)},
            function_parameters=("count",), value_dtypes={40: "int64"},
            lexical_read_scope=scope,
        )
        assert not shortfalls
        assert "planned_control" in module.functions
        assert not any("planned_region" in name for name in module.functions)
        row, = book.page(CONTROL_UNIFORM_DTYPE).rows()
        receipt = book.latest_ref(CONTROL_UNIFORM_DTYPE, row)
        assert canonical in {source for source, _ in book.edges_into(receipt)}
    finally:
        end_identity_book(token)
