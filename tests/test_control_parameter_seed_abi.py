"""A call-only tensor formal takes its physical seed from its exact ABI."""

import pytest

from src.common.tensors.fused_ir import Meta

from src.compiler.concordance_declarations import (
    CANONICAL_VALUE, INGESTION, INGEST_SOURCE, SSA_VALUE, VALUE_TYPE_ABI,
    ValueTypeABIAlternative, ValueTypeABIFact,
)
from src.compiler.control_source import ControlProgram, SequenceBlock
from src.compiler.identity_concordance import (
    Derived, Mode, Novel, begin_identity_book, end_identity_book,
)
from src.compiler.precompile_to_ssa import lower_control_program_to_ssa
from src.compiler.ssa_record_return_state import function_scope_of


@pytest.mark.parametrize("region_value_meta", [None, {2: Meta((), "unknown")}])
def test_call_only_tensor_parameter_uses_singleton_abi_without_region_meta(
    region_value_meta,
):
    book, token = begin_identity_book()
    try:
        lexical_scope = ("run46_tensor_octets_to_bytes", 0)
        canonical = book.post(
            CANONICAL_VALUE, (lexical_scope, 2), ("Input", "octets"),
            stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()),
            mode=Mode.CONCORD,
        )
        arm = ValueTypeABIAlternative(
            "AbstractTensor", "span", "int64", (), 1,
            (("program_abi", "octets"),), (canonical,),
            metadata_state="dynamic",
        )
        abi = book.post(
            VALUE_TYPE_ABI, (lexical_scope, 2),
            ValueTypeABIFact((arm.identity,), ((arm.identity, 0),), (arm,)),
            stage=INGESTION, provenance=Derived((canonical,)),
            mode=Mode.CONCORD,
        )
        function, shortfalls = lower_control_program_to_ssa(
            ControlProgram(SequenceBlock(())),
            function_name="run46_tensor_octets_to_bytes",
            lexical_read_scope=lexical_scope,
            value_name_histories={"octets": (2,)},
            parameter_names=("octets",),
            region_value_meta=region_value_meta or {},
        )
        assert shortfalls == ()
        (formal,) = function.args
        assert (formal.id, formal.dtype, formal.shape) == (2, "int64", ())
        value_cell = book.latest_ref(
            SSA_VALUE, (function_scope_of(function), 2),
        )
        fact = book.page(SSA_VALUE).cell(value_cell.row, value_cell.column)
        assert (fact.physical_dtype, fact.storage, fact.rank) == (
            "int64", "span", 1,
        )
        causes = {source for source, _stage in book.edges_into(value_cell)}
        assert {canonical, abi} <= causes
    finally:
        end_identity_book(token)
