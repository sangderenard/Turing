"""A call-only tensor formal takes its physical seed from its exact ABI."""

import pytest

from src.common.tensors.fused_ir import Meta

from src.compiler.concordance_declarations import (
    CANONICAL_VALUE, CONTROL_SSA_ENTRY, CONTROL_SSA_REGION, INGESTION,
    INGEST_SOURCE, REGION_SIGNATURE, SSA_VALUE, VALUE_TYPE_ABI,
    RegionSignature, SSAValueFact, SSAValueOrigin, ValueTypeABIAlternative,
    ValueTypeABIFact,
)
from src.compiler.control_source import ControlProgram, SequenceBlock
from src.compiler.identity_concordance import (
    Derived, Mode, Novel, begin_identity_book, end_identity_book,
)
from src.compiler.precompile_to_ssa import lower_control_program_to_ssa
from src.compiler.ssa_record_return_state import function_scope_of


@pytest.mark.parametrize("region_value_meta", [None, {2: Meta((), "unknown")}])
@pytest.mark.parametrize("preposted_unknown", [False, True])
def test_call_only_tensor_parameter_uses_singleton_abi_without_region_meta(
    region_value_meta, preposted_unknown,
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
        control_scope = "run46_tensor_octets_to_bytes@control:0"
        old_seed_cell = None
        if preposted_unknown:
            # Run47 had this earlier row: a region signature adopted the
            # graph ID before PARAMETER_SEED read its already posted ABI.
            signature = book.post(
                REGION_SIGNATURE, (control_scope, 7),
                RegionSignature((2,), ()), stage=CONTROL_SSA_REGION,
                provenance=Derived((canonical,)), mode=Mode.CONCORD,
            )
            old_seed_cell = book.post(
                SSA_VALUE, (control_scope, 2),
                SSAValueFact("unknown", (), SSAValueOrigin.ADOPTED_GRAPH_ID),
                stage=CONTROL_SSA_ENTRY,
                provenance=Derived((signature, canonical)), mode=Mode.REVISE,
            )
        function, shortfalls = lower_control_program_to_ssa(
            ControlProgram(SequenceBlock(())),
            function_name="run46_tensor_octets_to_bytes",
            lexical_read_scope=lexical_scope,
            tensor_shape_concordance_scope=control_scope,
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
        if old_seed_cell is not None:
            assert value_cell.column == old_seed_cell.column + 1
            assert old_seed_cell in causes
            old_fact = book.page(SSA_VALUE).cell(
                old_seed_cell.row, old_seed_cell.column,
            )
            assert old_fact.dtype == "unknown"
            assert (old_fact.physical_dtype, old_fact.storage, old_fact.rank) == (
                None, None, None,
            )
    finally:
        end_identity_book(token)
