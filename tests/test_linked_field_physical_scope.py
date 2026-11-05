"""A linked field contract belongs to its source control scope."""

import networkx as nx

from src.compiler.concordance_declarations import (
    CANONICAL_VALUE, CLASS_FIELD_DECLARATION, FieldFact,
    INGESTION, INGEST_SOURCE, SSA_VALUE,
)
from src.compiler.fortran_c_shell import _publish_linked_field_physical_rows
from src.compiler.identity_concordance import (
    Mode, Novel, begin_identity_book, end_identity_book,
)
from src.compiler.ssa_record_return_state import posted_ssa_value_fact
from src.transmogrifier.ssa import BasicBlock, Function, IRModule, SSAValue
from tests.compiler_provenance_inputs import publish_authored_ssa


def test_linked_field_physical_row_stays_in_source_family():
    book, token = begin_identity_book()
    try:
        value_id = 8
        source_scope = "source@control:0"
        helper_scope = "helper@control:0"

        def function(name, scope):
            formal = SSAValue(value_id, "int64")
            return Function(
                name, [formal], {"entry": BasicBlock("entry", [])},
                metadata={"tensor_shape_concordance_scope": scope},
            )

        source = function("source", source_scope)
        # Names deliberately imply the opposite ownership. The control
        # scopes and separate books remain the authority for value 8.
        region = function("compiler_broadcast_extents__region_0", source_scope)
        helper = function("source__planned_region_1", helper_scope)
        helper_book, helper_token = begin_identity_book()
        try:
            publish_authored_ssa({helper.name: helper})
        finally:
            end_identity_book(helper_token)
        helper.metadata["accepted_ssa_component"] = {
            "identity_book": helper_book,
            "artifact_sha256": "fixture-helper-digest",
        }
        module = IRModule({item.name: item for item in (source, region, helper)})
        publish_authored_ssa({source.name: source, region.name: region})

        graph = nx.DiGraph()
        graph.add_node(value_id, value_id=value_id)
        graph.graph.update(
            canonical_value_ids=True,
            lexical_read_scope=("source_reads", 0),
        )
        book.post(
            CANONICAL_VALUE, (("source_reads", 0), value_id), ("field",),
            stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()),
            mode=Mode.CONCORD,
        )
        declaration = book.post(
            CLASS_FIELD_DECLARATION,
            ("source_module", "SourceRecord", "field"),
            FieldFact("class", "AbstractTensor", ()),
            stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()),
            mode=Mode.CONCORD,
        )
        contract = {
            value_id: {
                "program_abi_parameter": "record_field",
                "dtype": "int64", "storage": "span", "rank": 1,
                "program_abi_field_declaration_cell": declaration,
            },
        }
        _publish_linked_field_physical_rows(module, source, graph, contract)

        for function in (source, region):
            posted = posted_ssa_value_fact(function, value_id, book=book)
            assert (posted.dtype, posted.storage, posted.rank) == (
                "int64", "span", 1,
            )
        assert source.args[0].accounting["program_abi_storage"] == "span"
        assert source.args[0].accounting["program_abi_rank"] == 1
        posted_source = book.latest_ref(SSA_VALUE, (source_scope, value_id))
        assert declaration in {
            cause for cause, _stage in book.edges_into(posted_source)
        }
        foreign = posted_ssa_value_fact(helper, value_id, book=helper_book)
        assert foreign.dtype == "int64"
        assert (foreign.storage, foreign.rank) == (None, None)
        assert helper.args[0].accounting == {}
        assert not tuple(helper_book.page(CLASS_FIELD_DECLARATION).rows())
    finally:
        end_identity_book(token)
