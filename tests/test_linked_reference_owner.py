"""Reference witnesses of accepted regions stay in their owning book."""

import hashlib

import pytest

from src.compiler.concordance_declarations import CROSS_FUNCTION_REFERENCES
from src.compiler.identity_concordance import (
    ConcordanceRefusal, begin_identity_book, end_identity_book,
)
from src.compiler.tensor_ssa_lowering import propagate_repository_ssa_call_metadata
from src.transmogrifier.ssa import BasicBlock, Function, Instr, IRModule, SSAValue
from tests.compiler_provenance_inputs import publish_authored_ssa


@pytest.mark.parametrize("valid_digest", (True, False))
def test_accepted_region_reference_witnesses_are_imported_from_the_owner(valid_digest):
    value = SSAValue(5, "int64")
    functions = {
        name: Function(name, [value], {
            "entry": BasicBlock("entry", [Instr("Ret", [value], None)]),
        }, metadata={
            "tensor_shape_concordance_scope": "accepted@control:0",
            "source_region_integral": {"owner": "accepted"},
        })
        for name in ("region_a", "region_b")
    }
    owner, token = begin_identity_book()
    try:
        publish_authored_ssa(functions)
    finally:
        end_identity_book(token)
    digest = hashlib.sha256(b"accepted-reference-fixture").hexdigest()
    for function in functions.values():
        function.metadata["accepted_ssa_component"] = {
            "schema": "turing.accepted-ssa-component.v1",
            "identity_book": owner,
            "artifact_sha256": digest if valid_digest else None,
        }

    destination, token = begin_identity_book()
    try:
        module = IRModule(functions, metadata={"identity_book": destination})
        if not valid_digest:
            with pytest.raises(ConcordanceRefusal, match="verified artifact digest"):
                propagate_repository_ssa_call_metadata(module)
            return
        assert not propagate_repository_ssa_call_metadata(module)
        row = ("accepted", 5)
        assert destination.page(CROSS_FUNCTION_REFERENCES).latest(row) == (
            "region_a", "region_b",
        )
        imported = destination.page("concordance_imported_cell")
        assert imported.rows()
        for row in imported.rows():
            assert row[0] == digest
            receipt = imported.latest(row)
            assert receipt.owner is owner
            owner._source_stamp(receipt.source)
    finally:
        end_identity_book(token)
