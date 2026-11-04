"""Source Ret metadata changes derive from its incoming Call projection."""

import pytest

from src.compiler.concordance_declarations import (
    SSA_VALUE, TENSOR_METADATA_RESULT_BINDING, TENSOR_SSA_LOWERING,
)
from src.compiler.identity_concordance import (
    ConcordanceRefusal, Derived, Mode, begin_identity_book, end_identity_book,
)
from src.compiler.precompile_to_ssa import _post_ssa_block_geometry
from src.compiler.ssa_record_return_state import function_scope_of
from src.transmogrifier.ssa import BasicBlock, Function, Instr, SSAValue
from tests.compiler_provenance_inputs import publish_authored_ssa


@pytest.mark.parametrize("mismatch", (None, "callee", "semantic", "metadata"))
def test_source_ret_geometry_requires_its_exact_incoming_projection(mismatch):
    book, token = begin_identity_book()
    try:
        tensor = SSAValue(4, "float64")
        metadata = tuple(SSAValue(identity, "int64") for identity in (5, 6, 7))
        other = (SSAValue(8, "int64"), SSAValue(9, "int64"))
        returned = Instr("Ret", [tensor], None)
        source = Function("source", [tensor], {
            "entry": BasicBlock("entry", [
                *(Instr("Const", [], value, attributes={"value": 1})
                  for value in (*metadata, *other)),
                returned,
            ]),
        })
        aggregate = SSAValue(20, "ssa.aggregate")
        projected = tuple(SSAValue(identity, "int64") for identity in (15, 16, 17))
        caller = Function("caller", list(projected), {
            "entry": BasicBlock("entry", [
                Instr("Call", [], aggregate, attributes={"callee": source.name}),
                Instr("Ret", [aggregate], None),
            ]),
        })
        publish_authored_ssa({source.name: source, caller.name: caller})
        scope = function_scope_of(source)
        prior = _post_ssa_block_geometry(book, scope, source)["entry"]
        # Every returned metadata SSA cell was already cited by the initial
        # geometry. Only the newer cross-frame projection explains this edit.
        row = (function_scope_of(caller), aggregate.id,
               "another_source" if mismatch == "callee" else source.name,
               9 if mismatch == "semantic" else tensor.id)
        fact = (row[3], (5, 6, 8) if mismatch == "metadata" else (5, 6, 7),
                tuple(value.id for value in projected))
        projection = book.post(
            TENSOR_METADATA_RESULT_BINDING, row, fact,
            stage=TENSOR_SSA_LOWERING,
            provenance=Derived(tuple(
                book.latest_ref(SSA_VALUE, (value_scope, value.id))
                for value_scope, values in (
                    (scope, (tensor, *metadata, *other)),
                    (function_scope_of(caller), (aggregate, *projected)),
                ) for value in values
            )), mode=Mode.CONCORD,
        )
        returned.args.extend(metadata)
        if mismatch is not None:
            with pytest.raises(ConcordanceRefusal, match="without a changed source"):
                _post_ssa_block_geometry(book, scope, source)
            return
        current = _post_ssa_block_geometry(book, scope, source)["entry"]
        assert current != prior
        assert projection in {parent for parent, _ in book.edges_into(current)}
        assert _post_ssa_block_geometry(book, scope, source)["entry"] == current
    finally:
        end_identity_book(token)
