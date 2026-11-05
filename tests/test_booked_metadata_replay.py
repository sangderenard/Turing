"""A pending tensor descriptor cannot replace a booked metadata Call proof."""

import pytest

from src.compiler import tensor_ssa_lowering
from src.compiler.concordance_declarations import (
    TENSOR_METADATA_CALL_BINDING, TENSOR_METADATA_RESULT_BINDING,
)
from src.compiler.identity_concordance import EDGE_PAGE, identity_book
from tests.test_dynamic_reshape_geometry import _lower


_SOURCE = """def root(bits: AbstractTensor, width: int) -> AbstractTensor:
    shaped = bits.reshape(-1, width)
    return shaped + 1
"""


@pytest.mark.parametrize("poison", (
    "booked_formal", "live_actual", "result_predecessor",
))
def test_pending_descriptor_replay_requires_exact_booked_call(poison):
    original = tensor_ssa_lowering.publish_computed_region_result_metadata
    poisoned = []

    def publish(module, **options):
        book = identity_book(module)
        page = book.page(TENSOR_METADATA_CALL_BINDING)
        candidates = tuple(
            row for row in page.rows()
            if str(row[4]).endswith("__planned_region_1")
            and row[1] in module.functions and row[4] in module.functions
        )
        if not poisoned and len(candidates) == 1:
            row = candidates[0]
            target = module.functions[row[4]]
            tensor_id = int(target.args[0].id)
            table = module.tensor_tables.get(target.name)
            if table is None or table.by_id(tensor_id) is None:
                caller = module.functions[row[1]]
                calls = tuple(
                    instruction for block in caller.blocks.values()
                    for instruction in block.instrs if instruction.op == "Call"
                    and instruction.res is not None
                    and int(instruction.res.id) == int(row[3])
                )
                assert len(calls) == 1
                call = calls[0]
                binding = book.latest_ref(TENSOR_METADATA_CALL_BINDING, row)
                result_causes = tuple(
                    (cause, stage) for cause, stage in book.edges_into(binding)
                    if cause.page == TENSOR_METADATA_RESULT_BINDING
                )
                assert len(result_causes) == 1
                actuals, formals = page.cell(row, binding.column)
                assert len(actuals) == len(formals) == 3
                if poison == "booked_formal":
                    page.cells[(row, binding.column)] = (
                        actuals, (formals[1], formals[1], formals[2]),
                    )
                elif poison == "live_actual":
                    call.args = (
                        call.args[0], call.args[2], call.args[1],
                        *call.args[3:],
                    )
                else:
                    result, stage = result_causes[0]
                    edge = (binding.key, result.key, stage.name)
                    edges = book.page(EDGE_PAGE)
                    edges.cells.pop((edge, 0))
                    edges.stamps.pop((edge, 0))
                    edges.row_columns.pop(edge)
                    edges.scopes[binding.key].pop(edge)
                poisoned.append(poison)
        return original(module, **options)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            tensor_ssa_lowering,
            "publish_computed_region_result_metadata", publish,
        )
        with pytest.raises(
            ValueError, match="computed tensor metadata has no exact live Call binding",
        ):
            _lower(_SOURCE)
    assert poisoned == [poison]
