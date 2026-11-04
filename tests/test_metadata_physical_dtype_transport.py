"""A Call transports the source row's physical dtype without a cache copy."""

from src.compiler.concordance_declarations import SSA_VALUE
from src.compiler.identity_concordance import begin_identity_book, end_identity_book
from src.compiler.tensor_ssa_lowering import propagate_repository_ssa_call_metadata
from src.transmogrifier.ssa import BasicBlock, Function, Instr, IRModule, SSAValue
from tests.compiler_provenance_inputs import publish_authored_ssa


def test_scalar_call_transport_reads_physical_dtype_from_source_cell():
    source = SSAValue(188, 'float64', accounting={
        'physical_dtype': 'float64', 'program_abi_storage': 'scalar',
        'program_abi_rank': 0,
    })
    formal = SSAValue(68, 'float64')
    callee = Function('binary_value', [formal], {
        'entry': BasicBlock('entry', [Instr('Ret', [formal], None)]),
    })
    caller = Function('binary_double', [source], {
        'entry': BasicBlock('entry', [
            Instr('Call', [source], None, attributes={'callee': callee.name}),
            Instr('Ret', [], None),
        ]),
    })
    functions = {fn.name: fn for fn in (caller, callee)}
    book, token = begin_identity_book()
    try:
        publish_authored_ssa(functions)
        source_cell = book.latest_ref(SSA_VALUE, (caller.name, source.id))
        # The real imported binary_double Load has an empty resident cache,
        # while the source cell on its exact binary_value edge proves f64.
        source.accounting = {}
        module = IRModule(functions, metadata={'identity_book': book})
        propagate_repository_ssa_call_metadata(module)
        target_cell = book.latest_ref(SSA_VALUE, (callee.name, formal.id))
        fact = book.page(SSA_VALUE).cell(target_cell.row, target_cell.column)
        assert (fact.physical_dtype, fact.storage, fact.rank) == (
            'float64', 'scalar', 0,
        )
        assert source_cell in {cell for cell, _ in book.edges_into(target_cell)}
    finally:
        end_identity_book(token)
