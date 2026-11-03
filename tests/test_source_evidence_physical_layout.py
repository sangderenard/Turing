"""Physical layout is read only from posted SSA storage fields."""

import ast
import contextlib
import io
from pathlib import Path

from src.common.tensors.topological_reducer import (
    _source_numeric_scope, reduce_abstract_tensor_topology,
)
from src.compiler.concordance_declarations import (
    SOURCE_VALUE_CLASS, SSA_VALUE, VALUE_TYPE_ABI,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import begin_identity_book, end_identity_book
from src.compiler.source_evidence_resolver import (
    PhysicalLayoutEvidence, UnresolvedSourceDemand, require_physical_layout,
)
from src.compiler.ssa_record_return_state import function_scope_of
from src.transmogrifier.graph.graph_express2 import ProcessGraph


ROOT = Path(__file__).resolve().parents[1]


def test_scalar_formal_reads_posted_physical_dtype_storage_and_rank():
    contract = ExtractionContract(
        ROOT / 'extraction_contracts/program_extraction.yaml'
    ).with_execution_file(
        ROOT / 'extraction_contracts/vehicle_full_native_execution.yaml'
    )
    module, _outputs, exports = lower_ast_source_to_ssa(
        'def root(x: int) -> int:\n    return x + 1\n',
        'root', name='physical_layout_evidence',
        extraction_contract=contract, progress=lambda _message: None,
    )
    function = module.functions[exports[0]]
    book = module.metadata['identity_book']
    scope = function_scope_of(function)
    value_id = int(function.args[0].id)
    before = tuple(book.clock)
    result = require_physical_layout(scope, value_id, book=book)
    assert isinstance(result, PhysicalLayoutEvidence)
    assert (result.physical_dtype, result.storage, result.rank, result.shape) == (
        'int64', 'scalar', 0, ())
    assert result.arena is False and result.device is None
    assert result.subject_cell == book.latest_ref(SSA_VALUE, (scope, value_id))
    assert result.proof_cells
    assert result.control_scope == scope
    # The planned-region literal now carries the exact physical ABI arm.
    region_literal = require_physical_layout(scope, 1, book=book)
    assert isinstance(region_literal, PhysicalLayoutEvidence)
    assert region_literal.physical_dtype == 'int64'
    assert any(cell.page == VALUE_TYPE_ABI
               for cell in region_literal.proof_cells)
    assert tuple(book.clock) == before


def test_source_semantic_class_does_not_synthesize_physical_layout():
    book, token = begin_identity_book()
    try:
        process = ProcessGraph(materialize_memory=False)
        with contextlib.redirect_stdout(io.StringIO()):
            process.build_from_ast(ast.parse(
                'def choose(first: int) -> int:\n    return first\n'))
        reduce_abstract_tensor_topology(process)
        graph = process.function_table.entry('choose').graph
        scope = _source_numeric_scope(graph)
        first = next(
            int(node) for node, data in graph.G.nodes(data=True)
            if str(data.get('type')) == 'Input'
            and (data.get('attributes') or {}).get('binding_name') == 'first'
        )
        assert book.latest_ref(SOURCE_VALUE_CLASS, (scope, first)) is not None
        before = tuple(book.clock)
        result = require_physical_layout(scope, first, book=book)
        assert isinstance(result, UnresolvedSourceDemand)
        assert result.missing == 'missing_ssa_value'
        assert tuple(book.clock) == before
    finally:
        end_identity_book(token)
