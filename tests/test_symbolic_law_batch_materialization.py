"""One symbolic law retains causal shape history across native batches."""

import pickle

import numpy as np
import pytest
import sympy as sp

from src.compiler.native_package import piece_from_law
from src.compiler.symbolic_equation_compiler import compile_sympy_equations
from src.compiler.identity_concordance import (
    REGISTRY, IdentityBook, Mode, Novel, Ref, Registry,
    concord_compiler_frame_formals, current_identity_book,
)
from src.compiler.concordance_declarations import (
    COPY_VALUE_SHAPE, CONTROL_SSA, FORMAL_ACTUAL, FORMAL_ACTUAL_OCCURRENCE,
    INGEST_SOURCE, SSA_SHAPE_MATERIALIZATION, SSA_VALUE, VALUE_SHAPE,
    SSAValueFact, SSAValueOrigin,
)
from src.transmogrifier.ssa import BasicBlock, Function, IRModule, Instr, SSAValue


@pytest.mark.parametrize("conditional", (False, True))
def test_same_symbolic_law_materializes_two_native_batch_contracts(tmp_path, conditional):
    x = sp.Symbol("x")
    expression = (sp.Piecewise((sp.Max(x, 1), x > 0), (sp.Integer(0), True))
                  if conditional else x + 1)
    law = compile_sympy_equations(
        (sp.Eq(sp.Symbol("result"), expression, evaluate=False),),
        name="law_batch_materialization",
    )
    book = current_identity_book()
    for index, batch in enumerate((9, 1, 9)):
        piece = piece_from_law(law, "law_batch_materialization", batch,
                               directory=tmp_path / str(index))
        assert current_identity_book() is book
        inputs = np.arange(batch, dtype=float)
        result, = piece(inputs)
        expected = (np.where(inputs > 0, np.maximum(inputs, 1), 0)
                    if conditional else inputs + 1)
        np.testing.assert_array_equal(result, expected)
    page = book.page(SSA_SHAPE_MATERIALIZATION)
    rows = [row for row in page.rows() if "law_batch_materialization" in row[0]]
    assert rows
    for row in rows:
        history = page.history(row)
        assert [fact.shape for _, fact in history] == [(9,), (1,), (9,)]
        for column, fact in history:
            sources = book.edges_into(Ref(SSA_SHAPE_MATERIALIZATION, row, column))
            assert len(sources) == 1
            source, _ = sources[0]
            assert source.page is VALUE_SHAPE
            shape_fact = book.page(VALUE_SHAPE).cells[(source.row, source.column)]
            assert shape_fact.shape == fact.shape
            copy_sources = book.edges_into(source)
            assert len(copy_sources) == 1
            copy_source, _ = copy_sources[0]
            assert copy_source.page is COPY_VALUE_SHAPE
            copy_fact = book.page(COPY_VALUE_SHAPE).cells[
                (copy_source.row, copy_source.column)
            ]
            assert copy_fact.shape == fact.shape
        print(f"same-book materialization {row}: "
              f"{[fact.shape for _, fact in history]} with exact source cells", flush=True)

    # A second statement about this exact call occurrence must still agree.
    # Reuse the real lowered call, change only its actual, and restore it.
    call = next(
        instruction for function in piece.module.functions.values()
        for block in function.blocks.values() for instruction in block.instrs
        if instruction.op == "Call"
        and instruction.attributes.get("callee") == "binary_scalar_double"
    )
    original = call.args[1]
    call.args[1] = call.args[0]
    try:
        with pytest.raises(ValueError, match="formal_actual_occurrence_concordance disagreement"):
            concord_compiler_frame_formals(piece.module)
    finally:
        call.args[1] = original


def test_persisted_legacy_formal_rows_keep_their_schema_and_history():
    # Reproduce a persisted pre-scope vocabulary through its public registry
    # declarations. The native test above exercises real lowering identities.
    registry = Registry()
    for page in REGISTRY.pages.values():
        if page is not FORMAL_ACTUAL_OCCURRENCE:
            registry.declare_page(page.name, page.row_fields, page.fact_type,
                                  private=page.name in REGISTRY.private_pages)
    for stage in REGISTRY.stages.values():
        registry.declare_stage(stage.name)
    for transform in REGISTRY.transforms.values():
        registry.declare_transform(transform.name, transform.arity)
    for reason in REGISTRY.reasons.values():
        registry.declare_reason(reason.name)
    book = IdentityBook(registry=registry)
    legacy_row = ("helper", 3, "caller", "entry", 0, 0)
    book.page(FORMAL_ACTUAL).concord(legacy_row, 20)
    scope = "caller@control:0"
    book.post(SSA_VALUE, (scope, 20),
              SSAValueFact("float64", (), SSAValueOrigin.ADOPTED_GRAPH_ID),
              stage=CONTROL_SSA, provenance=Novel(INGEST_SOURCE, ()),
              mode=Mode.CONCORD)
    actual = SSAValue(20, "float64", accounting={"linked_call_frame_storage": "helper"})
    callee = Function("helper", [SSAValue(3, "float64")], {
        "entry": BasicBlock("entry", [Instr("Ret", [], None)]),
    })
    caller = Function("caller", [actual], {
        "entry": BasicBlock("entry", [
            Instr("Call", [actual], None, attributes={"callee": "helper"}),
            Instr("Ret", [], None),
        ]),
    }, metadata={"tensor_shape_concordance_scope": scope})
    module = IRModule({"caller": caller, "helper": callee},
                      metadata={"identity_book": book})
    persisted = pickle.loads(pickle.dumps(module))
    loaded_book = persisted.metadata["identity_book"]
    legacy = loaded_book.page(FORMAL_ACTUAL)
    before = legacy.history(legacy_row)
    concord_compiler_frame_formals(persisted)
    assert legacy.history(legacy_row) == before
    assert len(loaded_book.registry.pages[FORMAL_ACTUAL.name].row_fields) == 6
    occurrence = loaded_book.page(FORMAL_ACTUAL_OCCURRENCE)
    assert occurrence.rows() == ((*legacy_row, scope),)
    ref = loaded_book.latest_ref(FORMAL_ACTUAL_OCCURRENCE, (*legacy_row, scope))
    sources = loaded_book.edges_into(ref)
    assert len(sources) == 1
    assert sources[0][0].page == SSA_VALUE
    assert sources[0][0].row == (scope, 20)
