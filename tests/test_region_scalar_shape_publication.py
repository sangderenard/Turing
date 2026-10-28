from pathlib import Path
import numpy as np
import pytest

from src.compiler.concordance_declarations import (
    SSA_VALUE, SSAValueFact, SSAValueOrigin, INGESTION, INGEST_SOURCE,
    CONTROL_CONST, PHI_LOOP_HEADER, CELL_SET,
    CANONICAL_VALUE, VALUE_TYPE_ABI, ValueTypeABIAlternative, ValueTypeABIFact,
)
from src.compiler.identity_concordance import (
    begin_identity_book, end_identity_book, Novel, Derived, Mode, NEW,
    SHAPE_STATE_PAGE, ConcordanceRefusal,
)
from src.compiler.structural_concordance import ssa_structure


@pytest.mark.parametrize("state,dtype,physical", [
    (((), "int64", 0, "static", None), "int64", ("int64", "scalar", 0)),
    (((), "int64", 1, "dynamic", None), "int64", ("int64", "span", 1)),
    (((), "float64", 0, "static", None), "int64", (None, None, None)),
    (((), "unknown", 0, "static", None), "unknown", (None, None, None)),
    (((), "ptr", 0, "static", None), "ptr", (None, None, None)),
])
def test_current_physical_view_uses_exact_shape_authority(state, dtype, physical):
    book, token = begin_identity_book()
    try:
        book.post(SSA_VALUE, ("owner", 73), SSAValueFact(dtype, (), SSAValueOrigin.ADOPTED_GRAPH_ID),
                  stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD)
        book.post(SHAPE_STATE_PAGE, (("shape", 0), 73), ("resolved", state),
                  stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD)
        view, _ = ssa_structure("owner", 73, shape_scope=("shape", 0), book=book)
        assert (view.physical_dtype, view.declared_storage, view.rank) == physical
    finally:
        end_identity_book(token)


def test_current_shape_view_invalidates_and_refuses_conflicting_declaration():
    book, token = begin_identity_book()
    try:
        raw = book.post(SSA_VALUE, ("owner", 73), SSAValueFact("int64", (), SSAValueOrigin.ADOPTED_GRAPH_ID),
                        stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD)
        shape = book.post(SHAPE_STATE_PAGE, (("shape", 0), 73),
                          ("resolved", ((), "int64", 0, "static", None)),
                          stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD)
        historical, first = ssa_structure("owner", 73, shape_scope=("shape", 0), book=book)
        book.post(SHAPE_STATE_PAGE, shape.row, ("resolved", ((), "int64", 1, "dynamic", None)),
                  stage=INGESTION, provenance=Derived((shape,)), mode=Mode.REVISE)
        current, second = ssa_structure("owner", 73, shape_scope=("shape", 0), book=book)
        assert historical.rank == 0 and current.rank == 1 and second.column > first.column
        book.post(SSA_VALUE, raw.row, SSAValueFact("int64", (), SSAValueOrigin.ADOPTED_GRAPH_ID,
                  physical_dtype="int64", storage="scalar", rank=0), stage=INGESTION,
                  provenance=Derived((raw,)), mode=Mode.REVISE)
        with pytest.raises(ConcordanceRefusal, match="physical concordance disagreement"):
            ssa_structure("owner", 73, shape_scope=("shape", 0), book=book)
    finally:
        end_identity_book(token)


def test_current_minted_const_and_phi_share_exact_operand_view():
    book, token = begin_identity_book()
    try:
        root = book.post(SSA_VALUE, ("owner", 0), SSAValueFact(None, (), SSAValueOrigin.ADOPTED_GRAPH_ID),
                         stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD)
        seed = book.post(SSA_VALUE, ("owner", NEW), SSAValueFact("int64", (), SSAValueOrigin.MINTED),
                         stage=INGESTION, provenance=Novel(CONTROL_CONST, (root,)), mode=Mode.REVISE)
        operands = book.post(CELL_SET, ("owner", 1), (seed.key,), stage=INGESTION,
                             provenance=Derived((seed,)), mode=Mode.CONCORD)
        phi = book.post(SSA_VALUE, ("owner", NEW), SSAValueFact("int64", (), SSAValueOrigin.MINTED),
                        stage=INGESTION, provenance=Novel(PHI_LOOP_HEADER, (operands,)), mode=Mode.REVISE)
        first, old = ssa_structure("owner", phi.row[1], book=book)
        assert (first.physical_dtype, first.declared_storage, first.rank) == ("int64", "scalar", 0)
        book.post(SSA_VALUE, seed.row, SSAValueFact("int64", (), SSAValueOrigin.MINTED,
                  physical_dtype="int64", storage="scalar", rank=0), stage=INGESTION,
                  provenance=Derived((seed,)), mode=Mode.REVISE)
        second, refreshed = ssa_structure("owner", phi.row[1], book=book)
        assert second == first and refreshed.column > old.column
    finally:
        end_identity_book(token)


def test_current_physical_view_tracks_exact_source_abi_revisions():
    book, token = begin_identity_book()
    try:
        source = book.post(CANONICAL_VALUE, ("source", 73), ("source",),
                           stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD)
        book.post(SSA_VALUE, ("owner", 73), SSAValueFact("int64", (), SSAValueOrigin.ADOPTED_GRAPH_ID),
                  stage=INGESTION, provenance=Derived((source,)), mode=Mode.CONCORD)
        def abi(storage, rank):
            return ValueTypeABIFact(("builtins.int",), (), (
                ValueTypeABIAlternative("builtins.int", storage, "int64", (), rank,
                                        (), (source,), "dynamic" if rank else "static"),
            ))
        declaration = book.post(VALUE_TYPE_ABI, source.row, abi("scalar", 0),
                                stage=INGESTION, provenance=Derived((source,)), mode=Mode.CONCORD)
        first, old = ssa_structure("owner", 73, book=book)
        assert (first.physical_dtype, first.declared_storage, first.rank) == ("int64", "scalar", 0)
        book.post(VALUE_TYPE_ABI, source.row, abi("span", 1), stage=INGESTION,
                  provenance=Derived((declaration, source)), mode=Mode.REVISE)
        second, current = ssa_structure("owner", 73, book=book)
        assert second.rank == 1 and second.declared_storage == "span" and current.column > old.column
    finally:
        end_identity_book(token)


@pytest.mark.parametrize("backend", ["c", "llvm", "fortran"])
def test_scalar_loop_update_native_and_audit(backend, tmp_path):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.identity_concordance import concordance_report

    source = '''def advance(begin: int, end: int):
    atom_index = begin
    while atom_index < end:
        atom_index += 1
    return atom_index
def root(begin: int, end: int):
    return advance(begin, end)
'''
    module, outputs, exports = lower_ast_source_to_ssa(
        source, "root", name="scalar_shape_publication",
        extraction_contract=Path(__file__).resolve().parents[1] / "extraction_contracts/program_extraction.yaml",
        progress=lambda _: None, progress_bars=False,
    )
    assert "0 finding(s)" in concordance_report(module)
    root = exports[0]
    args = module.functions[root].args
    feeds = {args[0].id: np.array([3], dtype="int64"), args[1].id: np.array([8], dtype="int64")}
    if backend == "c":
        from src.compiler.ssa_c_backend import emit_ssa_to_c
        artifact = emit_ssa_to_c(module, root)
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path)
        execution = artifact.prepare_execution(feeds)
    elif backend == "llvm":
        from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution
        artifact = emit_ssa_function_to_llvm(module, root)
        assert artifact.complete, artifact.shortfalls
        native = compile_artifact(artifact, directory=tmp_path)
        execution = prepare_artifact_execution(native, feeds)
    else:
        from src.compiler.ssa_fortran_backend import emit_module, FortranCoreNative, fortran_compiler
        from src.compiler.fortran_c_shell import compile_fortran_module_c_shell
        if fortran_compiler() is None:
            pytest.skip("Fortran compiler unavailable")
        artifact = emit_module(module, name="scalar_shape_publication", outputs={root: outputs[root]}, extra_roots=(root,), progress=lambda _: None)
        assert artifact.complete, artifact.shortfalls
        native = compile_fortran_module_c_shell(artifact, {}, tmp_path, entrypoint=root, library=True)
        execution = FortranCoreNative(native.executable_path, artifact.api.entry_point(root), artifact.api).prepare_execution(feeds)
    for begin, end in ((3, 8), (9, 4), (0, 1)):
        execution.buffers[args[0].id][0] = begin
        execution.buffers[args[1].id][0] = end
        execution.run()
        assert execution.buffers[outputs[root][0].id].item() == max(begin, end)
