import numpy as np
import pytest

from src.compiler.concordance_declarations import SSA_VALUE, SSAValueFact, SSAValueOrigin, INGESTION, INGEST_SOURCE
from src.compiler.identity_concordance import begin_identity_book, end_identity_book, Derived, Novel, Mode, ConcordanceRefusal
from src.compiler.structural_concordance import register_sequence_row_projections, function_value_structure
from src.transmogrifier.ssa import SSASequenceTable, SSASequenceDescriptor, SSAValue, Function, BasicBlock, Instr, IRModule


def _projection(book, *, descriptor_dtype="unknown"):
    column = SSAValue(38, "unknown", (1,))
    index, address, result = SSAValue(1, "int64"), SSAValue(2, "ptr"), SSAValue(41, "float64")
    scope = "root@control:0"
    source = book.post(SSA_VALUE, (scope, column.id),
                       SSAValueFact("unknown", (1,), SSAValueOrigin.ADOPTED_GRAPH_ID),
                       stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD)
    book.post(SSA_VALUE, (scope, result.id),
              SSAValueFact("float64", (), SSAValueOrigin.ADOPTED_GRAPH_ID),
              stage=INGESTION, provenance=Derived((source,)), mode=Mode.CONCORD)
    for value in (index, address):
        book.post(SSA_VALUE, (scope, value.id),
                  SSAValueFact(value.dtype, (), SSAValueOrigin.ADOPTED_GRAPH_ID),
                  stage=INGESTION, provenance=Derived((source,)), mode=Mode.CONCORD)
    table = SSASequenceTable(owner="root", book=book)
    table.register(SSASequenceDescriptor(
        38, (38,), 100, 101, column_dtypes=(descriptor_dtype,), column_shapes=((),),
    ), sources=(source,), stage=INGESTION)
    attrs = {"binding": "ssa_sequence_row_load", "sequence_id": 38, "column": 0}
    function = Function("root", [column], {"entry": BasicBlock("entry", [
        Instr("Const", [], index, attributes={"value": 0}),
        Instr("GetElementPtr", [column, index], address, attributes=attrs),
        Instr("Load", [address], result, attributes=attrs), Instr("Ret", [result], None),
    ])}, metadata={"tensor_shape_concordance_scope": scope})
    register_sequence_row_projections(function, table, book=book)
    return function, table, source, result


def test_sequence_element_follows_exact_column_revision_and_stays_unknown_first():
    book, token = begin_identity_book()
    try:
        function, table, source, result = _projection(book)
        unresolved, old = function_value_structure(function, result.id, book=book)
        assert unresolved.dtype is None and unresolved.physical_dtype is None
        book.post(SSA_VALUE, source.row,
                  SSAValueFact("uint64", (1,), SSAValueOrigin.ADOPTED_GRAPH_ID,
                               physical_dtype="uint64", storage="span", rank=1),
                  stage=INGESTION, provenance=Derived((source,)), mode=Mode.REVISE)
        current, new = function_value_structure(function, result.id, book=book)
        assert (current.dtype, current.physical_dtype, current.rank) == ("uint64", "uint64", 0)
        assert new.column > old.column
    finally:
        end_identity_book(token)


def test_sequence_known_schema_disagrees_with_different_column_type():
    book, token = begin_identity_book()
    try:
        function, table, source, result = _projection(book, descriptor_dtype="uint64")
        book.post(SSA_VALUE, source.row, SSAValueFact("float64", (1,), SSAValueOrigin.ADOPTED_GRAPH_ID),
                  stage=INGESTION, provenance=Derived((source,)), mode=Mode.REVISE)
        with pytest.raises(ConcordanceRefusal, match="sequence column type disagreement"):
            function_value_structure(function, result.id, book=book)
    finally:
        end_identity_book(token)


@pytest.mark.parametrize("backend", ["c", "llvm", "fortran"])
def test_native_sequence_element_refinement_keeps_uint64_high_bits(backend, tmp_path):
    from src.compiler.ir_indexing import _propagate_scalar_dtypes
    book, token = begin_identity_book()
    try:
        function, table, source, result = _projection(book)
        book.post(SSA_VALUE, source.row,
                  SSAValueFact("uint64", (1,), SSAValueOrigin.ADOPTED_GRAPH_ID,
                               physical_dtype="uint64", storage="span", rank=1),
                  stage=INGESTION, provenance=Derived((source,)), mode=Mode.REVISE)
        # The existing scalar settlement materializes the current concordance
        # view. Emission receives a typed Load, never an implicit float cast.
        _propagate_scalar_dtypes({"root": function})
        assert result.dtype == "uint64" and function.args[0].dtype == "uint64"
        # Emit this direct-IR fixture through the normal standalone-IR path;
        # the live-book assertions above concern the type projection itself.
        end_identity_book(token)
        token = None
        function.metadata = {}
        module = IRModule({"root": function})
        feeds = {38: np.array([2**64 - 1], dtype="uint64")}
        if backend == "c":
            from src.compiler.ssa_c_backend import emit_ssa_to_c
            artifact = emit_ssa_to_c(module, "root")
            assert artifact.complete, artifact.shortfalls
            artifact.compile(tmp_path)
            execution = artifact.prepare_execution(feeds)
        elif backend == "llvm":
            from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution
            artifact = emit_ssa_function_to_llvm(module, "root")
            assert artifact.complete, artifact.shortfalls
            native = compile_artifact(artifact, directory=tmp_path)
            execution = prepare_artifact_execution(native, feeds)
        else:
            from src.compiler.ssa_fortran_backend import emit_module, FortranCoreNative, fortran_compiler
            from src.compiler.fortran_c_shell import compile_fortran_module_c_shell
            if fortran_compiler() is None:
                pytest.skip("Fortran compiler unavailable")
            artifact = emit_module(module, name="sequence_element", outputs={"root": (result,)},
                                   extra_roots=("root",), progress=lambda _: None)
            assert artifact.complete, artifact.shortfalls
            native = compile_fortran_module_c_shell(artifact, {}, tmp_path, entrypoint="root", library=True)
            execution = FortranCoreNative(native.executable_path, artifact.api.entry_point("root"),
                                         artifact.api).prepare_execution(feeds)
        assert execution.run().buffers[result.id].item() == 2**64 - 1
    finally:
        if token is not None:
            end_identity_book(token)


@pytest.mark.parametrize("pool_dtype", ["int64", "unknown"])
def test_native_child_pool_storage_keeps_exact_descriptor_owner_and_layout(pool_dtype):
    from dataclasses import replace
    from src.transmogrifier.ssa import SSAChildTablePoolDescriptor, SSASequenceCapacityPolicy
    from src.compiler.structural_concordance import storage_structure
    from src.compiler.concordance_declarations import SEQUENCE_DESCRIPTOR
    book, token = begin_identity_book()
    try:
        source = book.post(SSA_VALUE, ("storage_owner", 80),
                          SSAValueFact("ptr", (), SSAValueOrigin.ADOPTED_GRAPH_ID),
                          stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD)
        pool = SSAChildTablePoolDescriptor(
            1, (90,), 92, 93, 94, status_value_id=95, column_dtypes=(pool_dtype,),
        )
        descriptor = SSASequenceDescriptor(
            80, (80, 81), 82, 83, status_address_id=84,
            column_dtypes=("int64", "int64"), key_columns=(0,), child_table_pool=pool,
            capacity_policy=SSASequenceCapacityPolicy.DYNAMIC, heap_owner_value_id=89,
        )
        table = SSASequenceTable(owner="storage_owner", book=book)
        table.register(descriptor, sources=(source,), stage=INGESTION)
        view, first = storage_structure("sequence", table.owner, 80, book=book)
        assert dict(view.children)[1].dtype == "int64"  # semantic column unchanged
        native_view = dict(view.children)["native_storage"]
        if pool_dtype == "unknown":
            assert native_view.kind == "unresolved" and native_view.children == ()
            return
        native = dict(native_view.children)
        for member, dtype in ((90, "int64"), (92, "int64"), (95, "int32")):
            cell = native[member]
            assert (cell.kind, cell.dtype, cell.physical_dtype, cell.storage) == ("reference", "ptr", "ptr", "heap_cell")
            assert dict(cell.children)["element"].dtype == dtype
            assert dict(cell.children)["owner"].value_id == 89
            assert cell.source == book.latest_ref(SEQUENCE_DESCRIPTOR, (table.owner, 80))
        assert native[89].storage == "heap_owner"
        revised_source = book.post(
            SSA_VALUE, source.row, SSAValueFact("ptr", (), SSAValueOrigin.ADOPTED_GRAPH_ID,
                                               physical_dtype="ptr", storage="reference", rank=0),
            stage=INGESTION, provenance=Derived((source,)), mode=Mode.REVISE,
        )
        with pytest.raises(ValueError, match="conflicting SSA sequence descriptor"):
            table.register(replace(descriptor, child_table_pool=replace(pool, column_dtypes=("uint64",))),
                           sources=(revised_source,), stage=INGESTION)
    finally:
        end_identity_book(token)
