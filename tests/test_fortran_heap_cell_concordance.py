"""Forwarded native heap cells keep their descriptor's physical ABI."""
import pytest

from src.compiler.concordance_declarations import SSA_VALUE, SSAValueFact, SSAValueOrigin, INGESTION, INGEST_SOURCE
from src.compiler.identity_concordance import begin_identity_book, end_identity_book, Novel, Derived, Mode
from src.compiler.ssa_fortran_backend import emit_function, FortranEmissionError
from src.transmogrifier.ssa import (
    SSASequenceDescriptor, SSAChildTablePoolDescriptor, SSASequenceCapacityPolicy,
    SSASequenceTable, SSAValue, Function, BasicBlock, Instr,
)


def _forwarded_pool(book, *, child_dtype="uint64"):
    # These are declared standalone SSA parameters. The storage descriptor,
    # rather than optional resident accounting copies, owns their cell roles.
    args = [SSAValue(10, "ptr"), SSAValue(11, "int64", (1,)),
            SSAValue(12, "int64"), SSAValue(13, "ptr", (1,)),
            SSAValue(14, "ptr"), SSAValue(15, "ptr"),
            SSAValue(16, "int64"), SSAValue(17, "int64"), SSAValue(18, "ptr")]
    source = None
    for value in args:
        source = book.post(
            SSA_VALUE, ("forwarded", value.id),
            SSAValueFact(value.dtype, value.shape, SSAValueOrigin.ADOPTED_GRAPH_ID),
            stage=INGESTION, mode=Mode.CONCORD,
            provenance=Novel(INGEST_SOURCE, ()) if source is None else Derived((source,)),
        )
    table = SSASequenceTable(owner="forwarded", book=book)
    table.register(SSASequenceDescriptor(
        10, (10,), 11, 12, column_dtypes=("int64",),
        capacity_policy=SSASequenceCapacityPolicy.DYNAMIC,
        heap_owner_value_id=13, writable=False,
        child_table_pool=SSAChildTablePoolDescriptor(
            0, (14,), 15, 16, 17, status_value_id=18,
            column_dtypes=(child_dtype,), key_columns=(),
        ),
    ), sources=(source,), stage=INGESTION)
    function = Function("forwarded", args, {"entry": BasicBlock("entry", [Instr("Ret", [], None)])})
    return function, table


def test_fortran_forwarded_pool_cells_use_canonical_storage_roles():
    book, token = begin_identity_book()
    try:
        function, table = _forwarded_pool(book)
        result = emit_function(function, sequence_table=table)
        assert result.complete, result.shortfalls
        assert {10, 13, 14, 15, 18}.issubset(result.reference_argument_ids)
        for value_id in (10, 13, 14, 15, 18):
            assert f"type(c_ptr), intent(inout) :: t{value_id}" in result.source
    finally:
        end_identity_book(token)


def test_fortran_resident_heap_layout_cannot_override_descriptor():
    book, token = begin_identity_book()
    try:
        function, table = _forwarded_pool(book)
        next(value for value in function.args if value.id == 14).accounting = {
            "heap_element_dtype": "float64",
        }
        with pytest.raises(FortranEmissionError, match="disagrees with descriptor element layout"):
            emit_function(function, sequence_table=table)
    finally:
        end_identity_book(token)


def test_fortran_unknown_owned_pool_layout_refuses_before_declaration():
    book, token = begin_identity_book()
    try:
        function, table = _forwarded_pool(book, child_dtype="unknown")
        with pytest.raises(FortranEmissionError, match="resolved concordance native-storage view"):
            emit_function(function, sequence_table=table)
    finally:
        end_identity_book(token)


def test_native_dynamic_slice_initializes_private_scalar_cells(tmp_path):
    from pathlib import Path
    import numpy as np
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa, compile_fortran_module_c_shell
    from src.compiler.ssa_fortran_backend import emit_module, FortranCoreNative
    from src.compiler.identity_concordance import CorrelationTable
    root = Path(__file__).resolve().parents[1]
    source = (root / "tests/native_semantics_corpus/c296_dynamic_loop_slice.py").read_text()
    module, outputs, exports = lower_ast_source_to_ssa(
        source, "root", name="dynamic_slice_cell_initialization",
        extraction_contract=root / "extraction_contracts/program_extraction.yaml",
        progress=lambda _: None, progress_bars=False,
    )
    assert not CorrelationTable.build(module).findings(module)
    artifact = emit_module(module, outputs={name: outputs[name] for name in exports},
                           extra_roots=exports, progress=lambda _: None)
    assert artifact.complete, artifact.shortfalls
    built = compile_fortran_module_c_shell(artifact, {}, tmp_path,
                                           entrypoint=exports[0], library=True)
    core = FortranCoreNative(built.executable_path, artifact.api.entry_point(exports[0]), artifact.api)
    parameters = dict(module.functions[exports[0]].metadata["parameter_names"])
    for n, end, expected in ((5, 3, 3), (0, 4, 0), (4, 0, 0)):
        prepared = core.prepare_execution({
            parameters["n"]: np.array([n], dtype="int64"),
            parameters["end"]: np.array([end], dtype="int64"),
        })
        # Reusing the normal prepared frame must reset the same existing
        # scalar cells as well as the private owner after its prior release.
        for _ in range(2):
            execution = prepared.run()
            assert execution.buffers[outputs[exports[0]][0].id].item() == expected
