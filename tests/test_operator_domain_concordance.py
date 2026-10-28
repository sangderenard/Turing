"""Operation domains retain their exact converted operands on the book."""
import pytest
from src.compiler.concordance_declarations import (
    CANONICAL_VALUE, INGESTION, INGEST_SOURCE, OPERATOR_RESULT_TYPE, SSA_VALUE,
    SSAValueFact, SSAValueOrigin,
)
from src.compiler.hierarchical_plan import PlanClosure, PlanLine, plan_region_to_ssa_instrs
from src.compiler.identity_concordance import (
    begin_identity_book, end_identity_book, Novel, Mode, Derived,
    SHAPE_STATE_PAGE, record_shape_transformation,
)
from src.compiler.structural_concordance import ssa_structure


@pytest.mark.parametrize("sibling_copy,multiple_owners,supply_scope", [
    (False, False, False), (True, False, False), (True, True, True), (True, True, False),
])
def test_integer_operation_domain_records_cast_and_literal_identities(sibling_copy, multiple_owners, supply_scope):
    book, token = begin_identity_book()
    try:
        scope = ("operation_source", 0)
        cells = {
            value: book.post(CANONICAL_VALUE, (scope, value), ("source", value),
                             stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD)
            for value in (1, 72, 73)
        }
        shape_cells = cells
        if sibling_copy:
            origin = cells
            stage = book.registry.declare_stage("read_scope_fork")
            def copied(destination):
                return {value: book.post(
                    CANONICAL_VALUE, (destination, value), ("source", value),
                    stage=stage, provenance=Derived((cell,)), mode=Mode.CONCORD,
                ) for value, cell in origin.items()}
            shape_cells = copied(("shape_copy", 0))
            scope = ("operation_copy", 1)
            cells = copied(scope)
        shape_scope = ("exact_shape_owner", 13)
        geometry = ((), "bool", 0, "static", None)
        for operand, role in ((1, "lhs"), (72, "rhs")):
            record_shape_transformation(
                shape_scope, operand, shape_scope, 73,
                stage="graph_tensor_descriptor", operation="Add",
                source_state=geometry if operand == 1 else None,
                target_state=geometry, role=role, source_cells=(shape_cells[operand],),
            )
        other_shape_scope = ("other_exact_shape_owner", 14)
        if multiple_owners:
            other_cells = copied(("other_shape_copy", 2))
            for operand, role in ((1, "lhs"), (72, "rhs")):
                record_shape_transformation(
                    other_shape_scope, operand, other_shape_scope, 73,
                    stage="graph_tensor_descriptor", operation="Add",
                    source_state=geometry if operand == 1 else None,
                    target_state=geometry, role=role, source_cells=(other_cells[operand],),
                )
        region = PlanClosure(
            "region_8", (1,), (
                PlanLine.create("Const", outputs=(72,), attributes={"value": 1}),
                PlanLine.create("Add", inputs=(1, 72), outputs=(73,), input_roles=("lhs", "rhs")),
            ), closure_id=8,
            value_shapes=((1, (), "bool"), (72, (), "int64"), (73, (), "bool")),
            value_ranks=((1, 0), (72, 0), (73, 0)),
        )
        if multiple_owners and not supply_scope:
            from src.compiler.identity_concordance import ConcordanceRefusal
            with pytest.raises(ConcordanceRefusal, match="ambiguous shape identities"):
                plan_region_to_ssa_instrs(region, function_scope="operation_owner", lexical_read_scope=scope)
            return
        instructions = plan_region_to_ssa_instrs(
            region, function_scope="operation_owner", lexical_read_scope=scope,
            shape_scope=shape_scope if supply_scope else None,
        )
        if multiple_owners:
            assert book.page(SHAPE_STATE_PAGE).latest((other_shape_scope, 73))[1] == geometry
        operation = next(item for item in instructions if item.op == "Add")
        assert operation.res.dtype == "int64"
        assert operation.args[0].id != 1  # existing common-domain conversion
        row = ("operation_owner", 8, "region_8", 73)
        assert book.page(OPERATOR_RESULT_TYPE).latest(row) == ("add", "int64")
        sources = {cell for cell, _ in book.edges_into(book.latest_ref(OPERATOR_RESULT_TYPE, row))}
        assert cells[73] in sources and cells[72] in sources
        assert book.latest_ref(SSA_VALUE, ("operation_owner", operation.args[0].id)) in sources
        current_shape = book.page(SHAPE_STATE_PAGE).latest((shape_scope, 73))
        assert current_shape[1] == ((), "int64", 0, "static", None)
        op_ref = book.latest_ref(OPERATOR_RESULT_TYPE, row)
        book.post(SSA_VALUE, ("operation_owner", 73),
                  SSAValueFact("int64", (), SSAValueOrigin.ADOPTED_GRAPH_ID),
                  stage=INGESTION, provenance=Derived((cells[73], op_ref)), mode=Mode.CONCORD)
        view, _ = ssa_structure("operation_owner", 73, shape_scope=shape_scope, book=book)
        assert (view.physical_dtype, view.rank, view.declared_storage) == ("int64", 0, "scalar")
    finally:
        end_identity_book(token)


import numpy as np
import pytest
from pathlib import Path


@pytest.mark.parametrize("backend", ["c", "llvm", "fortran"])
def test_boolean_integer_operation_native(backend, tmp_path):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.identity_concordance import concordance_report
    source = """def root(flag: bool):
    return flag + 1
"""
    module, outputs, exports = lower_ast_source_to_ssa(
        source, "root", name="bool_integer_update",
        extraction_contract=Path(__file__).resolve().parents[1] / "extraction_contracts/program_extraction.yaml",
        progress=lambda _: None, progress_bars=False,
    )
    assert "0 finding(s)" in concordance_report(module)
    root = exports[0]
    argument = module.functions[root].args[0]
    feeds = {argument.id: np.array([False], dtype="bool")}
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
        execution = prepare_artifact_execution(compile_artifact(artifact, directory=tmp_path), feeds)
    else:
        from src.compiler.ssa_fortran_backend import emit_module, FortranCoreNative, fortran_compiler
        from src.compiler.fortran_c_shell import compile_fortran_module_c_shell
        if fortran_compiler() is None:
            pytest.skip("Fortran compiler unavailable")
        artifact = emit_module(module, name="bool_integer_update", outputs={root: outputs[root]}, extra_roots=(root,), progress=lambda _: None)
        assert artifact.complete, artifact.shortfalls
        native = compile_fortran_module_c_shell(artifact, {}, tmp_path, entrypoint=root, library=True)
        execution = FortranCoreNative(native.executable_path, artifact.api.entry_point(root), artifact.api).prepare_execution(feeds)
    for flag in (False, True, False):
        execution.buffers[argument.id][0] = flag
        execution.run()
        assert execution.buffers[outputs[root][0].id].item() == int(flag) + 1


def test_unknown_operand_does_not_publish_a_numeric_domain_from_default_dtype():
    book, token = begin_identity_book()
    try:
        region = PlanClosure(
            "unknown_region", (1,), (
                PlanLine.create("Const", outputs=(2,), attributes={"value": 1}),
                PlanLine.create("Add", inputs=(1, 2), outputs=(3,), input_roles=("lhs", "rhs")),
            ), closure_id=9, value_shapes=((2, (), "int64"),),
        )
        plan_region_to_ssa_instrs(region, function_scope="unknown_owner")
        assert book.page(OPERATOR_RESULT_TYPE).latest(("unknown_owner", 9, "unknown_region", 3)) is None
    finally:
        end_identity_book(token)


def test_boolean_loop_without_representation_conversion_remains_refused():
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.identity_concordance import ConcordanceRefusal
    source = """def advance(begin: bool, end: int):
    atom_index = begin
    while atom_index < end:
        atom_index += 1
    return atom_index

def root(end: int):
    return advance(False, end)
"""
    # This needs a bool/int loop-carried representation transition. Recording
    # the Add domain alone must not bless incompatible Phi storage as native.
    with pytest.raises(ConcordanceRefusal, match="physical concordance disagreement"):
        lower_ast_source_to_ssa(
            source, "root", name="bool_integer_update",
            extraction_contract=Path(__file__).resolve().parents[1] / "extraction_contracts/program_extraction.yaml",
            progress=lambda _: None, progress_bars=False,
        )
