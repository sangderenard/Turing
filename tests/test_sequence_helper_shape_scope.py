"""Generated sequence storage helpers do not inherit a source graph namespace."""
from pathlib import Path

import numpy as np
import pytest

from src.compiler.fortran_c_shell import (
    lower_ast_source_to_ssa, compile_fortran_module_c_shell,
)
from src.compiler.identity_concordance import CorrelationTable


SOURCE = '''def batch(count: int):
    statuses = [0] * count
    tokens = [count] * count
    return 0, statuses, tokens
def root(n: int):
    envelope, statuses, tokens = batch(1)
    for index in range(n):
        envelope, statuses, tokens = batch(2)
        if envelope != 0 or statuses[0] != 0:
            return -1
    return len(statuses) * 1000 + len(tokens) * 100 + tokens[0]
'''


def test_registered_replacement_scope_is_independent():
    module, _outputs, _exports = lower_ast_source_to_ssa(
        SOURCE, "root", name="helper_scope_identity",
        extraction_contract=Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml",
        progress=lambda _: None, progress_bars=False,
    )
    assert CorrelationTable.build(module).findings(module) == []
    helpers = [function for function in module.functions.values()
               if function.metadata.get("ssa_sequence_operation") == "replace"]
    assert len(helpers) == 2
    for helper in helpers:
        assert "shape_scope" not in helper.metadata
        assert {value.dtype for value in helper.args} <= {"int", "int64"}


@pytest.mark.parametrize("lane", ["C", "LLVM", "Fortran"])
def test_registered_replacement_native_copy(tmp_path, lane):
    """Execute the incumbent replacement algorithm with explicit caller storage."""
    from src.compiler.ir_sequence_tables import lower_sequence_replace
    from src.transmogrifier.ssa import IRModule, SSASequenceDescriptor, SSASequenceTable
    destination = SSASequenceDescriptor(10, (10,), 11, 12, column_dtypes=("int64",))
    source = SSASequenceDescriptor(20, (20,), 21, 22, column_dtypes=("int64",))
    lowering = lower_sequence_replace(destination, source, function_name="root")
    assert lowering.complete, lowering.shortfalls
    helper = lowering.functions[0]
    module = IRModule({helper.name: helper}, sequence_tables={helper.name: SSASequenceTable(
        sequences={10: destination, 20: source},
    )})
    # The established replace ABI is destination(column,length,capacity),
    # source(column,length,capacity). These are its live formal identities.
    feeds = {}
    for index, argument in enumerate(helper.args):
        feeds[argument.id] = np.array(
            ([0, 0, 0, 99] if index == 0 else
             [2**53 + 1, 2**53 + 3, 7, 8] if index == 3 else
             [1] if index == 1 else [3] if index == 4 else [4]), dtype=np.int64,
        )
    result = helper.metadata["named_outputs"][0][1]
    if lane == "C":
        from src.compiler.ssa_c_backend import emit_ssa_to_c
        artifact = emit_ssa_to_c(module, helper.name)
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path)
        prepare = artifact.prepare_execution
    elif lane == "LLVM":
        from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution
        artifact = emit_ssa_function_to_llvm(module, helper.name)
        assert artifact.complete, artifact.shortfalls
        native = compile_artifact(artifact, directory=tmp_path)
        prepare = lambda values: prepare_artifact_execution(native, values)
    else:
        from src.compiler.ssa_fortran_backend import emit_module, FortranCoreNative, fortran_compiler
        if fortran_compiler() is None:
            pytest.skip("native Fortran compiler unavailable")
        returned = next(instruction.args for block in helper.blocks.values()
                        for instruction in block.instrs if instruction.op == "Ret")
        artifact = emit_module(module, outputs={helper.name: returned}, extra_roots=(helper.name,),
                               progress=lambda _: None)
        assert artifact.complete, artifact.shortfalls
        native = compile_fortran_module_c_shell(artifact, {}, tmp_path,
                                               entrypoint=helper.name, library=True)
        prepare = FortranCoreNative(native.executable_path,
                                   artifact.api.entry_point(helper.name),
                                   artifact.api).prepare_execution
    execution = prepare(feeds)
    buffers = execution.run().buffers
    assert buffers[helper.args[0].id].tolist()[:3] == [2**53 + 1, 2**53 + 3, 7]
    assert buffers[helper.args[1].id].item() == 3
    assert buffers[result].item() == 1


@pytest.mark.parametrize("lane", ["C", "LLVM", "Fortran"])
def test_returned_array_replacement_owns_helper_shape_scope(tmp_path, lane):
    module, outputs, exports = lower_ast_source_to_ssa(
        SOURCE, "root", name="helper_shape_scope",
        extraction_contract=Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml",
        progress=lambda _: None, progress_bars=False,
    )
    assert CorrelationTable.build(module).findings(module) == []
    replacements = [function for function in module.functions.values()
                    if function.metadata.get("ssa_sequence_operation") == "replace"]
    assert len(replacements) == 2
    for helper in replacements:
        assert "shape_scope" not in helper.metadata
        for value in helper.args:
            if (value.accounting or {}).get("compiler_frame_member") == 0:
                assert value.dtype == "int64"
    root = module.functions[exports[0]]
    parameter = dict(root.metadata["parameter_names"])["n"]
    result = next(instruction.args[0].id for block in root.blocks.values()
                  for instruction in block.instrs if instruction.op == "Ret")
    if lane == "C":
        from src.compiler.ssa_c_backend import emit_ssa_to_c
        artifact = emit_ssa_to_c(module, exports[0])
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path)
        prepare = artifact.prepare_execution
    elif lane == "LLVM":
        from src.compiler.ssa_llvm_backend import (
            emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution,
        )
        artifact = emit_ssa_function_to_llvm(module, exports[0])
        assert artifact.complete, artifact.shortfalls
        native = compile_artifact(artifact, directory=tmp_path)
        prepare = lambda feeds: prepare_artifact_execution(native, feeds)
    else:
        from src.compiler.ssa_fortran_backend import emit_module, FortranCoreNative, fortran_compiler
        if fortran_compiler() is None:
            pytest.skip("native Fortran compiler unavailable")
        artifact = emit_module(module, outputs={name: outputs[name] for name in exports},
                               extra_roots=exports, progress=lambda _: None)
        assert artifact.complete, artifact.shortfalls
        native = compile_fortran_module_c_shell(artifact, {}, tmp_path,
                                               entrypoint=exports[0], library=True)
        prepare = FortranCoreNative(native.executable_path,
                                   artifact.api.entry_point(exports[0]),
                                   artifact.api).prepare_execution
    for n in (0, 1, 5):
        actual = prepare({parameter: np.array([n], dtype=np.int64)}).run().buffers[result].item()
        assert actual == (1101 if n == 0 else 2202)
