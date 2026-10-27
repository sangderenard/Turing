"""Exact homogeneous element classes select the predicate subject ABI."""
from pathlib import Path

import pytest

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable
from src.compiler.ssa_c_backend import emit_ssa_to_c
from src.compiler.ssa_llvm_backend import (
    compile_artifact,
    emit_ssa_function_to_llvm,
    prepare_artifact_execution,
)
from src.compiler.ssa_fortran_backend import (
    FortranCoreNative,
    emit_module,
    fortran_compiler,
)
from src.compiler.fortran_c_shell import compile_fortran_module_c_shell


ROOT = Path(__file__).resolve().parents[1]

INT_BOOL_SOURCE = '''
def validate_int(offsets: list[int]) -> int:
    previous = 0
    for offset in offsets:
        if not isinstance(offset, int) or isinstance(offset, bool):
            return -1
        if offset < previous:
            return -2
        previous = offset
    return previous

def root() -> tuple[int, int]:
    return validate_int([0, 1]), validate_int([False, True])
'''

FLOAT_SOURCE = '''
def validate_float(offsets: list[float]) -> float:
    previous = 0.0
    for offset in offsets:
        if not isinstance(offset, float):
            return -1.0
        if offset < previous:
            return -2.0
        previous = offset
    return previous

def root() -> float:
    return validate_float([1.5])
'''


def _lower(source: str, name: str):
    return lower_ast_source_to_ssa(
        source,
        "root",
        name=name,
        extraction_contract=ROOT / "extraction_contracts/program_extraction.yaml",
        progress=lambda _message: None,
        progress_bars=False,
    )


@pytest.fixture(scope="module")
def int_bool_module():
    return _lower(INT_BOOL_SOURCE, "loop_predicate_intbool_abi")


def test_exact_int_and_bool_loop_predicates_execute_on_llvm(int_bool_module, tmp_path):
    module, outputs, exports = int_bool_module
    assert CorrelationTable.build(module).findings(module) == []

    root = exports[0]
    artifact = emit_ssa_function_to_llvm(module, root)
    assert artifact.complete, artifact.shortfalls
    native = compile_artifact(artifact, directory=tmp_path / "llvm")
    execution = prepare_artifact_execution(native, {}).run()

    assert [execution.buffers[value.id].item() for value in outputs[root]] == [1, -1]


def test_exact_int_and_bool_loop_predicates_execute_on_c(int_bool_module, tmp_path):
    module, outputs, exports = int_bool_module
    root = exports[0]
    artifact = emit_ssa_to_c(module, root)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "c")
    execution = artifact.prepare_execution({}).run()

    assert [execution.buffers[value.id].item() for value in outputs[root]] == [1, -1]


def test_exact_int_and_bool_loop_predicates_execute_on_fortran(int_bool_module, tmp_path):
    if fortran_compiler() is None:
        pytest.skip("native Fortran compiler unavailable")
    module, outputs, exports = int_bool_module
    root = exports[0]
    artifact = emit_module(
        module,
        name="loop_predicate_intbool_abi",
        outputs={root: outputs[root]},
        extra_roots=exports,
        progress=lambda _message: None,
    )
    assert artifact.complete, artifact.shortfalls
    native = compile_fortran_module_c_shell(
        artifact, {}, tmp_path / "fortran", entrypoint=root, library=True,
    )
    core = FortranCoreNative(
        native.executable_path, artifact.api.entry_point(root), artifact.api,
    )
    execution = core.prepare_execution({}).run()

    assert [execution.buffers[value.id].item() for value in outputs[root]] == [1, -1]


def test_exact_float_loop_predicate_uses_primitive_fortran_abi(tmp_path):
    if fortran_compiler() is None:
        pytest.skip("native Fortran compiler unavailable")
    module, outputs, exports = _lower(FLOAT_SOURCE, "loop_predicate_float_abi")
    assert CorrelationTable.build(module).findings(module) == []
    root = exports[0]
    artifact = emit_module(
        module,
        name="loop_predicate_float_abi",
        outputs={root: outputs[root]},
        extra_roots=exports,
        progress=lambda _message: None,
    )
    assert artifact.complete, artifact.shortfalls
    native = compile_fortran_module_c_shell(
        artifact, {}, tmp_path / "fortran", entrypoint=root, library=True,
    )
    core = FortranCoreNative(
        native.executable_path, artifact.api.entry_point(root), artifact.api,
    )
    execution = core.prepare_execution({}).run()

    assert execution.buffers[outputs[root][0].id].item() == 1.5


def test_mixed_int_bool_loop_predicate_keeps_explicit_native_refusal():
    source = '''
def validate_int(offsets: list[int]) -> int:
    for offset in offsets:
        if not isinstance(offset, int) or isinstance(offset, bool):
            return -1
    return 0

def root() -> int:
    return validate_int([0, True])
'''
    with pytest.raises(ValueError, match="unsupported mixed-element native type ABI"):
        _lower(source, "loop_predicate_mixed_abi")
