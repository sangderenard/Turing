"""Builtin bool arithmetic keeps Python's integer result ABI on native targets."""

import itertools
from pathlib import Path

import numpy as np
import pytest

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import (
    compile_fortran_module_c_shell,
    lower_ast_source_to_ssa,
)
from src.compiler.ssa_c_backend import emit_ssa_to_c
from src.compiler.ssa_fortran_backend import (
    FortranCoreNative,
    emit_module as emit_fortran_module,
    fortran_compiler,
)
from src.compiler.ssa_llvm_backend import (
    compile_artifact as compile_llvm_artifact,
    emit_ssa_function_to_llvm,
    prepare_artifact_execution as prepare_llvm_execution,
)


@pytest.mark.parametrize("backend", ["c", "llvm", "fortran"])
def test_python_bool_add_returns_integer_on_native_backend(tmp_path, backend):
    source = "def f(a: bool, b: bool):\n    return a + b, a - b, a * b\n"
    contract_path = (
        Path(__file__).resolve().parents[1]
        / "extraction_contracts"
        / "program_extraction.yaml"
    )
    contract = ExtractionContract(
        str(contract_path)
    )
    module, outputs, exports = lower_ast_source_to_ssa(
        source, "f", name="python_bool_add", extraction_contract=contract,
        progress=lambda _event: None, progress_bars=False,
    )
    root = module.functions[exports[0]]
    parameters = dict(root.metadata["parameter_names"])
    output_ids = next(
        [value.id for value in instruction.args]
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret"
    )
    planned_output_types = [
        value.dtype
        for region, values in outputs.items()
        if "planned_region" in region
        for value in values if value.id in output_ids
    ]
    assert planned_output_types == ["int64"] * 3

    cases = tuple(itertools.product((False, True), repeat=2))
    expected = {
        (left, right): (
            int(left) + int(right),
            int(left) - int(right),
            int(left) * int(right),
        )
        for left, right in cases
    }

    if backend == "c":
        artifact = emit_ssa_to_c(module, exports[0])
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path / "c-native")

        def execute(feeds):
            return artifact.prepare_execution(feeds).run()

    elif backend == "llvm":
        artifact = emit_ssa_function_to_llvm(module, exports[0])
        assert artifact.complete, artifact.shortfalls
        artifact = compile_llvm_artifact(
            artifact, directory=tmp_path / "llvm-native",
        )

        def execute(feeds):
            return prepare_llvm_execution(artifact, feeds).run()

    else:
        if fortran_compiler() is None:
            pytest.skip("native Fortran compiler unavailable")
        fortran_artifact = emit_fortran_module(
            module, name="python_bool_add", progress=lambda _event: None,
        )
        assert fortran_artifact.complete, fortran_artifact.shortfalls
        built = compile_fortran_module_c_shell(
            fortran_artifact, {}, tmp_path / "fortran-native",
            entrypoint=exports[0], library=True,
        )
        fortran_core = FortranCoreNative(
            built.executable_path,
            fortran_artifact.api.entry_point(exports[0]),
            fortran_artifact.api,
        )

        def execute(feeds):
            return fortran_core.prepare_execution(feeds).run()

    for left, right in cases:
        feeds = {
            int(parameters["a"]): np.asarray([left], dtype=np.bool_),
            int(parameters["b"]): np.asarray([right], dtype=np.bool_),
        }
        want = expected[(left, right)]
        execution = execute(feeds)
        actual = tuple(
            execution.buffers[value_id].item() for value_id in output_ids
        )
        assert all(
            execution.buffers[value_id].dtype == np.dtype("int64")
            for value_id in output_ids
        )
        assert actual == want, (backend, left, right, actual, want)


def test_bool_tensor_add_keeps_the_boolean_tensor_result_domain():
    contract_path = (
        Path(__file__).resolve().parents[1]
        / "extraction_contracts"
        / "program_extraction.yaml"
    )
    contract = ExtractionContract(
        str(contract_path)
    ).with_program_abi({
        "records": {},
        "bindings": [],
        "values": [
            {
                "function": "f", "parameter": name, "storage": "span",
                "dtype": "bool", "rank": 1, "shape": [2],
                "mutable": False, "python_type": "numpy.ndarray",
            }
            for name in ("a", "b")
        ],
    })
    _module, outputs, _exports = lower_ast_source_to_ssa(
        "def f(a, b):\n    return a + b\n",
        "f", name="bool_tensor_add_control", extraction_contract=contract,
        progress=lambda _event: None, progress_bars=False,
    )
    planned = outputs["bool_tensor_add_control__f__planned_region_0"]
    result = next(value for value in planned if value.id == 2)
    assert result.dtype == "bool"
    assert result.shape == (2,)


def test_single_leaf_llvm_bool_arithmetic_uses_integer_result(tmp_path):
    from src.transmogrifier.ssa import BasicBlock, Function, IRModule, Instr, SSAValue

    left, right = SSAValue(0, 'bool'), SSAValue(1, 'bool')
    results = [SSAValue(index + 2, 'int64') for index in range(3)]
    function = Function('bool_leaf', [left, right], {
        'entry': BasicBlock('entry', [
            *[Instr(op, [left, right], result)
              for op, result in zip(('Add', 'Sub', 'Mul'), results)],
            Instr('Ret', results, None),
        ]),
    })
    artifact = emit_ssa_function_to_llvm(IRModule({function.name: function}), function.name)
    assert artifact.complete, artifact.shortfalls
    native = compile_llvm_artifact(artifact, directory=tmp_path / 'leaf')
    for a, b in itertools.product((False, True), repeat=2):
        execution = prepare_llvm_execution(native, {
            left.id: np.asarray([a]), right.id: np.asarray([b]),
        }).run()
        assert tuple(execution.buffers[result.id].item() for result in results) == (
            int(a) + int(b), int(a) - int(b), int(a) * int(b),
        )
        assert all(execution.buffers[result.id].dtype == np.dtype('int64')
                   for result in results)
