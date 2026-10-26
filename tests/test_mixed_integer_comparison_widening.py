"""Canonical source comparisons widen only the exactly representable u32 side."""

from pathlib import Path

import numpy as np
import pytest

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import (
    compile_fortran_module_c_shell,
    lower_ast_source_to_ssa,
)


REPOSITORY = Path(__file__).resolve().parents[1]
CONTRACT = REPOSITORY / "extraction_contracts" / "program_extraction.yaml"
SOURCE = "def compare(values, scalar):\n    return values[0] < scalar\n"


def _lower(dtype: str, *, name: str, source: str = SOURCE):
    policy = ExtractionContract(CONTRACT).with_program_abi({
        "records": {},
        "bindings": [],
        "values": [
            {
                "function": "compare", "parameter": "values",
                "storage": "span", "dtype": dtype, "rank": 1,
                "shape": [1], "python_type": "builtins.list",
            },
            {
                "function": "compare", "parameter": "scalar",
                "storage": "scalar", "dtype": "int64", "rank": 0,
                "python_type": "builtins.int",
            },
        ],
    })
    return lower_ast_source_to_ssa(
        source, "compare", name=name, extraction_contract=policy,
        progress=lambda _message: None,
    )


LOOP_SOURCE = (
    "def count_loop(values):\n"
    "    i = 0\n"
    "    while i < values[0]:\n"
    "        i += 1\n"
    "    return i\n"
)


def _lower_loop_source(name: str):
    policy = ExtractionContract(CONTRACT).with_program_abi({
        "records": {},
        "bindings": [],
        "values": [{
            "function": "count_loop", "parameter": "values",
            "storage": "span", "dtype": "uint32", "rank": 1,
            "shape": [1], "python_type": "builtins.list",
        }],
    })
    return lower_ast_source_to_ssa(
        LOOP_SOURCE, "count_loop", name=name,
        extraction_contract=policy, progress=lambda _message: None,
    )


def _lower_integer_add(left_dtype: str, right_dtype: str, *, name: str, reverse=False):
    left, right = ("right", "left") if reverse else ("left", "right")
    source = f"def add(left, right):\n    return {left} + {right}\n"
    policy = ExtractionContract(CONTRACT).with_program_abi({
        "records": {},
        "bindings": [],
        "values": [
            {
                "function": "add", "parameter": "left",
                "storage": "scalar", "dtype": left_dtype, "rank": 0,
                "python_type": "builtins.int",
            },
            {
                "function": "add", "parameter": "right",
                "storage": "scalar", "dtype": right_dtype, "rank": 0,
                "python_type": "builtins.int",
            },
        ],
    })
    return lower_ast_source_to_ssa(
        source, "add", name=name, extraction_contract=policy,
        progress=lambda _message: None,
    )


@pytest.mark.parametrize("backend", ["c", "llvm", "fortran"])
def test_while_carried_python_int_seed_uses_existing_int64_comparison_domain(
    backend, tmp_path,
):
    module, outputs, exports = _lower_loop_source(f"u32_loop_seed_{backend}")
    from src.compiler.identity_concordance import concordance_report

    assert concordance_report(module).splitlines()[0].endswith("0 finding(s)")
    entry = exports[0]
    comparisons = [
        instruction
        for function_name, function in module.functions.items()
        if "count_loop" in function_name
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Lt"
    ]
    assert len(comparisons) == 1
    comparison = comparisons[0]
    assert tuple(argument.dtype for argument in comparison.args) == (
        "int64", "int64",
    )
    casts = {
        instruction.res.id: instruction
        for function_name, function in module.functions.items()
        if "count_loop" in function_name
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Cast"
    }
    cast = next(
        casts[argument.id]
        for argument in comparison.args
        if argument.id in casts
    )
    assert cast.args[0].dtype == "uint32"
    assert cast.res.dtype == "int64"

    root = module.functions[entry]
    parameter = dict(root.metadata["parameter_names"])["values"]
    result = outputs[entry][0]
    cases = (0, 1, 5)
    if backend == "c":
        from src.compiler.ssa_c_backend import emit_ssa_to_c

        artifact = emit_ssa_to_c(module, entry)
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path / "c")
        for value in cases:
            execution = artifact.prepare_execution({
                parameter: np.array([value], dtype=np.uint32),
            }).run()
            assert execution.buffers[result.id].item() == value
    elif backend == "llvm":
        from src.compiler.ssa_llvm_backend import (
            compile_artifact,
            emit_ssa_function_to_llvm,
            prepare_artifact_execution,
        )

        emitted = emit_ssa_function_to_llvm(module, entry)
        assert emitted.complete, emitted.shortfalls
        artifact = compile_artifact(emitted, directory=tmp_path / "llvm")
        for value in cases:
            execution = prepare_artifact_execution(artifact, {
                parameter: np.array([value], dtype=np.uint32),
            }).run()
            assert execution.buffers[result.id].item() == value
    else:
        from src.compiler.fortran_toolchain import (
            aggressive_fortran_flags,
            native_fortran_flags,
        )
        from src.compiler.ssa_fortran_backend import (
            FortranCoreNative,
            emit_module,
            fortran_compiler,
        )

        compiler = fortran_compiler()
        if compiler is None:
            pytest.skip("native Fortran compiler unavailable")
        try:
            native_fortran_flags(
                compiler, "unsigned(c_int32_t) :: capability_probe",
                aggressive_fortran_flags(compiler),
            )
        except ValueError as error:
            if str(error).startswith((
                "native unsigned Fortran requires a compiler supporting -funsigned;",
                "selected Fortran compiler/flags do not support exact native unsigned ISO C storage with -funsigned:",
            )):
                pytest.skip("selected Fortran toolchain lacks native unsigned ISO-C storage: " + str(error))
            raise
        emitted = emit_module(
            module, name=f"u32_loop_seed_{backend}", outputs=outputs,
            extra_roots=exports, progress=lambda _message: None,
        )
        assert emitted.complete, emitted.shortfalls
        built = compile_fortran_module_c_shell(
            emitted, {}, tmp_path / "fortran", entrypoint=entry,
            library=True,
        )
        core = FortranCoreNative(
            built.executable_path, emitted.api.entry_point(entry), emitted.api,
        )
        for value in cases:
            execution = core.prepare_execution({
                parameter: np.array([value], dtype=np.uint32),
            }).run()
            assert execution.buffers[result.id].item() == value


@pytest.mark.parametrize("backend", ["c", "llvm", "fortran"])
def test_canonical_uint32_int64_comparison_is_exact_at_native_boundaries(
    backend, tmp_path,
):
    module, outputs, exports = _lower("uint32", name=f"u32_i64_{backend}")
    from src.compiler.identity_concordance import concordance_report

    assert concordance_report(module).splitlines()[0].endswith("0 finding(s)")
    entry = exports[0]
    region = module.functions[f"{entry}__planned_region_1"]
    instructions = [
        instruction
        for block in region.blocks.values()
        for instruction in block.instrs
    ]
    cast = next(instruction for instruction in instructions if instruction.op == "Cast")
    comparison = next(instruction for instruction in instructions if instruction.op == "Lt")
    assert cast.args[0].dtype == "uint32"
    assert cast.res.dtype == "int64"
    assert cast.attributes["target_dtype"] == "int64"
    assert tuple(argument.dtype for argument in comparison.args) == (
        "int64", "int64",
    )

    root = module.functions[entry]
    parameters = dict(root.metadata["parameter_names"])
    results = outputs[entry]
    assert len(results) == 1
    feed_values = (
        (0, -(2**63), False),
        (0, -1, False),
        (1, 0, False),
        (2**32 - 1, 2**32 - 1, False),
        (2**32 - 1, 2**32, True),
        (2**32 - 1, 2**63 - 1, True),
    )

    if backend == "c":
        from src.compiler.ssa_c_backend import emit_ssa_to_c

        artifact = emit_ssa_to_c(module, entry)
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path / "c")
        for value, scalar, expected in feed_values:
            execution = artifact.prepare_execution({
                parameters["values"]: np.array([value], dtype=np.uint32),
                parameters["scalar"]: np.array([scalar], dtype=np.int64),
            }).run()
            assert bool(execution.buffers[results[0].id].item()) is expected
    elif backend == "llvm":
        from src.compiler.ssa_llvm_backend import (
            compile_artifact,
            emit_ssa_function_to_llvm,
            prepare_artifact_execution,
        )

        emitted = emit_ssa_function_to_llvm(module, entry)
        assert emitted.complete, emitted.shortfalls
        artifact = compile_artifact(emitted, directory=tmp_path / "llvm")
        for value, scalar, expected in feed_values:
            execution = prepare_artifact_execution(artifact, {
                parameters["values"]: np.array([value], dtype=np.uint32),
                parameters["scalar"]: np.array([scalar], dtype=np.int64),
            }).run()
            assert bool(execution.buffers[results[0].id].item()) is expected
    else:
        from src.compiler.fortran_toolchain import (
            aggressive_fortran_flags,
            native_fortran_flags,
        )
        from src.compiler.ssa_fortran_backend import (
            FortranCoreNative,
            emit_module,
            fortran_compiler,
        )

        compiler = fortran_compiler()
        if compiler is None:
            pytest.skip("native Fortran compiler unavailable")
        try:
            native_fortran_flags(
                compiler, "unsigned(c_int32_t) :: capability_probe",
                aggressive_fortran_flags(compiler),
            )
        except ValueError as error:
            if str(error).startswith((
                "native unsigned Fortran requires a compiler supporting -funsigned;",
                "selected Fortran compiler/flags do not support exact native unsigned ISO C storage with -funsigned:",
            )):
                pytest.skip("selected Fortran toolchain lacks native unsigned ISO-C storage: " + str(error))
            raise
        emitted = emit_module(
            module, name=f"u32_i64_{backend}", outputs=outputs,
            extra_roots=exports, progress=lambda _message: None,
        )
        assert emitted.complete, emitted.shortfalls
        built = compile_fortran_module_c_shell(
            emitted, {}, tmp_path / "fortran", entrypoint=entry,
            library=True,
        )
        core = FortranCoreNative(
            built.executable_path, emitted.api.entry_point(entry), emitted.api,
        )
        for value, scalar, expected in feed_values:
            execution = core.prepare_execution({
                parameters["values"]: np.array([value], dtype=np.uint32),
                parameters["scalar"]: np.array([scalar], dtype=np.int64),
            }).run()
            assert bool(execution.buffers[results[0].id].item()) is expected


@pytest.mark.parametrize("operation", ["<", "<=", ">", ">=", "==", "!="])
@pytest.mark.parametrize("reverse", [False, True])
def test_all_integer_comparisons_cast_only_the_uint32_operand(operation, reverse):
    left, right = ("scalar", "values[0]") if reverse else ("values[0]", "scalar")
    source = f"def compare(values, scalar):\n    return {left} {operation} {right}\n"
    module, _outputs, exports = _lower(
        "uint32", name=f"compare_{operation}_{reverse}", source=source,
    )
    region = module.functions[f"{exports[0]}__planned_region_1"]
    instructions = [
        instruction
        for block in region.blocks.values()
        for instruction in block.instrs
    ]
    casts = [instruction for instruction in instructions if instruction.op == "Cast"]
    compared = next(
        instruction for instruction in instructions
        if instruction.op in {"Lt", "Le", "Gt", "Ge", "Eq", "Ne"}
    )
    assert len(casts) == 1
    assert casts[0].args[0].dtype == "uint32"
    assert casts[0].res.dtype == "int64"
    widened_position = 1 if reverse else 0
    assert compared.args[widened_position].id == casts[0].res.id
    assert compared.args[1 - widened_position].dtype == "int64"
    assert tuple(argument.dtype for argument in compared.args) == (
        "int64", "int64",
    )
    from src.compiler.identity_concordance import concordance_report
    from src.compiler.ssa_c_backend import emit_ssa_to_c
    from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm
    from src.compiler.ssa_fortran_backend import emit_module

    assert concordance_report(module).splitlines()[0].endswith("0 finding(s)")
    assert emit_ssa_to_c(module, exports[0]).complete
    assert emit_ssa_function_to_llvm(module, exports[0]).complete
    assert emit_module(
        module, name=f"compare_{operation}_{reverse}", outputs=_outputs,
        extra_roots=exports, progress=lambda _message: None,
    ).complete


def test_canonical_uint64_int64_comparison_remains_refused():
    module, outputs, exports = _lower("uint64", name="u64_i64_refusal")
    entry = exports[0]
    region = module.functions[f"{entry}__planned_region_1"]
    instructions = [
        instruction
        for block in region.blocks.values()
        for instruction in block.instrs
    ]
    assert not any(instruction.op == "Cast" for instruction in instructions)
    assert next(
        instruction for instruction in instructions if instruction.op == "Lt"
    ).args[0].dtype == "uint64"

    from src.compiler.ssa_c_backend import emit_ssa_to_c
    from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm
    from src.compiler.ssa_fortran_backend import emit_module

    assert not emit_ssa_to_c(module, entry).complete
    assert not emit_ssa_function_to_llvm(module, entry).complete
    assert not emit_module(
        module, name="u64_i64_refusal", outputs=outputs,
        extra_roots=exports, progress=lambda _message: None,
    ).complete


@pytest.mark.parametrize("backend", ["c", "llvm", "fortran"])
@pytest.mark.parametrize("reverse", [False, True])
def test_int64_uint32_add_uses_exact_int64_inputs_and_existing_wrap_law(
    backend, reverse, tmp_path,
):
    module, outputs, exports = _lower_integer_add(
        "int64", "uint32", name=f"i64_u32_add_{backend}_{reverse}",
        reverse=reverse,
    )
    from src.compiler.identity_concordance import concordance_report

    assert concordance_report(module).splitlines()[0].endswith("0 finding(s)")
    entry = exports[0]
    region = module.functions[f"{entry}__planned_region_0"]
    instructions = [
        instruction
        for block in region.blocks.values()
        for instruction in block.instrs
    ]
    casts = [instruction for instruction in instructions if instruction.op == "Cast"]
    add = next(instruction for instruction in instructions if instruction.op == "Add")
    assert len(casts) == 1
    assert casts[0].args[0].dtype == "uint32"
    assert casts[0].res.dtype == "int64"
    assert tuple(argument.dtype for argument in add.args) == ("int64", "int64")
    assert add.res.dtype == "int64"

    root = module.functions[entry]
    parameters = dict(root.metadata["parameter_names"])
    result = outputs[entry][0]
    cases = (
        (2**63 - 1, 1, -(2**63)),
        (-(2**63), 2**32 - 1, -(2**63) + 2**32 - 1),
        (-(2**63), 1, -(2**63) + 1),
    )
    def feed(left_value, right_value):
        values = {"left": left_value, "right": right_value}
        return {
            parameters[name]: np.array(
                [values[name]],
                dtype=(np.uint32 if name == "right" else np.int64),
            )
            for name in ("left", "right")
        }

    if backend == "c":
        from src.compiler.ssa_c_backend import emit_ssa_to_c

        artifact = emit_ssa_to_c(module, entry)
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path / "c")
        for left_value, right_value, expected in cases:
            execution = artifact.prepare_execution(feed(left_value, right_value)).run()
            assert execution.buffers[result.id].item() == expected
    elif backend == "llvm":
        from src.compiler.ssa_llvm_backend import (
            compile_artifact,
            emit_ssa_function_to_llvm,
            prepare_artifact_execution,
        )

        emitted = emit_ssa_function_to_llvm(module, entry)
        assert emitted.complete, emitted.shortfalls
        artifact = compile_artifact(emitted, directory=tmp_path / "llvm")
        for left_value, right_value, expected in cases:
            execution = prepare_artifact_execution(
                artifact, feed(left_value, right_value),
            ).run()
            assert execution.buffers[result.id].item() == expected
    else:
        from src.compiler.fortran_toolchain import (
            aggressive_fortran_flags,
            native_fortran_flags,
        )
        from src.compiler.ssa_fortran_backend import (
            FortranCoreNative,
            emit_module,
            fortran_compiler,
        )

        compiler = fortran_compiler()
        if compiler is None:
            pytest.skip("native Fortran compiler unavailable")
        try:
            native_fortran_flags(
                compiler, "unsigned(c_int32_t) :: capability_probe",
                aggressive_fortran_flags(compiler),
            )
        except ValueError as error:
            if str(error).startswith((
                "native unsigned Fortran requires a compiler supporting -funsigned;",
                "selected Fortran compiler/flags do not support exact native unsigned ISO C storage with -funsigned:",
            )):
                pytest.skip("selected Fortran toolchain lacks native unsigned ISO-C storage: " + str(error))
            raise
        emitted = emit_module(
            module, name=f"i64_u32_add_{backend}_{reverse}", outputs=outputs,
            extra_roots=exports, progress=lambda _message: None,
        )
        assert emitted.complete, emitted.shortfalls
        built = compile_fortran_module_c_shell(
            emitted, {}, tmp_path / "fortran", entrypoint=entry,
            library=True,
        )
        core = FortranCoreNative(
            built.executable_path, emitted.api.entry_point(entry), emitted.api,
        )
        for left_value, right_value, expected in cases:
            execution = core.prepare_execution(feed(left_value, right_value)).run()
            assert execution.buffers[result.id].item() == expected


def test_canonical_uint64_int64_add_remains_refused():
    module, outputs, exports = _lower_integer_add(
        "uint64", "int64", name="u64_i64_add_refusal",
    )
    entry = exports[0]
    region = module.functions[f"{entry}__planned_region_0"]
    instructions = [
        instruction
        for block in region.blocks.values()
        for instruction in block.instrs
    ]
    assert not any(instruction.op == "Cast" for instruction in instructions)
    assert next(
        instruction for instruction in instructions if instruction.op == "Add"
    ).args[0].dtype == "uint64"

    from src.compiler.ssa_c_backend import emit_ssa_to_c
    from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm
    from src.compiler.ssa_fortran_backend import emit_module

    assert not emit_ssa_to_c(module, entry).complete
    assert not emit_ssa_function_to_llvm(module, entry).complete
    assert not emit_module(
        module, name="u64_i64_add_refusal", outputs=outputs,
        extra_roots=exports, progress=lambda _message: None,
    ).complete
