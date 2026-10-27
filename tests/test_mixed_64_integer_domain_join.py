"""Exact common domains for signed/unsigned scalar operands."""

from pathlib import Path

import numpy as np
import pytest

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import (
    compile_fortran_module_c_shell,
    lower_ast_source_to_ssa,
)
from src.transmogrifier.dtype_layout import (
    C_ABI_NAMES,
    C_LANE_NAMES,
    C_LANE_STORAGE_CLASSES,
    LLVM_LANE_NAMES,
    common_integer_dtype,
    c_abi_type,
    fortran_kind_table,
    llvm_type_for_dtype,
    shell_numpy_dtype,
)


REPOSITORY = Path(__file__).resolve().parents[1]
CONTRACT = REPOSITORY / "extraction_contracts" / "program_extraction.yaml"


def test_integer_domain_join_is_exact_and_internal_128_is_not_public_abi():
    assert common_integer_dtype(("uint32", "int64")) == "int64"
    assert common_integer_dtype(("int64", "uint32")) == "int64"
    assert common_integer_dtype(("uint64", "int64")) == "int128"
    assert common_integer_dtype(("int64", "uint64")) == "int128"
    assert common_integer_dtype(("uint64", "uint64")) == "uint64"
    assert common_integer_dtype(("int64", "int64")) == "int64"
    assert common_integer_dtype(("uint128", "int128")) is None
    assert common_integer_dtype(("opaque_ref", "int64")) is None

    assert "int128" not in C_ABI_NAMES
    assert "uint128" not in C_ABI_NAMES
    assert "int128" not in C_LANE_NAMES
    assert "uint128" not in C_LANE_NAMES
    assert "int128" not in LLVM_LANE_NAMES
    assert "uint128" not in LLVM_LANE_NAMES
    assert c_abi_type("ptr") == ("void *", "c_void_p")
    assert "int128" not in C_LANE_STORAGE_CLASSES
    assert "uint128" not in C_LANE_STORAGE_CLASSES
    assert shell_numpy_dtype("int128") is None
    assert shell_numpy_dtype("uint128") is None
    with pytest.raises(KeyError, match="public C ABI"):
        c_abi_type("int128")
    assert llvm_type_for_dtype("int128") == "i128"
    assert fortran_kind_table()["int128"] == "integer(kind=16)"


def _lower(*, reverse: bool, name: str):
    expression = "right + left" if reverse else "left + right"
    source = (
        "def compare(left, right):\n"
        f"    total = {expression}\n"
        "    return total > left\n"
    )
    policy = ExtractionContract(CONTRACT).with_program_abi({
        "records": {},
        "bindings": [],
        "values": [
            {
                "function": "compare", "parameter": "left",
                "storage": "scalar", "dtype": "uint64", "rank": 0,
                "python_type": "builtins.int",
            },
            {
                "function": "compare", "parameter": "right",
                "storage": "scalar", "dtype": "int64", "rank": 0,
                "python_type": "builtins.int",
            },
        ],
    })
    return lower_ast_source_to_ssa(
        source, "compare", name=name, extraction_contract=policy,
        progress=lambda _message: None,
    )


@pytest.mark.parametrize("backend", ["c", "llvm", "fortran"])
@pytest.mark.parametrize("reverse", [False, True])
def test_uint64_int64_add_and_comparison_use_internal_i128(
    backend, reverse, tmp_path,
):
    module, outputs, exports = _lower(
        reverse=reverse, name=f"integer_domain_{backend}_{reverse}",
    )
    from src.compiler.identity_concordance import concordance_report

    assert concordance_report(module).splitlines()[0].endswith("0 finding(s)")
    entry = exports[0]
    additions = [
        instruction
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Add"
        and instruction.res is not None
        and instruction.res.dtype == "int128"
    ]
    assert additions
    assert all(tuple(arg.dtype for arg in item.args) == ("int128", "int128")
               for item in additions)
    widening = [
        instruction
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Cast"
        and instruction.res is not None
        and instruction.res.dtype == "int128"
        and instruction.args
    ]
    assert {instruction.args[0].dtype for instruction in widening} >= {
        "uint64", "int64",
    }

    left_id = dict(module.functions[entry].metadata["parameter_names"])["left"]
    right_id = dict(module.functions[entry].metadata["parameter_names"])["right"]
    result_id = outputs[entry][0].id
    cases = (
        (2**63 + 17, 1, True),
        (2**64 - 1, -1, False),
        (0, -1, False),
        (2**53 + 1, 1, True),
        (17, 0, False),
    )
    if backend == "c":
        from src.compiler.ssa_c_backend import emit_ssa_to_c

        artifact = emit_ssa_to_c(module, entry)
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path / "c")
        prepare = artifact.prepare_execution
    elif backend == "llvm":
        from src.compiler.ssa_llvm_backend import (
            compile_artifact,
            emit_ssa_function_to_llvm,
            prepare_artifact_execution,
        )

        emitted = emit_ssa_function_to_llvm(module, entry)
        assert emitted.complete, emitted.shortfalls
        artifact = compile_artifact(emitted, directory=tmp_path / "llvm")
        prepare = lambda feeds: prepare_artifact_execution(artifact, feeds)
    else:
        from src.compiler.ssa_fortran_backend import (
            FortranCoreNative, emit_module, fortran_compiler,
        )

        compiler = fortran_compiler()
        if compiler is None:
            pytest.skip("native Fortran compiler unavailable")
        emitted = emit_module(
            module, outputs={name: outputs[name] for name in exports},
            extra_roots=exports, progress=lambda _message: None,
        )
        assert emitted.complete, emitted.shortfalls
        native = compile_fortran_module_c_shell(
            emitted, {}, tmp_path / "fortran", entrypoint=entry,
            library=True,
        )
        core = FortranCoreNative(
            native.executable_path, emitted.api.entry_point(entry), emitted.api,
        )
        prepare = core.prepare_execution

    for left, right, expected in cases:
        execution = prepare({
            left_id: np.array([left], dtype=np.uint64),
            right_id: np.array([right], dtype=np.int64),
        }).run()
        assert bool(execution.buffers[result_id].item()) is expected


def test_scalar_shaped_add_result_joins_against_unsigned_comparison_operand():
    source = (
        "def compare(left, right, limit):\n"
        "    total = left + right\n"
        "    return total > limit\n"
    )
    policy = ExtractionContract(CONTRACT).with_program_abi({
        "records": {},
        "bindings": [],
        "values": [
            {
                "function": "compare", "parameter": "left",
                "storage": "scalar", "dtype": "int64", "rank": 0,
                "python_type": "builtins.int",
            },
            {
                "function": "compare", "parameter": "right",
                "storage": "scalar", "dtype": "int64", "rank": 0,
                "python_type": "builtins.int",
            },
            {
                "function": "compare", "parameter": "limit",
                "storage": "scalar", "dtype": "uint64", "rank": 0,
                "python_type": "builtins.int",
            },
        ],
    })
    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "compare", name="scalar_shaped_add_comparison",
        extraction_contract=policy, progress=lambda _message: None,
    )
    from src.compiler.identity_concordance import concordance_report

    assert concordance_report(module).splitlines()[0].endswith("0 finding(s)")
    instructions = [
        instruction
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
    ]
    add = next(
        instruction for instruction in instructions
        if instruction.op == "Add" and instruction.res is not None
    )
    comparison = next(
        instruction for instruction in instructions
        if instruction.op == "Gt" and instruction.res is not None
    )
    assert add.res.dtype == "int64"
    assert tuple(argument.dtype for argument in comparison.args) == (
        "int128", "int128",
    )
    assert any(
        instruction.op == "Cast"
        and instruction.res is not None
        and instruction.res.dtype == "int128"
        and instruction.args
        and instruction.args[0].id == add.res.id
        for instruction in instructions
    )
