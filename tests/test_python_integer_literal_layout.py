from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.transmogrifier.dtype_layout import python_int_literal_dtype


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "extraction_contracts" / "program_extraction.yaml"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-(1 << 63), "int64"),
        ((1 << 63) - 1, "int64"),
        (1 << 63, "uint64"),
        ((1 << 64) - 1, "uint64"),
        (1 << 64, "int128"),
        (-((1 << 63) + 1), "int128"),
        (-(1 << 127), "int128"),
        ((1 << 127) - 1, "int128"),
    ],
)
def test_python_integer_literal_layout_uses_exact_declared_range(value, expected):
    assert python_int_literal_dtype(value) == expected


@pytest.mark.parametrize("value", [1 << 127, -(1 << 127) - 1])
def test_python_integer_literal_outside_declared_layouts_refuses(value):
    with pytest.raises(OverflowError, match="outside declared native layouts"):
        python_int_literal_dtype(value)


@pytest.mark.parametrize("backend", ["c", "llvm", "fortran"])
@pytest.mark.parametrize(
    ("literal", "expected_dtype", "feed", "expected"),
    [
        ("9223372036854775809", "uint64", -(1 << 63), True),
        ("18446744073709551615", "uint64", -(1 << 63), True),
        ("18446744073709551616", "int128", (1 << 63) - 1, True),
        ("-9223372036854775809", "int128", (1 << 63) - 1, False),
    ],
)
def test_large_python_integer_literal_compares_exactly_natively(
    backend, literal, expected_dtype, feed, expected, tmp_path,
):
    source = (
        "def compare(value):\n"
        "    return BOUND > value\n"
    )
    contract = ExtractionContract(CONTRACT).with_program_abi({
        "records": {},
        "bindings": [],
        "values": [{
            "function": "compare", "parameter": "value",
            "storage": "scalar", "dtype": "int64", "rank": 0,
            "python_type": "builtins.int",
        }],
    })
    module, outputs, exports = lower_ast_source_to_ssa(
        source, "compare", name=f"literal_{backend}_{literal}",
        python_bindings={"BOUND": int(literal)},
        extraction_contract=contract, progress=lambda _message: None,
    )

    from src.compiler.identity_concordance import concordance_report

    assert concordance_report(module).splitlines()[0].endswith("0 finding(s)")
    entry = exports[0]
    constants = [
        instruction.res
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Const" and instruction.res is not None
        and instruction.attributes.get("value") == int(literal)
    ]
    assert constants
    assert all(value.dtype == expected_dtype for value in constants), [
        (value.dtype, value.id) for value in constants
    ]
    result_id = outputs[entry][0].id
    parameter_id = dict(module.functions[entry].metadata["parameter_names"])[
        "value"
    ]

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
        from src.compiler.fortran_c_shell import compile_fortran_module_c_shell

        if fortran_compiler() is None:
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

    result = prepare({
        parameter_id: np.array([feed], dtype=np.int64),
    }).run().buffers[result_id].item()
    assert bool(result) is expected
