"""A forwarded sequence keeps its exact element class at a loop target."""

from pathlib import Path

import pytest

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.concordance_declarations import SOURCE_VALUE_CLASS
from src.compiler.identity_concordance import CorrelationTable


SOURCE = '''def inner(values: list[int]) -> int:
    for value in values:
        if isinstance(value, bool):
            return 4
        if not isinstance(value, int):
            return 5
    return 0

def outer(values: list[int]) -> int:
    return inner(values)

def root():
    return outer({sequence})
'''


def _lower(element_class):
    root = Path(__file__).resolve().parents[1]
    sequence = "[False, True]" if element_class == "bool" else "[0, 1]"
    return lower_ast_source_to_ssa(
        SOURCE.format(sequence=sequence),
        "root",
        name="forwarded_sequence_element_class",
        extraction_contract=root / "extraction_contracts/program_extraction.yaml",
        progress=lambda _: None,
        progress_bars=False,
    )


def _assert_exact_inner_class(module, expected_class, expected_dtype):
    book = module.metadata["identity_book"]
    page = book.page(SOURCE_VALUE_CLASS)
    inner_scopes = {
        function.metadata.get("source_numeric_scope")
        for function in module.functions.values()
        if "inner" in str(function.metadata.get("source_qualified_name") or "")
        or "inner" in function.name
    }
    targets = {}
    for row in page.rows():
        if not isinstance(row, tuple) or len(row) != 2 or row[0] not in inner_scopes:
            continue
        fact = page.latest(row)
        if isinstance(fact, tuple) and len(fact) >= 2:
            targets.setdefault(str(fact[0]), []).append((row, fact))

    matches = targets.get(expected_class, ())
    assert matches, f"inner loop target has no exact {expected_class} class row"
    row, fact = matches[0]
    target_ref = book.latest_ref(SOURCE_VALUE_CLASS, row)
    assert target_ref is not None
    owners = [
        function for function in module.functions.values()
        if function.metadata.get("source_numeric_scope") == row[0]
    ]
    assert owners, f"no inner SSA owner for exact loop class row {row!r}"
    matching_abi = []
    for function in owners:
        parameter_names = dict(function.metadata.get("parameter_names") or ())
        values_id = parameter_names.get("values")
        fact_abi = (function.metadata.get("native_value_type_abi") or {}).get(values_id)
        alternatives = tuple(getattr(fact_abi, "physical_alternatives", ()) or ())
        if any(
            getattr(alternative, "storage", None) == "span"
            and getattr(alternative, "dtype", None) == expected_dtype
            for alternative in alternatives
        ):
            matching_abi.append(function)
    assert matching_abi, (
        f"inner {expected_class} loop class is not backed by a {expected_dtype} sequence ABI"
    )


def _native_execution(backend, module, outputs, exports, tmp_path):
    entry = exports[0]
    if backend == "c":
        from src.compiler.ssa_c_backend import emit_ssa_to_c

        artifact = emit_ssa_to_c(module, entry)
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path)
        return artifact.prepare_execution
    if backend == "llvm":
        from src.compiler.ssa_llvm_backend import (
            compile_artifact,
            emit_ssa_function_to_llvm,
            prepare_artifact_execution,
        )

        artifact = emit_ssa_function_to_llvm(module, entry)
        assert artifact.complete, artifact.shortfalls
        native = compile_artifact(artifact, directory=tmp_path)
        return lambda feeds: prepare_artifact_execution(native, feeds)

    from src.compiler.fortran_c_shell import compile_fortran_module_c_shell
    from src.compiler.ssa_fortran_backend import (
        FortranCoreNative,
        emit_module,
        fortran_compiler,
    )

    if fortran_compiler() is None:
        pytest.skip("Fortran compiler unavailable")
    artifact = emit_module(
        module,
        outputs={entry: outputs[entry]},
        extra_roots=exports,
        progress=lambda _: None,
    )
    assert artifact.complete, artifact.shortfalls
    native = compile_fortran_module_c_shell(
        artifact, {}, tmp_path, entrypoint=entry, library=True,
    )
    api = artifact.api.entry_point(entry)
    return FortranCoreNative(
        native.executable_path, api, artifact.api,
    ).prepare_execution


@pytest.mark.parametrize("backend", ["c", "llvm", "fortran"])
@pytest.mark.parametrize(
    "element_class,dtype,expected",
    [
        ("bool", "bool", 4),
        ("int", "int64", 0),
    ],
)
def test_public_two_hop_forwarding_preserves_exact_loop_class(
    backend, element_class, dtype, expected, tmp_path,
):
    module, outputs, exports = _lower(element_class)
    assert not CorrelationTable.build(module).findings(module)
    _assert_exact_inner_class(module, f"builtins.{element_class}", dtype)

    entry = exports[0]
    prepare = _native_execution(backend, module, outputs, exports, tmp_path)
    execution = prepare({}).run()
    assert int(execution.buffers[outputs[entry][0].id].item()) == expected
