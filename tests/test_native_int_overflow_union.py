"""An unproven ``int`` result past int64 continues exactly (stage 2).

``def root(n): return n * n`` with ``n`` an int64 no interval bounds: the
returned product is the native union ``int64 | Precision``
(``python_int_overflow``).  The region computes it at int128, the frame
carries the int64 view, the Precision limbs and the tag, and the host reads
Python's exact value back through ``output_publication.return_value_plan``.
Both native lanes must return what Python returns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"

SQUARE = "def root(n):\n    return n * n\n"
NEGATE = "def root(n):\n    return -n\n"

SQUARES = (
    0, 3, -7, 3037000499, 3037000500, -3037000500,
    9223372036854775807, -9223372036854775808,
)


def _lowered(source):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.identity_concordance import CorrelationTable
    from src.compiler.output_publication import return_value_plan

    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "root", name="int_overflow_union",
        extraction_contract=str(CONTRACT), progress=lambda _message: None,
        root_calls=[{"n": 3}], root_signature={"n": "int"},
    )
    assert CorrelationTable.build(module).findings(module) == []
    root = module.functions[exports[0]]
    return module, exports[0], dict(root.metadata["parameter_names"]), (
        return_value_plan(root)
    )


def _run_c(tmp_path, source):
    from src.compiler.ssa_c_backend import emit_ssa_to_c

    module, export, parameters, plan = _lowered(source)
    artifact = emit_ssa_to_c(module, export)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "native")
    return parameters, plan, lambda feeds: artifact.prepare_execution(feeds).run()


def _run_llvm(tmp_path, source):
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )

    module, export, parameters, plan = _lowered(source)
    emitted = emit_ssa_function_to_llvm(module, export)
    assert not emitted.shortfalls, emitted.shortfalls
    artifact = compile_artifact(emitted, directory=tmp_path)
    return parameters, plan, lambda feeds: prepare_artifact_execution(
        artifact, feeds,
    ).run()


def _results(build, tmp_path, source, values):
    from src.compiler.output_publication import python_return_values

    parameters, plan, run = build(tmp_path, source)
    results = []
    for value in values:
        execution = run({parameters["n"]: np.array([value], dtype=np.int64)})
        (result,) = python_return_values(plan, execution.buffers)
        results.append(result)
    return results


@pytest.mark.parametrize("build", [_run_c, _run_llvm], ids=["c", "llvm"])
def test_square_past_int64_is_pythons_exact_int(tmp_path, build):
    results = _results(build, tmp_path, SQUARE, SQUARES)
    assert results == [n * n for n in SQUARES]
    assert all(type(result) is int for result in results)


@pytest.mark.parametrize("build", [_run_c, _run_llvm], ids=["c", "llvm"])
def test_negation_of_int64_min_is_pythons_exact_int(tmp_path, build):
    values = (5, -9223372036854775808)
    assert _results(build, tmp_path, NEGATE, values) == [-n for n in values]


def test_returned_unproven_product_is_the_int_precision_union():
    from src.compiler.python_int_overflow import union_alternatives

    _module, _export, _parameters, plan = _lowered(SQUARE)
    (row,) = plan
    assert row[0] == "union"
    assert sorted(kind for kind, _cells in row[2].values()) == [
        "int", "int_carrier",
    ]
    assert union_alternatives() == ("Precision", "builtins.int")


def test_python_exact_contract_is_what_continues_the_overflow():
    from src.compiler.python_int_overflow import python_int_continues_exactly
    from src.compiler.ssa_call_input_adapters import integer_arithmetic_policy

    assert integer_arithmetic_policy() == "python_exact"
    assert python_int_continues_exactly()


def test_carrier_limbs_read_back_exactly():
    from src.compiler.python_int_overflow import python_int_from_carrier

    value = (1 << 126) - (1 << 64) + 1
    assert python_int_from_carrier([2.0 ** 126, -(2.0 ** 64), 1.0]) == value
    with pytest.raises(ValueError):
        python_int_from_carrier([0.5])
