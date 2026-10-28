"""Python ``int`` arithmetic never wraps natively.

``a * b`` on ints no interval bounds is int64 arithmetic.  It wrapped at
int64 and returned a value Python never computes.  Until an overflowing
result continues as Precision limbs, both native lanes report it
(``native_faults.INT_OVERFLOW``) instead of returning the wrapped value.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"

SOURCE = '''
def mul(a: int, b: int) -> int:
    return a * b


def root(n: int):
    return mul(n, n) % 1000
'''


def _lowered():
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    module, _outputs, exports = lower_ast_source_to_ssa(
        SOURCE, "root", name="int_overflow_checked",
        extraction_contract=str(CONTRACT), progress=lambda _message: None,
    )
    root = module.functions[exports[0]]
    outputs = [
        int(value.id)
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret" and not instruction.attributes.get("fault_exit")
        for value in instruction.args
    ]
    return module, exports[0], dict(root.metadata["parameter_names"]), outputs


def _run_c(tmp_path, n):
    from src.compiler.ssa_c_backend import emit_ssa_to_c

    module, export, parameters, outputs = _lowered()
    artifact = emit_ssa_to_c(module, export)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "native")
    execution = artifact.prepare_execution(
        {parameters["n"]: np.array([n], dtype=np.int64)}
    ).run()
    return int(execution.buffers[outputs[0]].reshape(-1)[0])


def _run_llvm(tmp_path, n):
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )

    module, export, parameters, outputs = _lowered()
    emitted = emit_ssa_function_to_llvm(module, export)
    assert not emitted.shortfalls, emitted.shortfalls
    artifact = compile_artifact(emitted, directory=tmp_path)
    execution = prepare_artifact_execution(
        artifact, {parameters["n"]: np.array([n], dtype=np.int64)}
    ).run()
    return int(execution.buffers[outputs[0]].reshape(-1)[0])


@pytest.mark.parametrize("run", [_run_c, _run_llvm], ids=["c", "llvm"])
def test_in_range_product_matches_python(tmp_path, run):
    assert run(tmp_path, 3037000499) == (3037000499 * 3037000499) % 1000


@pytest.mark.parametrize("run", [_run_c, _run_llvm], ids=["c", "llvm"])
def test_product_past_int64_is_reported_not_wrapped(tmp_path, run):
    from src.compiler.native_faults import NativeIntOverflow

    with pytest.raises(NativeIntOverflow):
        run(tmp_path, 3037000500)


# --- the rest of Python's int arithmetic on an unproven int64 ------------
#
# Each program's ``root`` returns its int; the native result either equals
# Python's or raises the exception Python raises (NativeIntOverflow where
# Python would have continued past int64).

INT64_MIN = -(2 ** 63)

BINARY = {
    "shl": "a << b", "shr": "a >> b", "floordiv": "a // b", "mod": "a % b",
}


def _binary_source(expression):
    return f"""
def step(a: int, b: int) -> int:
    return {expression}


def root(a: int, b: int):
    return step(a, b) + 0
"""


UNARY_SOURCE = {
    "abs": """
def step(a: int) -> int:
    return abs(a)


def root(a: int, b: int):
    return step(a) + b
""",
    "neg": """
def step(a: int) -> int:
    return -a


def root(a: int, b: int):
    return step(a) + b
""",
    "square": """
def step(a: int) -> int:
    return a ** 2


def root(a: int, b: int):
    return step(a) + b
""",
    "loop_mul": """
def step(a: int, b: int) -> int:
    x = 1
    for _ in range(b):
        x *= a
    return x


def root(a: int, b: int):
    return step(a, b) + 0
""",
    "sum_list": """
def step(a: int, b: int) -> int:
    xs = [a, b, a]
    return sum(xs)


def root(a: int, b: int):
    return step(a, b) + 0
""",
    "sum_squares": """
def step(a: int, b: int) -> int:
    xs = [a, b]
    return sum(x * x for x in xs)


def root(a: int, b: int):
    return step(a, b) + 0
""",
    "builtin_pow": """
def step(a: int) -> int:
    return pow(a, 2)


def root(a: int, b: int):
    return step(a) + b
""",
    "loop_add": """
def step(a: int, b: int) -> int:
    x = 0
    for _ in range(b):
        x += a
    return x


def root(a: int, b: int):
    return step(a, b) + 0
""",
}


def _lower(source):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "root", name="int_python_semantics",
        extraction_contract=str(CONTRACT), progress=lambda _message: None,
    )
    root = module.functions[exports[0]]
    outputs = [
        int(value.id)
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret" and not instruction.attributes.get("fault_exit")
        for value in instruction.args
    ]
    return module, exports[0], dict(root.metadata["parameter_names"]), outputs


def _inputs(parameters, a, b):
    return {
        parameters["a"]: np.array([a], dtype=np.int64),
        parameters["b"]: np.array([b], dtype=np.int64),
    }


def _c_lane(tmp_path, source, a, b):
    from src.compiler.ssa_c_backend import emit_ssa_to_c

    module, export, parameters, outputs = _lower(source)
    artifact = emit_ssa_to_c(module, export)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "native")
    execution = artifact.prepare_execution(_inputs(parameters, a, b)).run()
    return int(execution.buffers[outputs[0]].reshape(-1)[0])


def _llvm_lane(tmp_path, source, a, b):
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )

    module, export, parameters, outputs = _lower(source)
    emitted = emit_ssa_function_to_llvm(module, export)
    assert not emitted.shortfalls, emitted.shortfalls
    artifact = compile_artifact(emitted, directory=tmp_path)
    execution = prepare_artifact_execution(
        artifact, _inputs(parameters, a, b)
    ).run()
    return int(execution.buffers[outputs[0]].reshape(-1)[0])


LANES = pytest.mark.parametrize("lane", [_c_lane, _llvm_lane], ids=["c", "llvm"])

EXACT = [
    ("shl", 3, 61), ("shl", -1, 63), ("shl", 0, 100), ("shl", -3, 3),
    ("shr", -8, 1), ("shr", -8, 100), ("shr", 8, 100), ("shr", INT64_MIN, 63),
    ("floordiv", -7, 2), ("floordiv", INT64_MIN, 1), ("floordiv", 7, -1),
    ("mod", -7, 2), ("mod", INT64_MIN, -1), ("mod", 7, -3),
    ("abs", -5, 0), ("abs", 2 ** 63 - 1, 0), ("neg", 2 ** 63 - 1, 0),
    ("square", 3037000499, 0),
    ("loop_mul", 3, 39), ("loop_add", 2 ** 62, 1),
    ("sum_list", 2 ** 61, 2 ** 61), ("sum_list", -5, 7),
    ("sum_squares", 2 ** 31, 2 ** 30), ("builtin_pow", -3037000499, 0),
]


def _source(program):
    return _binary_source(BINARY[program]) if program in BINARY else UNARY_SOURCE[program]


def _python(program, a, b):
    namespace = {}
    exec(_source(program), namespace)
    return namespace["root"](a, b)


@LANES
@pytest.mark.parametrize("program,a,b", EXACT)
def test_python_int_result_is_exact(tmp_path, lane, program, a, b):
    assert lane(tmp_path, _source(program), a, b) == _python(program, a, b)


FAULTS = [
    ("shl", 3, 62, "NativeIntOverflow"),
    ("shl", 1, 64, "NativeIntOverflow"),
    ("shl", 5, -1, "NativeValueError"),
    ("shr", 5, -1, "NativeValueError"),
    ("floordiv", INT64_MIN, -1, "NativeIntOverflow"),
    ("floordiv", 7, 0, "NativeZeroDivisionError"),
    ("mod", 7, 0, "NativeZeroDivisionError"),
    ("abs", INT64_MIN, 0, "NativeIntOverflow"),
    ("neg", INT64_MIN, 0, "NativeIntOverflow"),
    ("square", 3037000500, 0, "NativeIntOverflow"),
    ("loop_mul", 3, 40, "NativeIntOverflow"),
    ("loop_add", 2 ** 62, 2, "NativeIntOverflow"),
    ("sum_list", 2 ** 62, 0, "NativeIntOverflow"),
    ("sum_squares", 2 ** 31 * 3, 2 ** 31 * 3, "NativeIntOverflow"),
    ("builtin_pow", 3037000500, 0, "NativeIntOverflow"),
]


@LANES
@pytest.mark.parametrize("program,a,b,fault", FAULTS)
def test_python_int_exception_is_named(tmp_path, lane, program, a, b, fault):
    from src.compiler import native_faults

    python_exception = {
        "NativeValueError": ValueError,
        "NativeZeroDivisionError": ZeroDivisionError,
    }.get(fault)
    if python_exception is not None:
        with pytest.raises(python_exception):
            _python(program, a, b)
    with pytest.raises(getattr(native_faults, fault)):
        lane(tmp_path, _source(program), a, b)


# --- targets whose int64 is 32-bit or a double refuse, by name ----------


def _refusals(shortfalls):
    return [
        shortfall for shortfall in shortfalls
        if "python_int_overflow_checked" in str(shortfall.reason)
    ]


def test_javascript_refuses_unproven_int64_arithmetic():
    from src.compiler.ssa_javascript_backend import emit_ssa_module_to_javascript

    module, export, _parameters, _outputs = _lowered()
    artifact = emit_ssa_module_to_javascript(module, function_name=export)
    assert not artifact.complete
    assert any(shortfall.operation == "Mul" for shortfall in _refusals(artifact.shortfalls))


def test_wasm_refuses_unproven_int64_arithmetic():
    from src.compiler.ssa_wasm_backend import (
        emit_ssa_function_to_wasm, emit_ssa_module_to_wasm_core,
    )

    module, export, _parameters, _outputs = _lowered()
    core = emit_ssa_module_to_wasm_core(module, export)
    assert any(shortfall.operation == "Mul" for shortfall in _refusals(core.shortfalls))
    region = next(
        name for name in module.functions
        if "__mul__" in name and "planned_region" in name
    )
    direct = emit_ssa_function_to_wasm(module, region)
    assert any(shortfall.operation == "Mul" for shortfall in _refusals(direct.shortfalls))


# --- a proven interval needs no check, and every target may lower it ----

PROVEN_SOURCE = """
def step(a: int, b: int) -> int:
    x = 3
    if a > b:
        x = 7
    return x * x + 1


def root(a: int, b: int):
    return step(a, b) + 2
"""


def _proven_module():
    module, export, _parameters, _outputs = _lower(PROVEN_SOURCE)
    return module, export


def test_proven_int_arithmetic_carries_its_interval_and_is_not_checked():
    from src.compiler.ssa_c_backend import emit_ssa_to_c
    from src.compiler.ssa_llvm_backend import (
        python_int_operation, python_int_overflow_checked,
    )

    module, export = _proven_module()
    authored = [
        instruction
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
        if python_int_operation(instruction)
    ]
    assert {instruction.op for instruction in authored} >= {"Add", "Mul"}
    assert all(
        instruction.attributes.get("python_int_range") is not None
        and not python_int_overflow_checked(instruction)
        for instruction in authored
    )
    source = emit_ssa_to_c(module, export).source
    assert "_overflow(" not in source


@LANES
@pytest.mark.parametrize("a", [5, -1])
def test_proven_int_arithmetic_matches_python(tmp_path, lane, a):
    namespace = {}
    exec(PROVEN_SOURCE, namespace)
    assert lane(tmp_path, PROVEN_SOURCE, a, 0) == namespace["root"](a, 0)


def test_javascript_lowers_proven_int_arithmetic():
    from src.compiler.ssa_javascript_backend import emit_ssa_module_to_javascript

    module, export = _proven_module()
    artifact = emit_ssa_module_to_javascript(module, function_name=export)
    assert artifact.complete, artifact.shortfalls


def test_wasm_lowers_proven_int_arithmetic():
    from src.compiler.ssa_wasm_backend import emit_ssa_function_to_wasm

    module, _export = _proven_module()
    regions = [
        name for name in module.functions
        if "__step__" in name and "planned_region" in name
    ]
    assert regions
    for region in regions:
        artifact = emit_ssa_function_to_wasm(module, region)
        assert not _refusals(artifact.shortfalls), artifact.shortfalls


# --- a range loop's step never wraps past its stop ----------------------

RANGE_SOURCE = """
def step(a: int, b: int) -> int:
    count = 0
    for _ in range(a, b, STEP):
        count += 1
    return count


def root(a: int, b: int):
    return step(a, b) + 0
"""


@LANES
@pytest.mark.parametrize("step,a,b", [
    (3, 2 ** 63 - 5, 2 ** 63 - 1),
    (3, 0, 10),
    (-3, INT64_MIN + 5, INT64_MIN),
    (-3, 10, 0),
])
def test_range_step_near_int64_ends_where_python_does(tmp_path, lane, step, a, b):
    source = RANGE_SOURCE.replace("STEP", str(step))
    assert lane(tmp_path, source, a, b) == len(range(a, b, step))
