"""SSA exception record operators on the LLVM lane, at parity with the C lane.

The same lowered module is emitted to C (standalone shell) and to LLVM; both
link turing_validation_runtime's turing_error_packet ABI, so a raised
exception reaches each host as the same packet (type, message, suppressed
context) and a handled one returns the same outputs as Python."""
import ctypes
import subprocess

import numpy as np
import pytest

from tests.test_source_ssa_synchronization import compile_source
from src.compiler.concordance_declarations import (
    EMISSION_LLVM, LLVM_EXCEPTION_OPERATION, SOURCE_EXCEPTION_OPERATION,
)
from src.compiler.identity_concordance import Derived, Mode, concordance_report
from src.compiler.ssa_c_backend import emit_ssa_module_to_c
from src.compiler.ssa_llvm_backend import (
    compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
)
from src.transmogrifier.dtype_layout import c_lane_numpy_dtype


TRY_FINALLY_RAISE = '''def root(n: int) -> int:
    try:
        raise ValueError("authored failure")
    finally:
        value = 7
    return n
'''
NESTED_RETHROW = '''def root(n: int) -> int:
    value = 0
    try:
        try:
            raise ValueError("authored failure")
        except ValueError:
            value = 3
            raise
        finally:
            value = value + 2
    except ValueError:
        value = value * 10
    return value
'''
RAISE_FROM_NONE = '''def root(n: int) -> int:
    try:
        if n < 0:
            raise TypeError("negative")
    except TypeError:
        raise KeyError("hidden context") from None
    return n + 1
'''


class _Packet(ctypes.Structure):
    # turing_validation_runtime.h ``turing_error_packet``
    _fields_ = [
        ("code", ctypes.c_uint32), ("worker_token", ctypes.c_uint64),
        ("worker_function", ctypes.c_char_p), ("exception_type", ctypes.c_char_p),
        ("message", ctypes.c_char_p), ("suppress_context", ctypes.c_uint8),
    ]


def _python(source, n):
    namespace = {}
    exec(source, namespace)
    try:
        return ("ok", [namespace["root"](n)])
    except BaseException as error:
        return ("raised", f"builtins.{type(error).__name__}", str(error.args[0]))


def _llvm(module, tmp_path, n):
    root = module.functions["root__root"]
    emitted = emit_ssa_function_to_llvm(module, root.name)
    assert not emitted.shortfalls, emitted.shortfalls
    assert emitted.needs_validation_runtime
    artifact = compile_artifact(emitted, directory=tmp_path / "llvm")
    parameter = dict(root.metadata["parameter_names"]).get("n")
    feeds = {parameter: np.array([n], dtype=np.int64)} if parameter is not None else {}
    execution = prepare_artifact_execution(artifact, feeds)
    artifact.reset_validation_error()
    artifact.entry()(execution.pointers, execution.extents)
    take = artifact._library.turing_validation_error_take_packet
    take.argtypes, take.restype = [ctypes.POINTER(_Packet)], ctypes.c_int
    packet = _Packet()
    if take(ctypes.byref(packet)):
        return ("raised", packet.exception_type.decode(), packet.message.decode()), packet
    outputs = [
        value
        for value_id, role in zip(artifact.buffer_order, artifact.buffer_roles, strict=True)
        if role == "output"
        for value in np.asarray(execution.buffers[value_id]).reshape(-1).tolist()
    ]
    return ("ok", outputs), None


def _c(module, tmp_path, n):
    root = module.functions["root__root"]
    artifact = emit_ssa_module_to_c(module, root.name)
    assert artifact.complete, artifact.shortfalls
    parameter = dict(root.metadata["parameter_names"]).get("n")
    feeds = {parameter: np.array([n], dtype=np.int64)} if parameter is not None else {}
    executable = artifact.compile_standalone(tmp_path / "c", feeds)
    process = subprocess.run([str(executable.executable_path), "1", "0"], cwd=executable.directory,
                             capture_output=True, text=True, timeout=10)
    if process.returncode == 7:
        return process.stderr
    assert process.returncode == 0, process.stderr
    raw = executable.final_outputs_path.read_bytes()
    offset, outputs = 0, []
    for dtype, shape, role in zip(artifact.buffer_dtypes, artifact.buffer_shapes, artifact.buffer_roles, strict=True):
        count = int(np.prod(shape)) if shape else 1
        values = np.frombuffer(raw, dtype=c_lane_numpy_dtype(dtype), count=count, offset=offset)
        offset += values.nbytes
        if role == "output":
            outputs.extend(values.tolist())
    return ("ok", outputs)


@pytest.mark.parametrize("source, n, suppressed", [
    (TRY_FINALLY_RAISE, 4, 0),
    (NESTED_RETHROW, 0, None),
    (RAISE_FROM_NONE, 1, None),
    (RAISE_FROM_NONE, -1, 1),
])
def test_llvm_exception_records_match_python_and_the_c_lane(tmp_path, source, n, suppressed):
    expected = _python(source, n)
    module = compile_source(source)
    assert "0 finding(s)" in concordance_report(module)
    observed, packet = _llvm(module, tmp_path, n)
    assert observed == expected
    if packet is not None:
        assert packet.suppress_context == suppressed
    c_observed = _c(module, tmp_path, n)
    if expected[0] == "ok":
        assert c_observed == expected
    else:
        # The C shell reports the same packet the LLVM host took.
        assert f"type={expected[1]} message={expected[2]}" in c_observed
    book = module.metadata["identity_book"]
    policies = {book.page(LLVM_EXCEPTION_OPERATION).latest(row)[2]
                for row in book.page(LLVM_EXCEPTION_OPERATION).rows()}
    assert policies == {"native_record_and_control_edges"}
    assert "0 finding(s)" in concordance_report(module)


def test_audit_reads_the_llvm_exception_page_against_its_source_fact():
    module = compile_source(TRY_FINALLY_RAISE)
    emit_ssa_function_to_llvm(module, "root__root")
    assert "0 finding(s)" in concordance_report(module)
    book = module.metadata["identity_book"]
    row = next(row for row in book.page(LLVM_EXCEPTION_OPERATION).rows()
               if book.page(LLVM_EXCEPTION_OPERATION).latest(row)[0] == "ExceptionMake")
    source, function, _result = row
    assert source.page == SOURCE_EXCEPTION_OPERATION
    # A policy row naming a different operator than its source fact.
    book.post(LLVM_EXCEPTION_OPERATION, (source, function, -1),
              ("ExceptionMatch", (), "native_record_and_control_edges"), stage=EMISSION_LLVM,
              provenance=Derived((source,)), mode=Mode.CONCORD)
    report = concordance_report(module)
    assert "llvm-exception-provenance" in report
    assert "LLVM exception operation disagrees with its source identity" in report
