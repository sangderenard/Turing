"""Authored Conditions -> composed SSA -> standalone compiled C applications."""
import subprocess
import sys

import numpy as np
import pytest

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import concordance_report
from src.compiler.ssa_c_backend import emit_ssa_module_to_c
from src.transmogrifier.dtype_layout import c_lane_numpy_dtype
from tests.test_source_ssa_synchronization import CONTRACT


CASES = {
    "recursive_timeout": """import threading
def root():
    condition = threading.Condition()
    with condition:
        condition.acquire()
        notified = condition.wait(timeout=0.001)
        condition.release()
    return 9 if notified else 17
""",
    "handshake": """import threading
def root():
    condition = threading.Condition()
    shared = [0, 0]
    def worker():
        with condition:
            shared[0] = 1
            condition.notify()
            while shared[1] == 0:
                condition.wait()
            shared[0] = 3
    task = threading.Thread(target=worker)
    task.start()
    with condition:
        while shared[0] == 0:
            condition.wait()
        shared[1] = 2
        condition.notify_all()
    task.join()
    return shared[0] * 10 + shared[1]
""",
    "notify_count": """import threading
def root():
    condition = threading.Condition()
    shared = [0, 0, 0]
    def worker():
        with condition:
            shared[0] += 1
            condition.notify_all()
            while shared[1] == 0:
                condition.wait()
            shared[1] -= 1
            shared[2] += 1
            condition.notify_all()
    first = threading.Thread(target=worker)
    second = threading.Thread(target=worker)
    third = threading.Thread(target=worker)
    first.start()
    second.start()
    third.start()
    with condition:
        while shared[0] < 3:
            condition.wait()
        shared[1] = 1
        condition.notify(1)
        while shared[2] < 1:
            condition.wait()
        shared[1] = 2
        condition.notify_all()
    first.join()
    second.join()
    third.join()
    return shared[2]
""",
    "invalid_wait": """import threading
def root():
    condition = threading.Condition()
    try:
        condition.wait(timeout=0)
    except RuntimeError:
        return 17
    return 9
""",
    "invalid_notify": """import threading
def root():
    condition = threading.Condition()
    try:
        condition.notify()
    except RuntimeError:
        return 17
    return 9
""",
}

CASES["shared_rlock"] = CASES["handshake"].replace("condition = threading.Condition()",
    "lock = threading.RLock()\n    condition = threading.Condition(lock)")
CASES["shared_plain_lock"] = CASES["handshake"].replace("condition = threading.Condition()",
    "lock = threading.Lock()\n    condition = threading.Condition(lock=lock)")
CASES["separate_queues_same_lock"] = """import threading
def root():
    lock = threading.RLock()
    first = threading.Condition(lock)
    second = threading.Condition(lock)
    shared = [0, 0]
    def worker():
        with second:
            shared[0] = 1
            first.notify()
            notified = second.wait(timeout=0.05)
            shared[1] = 1 if notified else 2
    task = threading.Thread(target=worker)
    task.start()
    with first:
        while shared[0] == 0:
            first.wait()
        first.notify_all()
    task.join()
    return shared[1]
"""
CASES["negative_poll"] = CASES["recursive_timeout"].replace("timeout=0.001", "timeout=-1")
CASES["explicit_none"] = CASES["handshake"].replace("condition.wait()", "condition.wait(timeout=None)")
CASES["notification_not_sticky"] = """import threading
def root():
    condition = threading.Condition()
    with condition:
        condition.notify_all()
        condition.notify(0)
        notified = condition.wait(timeout=0)
    return 9 if notified else 17
"""
CASES["nested_return_cleanup"] = """import threading
def root():
    outer = threading.Condition()
    inner = threading.Condition()
    with outer:
        with inner:
            return 17
"""
CASES["nested_break_cleanup"] = """import threading
def root():
    outer = threading.Condition()
    inner = threading.Condition()
    with outer:
        for i in range(2):
            with inner:
                break
        outer.notify_all()
    return 17
"""
CASES["worker_try_loop"] = """import threading
def root():
    condition = threading.Condition()
    shared = [0]
    def worker():
        i = 0
        while i < 3:
            try:
                condition.wait(timeout=0)
            except RuntimeError:
                shared[0] = i + 1
            i += 1
    task = threading.Thread(target=worker)
    task.start()
    task.join()
    return shared[0]
"""
CASES["external_lock_scope"] = """import threading
def root():
    lock = threading.RLock()
    condition = threading.Condition(lock)
    with lock:
        condition.notify_all()
        result = condition.wait(timeout=0)
    return 9 if result else 17
"""
CASES["unrelated_wait"] = """class Other:
    def wait(self):
        return 23
def root():
    other = Other()
    return other.wait()
"""
CASES["park_error_restores_recursive_lock"] = """import threading
def root():
    condition = threading.Condition()
    with condition:
        condition.acquire()
        try:
            condition.wait(timeout=1e309)
        except OverflowError:
            condition.release()
            return 17
        condition.release()
    return 9
"""
CASES["invalid_acquire_timeout"] = """import threading
def root():
    condition = threading.Condition()
    try:
        condition.acquire(timeout=-2)
    except ValueError:
        return 17
    return 9
"""
CASES["try_around_resource"] = """import threading
def root():
    condition = threading.Condition()
    try:
        with condition:
            condition.wait(timeout=1e309)
    except OverflowError:
        return 17
    return 9
"""
CASES["cleanup_failure_is_caught"] = """import threading
def root():
    condition = threading.Condition()
    try:
        with condition:
            condition.release()
    except RuntimeError:
        return 17
    return 9
"""
CASES["try_resource_return"] = """import threading
def root():
    condition = threading.Condition()
    try:
        with condition:
            return 17
    except RuntimeError:
        return 9
"""
CASES["notify_one_bound"] = """import threading
def root():
    lock = threading.RLock()
    gate = threading.Condition(lock)
    progress = threading.Condition(lock)
    shared = [0, 0]
    def worker():
        with gate:
            shared[0] += 1
            progress.notify_all()
            notified = gate.wait(timeout=0.1)
            if notified:
                shared[1] += 1
    first = threading.Thread(target=worker)
    second = threading.Thread(target=worker)
    third = threading.Thread(target=worker)
    first.start()
    second.start()
    third.start()
    with progress:
        while shared[0] < 3:
            progress.wait()
        gate.notify(1)
    first.join()
    second.join()
    third.join()
    return 1 if shared[1] > 1 else 0
"""


def test_escaping_condition_requires_explicit_lifetime_transfer():
    module, _, exports = lower_ast_source_to_ssa("""import threading
def root():
    return threading.Condition()
""", "root", extraction_contract=CONTRACT, progress=lambda _: None, progress_bars=False)
    artifact = emit_ssa_module_to_c(module, exports[0])
    assert not artifact.complete
    assert any("ownership-transfer" in item.reason for item in artifact.shortfalls)


@pytest.mark.parametrize("preference", ["asap", "alap"])
@pytest.mark.parametrize("case", CASES)
def test_composed_condition_runs_in_standalone_c(tmp_path, case, preference):
    source = CASES[case]
    reference = subprocess.run([sys.executable, "-c", source + "\nprint(root())"],
                               capture_output=True, text=True, timeout=10, check=True)
    expected = int(reference.stdout.strip())
    module, _, exports = lower_ast_source_to_ssa(source, "root", extraction_contract=CONTRACT,
        schedule_preference=preference, progress=lambda _: None, progress_bars=False)
    assert "0 finding(s)" in concordance_report(module)
    stream = [i for f in module.functions.values() for b in f.blocks.values() for i in b.instrs]
    assert not any(i.op.startswith("Condition") for i in stream)
    artifact = emit_ssa_module_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    assert "turing_dispatch_condition_wait(" not in artifact.source
    assert "turing_dispatch_condition_notify(" not in artifact.source
    from src.compiler.concordance_declarations import C_SYNCHRONIZATION_POLICY, SSA_SYNCHRONIZATION
    book = module.metadata["identity_book"]
    for row in book.page(C_SYNCHRONIZATION_POLICY).rows():
        policy = book.page(C_SYNCHRONIZATION_POLICY).latest(row)
        if policy[4] != "host_synchronization":
            continue
        operator = book.page(SSA_SYNCHRONIZATION).cell(row[0].row, row[0].column)[1]
        assert policy[2] == 257
        assert policy[3] == {"MutexAcquire": 2, "MutexRelease": 4,
                             "MutexSuspend": 4, "MutexRestore": 2}.get(operator, 0)
    executable = artifact.compile_standalone(tmp_path / case, {})
    process = subprocess.run([str(executable.executable_path), "3", "1"], cwd=executable.directory,
                             capture_output=True, text=True, timeout=15)
    assert process.returncode == 0, process.stderr
    raw, offset, outputs = executable.final_outputs_path.read_bytes(), 0, []
    for dtype, shape, role in zip(artifact.buffer_dtypes, artifact.buffer_shapes, artifact.buffer_roles, strict=True):
        count = int(np.prod(shape)) if shape else 1
        values = np.frombuffer(raw, dtype=c_lane_numpy_dtype(dtype), count=count, offset=offset)
        offset += values.nbytes
        if role == "output":
            outputs.extend(values.tolist())
    assert outputs == [expected]
    assert "0 finding(s)" in concordance_report(module)
