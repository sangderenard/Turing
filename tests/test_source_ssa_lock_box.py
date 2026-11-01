"""Token-lock preparation from actual Python mutex objects and source calls."""

import pytest

from tests.test_source_ssa_synchronization import compile_source, synchronization
from src.compiler.identity_concordance import CorrelationTable
from src.compiler.concordance_declarations import SOURCE_MUTEX_BINDING


def test_plain_mutex_aliases_share_one_lock_box_entry_and_token_protocol():
    module = compile_source("""\
import threading
def root():
    mutex = threading.Lock()
    alias = mutex
    acquired = alias.acquire(blocking=False)
    held = mutex.locked()
    mutex.release()
    return acquired
""")
    assert module.lock_box.book is module.metadata["identity_book"]
    assert len(module.lock_box.entries) == 1
    entry, descriptor = next(iter(module.lock_box.entries.items()))
    assert descriptor[0] is False
    assert descriptor[2] == "mutex_handle"
    assert {"held_token", "owner_token", "depth", "generation"} <= set(descriptor[3])
    sync = synchronization(module)
    assert [i.op for i in sync] == ["MutexCreate", "WorkerToken", "MutexAcquire", "MutexAcquired",
                                    "MutexLocked", "WorkerToken", "MutexToken", "MutexRelease"]
    assert {i.attributes["lock_box_entry"] for i in sync} == {entry}
    acquire, acquired = sync[2:4]
    assert acquire.res.dtype == "ssa.lock_token"
    assert acquired.args[0] is acquire.res and acquired.res.dtype == "bool"
    assert not acquire.attributes["requires_concurrent_progress"]
    assert acquire.args[acquire.arg_roles.index("worker")] is sync[1].res
    release = sync[-1]
    assert release.args[release.arg_roles.index("token")] is sync[-2].res
    assert release.args[release.arg_roles.index("worker")] is sync[-3].res
    assert CorrelationTable._synchronization_findings(module) == []


def test_plain_and_reentrant_mutexes_keep_distinct_entries():
    module = compile_source("""\
import threading
def root():
    first = threading.Lock()
    second = threading.RLock()
    first.acquire()
    second.acquire()
    second.acquire()
    second.release()
    second.release()
    first.release()
    return 1
""")
    assert len(module.lock_box.entries) == 2
    assert {descriptor[0] for descriptor in module.lock_box.entries.values()} == {False, True}
    acquired = [i for i in synchronization(module) if i.op == "MutexAcquire"]
    assert acquired[1].attributes["lock_box_entry"] == acquired[2].attributes["lock_box_entry"]
    assert acquired[0].attributes["lock_box_entry"] != acquired[1].attributes["lock_box_entry"]


def test_worker_capture_keeps_the_parent_mutex_entry():
    module = compile_source("""\
import threading
def root():
    mutex = threading.Lock()
    def worker():
        mutex.acquire()
        mutex.release()
    first = threading.Thread(target=worker)
    first.start()
    first.join()
    return 1
""")
    (entry,) = module.lock_box.entries
    bindings = module.metadata["identity_book"].page(SOURCE_MUTEX_BINDING)
    assert len(tuple(bindings.rows())) >= 2
    assert {bindings.latest(row) for row in bindings.rows()} == {entry}


def test_mutex_context_cleanup_survives_an_early_return():
    module = compile_source("""\
import threading
def root():
    mutex = threading.RLock()
    with mutex as acquired:
        return acquired
""")
    assert len(module.lock_box.entries) == 1
    assert next(iter(module.lock_box.entries.values()))[0] is True
    sync = synchronization(module)
    assert [i.op for i in sync] == ["MutexCreate", "WorkerToken", "MutexAcquire", "MutexAcquired",
                                    "WorkerToken", "MutexToken", "MutexRelease"]


def test_explicit_mutex_intrinsics_use_the_same_lock_box():
    module = compile_source("""\
from src.compiler.synchronization import mutex_create, mutex_acquire, mutex_release
def root():
    mutex = mutex_create(256, False)
    acquired = mutex_acquire(mutex, False, -1)
    mutex_release(mutex)
    return acquired
""")
    assert len(module.lock_box.entries) == 1
    assert [i.op for i in synchronization(module)] == ["MutexCreate", "WorkerToken", "MutexAcquire", "MutexAcquired",
                                                      "WorkerToken", "MutexToken", "MutexRelease"]


def test_unrelated_acquire_and_release_methods_do_not_allocate_a_lock_box_entry():
    module = compile_source("""\
class Other:
    def acquire(self):
        return True
    def release(self):
        return None
def root():
    other = Other()
    acquired = other.acquire()
    other.release()
    return acquired
""")
    assert module.lock_box.entries == {}
    assert synchronization(module) == []


@pytest.mark.parametrize("preference", ["asap", "alap"])
def test_condition_reuses_lock_box_and_captured_mutex_protocol(preference):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.identity_concordance import concordance_report
    from src.transmogrifier.ssa import SSALockBox
    from tests.test_source_ssa_synchronization import CONTRACT
    module, _, _ = lower_ast_source_to_ssa("""import threading
def root():
    condition = threading.Condition()
    alias = condition
    def worker():
        alias.acquire()
        alias.release()
    task = threading.Thread(target=worker)
    task.start()
    task.join()
    return 1
""", "root", extraction_contract=CONTRACT, schedule_preference=preference,
        progress=lambda _: None, progress_bars=False)
    (entry,) = module.lock_box.entries
    descriptor = module.lock_box.entries[entry]
    assert descriptor[0] is True and descriptor[1][0] == 257
    assert descriptor[3] == SSALockBox.CONDITION_STATE_COLUMNS
    assert descriptor[4] == (("notification_head", 0), ("notification_tail", 0))
    stream = [i for f in module.functions.values() for b in f.blocks.values() for i in b.instrs]
    sync = [i for i in stream if i.attributes.get("synchronization_receipt")]
    assert [i.op for i in sync] == ["MutexCreate", "WorkerToken", "WaitQueueMutex", "MutexAcquire", "MutexAcquired",
                                    "WorkerToken", "WaitQueueMutex", "MutexToken", "MutexRelease"]
    assert {i.attributes["lock_box_entry"] for i in sync} == {entry}
    assert not {i.op for i in stream} & {"ConditionCreate", "ConditionAcquire", "ConditionRelease"}
    bindings = module.metadata["identity_book"].page(SOURCE_MUTEX_BINDING)
    assert len(tuple(bindings.rows())) >= 2
    assert {bindings.latest(row) for row in bindings.rows()} == {entry}
    assert "0 finding(s)" in concordance_report(module)


def test_condition_nonblocking_acquire_preserves_authored_operands():
    module = compile_source("""import threading
def root():
    condition = threading.Condition()
    acquired = condition.acquire(blocking=False)
    return acquired
""")
    stream = [i for f in module.functions.values() for b in f.blocks.values() for i in b.instrs]
    literals = {i.res.id: i.attributes["value"] for i in stream if i.op == "Const" and i.res is not None}
    (acquire,) = [i for i in stream if i.op == "MutexAcquire"]
    assert literals[acquire.args[acquire.arg_roles.index("blocking")].id] is False
    assert literals[acquire.args[acquire.arg_roles.index("timeout")].id] == -1
    assert not acquire.attributes["requires_concurrent_progress"]
    assert CorrelationTable._synchronization_findings(module) == []


def test_condition_supplied_mutex_is_not_silently_replaced_with_default_lock():
    module = compile_source("""import threading
def root():
    lock = threading.Lock()
    condition = threading.Condition(lock)
    return 1
""")
    sync = synchronization(module)
    assert [i.op for i in sync] == ["MutexCreate", "WaitQueueCreate"]
    assert sync[1].args[0] is sync[0].res
    assert sync[1].attributes["lock_box_entry"] == sync[0].attributes["lock_box_entry"]
    assert len(module.lock_box.entries) == 1
    assert next(iter(module.lock_box.entries.values()))[0] is False
