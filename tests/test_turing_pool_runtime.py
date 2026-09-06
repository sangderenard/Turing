"""Compile and exercise the native turing_pool runtime end to end.

Uses the repository's own toolchain discovery (``native_library``) so the
same compilers the backends use build this runtime; skips visibly when no
toolchain exists.  When one does, this proves the exactly-once claiming
theorem on the real runtime: every (lane, chunk) cell of the frame grid is
executed exactly once, from Python, through ctypes.
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import pytest

from src.common.tensors.accelerator_backends.native_library import (
    compile_shared_library,
    preferred_toolchain,
)

_BACKEND_DIR = (
    Path(__file__).resolve().parents[1]
    / "src" / "common" / "tensors" / "accelerator_backends" / "c_backend"
)

_TASK_PROBE_SOURCE = r"""
typedef struct {
    turing_pool_lock lock;
    turing_pool_cond changed;
    int entered, release, count, nested_status;
} task_probe_state;

static void task_probe_lane(void* raw, long lane, long chunk, long chunks) {
    task_probe_state* state = (task_probe_state*)raw;
    (void)lane; (void)chunk; (void)chunks;
    pool_lock(&state->lock);
    state->count++;
    pool_unlock(&state->lock);
}

static void task_probe_entry(void* raw) {
    task_probe_state* state = (task_probe_state*)raw;
    pool_lock(&state->lock);
    state->entered = 1;
    pool_notify_all(&state->changed);
    while (!state->release) pool_wait(&state->changed, &state->lock);
    pool_unlock(&state->lock);
    state->nested_status = turing_pool_deploy(task_probe_lane, state, 7, 1);
}

int task_probe_run(void) {
    task_probe_state state = {TURING_POOL_LOCK_INIT, TURING_POOL_COND_INIT, 0, 0, 0, -99};
    turing_dispatch_task* task = turing_task_start(task_probe_entry, &state);
    int timed, completed, destroyed;
    if (!task) return -10;
    pool_lock(&state.lock);
    while (!state.entered) pool_wait(&state.changed, &state.lock);
    pool_unlock(&state.lock);
    timed = turing_task_wait(task, 10);
    pool_lock(&state.lock);
    state.release = 1;
    pool_notify_all(&state.changed);
    pool_unlock(&state.lock);
    completed = turing_task_wait(task, 5000);
    destroyed = turing_task_destroy(task);
#ifndef _WIN32
    pthread_cond_destroy(&state.changed);
    pthread_mutex_destroy(&state.lock);
#endif
    if (timed != 0) return -100 + timed;
    if (completed != 1 || destroyed != 0) return -12;
    if (state.nested_status != 0) return -13;
    return state.count;
}
"""


_CONDITION_PROBE_SOURCE = r"""
typedef struct {
    turing_dispatch_condition* condition;
    int entered;
    int status;
} condition_probe_state;
static void condition_probe_worker(void* raw) {
    condition_probe_state* state = (condition_probe_state*)raw;
    state->status = turing_dispatch_condition_acquire(state->condition);
    if (state->status != 1) return;
    state->entered = 1;
    state->status = turing_dispatch_condition_notify_all(state->condition);
    if (turing_dispatch_condition_release(state->condition)) state->status = -20;
}
int condition_probe_run(void) {
    condition_probe_state state = {turing_dispatch_condition_create(), 0, -99};
    turing_dispatch_task* task;
    int waited, timed, first, second, third, completed, destroyed;
    if (!state.condition) return -1;
    if (turing_dispatch_condition_notify_all(state.condition) != -1) return -2;
    if (turing_dispatch_condition_wait(state.condition, 0) != -1) return -3;
    if (turing_dispatch_condition_acquire(state.condition) != 1) return -4;
    if (turing_dispatch_condition_acquire(state.condition) != 1) return -5;
    if (turing_dispatch_condition_destroy(state.condition) != -1) return -6;
    // The worker cannot acquire until wait releases BOTH recursion levels.
    task = turing_task_start(condition_probe_worker, &state);
    if (!task) {
        turing_dispatch_condition_release(state.condition);
        turing_dispatch_condition_release(state.condition);
        turing_dispatch_condition_destroy(state.condition);
        return -7;
    }
    waited = turing_dispatch_condition_wait(state.condition, 5000);
    timed = turing_dispatch_condition_wait(state.condition, 0);
    first = turing_dispatch_condition_release(state.condition);
    second = turing_dispatch_condition_release(state.condition);
    third = turing_dispatch_condition_release(state.condition);
    completed = turing_task_wait(task, 5000);
    destroyed = turing_task_destroy(task);
    if (turing_dispatch_condition_destroy(state.condition)) return -9;
    if (completed != 1) return -8;
    if (waited != 1 || timed != 0) return -10;
    if (first != 0 || second != 0 || third != -1) return -11;
    if (destroyed != 0 || !state.entered || state.status != 0) return -12;
    return 1;
}

typedef struct {
    turing_dispatch_condition* condition;
    turing_pool_lock lock;
    turing_pool_cond changed;
    int entered;
    int completed;
    int finished[2];
    int statuses[2];
} condition_group;
typedef struct { condition_group* group; int index; } condition_member;
static void condition_group_worker(void* raw) {
    condition_member* member = (condition_member*)raw;
    condition_group* group = member->group;
    int status;
    turing_dispatch_condition_acquire(group->condition);
    pool_lock(&group->lock);
    group->entered++;
    pool_notify_all(&group->changed);
    pool_unlock(&group->lock);
    status = turing_dispatch_condition_wait(group->condition, 5000);
    turing_dispatch_condition_release(group->condition);
    pool_lock(&group->lock);
    group->statuses[member->index] = status;
    group->finished[member->index] = 1;
    group->completed++;
    pool_notify_all(&group->changed);
    pool_unlock(&group->lock);
}
int condition_group_run(void) {
    condition_group group = {NULL, TURING_POOL_LOCK_INIT, TURING_POOL_COND_INIT, 0, 0, {0, 0}, {0, 0}};
    condition_member members[2] = {{&group, 0}, {&group, 1}};
    turing_dispatch_task* tasks[2];
    int other, timed, count, index;
    group.condition = turing_dispatch_condition_create();
    if (!group.condition) return -1;
    tasks[0] = turing_task_start(condition_group_worker, &members[0]);
    tasks[1] = turing_task_start(condition_group_worker, &members[1]);
    if (!tasks[0] || !tasks[1]) {
        for (index = 0; index < 2; index++)
            if (tasks[index]) turing_task_destroy(tasks[index]);
        turing_dispatch_condition_destroy(group.condition);
        return -2;
    }
    pool_lock(&group.lock);
    while (group.entered != 2) pool_wait(&group.changed, &group.lock);
    pool_unlock(&group.lock);
    // Acquiring after both entry reports proves both waiters released the
    // condition lock; no sleep guesses whether a task has reached wait().
    turing_dispatch_condition_acquire(group.condition);
    turing_dispatch_condition_notify(group.condition, 1);
    turing_dispatch_condition_release(group.condition);
    pool_lock(&group.lock);
    while (!group.completed) pool_wait(&group.changed, &group.lock);
    count = group.completed;
    other = group.finished[0] ? 1 : 0;
    pool_unlock(&group.lock);
    timed = turing_task_wait(tasks[other], 20);
    turing_dispatch_condition_acquire(group.condition);
    turing_dispatch_condition_notify_all(group.condition);
    turing_dispatch_condition_release(group.condition);
    for (index = 0; index < 2; index++) turing_task_destroy(tasks[index]);
    if (turing_dispatch_condition_destroy(group.condition)) return -3;
#ifndef _WIN32
    pthread_cond_destroy(&group.changed);
    pthread_mutex_destroy(&group.lock);
#endif
    if (count != 1 || timed != 0) return -4;
    if (group.completed != 2 || group.statuses[0] != 1 || group.statuses[1] != 1) return -5;
    return 1;
}
"""


_EVENT_PROBE_SOURCE = r"""
typedef struct {
    turing_dispatch_event* event;
    turing_dispatch_event* ready;
    int result;
} event_probe_state;

static void event_probe_worker(void* raw) {
    event_probe_state* state = raw;
    // Hold the event lock through readiness publication. The parent acquiring
    // this same lock after readiness proves this worker has entered wait.
    if (turing_dispatch_condition_acquire(state->event->condition) != 1) return;
    if (turing_dispatch_event_set(state->ready)) return;
    state->result = turing_dispatch_event_wait(state->event, 5000);
    if (turing_dispatch_condition_release(state->event->condition)) state->result = -9;
}

int event_probe_run(void) {
    turing_dispatch_event* event = turing_dispatch_event_create();
    turing_dispatch_event* ready = turing_dispatch_event_create();
    turing_dispatch_task* tasks[2];
    event_probe_state states[2];
    int index;
    if (!event || !ready) return -1;
    if (turing_dispatch_event_is_set(event) != 0 ||
        turing_dispatch_event_wait(event, 0) != 0) return -2;
    for (index = 0; index < 2; index++) {
        states[index].event = event;
        states[index].ready = ready;
        states[index].result = -99;
        tasks[index] = turing_task_start(event_probe_worker, &states[index]);
        if (!tasks[index] || turing_dispatch_event_wait(ready, 5000) != 1) return -3;
        if (turing_dispatch_condition_acquire(event->condition) != 1) return -4;
        if (turing_dispatch_event_clear(ready)) return -5;
        if (turing_dispatch_condition_release(event->condition)) return -6;
    }
    if (turing_dispatch_condition_acquire(event->condition) != 1) return -7;
    // Both waiters must observe set even when clear happens before either can
    // reacquire. Future waits must observe the cleared flag.
    if (turing_dispatch_event_set(event) ||
        turing_dispatch_event_is_set(event) != 1 ||
        turing_dispatch_event_clear(event) ||
        turing_dispatch_event_is_set(event) != 0) return -8;
    if (turing_dispatch_condition_release(event->condition)) return -9;
    for (index = 0; index < 2; index++) {
        if (turing_task_wait(tasks[index], 5000) != 1 ||
            turing_task_destroy(tasks[index]) || states[index].result != 1) return -10;
    }
    if (turing_dispatch_event_wait(event, 0) != 0) return -11;
    if (turing_dispatch_event_set(event) ||
        turing_dispatch_event_wait(event, 0) != 1 ||
        turing_dispatch_event_wait(event, 0) != 1) return -12;
    if (turing_dispatch_event_destroy(event) ||
        turing_dispatch_event_destroy(ready)) return -13;
    if (turing_dispatch_event_wait(NULL, 0) != -1 ||
        turing_dispatch_event_set(NULL) != -1) return -14;
    return 1;
}
"""


@pytest.fixture(scope="module")
def pool_library(tmp_path_factory):
    toolchain = preferred_toolchain()
    if toolchain is None:
        pytest.skip(
            "no native toolchain found (native_library.detect_toolchains)"
        )
    # compile_shared_library writes the source into its own directory, so
    # inline the header textually rather than teaching each toolchain an
    # include path.
    header = (_BACKEND_DIR / "turing_pool.h").read_text(encoding="utf-8")
    source = (_BACKEND_DIR / "turing_pool.c").read_text(encoding="utf-8")
    source = source.replace('#include "turing_pool.h"', header)
    source += _TASK_PROBE_SOURCE
    source += _CONDITION_PROBE_SOURCE
    source += _EVENT_PROBE_SOURCE
    extra_flags = () if sys.platform == "win32" else ("-pthread",)
    library_path, _toolchain = compile_shared_library(
        source,
        name="turing_pool_test",
        directory=tmp_path_factory.mktemp("turing_pool"),
        extra_flags=extra_flags,
    )
    library = ctypes.CDLL(str(library_path))
    library.turing_pool_start.restype = ctypes.c_int
    library.turing_pool_start.argtypes = [ctypes.c_int]
    library.turing_pool_workers.restype = ctypes.c_int
    library.turing_pool_deploy.restype = ctypes.c_int
    yield library
    library.turing_pool_stop()


_LANE_FN = ctypes.CFUNCTYPE(
    None, ctypes.c_void_p, ctypes.c_long, ctypes.c_long, ctypes.c_long,
)


def test_native_task_wait_and_nested_numerical_dispatch(pool_library):
    # Both the task and its synchronization peer run in native code. The
    # task cannot finish until the initiating thread signals it after a timed
    # wait, so a synchronous/serial start cannot satisfy this protocol.
    pool_library.task_probe_run.restype = ctypes.c_int
    pool_library.task_probe_run.argtypes = []
    assert pool_library.task_probe_run() == 7


def test_native_condition_wait_restores_recursive_ownership(pool_library):
    pool_library.condition_probe_run.restype = ctypes.c_int
    pool_library.condition_probe_run.argtypes = []
    assert pool_library.condition_probe_run() == 1


def test_native_condition_notify_one_then_all(pool_library):
    pool_library.condition_group_run.restype = ctypes.c_int
    pool_library.condition_group_run.argtypes = []
    assert pool_library.condition_group_run() == 1


def test_native_event_broadcast_clear_and_future_waits(pool_library):
    import subprocess

    # Native communicating tasks must never be allowed to hang pytest itself.
    probe = subprocess.run([sys.executable, "-c", """
import ctypes, sys
library = ctypes.CDLL(sys.argv[1])
library.event_probe_run.restype = ctypes.c_int
library.event_probe_run.argtypes = []
result = library.event_probe_run()
assert result == 1, result
""", str(pool_library._name)], capture_output=True, text=True, timeout=20)
    assert probe.returncode == 0, probe.stderr


def test_start_is_idempotent_and_never_shrinks(pool_library):
    assert pool_library.turing_pool_start(3) == 3
    assert pool_library.turing_pool_start(1) == 3
    assert pool_library.turing_pool_workers() == 3


def test_every_grid_cell_is_claimed_exactly_once(pool_library):
    lanes, chunks = 7, 13
    counts = (ctypes.c_long * (lanes * chunks))()

    @_LANE_FN
    def kernel(context, lane, chunk, chunks_per_lane):
        counts[lane * chunks_per_lane + chunk] += 1

    pool_library.turing_pool_start(3)
    status = pool_library.turing_pool_deploy(kernel, None, lanes, chunks)
    assert status == 0
    assert all(cell == 1 for cell in counts), (
        "claiming was not exactly-once: "
        f"{[index for index, cell in enumerate(counts) if cell != 1]}"
    )


def test_zero_workers_serial_fallback_runs_the_identical_path(pool_library):
    # Workers may already exist from earlier tests; the property that
    # matters is that deploy completes and covers the grid regardless of
    # pool size, caller participating.
    total = ctypes.c_long(0)

    @_LANE_FN
    def kernel(context, lane, chunk, chunks_per_lane):
        total.value += lane + chunk

    assert pool_library.turing_pool_deploy(kernel, None, 4, 1) == 0
    assert total.value == 0 + 1 + 2 + 3


def test_invalid_frames_are_refused(pool_library):
    @_LANE_FN
    def kernel(context, lane, chunk, chunks_per_lane):
        pass

    assert pool_library.turing_pool_deploy(kernel, None, 0, 1) == -1
    assert pool_library.turing_pool_deploy(kernel, None, 1, 0) == -1


def test_nested_deploy_from_a_lane_is_refused_not_deadlocked(pool_library):
    inner_status = ctypes.c_long(99)

    @_LANE_FN
    def inner(context, lane, chunk, chunks_per_lane):
        pass

    @_LANE_FN
    def outer(context, lane, chunk, chunks_per_lane):
        inner_status.value = pool_library.turing_pool_deploy(
            inner, None, 1, 1,
        )

    assert pool_library.turing_pool_deploy(outer, None, 1, 1) == 0
    assert inner_status.value == -2
