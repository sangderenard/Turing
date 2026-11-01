"""Exercise the real native byte deque, including communicating native tasks.

This verifies the C runtime ABI. It does not pretend to be Python-to-SSA lowering.
"""

import ctypes as C
from collections import deque
from pathlib import Path
import subprocess
import sys

import pytest

from src.common.tensors.accelerator_backends.native_library import (
    compile_shared_library, preferred_toolchain,
)

HOME = Path(__file__).resolve().parents[1] / "src/common/tensors/accelerator_backends/c_backend"


class Endpoint(C.Structure):
    _fields_ = [("queue", C.c_void_p), ("worker", C.c_uint64), ("end", C.c_int)]


class Record(C.Structure):
    _fields_ = [("length", C.c_size_t), ("type_tag", C.c_uint64),
                ("operation_contract", C.c_uint64), ("producer_worker", C.c_uint64)]


def bind(path):
    lib = C.CDLL(str(path))
    signatures = {
        "create": (C.c_void_p, [C.c_size_t, C.c_size_t]),
        "anchor": (C.c_int, [C.c_void_p, C.c_int, C.c_uint64, C.POINTER(Endpoint)]),
        "push": (C.c_int, [C.POINTER(Endpoint), C.c_void_p, C.c_size_t,
                           C.c_uint64, C.c_uint64, C.c_long]),
        "pop": (C.c_int, [C.POINTER(Endpoint), C.c_void_p, C.c_size_t,
                          C.POINTER(Record), C.c_long]),
        "close": (C.c_int, [C.c_void_p]),
        "size": (C.c_int, [C.c_void_p, C.POINTER(C.c_size_t)]),
        "destroy": (C.c_int, [C.c_void_p]),
    }
    for name, (result, arguments) in signatures.items():
        fn = getattr(lib, "turing_byte_queue_" + name)
        fn.restype, fn.argtypes = result, arguments
        setattr(lib, name, fn)
    return lib


# All worker activity below is compiled C using the existing native task ABI.
# The parent closes after producers complete; consumers drain before joining.
CONCURRENT_PROBE = r'''
typedef struct { uint64_t producer, sequence, check; } byte_probe_message;
typedef struct {
    turing_byte_queue* queue;
    turing_dispatch_event* start;
    int producers, consumers, items, push_end, pop_end;
} byte_probe_run;
typedef struct {
    byte_probe_run* run;
    int id, producer, result;
    unsigned int* seen;
} byte_probe_worker;

static void byte_probe_entry(void* raw) {
    byte_probe_worker* worker = raw;
    byte_probe_run* run = worker->run;
    turing_byte_endpoint endpoint;
    byte_probe_message message;
    turing_byte_record record;
    int index, status, next = 0;
    worker->result = -1;
    if (turing_dispatch_event_wait(run->start, 5000) != 1) return;
    if (turing_byte_queue_anchor(run->queue,
            worker->producer
                ? (run->push_end < 0 ? worker->id % 2 : run->push_end)
                : (run->pop_end < 0 ? worker->id % 2 : run->pop_end),
            (uint64_t)(worker->producer ? 0x10000 : 0x90000) + worker->id,
            &endpoint) != 1) return;
    if (worker->producer) {
        for (index = 0; index < run->items; index++) {
            message.producer = (uint64_t)worker->id;
            message.sequence = (uint64_t)index;
            message.check = message.producer * UINT64_C(104729) + message.sequence;
            status = turing_byte_queue_push(&endpoint, &message, sizeof(message),
                    (uint64_t)worker->id + 100, 0xabc, 2000);
            if (status != 1) { worker->result = -2; return; }
        }
    } else {
        for (;;) {
            status = turing_byte_queue_pop(&endpoint, &message, sizeof(message), &record, 2000);
            if (status == TURING_QUEUE_CLOSED) break;
            if (status != 1 || record.length != sizeof(message) ||
                message.producer >= (uint64_t)run->producers ||
                message.sequence >= (uint64_t)run->items ||
                message.check != message.producer * UINT64_C(104729) + message.sequence ||
                record.type_tag != message.producer + 100 || record.operation_contract != 0xabc ||
                record.producer_worker != message.producer + 0x10000) {
                worker->result = -3; return;
            }
            if (run->producers == 1 && run->consumers == 1 &&
                run->push_end != run->pop_end && message.sequence != (uint64_t)next++) {
                worker->result = -4; return;
            }
            worker->seen[message.producer * (uint64_t)run->items + message.sequence]++;
        }
    }
    worker->result = 0;
}

int byte_probe_exchange(int producers, int consumers, int items, int slots, int push_end, int pop_end) {
    byte_probe_run* run;
    byte_probe_worker* workers;
    turing_dispatch_task** tasks;
    int count, index, item, status = 0;
    if (producers < 1 || producers > 8 || consumers < 1 || consumers > 8 ||
        items < 1 || items > 4096 || slots < 1) return -10;
    count = producers + consumers;
    run = calloc(1, sizeof(*run));
    workers = calloc((size_t)count, sizeof(*workers));
    tasks = calloc((size_t)count, sizeof(*tasks));
    if (!run || !workers || !tasks) { free(run); free(workers); free(tasks); return -11; }
    run->queue = turing_byte_queue_create((size_t)slots, sizeof(byte_probe_message));
    run->start = turing_dispatch_event_create();
    run->producers = producers; run->consumers = consumers; run->items = items;
    run->push_end = push_end; run->pop_end = pop_end;
    if (!run->queue || !run->start) { status = -12; goto finish; }
    for (index = 0; index < count; index++) {
        workers[index].run = run;
        workers[index].producer = index < producers;
        workers[index].id = index < producers ? index : index - producers;
        workers[index].result = -99;
        if (index >= producers) {
            workers[index].seen = calloc((size_t)producers * (size_t)items, sizeof(unsigned int));
            if (!workers[index].seen) { status = -13; break; }
        }
        tasks[index] = turing_task_start(byte_probe_entry, &workers[index]);
        if (!tasks[index]) { status = -14; break; }
    }
    if (status) turing_byte_queue_close(run->queue);
    turing_dispatch_event_set(run->start);
    for (index = 0; index < producers; index++) {
        if (!tasks[index]) continue;
        if (turing_task_wait(tasks[index], 5000) != 1) {
            turing_byte_queue_close(run->queue);
            return -90; // keep heap state alive; isolated probe process will exit
        }
        if (workers[index].result) status = workers[index].result;
    }
    turing_byte_queue_close(run->queue);
    for (index = producers; index < count; index++) {
        if (!tasks[index]) continue;
        if (turing_task_wait(tasks[index], 5000) != 1) return -91;
        if (workers[index].result) status = workers[index].result;
    }
    if (!status) for (item = 0; item < producers * items; item++) {
        unsigned int observed = 0;
        for (index = producers; index < count; index++) observed += workers[index].seen[item];
        if (observed != 1) status = -15;
    }
finish:
    for (index = 0; index < count; index++) {
        if (tasks[index]) turing_task_destroy(tasks[index]);
        free(workers[index].seen);
    }
    if (run->queue) turing_byte_queue_destroy(run->queue);
    if (run->start) turing_dispatch_event_destroy(run->start);
    free(tasks); free(workers); free(run);
    return status;
}

// A second worker blocks on a full/empty queue. The waiter snapshot is read
// under the same monitor, proving registration before the parent closes it.
typedef struct {
    turing_byte_endpoint endpoint;
    turing_dispatch_event* ready;
    int push, result;
} byte_close_probe;
static void byte_close_entry(void* raw) {
    byte_close_probe* probe = raw;
    unsigned char value = 7;
    turing_byte_record record;
    turing_dispatch_event_set(probe->ready);
    probe->result = probe->push
        ? turing_byte_queue_push(&probe->endpoint, &value, 1, 0, 0, -1)
        : turing_byte_queue_pop(&probe->endpoint, &value, 1, &record, -1);
}
int byte_probe_close_waiter(int push) {
    byte_close_probe* probe = calloc(1, sizeof(*probe));
    turing_byte_queue* queue = turing_byte_queue_create(1, 1);
    turing_dispatch_task* task;
    unsigned char value = 1;
    uint64_t deadline, now;
    size_t readers, writers;
    int result;
    if (!probe || !queue) { free(probe); if(queue) turing_byte_queue_destroy(queue); return -20; }
    probe->ready = turing_dispatch_event_create(); probe->push = push;
    if (!probe->ready) { free(probe); turing_byte_queue_destroy(queue); return -21; }
    turing_byte_queue_anchor(queue, TURING_QUEUE_LEFT, UINT64_MAX, &probe->endpoint);
    if (push) turing_byte_queue_push(&probe->endpoint, &value, 1, 0, 0, 0);
    task = turing_task_start(byte_close_entry, probe);
    if (!task) { turing_dispatch_event_destroy(probe->ready); free(probe); turing_byte_queue_destroy(queue); return -22; }
    if (turing_dispatch_event_wait(probe->ready, 5000) != 1) {
        turing_byte_queue_close(queue); return -23;
    }
    if (byte_queue_now(&now)) { turing_byte_queue_close(queue); return -25; }
    deadline = now + 5000;
    for (;;) {
        if (turing_byte_queue_waiters(queue, &readers, &writers) != 1) return -26;
        if ((push ? writers : readers) == 1) break;
        if (byte_queue_now(&now) || now >= deadline) {
            turing_byte_queue_close(queue); return -27;
        }
    }
    turing_byte_queue_close(queue);
    if (turing_task_wait(task, 5000) != 1) return -24;
    result = probe->result;
    turing_task_destroy(task); turing_dispatch_event_destroy(probe->ready);
    turing_byte_queue_destroy(queue); free(probe);
    return result;
}
'''


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    if preferred_toolchain() is None:
        pytest.skip("no native C toolchain available")
    pool_header = (HOME / "turing_pool.h").read_text()
    queue_header = (HOME / "turing_byte_queue.h").read_text()
    pool = (HOME / "turing_pool.c").read_text().replace('#include "turing_pool.h"', pool_header)
    queue = (HOME / "turing_byte_queue.c").read_text().replace('#include "turing_pool.h"', pool_header)
    queue = queue.replace('#include "turing_byte_queue.h"', queue_header)
    path, _ = compile_shared_library(
        pool + "\n" + queue + CONCURRENT_PROBE,
        name="byte_queue_test", directory=tmp_path_factory.mktemp("byte_queue"),
        extra_flags=() if sys.platform == "win32" else ("-pthread",),
    )
    return path, bind(path)


@pytest.fixture
def queue(native):
    _, lib = native
    q = lib.create(3, 32)
    assert q
    yield lib, q
    assert lib.destroy(q) == 1


def endpoint(lib, q, end, worker):
    result = Endpoint()
    assert lib.anchor(q, end, worker, C.byref(result)) == 1
    return result


def push(lib, endpoint, data, tag=0, contract=0, timeout=0):
    buffer = C.create_string_buffer(data)
    return lib.push(C.byref(endpoint), buffer, len(data), tag, contract, timeout)


def pop(lib, endpoint, capacity=32, timeout=0):
    buffer, record = C.create_string_buffer(capacity), Record()
    status = lib.pop(C.byref(endpoint), buffer, capacity, C.byref(record), timeout)
    return status, bytes(buffer.raw[:record.length]), record


@pytest.mark.parametrize("push_end,pop_end", [(1, 0), (0, 1), (0, 0), (1, 1)])
def test_both_end_ordering_and_worker_anchors(queue, push_end, pop_end):
    lib, q = queue
    producer, consumer = endpoint(lib, q, push_end, 2**60 + 7), endpoint(lib, q, pop_end, 999)
    expected = deque()
    for value in (b"a\x00b", b"", bytes(range(32))):
        assert push(lib, producer, value, 0x17, 0xBEEF) == 1
        (expected.appendleft if push_end == 0 else expected.append)(value)
    while expected:
        status, payload, record = pop(lib, consumer)
        assert status == 1
        assert payload == (expected.popleft() if pop_end == 0 else expected.pop())
        assert (record.type_tag, record.operation_contract, record.producer_worker) == (0x17, 0xBEEF, 2**60 + 7)
    assert pop(lib, consumer)[0] == 0


def test_wraparound_mixed_end_operations_copy_and_aliases(queue):
    lib, q = queue
    left, right = endpoint(lib, q, 0, 0), endpoint(lib, q, 1, 2**64 - 1)
    alias = Endpoint(left.queue, 12345, left.end)
    expected = deque()
    for index in range(120):
        side = left if index % 3 else right
        payload = bytes([index % 256, 0, 255])
        assert push(lib, side, payload, index, 77) == 1
        (expected.appendleft if side.end == 0 else expected.append)(payload)
        if len(expected) >= 2:
            consumer = alias if index % 2 else right
            status, data, record = pop(lib, consumer)
            assert status == 1
            assert data == (expected.popleft() if consumer.end == 0 else expected.pop())
        count = C.c_size_t()
        assert lib.size(q, C.byref(count)) == 1 and count.value == len(expected)
    while expected:
        assert pop(lib, alias)[1] == expected.popleft()


def test_capacity_and_buffer_errors_do_not_consume_or_publish(queue):
    lib, q = queue
    out, inp = endpoint(lib, q, 1, 8), endpoint(lib, q, 0, 9)
    assert push(lib, out, b"x" * 33) == -4
    for data in (b"abcdef", b"second", b"third"):
        assert push(lib, out, data, 4, 55) == 1
    assert push(lib, out, b"fourth") == 0
    status, _, record = pop(lib, inp, 2)
    assert status == -3 and record.length == 6
    assert (record.type_tag, record.operation_contract) == (4, 55)
    assert pop(lib, inp)[1] == b"abcdef"
    assert push(lib, out, b"fourth") == 1
    assert [pop(lib, inp)[1] for _ in range(3)] == [b"second", b"third", b"fourth"]


def test_source_copy_and_destination_copy_do_not_share_queue_storage(queue):
    lib, q = queue
    out, inp = endpoint(lib, q, 1, 1), endpoint(lib, q, 0, 2)
    payload = C.create_string_buffer(b"old")
    assert lib.push(C.byref(out), payload, 3, 0, 0, 0) == 1
    payload[0] = b"X"
    destination, record = C.create_string_buffer(3), Record()
    assert lib.pop(C.byref(inp), destination, 3, C.byref(record), 0) == 1
    assert destination.raw == b"old"
    for _ in range(6):
        assert push(lib, out, b"new") == 1
        assert pop(lib, inp)[1] == b"new"
    assert destination.raw == b"old"


def test_empty_result_preserves_caller_buffers_and_metadata(queue):
    lib, q = queue
    end = endpoint(lib, q, 0, 17)
    destination, record = C.create_string_buffer(b"keep"), Record(12, 13, 14, 15)
    assert lib.pop(C.byref(end), destination, 4, C.byref(record), 0) == 0
    assert destination.value == b"keep"
    assert (record.length, record.type_tag, record.operation_contract, record.producer_worker) == (12, 13, 14, 15)


def test_close_drains_and_refuses_new_publications(queue):
    lib, q = queue
    out, inp = endpoint(lib, q, 1, 1), endpoint(lib, q, 0, 2)
    assert push(lib, out, b"before") == 1
    assert lib.close(q) == lib.close(q) == 1
    assert push(lib, out, b"after") == -2
    assert pop(lib, inp)[1] == b"before"
    assert pop(lib, inp)[0] == -2


def test_timeout_and_invalid_operations_leave_state_unchanged(queue):
    lib, q = queue
    end = endpoint(lib, q, 0, 7)
    assert pop(lib, end, timeout=5)[0] == 0
    assert push(lib, end, b"a", timeout=-2) == -1
    assert lib.anchor(q, 2, 0, C.byref(end)) == -1
    assert lib.push(C.byref(end), None, 1, 0, 0, 0) == -1
    assert lib.create(0, 4) is None
    assert lib.create(C.c_size_t(-1).value, 4) is None
    for _ in range(3):
        assert push(lib, end, b"a") == 1
    assert push(lib, end, b"b", timeout=5) == 0
    assert [pop(lib, end)[1] for _ in range(3)] == [b"a"] * 3


def test_zero_byte_slots_are_real_records(native):
    _, lib = native
    q = lib.create(1, 0)
    assert q
    try:
        end = endpoint(lib, q, 0, 17)
        assert lib.push(C.byref(end), None, 0, 5, 6, 0) == 1
        record = Record()
        assert lib.pop(C.byref(end), None, 0, C.byref(record), 0) == 1
        assert (record.length, record.type_tag, record.operation_contract) == (0, 5, 6)
        assert lib.pop(C.byref(end), None, 0, C.byref(record), 0) == 0
    finally:
        assert lib.destroy(q) == 1


@pytest.mark.parametrize("producers,consumers,slots,push_end,pop_end", [
    (1, 1, 1, 1, 0), (1, 1, 3, 0, 1),
    (4, 1, 7, 1, 0), (1, 4, 7, 1, 0),
    (4, 3, 7, 1, 0), (4, 3, 2, 0, 1),
    (3, 3, 3, 0, 0), (3, 3, 3, 1, 1),
    (4, 4, 3, -1, -1),
])
def test_native_tasks_deliver_each_record_exactly_once(native, producers, consumers, slots, push_end, pop_end):
    path, _ = native
    result = subprocess.run([sys.executable, "-c", """
import ctypes as C, sys
lib=C.CDLL(sys.argv[1]); fn=lib.byte_probe_exchange
fn.argtypes=[C.c_int]*6; fn.restype=C.c_int
result=fn(*map(int,sys.argv[2:])); print('native exchange:',result,flush=True)
assert result == 0, result
""", str(path), str(producers), str(consumers), "600", str(slots), str(push_end), str(pop_end)],
        capture_output=True, text=True, timeout=25)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("push", [0, 1])
def test_native_close_wakes_a_waiting_consumer_or_producer(native, push):
    path, _ = native
    result = subprocess.run([sys.executable, "-c", """
import ctypes as C, sys
lib=C.CDLL(sys.argv[1]); fn=lib.byte_probe_close_waiter
fn.argtypes=[C.c_int]; fn.restype=C.c_int
result=fn(int(sys.argv[2])); print('close waiter:',result,flush=True)
assert result == -2, result
""", str(path), str(push)], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
