"""Authored Python workload for source → SSA → standalone C benchmarking.

The byte/type/contract tuples are application records. The queue never decodes
payloads. This module is benchmark input, not an alternative Python executor.
"""
from collections import deque
import threading


class Channel:
    def __init__(self, capacity: int, end: int):
        self.records = deque()
        self.capacity = capacity
        self.end = end
        self.closed = False
        self.changed = threading.Condition()

    def put(self, record):
        with self.changed:
            while len(self.records) == self.capacity and not self.closed:
                self.changed.wait()
            if self.closed:
                raise RuntimeError("closed queue")
            payload, tag, contract, worker = record
            copied = (memoryview(payload).tobytes(), tag, contract, worker)
            if self.end == 0:
                self.records.appendleft(copied)
            else:
                self.records.append(copied)
            self.changed.notify_all()

    def get(self):
        with self.changed:
            while not self.records and not self.closed:
                self.changed.wait()
            if not self.records:
                raise RuntimeError("unexpected drained queue")
            record = self.records.pop() if self.end == 0 else self.records.popleft()
            payload, tag, contract, worker = record
            copied = (memoryview(payload).tobytes(), tag, contract, worker)
            self.changed.notify_all()
            return copied

    def close(self):
        with self.changed:
            self.closed = True
            self.changed.notify_all()


def root(lanes: int, hops: int, items: int, capacity: int) -> int:
    if lanes < 1 or hops < 0 or items < 1 or capacity < 1:
        raise ValueError("positive lanes/items/capacity and nonnegative hops required")
    # int32, int64, binary64, UTF-8 text, arbitrary binary: encoded bytes plus
    # convenience tags. Both implementations must execute this same workload.
    payloads = [b"\x01\x00\x00\x00", b"\xff\x00\x01\x02\x03\x04\x05\x06",
                b"\x00\x00\x00\x00\x00\x00\xf8\x3f", b"queue\xc3\xa9",
                b"\x00\xff\x7f\x80\x01\x02\x03"]
    channels = []
    i = 0
    while i < lanes * (hops + 1):
        channels.append(Channel(capacity, i % 2))
        i += 1
    results = [0] * lanes
    gate = threading.Event()

    def worker(lane: int, stage: int):
        gate.wait()
        item = 0
        checksum = 0
        while item < items:
            index = item % 5
            if stage == 0:
                payload = payloads[index]
                tag = index + 1
                contract = 4096 + tag + item * 16
            else:
                payload, tag, contract, publisher = channels[lane * (hops + 1) + stage - 1].get()
                if payload != payloads[index] or tag != index + 1 or contract != 4096 + tag + item * 16:
                    raise RuntimeError("payload, type, contract or order mismatch")
                if publisher != 1000 + lane * (hops + 2) + stage - 1:
                    raise RuntimeError("publisher context mismatch")
            if stage <= hops:
                channels[lane * (hops + 1) + stage].put(
                    (payload, tag, contract, 1000 + lane * (hops + 2) + stage))
            else:
                checksum += len(payload) * tag + payload[0]
            item += 1
        if stage <= hops:
            channels[lane * (hops + 1) + stage].close()
        else:
            results[lane] = checksum

    threads = []
    lane = 0
    while lane < lanes:
        stage = 0
        while stage < hops + 2:
            threads.append(threading.Thread(target=worker, args=(lane, stage)))
            stage += 1
        lane += 1
    for task in threads:
        task.start()
    gate.set()
    for task in threads:
        task.join()
    total = 0
    for value in results:
        total += value
    return total
