import time
import threading
from collections import deque
import numpy as np
import torch
from .base import verbose_log
from .lock import LockCommand

class AsyncGPUSyncWorker(threading.Thread):
    """
    Handles async, prioritized sync from pinned-CPU tensors to CUDA tensors.
    Uses lock commands via the manager to respect the lock graph.
    Now does non-blocking, region-wise sync with binary search fallback.
    """
    def __init__(self, buffer_sync, manager, poll_interval=0.001, kernel_size=16, stride=8):
        verbose_log(f"AsyncGPUSyncWorker.__init__()")
        super().__init__(daemon=True)
        self.buffer_sync = buffer_sync
        self.manager = manager
        self.queue = deque()       # fresh dirty keys
        self.lockwait_queue = deque()
        self.shutdown_flag = threading.Event()
        self.poll_interval = poll_interval
        self.kernel_size = kernel_size
        self.stride = stride

    def enqueue(self, key):
        verbose_log(f"AsyncGPUSyncWorker.enqueue(key={key})")
        self.queue.append(key)

    def run(self):
        verbose_log("AsyncGPUSyncWorker.run() started")
        while not self.shutdown_flag.is_set():
            if self.queue:
                key = self.queue.popleft()
            elif self.lockwait_queue:
                key = self.lockwait_queue.popleft()
            else:
                time.sleep(self.poll_interval)
                continue

            # Assume key is the buffer key (e.g. "positions"), get shape info
            arr = self.buffer_sync.host.get(key)
            if arr is None:
                continue
            N = arr.shape[0]
            page = 0  # If you have pages, set this accordingly

            # Start with full range, then binary search down
            stack = [(0, N)]
            while stack:
                idx_start, idx_end = stack.pop()
                if idx_end <= idx_start:
                    continue
                # Find all kernel regions covering this range
                regions = self.manager.lock_graph.find_covering_regions(
                    region_prefix="buf", page=page, idx_start=idx_start, idx_end=idx_end,
                    kernel_size=self.kernel_size, stride=self.stride
                )
                did_any = False
                for region in regions:
                    cmd = LockCommand('acquire', region, blocking=False)
                    self.manager.submit(cmd)
                    cmd.reply_event.wait()
                    if cmd.error:
                        # Could not acquire, subdivide further if possible
                        if idx_end - idx_start > 1:
                            mid = (idx_start + idx_end) // 2
                            stack.append((idx_start, mid))
                            stack.append((mid, idx_end))
                        continue
                    did_any = True
                    token = cmd.result
                    try:
                        # Only sync if dirty
                        if key in self.buffer_sync.host_dirty:
                            verbose_log(f"AsyncGPUSyncWorker: host->cpu sync for key {key} [{idx_start}:{idx_end}]")
                            arr = self.buffer_sync.host[key]
                            cpu_t = torch.from_numpy(arr[idx_start:idx_end]).pin_memory()
                            self.buffer_sync.cpu[key][idx_start:idx_end] = cpu_t
                            # Don't clear dirty flag unless all regions are done
                        cpu_t = self.buffer_sync.cpu.get(key)
                        if cpu_t is not None:
                            verbose_log(f"AsyncGPUSyncWorker: cpu->gpu sync for key {key} [{idx_start}:{idx_end}]")
                            if key not in self.buffer_sync.gpu or self.buffer_sync.gpu[key].device.type != 'cuda':
                                self.buffer_sync.gpu[key] = cpu_t.to('cuda', non_blocking=True)
                            else:
                                self.buffer_sync.gpu[key][idx_start:idx_end].copy_(cpu_t[idx_start:idx_end], non_blocking=True)
                    finally:
                        rel = LockCommand('release', region)
                        rel.token = token
                        self.manager.submit(rel)
                        rel.reply_event.wait()
                # If we did any region, mark as not dirty for this range
                if did_any and key in self.buffer_sync.host_dirty:
                    # Optionally, track which regions are clean
                    pass
            # After all, if all regions are clean, clear dirty flag
            # (You may want to track per-region dirty state for true correctness)

    def stop(self):
        verbose_log("AsyncGPUSyncWorker.stop()")
        self.shutdown_flag.set()


class AsyncCPUSyncWorker(threading.Thread):
    """
    Handles async sync between NumPy host and pinned-CPU torch tensors.
    Now does non-blocking, region-wise sync with binary search fallback.
    """
    def __init__(self, buffer_sync, manager, poll_interval=0.001, kernel_size=16, stride=8):
        verbose_log(f"AsyncCPUSyncWorker.__init__()")
        super().__init__(daemon=True)
        self.buffer_sync = buffer_sync
        self.manager = manager
        self.queue = deque()
        self.lockwait_queue = deque()
        self.shutdown_flag = threading.Event()
        self.poll_interval = poll_interval
        self.kernel_size = kernel_size
        self.stride = stride

    def enqueue(self, key):
        verbose_log(f"AsyncCPUSyncWorker.enqueue(key={key})")
        self.queue.append(key)

    def run(self):
        verbose_log("AsyncCPUSyncWorker.run() started")
        while not self.shutdown_flag.is_set():
            if self.queue:
                key = self.queue.popleft()
            elif self.lockwait_queue:
                key = self.lockwait_queue.popleft()
            else:
                time.sleep(self.poll_interval)
                continue

            arr = self.buffer_sync.host.get(key)
            if arr is None:
                continue
            N = arr.shape[0]
            page = 0  # If you have pages, set this accordingly

            stack = [(0, N)]
            while stack:
                idx_start, idx_end = stack.pop()
                if idx_end <= idx_start:
                    continue
                regions = self.manager.lock_graph.find_covering_regions(
                    region_prefix="buf", page=page, idx_start=idx_start, idx_end=idx_end,
                    kernel_size=self.kernel_size, stride=self.stride
                )
                did_any = False
                for region in regions:
                    cmd = LockCommand('acquire', region, blocking=False)
                    self.manager.submit(cmd)
                    cmd.reply_event.wait()
                    if cmd.error:
                        if idx_end - idx_start > 1:
                            mid = (idx_start + idx_end) // 2
                            stack.append((idx_start, mid))
                            stack.append((mid, idx_end))
                        continue
                    did_any = True
                    token = cmd.result
                    try:
                        if key in self.buffer_sync.host_dirty:
                            verbose_log(f"AsyncCPUSyncWorker: host->cpu sync for key {key} [{idx_start}:{idx_end}]")
                            arr = self.buffer_sync.host[key]
                            self.buffer_sync.cpu[key][idx_start:idx_end] = torch.from_numpy(arr[idx_start:idx_end]).pin_memory()
                        if key in self.buffer_sync.cpu_dirty:
                            verbose_log(f"AsyncCPUSyncWorker: cpu->host sync for key {key} [{idx_start}:{idx_end}]")
                            cpu_t = self.buffer_sync.cpu[key]
                            self.buffer_sync.host[key][idx_start:idx_end] = cpu_t[idx_start:idx_end].cpu().numpy()
                    finally:
                        rel = LockCommand('release', region)
                        rel.token = token
                        self.manager.submit(rel)
                        rel.reply_event.wait()
                if did_any and key in self.buffer_sync.host_dirty:
                    pass

    def stop(self):
        verbose_log("AsyncCPUSyncWorker.stop()")
        self.shutdown_flag.set()
