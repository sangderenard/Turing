#!/usr/bin/env python3
"""Backend-agnostic buffer coordinator for accelerator operations."""
from __future__ import annotations

try:
    from dataclasses import dataclass
    from typing import Any, Tuple
    from queue import Queue
    from concurrent.futures import Future
    import threading

    from .c_backend import CTensorOperations
    from .opengl_backend import OpenGLTensorOperations
except Exception:
    import sys
    print("Accelerator coordinator imports failed")
    sys.exit(1)
# --- END HEADER ---


@dataclass
class _BufferPair:
    """Internal double-buffer record."""

    front: Any
    back: Any
    queue: "Queue[Any]"
    event: threading.Event
    thread: threading.Thread

SYNC_TOKEN = object()
STOP_TOKEN = object()


# ########## STUB: Accelerator Buffer Coordinator ##########
# PURPOSE: Provide a unified interface that manages instruction chains and
#          contiguous buffers for both the C and OpenGL backends. A double
#          buffering scheme exposes a "front" buffer to Python while a "back"
#          buffer is mutated by the selected accelerator.
# EXPECTED BEHAVIOR: When implemented, this class will enqueue operations and
#          dispatch them to either a thread-saturated C backend or a compute
#          shader batch on the OpenGL backend. It abstracts buffer management so
#          client code remains agnostic to the underlying accelerator.
# INPUTS: Instruction objects describing tensor transformations and a selected
#         backend implementation.
# OUTPUTS: Finalized contiguous buffers ready for consumption by the caller.
# KEY ASSUMPTIONS/DEPENDENCIES: Requires either ``CTensorOperations`` or
#         ``OpenGLTensorOperations`` to be available. Assumes instruction chains
#         can be executed sequentially on a worker thread or GPU queue.
# TODO:
#   - Implement an instruction representation for tensor operations.
#   - Add worker thread or GPU dispatch logic.
#   - Provide synchronization primitives to retrieve completed buffers.
# NOTES: This is an architectural scaffold to pursue the conceptual flag
#         unifying C and OpenGL execution paths.
# ###########################################################################
class AcceleratorCoordinator:
    """Manage asynchronous accelerator buffers using double buffering."""

    def __init__(self, backend: str = "c") -> None:
        if backend == "c":
            self.ops = CTensorOperations()
        elif backend == "opengl":
            self.ops = OpenGLTensorOperations()
        else:
            raise ValueError(f"Unknown backend: {backend}")
        self._objects: dict[str, _BufferPair] = {}

    # ------------------------------------------------------------------
    # Context manager helpers
    # ------------------------------------------------------------------
    def __enter__(self) -> "AcceleratorCoordinator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        """Terminate all managed tensors and worker threads."""
        for name in list(self._objects):
            self.terminate(name)

    def terminate_all(self) -> None:
        """Alias for :meth:`close` for API completeness."""
        self.close()

    def create_tensor(self, name: str, data: Any) -> None:
        """Create a double-buffered tensor object."""
        if isinstance(data, self.ops.tensor_type_):
            tensor = data
        else:
            tensor = self.ops.tensor_from_list_(data, self.ops.float_dtype_, None)
        front = self.ops.clone_(tensor)
        back = self.ops.clone_(tensor)
        q: Queue[Any] = Queue()
        ev = threading.Event()
        t = threading.Thread(target=self._worker_loop, args=(name,), daemon=True)
        self._objects[name] = _BufferPair(front, back, q, ev, t)
        t.start()

    def enqueue(
        self,
        name: str,
        op: str,
        *args: Any,
        return_future: bool = False,
        **kwargs: Any,
    ) -> Future | None:
        """Queue an operation by name and optionally return a ``Future``."""
        pair = self._objects[name]
        fut: Future | None = Future() if return_future else None
        pair.queue.put((op, args, kwargs, fut))
        return fut

    def synchronize(self, name: str) -> None:
        """Block until queued work is applied and buffers swapped."""
        pair = self._objects[name]
        pair.queue.put((SYNC_TOKEN, None))
        pair.event.wait()
        pair.event.clear()

    def synchronize_async(self, name: str) -> Future:
        """Return a future resolved when pending work completes."""
        pair = self._objects[name]
        fut: Future = Future()
        pair.queue.put((SYNC_TOKEN, fut))
        return fut

    def get_tensor(self, name: str) -> Any:
        """Return the accessible front buffer, synchronizing if needed."""
        pair = self._objects[name]
        if not pair.queue.empty():
            self.synchronize(name)
        return self.ops.clone_(pair.front)

    def terminate(self, name: str) -> None:
        """Stop the worker thread for ``name`` and drop its buffers."""
        pair = self._objects.pop(name)
        pair.queue.put(STOP_TOKEN)
        pair.thread.join()

    def _worker_loop(self, name: str) -> None:
        pair = self._objects[name]
        while True:
            item = pair.queue.get()
            if item is STOP_TOKEN:
                break
            fut: Future | None = None
            if item is SYNC_TOKEN or (
                isinstance(item, tuple) and item[0] is SYNC_TOKEN
            ):
                if isinstance(item, tuple):
                    fut = item[1]
                pair.front, pair.back = pair.back, pair.front
                pair.event.set()
                if fut is not None:
                    fut.set_result(None)
                continue
            if isinstance(item, tuple) and len(item) == 4:
                op, args, kwargs, fut = item
            else:
                op, args, kwargs = item  # type: ignore[misc]
            func = getattr(self.ops, op)
            result = func(pair.back, *args, **kwargs)
            if isinstance(result, self.ops.tensor_type_):
                pair.back = result
            if fut is not None:
                fut.set_result(result)
