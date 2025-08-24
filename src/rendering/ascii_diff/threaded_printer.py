from __future__ import annotations

"""Threaded ASCII diff printer using DoubleBuffer and fast console output."""

import threading
import queue
import logging
import os
from typing import Optional
import logging

logger = logging.getLogger(__name__)

try:  # Windows-only fast console; fallback to normal print if unavailable
    from src.common.fast_console import cffiPrinter
except Exception as e:  # pragma: no cover - non-Windows or missing deps
    logger.warning("fast console unavailable; fallback mode active: %s", e)
    cffiPrinter = None  # type: ignore

from src.common.double_buffer import DoubleBuffer


logger = logging.getLogger(__name__)
if os.getenv("TURING_DEBUG"):
    if not logger.handlers:
        _h = logging.StreamHandler()
        _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        logger.addHandler(_h)
    logger.setLevel(logging.DEBUG)
else:
    logger.addHandler(logging.NullHandler())


class ThreadedAsciiDiffPrinter:
    """Background renderer consuming ASCII frames from a queue.

    Frames are written through a :class:`DoubleBuffer` to decouple producers
    from the rendering thread.  Output is sent to a fast console printer when
    available, otherwise ``print`` is used.
    """

    def __init__(self, buffer_size: int = 2) -> None:
        self._db = DoubleBuffer(roll_length=buffer_size, num_agents=2)
        self._queue: "queue.Queue[str | None]" = queue.Queue()
        self._stop = threading.Event()
        self._printer: Optional[cffiPrinter] = None
        if cffiPrinter is not None:
            # Use internal threading for console writes
            self._printer = cffiPrinter(threaded=True)
        self._thread = threading.Thread(target=self._render_loop, daemon=True)
        self._thread.start()

    def get_queue(self) -> "queue.Queue[str | None]":
        """Return the input queue for ASCII frames."""
        return self._queue

    def _render_loop(self) -> None:
        agent_writer, agent_reader = 0, 1
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.1)
                logger.debug("Retrieved frame from queue")
            except queue.Empty:
                continue
            if item is None:
                logger.debug("Stop sentinel received")
                self._queue.task_done()
                break
            self._db.write_frame(item, agent_idx=agent_writer)
            frame = self._db.read_frame(agent_idx=agent_reader)
            if frame is not None:
                if self._printer is not None:
                    logger.debug("Sending frame to cffiPrinter: %d chars", len(frame))
                    self._printer.print(frame)
                else:  # pragma: no cover - fallback path
                    logger.debug("Fallback printing path used")
                    print(frame, end="", flush=True)
            self._queue.task_done()
        if self._printer is not None:
            self._printer.flush()

    def stop(self) -> None:
        """Signal the rendering thread to terminate and wait for it."""
        self._stop.set()
        self._queue.put(None)
        self._thread.join()
        if self._printer is not None:
            self._printer.stop()
