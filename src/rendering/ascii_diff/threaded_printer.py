"""Threaded ASCII diff printer using DoubleBuffer and fast console output."""

from __future__ import annotations

import threading
import queue
import logging
import os

from src.common.fast_console import cffiPrinter
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
    from the rendering thread. Output is sent to a fast console printer.

    ``queue_maxsize`` controls the underlying queue capacity.  When ``0`` the
    queue is unbounded.  ``block_on_full`` selects whether producers block or
    drop frames when the queue is full.
    """

    def __init__(
        self,
        buffer_size: int = 2,
        *,
        queue_maxsize: int = 0,
        block_on_full: bool = True,
    ) -> None:
        self._db = DoubleBuffer(roll_length=buffer_size, num_agents=2)
        self._queue: "queue.Queue[str | None]" = queue.Queue(maxsize=queue_maxsize)
        self._block_on_full = block_on_full
        self._stop = threading.Event()
        self._printer = cffiPrinter(threaded=True)
        self._thread = threading.Thread(target=self._render_loop, daemon=True)
        self._thread.start()

    def enqueue(self, frame: str) -> None:
        """Queue ``frame`` for printing, obeying the saturation policy."""
        try:
            self._queue.put(frame, block=self._block_on_full)
        except queue.Full:
            logger.debug("Dropping frame due to full queue")

    def wait_until_empty(self) -> None:
        """Block until all queued frames have been processed."""
        self._queue.join()

    def _render_loop(self) -> None:
        agent_writer, agent_reader = 0, 1
        logger.debug("Starting render loop")
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.1)
                logger.debug("Retrieved frame from queue")
            except queue.Empty:
                import time
                time.sleep(0.01)
                continue
            if item is None:
                logger.debug("Stop sentinel received")
                self._queue.task_done()
                break
            self._db.write_frame(item, agent_idx=agent_writer)
            frame = self._db.read_frame(agent_idx=agent_reader)
            if frame is not None:
                logger.debug("Sending frame to cffiPrinter: %d chars", len(frame))
                self._printer.print(frame)
            self._queue.task_done()
        self._printer.flush()
        
    def stop(self) -> None:
        """Signal the rendering thread to terminate and wait for it."""
        self._stop.set()
        # Ensure the sentinel is enqueued even if the queue is full.
        self._queue.put(None, block=True)
        self._thread.join()
        self._printer.stop()
