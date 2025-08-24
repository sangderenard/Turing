
"""Fast console output using CFFI with per-platform back-ends.

On Windows the :mod:`kernel32` ``WriteConsole`` APIs are used while POSIX
platforms emit bytes directly via the C library ``write`` call.  A small
threaded queue is available to offload formatting work when needed.
"""

import cffi
import threading
import queue
import logging
import os
from typing import Optional


logger = logging.getLogger(__name__)
if os.getenv("TURING_DEBUG"):
    if not logger.handlers:
        _h = logging.StreamHandler()
        _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        logger.addHandler(_h)
    logger.setLevel(logging.DEBUG)
else:
    logger.addHandler(logging.NullHandler())


class cffiPrinter:
    def __init__(self, mode: str = "A", threaded: bool = False, queue_size: int = 128):
        """Create a new fast printer instance.

        Parameters
        ----------
        mode:
            Only relevant on Windows.  ``"A"`` routes to ``WriteConsoleA`` while
            ``"W"`` uses the wide-character variant.  POSIX platforms always emit
            UTF-8 bytes through ``write``.
        threaded:
            If ``True`` a background thread pulls strings from a queue to avoid
            blocking the caller.
        queue_size:
            Maximum number of queued print jobs when ``threaded`` is ``True``.
        """

        self.ffi = cffi.FFI()
        self.mode = mode.upper()
        self._is_windows = os.name == "nt"

        if self._is_windows:
            if self.mode == "A":
                self.ffi.cdef(
                    """
                    int __stdcall WriteConsoleA(void* hConsoleOutput, const char* lpBuffer,
                                               unsigned long nNumberOfCharsToWrite,
                                               unsigned long* lpNumberOfCharsWritten,
                                               void* lpReserved);
                    void* __stdcall GetStdHandle(int nStdHandle);
                    """
                )
            elif self.mode == "W":
                self.ffi.cdef(
                    """
                    int __stdcall WriteConsoleW(void* hConsoleOutput, const wchar_t* lpBuffer,
                                               unsigned long nNumberOfCharsToWrite,
                                               unsigned long* lpNumberOfCharsWritten,
                                               void* lpReserved);
                    void* __stdcall GetStdHandle(int nStdHandle);
                    """
                )
            else:  # pragma: no cover - defensive clause
                raise ValueError("mode must be 'A' or 'W'")

            self.C = self.ffi.dlopen("kernel32.dll")
            self.STD_OUTPUT_HANDLE = -11
            self._handle = self.C.GetStdHandle(self.STD_OUTPUT_HANDLE)
            if self._handle == self.ffi.NULL or self._handle == self.ffi.cast("void*", -1):
                raise OSError("GetStdHandle failed")
        else:
            # POSIX: stream bytes directly to stdout via write(2)
            self.ffi.cdef("ssize_t write(int fd, const void* buf, size_t count);")
            libc: Optional[object] = None
            for lib in (None, "libc.so.6", "libc.dylib"):
                try:
                    libc = self.ffi.dlopen(lib) if lib is not None else self.ffi.dlopen(None)
                    break
                except OSError:  # pragma: no cover - varies by platform
                    continue
            if libc is None:  # pragma: no cover - defensive
                raise OSError("Unable to load C library for write")
            self.C = libc
            self._fd = 1  # STDOUT

        self.threaded = threaded
        self._queue = queue.Queue(maxsize=queue_size) if threaded else None
        self._thread = None
        self._stop_event = threading.Event() if threaded else None
        if threaded:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def print(self, s: str) -> None:
        if not isinstance(s, str):
            s = str(s)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("printing %d chars", len(s))
        if self.threaded:
            try:
                self._queue.put(s, block=False)
            except queue.Full:
                # Optionally drop or block; here we drop if full
                pass
        else:
            self._print_direct(s)

    def _print_direct(self, s: str) -> None:
        if self._is_windows:
            written = self.ffi.new("unsigned long *")
            if self.mode == "A":
                b = s.encode("utf-8")
                n = len(b)
                ret = self.C.WriteConsoleA(self._handle, b, n, written, self.ffi.NULL)
                if ret == 0:
                    raise OSError("WriteConsoleA failed")
            else:
                w = s.encode("utf-16-le")
                n = len(w) // 2
                ret = self.C.WriteConsoleW(
                    self._handle,
                    self.ffi.from_buffer("wchar_t[]", w),
                    n,
                    written,
                    self.ffi.NULL,
                )
                if ret == 0:
                    raise OSError("WriteConsoleW failed")
        else:
            b = s.encode("utf-8")
            self.C.write(self._fd, b, len(b))

    def _worker(self):
        while not self._stop_event.is_set():
            try:
                s = self._queue.get(timeout=0.1)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("worker printing %d chars", len(s))
                self._print_direct(s)
                self._queue.task_done()
            except queue.Empty:
                continue

    def flush(self):
        if self.threaded and self._queue:
            self._queue.join()

    def stop(self):
        if self.threaded and self._stop_event:
            self._stop_event.set()
            if self._thread:
                self._thread.join()
