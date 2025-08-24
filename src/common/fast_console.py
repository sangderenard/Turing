
"""
Fast console output using CFFI and Windows API (WriteConsoleA/WriteConsoleW).
"""
import cffi
import threading
import queue
import time
import logging
import os


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
    def __init__(self, mode: str = 'A', threaded: bool = False, queue_size: int = 128):
        """
        mode: 'A' for WriteConsoleA (ANSI), 'W' for WriteConsoleW (Unicode)
        threaded: if True, use a background thread to print from a queue
        queue_size: max number of queued print jobs (threaded mode)
        """
        self.ffi = cffi.FFI()
        self.mode = mode.upper()
        if self.mode == 'A':
            self.ffi.cdef('''
                int WriteConsoleA(void* hConsoleOutput, const char* lpBuffer, unsigned long nNumberOfCharsToWrite, unsigned long* lpNumberOfCharsWritten, void* lpReserved);
                void* GetStdHandle(int nStdHandle);
            ''')
        elif self.mode == 'W':
            self.ffi.cdef('''
                int WriteConsoleW(void* hConsoleOutput, const wchar_t* lpBuffer, unsigned long nNumberOfCharsToWrite, unsigned long* lpNumberOfCharsWritten, void* lpReserved);
                void* GetStdHandle(int nStdHandle);
            ''')
        else:
            raise ValueError("mode must be 'A' or 'W'")
        self.C = self.ffi.dlopen('kernel32.dll')
        self.STD_OUTPUT_HANDLE = -11
        self._handle = self.C.GetStdHandle(self.STD_OUTPUT_HANDLE)

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
        written = self.ffi.new('unsigned long *')
        if self.mode == 'A':
            b = s.encode('utf-8')
            n = len(b)
            self.C.WriteConsoleA(self._handle, b, n, written, self.ffi.NULL)
        else:
            w = s.encode('utf-16-le')
            n = len(w) // 2
            self.C.WriteConsoleW(self._handle, self.ffi.from_buffer('wchar_t[]', w), n, written, self.ffi.NULL)

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
