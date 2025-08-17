from __future__ import annotations

"""Centralized, conspicuous debug logging for dt system and classic mechanics.

Use `enable(True)` (or set env TURING_DT_DEBUG=1) to turn on deep, step-by-step
logging across the dt controller, graph runner, engines, and threaded systems.

Helpers:
- dbg(name): namespaced logger under "dt.<name>"
- enable(flag): turn logging on/off globally
- is_enabled(): check global flag
- pretty_metrics(m): compact Metrics rendering for logs
- sample(seq, n): preview first/last items for readable dumps

By default, logging is quiet; enabling debug will configure a stream handler
on the root "dt" logger with a clear, timestamped format.
"""

import logging
import os
import threading
from typing import Any, Iterable

_ENABLED = bool(int(os.getenv("TURING_DT_DEBUG", "0") or "0"))
_LOCK = threading.Lock()


def enable(flag: bool = True, *, level: int = logging.DEBUG) -> None:
    """Enable or disable deep debug logging for the dt system."""
    global _ENABLED
    with _LOCK:
        _ENABLED = bool(flag)
        lg = logging.getLogger("dt")
        lg.propagate = False
        # Idempotent handler setup
        if _ENABLED:
            if not any(isinstance(h, logging.StreamHandler) for h in lg.handlers):
                h = logging.StreamHandler()
                fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
                h.setFormatter(logging.Formatter(fmt=fmt, datefmt="%H:%M:%S"))
                lg.addHandler(h)
            lg.setLevel(level)
        else:
            lg.setLevel(logging.CRITICAL)


def is_enabled() -> bool:
    return _ENABLED


def dbg(name: str) -> logging.Logger:
    """Return a child logger under the dt namespace."""
    lg = logging.getLogger(f"dt.{name}")
    # Ensure base logger respects global state even if enable() not called yet
    root = logging.getLogger("dt")
    if not root.handlers:
        # Default to quiet unless env toggled
        if _ENABLED:
            enable(True)
        else:
            root.setLevel(logging.CRITICAL)
    return lg


def pretty_metrics(m: Any) -> str:
    try:
        return (
            f"max_vel={getattr(m,'max_vel',None):.3e} max_flux={getattr(m,'max_flux',None):.3e} "
            f"div_inf={getattr(m,'div_inf',None):.3e} mass_err={getattr(m,'mass_err',None):.3e} "
            f"osc={getattr(m,'osc_flag',False)} stiff={getattr(m,'stiff_flag',False)}"
        )
    except Exception:
        return str(m)


def sample(seq: Iterable[Any], n: int = 3) -> str:
    """Return a human-friendly preview of a sequence."""
    try:
        lst = list(seq)
    except Exception:
        return str(seq)
    if len(lst) <= n:
        return repr(lst)
    head = ", ".join(repr(x) for x in lst[: n // 2])
    tail = ", ".join(repr(x) for x in lst[-(n - n // 2) :])
    return f"[{head}, …, {tail}] (n={len(lst)})"


__all__ = ["enable", "is_enabled", "dbg", "pretty_metrics", "sample"]
