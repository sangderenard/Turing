#!/usr/bin/env python3
"""Core functions for aligning system time with internet time."""
from __future__ import annotations

import datetime as _dt
import os
from typing import Optional
from . import _internet
# --- END HEADER ---

OFFSET_ENV = "SPEAKTOME_TIME_OFFSET"


def get_offset() -> float:
    """Return the current offset in seconds stored in ``OFFSET_ENV``."""
    try:
        return float(os.getenv(OFFSET_ENV, "0"))
    except ValueError:
        return 0.0


def set_offset(value: float) -> None:
    """Set ``OFFSET_ENV`` to ``value``."""
    os.environ[OFFSET_ENV] = str(value)


def sync_offset() -> float:
    """Synchronize ``OFFSET_ENV`` with internet time.

    The offset is calculated as ``internet_time - system_utc`` in seconds.
    If fetching internet time fails, the existing offset is kept.
    """
    try:
        internet = _internet.fetch_internet_utc()
    except Exception:
        return get_offset()

    system = _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc)
    offset = (internet - system).total_seconds()
    set_offset(offset)
    return offset


def adjust_datetime(dt: _dt.datetime) -> _dt.datetime:
    """Return ``dt`` plus the stored offset."""
    return dt + _dt.timedelta(seconds=get_offset())


def now(tz: Optional[_dt.tzinfo] = None) -> _dt.datetime:
    """Return :func:`datetime.datetime.now` adjusted by the offset."""
    return adjust_datetime(_dt.datetime.now(tz))


def utcnow() -> _dt.datetime:
    """Return :func:`datetime.datetime.utcnow` adjusted by the offset."""
    return adjust_datetime(_dt.datetime.utcnow())
