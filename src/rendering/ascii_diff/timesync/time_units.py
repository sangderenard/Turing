#!/usr/bin/env python3
"""Defines TimeUnit class and utilities for configurable clock units."""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Callable, Any, List, Tuple, Dict
# --- END HEADER ---

# --- Accessor Functions ---
# These functions extract a specific time component from a datetime or timedelta object.
# They should handle either type gracefully or be specific if needed.

def _get_dt_hour_12_float(obj: _dt.datetime | _dt.timedelta) -> float:
    return obj.hour % 12 + obj.minute / 60 if isinstance(obj, _dt.datetime) else 0.0

def _get_dt_hour_24_float(obj: _dt.datetime | _dt.timedelta) -> float:
    return obj.hour + obj.minute / 60 if isinstance(obj, _dt.datetime) else 0.0

def _get_dt_minute_float(obj: _dt.datetime | _dt.timedelta) -> float:
    return obj.minute + obj.second / 60 if isinstance(obj, _dt.datetime) else 0.0

def _get_dt_second_float(obj: _dt.datetime | _dt.timedelta) -> float:
    return obj.second + obj.microsecond / 1_000_000 if isinstance(obj, _dt.datetime) else 0.0

def _get_dt_millisecond_float(obj: _dt.datetime | _dt.timedelta) -> float:
    return obj.microsecond / 1000 if isinstance(obj, _dt.datetime) else 0.0

def _get_td_days_float(obj: _dt.datetime | _dt.timedelta) -> float:
    return obj.days if isinstance(obj, _dt.timedelta) else 0.0

def _get_td_hours_float(obj: _dt.datetime | _dt.timedelta) -> float:
    # Total hours in the timedelta, not just the hour part
    return (obj.total_seconds() / 3600) if isinstance(obj, _dt.timedelta) else 0.0

def _get_td_minutes_float(obj: _dt.datetime | _dt.timedelta) -> float:
    # Total minutes in the timedelta
    return (obj.total_seconds() / 60) if isinstance(obj, _dt.timedelta) else 0.0

def _get_td_seconds_float(obj: _dt.datetime | _dt.timedelta) -> float:
    return obj.total_seconds() if isinstance(obj, _dt.timedelta) else 0.0


ACCESSOR_REGISTRY: Dict[str, Callable[[_dt.datetime | _dt.timedelta], float]] = {
    "dt_hour12_float": _get_dt_hour_12_float,
    "dt_hour24_float": _get_dt_hour_24_float,
    "dt_minute_float": _get_dt_minute_float,
    "dt_second_float": _get_dt_second_float,
    "dt_millisecond_float": _get_dt_millisecond_float,
    "td_days_float": _get_td_days_float,
    "td_hours_float": _get_td_hours_float,
    "td_minutes_float": _get_td_minutes_float,
    "td_seconds_float": _get_td_seconds_float,
}

@dataclass
class TimeUnit:
    name: str
    max_value: float  # Value for a full cycle (e.g., 12 for hours on analog, 60 for minutes)
    accessor_key: str # Key into ACCESSOR_REGISTRY
    # Python's str.format() style, e.g., "{value:02.0f}" for digital
    # For analog, this can be used to derive hand properties or ignored
    format_template: str = "{value:02.0f}"
    # Analog clock hand properties (can be overridden by theme)
    hand_length_factor: float = 0.5 # Relative to clock radius
    hand_width_factor: float = 0.05 # Relative to clock effective diameter
    hand_color_key: str = "default_hand_color" # Key in theme palette for this hand's color

    def get_value(self, time_obj: _dt.datetime | _dt.timedelta) -> float:
        """Extracts the unit's value from the given time object."""
        accessor_func = ACCESSOR_REGISTRY.get(self.accessor_key)
        if accessor_func:
            return accessor_func(time_obj)
        # Fallback or error for unknown accessor
        print(f"Warning: Accessor key '{self.accessor_key}' not found for unit '{self.name}'. Returning 0.")
        return 0.0

    def format_value(self, time_obj: _dt.datetime | _dt.timedelta) -> str:
        """Formats the unit's value as a string for digital display."""
        value = self.get_value(time_obj)
        try:
            return self.format_template.format(value=value)
        except Exception as e: # Broad except for various formatting issues
            print(f"Warning: Could not format value {value} for unit '{self.name}' with template '{self.format_template}'. Error: {e}")
            return str(value) # Fallback to simple string conversion

    @classmethod
    def from_dict(cls, name: str, data: dict) -> TimeUnit:
        """Creates a TimeUnit instance from a dictionary (e.g., from JSON)."""
        return cls(name=name, **data)