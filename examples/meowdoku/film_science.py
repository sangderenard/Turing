"""Bounded, CPU-only direct sensor-pattern preview using Spectral's real profile.

This does not create a photo, trace rays, or alter a saved exposure. The same
profile and display functions as the worker produce the returned patch pixels.
"""
from __future__ import annotations

from functools import lru_cache
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading

from .marble_photo import PhotoError, _film_response

_LIMITS = {"exposure": (-6, 6, 0), "gamma": (.2, 3, 1),
           "red": (0, 3, 1), "green": (0, 3, 1), "blue": (0, 3, 1),
           "whitebalance": (-3, 3, 0), "tint": (-3, 3, 0), "intensity": (0, 3, 1),
           "normalization_white": (1e-6, 1e12, 1),
           "normalization_percentile": (1, 100, 99),
           "saturation": (0, 2, 1), "vibrance": (-1, 1, 0), "black_point": (0, .5, 0),
           "firefly_threshold": (2, 64, 8), "firefly_strength": (0, 1, 1)}
_CHOICES = {"normalization": ("auto99", "manual", "percentile", "specular"),
            "tone_curve": ("exponential", "linear"),
            "display_look": ("reference", "color_priority")}
_PREVIEW_LOCK = threading.Lock()


def _request(body: object) -> dict:
    if not isinstance(body, dict) or set(body) - {"film_response", "params"}:
        raise PhotoError("expected film_response and optional display params")
    response = _film_response(body.get("film_response"))
    params = body.get("params", {})
    if not isinstance(params, dict) or set(params) - (set(_LIMITS) | set(_CHOICES) | {"firefly_filter"}):
        raise PhotoError("unsupported film preview display parameter")
    enabled = params.get("firefly_filter", False)
    if not isinstance(enabled, bool):
        raise PhotoError("film preview firefly_filter must be a boolean")
    clean = {"firefly_filter": enabled}
    for name, (low, high, default) in _LIMITS.items():
        if name == "normalization_percentile" and params.get("normalization") == "specular":
            default = 95
        value = params.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PhotoError(f"film preview {name} must be a number")
        try:
            value = float(value)
        except (ValueError, OverflowError):
            raise PhotoError(f"film preview {name} must be finite") from None
        if not math.isfinite(value) or not low <= value <= high:
            raise PhotoError(f"film preview {name} is outside [{low}, {high}]")
        clean[name] = value
    for name, choices in _CHOICES.items():
        value = params.get(name, choices[0])
        if not isinstance(value, str) or value not in choices:
            raise PhotoError(f"unsupported film preview {name}")
        clean[name] = value
    if clean["normalization"] == "specular":
        clean["tone_curve"] = "linear"
        clean["display_look"] = "reference"
    return {"film_response": response, "params": clean}


@lru_cache(maxsize=64)
def _run_preview(root: str, revision: tuple, request_json: str) -> str:
    # Serialize only this bounded scientific helper, not the active GPU job.
    if not _PREVIEW_LOCK.acquire(timeout=2):
        raise PhotoError("Film preview is busy. Try again shortly.", 429)
    try:
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        result = subprocess.run(
            [sys.executable, "-B", "-m", "camera_designer.marble_color_profile"],
            cwd=root, input=request_json, text=True, encoding="utf-8",
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15,
            creationflags=flags,
        )
        if result.returncode or len(result.stdout) > 128 * 1024:
            raise PhotoError("Scientific film preview could not complete.", 503)
        try:
            value = json.loads(result.stdout)
        except (ValueError, TypeError):
            raise PhotoError("Scientific film preview returned an invalid result.", 503) from None
        if not isinstance(value, dict) or value.get("ok") is not True:
            raise PhotoError("Scientific film preview returned no usable result.", 503)
        return json.dumps(value, allow_nan=False)
    except (OSError, subprocess.TimeoutExpired):
        raise PhotoError("Scientific film preview is unavailable or timed out.", 503) from None
    finally:
        _PREVIEW_LOCK.release()


def preview_film(body: object, spectral_root: Path) -> dict:
    request = _request(body)
    root = Path(spectral_root).resolve()
    files = [root / "camera_designer" / name for name in
             ("marble_color_profile.py", "film_response.py", "spectral_sensor_basis.py")]
    try:
        revision = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in files)
    except OSError:
        raise PhotoError("Scientific film preview module is unavailable.", 503) from None
    encoded = json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False)
    # Return an isolated value so callers cannot mutate a cached response.
    return json.loads(_run_preview(str(root), revision, encoded))
