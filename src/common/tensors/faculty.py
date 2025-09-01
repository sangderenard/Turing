"""Faculty levels for runtime resources."""
from __future__ import annotations

try:
    import os
    from enum import IntEnum
    import importlib.util
except Exception:
    import sys
    print("faculty.py: Failed to import required modules.")
    sys.exit(1)


class Faculty(IntEnum):
    """Available compute/resource tiers."""

    PURE_PYTHON = 1  # No third-party numerical libs
    NUMPY = 2  # Research demo of algorithm
    TORCH = 3  # Performant production faculty
    PYGEO = 4  # NN programmable smart search
    CTENSOR = 5  # Experimental C backend



FORCE_ENV = "TENSOR_FACULTY"


def detect_faculty() -> Faculty:
    """Return the highest available Faculty tier based on installed packages.

    The environment variable ``TENSOR_FACULTY`` may be set to force a
    specific tier regardless of installed libraries.
    """
    forced = os.environ.get(FORCE_ENV)
    if forced:
        try:
            return Faculty[forced.upper()]
        except KeyError as exc:  # pragma: no cover - env misuse
            raise ValueError(f"Unknown faculty override: {forced}") from exc

    spec = importlib.util.find_spec
    if spec("numpy") is not None:
        return Faculty.NUMPY
    if spec("torch_geometric") is not None:
        return Faculty.PYGEO
    if spec("torch") is not None:
        return Faculty.TORCH
    return Faculty.PURE_PYTHON


DEFAULT_FACULTY = detect_faculty()


def available_faculties() -> list[Faculty]:
    """Return all faculty tiers available in the current environment."""
    levels = [Faculty.PURE_PYTHON]
    spec = importlib.util.find_spec
    if spec("numpy") is None:
        return levels
    levels.append(Faculty.NUMPY)
    if spec("torch") is None:
        return levels
    levels.append(Faculty.TORCH)
    if spec("torch_geometric") is not None:
        levels.append(Faculty.PYGEO)
    try:
        from .c_backend import CTensorOperations  # noqa: F401
        levels.append(Faculty.CTENSOR)
    except Exception:
        pass
    return levels
