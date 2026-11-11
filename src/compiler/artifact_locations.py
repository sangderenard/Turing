"""Writable compiler defaults, relocatable by the site job subprocess.

Explicit output directories still take precedence at each caller. Without the
site override the historical repository layout is preserved.
"""
from __future__ import annotations

import os
from pathlib import Path


def compiler_artifact_root() -> Path:
    configured = os.environ.get("TURING_COMPILER_ARTIFACT_ROOT")
    if configured:
        path = Path(configured)
        if not path.is_absolute():
            raise ValueError("TURING_COMPILER_ARTIFACT_ROOT must be absolute")
        return path.resolve()
    return Path(__file__).resolve().parents[2]
