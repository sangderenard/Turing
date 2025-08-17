# -*- coding: utf-8 -*-
"""Registry and forward-facing API for dt-graph integration.

- register_engine: simple entrypoint for engines
- list_engines, get_engine: query helpers
"""
from __future__ import annotations

from typing import Dict, Optional

from .engine_api import EngineRegistration


_REGISTRY: Dict[str, EngineRegistration] = {}


def register_engine(reg: EngineRegistration) -> None:
    if reg.name in _REGISTRY:
        raise ValueError(f"engine already registered: {reg.name}")
    _REGISTRY[reg.name] = reg


def list_engines() -> list[str]:
    return list(_REGISTRY.keys())


def get_engine(name: str) -> Optional[EngineRegistration]:
    return _REGISTRY.get(name)


__all__ = [
    "register_engine",
    "list_engines",
    "get_engine",
]
