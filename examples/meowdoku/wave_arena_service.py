"""Session-scoped browser admission for Spectral's existing T4 wave demo.

The numerical step and field presentation remain owned by Spectral Analyzer.
This module only bounds live sessions and uses the site's normal game cookie.
"""
from __future__ import annotations

from collections import OrderedDict
import importlib.util
from pathlib import Path
import sys
import threading
import time


MAX_ARENAS = 8
IDLE_SECONDS = 600


class WaveArenaService:
    def __init__(self, spectral_root: Path) -> None:
        self.spectral_root = Path(spectral_root).resolve()
        self.lock = threading.Lock()
        self.arenas: OrderedDict[str, tuple[object, float]] = OrderedDict()
        self.wave_type = None

    def _type(self):
        if self.wave_type is None:
            source = self.spectral_root / "wave_arena_web_demo.py"
            if not source.is_file():
                raise RuntimeError("Spectral T4 wave arena source is unavailable")
            root = str(self.spectral_root)
            if root not in sys.path:
                sys.path.insert(0, root)
            spec = importlib.util.spec_from_file_location(
                "spectral_wave_arena_site", source)
            if spec is None or spec.loader is None:
                raise RuntimeError("Spectral T4 wave arena source cannot be loaded")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.wave_type = module.WaveArena
        return self.wave_type

    def command(self, session: str, payload: dict | None = None) -> dict:
        if not session:
            raise ValueError("a site session is required")
        with self.lock:
            now = time.monotonic()
            for key, (_, seen) in list(self.arenas.items()):
                if now - seen > IDLE_SECONDS:
                    del self.arenas[key]
            arena = self.arenas.pop(session, (None, 0.0))[0]
            if payload is not None:
                if not isinstance(payload, dict):
                    raise ValueError("expected an arena command object")
                operation = payload.get("op")
                if operation == "configure":
                    config = {key: value for key, value in payload.items() if key != "op"}
                    if arena is None:
                        arena = self._type()(config)
                    else:
                        arena.configure(config)
                elif operation == "step":
                    if set(payload) - {"op", "count"}:
                        raise ValueError("unknown step parameter")
                    if arena is None:
                        arena = self._type()()
                    arena.step(int(payload.get("count", 1)))
                else:
                    raise ValueError("op must be configure or step")
            if arena is None:
                arena = self._type()()
            self.arenas[session] = (arena, now)
            while len(self.arenas) > MAX_ARENAS:
                self.arenas.popitem(last=False)
            return arena.snapshot()
