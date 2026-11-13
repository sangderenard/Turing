"""Per-account site access to Spectral's existing native T4 arena demonstration."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import threading
import time


class WaveArenaSite:
    def __init__(self, spectral_root: Path) -> None:
        self.spectral_root = Path(spectral_root).resolve()
        self._lock = threading.Lock()
        self._arenas: dict[int, tuple[object, float]] = {}
        self._arena_type = None

    def _type(self):
        if self._arena_type is None:
            source = self.spectral_root / "wave_arena_web_demo.py"
            if not source.is_file():
                raise RuntimeError("T4 wave arena is unavailable on this server")
            if str(self.spectral_root) not in sys.path:
                sys.path.insert(0, str(self.spectral_root))
            spec = importlib.util.spec_from_file_location("spectral_wave_arena_site", source)
            if spec is None or spec.loader is None:
                raise RuntimeError("T4 wave arena source cannot be loaded")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._arena_type = module.WaveArena
        return self._arena_type

    def request(self, account_id: int, payload: dict | None = None) -> dict:
        if not isinstance(account_id, int) or account_id <= 0:
            raise ValueError("a signed-in account is required")
        with self._lock:
            now = time.monotonic()
            self._arenas = {
                key: value for key, value in self._arenas.items()
                if now - value[1] < 1800 or key == account_id
            }
            if account_id not in self._arenas:
                try:
                    self._arenas[account_id] = (self._type()(), now)
                except (ImportError, OSError) as error:
                    raise RuntimeError("the native T4 wave kernel is unavailable") from error
                if len(self._arenas) > 32:
                    oldest = min((key for key in self._arenas if key != account_id),
                                 key=lambda key: self._arenas[key][1])
                    del self._arenas[oldest]
            arena, _ = self._arenas[account_id]
            if payload is not None:
                if not isinstance(payload, dict):
                    raise ValueError("expected a JSON object")
                command = dict(payload)
                operation = command.pop("op", None)
                if operation == "configure":
                    arena.configure(command)
                elif operation == "step":
                    if set(command) - {"count"}:
                        raise ValueError("unknown step parameter")
                    arena.step(int(command.get("count", 1)))
                else:
                    raise ValueError("op must be configure or step")
            self._arenas[account_id] = (arena, now)
            return arena.snapshot()
