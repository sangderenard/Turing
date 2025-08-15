from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any, Optional

@dataclass
class SimHooks:
    """Optional pre/post hooks for simulator steps.

    Each hook receives the simulator instance and the timestep ``dt``.
    Use to assemble custom simulator chains where one engine consults
    another before or after advancing.
    """
    pre: Optional[Callable[[Any, float], None]] = None
    post: Optional[Callable[[Any, float], None]] = None

    def run_pre(self, engine: Any, dt: float) -> None:
        if self.pre is not None:
            try:
                self.pre(engine, dt)
            except Exception:
                pass

    def run_post(self, engine: Any, dt: float) -> None:
        if self.post is not None:
            try:
                self.post(engine, dt)
            except Exception:
                pass
