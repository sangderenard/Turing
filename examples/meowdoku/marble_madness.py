"""A single playable marble in the existing 3-D fountain DEM law.

The compiled 4k SDF product remains the law. A very long emission
interval leaves ids 1..4095 parked while id 0 rolls and collides with the
hourglass toy's analytic ramps, throat and pegs. Keyboard/phone controls only
change the law's documented runtime gravity vector; this module adds no
integrator or collision approximation.
"""
from __future__ import annotations

import math

from meowdoku.fountain_game import FountainGame
from meowdoku import fountain3d as F
from meowdoku.toy_design import toy_parts


class MarbleMadness(FountainGame):
    game = "marble_madness"
    title = "Marble Madness"
    EMIT_INTERVAL = 1.0e9

    def __init__(self, *, aspect: float | None = None):
        super().__init__(size="4k", aspect=aspect)
        self.shape = "sdf_toy"
        self.inlet = "pour"
        self.fluid = "air"
        self.params.update({"wt0": 1.0, "wt1": 0.0, "wt2": 0.0, "wt3": 0.0,
                            "r0": 0.020, "irr0": 0.0, "spread": 0.0,
                            "nozzle_speed": 0.3, "fl_rho": F.FLUIDS["air"][0],
                            "fl_mu": F.FLUIDS["air"][1]})
        self.won = False
        self.seconds: float | None = None

    def uniforms(self, *, image: bool = False):
        values = super().uniforms(image=image)
        values["emit_rate"] = F._f32(self.EMIT_INTERVAL)
        return values

    def goal(self) -> dict[str, float]:
        p = self.resolved()
        geo = F.basin(self.n, self.aspect, F.cell_of(p), self.shape)
        cell, width, _ = F.window_fit(self.shape, geo)
        pool = toy_parts(width, cell, [species[1] for species in F.SPECIES])["pool"]
        return {"x": 0.0, "y": 0.5 * pool, "half_width": 0.30 * width,
                "ceiling": pool}

    def do(self, payload: dict) -> None:
        kind = str(payload.get("type"))
        if kind == "marble_restart":
            self.reset()
        elif kind == "fountain_param":
            super().do(payload)
        elif kind == "marble_finish":
            if self.won:
                return
            x, y, z = (float(payload[k]) for k in ("x", "y", "z"))
            seconds = float(payload["seconds"])
            goal = self.goal()
            u = {k: float(v) for k, v in self.uniforms().items()}
            distance = float(F.chamber_distance(u, x, y, z)[0].item())
            if not (math.isfinite(seconds) and 0 < seconds < 3600 and
                    all(math.isfinite(v) for v in (x, y, z)) and
                    abs(x - goal["x"]) <= goal["half_width"] and
                    0 <= y <= goal["ceiling"] and distance >= -0.005):
                raise ValueError("marble has not reached the goal")
            self.seconds = seconds
            self.won = True
        else:
            raise ValueError(f"unknown marble action {kind!r}")

    def reset(self) -> None:
        self.won = False
        self.seconds = None
        super().reset()

    def controls_ui(self):
        return [
            {"id": "marble_restart", "kind": "button", "label": "Restart marble", "action": "marble_restart", "group": "Run"},
            self._control("mu0", label="Sliding friction (contact μ)", group="Marble contact"),
            self._control("roll0", label="Rolling resistance", group="Marble contact"),
        ]

    def state(self):
        state = super().state()
        state["won"] = self.won
        state["status"] = (f"Goal reached in {self.seconds:.2f} s" if self.won else
                           "Guide the cat's-eye marble down the ramps and through the pegs · arrows/WASD or dual thumbsticks tilt X/Y from the backplate normal · optional phone tilt · R restarts")
        state["marble"] = {"goal": self.goal(), "seconds": self.seconds, "active_ids": 1}
        return state

    def result(self):
        if self.seconds is None:
            return None
        return {"score": max(1, round(100000 / self.seconds)), "detail": f"goal in {self.seconds:.2f} s"}
