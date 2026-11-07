"""The 3-D fountain's server side (``fountain3d``; the 2-D fountain is ``fountain.py`` and stays as it was): the game (phase, mix, size, the tunables, the image hand-off, verification) and the page's
controls (``controls_ui``).

Kept apart from ``fountain3d.py`` on purpose: the compiled products (``fountain_build``'s digest, ``LawPiece``) hash the law
module, so an edit to the page's controls here never forces the long rebuild.  ``fountain3d.FountainGame`` and the option
lists below are still reachable through ``meowdoku.fountain3d`` (its module ``__getattr__``).
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from meowdoku.fountain3d import *  # noqa: F401,F403  (the law, its constants and the host helpers)
from meowdoku.fountain3d import _f32  # noqa: F401
from meowdoku.games import Game

#: The time-strata rates the page offers (hue cycles over one emission): slower keeps wider bands of one colour in place.
STRATA_CHOICES = (0.5, 1.0, 2.0, 4.0, 8.0)
# the page's render looks for the particle field (web/particle_styles.js STYLES -- a test holds the two lists equal)
PARTICLE_STYLES = (("points", "points"), ("liquid", "liquid"), ("sand", "sand"), ("glow", "glow"), ("heat", "heat"),
                   ("stamp", "stamp (memory)"))
# the page's noise-filter replacement modes (web/noise_filter.js NOISE_MODES -- a test holds the two lists equal)
NOISE_MODES = (("mean", "mean colour"), ("spectral", "spectral class"))


class FountainGame(Game):
    """Server side of the fountain: phase, mix, size, the tunables, the image hand-off and
    verification by the law's native product.  It does not stream particles:
    the main page's generic compute runtime runs the compiled law (WebGPU
    compute or the wasm product, as the compiler's verdict and the browser
    allow) and draws the particles in the shared scene.

    Actions (``/api/action``): ``fountain_start`` (phase 1; optional ``mix``,
    ``mode``), ``fountain_mode``, ``fountain_speed`` (0 follows the newest frame), ``fountain_run`` (``continuous`` or ``settle``), ``fountain_size``,
    ``fountain_param`` (``name``, ``value``: one tunable of ``PARAMS``; the live ones reach the running law at the next batch,
    the restart ones begin a new run), ``fountain_choice`` (``name`` in ``shape``, ``inlet``, ``fluid`` and its ``value``),
    ``fountain_image`` (per-id colours as base64 bytes ``rgb``, three per id,
    or just ``count`` for sizes too large to ship), ``fountain_replay``
    (phase 2) and ``fountain_verify`` (run the law ``steps`` steps from round
    zero natively, small sizes only, and report a digest)."""

    game = "fountain3d"
    input = "compute"
    title = "Fountain 3D"

    def __init__(self, board_id: str = "", *, size: str = DEFAULT_SIZE, aspect: float | None = None):
        super().__init__(board_id)
        self.size = size
        self.mix, self.mode, self.speed = 0.5, 2, 1
        self.strata = STRATA
        self.run = "continuous"              # "continuous": never stops; "settle": stops at rest (the image replay needs it)
        self.aspect: float | None = aspect       # the screen's width / height when the run started; part of the recorded run
        self.params: dict[str, float] = {}       # the page's overrides of PARAMS (the rest are the defaults, the shares follow ``mix``)
        self.shape, self.inlet, self.fluid = "box", "fountain", DEFAULT_FLUID
        self.phase = "fill"
        self.epoch = 1                       # bumped whenever the page must restart the fountain
        self.image: np.ndarray | None = None
        self.has_image = False
        self.piece: FountainPiece | None = None
        self.verified: dict[str, Any] | None = None
        self.status = "fountain"

    @property
    def n(self) -> int:
        return SIZES[self.size][0]

    @property
    def size_tag(self) -> str:
        return self.size

    def resolved(self) -> dict[str, float]:
        p = resolve_params(self.mix, self.params)
        if "spread" not in self.params:
            p["spread"] = inlet_default_spread(self.inlet)
        return p

    def uniforms(self, *, image: bool = False) -> dict[str, np.ndarray]:
        return uniforms(n=self.n, mode=self.mode, use_image=1.0 if image else 0.0, mix=self.mix, aspect=self.aspect,
                        strata=self.strata, params=self.params, shape=self.shape, inlet=self.inlet)

    def columns(self, *, image: bool = False) -> dict[str, np.ndarray]:
        pad = SIZES[self.size][1]
        cols = initial_columns(pad, mix=self.mix, mode=self.mode, use_image=1.0 if image else 0.0,
                               active=self.n, aspect=self.aspect, strata=self.strata, params=self.params,
                               shape=self.shape, inlet=self.inlet)
        if image and self.image is not None:
            cols["ir"][: self.n], cols["ig"][: self.n], cols["ib"][: self.n] = (
                _f32(self.image[:, c] / 255.0) for c in range(3))
        return cols

    def reset(self) -> None:
        self.phase, self.verified = "fill", None
        self.epoch += 1

    def do(self, payload: dict) -> None:
        kind = str(payload.get("type"))
        if kind == "fountain_start":
            if "mix" in payload:
                self.mix = float(payload["mix"])
                for s in range(len(SPECIES)):                  # the poly mix is the macro that sets the four shares
                    self.params.pop(f"wt{s}", None)
            self.mode = int(payload.get("mode", self.mode))
            # a run's basin is fixed when it starts (and kept by its replay): a screen that rotates later changes nothing
            if payload.get("aspect"):
                self.aspect = float(payload["aspect"])
            self.reset()
        elif kind == "fountain_mode":
            self.mode = int(payload["mode"])
        elif kind == "fountain_strata":
            self.strata = float(payload["strata"])
        elif kind == "fountain_speed":
            self.speed = int(payload["speed"])
        elif kind == "fountain_param":
            name = str(payload["name"])
            if name not in PARAMS:
                raise ValueError(f"fountain_param: unknown tunable {name!r}")
            self.params[name] = min(max(float(payload["value"]), PARAMS[name][1]), PARAMS[name][2])
            if name in ("fl_rho", "fl_mu"):
                self.fluid = "custom"
            if PARAMS[name][5]:
                if payload.get("aspect"):
                    self.aspect = float(payload["aspect"])
                self.reset()
        elif kind == "fountain_choice":
            name, value = str(payload["name"]), str(payload["value"])
            if name == "shape":
                if value not in SHAPES:
                    raise ValueError(f"fountain_choice: unknown shape {value!r}")
                self.shape = value
                self.reset()
            elif name == "inlet":
                if value not in INLETS:
                    raise ValueError(f"fountain_choice: unknown inlet {value!r}")
                self.inlet = value
                self.params.pop("spread", None)
                self.reset()
            elif name == "fluid":
                if value not in FLUIDS:
                    raise ValueError(f"fountain_choice: unknown fluid {value!r}")
                self.fluid = value
                self.params["fl_rho"], self.params["fl_mu"] = FLUIDS[value]
            else:
                raise ValueError(f"fountain_choice: unknown choice {name!r}")
        elif kind == "fountain_run":
            run = str(payload["run"])
            if run not in ("continuous", "settle"):
                raise ValueError(f"fountain_run: unknown run {run!r}")
            self.run = run
            self.reset()
        elif kind == "fountain_size":
            tag = str(payload["size"])
            if tag not in SIZES:
                raise ValueError(f"fountain_size: unknown size {tag!r}")
            self.size = tag
            if payload.get("aspect"):
                self.aspect = float(payload["aspect"])
            self.image, self.has_image, self.piece = None, False, None
            self.reset()
        elif kind == "fountain_image":
            if "rgb" in payload:
                import base64
                raw = np.frombuffer(base64.b64decode(str(payload["rgb"])), dtype=np.uint8)
                if raw.size != self.n * 3:
                    raise ValueError(f"fountain_image: expected {self.n * 3} bytes, got {raw.size}")
                self.image = raw.reshape(self.n, 3).copy()
            elif int(payload.get("count", -1)) != self.n:
                raise ValueError("fountain_image: neither rgb bytes nor a matching count")
            self.has_image = True
        elif kind == "fountain_replay":
            if not self.has_image:
                raise ValueError("fountain_replay: supply an image first")
            self.phase, self.verified = "replay", None
            self.epoch += 1
        elif kind == "fountain_verify":
            self.verify(int(payload.get("steps", 600)), image=bool(payload.get("image", False)))
        else:
            raise ValueError(f"unknown fountain action {kind!r}")

    def verify(self, steps: int, *, image: bool = False) -> dict[str, Any]:
        active, pad = SIZES[self.size]
        if pad > 4096:
            raise ValueError("fountain_verify: the native lane is built for sizes up to 4096 ids")
        columns = self.columns(image=image)
        if self.piece is None:
            self.piece = make_piece(pad, active)
            self.piece.compile(columns)
        final = run_frames(self.piece, columns, steps)
        digest = hashlib.sha256(b"".join(np.ascontiguousarray(final[n], dtype=np.float32).tobytes()
                                         for n in STATE + ("cr", "cg", "cb"))).hexdigest()[:16]
        emitted = int((columns["pid"][:active] * float(columns["emit_rate"]) <= steps * DT).sum())
        self.verified = {"steps": steps, "lane": self.piece.lane, "digest": digest,
                         "emitted": emitted, "top": float(final["y"].max())}
        return self.verified

    def _control(self, name: str, kind: str = "range", **extra) -> dict[str, Any]:
        label, lo, hi, step, default, restart, group = PARAMS[name]
        value = self.resolved()[name]
        spec = {"id": name, "kind": kind, "label": label, "min": lo, "max": hi, "step": step, "value": value,
                "action": "fountain_param", "name": name, "field": "value", "group": group}
        if restart:
            spec["aspect"] = True
        spec.update(extra)
        return spec

    def controls_ui(self) -> list[dict[str, Any]]:
        sample = {"fit": "extent", "read": ["x", "y"], "into": ["ir", "ig", "ib"], "raster": [96, 72],
                  "post": {"action": "fountain_image", "field": "rgb", "max_ids": 65536}}
        sizes = [[tag, f"{count:,}"] for tag, (count, _pad) in SIZES.items()]
        p = self.resolved()
        out: list[dict[str, Any]] = [
            {"id": "start", "kind": "button", "label": "Start", "action": "fountain_start", "aspect": True},
            {"id": "image", "kind": "file", "label": "Image", "sample": sample},
            {"id": "camera", "kind": "file", "label": "Camera", "capture": "environment", "sample": sample},
            {"id": "replay", "kind": "button", "label": "Image replay", "action": "fountain_replay",
             "enabled": self.has_image},
            {"id": "run", "kind": "select", "label": "run", "value": self.run, "action": "fountain_run",
             "field": "run", "options": [["continuous", "continuous"], ["settle", "stop at rest"]]},
            # the run modes are the page's own (``local``): the compute runtime's recording, not a server action
            {"id": "rebirth", "kind": "button", "label": "Rebirth", "local": "rebirth"},
            {"id": "rec_replay", "kind": "button", "label": "Replay", "local": "replay"},
            {"id": "bounce", "kind": "button", "label": "Bounce", "local": "bounce"},
            {"id": "continue", "kind": "button", "label": "Continue", "local": "continue"},
            {"id": "tilt", "kind": "toggle", "label": "phone gravity", "local": "tilt", "value": True},
            {"id": "orbit", "kind": "toggle", "label": "auto-rotate", "local": "orbit", "value": False},
            # the particle field's render look is the page's own too (web/particle_styles.js STYLES: same ids, same order); render only
            {"id": "pstyle", "kind": "select", "label": "look", "local": "particle_style", "value": "points",
             "options": [[tag, label] for tag, label in PARTICLE_STYLES]},
            # the spectral noise filter over the look (web/noise_filter.js): speckle becomes a solid colour; also the page's own, render only
            {"id": "nf", "kind": "toggle", "label": "noise filter", "local": "noise_on", "value": False},
            {"id": "nfmode", "kind": "select", "label": "replace with", "local": "noise_mode", "value": "mean",
             "options": [[tag, label] for tag, label in NOISE_MODES]},
            {"id": "nfstrength", "kind": "range", "label": "strength", "min": 0, "max": 1, "step": 0.05, "local": "noise_strength", "value": 0.85},
            {"id": "nfthr", "kind": "range", "label": "threshold", "min": 0.2, "max": 0.9, "step": 0.05, "local": "noise_threshold", "value": 0.5},
            # a spinning case: gravity turns in the screen's plane at a set rate (a runtime input of the law, like the tilt)
            {"id": "spin", "kind": "range", "label": "case spin (Hz)", "min": -1, "max": 1, "step": 0.05, "local": "spin", "value": 0},
            {"id": "mix", "kind": "range", "label": "poly mix", "min": 0, "max": 1, "step": 0.05,
             "value": self.mix, "action": "fountain_start", "field": "mix", "aspect": True},
            {"id": "mode", "kind": "select", "label": "colour", "value": self.mode, "action": "fountain_mode",
             "field": "mode", "options": [[2, "time strata"], [0, "trig (atan2)"], [1, "rotation"]]},
            {"id": "strata", "kind": "select", "label": "colour rate", "value": self.strata, "action": "fountain_strata",
             "field": "strata", "options": [[s, f"{s:g} cycle" + ("" if s == 1 else "s")] for s in STRATA_CHOICES]},
            {"id": "size", "kind": "select", "label": "ids", "value": self.size, "action": "fountain_size",
             "field": "size", "options": sizes, "aspect": True},
            {"id": "speed", "kind": "select", "label": "speed", "value": self.speed, "action": "fountain_speed",
             "field": "speed", "options": [[1, "1x"], [2, "2x"], [4, "4x"], [0, "max"]]},
            {"id": "verify", "kind": "button", "label": "Verify", "action": "fountain_verify"},
        ]
        for s, (name, *_rest) in enumerate(SPECIES):                 # Grains: the species table
            for key, label in (("r", "radius"), ("rho", "density"), ("mu", "friction"), ("e", "restitution"),
                               ("roll", "roughness"), ("irr", "irregularity"), ("wt", "share")):
                ctl = self._control(f"{key}{s}")
                ctl["label"] = f"{name} {label}"
                ctl["species"] = s
                out.append(ctl)
        out.append({"id": "lognormal", "kind": "select", "label": "size distribution", "value": int(self.resolved()["lognormal"] > 0.5),
                    "action": "fountain_param", "name": "lognormal", "field": "value", "group": "Grains", "aspect": True,
                    "options": [[0, "uniform width"], [1, "lognormal sigma"]]})
        for name in ("bi_ratio", "bi_frac"):                       # Grains: the optional two-size (bidisperse) mode
            out.append(self._control(name))
        for name in ("omega", "hertz", "coh", "spin_drag", "temp", "nozzle_speed", "rate", "k_run"):       # Physics
            # the temperature is a logarithmic slider (0 = off, then kT from TEMP_MIN_LOG to TEMP_MAX)
            out.append(self._control(name, scale="log", log_min=TEMP_MIN_LOG) if name == "temp" else self._control(name))
        out.append({"id": "k_run_drop", "kind": "readout", "label": "dropped contacts", "group": "Physics", "diag": "n_drop"})
        out.append({"id": "fluid", "kind": "select", "label": "fluid", "value": self.fluid, "action": "fountain_choice",
                    "name": "fluid", "field": "value", "group": "Fluid",
                    "options": [[k, k] for k in FLUIDS] + ([["custom", "custom"]] if self.fluid == "custom" else [])})
        out.extend([self._control("fl_rho"), self._control("fl_mu")])
        out.append({"id": "shape", "kind": "select", "label": "polygon", "value": self.shape, "action": "fountain_choice",
                    "name": "shape", "field": "value", "group": "Chamber", "options": [[k, k] for k in SHAPES]})
        out.append({"id": "inlet", "kind": "select", "label": "inlet", "value": self.inlet, "action": "fountain_choice",
                    "name": "inlet", "field": "value", "group": "Chamber",
                    "options": [["fountain", "fan fountain"], ["jet", "colinear jet"], ["pour", "pour from above"], ["side", "side tube"]]})
        out.extend([self._control("depth"), self._control("spread")])
        for ctl in out:
            ctl.setdefault("group", "Run")
        return out

    def basin(self) -> dict[str, Any]:
        """The running basin: ``n`` ids at the aspect the run started with."""
        return basin(self.n, self.aspect, cell_of(self.resolved()))

    def state(self) -> dict[str, Any]:
        geo = self.basin()
        width, height = board_size(geo["view"])
        base = self.base_state(int(round(height)), int(round(width)))
        verified = self.verified
        text = {"fill": "phase 1: the fountain fills the chamber by contact forces; hue from the launch angle",
                "replay": "phase 2: the identical simulation, coloured by your image at each id's rest position"}[self.phase]
        base.update({
            "board": "free", "width": width, "height": height, "input": self.input,
            "palette": {"floor": [0.10, 0.11, 0.16]}, "cells": ["floor"], "marks": [], "hints": [],
            "pieces": [], "entities": [],
            "status": f"{text} · {self.n:,} ids" + (f" · server {verified['lane']}: {verified['emitted']} emitted, "
                                                    f"top {verified['top']:.3f}, digest {verified['digest']}" if verified else ""),
            "n": self.n, "phase": self.phase, "mix": self.mix, "mode": self.mode,
            "has_image": self.has_image, "verified": verified,
            "controls_ui": self.controls_ui(),
            "camera_gestures": {"orbit": "one", "zoom": True, "pan": True, "tilt": True},   # camera-first: the page's scene-camera layer
            "compute": {
                "manifest": "fountain3d.manifest.json", "size": self.size, "epoch": self.epoch,
                "uniforms": {k: float(v) for k, v in self.uniforms(image=self.phase == "replay").items()},
                "view": list(geo["view"]), "aspect": self.aspect, "depth": self.resolved()["depth"],
                "speed": self.speed, "named": ["ir", "ig", "ib"], "run": self.run,
                "tilt": tilt_inputs(self.n, self.aspect, cell_of(self.resolved())),
            },
        })
        return base

    def result(self):
        return None
