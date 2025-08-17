from __future__ import annotations

"""
Pygame demo: dt-graph orchestration with two craft system nodes + bath fluid.

Crafts are built from classic mechanics engines (thrusters, springs,
pneumatics, ground, integrator) inside nested rounds controlled by the dt
graph. Each craft round is wrapped as a DtCompatibleEngine so the parent
graph schedules it like a system node. The bath discrete fluid sim runs as a
regular engine in the same graph. The top-level dt remains centralized.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import math
import os
import argparse

try:
    import pygame
except Exception:  # pragma: no cover - headless environments
    pygame = None  # type: ignore

from ..dt_controller import STController, Targets, step_preview_once
from ..dt_graph import GraphBuilder, MetaLoopRunner
from ..engine_api import EngineRegistration
from ..roundnode_engine import RoundNodeEngine
from ..threaded_system import ThreadedSystemEngine
from ..rt_preview import RTPreviewConfig, RTPreviewState, compile_allocations, compute_penalty
from .engines import (
    DemoState,
    GravityEngine,
    ThrustersEngine,
    SpringEngine,
    PneumaticDamperEngine,
    GroundCollisionEngine,
    IntegratorEngine,
    MetaCollisionEngine,
)
from ..fluid_mechanics import BathDiscreteFluidEngine
from src.cells.bath.discrete_fluid import DiscreteFluid
from ..debug import enable as enable_debug
from ..dt_solver import BisectSolverConfig
# Tip: to run a bisect-based dt solver for a specific engine, pass
# EngineRegistration(..., solver_config=BisectSolverConfig(target=..., field="div_inf"))
# when building the graph. See src/common/dt_system/dt_solver.py for details.


# ---------------- Craft subgraph builder -------------------------

@dataclass
class Craft:
    name: str
    state: DemoState
    thrusters: ThrustersEngine
    round_engine: RoundNodeEngine
    system: ThreadedSystemEngine


def build_craft(name: str, anchor: Tuple[float, float], color=(255, 200, 40)) -> Craft:
    # Minimal 2-vertex "craft": anchor (fixed) + body mass on spring
    pos: List[Tuple[float, float]] = [anchor, (anchor[0] + 0.8, anchor[1] + 0.6)]
    vel = [(0.0, 0.0) for _ in pos]
    acc = [(0.0, 0.0) for _ in pos]
    mass = [0.0, 1.0]  # anchor mass 0 -> inert
    springs = [(0, 1)]
    rest_len = {(0, 1): 0.6}
    k_spring = {(0, 1): 40.0}
    pneu = {(0, 1): (2.0, 4.0)}
    s = DemoState(pos, vel, acc, mass, springs, rest_len, k_spring, pneu, ground_k=1200.0)

    # Engines composing the craft
    thr = ThrustersEngine(s, thrust=(0.0, 0.0))
    engines = [
        (f"{name}.thrusters", thr),
        (f"{name}.springs", SpringEngine(s)),
        (f"{name}.pneumatics", PneumaticDamperEngine(s)),
        (f"{name}.ground", GroundCollisionEngine(s)),
        (f"{name}.integrate", IntegratorEngine(s)),
    ]

    # Per-craft controller/targets; allow the craft to refine within the parent slice
    targets = Targets(cfl=0.9, div_max=1e3, mass_max=1e6)
    ctrl = STController(dt_min=1e-6)
    dx = 0.1
    regs: List[EngineRegistration] = [
        EngineRegistration(name=n, engine=e, targets=targets, dx=dx, localize=True) for (n, e) in engines
    ]
    gb = GraphBuilder(ctrl=ctrl, targets=targets, dx=dx)
    craft_round = gb.round(dt=0.016, engines=regs, schedule="sequential")
    rne = RoundNodeEngine(craft_round)

    # Threaded system wrapper captures a snapshot for UI
    def capture():
        body = s.pos[1]
        anc = s.pos[0]
        return {"craft": {"pos": np.array(body, dtype=float), "anchor": np.array(anc, dtype=float), "color": color}}

    sys = ThreadedSystemEngine(rne, capture=capture)
    return Craft(name=name, state=s, thrusters=thr, round_engine=rne, system=sys)


# ---------------- Parent graph + pygame loop ---------------------

def _run_demo(
    *, width: int = 1000, height: int = 700, fps: int = 60, debug: bool = False, classic: bool = False
) -> None:
    if debug:
        # Also set env so any subprocesses/threads see it
        os.environ["TURING_DT_DEBUG"] = "1"
        enable_debug(True)
    if pygame is None:
        raise RuntimeError("pygame not available")

    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("dt-graph: crafts + bath fluid")
    clock = pygame.time.Clock()

    # Optional simple perspective projection from 3D->2D, default enabled.
    projection_enabled = os.environ.get("TURING_2D", "0") in ("0", "false", "False", "no", "No", "")

    def project_point(p):
        """Project a 2D/3D world point to 2D world coords (pre-screen)."""
        try:
            x = float(p[0]); y = float(p[1])
            z = float(p[2]) if len(p) > 2 else 0.0  # type: ignore[arg-type]
        except Exception:
            # Fallback for dicts or malformed inputs
            try:
                x = float(p.get("x", 0.0)); y = float(p.get("y", 0.0)); z = float(p.get("z", 0.0))
            except Exception:
                x, y, z = 0.0, 0.0, 0.0
        if not projection_enabled:
            return (x, y)
        # Simple camera facing +z from z<0; origin-centered
        cam_z = -5.0
        z_cam = max(z - cam_z, 1e-3)
        f = 1.5  # focal length-ish scalar
        x2 = (x) * (f / z_cam)
        y2 = (y) * (f / z_cam)
        return (x2, y2)

    def world_to_screen(p):
        px, py = project_point(p)
        scale = 60.0
        return (int(px * scale + width * 0.15), int(height - (py * scale + height * 0.15)))

    # Build crafts
    craft_a = build_craft("A", anchor=(2.5, 3.0), color=(240, 80, 80))
    craft_b = build_craft("B", anchor=(6.5, 3.2), color=(80, 180, 255))

    # Bath discrete fluid (small dam break)
    fluid = DiscreteFluid.demo_dam_break(n_x=10, n_y=12, n_z=1, h=0.08)
    fluid_engine = BathDiscreteFluidEngine(fluid)

    # Top-level graph: gravity -> meta-collision(A,B) -> craft A (system) -> craft B (system) -> fluid
    targets = Targets(cfl=0.9, div_max=1e3, mass_max=1e6)
    ctrl = STController(dt_min=1e-6)
    dx = 0.1
    # Build meta collision engine that internally consults the spring/damper networks
    meta_collision = MetaCollisionEngine([craft_a.state, craft_b.state], restitution=0.25, friction_mu=0.6, body_radius=0.12)
    # Configure bisect solver to drive penetration (encoded in Metrics.div_inf) toward 0 within epsilon
    solver_cfg = BisectSolverConfig(target=0.0, eps=1e-5, field="div_inf", monotonic="increase", dt_min=1e-6)

    regs: List[EngineRegistration] = [
        EngineRegistration(name="gravity", engine=GravityEngine(craft_a.state), targets=targets, dx=dx, localize=False),
        EngineRegistration(name="collision", engine=meta_collision, targets=targets, dx=dx, localize=False, solver_config=solver_cfg),
        EngineRegistration(name="craftA", engine=craft_a.system, targets=targets, dx=dx, localize=False),
        EngineRegistration(name="craftB", engine=craft_b.system, targets=targets, dx=dx, localize=False),
        EngineRegistration(name="fluid", engine=fluid_engine, targets=targets, dx=float(fluid.kernel.h), localize=True),
    ]
    gb = GraphBuilder(ctrl=ctrl, targets=targets, dx=dx)
    top_round = gb.round(dt=1.0 / fps, engines=regs, schedule="sequential")
    runner = MetaLoopRunner()

    # Flatten node tree for HUD stats
    def _flatten_nodes(r):
        out = []
        for ch in r.children:
            out.append(ch)
            if hasattr(ch, "children"):
                for gch in ch.children:  # type: ignore[attr-defined]
                    out.append(gch)
        return out

    nodes_for_hud = _flatten_nodes(top_round)

    # Realtime preview state ----------------------------------------------
    # Default: preview mode ON; use --classic to fall back to superstep graph.
    rt_cfg = RTPreviewConfig(budget_ms=1000.0 / max(fps, 1), slack=0.92, beta=1.0, w_floor=0.25, ms_floor=0.25)
    rt_state = RTPreviewState()
    engine_ids = [reg.name for reg in regs]
    # Per-engine controllers (independent PI state) and current dt trackers
    per_ctrl: dict[str, STController] = {}
    dt_curr: dict[str, float] = {}
    for reg in regs:
        if reg.ctrl is not None:
            per_ctrl[reg.name] = reg.ctrl
        else:
            per_ctrl[reg.name] = STController(
                Kp=ctrl.Kp,
                Ki=ctrl.Ki,
                A=ctrl.A,
                shrink=ctrl.shrink,
                dt_min=ctrl.dt_min,
                dt_max=ctrl.dt_max,
            )
        dt_curr[reg.name] = 1.0 / max(fps, 1)

    last_a = {"pos": np.array(craft_a.state.pos[1], dtype=float), "anchor": np.array(craft_a.state.pos[0], dtype=float), "color": (240, 80, 80)}
    last_b = {"pos": np.array(craft_b.state.pos[1], dtype=float), "anchor": np.array(craft_b.state.pos[0], dtype=float), "color": (80, 180, 255)}

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        # WASD for craft A
        ax = float(keys[pygame.K_d]) - float(keys[pygame.K_a])
        ay = float(keys[pygame.K_w]) - float(keys[pygame.K_s])
        craft_a.thrusters.thrust = (ax * 10.0, ay * 10.0)
        # Arrows for craft B
        bx = float(keys[pygame.K_RIGHT]) - float(keys[pygame.K_LEFT])
        by = float(keys[pygame.K_UP]) - float(keys[pygame.K_DOWN])
        craft_b.thrusters.thrust = (bx * 10.0, by * 10.0)

        dt = clock.tick(fps) / 1000.0
        if classic:
            # Classic mode: run superstep via graph
            from ..dt import SuperstepPlan
            saved = top_round.plan
            top_round.plan = SuperstepPlan(round_max=float(dt), dt_init=max(float(dt), 1e-6))
            try:
                _res = runner.run_round(top_round)
            finally:
                top_round.plan = saved
        else:
            # Preview mode: single-step each engine with time allocations
            alloc = compile_allocations(rt_cfg, rt_state, engine_ids)

            # Build simple advance adapters inline (avoids touching the graph runner)
            for reg in regs:
                name = reg.name
                adv = lambda _s, _dt, _eng=reg.engine: _eng.step(_dt)
                m, dt_next, _used = step_preview_once(
                    state=None,
                    dt_current=dt_curr[name],
                    dx=reg.dx,
                    targets=reg.targets,
                    ctrl=per_ctrl[name],
                    advance=adv,
                    alloc_ms=float(alloc.get(name, rt_cfg.ms_floor)),
                    allow_exceptions=False,
                )
                # Update EMA penalty and proc time for next-frame allocations
                pen = compute_penalty(m, reg.targets)
                rt_state.update_penalty(name, pen, rt_cfg.ema_alpha)
                # If metrics recorded proc_ms, include it in EMA for potential future use
                try:
                    _proc_ms = float(getattr(m, "proc_ms", 0.0))
                    rt_state.update_proc_ms(name, _proc_ms, rt_cfg.ema_alpha)
                except Exception:
                    pass
                dt_curr[name] = float(dt_next)

        # Drain craft frames
        try:
            while True:
                frame = craft_a.system.output_queue.get_nowait()
                last_a = frame.get("craft", last_a)
        except Exception:
            pass
        try:
            while True:
                frame = craft_b.system.output_queue.get_nowait()
                last_b = frame.get("craft", last_b)
        except Exception:
            pass

        # Render ------------------------------------------------------
        screen.fill((8, 9, 12))  # slightly cooler dark bg for contrast

        # Fluid particles
        try:
            p = fluid.export_vertices()
            for i in range(min(p.shape[0], 2000)):
                # Accept 2D or 3D particle positions
                if p.shape[1] >= 3:
                    pos = (float(p[i, 0]), float(p[i, 1]), float(p[i, 2]))
                else:
                    pos = (float(p[i, 0]), float(p[i, 1]))
                pygame.draw.circle(screen, (90, 160, 255), world_to_screen(pos), 2)
        except Exception:
            pass

        def draw_craft(snapshot):
            def as_xy(p):
                try:
                    # Handle numpy arrays, lists, tuples
                    x = float(p[0])
                    y = float(p[1])
                    return (x, y)
                except Exception:
                    # Fallbacks: dict with x/y or insufficient data
                    try:
                        return (float(p.get("x", 0.0)), float(p.get("y", 0.0)))
                    except Exception:
                        return (0.0, 0.0)

            pos = as_xy(snapshot.get("pos", (0.0, 0.0)))
            anc = as_xy(snapshot.get("anchor", (0.0, 0.0)))
            color = snapshot.get("color", (200, 200, 200))
            sp_anc = world_to_screen(anc)
            sp_pos = world_to_screen(pos)

            def _is_valid_screen_pt(pt):
                if not isinstance(pt, (tuple, list)) or len(pt) != 2:
                    return False
                try:
                    x, y = float(pt[0]), float(pt[1])
                    return math.isfinite(x) and math.isfinite(y)
                except Exception:
                    return False

            if not _is_valid_screen_pt(sp_anc):
                print(
                    f"[draw_craft] invalid start_pos: anc={repr(anc)} -> screen={repr(sp_anc)} types: anc={type(anc)} screen={type(sp_anc)}"
                )
                assert _is_valid_screen_pt(sp_anc), f"invalid start_pos (2D) {sp_anc}"
            if not _is_valid_screen_pt(sp_pos):
                print(
                    f"[draw_craft] invalid end_pos: pos={repr(pos)} -> screen={repr(sp_pos)} types: pos={type(pos)} screen={type(sp_pos)}"
                )
                assert _is_valid_screen_pt(sp_pos), f"invalid end_pos (2D) {sp_pos}"

            # Force plain Python ints to appease pygame argument validation
            try:
                sp_anc_i = (int(sp_anc[0]), int(sp_anc[1]))
                sp_pos_i = (int(sp_pos[0]), int(sp_pos[1]))
            except Exception as e:
                print(f"[draw_craft] cast-to-int failed: anc={repr(sp_anc)} pos={repr(sp_pos)} err={e}")
                return

            try:
                pygame.draw.line(screen, (140, 140, 180), sp_anc_i, sp_pos_i, 2)
            except TypeError as e:
                print(
                    "[draw_craft] pygame.draw.line TypeError:", e,
                    " anc=", repr(sp_anc_i), " pos=", repr(sp_pos_i),
                    " raw_anc=", repr(anc), " raw_pos=", repr(pos)
                )
                # Skip drawing this craft this frame
                return
            try:
                pygame.draw.circle(screen, color, sp_pos_i, 10)
            except TypeError as e:
                print(
                    "[draw_craft] pygame.draw.circle TypeError:", e,
                    " center=", repr(sp_pos_i), " color=", repr(color)
                )

        draw_craft(last_a)
        draw_craft(last_b)

        # Ground (thicker, higher contrast)
        gy = 0.0
        gy_px = world_to_screen((0, gy))[1]
        pygame.draw.line(screen, (120, 120, 130), (0, gy_px), (width, gy_px), 3)

        # HUD: dt + per-node metrics
        font = pygame.font.SysFont("consolas", 15)
        hud_lines = []
        hud_lines.append("Controls: WASD / Arrows; Debug: TURING_DT_DEBUG=1")
        hud_lines.append(
            ("[classic] " if classic else "[preview] ")
            + ("proj " if projection_enabled else "2D ")
            + f"Frame dt={dt*1000.0:6.1f} ms  fps_target={fps}  budget={rt_cfg.budget_ms:4.1f}ms"
        )
        if classic:
            # Top-round metrics
            top_m = runner.get_latest_metrics(top_round)
            if top_m is not None:
                hud_lines.append(
                    f"top: max_vel={top_m.max_vel:6.3f}  div_inf={top_m.div_inf:7.4f}  mass_err={top_m.mass_err:7.2e}"
                )
            # Child nodes
            for ch in nodes_for_hud:
                label = getattr(ch, "label", "node")
                m = runner.get_latest_metrics(ch)  # type: ignore[arg-type]
                if m is None:
                    continue
                hud_lines.append(
                    f"{label[:20]:20}  v={m.max_vel:6.3f}  pen={m.div_inf:7.4f}  mass={m.mass_err:6.2e}"
                )
        else:
            # Preview HUD: per-engine live stats
            alloc = compile_allocations(rt_cfg, rt_state, engine_ids)
            for reg in regs:
                name = reg.name
                ms = float(alloc.get(name, rt_cfg.ms_floor))
                ema_ms = float(rt_state.proc_ms_ma.get(name, 0.0))
                ema_pen = float(rt_state.penalty_ma.get(name, 1.0))
                hud_lines.append(
                    f"{name[:20]:20}  alloc={ms:5.2f}ms  ema_cost={ema_ms:5.2f}ms  pen={ema_pen:6.3f}  dt={dt_curr[name]:7.4f}"
                )
        # Draw HUD panel
        pad, lh = 8, 18
        box_w = 460
        box_h = lh * (len(hud_lines) + 1)
        srf = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        srf.fill((0, 0, 0, 120))
        screen.blit(srf, (10, 10))
        y = 10 + pad
        for line in hud_lines:
            txt = font.render(line, True, (235, 238, 245))
            screen.blit(txt, (10 + pad, y))
            y += lh

        pygame.display.flip()

    craft_a.system.stop(); craft_b.system.stop()
    pygame.quit()


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description="dt-graph demo: crafts + bath fluid")
    parser.add_argument("--debug", action="store_true", help="enable deep dt debug logging")
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--height", type=int, default=700)
    parser.add_argument("--fps", type=int, default=60)
    args = parser.parse_args()
    # Default to preview mode; set TURING_CLASSIC=1 to force classic if desired
    classic_env = os.environ.get("TURING_CLASSIC", "0") not in ("0", "false", "False", "no", "No", "")
    _run_demo(width=args.width, height=args.height, fps=args.fps, debug=args.debug, classic=classic_env)
