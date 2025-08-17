from __future__ import annotations

"""
Pygame demo: dt-graph orchestration with two craft system nodes.

Crafts are built from classic mechanics engines (thrusters, springs,
pneumatics, ground, integrator) inside nested rounds controlled by the dt
graph. Each craft round is wrapped as a DtCompatibleEngine so the parent
graph schedules it like a system node. This demo intentionally avoids any
dependency on the bath fluid modules.
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

from ..dt_controller import STController, Targets, step_realtime_once
from ..dt_graph import GraphBuilder, MetaLoopRunner
from ..engine_api import EngineRegistration
from ..roundnode_engine import RoundNodeEngine
from ..threaded_system import ThreadedSystemEngine
from ..realtime import (
    RealtimeConfig,
    RealtimeState,
    compile_allocations,
    compute_penalty,
    compute_minimum_budget,
    compute_normalized_weights,
)
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
from ..solids.api import GLOBAL_WORLD, WorldPlane, MATERIAL_ELASTIC
from ..state_table import sync_engine_from_table, publish_engine_to_table
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


def build_craft(name: str, anchor: Tuple[float, float], color=(255, 200, 40), *, classic: bool, realtime_config=None, realtime_state=None) -> Craft:
    # 4-node square craft: fully connected with diagonals, equal masses
    ax, ay = float(anchor[0]), float(anchor[1])
    size = 0.8
    # Nodes in CCW order
    pos: List[Tuple[float, float]] = [
        (ax, ay),
        (ax + size, ay),
        (ax + size, ay + size),
        (ax, ay + size),
    ]
    vel = [(0.0, 0.0) for _ in pos]
    acc = [(0.0, 0.0) for _ in pos]
    mass = [1.0 for _ in pos]
    # Edges: perimeter + diagonals
    springs = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]
    rest_len = {}
    k_spring = {}
    pneu = {}
    base_k = 60.0
    for (i, j) in springs:
        dx = pos[j][0] - pos[i][0]
        dy = pos[j][1] - pos[i][1]
        L0 = float((dx * dx + dy * dy) ** 0.5)
        rest_len[(i, j)] = L0
        # Slightly softer diagonals
        k = base_k * (0.8 if abs(i - j) == 2 else 1.0)
        k_spring[(i, j)] = k
        # Directional damping (along, against)
        pneu[(i, j)] = (3.0, 5.0)
    s = DemoState(
        pos, vel, acc, mass, springs, rest_len, k_spring, pneu,
        ground_k=1200.0,
        spring_eff=0.98, thruster_eff=0.92, pneumatic_eff=0.95, linear_drag=0.15,
    )

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
        EngineRegistration(name=n, engine=e, targets=targets, dx=dx, localize=classic) for (n, e) in engines
    ]
    gb = GraphBuilder(ctrl=ctrl, targets=targets, dx=dx)
    # Use the craft name as the parent_label to ensure all node labels are unique per craft
    craft_round = gb.round(dt=0.016, engines=regs, schedule="sequential", realtime_config=realtime_config, realtime_state=realtime_state, parent_label=name)
    rne = RoundNodeEngine(craft_round)

    # Threaded system wrapper captures a snapshot for UI
    def capture():
        nodes = np.array(s.pos, dtype=float)
        edges = np.array(s.springs, dtype=int)
        # Center of mass for quick indicator
        com = nodes.mean(axis=0) if len(nodes) > 0 else np.array([0.0, 0.0])
        return {"craft": {"nodes": nodes, "edges": edges, "com": com, "color": color}}

    sys = ThreadedSystemEngine(rne, capture=capture, realtime=not classic)
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

    # Set up a simple world cage so fluids wrap within bounds and crafts hit walls
    world = GLOBAL_WORLD
    cage_min = np.array([-2.0, -1.5, -2.0], dtype=float)
    cage_max = np.array([9.0, 4.0, 2.0], dtype=float)
    world.bounds = (
        (float(cage_min[0]), float(cage_min[1]), float(cage_min[2])),
        (float(cage_max[0]), float(cage_max[1]), float(cage_max[2])),
    )
    world.fluid_mode = "wrap"
    world.planes = [
        WorldPlane(normal=np.array([0.0, 1.0, 0.0], dtype=float), offset=0.0, material=MATERIAL_ELASTIC),
        WorldPlane(normal=np.array([0.0, -1.0, 0.0], dtype=float), offset=float(cage_max[1]), material=MATERIAL_ELASTIC),
        WorldPlane(normal=np.array([1.0, 0.0, 0.0], dtype=float), offset=float(-cage_min[0]), material=MATERIAL_ELASTIC),
        WorldPlane(normal=np.array([-1.0, 0.0, 0.0], dtype=float), offset=float(cage_max[0]), material=MATERIAL_ELASTIC),
        WorldPlane(normal=np.array([0.0, 0.0, 1.0], dtype=float), offset=float(-cage_min[2]), material=MATERIAL_ELASTIC),
        WorldPlane(normal=np.array([0.0, 0.0, -1.0], dtype=float), offset=float(cage_max[2]), material=MATERIAL_ELASTIC),
    ]

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

    # Realtime state --------------------------------------------------------
    # Default: realtime mode ON; use --classic to fall back to superstep graph.
    rt_cfg = RealtimeConfig(budget_ms=1000.0 / max(fps, 1), slack=0.92, beta=1.0, w_floor=0.25, ms_floor=0.25)
    rt_state = RealtimeState()

    # Build crafts, passing the root's realtime config/state
    craft_a = build_craft("A", anchor=(2.5, 3.0), color=(240, 80, 80), classic=classic, realtime_config=rt_cfg, realtime_state=rt_state)
    craft_b = build_craft("B", anchor=(6.5, 3.2), color=(80, 180, 255), classic=classic, realtime_config=rt_cfg, realtime_state=rt_state)

    # Bath discrete fluid: construct via engine convenience (n) and derive bounds from world planes
    fluid_engine = BathDiscreteFluidEngine(n=100, world=world)

    # Top-level graph: gravity -> meta-collision(A,B) -> craft A (system) -> craft B (system) -> fluid
    targets = Targets(cfl=0.9, div_max=1e3, mass_max=1e6)
    ctrl = STController(dt_min=1e-6)
    dx = 0.1
    # Build meta collision engine that internally consults the spring/damper networks
    meta_collision = MetaCollisionEngine([craft_a.state, craft_b.state], restitution=0.25, friction_mu=0.6, body_radius=0.12)
    # Configure bisect solver to drive penetration (encoded in Metrics.div_inf) toward 0 within epsilon
    solver_cfg = BisectSolverConfig(target=0.0, eps=1e-5, field="div_inf", monotonic="increase", dt_min=1e-6)

    # To use the rigid body engine as a constraint/finalizer, add it as the last engine in your engine registration list:
    from .rigid_body_engine import RigidBodyEngine, WorldAnchor, WorldObjectLink, COM
    world_anchors = [WorldAnchor(position=(0.0, 0.0))]
    object_anchors = [(0, "craft_a", COM, 0.0),
                      (0, "craft_b", COM, 0.0)]#(world_anchor_index, vertex_set_identifier, set_index, mass)

    links = [WorldObjectLink(world_anchor=world_anchors[object_anchors[i][0]], object_anchor=object_anchors[i], link_type='steel_beam', properties={'length': 1.0, 'k': 10000.0}) for i in range(len(object_anchors))]
    rigid_engine = RigidBodyEngine(links)

    regs: List[EngineRegistration] = [
        EngineRegistration(name="gravity", engine=GravityEngine(craft_a.state), targets=targets, dx=dx, localize=False),
        EngineRegistration(name="collision", engine=meta_collision, targets=targets, dx=dx, localize=False, solver_config=solver_cfg),
        EngineRegistration(name="craftA", engine=craft_a.system, targets=targets, dx=dx, localize=False),
        EngineRegistration(name="craftB", engine=craft_b.system, targets=targets, dx=dx, localize=False),
        EngineRegistration(
            name="rigid_body_constraint",
            engine=rigid_engine,
            targets=targets,
            dx=dx,
            localize=False
        ),
        EngineRegistration(name="fluid", engine=fluid_engine, targets=targets, dx=dx, localize=True),
    ]

    # Realtime state --------------------------------------------------------
    # Default: realtime mode ON; use --classic to fall back to superstep graph.
    rt_cfg = RealtimeConfig(budget_ms=1000.0 / max(fps, 1), slack=0.92, beta=1.0, w_floor=0.25, ms_floor=0.25)
    rt_state = RealtimeState()
    gb = GraphBuilder(ctrl=ctrl, targets=targets, dx=dx)
    top_round = gb.round(dt=1.0 / fps, engines=regs, schedule="sequential", realtime_config=rt_cfg, realtime_state=rt_state)

    # --- StateTable for dt_tape HUD wiring and metaloop constructor---
    from ..state_table import StateTable
    state_table = StateTable()
    runner = MetaLoopRunner(realtime_config=rt_cfg, realtime_state=rt_state, realtime=not classic, state_table=state_table)
    runner.set_process_graph(top_round, schedule_method="asap", schedule_order="dependency")


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


    engine_ids = [reg.name for reg in regs]
    # Per-engine controllers (independent PI state) and current dt trackers
    per_ctrl: dict[str, STController] = {}
    dt_curr: dict[str, float] = {}
    # Per-engine mutable state passed to realtime step (engines with step_with_state can use this)
    per_state: dict[str, dict] = {}
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
        # Start realtime dt at budget rather than controller dt to avoid tiny values
        dt_curr[reg.name] = 1.0 / max(fps, 1)
        # Initialize an empty state dict; engines that don't use it will ignore it
        per_state[reg.name] = {}

    last_a = {"pos": np.array(craft_a.state.pos[1], dtype=float), "anchor": np.array(craft_a.state.pos[0], dtype=float), "color": (240, 80, 80)}
    last_b = {"pos": np.array(craft_b.state.pos[1], dtype=float), "anchor": np.array(craft_b.state.pos[0], dtype=float), "color": (80, 180, 255)}


    running = True
    paused = True  # Start paused
    reset_requested = False

    def reset_scene():
        nonlocal craft_a, craft_b, fluid_engine, regs, gb, top_round, runner, nodes_for_hud, last_a, last_b, state_table
        # Rebuild crafts and engines
        craft_a = build_craft("A", anchor=(2.5, 3.0), color=(240, 80, 80), classic=classic)
        craft_b = build_craft("B", anchor=(6.5, 3.2), color=(80, 180, 255), classic=classic)
        fluid_engine = BathDiscreteFluidEngine(n=100, world=world)
        meta_collision = MetaCollisionEngine([craft_a.state, craft_b.state], restitution=0.25, friction_mu=0.6, body_radius=0.12)
        solver_cfg = BisectSolverConfig(target=0.0, eps=1e-5, field="div_inf", monotonic="increase", dt_min=1e-6)
        regs = [
            EngineRegistration(name="gravity", engine=GravityEngine(craft_a.state), targets=targets, dx=dx, localize=False),
            EngineRegistration(name="collision", engine=meta_collision, targets=targets, dx=dx, localize=False, solver_config=solver_cfg),
            EngineRegistration(name="craftA", engine=craft_a.system, targets=targets, dx=dx, localize=False),
            EngineRegistration(name="craftB", engine=craft_b.system, targets=targets, dx=dx, localize=False),
            EngineRegistration(name="fluid", engine=fluid_engine, targets=targets, dx=dx, localize=True),
        ]
        gb = GraphBuilder(ctrl=ctrl, targets=targets, dx=dx)
        top_round = gb.round(dt=1.0 / fps, engines=regs, schedule="sequential")
        from ..state_table import StateTable
        state_table = StateTable()
        runner = MetaLoopRunner(state_table=state_table)
        runner.set_process_graph(top_round, schedule_method="asap", schedule_order="dependency")
        def _flatten_nodes(r):
            out = []
            for ch in r.children:
                out.append(ch)
                if hasattr(ch, "children"):
                    for gch in ch.children:
                        out.append(gch)
            return out
        nodes_for_hud = _flatten_nodes(top_round)
        last_a = {"pos": np.array(craft_a.state.pos[1], dtype=float), "anchor": np.array(craft_a.state.pos[0], dtype=float), "color": (240, 80, 80)}
        last_b = {"pos": np.array(craft_b.state.pos[1], dtype=float), "anchor": np.array(craft_b.state.pos[0], dtype=float), "color": (80, 180, 255)}

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RETURN:
                    reset_requested = True

        if reset_requested:
            reset_scene()
            reset_requested = False
            paused = True

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
        if not paused:
            if classic:
                from ..dt import SuperstepPlan
                saved = top_round.plan
                top_round.plan = SuperstepPlan(round_max=float(dt), dt_init=max(float(dt), 1e-6))
                try:
                    _res = runner.run_round()
                finally:
                    top_round.plan = saved
            else:
                
                _res = runner.run_round()

        # Drain craft frames (always, so rendering is up to date)
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

        # Rendering and HUD should always run, regardless of pause
        screen.fill((8, 9, 12))  # slightly cooler dark bg for contrast

        # Fluid particles
        try:
            p = fluid_engine.sim.export_vertices()  # type: ignore[union-attr]
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
            color = snapshot.get("color", (200, 200, 200))
            nodes = snapshot.get("nodes")
            edges = snapshot.get("edges")
            com = snapshot.get("com")

            def _is_valid_screen_pt(pt):
                if not isinstance(pt, (tuple, list)) or len(pt) != 2:
                    return False
                try:
                    x, y = float(pt[0]), float(pt[1])
                    return math.isfinite(x) and math.isfinite(y)
                except Exception:
                    return False

            if nodes is not None and edges is not None:
                # Draw edges and nodes
                try:
                    for (i, j) in edges:
                        pi = world_to_screen(nodes[int(i)])
                        pj = world_to_screen(nodes[int(j)])
                        pygame.draw.line(screen, (140, 140, 180), (int(pi[0]), int(pi[1])), (int(pj[0]), int(pj[1])), 2)
                    for p in nodes:
                        sp = world_to_screen(p)
                        pygame.draw.circle(screen, color, (int(sp[0]), int(sp[1])), 7)
                    if com is not None:
                        spc = world_to_screen(com)
                        pygame.draw.circle(screen, (250, 240, 240), (int(spc[0]), int(spc[1])), 4)
                except Exception:
                    pass
                return

            # Back-compat: draw a tethered point if only pos/anchor provided
            def as_xy(p):
                try:
                    x = float(p[0]); y = float(p[1])
                    return (x, y)
                except Exception:
                    try:
                        return (float(p.get("x", 0.0)), float(p.get("y", 0.0)))
                    except Exception:
                        return (0.0, 0.0)

            pos = as_xy(snapshot.get("pos", (0.0, 0.0)))
            anc = as_xy(snapshot.get("anchor", (0.0, 0.0)))
            sp_anc = world_to_screen(anc)
            sp_pos = world_to_screen(pos)
            if _is_valid_screen_pt(sp_anc) and _is_valid_screen_pt(sp_pos):
                pygame.draw.line(screen, (140, 140, 180), (int(sp_anc[0]), int(sp_anc[1])), (int(sp_pos[0]), int(sp_pos[1])), 2)
                pygame.draw.circle(screen, color, (int(sp_pos[0]), int(sp_pos[1])), 10)

        draw_craft(last_a)
        draw_craft(last_b)

        # Ground (thicker, higher contrast)
        gy = 0.0
        gy_px = world_to_screen((0, gy))[1]
        pygame.draw.line(screen, (120, 120, 130), (0, gy_px), (width, gy_px), 3)

        # HUD: dt + per-node metrics
        font = pygame.font.SysFont("consolas", 15)
        hud_lines = []
        hud_lines.append("Controls: SPACE=play/pause, ENTER=reset, WASD / Arrows; Debug: TURING_DT_DEBUG=1")
        if classic:
            hud_lines.append(
                ("[classic] " if classic else "[realtime] ")
                + ("proj " if projection_enabled else "2D ")
                + f"Frame dt={dt*1000.0:6.1f} ms  fps_target={fps}  budget={rt_cfg.budget_ms:4.1f}ms"
            )
        else:
            hud_lines.append(
                ("[classic] " if classic else "[realtime] ")
                + ("proj " if projection_enabled else "2D ")
                + f"Frame dt={dt*1000.0:6.1f} ms  fps_target={fps}  budget={rt_cfg.budget_ms:4.1f}ms"
            )
            # Show minimum budget vs target and slack weights summary
            try:
                _, min_total_ms = compute_minimum_budget(rt_cfg, rt_state, engine_ids)
                hud_lines.append(
                    f"min_budget={min_total_ms:5.2f}ms  target_pool={rt_cfg.slack*rt_cfg.budget_ms:5.2f}ms"
                )
            except Exception:
                pass
        # --- HUD wiring: always use StateTable's dt_tape for per-node data ---
        def get_dt_tape(label, field):
            return state_table.get('dt_tape', label, field)

        if classic:
            # Top-round metrics from dt_tape
            top_label = getattr(top_round, 'label', None)
            if top_label:
                m = get_dt_tape(top_label, 'metrics')
                if m is not None:
                    hud_lines.append(
                        f"top: max_vel={getattr(m, 'max_vel', 0.0):6.3f}  div_inf={getattr(m, 'div_inf', 0.0):7.4f}  mass_err={getattr(m, 'mass_err', 0.0):7.2e}"
                    )
            # Child nodes
            for ch in nodes_for_hud:
                label = getattr(ch, "label", "node")
                m = get_dt_tape(label, 'metrics')
                if m is None:
                    continue
                hud_lines.append(
                    f"{label[:20]:20}  v={getattr(m, 'max_vel', 0.0):6.3f}  pen={getattr(m, 'div_inf', 0.0):7.4f}  mass={getattr(m, 'mass_err', 0.0):6.2e}"
                )
        else:
            # Realtime HUD: per-engine live stats from dt_tape
            alloc = compile_allocations(rt_cfg, rt_state, engine_ids)
            for reg in regs:
                name = reg.name
                label = f"advance:{name}"
                ms = float(alloc.get(name, rt_cfg.ms_floor))
                ema_ms = float(rt_state.proc_ms_ma.get(name, 0.0))
                ema_pen = float(rt_state.penalty_ma.get(name, 1.0))
                dt_val = get_dt_tape(label, 'dt')
                if dt_val is not None:
                    dt_str = f"{float(dt_val)*1000.0:6.3f}ms"
                else:
                    dt_str = "N/A"
                hud_lines.append(
                    f"{name[:20]:20}  alloc={ms:5.2f}ms  ema_cost={ema_ms:5.2f}ms  pen={ema_pen:6.3f}  dt={dt_str}"
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




# This ensures the rigid body constraint is enforced after all other steps.
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
