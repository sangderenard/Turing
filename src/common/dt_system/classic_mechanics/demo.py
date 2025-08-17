# -*- coding: utf-8 -*-
"""Text-mode demo: orchestrate multiple engines via dt graph → ILP schedule.

Build a small spring network with per-edge pneumatics, ground collisions,
constant gravity and optional thrusters. Use MetaLoopRunner to step one frame,
materialize a process graph via dt→nx adapter, feed ILPScheduler and print
schedule, lifespans, and a brief per-engine execution trace.
"""
from __future__ import annotations

from typing import List, Tuple

from ..dt_controller import STController, Targets
from ..dt_graph import GraphBuilder, MetaLoopRunner
from ..engine_api import EngineRegistration
from .engines import (
    DemoState,
    GravityEngine,
    ThrustersEngine,
    SpringEngine,
    PneumaticDamperEngine,
    GroundCollisionEngine,
    IntegratorEngine,
)
from ..dt_process_adapter import schedule_dt_round


def make_state() -> DemoState:
    # 3-vertex triangle over ground with two springs
    pos: List[Tuple[float, float]] = [(0.0, 0.5), (1.0, 0.6), (0.5, 1.1)]
    vel = [(0.0, 0.0) for _ in pos]
    acc = [(0.0, 0.0) for _ in pos]
    mass = [1.0, 1.0, 1.0]
    springs = [(0, 1), (1, 2), (2, 0)]
    rest_len = {(i, j): 1.0 for (i, j) in springs}
    k_spring = {(i, j): 50.0 for (i, j) in springs}
    pneu = {(i, j): (2.0, 4.0) for (i, j) in springs}
    return DemoState(pos, vel, acc, mass, springs, rest_len, k_spring, pneu)


def build_demo_round(dt: float = 0.016):
    s = make_state()

    engines = [
        ("gravity", GravityEngine(s)),
        ("thrusters", ThrustersEngine(s, thrust=(0.0, 0.0))),
        ("springs", SpringEngine(s)),
        ("pneumatics", PneumaticDamperEngine(s)),
        ("ground", GroundCollisionEngine(s)),
        ("integrate", IntegratorEngine(s)),
    ]

    regs = []
    # Use uniform targets; demo engines keep Metrics small
    targets = Targets(cfl=0.9, div_max=1e3, mass_max=1e6)
    ctrl = STController(dt_min=1e-6)
    dx = 0.1

    for name, eng in engines:
        regs.append(EngineRegistration(name=name, engine=eng, targets=targets, dx=dx))

    gb = GraphBuilder(ctrl=ctrl, targets=targets, dx=dx)
    # sequential here to demonstrate explicit dependency chain; switch to
    # interleave/parallel to change the adapter’s edges
    round_node = gb.round(dt=dt, engines=regs, schedule="sequential")

    return s, regs, round_node


def run_demo(dt: float = 0.016, steps: int = 1) -> None:
    state, regs, round_node = build_demo_round(dt=dt)

    # 1) Materialize process graph and compute schedule metadata
    levels, ig, lifespans, proc_nx = schedule_dt_round(round_node, method="asap", order="dependency")

    print("=== Process graph nodes (dt→nx) ===")
    for nid, d in proc_nx.nodes(data=True):
        print(f"  {nid}: {d.get('label')} parents={d.get('parents')} children={d.get('children')}")

    print("\n=== ASAP levels ===")
    for nid, lvl in sorted(levels.items(), key=lambda x: x[1]):
        print(f"  {proc_nx.nodes[nid].get('label')}: L{lvl}")

    print("\n=== Lifespans (start,end) ===")
    for nid, span in lifespans.items():
        print(f"  {proc_nx.nodes[nid].get('label')}: {span}")

    print("\n=== Interference edges ===")
    for u, v in ig.edges():
        print(f"  {proc_nx.nodes[u].get('label')} ↔ {proc_nx.nodes[v].get('label')}")

    # 2) Execute one or more frames via MetaLoopRunner
    runner = MetaLoopRunner()
    for k in range(steps):
        res = runner.run_round(round_node)
        print(f"\n=== Frame {k} advanced {res.advanced:.6f}s, dt_next={res.dt_next:.6f} ===")
        for reg in regs:
            print(f"  ran engine: {reg.name}")
        print("  state.pos:")
        for i, p in enumerate(state.pos):
            print(f"    v{i}: ({p[0]:.4f}, {p[1]:.4f})")


if __name__ == "__main__":
    run_demo(0.016, steps=2)
