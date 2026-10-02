"""Lane FD probe: differentiation feasibility for the orbital collocation planner.

One measurement per invocation (slow disk; keep each run small):

    python tools/compiler_probes/probe_collocation_jacobian.py kepler
    python tools/compiler_probes/probe_collocation_jacobian.py sparsity
    python tools/compiler_probes/probe_collocation_jacobian.py jacobian

Results are recorded in docs/DIFFERENTIATION_FEASIBILITY_2026-10-02.md.
Probe only: edits nothing, commits nothing.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

TURING = Path(__file__).resolve().parents[2]
ROOT = TURING.parent
ENGINE_TOY = ROOT / "engine_toy"
for p in (str(TURING), str(ENGINE_TOY)):
    if p not in sys.path:
        sys.path.insert(0, p)


def stamp(msg: str, t0=[time.perf_counter()]) -> None:
    print(f"[{time.perf_counter() - t0[0]:8.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------- item 2
def measure_kepler() -> int:
    """dE/dM of the converged compiled Kepler solve vs 1/(1 - e cos E)."""
    import numpy as np
    import sympy as sp

    stamp("importing orbital_plan")
    import orbital_plan as op

    # implicit function theorem on the stage law's residual (symbolic)
    E, M, e = sp.symbols("E M e", real=True)
    R = E - e * sp.sin(E) - M
    implicit = sp.simplify(-sp.diff(R, M) / sp.diff(R, E))
    stamp(f"symbolic dE/dM = {implicit}")

    plan = op.hohmann_plan(3.986004418e14, 7.0e6, 42164.0e3,
                           t_burn1=300.0, phase=0.0)
    stamp(f"plan e_t={plan.e_transfer:.6f} T={plan.transfer_time:.3f}")
    # 64 interior samples of the transfer leg (one batch)
    t = plan.t_burn1 + plan.transfer_time * np.linspace(0.02, 0.98, 64)
    worst = {}
    for rel_step in (1e-4, 1e-5, 1e-6):
        dt = rel_step * plan.transfer_time
        cp, Ep, _ = op._solve_kepler(plan, t + dt)
        cm, Em, _ = op._solve_kepler(plan, t - dt)
        c0, E0, res0 = op._solve_kepler(plan, t)
        fd = (Ep - Em) / (cp["M"] - cm["M"])
        ecc = c0["e"]
        analytic = 1.0 / (1.0 - ecc * np.cos(E0))
        rel = np.abs(fd - analytic) / np.abs(analytic)
        worst[rel_step] = float(rel.max())
        stamp(f"step {rel_step:g} T: max rel |FD - 1/(1-e cosE)| = "
              f"{rel.max():.3e}  (median {np.median(rel):.3e}); "
              f"dE/dM range [{analytic.min():.4f}, {analytic.max():.4f}]; "
              f"max solve residual {np.max(res0):.2e}")
    print("RESULT kepler", worst)
    return 0


# ------------------------------------------------------- the collocation slice
CENTERS = (((0.0, 0.0, 0.0), 3.986004418e14),          # Earth
           ((3.844e8, 0.0, 0.0), 4.9048695e12))        # Moon
MASS = 1000.0
MAX_THRUST = 100.0e3
ALPHA, BETA, BARRIER, FUEL_BUDGET = 1.0e-4, 1.0, 1.0e3, 5.0e6


def slice_laws(k: int = 0):
    """One symplectic-Euler collocation slice spelled on the jumper's own
    law builders (actuation eq_TS1_2 -> gravity eq_N4_1 over two centers ->
    momentum eq_N1_2 -> position eq_N1_1), same xreplace as
    ``orbital_jumper_dt_pieces``.  Returns (residuals[6], cost, variables).

    Variables: r_k(3), p_k(3), u_k(6), r_k1(3), p_k1(3), dt_k.  Design
    constants (centers, mus, mass, thrust, directions, throttle box) are
    substituted as numbers, as the piece's columns would hold them.
    """
    import sympy as sp
    import honorary_engine_equation_catalogue as honorary
    import orbital_jumper as oj
    from orbital_actuation import (AXES, actuation_force_rhs, six_axis_jumper,
                                   thrust_magnitude, thruster_symbols)

    design = six_axis_jumper(MAX_THRUST, MASS)
    n_u = design.thruster_count
    dt, mass = sp.symbols("dt mass")
    position = {a: sp.Symbol(f"position_{a}") for a in AXES}
    momentum = {a: sp.Symbol(f"momentum_{a}") for a in AXES}

    constants = {mass: MASS}
    for index, (center, mu) in enumerate(CENTERS):
        csym, musym = oj._center_symbols(index)
        constants[musym] = mu
        for slot, a in enumerate(AXES):
            constants[csym[a]] = center[slot]
    for index, thruster in enumerate(design.thrusters):
        s = thruster_symbols(index)
        constants[s["max_thrust"]] = thruster.max_thrust_n
        constants[s["throttle_min"]] = thruster.throttle_min
        constants[s["throttle_max"]] = thruster.throttle_max
        for slot, a in enumerate(AXES):
            constants[s["direction"][a]] = thruster.direction[slot]

    # the pieces' laws, composed by substitution in causal order
    force = {a: oj.gravity_force_rhs(a, len(CENTERS))
             + actuation_force_rhs(a, n_u) for a in AXES}
    p_next = {a: momentum[a] + dt * honorary.eq_N1_2.rhs.xreplace(
        {honorary.F_i(honorary.t): force[a]}) for a in AXES}
    r_next = {a: position[a] + dt * honorary.eq_N1_1.rhs.xreplace(
        {honorary.m_i(honorary.t): mass, honorary.p_i(honorary.t): p_next[a]})
        for a in AXES}
    rate = sum((thrust_magnitude(i) for i in range(n_u)), sp.Integer(0))

    rk = {a: sp.Symbol(f"r{k}_{a}", real=True) for a in AXES}
    pk = {a: sp.Symbol(f"p{k}_{a}", real=True) for a in AXES}
    rk1 = {a: sp.Symbol(f"r{k + 1}_{a}", real=True) for a in AXES}
    pk1 = {a: sp.Symbol(f"p{k + 1}_{a}", real=True) for a in AXES}
    uk = [sp.Symbol(f"u{k}_{i}", real=True) for i in range(n_u)]
    dtk = sp.Symbol(f"dt{k}", positive=True)
    rename = {**{position[a]: rk[a] for a in AXES},
              **{momentum[a]: pk[a] for a in AXES},
              **{thruster_symbols(i)["throttle"]: uk[i] for i in range(n_u)},
              dt: dtk}

    def close(expr):
        return expr.xreplace(constants).xreplace(rename)

    residuals = [pk1[a] - close(p_next[a]) for a in AXES] + \
                [rk1[a] - close(r_next[a]) for a in AXES]
    fuel_rate = close(rate)
    cost = (ALPHA * fuel_rate + BETA * dtk
            - BARRIER * sp.log(FUEL_BUDGET - dtk * fuel_rate))
    variables = ([rk[a] for a in AXES] + [pk[a] for a in AXES] + uk
                 + [rk1[a] for a in AXES] + [pk1[a] for a in AXES] + [dtk])
    return residuals, cost, variables


def sample_point(variables):
    """A sample point inside the throttle box (no clamp kink)."""
    import numpy as np
    r = np.array([7.0e6, 1.0e5, 2.0e4])
    p = MASS * np.array([-100.0, 7500.0, 30.0])
    u = np.array([0.3, 0.1, 0.6, 0.2, 0.45, 0.05])
    dt = 10.0
    r1 = r + dt * p / MASS + np.array([3.0, -2.0, 1.0])
    p1 = p + np.array([-80.0, 20.0, 5.0]) * MASS * 0.0 + np.array(
        [-8.1e4, 1.2e3, 4.0e2])
    values = list(r) + list(p) + list(u) + list(r1) + list(p1) + [dt]
    return dict(zip(variables, values))


# --------------------------------------------------------------------- item 4
def _cpr_colors(pattern, order):
    """Greedy Curtis-Powell-Reid column coloring: a column takes the
    smallest color none of its row-neighbours' columns already hold."""
    rows_of = [set(pattern[c]) for c in range(len(pattern))]
    cols_in_row = {}
    for c, rows in enumerate(rows_of):
        for r in rows:
            cols_in_row.setdefault(r, []).append(c)
    color = {}
    for c in order:
        taken = {color[o] for r in rows_of[c] for o in cols_in_row[r]
                 if o in color}
        k = 0
        while k in taken:
            k += 1
        color[c] = k
    return max(color.values()) + 1


def measure_sparsity() -> int:
    import sympy as sp
    stamp("building slice laws (imports orbital_jumper)")
    res0, cost0, vars0 = slice_laws(0)
    stamp("slice built")
    # validate free-symbol structure against sympy's own Jacobian, slice 0
    J = sp.Matrix(res0).jacobian(vars0)
    sym_nnz = {(i, j) for i in range(J.rows) for j in range(J.cols)
               if J[i, j] != 0}
    fs_nnz = {(i, j) for i, row in enumerate(res0)
              for j, v in enumerate(vars0) if row.has(v)}
    stamp(f"slice: residual Jacobian {J.rows}x{J.cols}, nnz {len(sym_nnz)}; "
          f"free-symbol pattern equal: {sym_nnz == fs_nnz}")
    names = [str(v) for v in vars0]
    for i in range(J.rows):
        print("   row", i, "".join("x" if (i, j) in sym_nnz else "."
                                     for j in range(J.cols)))
    print("   cols", " ".join(names))
    cost_cols = [str(v) for v in vars0 if cost0.has(v)]
    stamp(f"cost depends on: {cost_cols}")

    for N in (1, 2, 5, 20, 100):
        # structure of slice k = slice-0 pattern with columns shifted
        # (the laws are identical per slice; only the names change)
        x_cols = 6
        u_cols = 6
        per_slice = x_cols + u_cols + 1                 # x_k, u_k, dt_k
        n_cols = per_slice * N + x_cols                 # plus x_N
        col_index = {}
        for k in range(N):
            base = per_slice * k
            col_index[k] = (list(range(base, base + 12))           # x_k,u_k
                            + list(range(base + per_slice,
                                         base + per_slice + 6))    # x_k1
                            + [base + 12])                         # dt_k
        pattern = [[] for _ in range(n_cols)]
        for k in range(N):
            for (i, j) in fs_nnz:
                pattern[col_index[k][j]].append(6 * k + i)
        bandwidth = max(abs(r - c) for c in range(n_cols) for r in pattern[c])
        natural = _cpr_colors(pattern, range(n_cols))
        degree = sorted(range(n_cols), key=lambda c: -len(pattern[c]))
        largest = _cpr_colors(pattern, degree)
        max_row = max(sum(1 for c in range(n_cols) if r in pattern[c])
                      for r in range(6 * N))
        nnz = sum(len(p) for p in pattern)
        stamp(f"N={N:4d}: J {6 * N}x{n_cols}, nnz {nnz} "
              f"({nnz / (6 * N * n_cols):.3%}), max row nnz {max_row}, "
              f"|row-col| <= {bandwidth}; CPR colors natural {natural}, "
              f"largest-first {largest}")
    return 0


def _ingest_slice():
    """sympy slice -> ProcessGraph via the repository's own ingestion
    (``ingest_sympy_expressions``, strict).  Returns everything item 1
    needs."""
    from src.compiler.symbolic_process_graph import ingest_sympy_expressions
    from src.transmogrifier.graph.graph_express2 import ProcessGraph

    residuals, cost, variables = slice_laws(0)
    exprs = list(residuals) + [cost]
    names = [f"res_{i}" for i in range(6)] + ["cost"]
    graph = ProcessGraph(materialize_memory=False)
    roots = ingest_sympy_expressions(graph, exprs, output_names=names,
                                     strict=True)
    by_name = {}
    for node, data in graph.G.nodes(data=True):
        expr = data.get("expr_obj")
        if expr is not None and expr in variables:
            by_name[str(expr)] = int(node)
    return graph, roots, exprs, variables, by_name


def measure_jacobian() -> int:
    import collections
    import numpy as np
    import sympy as sp
    from src.compiler.process_graph_autograd import (
        ProcessGraphAutogradError, differentiate_process_graph)

    stamp("building + ingesting slice")
    graph, roots, exprs, variables, by_name = _ingest_slice()
    ops = collections.Counter(str(d.get("op")) for _n, d in graph.G.nodes(data=True))
    stamp(f"forward ProcessGraph: {graph.G.number_of_nodes()} nodes; ops {dict(ops)}")
    id_name = {n: k for k, n in by_name.items()}
    for node, data in graph.G.nodes(data=True):
        if str(data.get("op")) in {"Max", "Min"}:
            print(f"   {node} {data.get('op')} parents "
                  f"{[(p, graph.G.nodes[p].get('op'), id_name.get(p, graph.G.nodes[p].get('constant'))) for p, _r in data.get('parents', ())]}")
    if "--relabel-minmax" in sys.argv:
        # SIMULATES the proposed one-line edit (not a fix): binary sympy
        # Max/Min carried as the registry's elementwise maximum/minimum
        # instead of casefolding onto the unary reductions max/min.
        for _node, data in graph.G.nodes(data=True):
            if str(data.get("op")) in {"Max", "Min"}:
                new = {"Max": "maximum", "Min": "minimum"}[str(data["op"])]
                data["op"] = data["type"] = new
        stamp("relabelled Max/Min -> maximum/minimum (probe-local)")
    missing_vars = [str(v) for v in variables if str(v) not in by_name]
    stamp(f"variable leaves found {len(by_name)}/{len(variables)}; "
          f"missing {missing_vars}; leaf ops "
          f"{sorted({graph.G.nodes[n].get('op') for n in by_name.values()})}")

    refused = []
    adjoints = {}
    for index, root in enumerate(roots):
        wrt = [by_name[str(v)] for v in variables
               if str(v) in by_name and exprs[index].has(v)]
        try:
            adjoints[index] = differentiate_process_graph(
                graph, outputs=[root], wrt=wrt)
        except ProcessGraphAutogradError as exc:
            refused.append((index, str(exc)))
            print(f"   output {index}: REFUSED: {exc}")
    stamp(f"differentiated {len(adjoints)}/{len(roots)}; refusals "
          f"{len(refused)}")
    for index, adj in adjoints.items():
        rules = adj.backward.G.graph.get("backward_rule_nodes") or {}
        print(f"   output {index}: backward nodes "
              f"{adj.backward.G.number_of_nodes()}, rules "
              f"{sorted(set(map(str, rules.values())))}")
    if "--render" in sys.argv and not refused:
        return _render_and_check(exprs, variables, by_name, adjoints)
    if "--compile" not in sys.argv:
        return 0 if not refused else 1
    return _compile_and_check(graph, roots, exprs, variables, by_name, adjoints)


def _render_and_check(exprs, variables, by_name, adjoints):
    """Numeric check of the graph adjoint WITHOUT compiling: each row's
    fused motion (unit seed) rendered by the repository's own
    ``process_graph_to_sympy_expressions`` and evaluated at the point."""
    import numpy as np
    import sympy as sp
    from src.compiler.process_graph_autograd import fuse_forward_loss_backward
    from src.compiler.symbolic_process_graph import (
        process_graph_to_sympy_expressions)

    point = sample_point(variables)
    J_sym = sp.Matrix(exprs).jacobian(variables)
    subs30 = {v: sp.Float(point[v], 30) for v in variables}
    J_ref = np.array(J_sym.evalf(30, subs=subs30).tolist(), dtype=float)
    J_graph = np.zeros_like(J_ref)
    for row, adj in adjoints.items():
        motion = fuse_forward_loss_backward(adj)
        grad_map = dict(motion.graph.G.graph["gradient_outputs"])
        gids = list(grad_map)                       # forward (variable) ids
        rendered = process_graph_to_sympy_expressions(
            motion.graph, [grad_map[g] for g in gids])
        free = set().union(*(e.free_symbols for e in rendered))
        by_str = {str(v): v for v in variables}
        unknown = [s for s in free if str(s) not in by_str]
        if unknown:
            print(f"   row {row}: rendered free symbols not variables: {unknown[:5]}")
        sub = {s: sp.Float(point[by_str[str(s)]], 30) for s in free
               if str(s) in by_str}
        id_to_col = {by_name[str(v)]: c for c, v in enumerate(variables)}
        for gid, expr in zip(gids, rendered):
            value = expr.evalf(30, subs=sub)
            if not value.is_number:
                funcs = sorted({type(f).__name__ for f in
                                value.atoms(sp.Function)})
                print(f"   row {row} grad {gid}: not numeric after subs; "
                      f"functions {funcs}; text {str(expr)[:300]}")
                return 1
            J_graph[row, id_to_col[int(gid)]] = float(value)
    nz = J_ref != 0
    rel = np.abs(J_graph - J_ref)[nz] / np.abs(J_ref)[nz]
    stray = np.abs(J_graph[~nz]).max() if (~nz).any() else 0.0
    worst = np.unravel_index(np.argmax(np.where(
        nz, np.abs(J_graph - J_ref) / np.maximum(np.abs(J_ref), 1e-300), 0)),
        J_ref.shape)
    print(f"   graph adjoint (rendered) vs sympy jacobian {J_ref.shape}: nnz "
          f"{int(nz.sum())}, max rel {rel.max():.3e}, median "
          f"{np.median(rel):.3e}; max |graph| on zeros {stray:.3e}; worst "
          f"row {worst[0]} {variables[worst[1]]}: {J_graph[worst]:.17g} vs "
          f"{J_ref[worst]:.17g}")
    print("RESULT render max_rel", float(rel.max()))
    return 0


def _compile_and_check(graph, roots, exprs, variables, by_name, adjoints):
    """One forward+backward motion over all 7 outputs with explicit seed
    inputs; native run once per one-hot seed = one Jacobian row."""
    import ctypes
    import tempfile
    import numpy as np
    import sympy as sp
    from src.compiler.process_graph_autograd import (
        differentiate_process_graph, fuse_forward_loss_backward,
        lower_training_motion_to_repository_ssa)
    from src.compiler.ssa_llvm_backend import (compile_artifact,
                                               emit_ssa_function_to_llvm)

    wrt = [by_name[str(v)] for v in variables]
    adjoint = differentiate_process_graph(graph, outputs=list(roots), wrt=wrt)
    motion = fuse_forward_loss_backward(adjoint, unit_loss_seed=False)
    stamp(f"motion: {motion.graph.G.number_of_nodes()} nodes, seeds "
          f"{dict(motion.seed_value_ids)}")
    lowering = lower_training_motion_to_repository_ssa(motion)
    stamp(f"lowered to repository SSA: shortfalls {lowering.shortfalls}; "
          f"outputs {len(lowering.outputs)}")
    llvm = emit_ssa_function_to_llvm(lowering.module, lowering.function_name,
                                     entry_name=lowering.function_name)
    stamp(f"LLVM emitted: shortfalls {llvm.shortfalls}")
    native = compile_artifact(llvm, directory=Path(tempfile.mkdtemp(
        prefix="fd_colloc_")))
    stamp(f"native built: {native.library_path}")
    entry = native.entry()

    point = sample_point(variables)
    inputs = {by_name[str(v)]: point[v] for v in variables}
    seed_of = dict(motion.seed_value_ids)
    J_native = np.zeros((len(roots), len(variables)))
    values = np.zeros(len(roots))
    for row, root in enumerate(roots):
        feed = dict(inputs)
        for r_index, r_root in enumerate(roots):
            feed[seed_of[int(r_root)]] = 1.0 if r_index == row else 0.0
        buffers = {
            vid: (np.full(shape or (), feed[vid], dtype=np.float64)
                  if vid in feed else np.zeros(shape or (), dtype=np.float64))
            for vid, shape in zip(native.buffer_order, native.buffer_shapes)}
        pointers = (ctypes.c_void_p * len(native.buffer_order))(*(
            ctypes.c_void_p(buffers[vid].ctypes.data)
            for vid in native.buffer_order))
        extents = (ctypes.c_int32 * len(native.extent_order))()
        entry(pointers, extents)
        values[row] = float(np.asarray(
            buffers[lowering.outputs[f"loss_{row}"]]).reshape(-1)[0])
        for col, v in enumerate(variables):
            key = f"grad_{by_name[str(v)]}"
            J_native[row, col] = float(np.asarray(
                buffers[lowering.outputs[key]]).reshape(-1)[0])
    stamp("native rows evaluated")

    J_sym = sp.Matrix(exprs).jacobian(variables)
    subs = {v: sp.Float(point[v], 30) for v in variables}
    J_ref = np.array(J_sym.evalf(30, subs=subs).tolist(), dtype=float)
    f_ref = np.array([float(e.evalf(30, subs=subs)) for e in exprs])
    scale = np.maximum(np.abs(J_ref), 1e-300)
    nz = J_ref != 0
    rel = np.abs(J_native - J_ref)[nz] / scale[nz]
    stray = np.abs(J_native[~nz]).max() if (~nz).any() else 0.0
    print("   forward values native vs sympy, max rel",
          float(np.max(np.abs(values - f_ref) / np.maximum(np.abs(f_ref), 1e-300))))
    print(f"   Jacobian {J_ref.shape}: nnz {int(nz.sum())}, max rel err "
          f"{rel.max():.3e}, median {np.median(rel):.3e}; max |native| on "
          f"structural zeros {stray:.3e}")
    worst = np.unravel_index(np.argmax(np.where(nz, np.abs(J_native - J_ref)
                                                / scale, 0)), J_ref.shape)
    print(f"   worst entry row {worst[0]} col {variables[worst[1]]}: native "
          f"{J_native[worst]:.17g} ref {J_ref[worst]:.17g}")
    print("RESULT jacobian max_rel", float(rel.max()))
    return 0


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "kepler"
    sys.exit({"kepler": measure_kepler, "sparsity": measure_sparsity,
              "jacobian": measure_jacobian}[which]())
