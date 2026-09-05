"""Audit: raw vs control-discounted ancestry on a real retained-loop graph.

Reuses tools/repro_step_with_dt_control_used.py's source assembly, but stops
the pipeline right after ``reduce_scheduled_shader_regions`` runs on the
first graph that carries a ``recursion_table`` (the retained ``while True``
retry loop of ``step_with_dt_control_used``).  Nothing under src/ is edited;
the scheduler is wrapped in-process only.  Read-only analysis follows.
"""

from __future__ import annotations

import ast
import inspect
import sys
import time
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.compiler.glsl_deployment_strategy as gds  # noqa: E402
import src.compiler.loop_composer as lc  # noqa: E402
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa  # noqa: E402
from src.compiler.extraction_contract import ExtractionContract  # noqa: E402
from src.compiler.vehicle_python_compilation import (  # noqa: E402
    balloon_tire_managed_extraction_contract, BalloonTireManagedState,
)
from src.common.dt_system.dt_controller import (  # noqa: E402
    _restore_type, Targets, _shadow_dt_limit, _energy_time_limit,
    _no_exchange_observed, _apply_energy_sidechain, STController,
    _propose_dt_pen, step_with_dt_control_used,
)
from src.common.dt_system.dt_scaler import Metrics, _scalar, coerce_metrics  # noqa: E402
from src.common.dt_system.shadow import shadow_dt_limit  # noqa: E402

CONTRACTS = Path(__file__).resolve().parents[1] / "extraction_contracts"


class _Captured(Exception):
    pass


CAPTURE: dict = {}
PRE_STATES: list = []


def _base_records():
    import numpy as np

    stub = BalloonTireManagedState.__new__(BalloonTireManagedState)
    for name in (
        "inputs", "state", "output", "wheel_input_indices", "rest",
        "face_vertices", "face_rest", "face_scatter", "bending_incidence",
        "bending_scatter", "bending_weight", "vertex_area", "bead_mask",
        "face_material", "telemetry",
    ):
        setattr(stub, name, np.zeros((1,), dtype=np.float64))
    return balloon_tire_managed_extraction_contract(stub).program_abi.receipt()


def _install_probes():
    original_reduce = gds.reduce_scheduled_shader_regions

    def probe_reduce(graph, executable_node_ids, **kwargs):
        plan = original_reduce(graph, executable_node_ids, **kwargs)
        if graph.G.graph.get("recursion_table"):
            CAPTURE["graph"] = graph
            CAPTURE["executable"] = tuple(executable_node_ids)
            CAPTURE["control_node_ids"] = frozenset(
                kwargs.get("control_node_ids", ())
            )
            CAPTURE["plan"] = plan
            CAPTURE["partition_keys"] = kwargs.get("partition_keys")
            raise _Captured()
        return plan

    gds.reduce_scheduled_shader_regions = probe_reduce

    original_evaporate = lc.evaporate_unrolled_loops

    def probe_evaporate(graph, plans):
        plans = tuple(plans)
        PRE_STATES.append((
            "evaporate_entry",
            graph.G.graph.get("function_name"),
            nx.is_directed_acyclic_graph(graph.G),
            bool(graph.G.graph.get("recursion_table")),
            tuple(str(p.strategy) for p in plans),
        ))
        return original_evaporate(graph, plans)

    gds.evaporate_unrolled_loops = probe_evaporate

    original_materialize = lc.materialize_retained_loop_ports

    def probe_materialize(graph, plans):
        plans = tuple(plans)
        PRE_STATES.append((
            "materialize_entry",
            graph.G.graph.get("function_name"),
            nx.is_directed_acyclic_graph(graph.G),
            bool(graph.G.graph.get("recursion_table")),
            tuple(str(p.strategy) for p in plans),
        ))
        result = original_materialize(graph, plans)
        CAPTURE.setdefault("loop_plans", {})[
            graph.G.graph.get("function_name")
        ] = result
        return result

    gds.materialize_retained_loop_ports = probe_materialize


def _source() -> str:
    real_source = "\n\n".join((
        inspect.getsource(_scalar),
        inspect.getsource(coerce_metrics),
        inspect.getsource(_restore_type),
        inspect.getsource(shadow_dt_limit),
        inspect.getsource(_shadow_dt_limit),
        inspect.getsource(_energy_time_limit),
        inspect.getsource(_no_exchange_observed),
        inspect.getsource(_apply_energy_sidechain),
        inspect.getsource(_propose_dt_pen),
        inspect.getsource(step_with_dt_control_used),
    ))
    advance_source = (
        "def advance(state, dt):\n"
        "    return True, Metrics(\n"
        "        max_vel=float(state.state[0]),\n"
        "        max_flux=float(state.state[0]),\n"
        "        div_inf=0.0,\n"
        "        mass_err=0.0,\n"
        "    )\n"
    )
    root_source = (
        "def root(state, dt, dx, targets, ctrl):\n"
        "    return step_with_dt_control_used(\n"
        "        state, dt, dx, targets, ctrl, advance,\n"
        "        rollback=True,\n"
        "    )\n"
    )
    return real_source + "\n\n" + advance_source + "\n\n" + root_source


def _describe(G, node_id):
    data = G.nodes[node_id]
    expression = data.get("expr_obj")
    return (
        f"{node_id}:{data.get('type') or data.get('op')}"
        f"@{getattr(expression, 'lineno', '-')}"
        f"{'/' + type(expression).__name__ if expression is not None else ''}"
    )


def analyse() -> None:
    graph = CAPTURE["graph"]
    G = graph.G
    fn = G.graph.get("function_name")
    control = CAPTURE["control_node_ids"]
    print(f"\n=== captured graph fn={fn} nodes={G.number_of_nodes()} "
          f"edges={G.number_of_edges()} dag={nx.is_directed_acyclic_graph(G)}")
    print("pre-states:", *PRE_STATES, sep="\n  ")
    table = G.graph.get("recursion_table") or {}
    for key, record in table.items():
        print(f"recursion_table[{key}] members={len(record['members'])} "
              f"control_members={[_describe(G, n) for n in record['control_members']]} "
              f"feedback={len(record['feedback'])}")
    discounted = G.copy()
    discounted.remove_edges_from(tuple(
        (l, r) for l, r in discounted.edges if l in control or r in control
    ))
    print(f"discounted dag={nx.is_directed_acyclic_graph(discounted)} "
          f"removed={G.number_of_edges() - discounted.number_of_edges()} edges")
    sccs = [c for c in nx.strongly_connected_components(G) if len(c) > 1]
    for scc in sccs:
        types = {}
        for n in scc:
            t = str(G.nodes[n].get("type") or G.nodes[n].get("op"))
            types[t] = types.get(t, 0) + 1
        print(f"SCC size={len(scc)} types={types}")
        # which edges of the SCC are incident to control nodes
        internal = [(l, r) for l, r in G.edges if l in scc and r in scc]
        ctrl_edges = [(l, r) for l, r in internal if l in control or r in control]
        print(f"  internal edges={len(internal)} control-incident={len(ctrl_edges)}")
        for l, r in ctrl_edges[:12]:
            print(f"    {_describe(G, l)} -> {_describe(G, r)} role={G.edges[l, r].get('role')}")

    scc_nodes = set().union(*sccs) if sccs else set()

    # --- Site A: _build_shell_hierarchy_plan has_path(call, definition)
    identities = G.graph.get("identity_table") or {}
    print("\n--- site A: nx.has_path(G, call, definition) wrap-around (glsl 2054/2172)")
    flips = 0
    for call_id, data in G.nodes(data=True):
        attributes = data.get("attributes") or {}
        if attributes.get("callee_ref") is None and attributes.get("method_ref") is None:
            continue
        if call_id not in scc_nodes:
            continue
        call_line = getattr(data.get("expr_obj"), "lineno", None)
        for name, history in identities.items():
            for definition in history:
                definition = int(definition)
                if definition not in G or definition == call_id:
                    continue
                d_line = getattr(G.nodes[definition].get("expr_obj"), "lineno", None)
                raw = nx.has_path(G, call_id, definition)
                disc = nx.has_path(discounted, call_id, definition)
                if raw and not disc and d_line is not None and call_line is not None and d_line < call_line:
                    flips += 1
                    if flips <= 8:
                        print(f"  call {_describe(G, call_id)} name={name!r} "
                              f"preceding def {_describe(G, definition)}: raw has_path=True, discounted=False")
    print(f"  total preceding definitions wrongly excluded by raw has_path: {flips}")

    # --- Site B: _ordinary_conditional_control_programs ancestors(predicate)
    print("\n--- site B: nx.ancestors(G, predicate) for source ifs (glsl 7167)")
    memberships = gds._branch_compartments(graph)
    for control_id, record in gds._source_control_records(G).items():
        expression = record.get("expression")
        if not isinstance(expression, ast.If):
            continue
        predicate_id = gds._retained_control_value_id(G, record.get("predicate_id"), expression.test)
        if predicate_id is None or predicate_id not in scc_nodes:
            continue
        scope = memberships.get(int(predicate_id), frozenset())
        raw = {int(c) for c in nx.ancestors(G, int(predicate_id)) if memberships.get(int(c), frozenset()) == scope}
        disc = {int(c) for c in nx.ancestors(discounted, int(predicate_id)) if memberships.get(int(c), frozenset()) == scope}
        extra = raw - disc
        end = getattr(expression, "end_lineno", expression.lineno)
        later = [n for n in extra if (getattr(G.nodes[n].get("expr_obj"), "lineno", -1) or -1) > end]
        print(f"  if@{expression.lineno}-{end} predicate={_describe(G, predicate_id)} "
              f"same-scope ancestors raw={len(raw)} discounted={len(disc)} "
              f"extra={len(extra)} of which lexically AFTER the if: {len(later)}")
        for n in sorted(later)[:6]:
            print(f"      after-if node pulled into predicate prelude: {_describe(G, n)}")

    # --- Site C: _control_partition_keys descendants(callsite)
    print("\n--- site C: nx.descendants(G, callsite) partition keys (glsl 7629)")
    for call_id, data in G.nodes(data=True):
        attributes = data.get("attributes") or {}
        if attributes.get("callee_ref") is None and attributes.get("method_ref") is None:
            continue
        if call_id not in scc_nodes:
            continue
        raw = nx.descendants(G, call_id)
        disc = nx.descendants(discounted, call_id)
        line = getattr(data.get("expr_obj"), "lineno", None)
        before = [n for n in raw - disc if line is not None and (getattr(G.nodes[n].get("expr_obj"), "lineno", None) or 10**9) < line]
        print(f"  call {_describe(G, call_id)}: descendants raw={len(raw)} discounted={len(disc)} "
              f"extra={len(raw - disc)}; extra nodes lexically BEFORE the call: {len(before)}")

    # --- Site D: _dependency_order fallback vs discounted DAG order
    print("\n--- site D: _dependency_order fallback (glsl 152) vs discounted DAG")
    order = gds._dependency_order(graph)
    index = {n: i for i, n in enumerate(order)}
    violations = [(l, r) for l, r in discounted.edges if index[l] > index[r]]
    print(f"  order length={len(order)} discounted edges violated by fallback order: {len(violations)}")
    for l, r in violations[:8]:
        print(f"    {_describe(G, l)} (pos {index[l]}) -> {_describe(G, r)} (pos {index[r]})")

    # --- Site E: coordinator body cone (glsl 9222/9231)
    print("\n--- site E: coordinator invalidation cone (glsl 9222-9234)")
    for plans in (CAPTURE.get("loop_plans") or {}).values():
        for plan in plans:
            loop_id = int(plan.loop.node_id)
            if loop_id not in G:
                continue
            body = {int(b) for b in plan.loop.body_nodes if int(b) in G}
            raw_inv = set(body)
            for b in body:
                raw_inv.update(nx.descendants(G, b))
            invariant = set(nx.ancestors(G, loop_id)) - body
            after_rule = raw_inv - invariant
            disc_inv = set(body)
            for b in body:
                disc_inv.update(nx.descendants(discounted, b))
            scc_here = next((s for s in sccs if loop_id in s), set())
            print(f"  loop {_describe(G, loop_id)} strategy={plan.strategy} body={len(body)} "
                  f"scc={len(scc_here)} scc-not-body={len(scc_here - body)} "
                  f"raw_cone={len(raw_inv)} 'invariant_ancestors'={len(invariant)} "
                  f"(of which in SCC: {len(invariant & scc_here)}) "
                  f"after_rule={len(after_rule)} discounted_cone={len(disc_inv)}")
            print(f"    SCC nodes un-invalidated by the invented rule: "
                  f"{[_describe(G, n) for n in sorted(invariant & scc_here)][:10]}")

    # --- Site F: loop_composer 3487 region dependency parent walk
    print("\n--- site F: region-dependency parent walk (loop_composer 3487)")
    plan = CAPTURE["plan"]
    regions = [tuple(int(n) for n in d.node_ids) for d in plan.dispatches]
    owner = {n: i for i, r in enumerate(regions) for n in r}
    def region_graph(stop_at_control: bool):
        deps = nx.DiGraph()
        deps.add_nodes_from(range(len(regions)))
        for i, r in enumerate(regions):
            for n in r:
                pending = [int(p) for p, _ in (G.nodes[n].get("parents") or ())]
                seen = set()
                while pending:
                    p = pending.pop()
                    if p in seen or p not in G:
                        continue
                    seen.add(p)
                    if stop_at_control and p in control:
                        continue
                    o = owner.get(p)
                    if o is not None:
                        if o != i:
                            deps.add_edge(o, i)
                        continue
                    pending.extend(int(gp) for gp, _ in (G.nodes[p].get("parents") or ()))
        return deps
    raw_deps = region_graph(False)
    disc_deps = region_graph(True)
    print(f"  regions={len(regions)} raw region DAG={nx.is_directed_acyclic_graph(raw_deps)} "
          f"discounted region DAG={nx.is_directed_acyclic_graph(disc_deps)}")
    if not nx.is_directed_acyclic_graph(raw_deps):
        cycles = list(nx.simple_cycles(raw_deps))
        print(f"  raw region cycles: {cycles[:5]}")
        def earliest(i):
            return min((getattr(G.nodes[n].get('expr_obj'), 'lineno', 10**9) or 10**9, n) for n in regions[i])
        condensed = nx.condensation(raw_deps)
        cond_order = [m for c in nx.lexicographical_topological_sort(
            condensed, key=lambda c: min(earliest(m) for m in condensed.nodes[c]['members']))
            for m in sorted(condensed.nodes[c]['members'], key=earliest)]
        pos = {r: i for i, r in enumerate(cond_order)}
        bad = [(a, b) for a, b in disc_deps.edges if pos[a] > pos[b]]
        print(f"  condensation order violates {len(bad)} same-iteration region dependencies: {bad[:8]}")


def main() -> int:
    _install_probes()
    base = _base_records()
    contract_abi = {
        "records": {
            "Metrics": base["records"]["Metrics"],
            "Targets": base["records"]["Targets"],
            "STController": base["records"]["STController"],
            "BalloonTireManagedState": base["records"]["BalloonTireManagedState"],
        },
        "bindings": [
            {"function": "*", "parameter": "metrics", "record": "Metrics"},
            {"function": "*", "parameter": "targets", "record": "Targets"},
            {"function": "*", "parameter": "ctrl", "record": "STController"},
            {"function": "*", "parameter": "state", "record": "BalloonTireManagedState"},
        ],
        "values": [],
    }
    policy = ExtractionContract(CONTRACTS / "program_extraction.yaml").with_program_abi(contract_abi)
    t0 = time.time()
    try:
        lower_ast_source_to_ssa(_source(), "root", name="step_used", extraction_contract=policy)
        print(f"no retained-loop graph reached the scheduler ({time.time()-t0:.1f}s)")
        return 1
    except _Captured:
        print(f"captured after {time.time()-t0:.1f}s")
    analyse()
    analyse_more()
    analyse_bindings()
    return 0



def analyse_more() -> None:
    graph = CAPTURE["graph"]
    G = graph.G
    control = CAPTURE["control_node_ids"]
    discounted = G.copy()
    discounted.remove_edges_from(tuple(
        (l, r) for l, r in discounted.edges if l in control or r in control
    ))
    sccs = [c for c in nx.strongly_connected_components(G) if len(c) > 1]
    scc = set().union(*sccs) if sccs else set()
    plans = (CAPTURE.get("loop_plans") or {}).get(G.graph.get("function_name"), ())
    print("\n--- SCC member detail")
    body_by_loop = {int(p.loop.node_id): {int(b) for b in p.loop.body_nodes} for p in plans}
    for n in sorted(scc):
        d = G.nodes[n]
        a = d.get("attributes") or {}
        owners = [l for l, b in body_by_loop.items() if n in b]
        print(f"  {_describe(G, n)} expr_obj={type(d.get('expr_obj')).__name__} "
              f"source_type={a.get('source_type')} region={a.get('recursion_region_id')} "
              f"parents={[(int(p), r) for p, r in d.get('parents') or ()]} in_body_of={owners}")
    print("\n--- broadened site B: every source-if predicate, raw vs discounted same-scope ancestors")
    memberships = gds._branch_compartments(graph)
    changed = 0
    for control_id, record in gds._source_control_records(G).items():
        expression = record.get("expression")
        if not isinstance(expression, ast.If):
            continue
        predicate_id = gds._retained_control_value_id(G, record.get("predicate_id"), expression.test)
        if predicate_id is None:
            continue
        scope = memberships.get(int(predicate_id), frozenset())
        raw = {int(c) for c in nx.ancestors(G, int(predicate_id)) if memberships.get(int(c), frozenset()) == scope}
        disc = {int(c) for c in nx.ancestors(discounted, int(predicate_id)) if memberships.get(int(c), frozenset()) == scope}
        if raw != disc:
            changed += 1
            end = getattr(expression, "end_lineno", expression.lineno)
            lost = disc - raw
            extra = raw - disc
            later = [n for n in extra if (getattr(G.nodes[n].get("expr_obj"), "lineno", None) or -1) > end]
            print(f"  if@{expression.lineno}-{end} pred={_describe(G, predicate_id)} in_scc={predicate_id in scc} "
                  f"raw={len(raw)} disc={len(disc)} raw-only={len(extra)} (after-if: {len(later)}) disc-only={len(lost)}")
    print(f"  predicates whose same-scope ancestry differs: {changed}")
    print("\n--- broadened site C: every callsite, raw vs discounted descendants")
    for call_id, data in G.nodes(data=True):
        a = data.get("attributes") or {}
        if a.get("callee_ref") is None and a.get("method_ref") is None:
            continue
        raw = nx.descendants(G, call_id)
        disc = nx.descendants(discounted, call_id)
        if raw != disc:
            line = getattr(data.get("expr_obj"), "lineno", None)
            before = [n for n in raw - disc if line is not None and (getattr(G.nodes[n].get("expr_obj"), "lineno", None) or 10**9) < line]
            print(f"  call {_describe(G, call_id)} in_scc={call_id in scc} raw={len(raw)} disc={len(disc)} "
                  f"raw-only={len(raw - disc)} (lexically before call: {len(before)}) ")
    print("\n--- site E detail: are the un-invalidated SCC effect nodes reachable as 'controlled' by any other rule?")
    loop_inputs = [n for n, d in G.nodes(data=True) if d.get("type") == "Input" and (d.get("attributes") or {}).get("binding_kind") in {"loop", "exception"}]
    via_binding = set()
    for b in loop_inputs:
        via_binding.update(nx.descendants(G, b))
    for n in sorted(scc):
        if n in control:
            continue
        print(f"  {_describe(G, n)}: descendant of a loop/exception Input={n in via_binding}; "
              f"has expr_obj={G.nodes[n].get('expr_obj') is not None}")

def analyse_bindings() -> None:
    graph = CAPTURE["graph"]
    G = graph.G
    control = CAPTURE["control_node_ids"]
    discounted = G.copy()
    discounted.remove_edges_from(tuple(
        (l, r) for l, r in discounted.edges if l in control or r in control
    ))
    sccs = [c for c in nx.strongly_connected_components(G) if len(c) > 1]
    scc = set().union(*sccs) if sccs else set()
    plans = (CAPTURE.get("loop_plans") or {}).get(G.graph.get("function_name"), ())
    print("\n--- site G: per-loop target-binding descendants (glsl 9151 / 10673) raw vs discounted")
    for plan in plans:
        for name, binding in plan.loop.target_bindings:
            binding = int(binding)
            if binding not in G:
                continue
            raw = nx.descendants(G, binding)
            disc = nx.descendants(discounted, binding)
            extra_nodes = raw - disc
            print(f"  loop {_describe(G, int(plan.loop.node_id))} target {name!r}={_describe(G, binding)}: "
                  f"raw={len(raw)} disc={len(disc)} raw-only={len(extra_nodes)} "
                  f"raw-only SCC effect nodes={[_describe(G, n) for n in sorted(extra_nodes & scc)]}")
    print("\n--- site H: downstream_loop_nodes (glsl 9269) raw vs discounted")
    loop_ids = [int(p.loop.node_id) for p in plans if int(p.loop.node_id) in G]
    for cand in loop_ids:
        raw_owners = [o for o in loop_ids if o != cand and o in nx.ancestors(G, cand)]
        disc_owners = [o for o in loop_ids if o != cand and o in nx.ancestors(discounted, cand)]
        if raw_owners != disc_owners:
            print(f"  {_describe(G, cand)}: raw ancestors-loops={raw_owners} discounted={disc_owners}")
    print("  (only loops whose verdict differs are listed)")


if __name__ == "__main__":
    raise SystemExit(main())
