"""Synthetic repro: which cached-id ledgers does rewire_continuation leave stale?

Builds a tiny canonical-id ProcessGraph by hand (no front-end, no build) and
runs the real ``materialize_retained_loop_ports`` over it, then prints every
cached id ledger next to the live edge it duplicates.

Scenarios:
  1. Two retained loops that both mutate one state ``s`` (OPAQUE effects).
  2. One retained loop carrying ``acc`` whose result feeds an authored Tuple
     (``aggregate_leaf_value_ids``) and a ``list(acc)`` materializer
     (``materialized_source_value_ids``).
  3. ``_replace_parent_value`` (evaporator) over the same consumers.
  4. ``_alias_projection_to_member`` (callsite specialization) over a Tuple
     whose leaf is an aliased projection.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.compiler.loop_composer import (  # noqa: E402
    LoopDescriptor, LoopPlan, LoopStrategy,
    materialize_retained_loop_ports, _replace_parent_value,
)
from src.compiler.loop_ir import (  # noqa: E402
    LoopStateEffect, LoopStateEffectMode,
)


class Graph:
    def __init__(self):
        self.G = nx.DiGraph()
        self.G.graph["canonical_value_ids"] = True
        self.G.graph["identity_table"] = {}
        self.roots = []
        self.levels = {}

    def add(self, node_id, type_, parents=(), attributes=None, label=None):
        self.G.add_node(
            node_id, type=type_, op=type_.lower(), label=label or type_,
            expr_obj=None, value_id=node_id, parents=list(parents),
            children=[], attributes=dict(attributes or {}), tensor={},
        )
        for parent, role in parents:
            self.G.add_edge(parent, node_id, role=role)
            self.G.nodes[parent]["children"].append((node_id, role))


def show(graph, node_id, *keys):
    data = graph.G.nodes[node_id]
    attrs = data.get("attributes") or {}
    ledger = {key: attrs.get(key) for key in keys if key in attrs}
    print(f"  node {node_id:>3} {data['label']:<22} parents={data['parents']}"
          f"  cached={ledger}")


def parents_cycle(graph):
    g = nx.DiGraph()
    for node_id, data in graph.G.nodes(data=True):
        g.add_node(node_id)
        for parent, _role in data.get("parents") or ():
            if parent in graph.G:
                g.add_edge(parent, node_id)
    return next(iter(nx.simple_cycles(g)), None)


def scenario_two_loops_same_state():
    print("\n=== 1. two retained loops mutating the same state s ===")
    g = Graph()
    g.add(0, "Input", label="s", attributes={"binding_name": "s"})
    g.add(1, "Input", label="x", attributes={"binding_name": "x"})
    g.add(2, "For", label="loopA")
    g.add(3, "Call", [(0, "operand"), (1, "arg0")], label="effectA s.append")
    g.add(4, "For", label="loopB")
    g.add(5, "Call", [(0, "operand"), (1, "arg0")], label="effectB s.append")
    g.add(6, "Return", [(0, "value")], label="return s")
    g.roots = [6]

    def plan(loop_id, effect_id):
        loop = LoopDescriptor(
            node_id=loop_id, source_type="For", target="i",
            iterator_kind="arithmetic_sequence", body_nodes=(effect_id,),
            condition_nodes=(), start=0, stop=3, step=1,
            state_effects=(LoopStateEffect(
                state_name="s", operator="append", state_input_id=0,
                effect_node_id=effect_id, argument_value_ids=(1,),
                mode=LoopStateEffectMode.OPAQUE,
            ),),
        )
        return LoopPlan(loop=loop, strategy=LoopStrategy.NATIVE_SOURCE,
                        reason="test")

    plans = materialize_retained_loop_ports(g, (plan(2, 3), plan(4, 5)))
    for node_id in sorted(g.G.nodes):
        show(g, node_id, "value_source_id", "state_input_id")
    for p in plans:
        for effect in p.loop.state_effects:
            port = g.G.nodes[effect.state_output_id]
            edge = [int(a) for a, r in port["parents"] if r == "state"]
            print(f"  plan loop {p.loop.node_id}: cached state_input_id="
                  f"{effect.state_input_id}  state-port {effect.state_output_id}"
                  f" 'state' edge={edge}  effect-node {effect.effect_node_id}"
                  f" operand edge="
                  f"{[int(a) for a, r in g.G.nodes[effect.effect_node_id]['parents'] if r == 'operand']}")
    print(f"  roots={g.roots}")
    print(f"  parents-graph cycle: {parents_cycle(g)}")


def scenario_carried_into_tuple_and_materializer():
    print("\n=== 2. carried loop result consumed by Tuple + list() ===")
    g = Graph()
    g.add(0, "Input", label="acc0", attributes={"binding_name": "acc"})
    g.add(1, "Input", label="x", attributes={"binding_name": "x"})
    g.add(2, "For", label="loopC")
    g.add(3, "Add", [(0, "left"), (1, "right")], label="acc_updated")
    g.add(4, "Tuple", [(3, "elts"), (1, "elts")], label="(acc, x)",
          attributes={"producer_kind": "aggregate", "aggregate_kind": "tuple",
                      "aggregate_leaf_value_ids": (3, 1)})
    g.add(5, "Call", [(3, "arg:0")], label="list(acc)",
          attributes={"producer_kind": "aggregate_materialization",
                      "materialized_source_value_ids": (3,)})
    g.roots = [4, 5]
    loop = LoopDescriptor(
        node_id=2, source_type="For", target="i",
        iterator_kind="arithmetic_sequence", body_nodes=(3,),
        condition_nodes=(), start=0, stop=3, step=1,
        carried_bindings=(("acc", 0, 3),),
    )
    materialize_retained_loop_ports(
        g, (LoopPlan(loop=loop, strategy=LoopStrategy.NATIVE_SOURCE,
                     reason="test"),))
    for node_id in sorted(g.G.nodes):
        show(g, node_id, "value_source_id", "aggregate_leaf_value_ids",
             "materialized_source_value_ids")
    print(f"  roots={g.roots}")


def scenario_replace_parent_value():
    print("\n=== 3. _replace_parent_value (evaporator) over the same consumers ===")
    g = Graph()
    g.add(1, "Input", label="x")
    g.add(3, "Add", [(1, "left")], label="updated(body)")
    g.add(9, "Add", [(1, "left")], label="final clone")
    g.add(4, "Tuple", [(3, "elts"), (1, "elts")], label="(acc, x)",
          attributes={"producer_kind": "aggregate", "aggregate_kind": "tuple",
                      "aggregate_leaf_value_ids": (3, 1)})
    g.add(5, "Call", [(3, "arg:0")], label="list(acc)",
          attributes={"producer_kind": "aggregate_materialization",
                      "materialized_source_value_ids": (3,)})
    g.add(6, "Collection", label="collection")
    g.add(7, "Tuple", [(6, "elts")], label="tuple(collection)",
          attributes={"producer_kind": "aggregate", "aggregate_kind": "tuple",
                      "aggregate_leaf_value_ids": (6,)})
    g.add(10, "Const", label="v0")
    g.add(11, "Const", label="v1")
    _replace_parent_value(g, 3, (9,))
    _replace_parent_value(g, 6, (10, 11))
    for node_id in (4, 5, 7):
        show(g, node_id, "aggregate_leaf_value_ids",
             "materialized_source_value_ids")


def scenario_alias_projection():
    print("\n=== 4. _alias_projection_to_member over a Tuple leaf ===")
    started = time.perf_counter()
    from src.compiler.glsl_deployment_strategy import (
        _alias_projection_to_member, _structured_output_descriptor,
    )
    print(f"  (glsl import {time.perf_counter() - started:.1f}s)")
    g = Graph()
    g.add(0, "Input", label="formal")
    g.add(1, "Constant", label="0")
    g.G.nodes[1]["constant"] = 0
    g.add(2, "Indexed", [(0, "base"), (1, "index")], label="formal[0]")
    g.add(4, "Input", label="y")
    g.G.nodes[4]["tensor"] = {"shape": (3,), "dtype": "float64"}
    g.add(3, "Tuple", [(2, "elts"), (4, "elts")], label="(formal[0], y)",
          attributes={"producer_kind": "aggregate", "aggregate_kind": "tuple",
                      "aggregate_leaf_value_ids": (2, 4)})
    g.add(5, "Input", label="formal[0] member")
    g.G.nodes[5]["tensor"] = {"shape": (2,), "dtype": "float64"}
    g.roots = [3]
    _alias_projection_to_member(g, 2, 5)
    show(g, 3, "aggregate_leaf_value_ids")
    print(f"  node 2 still in graph: {2 in g.G}")
    print(f"  _structured_output_descriptor(3) = "
          f"{_structured_output_descriptor(g, 3)}")


if __name__ == "__main__":
    scenario_two_loops_same_state()
    scenario_carried_into_tuple_and_materializer()
    scenario_replace_parent_value()
    scenario_alias_projection()
