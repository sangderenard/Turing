"""Real demo: spectral graph analysis on an actual compiled-program graph.

Builds a ProcessGraph for a small but genuine program:

    x = init(x0)
    for i in range(5):        # a real loop, closed by a real loop-back edge
        x = add(x, step(i))   # loop-carried: x depends on its own previous value
    y = scale(x)               # a straight-line tail after the loop exits

Then runs it through field_from_process_graph -- the actual adapter, not a
hand-built InfluenceField -- and through analyze_graph_spectrum.

    python -m examples.spectral_graph_analysis_demo
"""
from __future__ import annotations

import numpy as np

from src.transmogrifier.graph.graph_express2 import ProcessGraph
from src.compiler.influence_field import InfluenceContract, field_from_process_graph, DYNAMIC
from src.compiler.shell_telemetry import TelemetryChannel
from src.compiler.spectral_graph_analysis import analyze_graph_spectrum, profile_projection


def _add_node(graph, node_id, op, parents=(), *, label=None):
    parent_items = [(parent, f"arg{index}") for index, parent in enumerate(parents)]
    graph.G.add_node(
        node_id, op=op, type=op, label=label or op, parents=parent_items,
        children=[], attributes={}, extra_args={}, tensor={}, control={},
        constant=None, expr_obj=None, store_id=None,
    )
    for parent, role in parent_items:
        graph.G.add_edge(parent, node_id, role=role)
        graph.G.nodes[parent]["children"].append((node_id, role))


def build_program_graph() -> ProcessGraph:
    graph = ProcessGraph(materialize_memory=False)

    # Straight-line setup.
    _add_node(graph, 1, "input", label="x0")

    # A real loop, unrolled here so its shape is explicit, but genuinely
    # loop-carried: each iteration's accumulator depends on the previous
    # one, and the last iteration closes back to the first with a real
    # "loop-back"-roled edge, exactly as a real compiler's control-flow
    # lowering would mark it.
    header = 1
    prev = header
    for i in range(5):
        step_id = 100 + i
        acc_id = 200 + i
        # Realistic: this iteration's step value depends on the loop
        # variable, so it is actually reachable by propagate() -- an
        # unreferenced constant with no incoming edges is not what a real
        # per-iteration computation looks like.
        _add_node(graph, step_id, "step_const", (header,), label=f"step[{i}]")
        _add_node(graph, acc_id, "add", (prev, step_id), label=f"acc[{i}]")
        prev = acc_id

    latch = prev
    # Close the loop for real: latch -> header, marked as the compiler
    # itself would mark it. Direct role assignment (not through _add_node,
    # which only knows plain "argN" parent roles) matches how
    # field_from_process_graph reads role off edge data.
    graph.G.add_edge(latch, header, role="loop-back")

    # A straight-line tail after the loop exits.
    _add_node(graph, 300, "scale", (latch,), label="y")
    graph.roots = [300]
    return graph


def main() -> None:
    graph = build_program_graph()
    contract = InfluenceContract(enabled=True, categories=(DYNAMIC,))

    channel = TelemetryChannel(name="spectral-demo")
    field = field_from_process_graph(graph, contract, profile_channel=channel)
    field.propagate()

    print(f"nodes: {len(field.node_keys())}")
    print(f"edges: {len(field.edge_list())}")

    decomposition = analyze_graph_spectrum(field)

    print()
    print(f"whole graph: method={decomposition.whole_graph.method}  "
          f"n={len(decomposition.whole_graph.node_order)}  "
          f"spectral_gap={decomposition.whole_graph.spectral_gap():.6f}")

    print()
    print(f"loop regions found: {len(decomposition.loop_regions)}")
    for region in decomposition.loop_regions:
        print(f"  header={region.loop.header} latch={region.loop.latch} "
              f"n={len(region.node_order)} method={region.method} "
              f"spectral_gap={region.spectral_gap():.6f}")
        ev = np.asarray(region.eigenvalues.data if hasattr(region.eigenvalues, "data") else region.eigenvalues)
        print(f"    eigenvalues: {np.sort(ev)}")

        try:
            phase, intensity = profile_projection(region, field)
            intensity_np = np.asarray(intensity.data if hasattr(intensity, "data") else intensity)
            print(f"    real per-node intensity (profiled ns-derived), per "
                  f"node first sample: {intensity_np[:, 0]}")
        except ValueError as exc:
            # Real relaxation dynamics: nodes get popped different numbers
            # of times (merge points revisit, decay recirculates), so the
            # region-wide batched projection correctly refuses rather than
            # padding/truncating real measured data. Fall back to reading
            # each node's own clock individually, exactly as the error
            # message suggests.
            print(f"    region-wide profile_projection refused: {exc}")
            print("    reading node_phase_clock per node instead:")
            for key in region.node_order:
                clock = field.node_phase_clock(key)
                print(f"      node {key}: {clock.sample_count} real ticks, "
                      f"{clock.elapsed_seconds * 1e6:.2f}us total")


if __name__ == "__main__":
    main()
