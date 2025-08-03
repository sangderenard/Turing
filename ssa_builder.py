from __future__ import annotations

from typing import Dict, List, Tuple

from turing_provenance import ProvenanceGraph
from loop_structure import LoopStructureAnalyzer, LoopInfo


def insert_phi_nodes(pg: ProvenanceGraph, loops: List[LoopInfo]) -> Dict[int, List[Tuple[int, int, int]]]:
    """Return mapping ``header -> [(arg_pos, init_src, back_src), ...]``."""
    phi_map: Dict[int, List[Tuple[int, int, int]]] = {}
    for info in loops:
        if info.loop_vars:
            phi_map[info.header] = list(info.loop_vars)
    return phi_map


def rename_to_ssa(pg: ProvenanceGraph, phi_map: Dict[int, List[Tuple[int, int, int]]]) -> Dict[int, str]:
    """Assign human readable names to all node results and φ-nodes."""
    names: Dict[int, str] = {n.idx: f"%n{n.idx}" for n in pg.nodes}
    for header, vars in phi_map.items():
        for arg_pos, _, _ in vars:
            names[(header << 16) + arg_pos] = f"%n{header}.phi{arg_pos}"
    return names


def graph_to_ssa_with_loops(pg: ProvenanceGraph) -> str:
    """Emit a tiny LLVM-like textual SSA for ``pg`` handling natural loops."""
    analyzer = LoopStructureAnalyzer(pg)
    loops = analyzer.find_loops()
    phi_map = insert_phi_nodes(pg, loops)
    names = rename_to_ssa(pg, phi_map)

    # Precompute incoming edges per node
    incoming: Dict[int, Dict[int, int]] = {n.idx: {} for n in pg.nodes}
    for e in pg.edges:
        incoming[e.dst_idx][e.arg_pos] = e.src_idx

    lines: List[str] = []
    for node in pg.nodes:
        # emit φ-nodes if this is a loop header
        if node.idx in phi_map:
            for arg_pos, init_src, back_src in phi_map[node.idx]:
                phi_name = names[(node.idx << 16) + arg_pos]
                init_name = names.get(init_src, f"%n{init_src}" if init_src >= 0 else "undef")
                back_name = names[back_src]
                lines.append(
                    f"{phi_name} = phi [{init_name}, %{init_src if init_src >= 0 else 'pre'}], [{back_name}, %{back_src}]"
                )
        # now emit the actual operation
        args = []
        for idx in range(len(node.args)):
            src = incoming[node.idx].get(idx)
            args.append(names.get(src, f"%n{src}" if src is not None else "undef"))
        lines.append(f"{names[node.idx]} = {node.op} {', '.join(args)}")
    return "\n".join(lines)
