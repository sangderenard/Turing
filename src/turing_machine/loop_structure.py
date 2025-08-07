from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Set, Tuple

from .turing_provenance import ProvenanceGraph, ProvEdge, ProvNode


@dataclass
class LoopInfo:
    """Container describing a natural loop."""

    header: int
    body: Set[int]
    latch: int
    preheader: int
    loop_vars: List[Tuple[int, int, int]]  # (arg_pos, init_src, back_src)


class LoopStructureAnalyzer:
    """Analyse a ``ProvenanceGraph`` to find natural loops.

    This is a *minimal* implementation tailored for the unit tests in this
    kata.  It purposefully ignores many edge cases of full compiler loop
    detection but follows the outline from the project brief:

    * compute strongly connected components via Tarjan's algorithm
    * for each non‑trivial SCC, identify header and latch via a back edge
      ``u -> h`` where ``h`` dominates ``u``
    * determine the preheader and loop carried variables feeding the header
    """

    def __init__(self, graph: ProvenanceGraph):
        self.graph = graph
        self.succ: Dict[int, Set[int]] = {n.idx: set() for n in graph.nodes}
        self.pred: Dict[int, Set[int]] = {n.idx: set() for n in graph.nodes}
        for e in graph.edges:
            self.succ[e.src_idx].add(e.dst_idx)
            self.pred[e.dst_idx].add(e.src_idx)

    # ------------------------------------------------------------------
    # Tarjan SCC
    # ------------------------------------------------------------------
    def _tarjan(self) -> List[Set[int]]:
        index = 0
        stack: List[int] = []
        on_stack: Set[int] = set()
        indices: Dict[int, int] = {}
        lowlink: Dict[int, int] = {}
        result: List[Set[int]] = []

        for v in self.succ.keys():
            if v in indices:
                continue

            indices[v] = index
            lowlink[v] = index
            index += 1
            stack.append(v)
            on_stack.add(v)

            call_stack: List[Tuple[int, Iterator[int]]] = [(v, iter(self.succ[v]))]
            while call_stack:
                node, it = call_stack[-1]
                try:
                    w = next(it)
                except StopIteration:
                    call_stack.pop()
                    if call_stack:
                        parent, _ = call_stack[-1]
                        lowlink[parent] = min(lowlink[parent], lowlink[node])
                    if lowlink[node] == indices[node]:
                        scc: Set[int] = set()
                        while True:
                            w = stack.pop()
                            on_stack.remove(w)
                            scc.add(w)
                            if w == node:
                                break
                        result.append(scc)
                    continue

                if w not in indices:
                    indices[w] = index
                    lowlink[w] = index
                    index += 1
                    stack.append(w)
                    on_stack.add(w)
                    call_stack.append((w, iter(self.succ[w])))
                elif w in on_stack:
                    lowlink[node] = min(lowlink[node], indices[w])

        return result

    # ------------------------------------------------------------------
    # Dominators (simple iterative algorithm)
    # ------------------------------------------------------------------
    def _dominators(self, start: int = 0) -> Dict[int, Set[int]]:
        nodes = list(self.succ.keys())
        dom: Dict[int, Set[int]] = {n: set(nodes) for n in nodes}
        dom[start] = {start}
        changed = True
        while changed:
            changed = False
            for n in nodes:
                if n == start:
                    continue
                preds = self.pred[n]
                if not preds:
                    new_dom = {n}
                else:
                    it = iter(preds)
                    new_dom = dom[next(it)].copy()
                    for p in it:
                        new_dom &= dom[p]
                    new_dom.add(n)
                if new_dom != dom[n]:
                    dom[n] = new_dom
                    changed = True
        return dom

    # ------------------------------------------------------------------
    def find_loops(self) -> List[LoopInfo]:
        loops: List[LoopInfo] = []
        sccs = self._tarjan()
        dom = self._dominators(0 if self.graph.nodes else 0)
        for scc in sccs:
            if len(scc) == 1:
                n = next(iter(scc))
                if n not in self.succ[n]:
                    continue
            # identify *all* back edges within the SCC
            backedges: List[Tuple[int, int]] = []
            for u in scc:
                for v in self.succ[u]:
                    if v in scc and v in dom[u]:
                        backedges.append((u, v))
            for latch, header in backedges:
                preds = self.pred[header]
                pre = next(iter(preds - scc)) if (preds - scc) else -1
                loop_vars: List[Tuple[int, int, int]] = []
                for e in self.graph.edges:
                    if e.dst_idx == header and e.src_idx == latch:
                        init_src = -1
                        for e2 in self.graph.edges:
                            if (
                                e2.dst_idx == header
                                and e2.arg_pos == e.arg_pos
                                and e2.src_idx == pre
                            ):
                                init_src = e2.src_idx
                                break
                        loop_vars.append((e.arg_pos, init_src, e.src_idx))
                loops.append(LoopInfo(header, scc, latch, pre, loop_vars))
        return loops
