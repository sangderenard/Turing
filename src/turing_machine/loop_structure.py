from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Set, Tuple

from .turing_provenance import ProvenanceGraph, ProvEdge, ProvNode
import networkx as nx
from collections import defaultdict


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
    # Self-contained Lengauer–Tarjan dominator algorithm
    # ------------------------------------------------------------------
    def _lengauer_tarjan(self, start: int = 0) -> Dict[int, int]:
        """
        Compute immediate dominators using the Lengauer-Tarjan algorithm.
        Returns a dict mapping each node to its immediate dominator (None for start).
        """
        # initialization
        semi: Dict[int, int] = {}
        parent: Dict[int, int] = {}
        vertex: Dict[int, int] = {}
        bucket: Dict[int, List[int]] = defaultdict(list)
        pred: Dict[int, List[int]] = defaultdict(list)
        ancestor: Dict[int, int] = {}
        label: Dict[int, int] = {}
        dom: Dict[int, int] = {}
        N = 0
        # Step 1: DFS numbering
        def dfs(v: int):
            nonlocal N
            N += 1
            semi[v] = N
            vertex[N] = v
            label[v] = v
            ancestor[v] = None
            for w in self.succ.get(v, []):
                if w not in semi:
                    parent[w] = v
                    dfs(w)
                pred[w].append(v)
        dfs(start)
        # helper functions
        def compress(u: int):
            if ancestor.get(ancestor.get(u)) is not None:
                compress(ancestor[u])
                if semi[label[ancestor[u]]] < semi[label[u]]:
                    label[u] = label[ancestor[u]]
                ancestor[u] = ancestor[ancestor[u]]
        def eval(u: int) -> int:
            if ancestor.get(u) is None:
                return label[u]
            compress(u)
            if semi[label[u]] >= semi[label[ancestor[u]]]:
                return label[ancestor[u]]
            return label[u]
        # Step 2: compute semidominators
        for i in range(N, 1, -1):
            w = vertex[i]
            for v in pred[w]:
                u = eval(v)
                semi[w] = min(semi[w], semi[u])
            bucket[vertex[semi[w]]].append(w)
            ancestor[w] = parent[w]
            for v in bucket[parent[w]]:
                u = eval(v)
                dom[v] = u if semi[u] < semi[v] else parent[w]
            bucket[parent[w]].clear()
        # Step 3: explicit dominator assignment
        for i in range(2, N+1):
            w = vertex[i]
            if dom[w] != vertex[semi[w]]:
                dom[w] = dom[dom[w]]
        dom[start] = None
        return dom

    def _compute_idoms(self, start: int = 0) -> Dict[int, int]:
        """Return mapping node->immediate dominator using self-contained LT."""
        return self._lengauer_tarjan(start)

    # ------------------------------------------------------------------
    def find_loops(self) -> List[LoopInfo]:
        # debug logging: graph size
        total_nodes = len(self.succ)
        total_edges = sum(len(v) for v in self.succ.values())
        print(
            f"[LoopStructureAnalyzer] find_loops start: total_nodes={total_nodes}, total_edges={total_edges}"
        )
        loops: List[LoopInfo] = []
        sccs = self._tarjan()
        # Only compute dominators when loops exist
        nontrivial = [scc for scc in sccs if len(scc) > 1 or any(n in self.succ[n] for n in scc)]
        if not nontrivial:
            return []
        idom = self._compute_idoms(0 if self.graph.nodes else 0)
        # helper: check if 'a' dominates 'b'
        def dominates(a: int, b: int) -> bool:
            while b != a and b in idom and idom[b] != b:
                b = idom[b]
            return b == a

        for scc in nontrivial:
            if len(scc) == 1:
                n = next(iter(scc))
                if n not in self.succ[n]:
                    continue
            # identify *all* back edges within the SCC
            backedges: List[Tuple[int, int]] = []
            for u in scc:
                for v in self.succ[u]:
                    if v in scc and dominates(v, u):
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
