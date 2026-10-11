"""Bounded dispatch residency and dependency-only worker proposals.

The planner still owns graph carving and every IdentityBook write. Workers
only run the existing scheduler and registered topology foundation over a
sealed, small dependency snapshot. Admission includes completed-but-unpublished
results: a slow first task cannot accumulate an unbounded reorder buffer.

Region topology uses the existing BookSpill segment codec, in a separate
short-lived store. ProcessGraph headers, mutable graph metadata and semantic
leaves (including AST identity) stay shared with the coordinator. This is NOT
out-of-core source ingestion or SSA. Only cached dispatch topology is bounded.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import hashlib
from pathlib import Path
import threading
import time
import tempfile
from typing import Any

import networkx as nx


_BOOTSTRAPPING = ContextVar("compiler_graph_foundation_bootstrap", default=False)


def ensure_graph_foundation(progress=None) -> None:
    """Explicit worker policy builds OUR kernel, before the program book opens."""
    from .work_contract import active_contract
    from .native_compiler_accelerators import (
        TOPOLOGICAL_ORDER, DEPENDENCY_LEVELS, compiler_accelerators,
        compile_and_register_topology_accelerator,
        compile_and_register_dependency_levels_accelerator,
    )

    if _BOOTSTRAPPING.get() or active_contract().graph_workers == 1:
        return
    missing_topology = compiler_accelerators.resolve(TOPOLOGICAL_ORDER).tier != "native"
    missing_levels = compiler_accelerators.resolve(DEPENDENCY_LEVELS).tier != "native"
    if not missing_topology and not missing_levels:
        return
    if progress is not None:
        progress("graph-work: compiling authored graph foundation through our source compiler")
    token = _BOOTSTRAPPING.set(True)
    try:
        directory = Path(__file__).resolve().parents[2] / "build" / "compiler_foundations"
        directory.mkdir(parents=True, exist_ok=True)
        # Independent compiler processes must never rewrite a DLL another
        # process has loaded. Keep each build's existing source/book receipts.
        directory = Path(tempfile.mkdtemp(prefix="graph_", dir=directory))
        if missing_topology:
            compile_and_register_topology_accelerator(directory / "graph_topology", progress=progress)
        if missing_levels:
            compile_and_register_dependency_levels_accelerator(directory / "graph_levels", progress=progress)
    finally:
        _BOOTSTRAPPING.reset(token)


@dataclass(frozen=True)
class DependencySnapshot:
    records: tuple
    edges: tuple
    preference: str
    digest: str


@dataclass(frozen=True)
class ScheduleProposal:
    digest: str
    levels: dict
    schedule: tuple
    provider: str
    worker: int
    seconds: float = 0.0


def dependency_snapshot(subgraph, preference: str) -> DependencySnapshot:
    # Only the existing scheduler's inputs. No AST, Python bindings, scope,
    # tensor descriptors, book, or graph observer enters a worker.
    records = tuple(
        (node, tuple(data.get("parents", ())), tuple(data.get("children", ())))
        for node, data in subgraph.G.nodes(data=True)
    )
    edges = tuple(subgraph.G.edges)
    digest = hashlib.sha256(repr((records, edges, preference)).encode()).hexdigest()
    return DependencySnapshot(records, edges, preference, digest)


def schedule_dependency_snapshot(snapshot: DependencySnapshot) -> ScheduleProposal:
    started = time.perf_counter()
    from .native_compiler_accelerators import (
        TOPOLOGICAL_ORDER, DEPENDENCY_LEVELS, compiler_accelerators,
        lexicographical_topological_order, dependency_levels,
    )

    graph = nx.DiGraph()
    graph.add_nodes_from((node, {"parents": parents, "children": children})
                         for node, parents, children in snapshot.records)
    graph.add_edges_from(snapshot.edges)
    order = lexicographical_topological_order(graph, key=int)
    # Serial work retains ILPScheduler as its oracle; threaded work executes
    # the same equations in an authored, self-hosted integer-span foundation.
    levels = dependency_levels(graph, order=order, preference=snapshot.preference)
    buckets = {}
    for node in order:
        buckets.setdefault(int(levels.get(node, 0)), []).append(node)
    schedule = tuple((level, tuple(buckets.get(level, ()))) for level in sorted(set(levels.values())))
    return ScheduleProposal(snapshot.digest, levels, schedule,
                            "+".join(compiler_accelerators.resolve(semantic).name
                                     for semantic in (TOPOLOGICAL_ORDER, DEPENDENCY_LEVELS)),
                            threading.get_ident(), time.perf_counter() - started)


def publish_schedule(subgraph, snapshot, proposal) -> None:
    # Validate before touching live metadata. A worker completion is not
    # authority to publish into a graph changed since its admission.
    if proposal.digest != snapshot.digest or dependency_snapshot(subgraph, snapshot.preference).digest != snapshot.digest:
        raise RuntimeError("stale dispatch scheduling proposal")
    subgraph.levels = proposal.levels
    subgraph.scheduler.levels_asap = (
        proposal.levels if snapshot.preference == "asap" else subgraph.scheduler.levels_asap
    )
    if snapshot.preference == "alap":
        subgraph.scheduler.levels_alap = proposal.levels
    subgraph.G.graph["compartment_schedule_preference"] = snapshot.preference
    subgraph.G.graph["compartment_schedule"] = proposal.schedule


class _FrozenDict(dict):
    def _refuse(self, *_args, **_kwargs):
        raise RuntimeError("dispatch node attributes require a resident_region write lease")
    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _refuse

    def __reduce__(self):
        return dict, (dict(self),)


class _FrozenList(list):
    def _refuse(self, *_args, **_kwargs):
        raise RuntimeError("dispatch node attributes require a resident_region write lease")
    __setitem__ = __delitem__ = append = extend = insert = pop = remove = clear = sort = reverse = __iadd__ = __imul__ = _refuse

    def __reduce__(self):
        return list, (list(self),)


class _LeasedDict(dict):
    """Even a dictionary escaping a write lease cannot later mutate stale data."""
    def _check(self):
        self._pool._check()
        entry = self._pool.entries.get(self._identity)
        if entry is None or not entry.pins:
            raise RuntimeError("dispatch node attributes require a resident_region write lease")
        if entry.generation != self._generation:
            raise RuntimeError("dispatch attribute view belongs to an expired residency generation")

    def __setitem__(self, key, value):
        self._check()
        dict.__setitem__(self, key, _managed(value, self._pool, self._identity))

    def __delitem__(self, key):
        self._check()
        dict.__delitem__(self, key)

    def update(self, *args, **kwargs):
        self._check()
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def setdefault(self, key, default=None):
        self._check()
        if key not in self:
            self[key] = default
        return self[key]

    def pop(self, *args):
        self._check()
        return dict.pop(self, *args)

    def popitem(self):
        self._check()
        return dict.popitem(self)

    def clear(self):
        self._check()
        dict.clear(self)

    def __ior__(self, other):
        self.update(other)
        return self


class _LeasedList(list):
    def _check(self):
        self._pool._check()
        entry = self._pool.entries.get(self._identity)
        if entry is None or not entry.pins:
            raise RuntimeError("dispatch node attributes require a resident_region write lease")
        if entry.generation != self._generation:
            raise RuntimeError("dispatch attribute view belongs to an expired residency generation")

    def __setitem__(self, key, value):
        self._check()
        value = ([_managed(v, self._pool, self._identity) for v in value]
                 if isinstance(key, slice) else _managed(value, self._pool, self._identity))
        list.__setitem__(self, key, value)

    def __delitem__(self, key):
        self._check()
        list.__delitem__(self, key)

    def append(self, value):
        self._check()
        list.append(self, _managed(value, self._pool, self._identity))

    def extend(self, values):
        self._check()
        list.extend(self, [_managed(v, self._pool, self._identity) for v in values])

    def insert(self, key, value):
        self._check()
        list.insert(self, key, _managed(value, self._pool, self._identity))

    def _mutator(name):
        def checked(self, *args, **kwargs):
            self._check()
            return getattr(list, name)(self, *args, **kwargs)
        return checked
    pop = _mutator("pop")
    remove = _mutator("remove")
    clear = _mutator("clear")
    sort = _mutator("sort")
    reverse = _mutator("reverse")
    __imul__ = _mutator("__imul__")

    def __iadd__(self, other):
        self.extend(other)
        return self


class _LeasedSet(set):
    _check = _LeasedDict._check

    def _mutator(name):
        def checked(self, *args, **kwargs):
            self._check()
            return getattr(set, name)(self, *args, **kwargs)
        return checked
    add = _mutator("add")
    remove = _mutator("remove")
    discard = _mutator("discard")
    pop = _mutator("pop")
    clear = _mutator("clear")
    update = _mutator("update")
    intersection_update = _mutator("intersection_update")
    difference_update = _mutator("difference_update")
    symmetric_difference_update = _mutator("symmetric_difference_update")
    __ior__ = _mutator("__ior__")
    __iand__ = _mutator("__iand__")
    __ixor__ = _mutator("__ixor__")
    __isub__ = _mutator("__isub__")


class _FrozenSet(set):
    _refuse = _FrozenDict._refuse
    add = remove = discard = pop = clear = update = _refuse
    intersection_update = difference_update = symmetric_difference_update = _refuse
    __ior__ = __iand__ = __ixor__ = __isub__ = _refuse

    def __reduce__(self):
        return set, (tuple(self),)


def _managed(value, pool, identity, memo=None):
    if (isinstance(value, (_LeasedDict, _LeasedList, _LeasedSet))
            and value._pool is pool and value._identity == identity
            and value._generation == pool.entries[identity].generation):
        return value  # Already belongs to this exact writable generation.
    memo = {} if memo is None else memo
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, (dict, list)):
        result = _LeasedDict() if isinstance(value, dict) else _LeasedList()
        result._pool, result._identity = pool, identity
        result._generation = pool.entries[identity].generation
        memo[id(value)] = result
        if isinstance(value, dict):
            for key, item in value.items():
                dict.__setitem__(result, key, _managed(item, pool, identity, memo))
        else:
            list.extend(result, (_managed(item, pool, identity, memo) for item in value))
        return result
    if isinstance(value, tuple):
        result = tuple(_managed(item, pool, identity, memo) for item in value)
        incumbent = memo.get(id(value))
        if incumbent is not None:
            return incumbent
        memo[id(value)] = result
        return result
    if isinstance(value, set):
        result = _LeasedSet(value)
        result._pool, result._identity = pool, identity
        result._generation = pool.entries[identity].generation
        memo[id(value)] = result
        return result
    return value


def _freeze(value, memo=None):
    """Escaping read views cannot write a stale, evicted payload."""
    memo = {} if memo is None else memo
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, dict):
        result = _FrozenDict()
        memo[id(value)] = result
        for key, item in value.items():
            dict.__setitem__(result, key, _freeze(item, memo))
        return result
    if isinstance(value, list):
        result = _FrozenList()
        memo[id(value)] = result
        list.extend(result, (_freeze(item, memo) for item in value))
        return result
    if type(value) is tuple:
        result = tuple(_freeze(item, memo) for item in value)
        incumbent = memo.get(id(value))
        if incumbent is not None:
            return incumbent
        memo[id(value)] = result
        return result
    if isinstance(value, set):
        result = _FrozenSet(value)
        memo[id(value)] = result
        return result
    # Semantic leaves are coordinator inputs, never invented or copied here.
    return value


@dataclass(frozen=True)
class _Leaf:
    identity: int


@dataclass
class _RegionEntry:
    segment: Any = None
    pins: int = 0
    dirty: bool = True
    managed: bool = False
    generation: int = 0


class _RegionMap(Mapping):
    """Permanent view: cached NetworkX views never pin an old topology."""
    def __init__(self, pool, identity, kind):
        self.pool, self.identity, self.kind = pool, identity, kind

    def __len__(self):
        return len(self.pool.payload(self.identity)[self.kind])

    def __iter__(self):
        # A reader can touch another region while this iteration is suspended.
        return iter(tuple(self.pool.payload(self.identity)[self.kind]))

    def __getitem__(self, key):
        payload = self.pool.payload(self.identity)
        if self.pool.entries[self.identity].pins:
            return payload[self.kind][key]
        frozen = self.pool.read_views.get(self.identity)
        if frozen is None:
            frozen = self.pool.read_views[self.identity] = _freeze(payload)
        return frozen[self.kind][key]


class _RegionGraph(nx.DiGraph):
    def __init__(self, pool, identity, metadata):
        # Deliberately retain the NetworkX graph class/API, not a graph-shaped
        # stand-in. Only its dictionary ownership changes.
        self.graph = metadata
        self._node = _RegionMap(pool, identity, "nodes")
        self._succ = self._adj = _RegionMap(pool, identity, "succ")
        self._pred = _RegionMap(pool, identity, "pred")
        self.__networkx_cache__ = {}
        self._region_pool, self._region_identity = pool, identity

    def __reduce__(self):
        # Existing checkpoint/digest readers see an ordinary materialized graph,
        # never a pickle of the pool, its file handles or other regions.
        return (_restore_graph, (dict(self.graph), list(self.nodes(data=True)), list(self.edges(data=True))))


def _restore_graph(metadata, nodes, edges):
    graph = nx.DiGraph()
    graph.graph.update(metadata)
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph


class RegionTopologyPool:
    """One compile's LRU; all three region collections share these headers."""
    def __init__(self, book, resident_limit: int):
        from .identity_spill import BookSpill
        from .concordance_declarations import COMPILE_POLICY, MEMORY_REGULATION, POLICY_DECLARATION
        from .identity_concordance import Mode, Novel

        book.post(COMPILE_POLICY, ("graph_resident_regions",), str(resident_limit),
                  stage=MEMORY_REGULATION, provenance=Novel(POLICY_DECLARATION, ()), mode=Mode.CONCORD)
        self.book = book
        self.limit = resident_limit
        self.store = BookSpill(book.registry)
        self.owner = threading.get_ident()
        self.entries = {}
        self.resident = OrderedDict()
        self.read_views = {}
        self.leaves = {}
        self.closed = False
        self.executor = None
        self.worker_count = None
        self.reservations = 0
        self.stats = dict(regions=0, resident_peak=0, live_topology_peak=0, spills=0, reloads=0,
                          spill_raw_bytes=0, spill_stored_bytes=0,
                          admitted=0, published=0, outstanding_peak=0,
                          admission_limit=0, worker_threads=set(),
                          carve_seconds=0.0, worker_seconds=0.0,
                          wait_seconds=0.0, publish_seconds=0.0)

    def _check(self):
        if self.closed:
            raise RuntimeError("dispatch topology store is closed")
        if threading.get_ident() != self.owner:
            raise RuntimeError("dispatch topology/provenance is coordinator-owned")

    def _codec(self, value, *, decode=False, memo=None):
        # Reuse BookSpill's registry-aware framing. Semantic leaves are shared
        # pointers on the coordinator, not recursive copies of bound programs.
        memo = {} if memo is None else memo
        if decode and isinstance(value, _Leaf):
            return self.leaves[value.identity]
        if id(value) in memo:
            return memo[id(value)]
        if isinstance(value, (dict, list)):
            result = {} if isinstance(value, dict) else []
            memo[id(value)] = result
            if isinstance(value, dict):
                result.update((self._codec(k, decode=decode, memo=memo), self._codec(v, decode=decode, memo=memo))
                              for k, v in value.items())
            else:
                result.extend(self._codec(v, decode=decode, memo=memo) for v in value)
            return result
        if isinstance(value, (tuple, set, frozenset)):
            constructor = tuple if isinstance(value, tuple) else set if isinstance(value, set) else frozenset
            result = constructor(self._codec(v, decode=decode, memo=memo) for v in value)
            incumbent = memo.get(id(value))
            if incumbent is not None:
                return incumbent
            memo[id(value)] = result
            return result
        if value is None or type(value) in {str, bytes, int, float, complex, bool}:
            return value
        if decode:
            return value
        self.leaves[id(value)] = value
        return _Leaf(id(value))

    def _trim(self, incoming=0):
        while len(self.resident) + self.reservations + incoming > self.limit:
            victim = next((key for key in self.resident if not self.entries[key].pins), None)
            if victim is None:
                raise RuntimeError("dispatch residency bound exhausted by active write leases")
            entry = self.entries[victim]
            payload = self.resident[victim]
            if entry.dirty or entry.segment is None:
                entry.segment = self.store.write("graph-region", str(victim), self._codec(payload),
                                                 cells=len(payload["nodes"]), rows=len(payload["nodes"]))
                entry.dirty = False
                self.stats["spills"] += 1
                self.stats["spill_raw_bytes"] += entry.segment.raw
                self.stats["spill_stored_bytes"] += entry.segment.stored
            del self.resident[victim]
            entry.generation += 1
            entry.managed = False
            self.read_views.pop(victim, None)

    def reserve(self):
        """Reserve a full topology slot BEFORE the coordinator carves a job."""
        self._check()
        self._trim(incoming=1)
        self.reservations += 1
        self.stats["live_topology_peak"] = max(
            self.stats["live_topology_peak"], len(self.resident) + self.reservations,
        )

    def abandon_reservation(self):
        self._check()
        if not self.reservations:
            raise RuntimeError("no dispatch admission reservation to abandon")
        self.reservations -= 1

    def add(self, subgraph, *, reserved=False):
        self._check()
        if reserved:
            self.abandon_reservation()
        self._trim(incoming=1)
        identity = self.stats["regions"]
        original = subgraph.G
        self.entries[identity] = _RegionEntry()
        # Ownership transfers here. Do not recursively wrap a topology the
        # source-SSA path may NEVER read. Mutable guards are installed on the
        # first write lease; immutable guards on the first read view.
        self.resident[identity] = dict(nodes=original._node, succ=original._succ,
                                      pred=original._pred, levels=subgraph.levels)
        graph = _RegionGraph(self, identity, original.graph)
        subgraph.G = graph
        subgraph.scheduler.G = graph
        # Scheduler caches are the same levels, not a second permanent copy.
        subgraph.scheduler.levels_asap = {}
        subgraph.scheduler.levels_alap = {}
        subgraph.levels = _RegionMap(self, identity, "levels")
        self.stats["regions"] += 1
        self.stats["resident_peak"] = max(self.stats["resident_peak"], len(self.resident))
        self.stats["live_topology_peak"] = max(
            self.stats["live_topology_peak"], len(self.resident) + self.reservations,
        )
        return subgraph

    def payload(self, identity):
        self._check()
        if identity not in self.resident:
            self._trim(incoming=1)
            self.resident[identity] = self._codec(self.store.read(self.entries[identity].segment), decode=True)
            self.entries[identity].managed = False
            self.stats["reloads"] += 1
            self.stats["resident_peak"] = max(self.stats["resident_peak"], len(self.resident))
        self.resident.move_to_end(identity)
        return self.resident[identity]

    @contextmanager
    def lease(self, identity):
        self._check()
        payload = self.payload(identity)
        entry = self.entries[identity]
        if not entry.managed:
            payload = self.resident[identity] = _managed(payload, self, identity)
            entry.managed = True
        entry.pins += 1
        self.read_views.pop(identity, None)
        try:
            yield payload
        finally:
            entry.pins -= 1
            entry.dirty = True
            self.read_views.pop(identity, None)

    def release(self, identity):
        self._check()
        if self.entries[identity].pins:
            raise RuntimeError("cannot retire a leased dispatch region")
        self.resident.pop(identity, None)
        self.read_views.pop(identity, None)
        self.entries.pop(identity)

    def summary(self):
        return {**self.stats, "worker_threads": len(self.stats["worker_threads"]),
                "resident_limit": self.limit, "resident_now": len(self.resident),
                "source_and_ssa_paged": False}

    def worker_executor(self, workers):
        self._check()
        if workers == 1:
            return None
        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="compiler-graph")
            self.worker_count = workers
        elif self.worker_count != workers:
            raise RuntimeError("graph worker policy changed during an active compile")
        return self.executor

    def close(self):
        if self.closed:
            return
        self._check()
        if any(entry.pins for entry in self.entries.values()):
            raise RuntimeError("dispatch write lease survived compile end")
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
        self.resident.clear()
        self.read_views.clear()
        self.entries.clear()
        self.leaves.clear()
        self.reservations = 0
        self.closed = True
        self.store._finalizer()  # Existing BookSpill cleanup, only this owned store.


def resident_region(subgraph):
    graph = subgraph.G
    if isinstance(graph, _RegionGraph):
        return graph._region_pool.lease(graph._region_identity)
    return nullcontext()


def region_write_lease(function):
    @wraps(function)
    def leased(subgraph, *args, **kwargs):
        with resident_region(subgraph):
            return function(subgraph, *args, **kwargs)
    return leased


class RegionProjectionSequence(Sequence):
    """Existing region constructors, deferred until their consumers need them."""
    def __init__(self, count, factory):
        self.count, self.factory, self.materialized = count, factory, {}

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(self.count)))
        if index < 0:
            index += self.count
        if not 0 <= index < self.count:
            raise IndexError(index)
        if index not in self.materialized:
            self.materialized[index] = self.factory(index)
        return self.materialized[index]


def retire_region(subgraph):
    graph = getattr(subgraph, "G", None)
    if isinstance(graph, _RegionGraph) and graph._region_identity in graph._region_pool.entries:
        graph._region_pool.release(graph._region_identity)


def graph_work_settings():
    from .work_contract import active_contract
    contract = active_contract()
    workers = 1 if _BOOTSTRAPPING.get() else contract.graph_workers
    limit = contract.graph_resident_regions
    return workers, limit, min(workers, limit) if limit is not None else workers


def bounded_schedule_regions(regions, *, workers, admission_limit, finish, stats=None, executor=None):
    """Carve on the coordinator, admit W snapshots, publish in ordinal order."""
    from .native_compiler_accelerators import TOPOLOGICAL_ORDER, DEPENDENCY_LEVELS, compiler_accelerators
    if workers > 1 and any(compiler_accelerators.resolve(semantic).tier != "native"
                           for semantic in (TOPOLOGICAL_ORDER, DEPENDENCY_LEVELS)):
        raise RuntimeError("parallel region scheduling requires our compiled topology and level foundations")
    if workers < 1 or admission_limit < 1:
        raise ValueError("graph workers and admission limit must be positive")
    iterator = iter(regions)
    queue = deque()
    owns_executor = executor is None and workers > 1
    if owns_executor:
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="compiler-graph")
    if stats is not None:
        stats["admission_limit"] = max(stats["admission_limit"], admission_limit)
    try:
        exhausted = False
        while queue or not exhausted:
            while not exhausted and len(queue) < admission_limit:
                try:
                    started = time.perf_counter()
                    index, subgraph, snapshot = next(iterator)
                    carved_seconds = time.perf_counter() - started
                except StopIteration:
                    exhausted = True
                    break
                proposal = (executor.submit(schedule_dependency_snapshot, snapshot) if executor is not None
                            else schedule_dependency_snapshot(snapshot))
                queue.append((index, subgraph, snapshot, proposal))
                if stats is not None:
                    stats["carve_seconds"] += carved_seconds
                    stats["admitted"] += 1
                    stats["outstanding_peak"] = max(stats["outstanding_peak"], len(queue))
            if not queue:
                break
            index, subgraph, snapshot, pending = queue.popleft()
            started = time.perf_counter()
            proposal = pending.result() if executor is not None else pending
            if stats is not None:
                stats["wait_seconds"] += time.perf_counter() - started
                stats["worker_seconds"] += proposal.seconds
            started = time.perf_counter()
            publish_schedule(subgraph, snapshot, proposal)
            finish(index, subgraph, snapshot, proposal)
            if stats is not None:
                stats["publish_seconds"] += time.perf_counter() - started
                stats["published"] += 1
                if executor is not None:
                    stats["worker_threads"].add(proposal.worker)
            # Drop the last proposal and graph before admitting another window.
            del subgraph, snapshot, pending, proposal
    finally:
        for _index, _graph, _snapshot, pending in queue:
            if executor is not None:
                pending.cancel()
        if owns_executor:
            executor.shutdown(wait=True, cancel_futures=True)


def post_graph_work_publication(graph, subgraph, index, snapshot, proposal, admission_limit):
    from .concordance_declarations import (
        COMPILE_POLICY, DEPLOYMENT_REGION, GRAPH_WORK_PUBLICATION,
        GraphWorkPublication, MEMORY_REGULATION, POLICY_DECLARATION, PLANNER_REGION_CARVE,
        SCOPE_REGISTRY, SCOPE_ORIGIN,
    )
    from .identity_concordance import current_identity_book, Derived, Mode, Novel
    book = current_identity_book()
    scope = graph.G.graph.get("planning_scope")
    region = book.latest_ref(DEPLOYMENT_REGION, (scope, index)) if scope is not None else None
    if region is None:
        return
    workers, _limit, _admission = graph_work_settings()
    for field, value in (("graph_workers", workers), ("graph_admission_limit", admission_limit)):
        book.post(COMPILE_POLICY, (field,), str(value), stage=MEMORY_REGULATION,
                  provenance=Novel(POLICY_DECLARATION, ()), mode=Mode.CONCORD)
    policy = book.latest_ref(COMPILE_POLICY, ("graph_admission_limit",))
    read_scope = subgraph.G.graph.get("lexical_read_scope")
    read_scope = None if read_scope is None else tuple(read_scope)
    sources = [region, policy]
    sources.append(book.latest_ref(COMPILE_POLICY, ("graph_workers",)))
    if read_scope is not None:
        # The carved copy's book-minted scope, not just its source's ordinal.
        # This distinguishes re-carvings under the same planning region and
        # connects the sealed snapshot to the existing copy-on-read backbone.
        for page, row in ((SCOPE_REGISTRY, read_scope), (SCOPE_ORIGIN, (read_scope,))):
            cell = book.latest_ref(page, row)
            if cell is not None:
                sources.append(cell)
    fact = GraphWorkPublication(snapshot.digest, snapshot.preference, len(snapshot.records),
                                len(snapshot.edges), proposal.provider, admission_limit, read_scope)
    # Re-planning the same ordinal can update its sealed input. The region
    # remains the authority; this is a revision, not another guessed identity.
    from .planner_concordance_posts import _post_if_changed
    _post_if_changed(GRAPH_WORK_PUBLICATION, (scope, index), fact,
                     stage=PLANNER_REGION_CARVE, cells=tuple(sources))
