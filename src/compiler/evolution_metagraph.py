"""Thin append-only provenance graph beside compiler IR transformations.

IR objects remain authoritative for semantics.  This metagraph records only
stable graph/component identities and transformation events so observers can
replay how one representation consumed another.  When no recorder is active,
the convenience functions are no-ops and compiler behavior is unchanged.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from dataclasses import fields, is_dataclass
import threading
import time
from typing import Any, Callable, Iterable, Mapping


@dataclass(frozen=True, slots=True, order=True)
class EvolutionGraphRef:
    id: str
    stage: str
    label: str


@dataclass(frozen=True, slots=True, order=True)
class EvolutionComponentRef:
    graph_id: str
    local_id: str


@dataclass(frozen=True, slots=True)
class EvolutionComponent:
    ref: EvolutionComponentRef
    label: str
    kind: str
    attributes: Mapping[str, Any] = field(default_factory=dict)
    token_id: int | None = None


@dataclass(frozen=True, slots=True)
class EvolutionEvent:
    sequence: int
    captured_ns: int
    kind: str
    graph: EvolutionGraphRef | None = None
    component: EvolutionComponentRef | None = None
    sources: tuple[EvolutionComponentRef, ...] = ()
    detail: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvolutionSnapshot:
    graphs: tuple[EvolutionGraphRef, ...]
    components: tuple[EvolutionComponent, ...]
    events: tuple[EvolutionEvent, ...]


Subscriber = Callable[[EvolutionEvent], None]


class TokenPathAtlas:
    """Stable integer identities for structural and vocabulary token paths.

    Text labels remain diagnostics. Equality and cross-representation lineage
    use the returned integer identity, while the retained path explains which
    namespace/token composition produced it.
    """

    def __init__(self) -> None:
        self._paths: list[tuple[int, ...]] = []
        self._index: dict[tuple[int, ...], int] = {}

    def consume(self, path: Iterable[int]) -> int:
        encoded = tuple(int(token) for token in path)
        token = self._index.get(encoded)
        if token is not None:
            return token
        token = len(self._paths)
        self._paths.append(encoded)
        self._index[encoded] = token
        return token

    def path(self, token: int) -> tuple[int, ...]:
        return self._paths[int(token)]

    def snapshot(self) -> tuple[tuple[int, ...], ...]:
        return tuple(self._paths)

    def __len__(self) -> int:
        return len(self._paths)


class EvolutionMetaGraph:
    """Thread-safe provenance ledger with no dependency on a concrete IR."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._graphs: dict[str, EvolutionGraphRef] = {}
        self._components: dict[EvolutionComponentRef, EvolutionComponent] = {}
        self._events: list[EvolutionEvent] = []
        self._subscribers: list[Subscriber] = []
        self._next_graph = 0
        self._next_sequence = 0
        self._artifact_graphs: dict[int, EvolutionGraphRef] = {}
        self._artifact_components: dict[int, EvolutionComponentRef] = {}

    def _publish(
        self,
        kind: str,
        *,
        graph: EvolutionGraphRef | None = None,
        component: EvolutionComponentRef | None = None,
        sources: Iterable[EvolutionComponentRef] = (),
        detail: Mapping[str, Any] | None = None,
    ) -> EvolutionEvent:
        with self._lock:
            event = EvolutionEvent(
                sequence=self._next_sequence,
                captured_ns=time.perf_counter_ns(),
                kind=str(kind),
                graph=graph,
                component=component,
                sources=tuple(sources),
                detail=dict(detail or {}),
            )
            self._next_sequence += 1
            self._events.append(event)
            subscribers = tuple(self._subscribers)
        for subscriber in subscribers:
            subscriber(event)
        return event

    def open_graph(self, stage: str, label: str = "") -> EvolutionGraphRef:
        with self._lock:
            graph_id = f"{stage}:{self._next_graph}"
            self._next_graph += 1
            graph = EvolutionGraphRef(graph_id, str(stage), str(label or stage))
            self._graphs[graph_id] = graph
        self._publish("graph-open", graph=graph)
        return graph

    def close_graph(self, graph: EvolutionGraphRef) -> EvolutionEvent:
        return self._publish("graph-close", graph=graph)

    def bind_artifact(self, artifact: Any, graph: EvolutionGraphRef) -> None:
        """Associate an in-memory IR object without retaining the object."""

        with self._lock:
            self._artifact_graphs[id(artifact)] = graph

    def graph_for_artifact(self, artifact: Any) -> EvolutionGraphRef | None:
        with self._lock:
            return self._artifact_graphs.get(id(artifact))

    def has_component(self, ref: EvolutionComponentRef) -> bool:
        with self._lock:
            return ref in self._components

    def bind_component(self, artifact: Any, ref: EvolutionComponentRef) -> None:
        with self._lock:
            self._artifact_components[id(artifact)] = ref

    def component_for_artifact(self, artifact: Any) -> EvolutionComponentRef | None:
        with self._lock:
            return self._artifact_components.get(id(artifact))

    def component(
        self,
        graph: EvolutionGraphRef,
        local_id: Any,
        *,
        label: str,
        kind: str,
        attributes: Mapping[str, Any] | None = None,
        consumes: Iterable[EvolutionComponentRef] = (),
        token_id: int | None = None,
    ) -> EvolutionComponentRef:
        ref = EvolutionComponentRef(graph.id, str(local_id))
        record = EvolutionComponent(
            ref,
            str(label),
            str(kind),
            dict(attributes or {}),
            None if token_id is None else int(token_id),
        )
        with self._lock:
            previous = self._components.get(ref)
            self._components[ref] = record
        event_kind = "component-spawn" if previous is None else "component-update"
        self._publish(
            event_kind,
            graph=graph,
            component=ref,
            sources=consumes,
            detail={"label": record.label, "kind": record.kind},
        )
        return ref

    def relationship(
        self,
        graph: EvolutionGraphRef,
        source: EvolutionComponentRef,
        target: EvolutionComponentRef,
        *,
        role: str = "data",
        role_token_id: int | None = None,
    ) -> EvolutionEvent:
        return self._publish(
            "component-link",
            graph=graph,
            component=target,
            sources=(source,),
            detail={
                "role": str(role),
                **(
                    {}
                    if role_token_id is None
                    else {"role_token_id": int(role_token_id)}
                ),
            },
        )

    def handoff(
        self,
        target: EvolutionComponentRef,
        sources: Iterable[EvolutionComponentRef],
        *,
        transformation: str,
        detail: Mapping[str, Any] | None = None,
    ) -> EvolutionEvent:
        target_graph = self._graphs.get(target.graph_id)
        return self._publish(
            "component-handoff",
            graph=target_graph,
            component=target,
            sources=sources,
            detail={"transformation": str(transformation), **dict(detail or {})},
        )

    def subscribe(self, callback: Subscriber, *, replay: bool = False):
        with self._lock:
            self._subscribers.append(callback)
            existing = tuple(self._events) if replay else ()
        for event in existing:
            callback(event)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)

        return unsubscribe

    def snapshot(self) -> EvolutionSnapshot:
        with self._lock:
            return EvolutionSnapshot(
                tuple(self._graphs.values()),
                tuple(self._components.values()),
                tuple(self._events),
            )

    def to_token_multidigraph(self):
        """Project all component and lineage events without collapsing edges."""

        import networkx as nx

        snapshot = self.snapshot()
        graph = nx.MultiDiGraph()
        graph_refs = {item.id: item for item in snapshot.graphs}
        for component in snapshot.components:
            owner = graph_refs.get(component.ref.graph_id)
            graph.add_node(
                (component.ref.graph_id, component.ref.local_id),
                token_id=component.token_id,
                diagnostic=component.label,
                kind=component.kind,
                stage=None if owner is None else owner.stage,
                **dict(component.attributes),
            )
        for event in snapshot.events:
            if event.component is None or not event.sources:
                continue
            target = (event.component.graph_id, event.component.local_id)
            for position, source in enumerate(event.sources):
                graph.add_edge(
                    (source.graph_id, source.local_id),
                    target,
                    key=(event.sequence, position),
                    event=event.kind,
                    sequence=event.sequence,
                    **dict(event.detail),
                )
        return graph


_ACTIVE_EVOLUTION: ContextVar[EvolutionMetaGraph | None] = ContextVar(
    "turing_active_evolution_metagraph",
    default=None,
)


def active_evolution_metagraph() -> EvolutionMetaGraph | None:
    return _ACTIVE_EVOLUTION.get()


@contextmanager
def record_evolution(metagraph: EvolutionMetaGraph | None = None):
    """Attach an optional metagraph to compiler work in this context."""

    graph = metagraph or EvolutionMetaGraph()
    token = _ACTIVE_EVOLUTION.set(graph)
    try:
        yield graph
    finally:
        _ACTIVE_EVOLUTION.reset(token)


def record_fused_program_evolution(
    program: Any,
    *,
    source_graph: EvolutionGraphRef | None = None,
    label: str = "numeric precompile",
) -> EvolutionGraphRef | None:
    """Record an exact value-ID handoff into an existing FusedProgram."""

    metagraph = active_evolution_metagraph()
    if metagraph is None:
        return None
    existing = metagraph.graph_for_artifact(program)
    if existing is not None:
        return existing
    graph = metagraph.open_graph("precompile", label)
    metagraph.bind_artifact(program, graph)

    def source_for(value_id: Any) -> tuple[EvolutionComponentRef, ...]:
        if source_graph is None:
            return ()
        ref = EvolutionComponentRef(source_graph.id, str(value_id))
        return (ref,) if metagraph.has_component(ref) else ()

    for value_id in sorted(program.feeds):
        sources = source_for(value_id)
        target = metagraph.component(
            graph,
            value_id,
            label=f"feed {value_id}",
            kind="feed",
            consumes=sources,
        )
        if sources:
            metagraph.handoff(target, sources, transformation="process-graph-to-precompile")
    for step in program.steps:
        sources = source_for(step.result_id)
        target = metagraph.component(
            graph,
            step.result_id,
            label=str(step.op_name),
            kind="operation",
            attributes={"step_id": int(step.step_id)},
            consumes=sources,
        )
        if sources:
            metagraph.handoff(target, sources, transformation="process-graph-to-precompile")
        for position, value_id in enumerate(step.input_ids):
            input_ref = EvolutionComponentRef(graph.id, str(value_id))
            if metagraph.has_component(input_ref):
                metagraph.relationship(graph, input_ref, target, role=f"arg{position}")
    metagraph.close_graph(graph)
    return graph


def record_control_program_evolution(
    program: Any,
    *,
    label: str = "control program",
) -> EvolutionGraphRef | None:
    """Record the existing control dataclass tree without copying semantics."""

    metagraph = active_evolution_metagraph()
    if metagraph is None:
        return None
    existing = metagraph.graph_for_artifact(program)
    if existing is not None:
        return existing
    graph = metagraph.open_graph("control-ir", label)
    metagraph.bind_artifact(program, graph)
    seen: set[int] = set()

    def visit(value: Any, path: str, parent: EvolutionComponentRef | None = None):
        if not is_dataclass(value) or id(value) in seen:
            return
        seen.add(id(value))
        ref = metagraph.component(
            graph,
            path,
            label=type(value).__name__,
            kind="control",
        )
        metagraph.bind_component(value, ref)
        if parent is not None:
            metagraph.relationship(graph, parent, ref, role="contains")
        for descriptor in fields(value):
            child = getattr(value, descriptor.name)
            if is_dataclass(child):
                visit(child, f"{path}.{descriptor.name}", ref)
            elif isinstance(child, (tuple, list)):
                for index, item in enumerate(child):
                    if is_dataclass(item):
                        visit(item, f"{path}.{descriptor.name}[{index}]", ref)

    visit(program.root, "root")
    return graph


__all__ = [
    "EvolutionComponent",
    "EvolutionComponentRef",
    "EvolutionEvent",
    "EvolutionGraphRef",
    "EvolutionMetaGraph",
    "EvolutionSnapshot",
    "TokenPathAtlas",
    "active_evolution_metagraph",
    "record_evolution",
    "record_fused_program_evolution",
    "record_control_program_evolution",
]
