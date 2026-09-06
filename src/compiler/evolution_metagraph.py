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

    def ingest_event(self, event: EvolutionEvent) -> None:
        """Replay one authoritative event received from an isolated compiler.

        The event keeps its compiler-assigned sequence. This is intentionally
        narrower than `_publish`: it reconstructs just enough ledger state for
        snapshots and subscribers without inventing a second mutation.
        """

        with self._lock:
            if self._events and event.sequence <= self._events[-1].sequence:
                raise ValueError(
                    "external evolution events must arrive in increasing order"
                )
            if event.graph is not None:
                self._graphs[event.graph.id] = event.graph
            if (
                event.component is not None
                and event.kind in {"component-spawn", "component-update"}
            ):
                self._components[event.component] = EvolutionComponent(
                    event.component,
                    str(event.detail.get("label") or event.component.local_id),
                    str(event.detail.get("kind") or "component"),
                    dict(event.detail.get("attributes") or {}),
                    event.detail.get("token_id"),
                )
            self._events.append(event)
            self._next_sequence = max(self._next_sequence, event.sequence + 1)
            subscribers = tuple(self._subscribers)
        for subscriber in subscribers:
            subscriber(event)

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

    def finalize_schedule(
        self,
        graph: EvolutionGraphRef,
        levels: Mapping[Any, int],
        *,
        method: str,
        order: str,
    ) -> EvolutionEvent:
        """Publish one scheduler-authored, immutable component schedule.

        The caller is the IR scheduler that determines program order.  This
        ledger records its result verbatim; observers may replay or present
        the groups but must not reconstruct scheduling from topology.
        """

        normalized = tuple(
            sorted(
                ((str(local_id), int(level)) for local_id, level in levels.items()),
                key=lambda item: (item[1], item[0]),
            )
        )
        distinct_levels = tuple(sorted({level for _local_id, level in normalized}))
        ordinal = {level: index for index, level in enumerate(distinct_levels)}
        return self._publish(
            "graph-schedule-finalized",
            graph=graph,
            detail={
                "schema": "process-graph-schedule-v1",
                "method": str(method),
                "order": str(order),
                "levels": normalized,
                "groups": tuple(
                    (level, ordinal[level]) for level in distinct_levels
                ),
            },
        )

    def finalize_execution_plan(
        self,
        graph: EvolutionGraphRef,
        frames: Iterable[Mapping[str, Any]],
    ) -> EvolutionEvent:
        """Publish the accepted compiled product's structured replay frames.

        Unlike ``finalize_schedule``, this is not a ProcessGraph level map.
        Its caller owns the prepared ControlProgram/deployment product and
        supplies the exact component memberships of each symbolic execution
        frame.  A frame may describe a branch alternative or one structural
        loop iteration; observers must retain that qualification rather than
        claiming a runtime predicate value or trip count was observed.
        """

        normalized = []
        for ordinal, frame in enumerate(frames):
            components = tuple(
                (str(ref.graph_id), str(ref.local_id))
                if isinstance(ref, EvolutionComponentRef)
                else (str(ref[0]), str(ref[1]))
                for ref in frame.get("components", ())
            )
            normalized.append({
                "group": int(frame.get("group", ordinal)),
                "kind": str(frame.get("kind", "execution")),
                "label": str(frame.get("label", "execution")),
                "control_path": tuple(map(str, frame.get("control_path", ()))),
                "loop_depth": int(frame.get("loop_depth", 0)),
                "branch": frame.get("branch"),
                "deployment_region": frame.get("deployment_region"),
                "deployment_lanes": tuple(map(int, frame.get("deployment_lanes", ()))),
                "recurrent": bool(frame.get("recurrent", False)),
                "components": components,
            })
        return self._publish(
            "compiled-execution-plan-finalized",
            graph=graph,
            detail={
                "schema": "compiled-execution-plan-v1",
                "frames": tuple(normalized),
            },
        )

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
            detail={
                "label": record.label,
                "kind": record.kind,
                "attributes": dict(record.attributes),
            },
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


def record_compiled_execution_evolution(
    program: Any,
    *,
    region_graphs: Mapping[int, EvolutionGraphRef | None],
    region_programs: Mapping[int, Any],
    label: str = "accepted compiled execution",
) -> EvolutionGraphRef | None:
    """Record ControlProgram, deployment, and dispatch execution structure.

    This observes the planner-owned product after scheduling and region
    formation.  It does not evaluate predicates or invent loop trip counts:
    conditional arms are labelled alternatives and loops contribute one
    recurrent structural iteration with an explicit backedge.
    """

    metagraph = active_evolution_metagraph()
    if metagraph is None or program is None:
        return None

    from .control_source import (
        CallBlock,
        ConditionalBlock,
        LoopBlock,
        LoopControlBlock,
        ParallelDeployment,
        SequenceBlock,
        StateMachineTick,
        StatementBlock,
        StreamPublishBlock,
        ValidationBlock,
        WhileBlock,
    )
    import re

    graph = metagraph.open_graph("execution-plan", label)
    metagraph.bind_artifact(program, graph)
    component_records = {
        component.ref: component
        for component in metagraph.snapshot().components
    }
    next_id = 0
    frames: list[dict[str, Any]] = []
    region_bounds: dict[int, tuple[EvolutionComponentRef, EvolutionComponentRef]] = {}

    def node(label: str, kind: str, path: tuple[str, ...], **attributes: Any):
        nonlocal next_id
        local_id = next_id
        next_id += 1
        return metagraph.component(
            graph,
            local_id,
            label=label,
            kind=kind,
            attributes={"control_path": path, **attributes},
        )

    def link(source: EvolutionComponentRef, target: EvolutionComponentRef, role: str):
        metagraph.relationship(graph, source, target, role=role)

    def frame(
        control: EvolutionComponentRef,
        *,
        kind: str,
        label: str,
        path: tuple[str, ...],
        components: Iterable[EvolutionComponentRef] = (),
        loop_depth: int = 0,
        branch: str | None = None,
        deployment_region: int | None = None,
        deployment_lanes: Iterable[int] = (),
        recurrent: bool = False,
    ) -> dict[str, Any]:
        return {
            "kind": kind,
            "label": label,
            "control_path": path,
            "components": (control, *tuple(components)),
            "loop_depth": loop_depth,
            "branch": branch,
            "deployment_region": deployment_region,
            "deployment_lanes": tuple(deployment_lanes),
            "recurrent": recurrent,
        }

    def sequence(parts: Iterable[tuple[list[Any], tuple[Any, ...], tuple[Any, ...]]]):
        combined: list[dict[str, Any]] = []
        entries: tuple[EvolutionComponentRef, ...] = ()
        exits: tuple[EvolutionComponentRef, ...] = ()
        for part_frames, part_entries, part_exits in parts:
            if not part_entries:
                continue
            if not entries:
                entries = part_entries
            for previous in exits:
                for following in part_entries:
                    link(previous, following, "control-next")
            combined.extend(part_frames)
            exits = part_exits
        return combined, entries, exits

    def region_fragment(region_index: int, path: tuple[str, ...], loop_depth: int):
        region_index = int(region_index)
        enter = node(
            f"dispatch {region_index}", "dispatch-enter", path,
            region_index=region_index,
        )
        leave = node(
            f"dispatch {region_index} complete", "dispatch-exit", path,
            region_index=region_index,
        )
        region_bounds.setdefault(region_index, (enter, leave))
        region_graph = region_graphs.get(region_index)
        captured = region_programs.get(region_index)
        numeric = getattr(captured, "program", captured)
        region_components: dict[str, EvolutionComponentRef] = {}
        if region_graph is not None:
            region_components = {
                ref.local_id: ref
                for ref in component_records
                if ref.graph_id == region_graph.id
            }
        local_frames = [frame(
            enter, kind="dispatch-enter", label=f"dispatch {region_index}",
            path=path, loop_depth=loop_depth,
        )]
        previous: tuple[EvolutionComponentRef, ...] = (enter,)
        feeds = tuple(
            region_components[str(value_id)]
            for value_id in sorted(getattr(numeric, "feeds", ()) or ())
            if str(value_id) in region_components
        )
        for feed in feeds:
            link(enter, feed, "dispatch-feed")
        if feeds:
            local_frames[0]["components"] += feeds
            previous = feeds
        for step in tuple(getattr(numeric, "steps", ()) or ()):
            ref = region_components.get(str(step.result_id))
            if ref is None:
                continue
            local_frames.append(frame(
                ref,
                kind="dispatch-instruction",
                label=str(step.op_name),
                path=path,
                loop_depth=loop_depth,
            ))
            previous = (ref,)
        outputs = tuple(
            region_components[str(value_id)]
            for value_id in (getattr(numeric, "outputs", {}) or {}).values()
            if str(value_id) in region_components
        )
        for source in outputs or previous:
            link(source, leave, "dispatch-result")
        local_frames.append(frame(
            leave, kind="dispatch-exit",
            label=f"dispatch {region_index} complete", path=path,
            loop_depth=loop_depth,
        ))
        return local_frames, (enter,), (leave,)

    def walk(
        block: Any,
        path: tuple[str, ...] = ("root",),
        loop_depth: int = 0,
        branch: str | None = None,
    ):
        if isinstance(block, SequenceBlock):
            return sequence(
                walk(child, (*path, f"sequence[{index}]"), loop_depth, branch)
                for index, child in enumerate(block.blocks)
            )
        if isinstance(block, StatementBlock):
            parts = []
            for index, line in enumerate(block.lines):
                match = re.fullmatch(r"__scheduled_region_(\d+)__", str(line))
                if match:
                    parts.append(region_fragment(
                        int(match.group(1)), (*path, f"statement[{index}]"), loop_depth,
                    ))
                    continue
                ref = node(str(line), "control-statement", path, branch=branch)
                parts.append(([
                    frame(ref, kind="control-statement", label=str(line), path=path,
                          loop_depth=loop_depth, branch=branch)
                ], (ref,), (ref,)))
            return sequence(parts)
        if isinstance(block, ConditionalBlock):
            test = node(
                f"if value {block.predicate_value_id}", "branch", path,
                predicate_value_id=int(block.predicate_value_id),
                expect_true=bool(block.expect_true),
            )
            merge = node("branch merge", "branch-merge", path)
            true_part = walk(block.body, (*path, "true"), loop_depth, "true")
            false_part = (
                walk(block.orelse, (*path, "false"), loop_depth, "false")
                if block.orelse is not None else ([], (), ())
            )
            for entry in true_part[1]:
                link(test, entry, "branch-true")
            for entry in false_part[1]:
                link(test, entry, "branch-false")
            for exit_ref in (*true_part[2], *false_part[2]):
                link(exit_ref, merge, "branch-merge")
            if not false_part[1]:
                link(test, merge, "branch-false")
            return ([
                frame(test, kind="branch", label="predicate", path=path,
                      loop_depth=loop_depth),
                *true_part[0], *false_part[0],
                frame(merge, kind="branch-merge", label="branch merge", path=path,
                      loop_depth=loop_depth),
            ], (test,), (merge,))
        if isinstance(block, LoopBlock):
            header = node(
                f"for {block.induction}: {block.start}:{block.stop}:{block.step}",
                "loop-header", path,
                induction=str(block.induction), start=str(block.start),
                stop=str(block.stop), step=str(block.step),
                comparison=str(block.comparison),
                parallel_iterations=bool(block.parallel_iterations),
                recursion_region_id=block.recursion_region_id,
            )
            body = walk(block.body, (*path, "iteration"), loop_depth + 1, branch)
            latch = node("loop latch", "loop-latch", path)
            exit_ref = node("loop exit", "loop-exit", path)
            for entry in body[1]:
                link(header, entry, "loop-body")
            if not body[1]:
                link(header, latch, "loop-body")
            for body_exit in body[2]:
                link(body_exit, latch, "loop-latch")
            link(latch, header, "loop-back")
            link(header, exit_ref, "loop-exit")
            return ([
                frame(header, kind="loop-header", label="loop test", path=path,
                      loop_depth=loop_depth, recurrent=True),
                *body[0],
                frame(latch, kind="loop-latch", label="next iteration", path=path,
                      loop_depth=loop_depth + 1, recurrent=True),
                frame(exit_ref, kind="loop-exit", label="loop exit", path=path,
                      loop_depth=loop_depth),
            ], (header,), (exit_ref,))
        if isinstance(block, WhileBlock):
            condition = walk(block.condition, (*path, "condition"), loop_depth, branch)
            test = node(
                f"while value {block.predicate_value_id}", "loop-header", path,
                predicate_value_id=int(block.predicate_value_id),
                recursion_region_id=block.recursion_region_id,
            )
            body = walk(block.body, (*path, "iteration"), loop_depth + 1, branch)
            exit_ref = node("while exit", "loop-exit", path)
            for source in condition[2]:
                link(source, test, "loop-test")
            for entry in body[1]:
                link(test, entry, "loop-body")
            for source in body[2]:
                for entry in condition[1]:
                    link(source, entry, "loop-back")
            link(test, exit_ref, "loop-exit")
            return ([
                *condition[0],
                frame(test, kind="loop-header", label="while test", path=path,
                      loop_depth=loop_depth, recurrent=True),
                *body[0],
                frame(exit_ref, kind="loop-exit", label="while exit", path=path,
                      loop_depth=loop_depth),
            ], condition[1] or (test,), (exit_ref,))
        if isinstance(block, ParallelDeployment):
            deploy = node(
                "parallel deploy", "deployment", path,
                lane_count=len(block.lanes),
                schedule_preference=str(block.schedule_preference),
            )
            join = node("deployment join", "deployment-join", path)
            lanes = [
                walk(lane, (*path, f"lane[{index}]"), loop_depth, branch)
                for index, lane in enumerate(block.lanes)
            ]
            for index, lane in enumerate(lanes):
                for entry in lane[1]:
                    link(deploy, entry, f"deployment-lane-{index}")
                for exit_lane in lane[2]:
                    link(exit_lane, join, "deployment-join")
            parallel_frames: list[dict[str, Any]] = []
            width = max((len(lane[0]) for lane in lanes), default=0)
            for offset in range(width):
                members = []
                for lane in lanes:
                    if offset < len(lane[0]):
                        members.extend(lane[0][offset]["components"])
                parallel_frames.append({
                    "kind": "deployment-lanes",
                    "label": "parallel deployment lanes",
                    "control_path": path,
                    "components": tuple(dict.fromkeys(members)),
                    "loop_depth": loop_depth,
                    "branch": branch,
                    "deployment_lanes": tuple(range(len(lanes))),
                })
            return ([
                frame(deploy, kind="deployment", label="deploy", path=path,
                      loop_depth=loop_depth),
                *parallel_frames,
                frame(join, kind="deployment-join", label="join", path=path,
                      loop_depth=loop_depth),
            ], (deploy,), (join,))
        if isinstance(block, CallBlock):
            call = node(f"call {block.callsite_id}", "call", path,
                        callsite_id=int(block.callsite_id))
            callee = walk(block.callee, (*path, "callee"), loop_depth, branch)
            returned = node(f"return {block.callsite_id}", "call-return", path)
            for entry in callee[1]:
                link(call, entry, "call-enter")
            for source in callee[2]:
                link(source, returned, "call-return")
            return ([
                frame(call, kind="call", label="call", path=path,
                      loop_depth=loop_depth),
                *callee[0],
                frame(returned, kind="call-return", label="return", path=path,
                      loop_depth=loop_depth),
            ], (call,), (returned,))
        if isinstance(block, StateMachineTick):
            dispatch = node(f"state {block.state}", "state-dispatch", path,
                            state=str(block.state))
            merge = node("state merge", "state-merge", path)
            case_frames = []
            for name, case in block.cases:
                part = walk(case, (*path, f"case[{name}]"), loop_depth, str(name))
                for entry in part[1]:
                    link(dispatch, entry, f"state-{name}")
                for source in part[2]:
                    link(source, merge, "state-merge")
                case_frames.extend(part[0])
            if block.default is not None:
                part = walk(block.default, (*path, "default"), loop_depth, "default")
                for entry in part[1]:
                    link(dispatch, entry, "state-default")
                for source in part[2]:
                    link(source, merge, "state-merge")
                case_frames.extend(part[0])
            return ([
                frame(dispatch, kind="state-dispatch", label="state dispatch", path=path,
                      loop_depth=loop_depth),
                *case_frames,
                frame(merge, kind="state-merge", label="state merge", path=path,
                      loop_depth=loop_depth),
            ], (dispatch,), (merge,))

        kind = (
            "loop-control" if isinstance(block, LoopControlBlock)
            else "validation" if isinstance(block, ValidationBlock)
            else "stream-publish" if isinstance(block, StreamPublishBlock)
            else type(block).__name__.lower()
        )
        ref = node(type(block).__name__, kind, path, branch=branch)
        return ([frame(ref, kind=kind, label=type(block).__name__, path=path,
                       loop_depth=loop_depth, branch=branch)], (ref,), (ref,))

    root_frames, _entries, _exits = walk(program.root)
    frames.extend(root_frames)

    # Deployment-region records are durable parallelism proofs beside the
    # lexical control tree. Connect those proofs to the exact dispatch
    # boundaries without replaying them as an invented second execution.
    for deployment in tuple(getattr(program, "deployment_regions", ()) or ()):
        deploy = node(
            f"deployment {deployment.region_id}", "deployment-proof",
            ("deployments", str(deployment.region_id)),
            deployment_region=int(deployment.region_id),
            schedule=str(deployment.schedule),
            schedule_preference=str(deployment.schedule_preference),
            iteration_space=deployment.iteration_space,
            scale=int(deployment.scale),
            join_mode=deployment.join.mode.value,
            origin=str(deployment.origin),
        )
        joined = node(
            f"deployment {deployment.region_id} join", "deployment-proof-join",
            ("deployments", str(deployment.region_id)),
            deployment_region=int(deployment.region_id),
            join_mode=deployment.join.mode.value,
        )
        deployment_frame_indices: list[int] = []
        for lane in deployment.lanes:
            lane_ref = node(
                f"lane {lane.index}", "deployment-lane",
                ("deployments", str(deployment.region_id), f"lane[{lane.index}]"),
                deployment_region=int(deployment.region_id),
                deployment_lane=int(lane.index),
            )
            link(deploy, lane_ref, "deployment-lane")
            for region_index in lane.region_indices:
                bounds = region_bounds.get(int(region_index))
                if bounds is None:
                    continue
                link(lane_ref, bounds[0], "dispatch-membership")
                link(bounds[1], joined, "deployment-join")
                matching = [
                    index
                    for index, authored_frame in enumerate(frames)
                    if bounds[0] in authored_frame["components"]
                    or bounds[1] in authored_frame["components"]
                ]
                deployment_frame_indices.extend(matching)
                if matching:
                    first = frames[min(matching)]
                    first["components"] = tuple(dict.fromkeys((
                        *first["components"], deploy, lane_ref,
                    )))
                    first["deployment_region"] = int(deployment.region_id)
                    first["deployment_lanes"] = tuple(dict.fromkeys((
                        *first.get("deployment_lanes", ()), int(lane.index),
                    )))
        if deployment_frame_indices:
            final_frame = frames[max(deployment_frame_indices)]
            final_frame["components"] = tuple(dict.fromkeys((
                *final_frame["components"], joined,
            )))
            final_frame["deployment_region"] = int(deployment.region_id)

    metagraph.finalize_execution_plan(graph, frames)
    metagraph.close_graph(graph)
    return graph


def extend_compiled_execution_lineage(
    graph: EvolutionGraphRef | None,
) -> EvolutionEvent | None:
    """Attach later SSA/backend descendants to existing execution frames."""

    metagraph = active_evolution_metagraph()
    if metagraph is None or graph is None:
        return None
    snapshot = metagraph.snapshot()
    authored = next((
        event
        for event in reversed(snapshot.events)
        if event.kind == "compiled-execution-plan-finalized"
        and event.graph == graph
    ), None)
    if authored is None:
        return None
    descendants: dict[EvolutionComponentRef, set[EvolutionComponentRef]] = {}
    ancestors: dict[EvolutionComponentRef, set[EvolutionComponentRef]] = {}
    for event in snapshot.events:
        if event.kind != "component-handoff" or event.component is None:
            continue
        for source in event.sources:
            descendants.setdefault(source, set()).add(event.component)
            ancestors.setdefault(event.component, set()).add(source)

    expanded = []
    for raw_frame in authored.detail.get("frames", ()):
        roots = {
            EvolutionComponentRef(str(graph_id), str(local_id))
            for graph_id, local_id in raw_frame.get("components", ())
        }
        lineage_ancestors = set(roots)
        pending = list(roots)
        while pending:
            target = pending.pop()
            for source in ancestors.get(target, ()):
                if source not in lineage_ancestors:
                    lineage_ancestors.add(source)
                    pending.append(source)
        lineage_descendants = set(roots)
        pending = list(roots)
        while pending:
            source = pending.pop()
            for target in descendants.get(source, ()):
                if target not in lineage_descendants:
                    lineage_descendants.add(target)
                    pending.append(target)
        closure = lineage_ancestors | lineage_descendants
        expanded.append({
            **dict(raw_frame),
            "components": tuple(sorted(closure)),
        })
    return metagraph.finalize_execution_plan(graph, expanded)


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
    "record_compiled_execution_evolution",
    "extend_compiled_execution_lineage",
    "record_fused_program_evolution",
    "record_control_program_evolution",
]
