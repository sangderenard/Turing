"""Deterministic project division at authored function-call boundaries.

The compiler may retain whole-project source and name context without keeping
every function's ProcessGraph deployment resident at once.  This module turns
the resolved FunctionTable call graph into dependency-ordered compilation
units. A unit is one strongly connected component of the planning relation:
authored calls plus explicitly indivisible specialization edges. Ordinary
functions remain independent; cycles and unresolved record-specialization
components stay together until a correct native frontier exists.

Qualified authored names are the durable identities.  Function references are
carried only as the already-resolved correlation needed to select graphs in
this particular compilation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


_CALL_REFERENCE_FIELDS = (
    "callee_ref",
    "method_ref",
    "constructor_ref",
    "first_class_function_ref",
)


@dataclass(frozen=True)
class CompilationUnit:
    """One independently lowerable call-graph component."""

    qualified_names: tuple[str, ...]
    function_references: tuple[int, ...]
    dependency_units: tuple[int, ...]
    external_references: tuple[int, ...]
    dynamic_call_nodes: tuple[tuple[int, int], ...]
    source_nodes: int
    recursive: bool

    def to_mapping(self) -> dict[str, Any]:
        return {
            "qualified_names": list(self.qualified_names),
            "function_references": list(self.function_references),
            "dependency_units": list(self.dependency_units),
            "external_references": list(self.external_references),
            "dynamic_call_nodes": [list(item) for item in self.dynamic_call_nodes],
            "source_nodes": int(self.source_nodes),
            "recursive": bool(self.recursive),
        }


@dataclass(frozen=True)
class CompilationUnitPlan:
    """Dependency-first division of a resolved project FunctionTable."""

    units: tuple[CompilationUnit, ...]
    reference_to_unit: Mapping[int, int]

    def unit_for_reference(self, reference: int) -> CompilationUnit:
        return self.units[self.reference_to_unit[int(reference)]]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": "turing.compilation-unit-plan.v1",
            "order": "dependencies-first",
            "units": [unit.to_mapping() for unit in self.units],
            "reference_to_unit": {
                str(reference): int(unit)
                for reference, unit in sorted(self.reference_to_unit.items())
            },
        }


def compilation_growth_priorities(
    units: Iterable[Mapping[str, Any]], *, accepted_units: Iterable[int] = (),
) -> tuple[dict[str, Any], ...]:
    """Rank ready cuts by the costly enclosing cone they can relieve.

    Source-node counts are a structural estimate, not measured compile time.
    Walk only the supplied SCC dependency edges. Native scheduling supplies
    resolved FunctionTable edges; authored-source triage remains advisory.
    Each cone counts a shared
    dependency once; cycles must already have been contracted by the planner.
    The receipt retains the enclosing-to-interior chain that chose each cut.
    """
    selected = tuple(units)
    accepted = set(map(int, accepted_units))
    if any(not 0 <= index < len(selected) for index in accepted):
        raise ValueError("accepted compilation growth unit lies outside the plan")
    cones: dict[int, set[int]] = {}
    active: set[int] = set()

    def cone(index: int) -> set[int]:
        if index in active:
            raise ValueError("compilation growth requires a contracted dependency DAG")
        if index in cones:
            return cones[index]
        active.add(index)
        members = {index}
        for dependency in selected[index].get("dependency_units", ()):
            dependency = int(dependency)
            if not 0 <= dependency < len(selected):
                raise ValueError("compilation growth dependency lies outside the plan")
            members.update(cone(dependency))
        active.remove(index)
        cones[index] = members
        return members

    source_weights = [max(0, int(unit.get("source_nodes", 0))) for unit in selected]
    weights = [0 if index in accepted else weight for index, weight in enumerate(source_weights)]
    for index in range(len(selected)):
        cone(index)
    costs = [sum(weights[member] for member in cones[index]) for index in range(len(selected))]
    owners = [max((owner for owner in cones if index in cones[owner]),
                  key=lambda owner: (costs[owner], weights[owner], -owner))
              for index in range(len(selected))]
    receipts = []
    for index, owner in enumerate(owners):
        chain = [owner]
        while chain[-1] != index:
            candidates = [int(dependency) for dependency in selected[chain[-1]].get("dependency_units", ())
                          if index in cones[int(dependency)]]
            chain.append(max(candidates, key=lambda dependency: (costs[dependency], -dependency)))
        receipts.append({"unit_index": index, "source_nodes": source_weights[index],
                         "remaining_source_nodes": weights[index],
                         "dependency_cone_source_nodes": costs[index],
                         "enclosing_cone_source_nodes": costs[owner],
                         "enclosing_unit_count": sum(index in members for other, members in cones.items()
                                                     if other != index and other not in accepted),
                         "repeated_source_nodes_avoided": costs[index] *
                             sum(index in members for other, members in cones.items()
                                 if other != index and other not in accepted),
                         "enclosing_unit_chain": chain,
                         "reason": "remaining-dependency-cone-source-nodes"})
    return tuple(receipts)


def _entry_graph(entry: Any) -> Any | None:
    return getattr(getattr(entry, "graph", None), "G", None)


def _reference_sort_key(reference: int, names: Mapping[int, str]) -> tuple[str, int]:
    return names[int(reference)], int(reference)


def _call_references(graph: Any) -> tuple[set[int], list[int]]:
    references: set[int] = set()
    dynamic_nodes: list[int] = []
    for node_id, data in graph.nodes(data=True):
        attributes = data.get("attributes") or {}
        selected = None
        for field in _CALL_REFERENCE_FIELDS:
            if attributes.get(field) is not None:
                selected = int(attributes[field])
                references.add(selected)
        operation = str(data.get("op") or data.get("type") or "").casefold()
        if operation in {"call", "methodcall", "functioncall"} and selected is None:
            dynamic_nodes.append(int(node_id))
    return references, dynamic_nodes


def _strong_components(
    references: Iterable[int],
    edges: Mapping[int, set[int]],
    names: Mapping[int, str],
) -> tuple[tuple[int, ...], ...]:
    """Tarjan SCCs with deterministic source-name traversal."""

    index = 0
    indices: dict[int, int] = {}
    lowlinks: dict[int, int] = {}
    stack: list[int] = []
    stacked: set[int] = set()
    components: list[tuple[int, ...]] = []

    def visit(reference: int) -> None:
        nonlocal index
        indices[reference] = index
        lowlinks[reference] = index
        index += 1
        stack.append(reference)
        stacked.add(reference)
        for callee in sorted(
            edges.get(reference, ()),
            key=lambda item: _reference_sort_key(item, names),
        ):
            if callee not in indices:
                visit(callee)
                lowlinks[reference] = min(lowlinks[reference], lowlinks[callee])
            elif callee in stacked:
                lowlinks[reference] = min(lowlinks[reference], indices[callee])
        if lowlinks[reference] != indices[reference]:
            return
        component = []
        while True:
            member = stack.pop()
            stacked.remove(member)
            component.append(member)
            if member == reference:
                break
        components.append(tuple(sorted(
            component, key=lambda item: _reference_sort_key(item, names),
        )))

    for reference in sorted(
        references, key=lambda item: _reference_sort_key(item, names),
    ):
        if reference not in indices:
            visit(reference)
    return tuple(components)


def plan_compilation_units(
    function_table: Any, *, indivisible_edges: Iterable[tuple[int, int]] = (),
) -> CompilationUnitPlan:
    """Divide graph-backed functions into dependency-first SCC units.

    The input must already have completed source ingestion and topology
    reduction, because resolved ``callee_ref``/``method_ref`` correlations are
    what make the division exact.  No ProcessGraph is copied by this pass.
    """

    entries = {
        int(entry.reference.address): entry
        for entry in function_table
        if _entry_graph(entry) is not None
    }
    names = {
        reference: str(entry.qualified_name)
        for reference, entry in entries.items()
    }
    internal_references = set(entries)
    edges: dict[int, set[int]] = {}
    external: dict[int, set[int]] = {}
    dynamic: dict[int, tuple[int, ...]] = {}
    source_nodes: dict[int, int] = {}
    for reference, entry in entries.items():
        graph = _entry_graph(entry)
        called, dynamic_nodes = _call_references(graph)
        edges[reference] = called & internal_references
        external[reference] = called - internal_references
        dynamic[reference] = tuple(sorted(dynamic_nodes))
        source_nodes[reference] = int(graph.number_of_nodes())

    # A native frontier is legal only after its callsite specialization has
    # settled. Contract an explicitly coupled edge by adding its reverse to
    # the planning relation, not by expanding a worker's dependency closure.
    # The semantic call graph stays unchanged (including recursion claims).
    semantic_components = _strong_components(entries, edges, names)
    recursive_references = {
        member for component in semantic_components
        if len(component) > 1 or any(member in edges[member] for member in component)
        for member in component
    }
    planning_edges = {reference: set(callees) for reference, callees in edges.items()}
    for caller, callee in indivisible_edges:
        caller, callee = int(caller), int(callee)
        if caller not in entries or callee not in edges.get(caller, ()):
            raise ValueError(f"indivisible edge is not a resolved authored call: {(caller, callee)!r}")
        planning_edges[callee].add(caller)
    components = _strong_components(entries, planning_edges, names)
    component_by_reference = {
        reference: component_index
        for component_index, component in enumerate(components)
        for reference in component
    }
    dependencies = {
        component_index: {
            component_by_reference[callee]
            for caller in component
            for callee in edges[caller]
            if component_by_reference[callee] != component_index
        }
        for component_index, component in enumerate(components)
    }
    component_key = {
        index: tuple(names[reference] for reference in component)
        for index, component in enumerate(components)
    }
    ordered_components: list[int] = []
    visited: set[int] = set()

    def order(component_index: int) -> None:
        if component_index in visited:
            return
        visited.add(component_index)
        for dependency in sorted(
            dependencies[component_index], key=component_key.__getitem__,
        ):
            order(dependency)
        ordered_components.append(component_index)

    for component_index in sorted(component_key, key=component_key.__getitem__):
        order(component_index)

    output_index = {
        component_index: index
        for index, component_index in enumerate(ordered_components)
    }
    units = []
    reference_to_unit: dict[int, int] = {}
    for component_index in ordered_components:
        component = components[component_index]
        unit_index = output_index[component_index]
        for reference in component:
            reference_to_unit[reference] = unit_index
        units.append(CompilationUnit(
            qualified_names=tuple(names[reference] for reference in component),
            function_references=component,
            dependency_units=tuple(sorted(
                (output_index[item] for item in dependencies[component_index]),
            )),
            external_references=tuple(sorted({
                item for reference in component for item in external[reference]
            })),
            dynamic_call_nodes=tuple(sorted(
                (reference, node_id)
                for reference in component
                for node_id in dynamic[reference]
            )),
            source_nodes=sum(source_nodes[reference] for reference in component),
            recursive=bool(set(component) & recursive_references),
        ))
    return CompilationUnitPlan(tuple(units), reference_to_unit)


def native_interface_coupling_edges(function_table: Any) -> tuple[dict[str, Any], ...]:
    """Close identity-changing record transfers over exact forwarding edges.

    root(Actual)->relay(Declared)->leaf(Declared) formerly published generic
    headers and silently returned 16 instead of 59. The ordinary planner
    specializes these bodies to the actual record. Keep that causal component
    together until its accepted ABI exists; unrelated scalar calls remain
    separate. This is finite reachability over parameter identities already
    present in the source graphs, not matching physical fields by name/type.
    """
    from collections import deque
    from .glsl_deployment_strategy import (
        _callsite_parameter_name, _expanded_callsite_argument_edges,
        _method_parameter_layout,
    )

    entries = {int(entry.reference.address): entry for entry in function_table
               if _entry_graph(entry) is not None}
    forwarding: dict[tuple[int, str], list[dict[str, Any]]] = {}
    coupled: dict[tuple[int, int, int, str], dict[str, Any]] = {}
    pending = deque()
    visited = set()
    for caller_ref, entry in entries.items():
        caller = entry.graph
        records = caller.G.graph.get("parameter_record_abi") or {}
        identities = caller.G.graph.get("identity_table") or {}
        record_by_value = {}
        for parameter, record in records.items():
            for value_id in identities.get(parameter, ()):
                record_by_value.setdefault(int(value_id), []).append((str(parameter), record))
        for callsite, data in caller.G.nodes(data=True):
            attrs = data.get("attributes") or {}
            reference = attrs.get("callee_ref", attrs.get("method_ref"))
            if reference is None or int(reference) not in entries:
                continue
            callee_ref = int(reference)
            callee = entries[callee_ref].graph
            receiver, positional, _all = _method_parameter_layout(callee.G)
            declared = callee.G.graph.get("parameter_record_abi") or {}
            for actual, role in _expanded_callsite_argument_edges(caller, int(callsite)):
                parameter = _callsite_parameter_name(str(role), receiver, positional)
                matches = record_by_value.get(int(actual), ())
                if parameter not in declared or len(matches) != 1:
                    continue
                caller_parameter, actual_record = matches[0]
                row = {"caller_reference": caller_ref, "callsite_id": int(callsite),
                       "actual_value_id": int(actual), "caller_parameter": caller_parameter,
                       "callee_reference": callee_ref, "callee_parameter": str(parameter),
                       "actual_record_identity": str(actual_record["identity"]),
                       "declared_record_identity": str(declared[parameter]["identity"])}
                forwarding.setdefault((caller_ref, caller_parameter), []).append(row)
                if row["actual_record_identity"] != row["declared_record_identity"]:
                    coupled[(caller_ref, int(callsite), callee_ref, str(parameter))] = row
                    pending.append((callee_ref, str(parameter)))
    while pending:
        endpoint = pending.popleft()
        if endpoint in visited:
            continue
        visited.add(endpoint)
        for row in forwarding.get(endpoint, ()):
            key = (row["caller_reference"], row["callsite_id"],
                   row["callee_reference"], row["callee_parameter"])
            coupled[key] = row
            pending.append((row["callee_reference"], row["callee_parameter"]))
    return tuple(coupled[key] for key in sorted(coupled))


def record_compilation_unit_plan(
    graph: Any, *, indivisible_edges: Iterable[tuple[int, int]] = (),
) -> CompilationUnitPlan:
    """Plan the resolved project and retain its portable receipt on the graph."""

    plan = plan_compilation_units(graph.function_table, indivisible_edges=indivisible_edges)
    graph.G.graph["compilation_unit_plan"] = plan.to_mapping()
    return plan


__all__ = [
    "CompilationUnit",
    "CompilationUnitPlan",
    "compilation_growth_priorities",
    "plan_compilation_units",
    "record_compilation_unit_plan",
    "native_interface_coupling_edges",
]
