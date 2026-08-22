"""Deterministic project division at authored function-call boundaries.

The compiler may retain whole-project source and name context without keeping
every function's ProcessGraph deployment resident at once.  This module turns
the resolved FunctionTable call graph into dependency-ordered compilation
units.  A unit contains exactly one call-graph strongly connected component:
ordinary functions remain independently compilable, while mutual recursion is
kept together because no correct linker boundary exists inside the cycle.

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


def plan_compilation_units(function_table: Any) -> CompilationUnitPlan:
    """Divide all graph-backed functions into dependency-first SCC units.

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

    components = _strong_components(entries, edges, names)
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
            recursive=(
                len(component) > 1
                or any(reference in edges[reference] for reference in component)
            ),
        ))
    return CompilationUnitPlan(tuple(units), reference_to_unit)


def record_compilation_unit_plan(graph: Any) -> CompilationUnitPlan:
    """Plan the resolved project and retain its portable receipt on the graph."""

    plan = plan_compilation_units(graph.function_table)
    graph.G.graph["compilation_unit_plan"] = plan.to_mapping()
    return plan


__all__ = [
    "CompilationUnit",
    "CompilationUnitPlan",
    "plan_compilation_units",
    "record_compilation_unit_plan",
]
