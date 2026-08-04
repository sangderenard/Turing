"""Explicit reference tables carried by a compiled ProcessGraph shell.

The ProcessGraph remains the authority for topology.  These tables give a
backend-facing shell compact, monotonically indexed views of the references
that survive at its boundary:

* functions visible through the shared function tables;
* literal constants owned by this graph;
* input, output, and field-addressable memory references;
* correlations back to ProcessGraph nodes and source references.

They deliberately contain no allocation policy and perform no lowering.
Backends may later replace the local IDs with addresses, bindings, offsets, or
inlined definitions without losing the source correlation.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class ShellFunctionReference:
    """One shell-local function-table slot."""

    index: int
    namespace: str
    source_address: int | None
    name: str
    qualified_name: str
    graph_backed: bool
    external: bool


@dataclass(frozen=True)
class ShellConstantReference:
    """One literal constant owned by the shell graph."""

    index: int
    value: Any
    value_type: str


@dataclass(frozen=True)
class ShellMemoryReference:
    """One boundary or derived storage reference."""

    index: int
    graph_node_id: Any
    name: str
    roles: tuple[str, ...]
    base_node_id: Any | None = None
    field_name: str | None = None


@dataclass(frozen=True)
class ShellReferenceCorrelation:
    """Trace one local table slot back to its ProcessGraph origin."""

    table: str
    index: int
    graph_node_id: Any | None
    source_kind: str
    source_reference: Any | None = None
    source_name: str | None = None


@dataclass(frozen=True)
class ShellRecursionReference:
    """One shell-local reference to a cached recursive SCC."""

    index: int
    region_id: int
    lower_as: str
    members: tuple[int, ...]
    control_ir: bool = True
    control_members: tuple[int, ...] = ()
    incoming: tuple[tuple[int, int, str], ...] = ()
    outgoing: tuple[tuple[int, int, str], ...] = ()
    feedback: tuple[tuple[int, int, str], ...] = ()


@dataclass
class ShellReferenceTables:
    """Ordinary indexed lists installed on one deployment shell."""

    functions: list[ShellFunctionReference]
    constants: list[ShellConstantReference]
    memory: list[ShellMemoryReference]
    correlations: list[ShellReferenceCorrelation]
    recursion: list[ShellRecursionReference] = field(default_factory=list)

    def copy(self) -> "ShellReferenceTables":
        return ShellReferenceTables(
            functions=list(self.functions),
            constants=list(self.constants),
            memory=list(self.memory),
            correlations=list(self.correlations),
            recursion=list(self.recursion),
        )


@dataclass(frozen=True)
class MapDependencyRegions:
    """Function compartments retained for execution and object-map reasons.

    ``runtime`` is the strict call closure of one entrypoint. ``mapped`` is
    the set of graph-backed methods named by retained class maps. ``retained``
    is their union, while ``map_only`` keeps non-runtime methods identifiable
    without pretending they are part of the current execution closure.
    """

    runtime: tuple[int, ...]
    mapped: tuple[int, ...]
    retained: tuple[int, ...]
    map_only: tuple[int, ...]
    bindings: tuple[tuple[str, int], ...]


PermissionEvaluator = Callable[[str, tuple[str, ...]], bool]


@dataclass(frozen=True)
class ClassNavigationMember:
    """One dot-addressable class member in the navigation LUT."""

    name: str
    identity: str
    kind: str
    storage: str | None
    function_reference: int | None
    permissions: tuple[str, ...]


@dataclass(frozen=True)
class ClassNavigationRecord:
    """One class's static space and method/function-table references."""

    identity: str
    permissions: tuple[str, ...]
    members: tuple[ClassNavigationMember, ...]
    instantiation_functions: tuple[int, ...]


@dataclass(frozen=True)
class ClassNavigationTable:
    """Lookup table for class construction and ``.`` member navigation."""

    classes: tuple[ClassNavigationRecord, ...]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "classes": [
                {
                    "identity": record.identity,
                    "permissions": list(record.permissions),
                    "instantiation_functions": list(
                        record.instantiation_functions
                    ),
                    "members": [
                        {
                            "name": member.name,
                            "identity": member.identity,
                            "kind": member.kind,
                            "storage": member.storage,
                            "function_reference": member.function_reference,
                            "permissions": list(member.permissions),
                        }
                        for member in record.members
                    ],
                }
                for record in self.classes
            ]
        }

    def class_record(self, identity: str) -> ClassNavigationRecord:
        matches = [item for item in self.classes if item.identity == identity]
        if len(matches) != 1:
            raise KeyError(f"unknown or ambiguous class identity {identity!r}")
        return matches[0]

    @staticmethod
    def _require(
        evaluator: PermissionEvaluator,
        identity: str,
        permissions: tuple[str, ...],
    ) -> None:
        if not evaluator(identity, permissions):
            raise PermissionError(f"access denied to {identity!r}")

    def instantiate(
        self,
        class_identity: str,
        evaluator: PermissionEvaluator,
    ) -> tuple[int, ...]:
        """Return the permitted constructor function-table references."""

        record = self.class_record(class_identity)
        self._require(evaluator, record.identity, record.permissions)
        constructors = {
            member.function_reference: member
            for member in record.members
            if member.name in {"__new__", "__init__"}
            and member.function_reference is not None
        }
        for reference in record.instantiation_functions:
            member = constructors[reference]
            self._require(evaluator, member.identity, member.permissions)
        return record.instantiation_functions

    def resolve_dot(
        self,
        class_identity: str,
        member_name: str,
        evaluator: PermissionEvaluator,
    ) -> ClassNavigationMember:
        """Resolve ``class_identity.member_name`` after permission checks."""

        record = self.class_record(class_identity)
        self._require(evaluator, record.identity, record.permissions)
        matches = [item for item in record.members if item.name == member_name]
        if len(matches) != 1:
            raise KeyError(
                f"unknown or ambiguous member {class_identity}.{member_name}"
            )
        member = matches[0]
        self._require(evaluator, member.identity, member.permissions)
        return member


def build_class_navigation_table(graph: Any) -> ClassNavigationTable:
    """Bind ingested class-map identities to the existing function table."""

    function_table = getattr(graph, "function_table", None)
    if function_table is None:
        raise ValueError("class navigation requires a function table")
    records = []
    for object_map in (graph.G.graph.get("map_ir") or {}).get("objects", ()):
        class_identity = str(object_map["class_name"])
        members = []
        for attribute in object_map.get("attributes", ()):
            members.append(ClassNavigationMember(
                name=str(attribute["name"]),
                identity=str(attribute["identity"]),
                kind="attribute",
                storage=str(attribute["storage"]),
                function_reference=None,
                permissions=tuple(attribute.get("permissions", ())),
            ))
        for method in object_map.get("methods", ()):
            identity = str(method["graph_identity"])
            try:
                entry = function_table.entry(identity)
            except KeyError:
                reference = None
            else:
                reference = (
                    int(entry.reference.address) if entry.graph is not None else None
                )
            members.append(ClassNavigationMember(
                name=str(method["name"]),
                identity=identity,
                kind="method",
                storage=None,
                function_reference=reference,
                permissions=tuple(method.get("permissions", ())),
            ))
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError(f"class {class_identity!r} has ambiguous member names")
        constructors = tuple(
            member.function_reference
            for constructor_name in ("__new__", "__init__")
            for member in members
            if member.name == constructor_name
            and member.function_reference is not None
        )
        records.append(ClassNavigationRecord(
            identity=class_identity,
            permissions=tuple(object_map.get("permissions", ())),
            members=tuple(members),
            instantiation_functions=constructors,
        ))
    identities = [record.identity for record in records]
    if len(identities) != len(set(identities)):
        raise ValueError("class navigation contains duplicate class identities")
    return ClassNavigationTable(tuple(records))


def build_map_dependency_regions(graph: Any, entrypoint: str) -> MapDependencyRegions:
    """Combine strict runtime closure with map-level class retention."""

    function_table = getattr(graph, "function_table", None)
    if function_table is None:
        raise ValueError("dependency regions require a function table")
    try:
        entry = function_table.entry(entrypoint)
    except KeyError as exc:
        raise ValueError(f"unknown dependency entrypoint {entrypoint!r}") from exc

    runtime: set[int] = set()
    pending = [int(entry.reference.address)]
    while pending:
        reference = pending.pop()
        if reference in runtime:
            continue
        runtime.add(reference)
        function = function_table.entry(reference)
        function_graph = function.graph
        if function_graph is None:
            continue
        for _node_id, data in function_graph.G.nodes(data=True):
            attributes = data.get("attributes") or {}
            callee = attributes.get("callee_ref")
            if callee is None:
                callee = attributes.get("method_ref")
            if callee is not None:
                pending.append(int(callee))

    mapped: set[int] = set()
    bindings: list[tuple[str, int]] = []
    map_ir = graph.G.graph.get("map_ir") or {}
    selected_classes = (
        set(map(str, map_ir["selected_class_identities"]))
        if "selected_class_identities" in map_ir
        else None
    )
    for mapped_graph in map_ir.get("graphs", ()):
        identity = mapped_graph.get("identity")
        if not identity:
            continue
        if (
            selected_classes is not None
            and str(identity).rsplit(".", 1)[0] not in selected_classes
        ):
            continue
        try:
            mapped_entry = function_table.entry(str(identity))
        except KeyError:
            continue
        if mapped_entry.graph is not None:
            reference = int(mapped_entry.reference.address)
            mapped.add(reference)
            bindings.append((str(identity), reference))

    retained = runtime | mapped
    return MapDependencyRegions(
        runtime=tuple(sorted(runtime)),
        mapped=tuple(sorted(mapped)),
        retained=tuple(sorted(retained)),
        map_only=tuple(sorted(mapped - runtime)),
        bindings=tuple(bindings),
    )


def _ordered_nodes(graph: Any) -> list[Any]:
    try:
        import networkx as nx

        return list(nx.lexicographical_topological_sort(graph.G, key=str))
    except nx.NetworkXUnfeasible:
        # Retained loop-carried feedback is recorded as an irreducible SCC by
        # loop_composer.  Reference-table numbering needs a stable view, not
        # permission to erase that recursion.
        return sorted(
            graph.G,
            key=lambda node_id: (
                int(graph.levels.get(node_id, 0)),
                str(node_id),
            ),
        )
    except (ImportError, TypeError, ValueError):
        return list(graph.G)


def _constant_value(data: dict[str, Any]) -> tuple[bool, Any]:
    expression = data.get("expr_obj")
    if isinstance(expression, ast.Constant):
        return True, expression.value
    if "constant" in data:
        return True, data["constant"]
    attributes = data.get("attributes") or {}
    if str(data.get("type")) in {"Const", "const", "Constant"}:
        return True, attributes.get("value")
    return False, None


def _function_usage(
    graph: Any,
    node_id: Any,
    data: dict[str, Any],
) -> tuple[str, int | str] | None:
    attributes = data.get("attributes") or {}
    if attributes.get("callee_ref") is not None:
        return "graph", int(attributes["callee_ref"])
    if attributes.get("external_callee_ref") is not None:
        return "external", int(attributes["external_callee_ref"])
    if attributes.get("static_python_reference"):
        return "static", str(attributes["static_python_reference"])

    # Attribute-call syntax has already been reduced to an operation name at
    # this point.  Correlate that operation with an existing function-table
    # entry without restoring Python attribute or bound-method semantics.
    expression = data.get("expr_obj")
    if isinstance(expression, ast.Call) and isinstance(
        expression.func,
        ast.Attribute,
    ):
        table = getattr(graph, "function_table", None)
        if table is not None:
            reference = table.reference(str(data.get("type")))
            if reference is not None:
                return "graph", int(reference.address)
    return None


def build_shell_reference_tables(graph: Any) -> ShellReferenceTables:
    """Package one graph's visible references into monotonic local lists."""

    functions: list[ShellFunctionReference] = []
    constants: list[ShellConstantReference] = []
    memory: list[ShellMemoryReference] = []
    correlations: list[ShellReferenceCorrelation] = []
    recursion: list[ShellRecursionReference] = []
    function_indices: dict[tuple[str, int | str], int] = {}

    for region_id, record in sorted(
        (graph.G.graph.get("recursion_table") or {}).items()
    ):
        index = len(recursion)
        recursion.append(ShellRecursionReference(
            index=index,
            region_id=int(region_id),
            lower_as=str(record.get("lower_as", "while")),
            members=tuple(map(int, record.get("members", ()))),
            control_ir=bool(record.get("control_ir", True)),
            control_members=tuple(map(
                int, record.get("control_members", ())
            )),
            incoming=tuple(record.get("incoming", ())),
            outgoing=tuple(record.get("outgoing", ())),
            feedback=tuple(record.get("feedback", ())),
        ))
        correlations.append(ShellReferenceCorrelation(
            table="recursion",
            index=index,
            graph_node_id=None,
            source_kind="strongly_connected_component",
            source_reference=int(region_id),
            source_name=f"recursion_region_{int(region_id)}",
        ))

    function_table = getattr(graph, "function_table", None)
    if function_table is not None:
        for entry in sorted(
            function_table,
            key=lambda item: item.reference.address,
        ):
            index = len(functions)
            key = ("graph", int(entry.reference.address))
            function_indices[key] = index
            functions.append(
                ShellFunctionReference(
                    index=index,
                    namespace="graph",
                    source_address=int(entry.reference.address),
                    name=str(entry.name),
                    qualified_name=str(entry.qualified_name),
                    graph_backed=entry.graph is not None,
                    external=False,
                )
            )
            correlations.append(
                ShellReferenceCorrelation(
                    table="functions",
                    index=index,
                    graph_node_id=None,
                    source_kind="function_table",
                    source_reference=int(entry.reference.address),
                    source_name=str(entry.qualified_name),
                )
            )

    external_table = getattr(graph, "external_function_table", None)
    if external_table is not None:
        for entry in sorted(
            external_table,
            key=lambda item: item.reference.address,
        ):
            index = len(functions)
            key = ("external", int(entry.reference.address))
            function_indices[key] = index
            functions.append(
                ShellFunctionReference(
                    index=index,
                    namespace="external",
                    source_address=int(entry.reference.address),
                    name=str(entry.name),
                    qualified_name=str(entry.qualified_name),
                    graph_backed=False,
                    external=True,
                )
            )
            correlations.append(
                ShellReferenceCorrelation(
                    table="functions",
                    index=index,
                    graph_node_id=None,
                    source_kind="external_function_table",
                    source_reference=int(entry.reference.address),
                    source_name=str(entry.qualified_name),
                )
            )

    ordered_nodes = _ordered_nodes(graph)
    output_names = tuple(graph.G.graph.get("function_outputs", ()))
    identities = dict(graph.G.graph.get("identity_table", {}) or {})
    output_node_names = {
        node_id: name
        for name in output_names
        for node_id in identities.get(name, ())[-1:]
    }
    output_nodes = set(output_node_names)

    for node_id in ordered_nodes:
        data = graph.G.nodes[node_id]
        usage = _function_usage(graph, node_id, data)
        if usage is not None:
            index = function_indices.get(usage)
            if index is None and usage[0] == "static":
                index = len(functions)
                function_indices[usage] = index
                functions.append(
                    ShellFunctionReference(
                        index=index,
                        namespace="static",
                        source_address=None,
                        name=str(usage[1]).rsplit(".", 1)[-1],
                        qualified_name=str(usage[1]),
                        graph_backed=False,
                        external=True,
                    )
                )
            if index is not None:
                correlations.append(
                    ShellReferenceCorrelation(
                        table="functions",
                        index=index,
                        graph_node_id=node_id,
                        source_kind="call",
                        source_reference=usage[1],
                        source_name=functions[index].qualified_name,
                    )
                )

        is_constant, value = _constant_value(data)
        if is_constant:
            index = len(constants)
            constants.append(
                ShellConstantReference(
                    index=index,
                    value=value,
                    value_type=type(value).__name__,
                )
            )
            correlations.append(
                ShellReferenceCorrelation(
                    table="constants",
                    index=index,
                    graph_node_id=node_id,
                    source_kind="literal",
                    source_reference=node_id,
                    source_name=str(data.get("label", value)),
                )
            )

        expression = data.get("expr_obj")
        node_type = str(data.get("type"))
        is_input = node_type in {"Input", "input"}
        is_attribute = isinstance(expression, ast.Attribute)
        is_output = node_id in output_nodes
        if not (is_input or is_attribute or is_output):
            continue
        roles = []
        if is_input:
            roles.append("input")
        if is_attribute:
            roles.append("attribute")
        if is_output:
            roles.append("output")
        attributes = data.get("attributes") or {}
        field_name = expression.attr if is_attribute else None
        base_node_id = next(
            (
                parent
                for parent, role in data.get("parents", ())
                if str(role) in {"value", "base", "operand"}
            ),
            None,
        )
        name = str(
            output_node_names.get(
                node_id,
                attributes.get(
                    "binding_name",
                    field_name or data.get("label", node_id),
                ),
            )
        )
        index = len(memory)
        memory.append(
            ShellMemoryReference(
                index=index,
                graph_node_id=node_id,
                name=name,
                roles=tuple(roles),
                base_node_id=base_node_id,
                field_name=field_name,
            )
        )
        correlations.append(
            ShellReferenceCorrelation(
                table="memory",
                index=index,
                graph_node_id=node_id,
                source_kind="+".join(roles),
                source_reference=node_id,
                source_name=name,
            )
        )

    return ShellReferenceTables(
        functions=functions,
        constants=constants,
        memory=memory,
        correlations=correlations,
        recursion=recursion,
    )


__all__ = [
    "ClassNavigationMember",
    "ClassNavigationRecord",
    "ClassNavigationTable",
    "MapDependencyRegions",
    "ShellConstantReference",
    "ShellFunctionReference",
    "ShellMemoryReference",
    "ShellReferenceCorrelation",
    "ShellRecursionReference",
    "ShellReferenceTables",
    "build_map_dependency_regions",
    "build_class_navigation_table",
    "build_shell_reference_tables",
]
