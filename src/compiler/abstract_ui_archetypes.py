"""Reusable AbstractUI archetypes as transactional living-document edits.

An intention describes meaning.  An archetype describes a construction a user
may instantiate into the program.  Instantiation creates identified nodes and
edges, publishes a symbol into a hierarchical namespace, derives an
``IntelliType`` from the created structure, and records the actor/location that
caused the edit.

The deliberately plain fluent surface mirrors the motivating sketch::

    panel = (
        box.with_(class_context, inside)
        .with_(buttons.with_(label, front), front)
        .with_(displays, top_front)
        .connect(displays, class_context.members)
        .connect(buttons, class_context.methods)
    )

Python cannot bind a value named ``class`` or spell ``top-front`` as an
identifier, so the source-safe names are ``class_context`` and ``top_front``;
their canonical spellings remain ``class`` and ``top-front``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import inspect
from typing import Any, Iterable, Mapping, Sequence


ABSTRACT_UI_ARCHETYPE_VERSION = "abstract-ui-archetype-v0"


def _stable_id(*parts: str) -> str:
    readable = "/".join(
        str(part).strip().replace(" ", "-") for part in parts if str(part).strip()
    )
    digest = hashlib.sha256("\0".join(map(str, parts)).encode("utf-8")).hexdigest()[:12]
    return f"{readable}@{digest}"


def _frozen_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, Mapping):
        return tuple((str(key), _frozen_value(item)) for key, item in value.items())
    if isinstance(value, (tuple, list, set, frozenset)):
        return tuple(_frozen_value(item) for item in value)
    return str(value)


@dataclass(frozen=True, slots=True)
class Placement:
    name: str


@dataclass(frozen=True, slots=True)
class ContextReference:
    path: tuple[str, ...]

    def __getattr__(self, name: str) -> "ContextReference":
        if name.startswith("_"):
            raise AttributeError(name)
        return ContextReference((*self.path, name.rstrip("_")))

    @property
    def spelling(self) -> str:
        return ".".join(self.path)


@dataclass(frozen=True, slots=True)
class ArchetypePrototype:
    name: str
    capabilities: tuple[str, ...] = ()

    def with_(
        self,
        child: "ArchetypePrototype | ArchetypeRecipe | ContextReference",
        placement: Placement,
    ) -> "ArchetypeRecipe":
        return ArchetypeRecipe(self).with_(child, placement)

    def connect(
        self,
        source: "ArchetypePrototype | ContextReference | str",
        target: "ArchetypePrototype | ContextReference | str",
        relationship: str = "connects",
    ) -> "ArchetypeRecipe":
        return ArchetypeRecipe(self).connect(source, target, relationship)


@dataclass(frozen=True, slots=True)
class ArchetypeInclusion:
    child: ArchetypePrototype | "ArchetypeRecipe" | ContextReference
    placement: Placement
    ordinal: int


@dataclass(frozen=True, slots=True)
class ArchetypeConnection:
    source: ArchetypePrototype | ContextReference | str
    target: ArchetypePrototype | ContextReference | str
    relationship: str
    ordinal: int


@dataclass(frozen=True, slots=True)
class ArchetypeStatement:
    ordinal: int
    operation: str
    source: str
    target: str
    relationship: str


def _selector_spelling(value: Any) -> str:
    if isinstance(value, ArchetypePrototype):
        return value.name
    if isinstance(value, ContextReference):
        return value.spelling
    if isinstance(value, ArchetypeRecipe):
        return value.root.name
    return str(value)


@dataclass(frozen=True, slots=True)
class ArchetypeRecipe:
    root: ArchetypePrototype
    inclusions: tuple[ArchetypeInclusion, ...] = ()
    connections: tuple[ArchetypeConnection, ...] = ()
    statements: tuple[ArchetypeStatement, ...] = ()

    def with_(
        self,
        child: ArchetypePrototype | "ArchetypeRecipe" | ContextReference,
        placement: Placement,
    ) -> "ArchetypeRecipe":
        if not isinstance(placement, Placement):
            raise TypeError("archetype placement must be a Placement token")
        ordinal = len(self.statements)
        return replace(
            self,
            inclusions=(*self.inclusions, ArchetypeInclusion(
                child, placement, len(self.inclusions),
            )),
            statements=(*self.statements, ArchetypeStatement(
                ordinal, "with", self.root.name,
                _selector_spelling(child), placement.name,
            )),
        )

    def connect(
        self,
        source: ArchetypePrototype | ContextReference | str,
        target: ArchetypePrototype | ContextReference | str,
        relationship: str = "connects",
    ) -> "ArchetypeRecipe":
        ordinal = len(self.statements)
        relationship = str(relationship)
        return replace(
            self,
            connections=(*self.connections, ArchetypeConnection(
                source, target, relationship, len(self.connections),
            )),
            statements=(*self.statements, ArchetypeStatement(
                ordinal, "connect", _selector_spelling(source),
                _selector_spelling(target), relationship,
            )),
        )


@dataclass(frozen=True, slots=True)
class LivingNode:
    identity: str
    kind: str
    name: str
    archetype: str | None = None
    properties: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class LivingEdge:
    source: str
    target: str
    relationship: str
    ordinal: int
    properties: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class IntelliSlot:
    name: str
    identity: str
    kind: str
    placement: str
    capabilities: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class IntelliBinding:
    source: str
    target: str
    relationship: str


@dataclass(frozen=True, slots=True)
class IntelliType:
    """Structural type inferred from one archetype instance graph."""

    identity: str
    name: str
    root_archetype: str
    capabilities: tuple[str, ...]
    slots: tuple[IntelliSlot, ...]
    bindings: tuple[IntelliBinding, ...]
    recipe_name: str


@dataclass(frozen=True, slots=True)
class NamespaceSymbol:
    path: tuple[str, ...]
    identity: str
    intellitype: IntelliType
    revision: int

    @property
    def qualified_name(self) -> str:
        return ".".join(self.path)


@dataclass(frozen=True, slots=True)
class LivingNamespace:
    symbols: tuple[NamespaceSymbol, ...] = ()

    def resolve(self, path: str | Sequence[str]) -> NamespaceSymbol:
        pieces = tuple(str(path).split(".")) if isinstance(path, str) else tuple(map(str, path))
        for symbol in self.symbols:
            if symbol.path == pieces:
                return symbol
        raise KeyError(".".join(pieces))

    def publish(self, symbol: NamespaceSymbol) -> "LivingNamespace":
        if any(existing.path == symbol.path for existing in self.symbols):
            raise ValueError(f"namespace symbol already exists: {symbol.qualified_name}")
        return LivingNamespace((*self.symbols, symbol))


@dataclass(frozen=True, slots=True)
class LivingDocument:
    identity: str
    revision: int = 0
    nodes: tuple[LivingNode, ...] = ()
    edges: tuple[LivingEdge, ...] = ()
    namespace: LivingNamespace = field(default_factory=LivingNamespace)

    @staticmethod
    def empty(identity: str = "abstract-ui-document") -> "LivingDocument":
        return LivingDocument(str(identity))


@dataclass(frozen=True, slots=True)
class LivingDocumentEdit:
    identity: str
    action: str
    actor: str
    location: str
    before_revision: int
    after_revision: int
    added_nodes: tuple[LivingNode, ...]
    added_edges: tuple[LivingEdge, ...]
    published_symbols: tuple[NamespaceSymbol, ...]
    statements: tuple[ArchetypeStatement, ...]

    def to_mapping(self) -> dict[str, Any]:
        """Serializable sidecar suitable for a later typed DualIR member."""

        return {
            "schema": ABSTRACT_UI_ARCHETYPE_VERSION,
            "identity": self.identity,
            "action": self.action,
            "actor": self.actor,
            "location": self.location,
            "before_revision": self.before_revision,
            "after_revision": self.after_revision,
            "added_nodes": [
                {
                    "identity": node.identity,
                    "kind": node.kind,
                    "name": node.name,
                    "archetype": node.archetype,
                    "properties": dict(node.properties),
                }
                for node in self.added_nodes
            ],
            "added_edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "relationship": edge.relationship,
                    "ordinal": edge.ordinal,
                    "properties": dict(edge.properties),
                }
                for edge in self.added_edges
            ],
            "published_symbols": [
                {
                    "path": symbol.path,
                    "identity": symbol.identity,
                    "intellitype": symbol.intellitype.identity,
                }
                for symbol in self.published_symbols
            ],
            "statements": [
                {
                    "ordinal": statement.ordinal,
                    "operation": statement.operation,
                    "source": statement.source,
                    "target": statement.target,
                    "relationship": statement.relationship,
                }
                for statement in self.statements
            ],
        }


@dataclass(frozen=True, slots=True)
class ArchetypeInstantiation:
    document: LivingDocument
    edit: LivingDocumentEdit
    symbol: NamespaceSymbol
    intellitype: IntelliType
    root_identity: str


@dataclass(frozen=True, slots=True)
class ArchetypeContext:
    namespace_path: tuple[str, ...]
    bindings: Mapping[str, Any]
    actor: str = "user"
    location: str = "here"


@dataclass(frozen=True, slots=True)
class _ResolvedReference:
    identity: str
    name: str
    kind: str
    value: Any


def _class_members(value: Any) -> tuple[str, ...]:
    if isinstance(value, type):
        annotations = tuple((vars(value).get("__annotations__") or {}).keys())
        ordinary = tuple(
            name for name, item in vars(value).items()
            if not name.startswith("_")
            and not inspect.isfunction(item)
            and not isinstance(item, (staticmethod, classmethod, property, type))
        )
        return tuple(dict.fromkeys((*annotations, *ordinary)))
    fields = getattr(value, "fields", ())
    return tuple(str(field.name) for field in fields)


def _class_methods(value: Any) -> tuple[str, ...]:
    if isinstance(value, type):
        return tuple(
            name for name, item in vars(value).items()
            if not name.startswith("_")
            and (
                inspect.isfunction(item)
                or isinstance(item, (staticmethod, classmethod))
            )
        )
    methods = getattr(value, "methods", ())
    return tuple(str(method.name) for method in methods)


def _binding_identity(value: Any) -> str:
    if isinstance(value, type):
        return f"python:{value.__module__}.{value.__qualname__}"
    identity = getattr(value, "identity", None)
    if identity is not None:
        return str(identity)
    return f"value:{type(value).__module__}.{type(value).__qualname__}"


def _resolve_context_reference(
    reference: ContextReference,
    context: ArchetypeContext,
) -> _ResolvedReference:
    root = reference.path[0]
    if root not in context.bindings:
        raise KeyError(f"archetype context has no binding {root!r}")
    value = context.bindings[root]
    root_identity = _binding_identity(value)
    if len(reference.path) == 1:
        return _ResolvedReference(root_identity, root, "context-object", value)
    tail = reference.path[1:]
    if tail == ("members",):
        resolved = _class_members(value)
        kind = "program-member-set"
    elif tail == ("methods",):
        resolved = _class_methods(value)
        kind = "program-method-set"
    else:
        resolved = value
        for piece in tail:
            if isinstance(resolved, Mapping):
                resolved = resolved[piece]
            else:
                resolved = getattr(resolved, piece)
        kind = "context-reference"
    return _ResolvedReference(
        f"{root_identity}/{'/'.join(tail)}",
        reference.spelling,
        kind,
        resolved,
    )


def _part_selector_name(value: Any) -> str:
    if isinstance(value, ArchetypePrototype):
        return value.name
    if isinstance(value, ArchetypeRecipe):
        return value.root.name
    return _selector_spelling(value)


@dataclass
class _InstantiationBuilder:
    recipe_name: str
    instance_path: tuple[str, ...]
    context: ArchetypeContext
    nodes: list[LivingNode] = field(default_factory=list)
    edges: list[LivingEdge] = field(default_factory=list)
    slots: list[IntelliSlot] = field(default_factory=list)
    bindings: list[IntelliBinding] = field(default_factory=list)
    part_ids: dict[str, list[str]] = field(default_factory=dict)
    reference_ids: dict[str, str] = field(default_factory=dict)
    capabilities: list[str] = field(default_factory=list)

    def add_reference(self, reference: ContextReference) -> str:
        spelling = reference.spelling
        cached = self.reference_ids.get(spelling)
        if cached is not None:
            return cached
        resolved = _resolve_context_reference(reference, self.context)
        properties = (
            ("reference", spelling),
            ("value", _frozen_value(resolved.value)),
        )
        self.nodes.append(LivingNode(
            resolved.identity, resolved.kind, resolved.name,
            properties=properties,
        ))
        self.reference_ids[spelling] = resolved.identity
        return resolved.identity

    def add_recipe(
        self,
        recipe: ArchetypeRecipe,
        *,
        parent_id: str | None,
        placement: Placement | None,
        path: tuple[int, ...],
    ) -> str:
        part_id = _stable_id(*self.instance_path, recipe.root.name, *map(str, path))
        self.nodes.append(LivingNode(
            part_id, "archetype-part", recipe.root.name, recipe.root.name,
            (("capabilities", recipe.root.capabilities),),
        ))
        self.part_ids.setdefault(recipe.root.name, []).append(part_id)
        self.capabilities.extend(recipe.root.capabilities)
        if parent_id is not None and placement is not None:
            self.edges.append(LivingEdge(
                part_id, parent_id, "contained-by", len(self.edges),
                (("placement", placement.name),),
            ))
            self.slots.append(IntelliSlot(
                recipe.root.name, part_id, "archetype-part", placement.name,
                recipe.root.capabilities,
            ))

        for inclusion in recipe.inclusions:
            child = inclusion.child
            if isinstance(child, ContextReference):
                child_id = self.add_reference(child)
                self.edges.append(LivingEdge(
                    child_id, part_id, "contained-by", inclusion.ordinal,
                    (("placement", inclusion.placement.name),),
                ))
                self.slots.append(IntelliSlot(
                    child.spelling, child_id, "context-reference",
                    inclusion.placement.name,
                ))
            else:
                child_recipe = (
                    child if isinstance(child, ArchetypeRecipe)
                    else ArchetypeRecipe(child)
                )
                self.add_recipe(
                    child_recipe,
                    parent_id=part_id,
                    placement=inclusion.placement,
                    path=(*path, inclusion.ordinal),
                )

        for connection in recipe.connections:
            source_id = self.resolve_selector(connection.source)
            target_id = self.resolve_selector(connection.target)
            self.edges.append(LivingEdge(
                source_id, target_id, connection.relationship,
                connection.ordinal,
            ))
            self.bindings.append(IntelliBinding(
                source_id, target_id, connection.relationship,
            ))
        return part_id

    def resolve_selector(self, selector: Any) -> str:
        if isinstance(selector, ContextReference):
            return self.add_reference(selector)
        name = _part_selector_name(selector)
        identities = self.part_ids.get(name, ())
        if not identities:
            raise KeyError(f"archetype has no instantiated part {name!r}")
        if len(identities) != 1:
            raise ValueError(
                f"archetype selector {name!r} is ambiguous across {len(identities)} parts"
            )
        return identities[0]


@dataclass(frozen=True, slots=True)
class ArchetypeLibrary:
    definitions: tuple[tuple[str, ArchetypeRecipe], ...] = ()

    def define(self, name: str, recipe: ArchetypeRecipe) -> "ArchetypeLibrary":
        name = str(name)
        if not name or "." in name:
            raise ValueError("archetype name must be one non-empty namespace segment")
        if any(existing == name for existing, _ in self.definitions):
            raise ValueError(f"archetype already defined: {name}")
        return ArchetypeLibrary((*self.definitions, (name, recipe)))

    def recipe(self, name: str) -> ArchetypeRecipe:
        for candidate, recipe in self.definitions:
            if candidate == name:
                return recipe
        raise KeyError(name)

    def instantiate(
        self,
        name: str,
        *,
        document: LivingDocument,
        context: ArchetypeContext,
        symbol_name: str | None = None,
    ) -> ArchetypeInstantiation:
        recipe = self.recipe(name)
        symbol_name = str(symbol_name or name)
        symbol_path = (*context.namespace_path, symbol_name)
        if any(symbol.path == symbol_path for symbol in document.namespace.symbols):
            raise ValueError(
                f"namespace symbol already exists: {'.'.join(symbol_path)}"
            )
        instance_identity = _stable_id(document.identity, *symbol_path)
        builder = _InstantiationBuilder(name, symbol_path, context)
        root_identity = builder.add_recipe(
            recipe, parent_id=None, placement=None, path=(),
        )
        capabilities = tuple(dict.fromkeys(builder.capabilities))
        intellitype = IntelliType(
            f"intellitype:{instance_identity}",
            symbol_name,
            recipe.root.name,
            capabilities,
            tuple(builder.slots),
            tuple(builder.bindings),
            name,
        )
        symbol = NamespaceSymbol(
            symbol_path, root_identity, intellitype, document.revision + 1,
        )

        existing_nodes = {node.identity: node for node in document.nodes}
        conflicting_nodes = [
            node.identity for node in builder.nodes
            if node.identity in existing_nodes and existing_nodes[node.identity] != node
        ]
        if conflicting_nodes:
            raise ValueError(
                f"archetype instantiation collides with existing nodes: "
                f"{sorted(conflicting_nodes)!r}"
            )
        # Context anchors belong to the living document rather than to any one
        # archetype instance.  A second instance may therefore refer to the same
        # class/member/method nodes, provided their definitions are identical.
        added_nodes = tuple(
            node for node in builder.nodes if node.identity not in existing_nodes
        )
        existing_edges = set(document.edges)
        added_edges = tuple(edge for edge in builder.edges if edge not in existing_edges)
        namespace = document.namespace.publish(symbol)
        new_document = LivingDocument(
            document.identity,
            document.revision + 1,
            (*document.nodes, *added_nodes),
            (*document.edges, *added_edges),
            namespace,
        )
        edit_identity = _stable_id(
            document.identity, str(new_document.revision), "instantiate", name,
        )
        edit = LivingDocumentEdit(
            edit_identity,
            "instantiate-archetype",
            context.actor,
            context.location,
            document.revision,
            new_document.revision,
            added_nodes,
            added_edges,
            (symbol,),
            recipe.statements,
        )
        return ArchetypeInstantiation(
            new_document, edit, symbol, intellitype, root_identity,
        )


# Canonical starter prototypes -----------------------------------------------------
box = ArchetypePrototype("box", ("container", "interior"))
buttons = ArchetypePrototype("buttons", ("action-host", "method-binding"))
displays = ArchetypePrototype("displays", ("display-host", "member-binding"))
label = ArchetypePrototype("label", ("description", "accessible-name"))
panel_surface = ArchetypePrototype("panel-surface", ("container", "presentation"))
door = ArchetypePrototype("door", ("portal", "action-host"))
room = ArchetypePrototype("room", ("container", "interior", "enterable"))
track = ArchetypePrototype("track", ("navigation", "traversable"))

inside = Placement("inside")
front = Placement("front")
top_front = Placement("top-front")
back = Placement("back")
left = Placement("left")
right = Placement("right")
top = Placement("top")
bottom = Placement("bottom")

class_context = ContextReference(("class",))
program_class = class_context


def class_panel_recipe() -> ArchetypeRecipe:
    """The motivating panel recipe, expressed through source-safe Python."""

    return (
        box.with_(class_context, inside)
        .with_(buttons.with_(label, front), front)
        .with_(displays, top_front)
        .connect(displays, class_context.members, "displays")
        .connect(buttons, class_context.methods, "invokes")
    )


__all__ = [
    "ABSTRACT_UI_ARCHETYPE_VERSION",
    "ArchetypeConnection",
    "ArchetypeContext",
    "ArchetypeInclusion",
    "ArchetypeInstantiation",
    "ArchetypeLibrary",
    "ArchetypePrototype",
    "ArchetypeRecipe",
    "ArchetypeStatement",
    "ContextReference",
    "IntelliBinding",
    "IntelliSlot",
    "IntelliType",
    "LivingDocument",
    "LivingDocumentEdit",
    "LivingEdge",
    "LivingNamespace",
    "LivingNode",
    "NamespaceSymbol",
    "Placement",
    "back",
    "bottom",
    "box",
    "buttons",
    "class_context",
    "class_panel_recipe",
    "displays",
    "door",
    "front",
    "inside",
    "label",
    "left",
    "panel_surface",
    "program_class",
    "right",
    "room",
    "top",
    "top_front",
    "track",
]
