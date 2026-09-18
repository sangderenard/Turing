"""HTML containment ingestion for the language-neutral ``ProcessGraph``.

HTML is not treated as a publishing wrapper here.  It is a source language
whose document tree is normalized into the same dependency-graph shape used
by the other compiler frontends.  The orientation is intentional::

    content ----> containing element ----> document/system root

In other words, an interface container *depends on* its ordered contents.
That makes the synthetic :class:`InterfaceRoot` the graph root in the same
sense that a SymPy expression's final operation is its graph root, while the
inverse reading of each edge remains the familiar "container contains child."

The neutral node vocabulary is deliberately small:

``InterfaceRoot``
    The host/system container.  An HTML document is one source-language
    spelling of this concept; a Java scene or native window can map to it too.
``InterfaceContainer``
    Any nested interface entity.  Exact HTML tags and their capabilities are
    vocabulary data, not new graph node classes.
``InterfaceContent``
    Text, comments, and declarations in authored order.

CSS selectors/rules and program bindings are separate dependency relations.
They are not guessed by this first containment pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from html import escape
from html.parser import HTMLParser
from typing import Any, Mapping, Sequence


HTML_GRAPH_SCHEMA_VERSION = "interface-containment-v1"


class InterfaceNodeName(str, Enum):
    """Canonical node names shared by UI-capable source languages."""

    ROOT = "InterfaceRoot"
    CONTAINER = "InterfaceContainer"
    CONTENT = "InterfaceContent"


class InterfaceCapability(str, Enum):
    """Coarse behavior supplied by a source-language vocabulary token."""

    DOCUMENT = "document"
    STRUCTURE = "structure"
    METADATA = "metadata"
    FORM = "form"
    VALUE = "value"
    ACTION = "action"
    RESOURCE = "resource"
    GENERIC = "generic"


@dataclass(frozen=True)
class HTMLTagSpec:
    """Meaning of one HTML spelling in the neutral container vocabulary."""

    capability: InterfaceCapability
    void: bool = False


# This is a supported compiler profile, not a claim to duplicate the entire
# WHATWG element catalog.  Unknown elements remain losslessly representable as
# generic containers and are reported as vocabulary shortfalls.
HTML_TAG_VOCABULARY: Mapping[str, HTMLTagSpec] = {
    "html": HTMLTagSpec(InterfaceCapability.DOCUMENT),
    "head": HTMLTagSpec(InterfaceCapability.METADATA),
    "body": HTMLTagSpec(InterfaceCapability.STRUCTURE),
    "title": HTMLTagSpec(InterfaceCapability.METADATA),
    "meta": HTMLTagSpec(InterfaceCapability.METADATA, void=True),
    "link": HTMLTagSpec(InterfaceCapability.RESOURCE, void=True),
    "style": HTMLTagSpec(InterfaceCapability.RESOURCE),
    "script": HTMLTagSpec(InterfaceCapability.RESOURCE),
    "div": HTMLTagSpec(InterfaceCapability.STRUCTURE),
    "form": HTMLTagSpec(InterfaceCapability.FORM),
    "label": HTMLTagSpec(InterfaceCapability.FORM),
    "input": HTMLTagSpec(InterfaceCapability.VALUE, void=True),
    "select": HTMLTagSpec(InterfaceCapability.VALUE),
    "option": HTMLTagSpec(InterfaceCapability.VALUE),
    "button": HTMLTagSpec(InterfaceCapability.ACTION),
    "textarea": HTMLTagSpec(InterfaceCapability.VALUE),
    "output": HTMLTagSpec(InterfaceCapability.VALUE),
}


# Mirrors role_schemas' node-name -> directional-child-role description.  The
# direct ingester below writes canonical ProcessGraph nodes because source
# attributes and deterministic positions must survive, but publishing the
# schema independently lets other frontends construct the same neutral tree.
HTML_NODE_SCHEMA: Mapping[str, Mapping[str, Mapping[str, str]]] = {
    InterfaceNodeName.ROOT.value: {"up": {"content": "many"}, "down": {}},
    InterfaceNodeName.CONTAINER.value: {
        "up": {"content": "many"}, "down": {},
    },
    InterfaceNodeName.CONTENT.value: {"up": {}, "down": {}},
}


@dataclass(frozen=True, order=True)
class HTMLVocabularyShortfall:
    """A source tag preserved as a generic container outside the profile."""

    path: tuple[int, ...]
    tag: str

    def format(self) -> str:
        position = ".".join(map(str, self.path)) or "root"
        return f"unsupported-html-tag: <{self.tag}> at {position}"


class HTMLGraphError(ValueError):
    """The source cannot form an unambiguous ordered containment graph."""


@dataclass
class HTMLPageLowering:
    graph: Any
    root: int
    shortfalls: tuple[HTMLVocabularyShortfall, ...]

    @property
    def complete(self) -> bool:
        return not self.shortfalls


@dataclass
class _SourceContent:
    kind: str
    value: str


@dataclass
class _SourceContainer:
    tag: str | None
    attributes: tuple[tuple[str, str | None], ...] = ()
    content: list["_SourceContainer | _SourceContent"] = field(
        default_factory=list
    )
    self_closing: bool = False


class _ContainmentParser(HTMLParser):
    """Strictly nest HTML into the compiler's small containment profile."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = _SourceContainer(None)
        self._stack = [self.root]

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]],
    ) -> None:
        tag = tag.casefold()
        spec = HTML_TAG_VOCABULARY.get(tag)
        child = _SourceContainer(
            tag,
            tuple((name.casefold(), value) for name, value in attrs),
            self_closing=bool(spec and spec.void),
        )
        self._stack[-1].content.append(child)
        if not (spec and spec.void):
            self._stack.append(child)

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]],
    ) -> None:
        child = _SourceContainer(
            tag.casefold(),
            tuple((name.casefold(), value) for name, value in attrs),
            self_closing=True,
        )
        self._stack[-1].content.append(child)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if len(self._stack) == 1 or self._stack[-1].tag != tag:
            open_tag = self._stack[-1].tag if len(self._stack) > 1 else None
            raise HTMLGraphError(
                f"closing </{tag}> does not match open "
                f"{('<' + open_tag + '>') if open_tag else 'document root'}"
            )
        self._stack.pop()

    def handle_data(self, data: str) -> None:
        if data:
            self._stack[-1].content.append(_SourceContent("text", data))

    def handle_comment(self, data: str) -> None:
        self._stack[-1].content.append(_SourceContent("comment", data))

    def handle_decl(self, decl: str) -> None:
        self._stack[-1].content.append(_SourceContent("declaration", decl))

    def finish(self) -> _SourceContainer:
        self.close()
        if len(self._stack) != 1:
            unclosed = ", ".join(
                f"<{node.tag}>" for node in self._stack[1:]
            )
            raise HTMLGraphError(f"unclosed HTML containers: {unclosed}")
        return self.root


def _next_integer_node_id(graph: Any) -> int:
    return max(
        (node for node in graph.G if isinstance(node, int)),
        default=-1,
    ) + 1


def _container_label(tag: str, attributes: Sequence[tuple[str, str | None]]) -> str:
    by_name = dict(attributes)
    identity = f"#{by_name['id']}" if by_name.get("id") else ""
    classes = "".join(
        f".{token}" for token in (by_name.get("class") or "").split()
    )
    return f"<{tag}{identity}{classes}>"


def ingest_html_document(
    graph: Any,
    source: str,
    *,
    strict_vocabulary: bool = False,
) -> HTMLPageLowering:
    """Parse ``source`` into ordered neutral interface dependencies.

    Every source occurrence receives a monotonic integer identity in preorder
    and an explicit ``position`` tuple.  Containment edges point child ->
    container and carry their sibling ``ordinal``.  Unknown tags are retained
    as generic containers; ``strict_vocabulary=True`` rejects them after the
    complete shortfall set has been collected.
    """

    parser = _ContainmentParser()
    parser.feed(source)
    source_root = parser.finish()
    next_id = [_next_integer_node_id(graph)]
    shortfalls: list[HTMLVocabularyShortfall] = []

    def allocate() -> int:
        node_id = next_id[0]
        next_id[0] += 1
        return node_id

    def add_node(
        source_node: _SourceContainer | _SourceContent,
        path: tuple[int, ...],
    ) -> int:
        node_id = allocate()
        if isinstance(source_node, _SourceContent):
            node_type = InterfaceNodeName.CONTENT.value
            label = source_node.value
            semantic_attributes: dict[str, Any] = {
                "source_language": "html",
                "position": path,
                "content_kind": source_node.kind,
                "value": source_node.value,
            }
            content = ()
            constant: Any = source_node.value
        elif source_node.tag is None:
            node_type = InterfaceNodeName.ROOT.value
            label = "interface root"
            semantic_attributes = {
                "source_language": "html",
                "position": path,
                "capability": InterfaceCapability.DOCUMENT.value,
                "schema_version": HTML_GRAPH_SCHEMA_VERSION,
            }
            content = tuple(source_node.content)
            constant = None
        else:
            node_type = InterfaceNodeName.CONTAINER.value
            spec = HTML_TAG_VOCABULARY.get(source_node.tag)
            if spec is None:
                shortfalls.append(HTMLVocabularyShortfall(path, source_node.tag))
                spec = HTMLTagSpec(InterfaceCapability.GENERIC)
            label = _container_label(source_node.tag, source_node.attributes)
            semantic_attributes = {
                "source_language": "html",
                "position": path,
                "vocabulary": source_node.tag,
                "capability": spec.capability.value,
                "source_attributes": source_node.attributes,
                "void": spec.void,
                "self_closing": source_node.self_closing,
                "vocabulary_registered": source_node.tag in HTML_TAG_VOCABULARY,
            }
            content = tuple(source_node.content)
            constant = None

        graph.G.add_node(
            node_id,
            type=node_type,
            op=None,
            label=label,
            expr_obj=source_node,
            attributes=semantic_attributes,
            constant=constant,
            tensor={},
            bit_quanta={},
            parents=[],
            children=[],
        )
        graph.node_map[node_id] = source_node

        for ordinal, child in enumerate(content):
            child_id = add_node(child, (*path, ordinal))
            graph.G.add_edge(
                child_id,
                node_id,
                role="content",
                ordinal=ordinal,
                relationship="contained-by",
            )
            graph.G.nodes[node_id]["parents"].append((child_id, "content"))
            graph.G.nodes[child_id]["children"].append((node_id, "content"))
        return node_id

    root = add_node(source_root, ())
    if strict_vocabulary and shortfalls:
        detail = "; ".join(item.format() for item in shortfalls)
        raise HTMLGraphError(detail)

    graph.roots = [root]
    graph.domain_shape = (1,)
    graph.G.graph["interface_schema_version"] = HTML_GRAPH_SCHEMA_VERSION
    graph.G.graph["source_language"] = "html"
    graph.G.graph["html_vocabulary_shortfalls"] = tuple(
        item.format() for item in shortfalls
    )
    return HTMLPageLowering(graph, root, tuple(shortfalls))


def _ordered_content_ids(graph: Any, container_id: int) -> list[int]:
    incoming = list(graph.G.predecessors(container_id))
    return sorted(
        incoming,
        key=lambda child_id: (
            graph.G.edges[child_id, container_id].get("ordinal", 0),
            child_id,
        ),
    )


def html_source_from_graph(graph: Any, root: int | None = None) -> str:
    """Render a neutral interface containment graph as normalized HTML."""

    if root is None:
        if len(graph.roots) != 1:
            raise ValueError("HTML emission requires exactly one interface root")
        root = graph.roots[0]

    def render(node_id: int) -> str:
        data = graph.G.nodes[node_id]
        node_type = data.get("type")
        attributes = data.get("attributes") or {}
        if node_type == InterfaceNodeName.ROOT.value:
            return "".join(
                render(child_id)
                for child_id in _ordered_content_ids(graph, node_id)
            )
        if node_type == InterfaceNodeName.CONTENT.value:
            value = str(attributes.get("value", ""))
            kind = attributes.get("content_kind", "text")
            if kind == "comment":
                return f"<!--{value}-->"
            if kind == "declaration":
                return f"<!{value}>"
            return escape(value, quote=False)
        if node_type != InterfaceNodeName.CONTAINER.value:
            raise ValueError(
                f"node {node_id!r} of type {node_type!r} is not HTML-emittable"
            )

        tag = str(attributes["vocabulary"])
        rendered_attributes = []
        for name, value in attributes.get("source_attributes", ()):
            if value is None:
                rendered_attributes.append(str(name))
            else:
                rendered_attributes.append(
                    f'{name}="{escape(str(value), quote=True)}"'
                )
        suffix = f" {' '.join(rendered_attributes)}" if rendered_attributes else ""
        opening = f"<{tag}{suffix}>"
        if attributes.get("void"):
            return opening
        body = "".join(
            render(child_id)
            for child_id in _ordered_content_ids(graph, node_id)
        )
        return f"{opening}{body}</{tag}>"

    return render(root)


__all__ = [
    "HTML_GRAPH_SCHEMA_VERSION",
    "HTML_NODE_SCHEMA",
    "HTML_TAG_VOCABULARY",
    "HTMLGraphError",
    "HTMLPageLowering",
    "HTMLTagSpec",
    "HTMLVocabularyShortfall",
    "InterfaceCapability",
    "InterfaceNodeName",
    "html_source_from_graph",
    "ingest_html_document",
]
