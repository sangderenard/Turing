"""HTML is a first-class containment language for ProcessGraph."""

import pytest

from src.compiler.html_process_graph import (
    HTMLGraphError,
    InterfaceNodeName,
    html_source_from_graph,
    ingest_html_document,
)
from src.transmogrifier.graph.graph_express2 import ProcessGraph


def _ingest(source: str, **kwargs):
    graph = ProcessGraph(materialize_memory=False, source_language="html")
    return ingest_html_document(graph, source, **kwargs)


def _node_by_position(graph, position):
    return next(
        (node_id, data)
        for node_id, data in graph.G.nodes(data=True)
        if tuple((data.get("attributes") or {}).get("position", ())) == position
    )


def test_document_root_depends_on_ordered_nested_containers():
    lowering = _ingest(
        '<div id="app"><form><label>Name<input name="name"></label>'
        '<button type="submit">Save</button></form></div>'
    )
    graph = lowering.graph

    assert graph.roots == [lowering.root]
    assert graph.G.nodes[lowering.root]["type"] == InterfaceNodeName.ROOT.value

    div_id, div = _node_by_position(graph, (0,))
    form_id, form = _node_by_position(graph, (0, 0))
    assert div["attributes"]["vocabulary"] == "div"
    assert form["attributes"]["vocabulary"] == "form"
    assert graph.G.has_edge(div_id, lowering.root)
    assert graph.G.has_edge(form_id, div_id)
    assert graph.G.edges[form_id, div_id]["relationship"] == "contained-by"


def test_every_html_element_uses_one_neutral_container_node_name():
    lowering = _ingest(
        "<div><form><input><select><option>x</option></select>"
        "<button>go</button></form></div>"
    )
    element_types = {
        data["type"]
        for _, data in lowering.graph.G.nodes(data=True)
        if (data.get("attributes") or {}).get("vocabulary")
    }
    assert element_types == {InterfaceNodeName.CONTAINER.value}

    capabilities = {
        data["attributes"]["vocabulary"]: data["attributes"]["capability"]
        for _, data in lowering.graph.G.nodes(data=True)
        if (data.get("attributes") or {}).get("vocabulary")
    }
    assert capabilities["div"] == "structure"
    assert capabilities["form"] == "form"
    assert capabilities["input"] == "value"
    assert capabilities["button"] == "action"


def test_positions_and_edge_ordinals_preserve_authored_appearance_order():
    lowering = _ingest("<div>first<input name=a>last</div>")
    graph = lowering.graph
    div_id, _ = _node_by_position(graph, (0,))
    dependencies = sorted(
        graph.G.predecessors(div_id),
        key=lambda child: graph.G.edges[child, div_id]["ordinal"],
    )
    assert [graph.G.nodes[node]["attributes"]["position"] for node in dependencies] == [
        (0, 0), (0, 1), (0, 2),
    ]
    assert [graph.G.edges[node, div_id]["ordinal"] for node in dependencies] == [
        0, 1, 2,
    ]


def test_unknown_tag_is_preserved_and_reported_or_rejected():
    lowering = _ingest("<widget-box>value</widget-box>")
    assert not lowering.complete
    assert lowering.shortfalls[0].tag == "widget-box"
    _, widget = _node_by_position(lowering.graph, (0,))
    assert widget["attributes"]["capability"] == "generic"

    with pytest.raises(HTMLGraphError, match="unsupported-html-tag"):
        _ingest("<widget-box></widget-box>", strict_vocabulary=True)


def test_normalized_html_round_trip_preserves_structure_attributes_and_text():
    source = (
        '<!DOCTYPE html><html><body><div class="knob selected">'
        '<label for="gain">Gain &amp; level</label>'
        '<input id="gain" type="range" disabled><!--control-->'
        '</div></body></html>'
    )
    lowering = _ingest(source)
    assert html_source_from_graph(lowering.graph) == (
        '<!DOCTYPE html><html><body><div class="knob selected">'
        '<label for="gain">Gain &amp; level</label>'
        '<input id="gain" type="range" disabled><!--control-->'
        '</div></body></html>'
    )


def test_malformed_nesting_is_rejected_instead_of_ambiguously_recovered():
    with pytest.raises(HTMLGraphError, match="does not match"):
        _ingest("<div><form></div></form>")
