"""Exact parent-edge propagation must not scan the graph per chain link."""
import copy
import types
import random
import ast
import sys
from pathlib import Path

import networkx as nx
import pytest

from src.compiler import identity_concordance as concordance
sys.path.insert(0, str(Path(__file__).resolve().parent))
from abi_publication_full_scan import publish_program_abi_graph_identities as full_scan


class CountedNodes:
    def __init__(self, nodes):
        self.nodes = nodes
        self.visits = 0

    def __call__(self, *args, **kwargs):
        for item in self.nodes(*args, **kwargs):
            self.visits += 1
            yield item

    def __getitem__(self, key):
        self.visits += 1
        return self.nodes[key]

    def __iter__(self):
        return iter(self.nodes)

    def __len__(self):
        return len(self.nodes)


def chain(size, reverse=True):
    graph = nx.DiGraph()
    graph.graph.update(
        function_name="abi_chain",
        parameter_record_abi={"root": {"identity": "Node"}},
        identity_table={"root": (0,)},
        program_abi={"records": {"Node": {"identity": "Node", "fields": {
            "next": {"storage": "record", "record": "Node"},
        }}}},
    )
    for node in (reversed(range(size)) if reverse else range(size)):
        graph.add_node(node, value_id=node,
                       type="Input" if node == 0 else "GetAttr",
                       parents=[] if node == 0 else [(node - 1, "value")],
                       attributes={} if node == 0 else {"attribute": "next"})
    # Publication's authoritative dependencies are the actual parent records,
    # not a guessed NetworkX adjacency (which can precede edge rebuilding).
    return graph


def run(function, graph):
    graph = copy.deepcopy(graph)
    counted = CountedNodes(graph.nodes)
    graph.nodes = counted
    book, token = concordance.begin_identity_book()
    try:
        function(graph, force=False)
        rounds = [page for name, page in book.pages.items()
                  if name == "fixed_point_round"]
        facts = {node: {key: ast.dump(value, include_attributes=True)
                       if isinstance(value, ast.AST) else value
                       for key, value in data.items()}
                 for node, data in graph.nodes.nodes(data=True)}
        shapes = book.page("shape_transformation_state")
        shape_facts = {row: shapes.latest(row) for row in shapes.rows()}
        return facts, counted.visits, shape_facts
    finally:
        concordance.end_identity_book(token)


ORIGINAL = types.FunctionType(full_scan.__code__, concordance.__dict__)
ORIGINAL.__kwdefaults__ = full_scan.__kwdefaults__


@pytest.mark.parametrize("size", [8, 32, 128])
@pytest.mark.parametrize("reverse", [False, True])
def test_worklist_preserves_exact_final_identities(size, reverse):
    graph = chain(size, reverse)
    old, old_visits, _ = run(ORIGINAL, graph)
    new, visits, _ = run(concordance.publish_program_abi_graph_identities, graph)
    assert new == old
    assert all(data["attributes"]["program_abi_record_identity"] == "Node"
               for data in new.values())
    assert visits <= 20 * size, (visits, old_visits)


def test_loop_target_writes_invalidate_actual_consumers():
    graph = chain(4)
    graph.graph["parameter_record_abi"] = {}
    graph.nodes[0]["attributes"] = {"program_abi_sequence_row_identity": "Node"}
    graph.nodes[2].update(type="For", parents=[(0, "iterable")],
                          attributes={"loop_target_bindings": {"item": 1}})
    graph.nodes[1].update(type="Input", parents=[], attributes={})
    graph.nodes[3]["parents"] = [(1, "value")]
    old, _, _ = run(ORIGINAL, graph)
    new, _, _ = run(concordance.publish_program_abi_graph_identities, graph)
    assert new == old
    assert new[3]["attributes"]["program_abi_record_identity"] == "Node"


def test_span_shape_publication_is_the_same_book_fact():
    graph = chain(32)
    graph.graph["shape_scope"] = ("abi_worklist", "span")
    graph.graph["program_abi"]["records"]["Node"]["fields"]["column"] = {
        "storage": "span", "shape": (8,), "dtype": "float64", "rank": 1,
    }
    graph.nodes[31]["attributes"]["attribute"] = "column"
    old, _, old_shapes = run(ORIGINAL, graph)
    new, _, new_shapes = run(concordance.publish_program_abi_graph_identities, graph)
    assert new == old
    assert new_shapes == old_shapes
    assert old_shapes, "the original pass must publish the span's shape"


def test_record_constructor_append_publishes_exact_sequence_row_identity():
    graph = nx.DiGraph()
    graph.graph.update(
        function_name="record_append",
        parameter_record_abi={},
        identity_table={},
        program_abi={"records": {"Finding": {
            "identity": "Finding",
            "fields": {
                "kind": {"storage": "span", "dtype": "int64"},
                "value_id": {
                    "storage": "scalar", "dtype": "int64",
                    "optional": True,
                },
            },
        }}},
    )
    graph.add_node(
        0, value_id=0, type="List", parents=[],
        attributes={"aggregate_kind": "list"},
        expr_obj=ast.parse("[]", mode="eval").body,
    )
    graph.add_node(
        1, value_id=1, type="Call", op="Call", parents=[],
        attributes={"class_ref": "Finding"},
        expr_obj=ast.parse("Finding('bad', None)", mode="eval").body,
    )
    graph.add_node(
        2, value_id=2, type="append", op="append",
        parents=[(0, "operand"), (1, "arg:0")], attributes={},
        expr_obj=ast.parse(
            "found.append(Finding('bad', None))", mode="eval",
        ).body,
    )

    published, _, _ = run(
        concordance.publish_program_abi_graph_identities, graph,
    )

    assert published[1]["attributes"][
        "program_abi_record_identity"
    ] == "Finding"
    assert published[0]["attributes"][
        "program_abi_sequence_row_identity"
    ] == "Finding"


@pytest.mark.parametrize("seed", range(40))
def test_mixed_identity_rules_match_original_round_order(seed):
    rng = random.Random(seed)
    graph = chain(45)
    order = list(graph)
    rng.shuffle(order)
    graph = nx.DiGraph(graph.subgraph(order))
    # Explicitly randomize insertion order, preserving metadata/parent facts.
    data = dict(graph.nodes(data=True))
    graph.clear()
    graph.graph.update(chain(1).graph)
    for node in order:
        graph.add_node(node, **data[node])
    for node in range(1, 45):
        kind = rng.choice(("GetAttr", "Indexed", "Phi", "BoolOp", "Call",
                           "For", "LoopResult", "Comprehension"))
        parents = rng.sample(range(node), min(node, rng.randint(1, 3)))
        role = {"GetAttr": "value", "Indexed": "base", "Call": "arg:0",
                "For": "iterable", "LoopResult": "value",
                "Comprehension": "elt"}.get(kind, "value")
        attributes = {"attribute": "next"}
        expression = None
        if kind == "Call":
            expression = ast.parse("list(value)", mode="eval").body
        if kind == "For":
            attributes["loop_target_bindings"] = {"target": rng.randrange(45)}
        if kind == "LoopResult":
            attributes["result_kind"] = "collection"
        if kind == "Comprehension":
            expression = ast.parse("[value for value in rows]", mode="eval").body
        graph.nodes[node].update(type=kind, attributes=attributes,
                                 expr_obj=expression,
                                 parents=[(parent, role) for parent in parents])
        if rng.random() < .2:
            graph.nodes[node]["attributes"][rng.choice((
                "program_abi_record_identity", "program_abi_sequence_row_identity",
                "program_abi_indexed_value_identity"))] = rng.choice(("Node", "Other"))
    old, _, _ = run(ORIGINAL, graph)
    new, _, _ = run(concordance.publish_program_abi_graph_identities, graph)
    assert new == old
