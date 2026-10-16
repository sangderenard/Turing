"""NetworkX-shaped access to the physical BitTensor graph.

This is the repository's existing emulation boundary.  It intentionally does
not inherit from or contain a NetworkX graph.  Topology lives in the memory
graph's NodeEntry and EdgeEntry allocations; the dictionaries below retain the
Python semantic payload attached to those physical identities.
"""

from collections.abc import Mapping


class _NodeView(Mapping):
    def __init__(self, graph):
        self._graph = graph

    def __getitem__(self, node):
        return self._graph._node[node]

    def __iter__(self):
        return iter(self._graph._node)

    def __len__(self):
        return len(self._graph._node)

    def __call__(self, data=False, default=None):
        if data is False:
            return iter(self._graph._node)
        if data is True:
            return iter(self._graph._node.items())
        return (
            (node, attributes.get(data, default))
            for node, attributes in self._graph._node.items()
        )


class _EdgeView(Mapping):
    def __init__(self, graph, incoming=False):
        self._graph = graph
        self._incoming = incoming

    def _triples(self, nbunch=None):
        graph = self._graph
        if self._incoming:
            for target in graph.nbunch_iter(nbunch):
                for source, attributes in graph._pred[target].items():
                    yield source, target, attributes
        else:
            for source in graph.nbunch_iter(nbunch):
                for target, attributes in graph._succ[source].items():
                    yield source, target, attributes

    def __getitem__(self, edge):
        source, target = edge[:2]
        return self._graph._succ[source][target]

    def __iter__(self):
        return ((source, target) for source, target, _ in self._triples())

    def __len__(self):
        return sum(len(targets) for targets in self._graph._succ.values())

    def __call__(self, nbunch=None, data=False, default=None):
        triples = self._triples(nbunch)
        if data is False:
            return ((source, target) for source, target, _ in triples)
        if data is True:
            return triples
        return (
            (source, target, attributes.get(data, default))
            for source, target, attributes in triples
        )


class _DegreeView(Mapping):
    def __init__(self, graph, direction):
        self._graph = graph
        self._direction = direction

    def __getitem__(self, node):
        incoming = len(self._graph._pred[node])
        outgoing = len(self._graph._succ[node])
        if self._direction == "in":
            return incoming
        if self._direction == "out":
            return outgoing
        return incoming + outgoing

    def __iter__(self):
        return iter(self._graph._node)

    def __len__(self):
        return len(self._graph._node)

    def __call__(self, nbunch=None, weight=None):
        def value(node):
            if weight is None:
                return self[node]
            incoming = sum(
                attributes.get(weight, 1)
                for attributes in self._graph._pred[node].values()
            )
            outgoing = sum(
                attributes.get(weight, 1)
                for attributes in self._graph._succ[node].values()
            )
            if self._direction == "in":
                return incoming
            if self._direction == "out":
                return outgoing
            return incoming + outgoing

        try:
            single = nbunch is not None and nbunch in self._graph
        except TypeError:
            single = False
        if single:
            return value(nbunch)
        return ((node, value(node)) for node in self._graph.nbunch_iter(nbunch))


class NetworkxEmulation:
    """Directed graph protocol backed by ``BitTensorMemoryGraph`` records."""

    def __init__(self, bit_tensor_memory_graph=None):
        # NetworkX's pure algorithms construct ``G.__class__()`` for results.
        # That is an algorithm protocol, not authority over storage: give the
        # result a fresh physical arena of the same repository type.
        if bit_tensor_memory_graph is None:
            from ..memory_graph import BitTensorMemoryGraph
            bit_tensor_memory_graph = BitTensorMemoryGraph(size=512)
            bit_tensor_memory_graph.G = self
        self.bit_tensor_memory_graph = bit_tensor_memory_graph
        self.graph = {}
        self._node = {}
        self._succ = {}
        self._pred = {}
        self._install_views()

    def _install_views(self):
        self._adj = self._succ
        self.nodes = _NodeView(self)
        self.edges = _EdgeView(self)
        self.out_edges = self.edges
        self.in_edges = _EdgeView(self, incoming=True)
        self.degree = _DegreeView(self, "both")
        self.in_degree = _DegreeView(self, "in")
        self.out_degree = _DegreeView(self, "out")

    def __getstate__(self):
        state = dict(self.__dict__)
        for name in (
            "nodes", "edges", "out_edges", "in_edges", "degree",
            "in_degree", "out_degree", "_adj",
        ):
            state.pop(name, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._install_views()

    @property
    def adj(self):
        return self._succ

    @property
    def succ(self):
        return self._succ

    @property
    def pred(self):
        return self._pred

    @property
    def name(self):
        return self.graph.get("name", "")

    @name.setter
    def name(self, value):
        self.graph["name"] = value

    def __iter__(self):
        return iter(self._node)

    def __contains__(self, node):
        return node in self._node

    def __len__(self):
        return len(self._node)

    def __getitem__(self, node):
        return self._succ[node]

    def is_directed(self):
        return True

    def is_multigraph(self):
        return False

    def has_node(self, node):
        return node in self._node

    def has_edge(self, source, target):
        return source in self._succ and target in self._succ[source]

    def number_of_nodes(self):
        return len(self._node)

    def number_of_edges(self, source=None, target=None):
        if source is not None and target is not None:
            return int(self.has_edge(source, target))
        return len(self.edges)

    order = number_of_nodes

    def size(self, weight=None):
        if weight is None:
            return len(self.edges)
        return sum(
            attributes.get(weight, 1)
            for _, _, attributes in self.edges(data=True)
        )

    def nbunch_iter(self, nbunch=None):
        if nbunch is None:
            return iter(self._node)
        try:
            if nbunch in self._node:
                return iter((nbunch,))
        except TypeError:
            pass
        return (node for node in nbunch if node in self._node)

    def neighbors(self, node):
        return iter(self._succ[node])

    successors = neighbors

    def predecessors(self, node):
        return iter(self._pred[node])

    def get_edge_data(self, source, target, default=None):
        return self._succ.get(source, {}).get(target, default)

    def _add_node_topology(self, node, attributes=None):
        if node not in self._node:
            self._node[node] = {}
            self._succ[node] = {}
            self._pred[node] = {}
        if attributes:
            self._node[node].update(attributes)
        return node

    def add_node(self, node_for_adding, **attr):
        if node_for_adding not in self._node:
            self.bit_tensor_memory_graph._add_node_record(node_for_adding)
        return self._add_node_topology(node_for_adding, attr)

    def _add_edge_topology(self, source, target, attributes=None):
        self._add_node_topology(source)
        self._add_node_topology(target)
        edge = self._succ[source].get(target)
        if edge is None:
            edge = {}
            self._succ[source][target] = edge
            self._pred[target][source] = edge
        if attributes:
            edge.update(attributes)
        return source, target

    def add_edge(self, source, target, **attr):
        if source not in self._node:
            self.add_node(source)
        if target not in self._node:
            self.add_node(target)
        if not self.has_edge(source, target):
            self.bit_tensor_memory_graph._add_edge_record(source, target)
        return self._add_edge_topology(source, target, attr)

    def add_nodes_from(self, nodes_for_adding, **attr):
        for item in nodes_for_adding:
            if (
                isinstance(item, tuple) and len(item) == 2
                and isinstance(item[1], Mapping)
            ):
                node, node_attributes = item
                merged = dict(attr)
                merged.update(node_attributes)
                self.add_node(node, **merged)
            else:
                self.add_node(item, **attr)

    def add_edges_from(self, ebunch_to_add, **attr):
        for edge in ebunch_to_add:
            source, target = edge[:2]
            merged = dict(attr)
            if len(edge) == 3:
                merged.update(edge[2])
            self.add_edge(source, target, **merged)

    def add_weighted_edges_from(self, ebunch_to_add, weight="weight", **attr):
        for source, target, edge_weight in ebunch_to_add:
            merged = dict(attr)
            merged[weight] = edge_weight
            self.add_edge(source, target, **merged)

    def remove_edge(self, source, target):
        if not self.has_edge(source, target):
            raise KeyError((source, target))
        self.bit_tensor_memory_graph._remove_edge_record(source, target)
        del self._succ[source][target]
        del self._pred[target][source]

    def remove_node(self, node):
        if node not in self._node:
            raise KeyError(node)
        for source in tuple(self._pred[node]):
            self.remove_edge(source, node)
        for target in tuple(self._succ[node]):
            self.remove_edge(node, target)
        self.bit_tensor_memory_graph._remove_node_record(node)
        del self._node[node]
        del self._succ[node]
        del self._pred[node]

    def remove_edges_from(self, ebunch):
        for edge in tuple(ebunch):
            source, target = edge[:2]
            if self.has_edge(source, target):
                self.remove_edge(source, target)

    def remove_nodes_from(self, nodes):
        for node in tuple(nodes):
            if node in self:
                self.remove_node(node)

    def clear_edges(self):
        self.remove_edges_from(tuple(self.edges))

    def clear(self):
        self.remove_nodes_from(tuple(self.nodes))
        self.graph.clear()

    def to_edges(self, target):
        return list(self.in_edges(target))

    def from_edges(self, source):
        return list(self.out_edges(source))

    def copy(self, as_view=False):
        if as_view:
            return self
        return self._induced_copy(set(self._node))

    def _induced_copy(self, selected):
        backing_type = type(self.bit_tensor_memory_graph)
        copied = backing_type(
            size=max(512, self.bit_tensor_memory_graph.hard_memory.size)
        ).G
        copied.graph.update(self.graph.copy())
        copied.add_nodes_from(
            (node, self._node[node].copy()) for node in selected
        )
        copied.add_edges_from(
            (source, target, attributes.copy())
            for source, target, attributes in self.edges(data=True)
            if source in selected and target in selected
        )
        return copied

    def subgraph(self, nodes):
        return self._induced_copy({node for node in nodes if node in self})

    def edge_subgraph(self, edges):
        selected = {
            tuple(edge[:2]) for edge in edges if self.has_edge(*edge[:2])
        }
        copied = self._induced_copy({node for edge in selected for node in edge})
        copied.remove_edges_from(
            edge for edge in tuple(copied.edges) if edge not in selected
        )
        return copied

    def reverse(self, copy=True):
        backing_type = type(self.bit_tensor_memory_graph)
        reversed_graph = backing_type(
            size=max(512, self.bit_tensor_memory_graph.hard_memory.size)
        ).G
        reversed_graph.graph.update(self.graph.copy())
        reversed_graph.add_nodes_from(
            (node, attributes.copy())
            for node, attributes in self.nodes(data=True)
        )
        reversed_graph.add_edges_from(
            (target, source, attributes.copy())
            for source, target, attributes in self.edges(data=True)
        )
        return reversed_graph


meta_nodes = {}
root_meta_nodes = set()
master_graph = None
