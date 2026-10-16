"""NetworkX contract joined to the physical BitTensor record arena."""

import networkx as nx


class NetworkxEmulation(nx.DiGraph):
    """A DiGraph whose topology also reserves fixed-width physical records."""

    def __init__(self, bit_tensor_memory_graph):
        super().__init__()
        self.bit_tensor_memory_graph = bit_tensor_memory_graph

    def add_node(self, node_for_adding, **attr):
        if node_for_adding not in self._node:
            self.bit_tensor_memory_graph._add_node_record(node_for_adding)
        super().add_node(node_for_adding, **attr)
        return node_for_adding

    def add_edge(self, u_of_edge, v_of_edge, **attr):
        if u_of_edge not in self._node:
            self.add_node(u_of_edge)
        if v_of_edge not in self._node:
            self.add_node(v_of_edge)
        if not super().has_edge(u_of_edge, v_of_edge):
            self.bit_tensor_memory_graph._add_edge_record(u_of_edge, v_of_edge)
        super().add_edge(u_of_edge, v_of_edge, **attr)
        return (u_of_edge, v_of_edge)

    def add_nodes_from(self, nodes_for_adding, **attr):
        for item in nodes_for_adding:
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], dict):
                node, node_attributes = item
                merged = attr.copy()
                merged.update(node_attributes)
                self.add_node(node, **merged)
            else:
                self.add_node(item, **attr)

    def add_edges_from(self, ebunch_to_add, **attr):
        for edge in ebunch_to_add:
            if len(edge) == 3:
                source, target, edge_attributes = edge
                merged = attr.copy()
                merged.update(edge_attributes)
                self.add_edge(source, target, **merged)
            else:
                source, target = edge
                self.add_edge(source, target, **attr)

    def remove_edge(self, u, v):
        self.bit_tensor_memory_graph._remove_edge_record(u, v)
        super().remove_edge(u, v)

    def remove_node(self, n):
        for source, target in tuple(self.in_edges(n)) + tuple(self.out_edges(n)):
            if self.has_edge(source, target):
                self.remove_edge(source, target)
        self.bit_tensor_memory_graph._remove_node_record(n)
        super().remove_node(n)

    def remove_edges_from(self, ebunch):
        for source, target in tuple(ebunch):
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
            return nx.graphviews.generic_graph_view(self)
        backing_type = type(self.bit_tensor_memory_graph)
        copied_backing = backing_type(size=self.bit_tensor_memory_graph.hard_memory.size)
        copied = copied_backing.G
        copied.graph.update(self.graph.copy())
        for node, attributes in self.nodes(data=True):
            copied.add_node(node, **attributes.copy())
        for source, target, attributes in self.edges(data=True):
            copied.add_edge(source, target, **attributes.copy())
        return copied


meta_nodes = {}
root_meta_nodes = set()
master_graph = None
