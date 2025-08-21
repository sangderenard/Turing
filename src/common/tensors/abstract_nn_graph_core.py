from abc import ABC
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import networkx as nx

class AbstractNNGraphCore(ABC):
    """
    Graph-first mixin for AbstractTensor, for NN/segment management.
    Handles segment maps, segment registration, and NN graph assembly.
    """
    @dataclass
    class SegmentMap:
        """
        Describes a subnetwork/segment for graph-NN assembly.
        nodes: List of node ids in this segment
        edges: List of (src, dst) tuples (optionally with edge keys)
        node_props: Dict[node_id, dict] of node attributes
        edge_props: Dict[edge_tuple, dict] of edge attributes
        label: Optional segment label or id
        """
        nodes: list = field(default_factory=list)
        edges: list = field(default_factory=list)
        node_props: Dict = field(default_factory=dict)
        edge_props: Dict = field(default_factory=dict)
        label: str = None

    @staticmethod
    def create_segment_map(description):
        if isinstance(description, str):
            raise NotImplementedError("String DSL parsing not yet implemented.")
        if isinstance(description, dict):
            segments = list(description.values())
        else:
            segments = list(description)
        segmap = {}
        prev_nodes = None
        prev_label = None
        for idx, seg in enumerate(segments):
            label = seg.get('label')
            if not label:
                label = f'segment_{idx}'
            seg_type = seg.get('type', label)
            nodes = seg.get('nodes')
            num_nodes = seg.get('num_nodes')
            if nodes is None:
                base = seg_type
                if num_nodes is None:
                    raise ValueError(f"Segment '{label}' must specify 'nodes' or 'num_nodes'.")
                nodes = [f"{base}_{i}" for i in range(num_nodes)]
            edges = seg.get('edges')
            mode = seg.get('mode', 'fully_connected')
            if edges is None and prev_nodes is not None:
                if mode == 'fully_connected':
                    edges = [(src, dst) for src in prev_nodes for dst in nodes]
                elif mode == 'one_to_one':
                    if len(prev_nodes) != len(nodes):
                        raise ValueError(f"Cannot do one_to_one from {prev_label} to {label}: size mismatch.")
                    edges = list(zip(prev_nodes, nodes))
                elif mode == 'custom':
                    edges = []
                else:
                    raise ValueError(f"Unknown mode '{mode}' for segment '{label}'.")
            elif edges is None:
                edges = []
            extra = {k: v for k, v in seg.items() if k not in {'label', 'type', 'nodes', 'num_nodes', 'edges', 'mode', 'node_props', 'edge_props'}}
            segmap[label] = AbstractNNGraphCore.SegmentMap(
                nodes=nodes,
                edges=edges,
                node_props=seg.get('node_props', {}),
                edge_props=seg.get('edge_props', {}),
                label=label
            )
            for k, v in extra.items():
                setattr(segmap[label], k, v)
            prev_nodes = nodes
            prev_label = label
        return segmap

    def __init__(self, backing = None, in_nodes: Optional[List[int]] = None, out_nodes: Optional[List[int]] = None, track_time: bool = False):
        self.backing = backing if backing is not None else nx.DiGraph()
        self.track_time = track_time
        self.in_nodes = in_nodes if in_nodes is not None else []
        self.out_nodes = out_nodes if out_nodes is not None else []
        self.n_in = len(self.in_nodes)
        self.n_out = len(self.out_nodes)
        self.NN = None

    def _register_NN(self, nn, segment_map: Optional[Dict] = None, flags: Optional[dict] = None):
        self.NN = nn
        flags = flags or {}
        autowrap = flags.get('autowrap_inputs', True)
        if segment_map is None and hasattr(nn, "segments"):
            segment_map = nn.segments
        if segment_map is None:
            raise ValueError("No segment map provided and network has no .segments attribute.")
        for seg_id, seg in segment_map.items():
            if isinstance(seg, dict):
                seg = AbstractNNGraphCore.SegmentMap(**seg)
            for n in seg.nodes:
                props = dict(segment=seg_id)
                props.update(seg.node_props.get(n, {}))
                self.backing.add_node(n, **props)
            for e in seg.edges:
                src, dst = e[:2]
                props = dict(segment=seg_id)
                props.update(seg.edge_props.get(e, {}))
                self.backing.add_edge(src, dst, **props)
        if autowrap:
            if hasattr(nn, 'in_nodes') and hasattr(nn, 'out_nodes'):
                self.in_nodes = nn.in_nodes
                self.out_nodes = nn.out_nodes
