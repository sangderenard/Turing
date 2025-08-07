import pytest

from src.turing_machine.turing_provenance import ProvenanceGraph, ProvNode, ProvEdge
from src.compiler.ssa_builder import process_graph_to_ssa_instrs
from src.compiler.tape_compiler import TapeCompiler
from src.compiler.process_graph_helper import provenance_to_process_graph, reduce_cycles_to_mu


@pytest.mark.xfail(reason="ProcessGraph build_from_expression requires full memory context")
def test_process_graph_ssa_and_compile():
    """End-to-end check from provenance -> ProcessGraph -> SSA -> tape instructions."""
    pg = ProvenanceGraph()
    # simple nand chain: a = zeros(); b = nand(a, a)
    pg._nodes = [
        ProvNode(0, 'zeros', (2,), {}, 0),  # produce constant via zeros
        ProvNode(1, 'nand', (0, 0), {}, 1),
    ]
    pg._edges = [
        ProvEdge(0, 1, 0),
        ProvEdge(0, 1, 1),
    ]

    proc = provenance_to_process_graph(pg)
    instrs = process_graph_to_ssa_instrs(proc)
    assert any(i.op == 'nand' for i in instrs)

    tc = TapeCompiler(pg, bit_width=1)
    _, instructions, _ = tc.compile_ssa(instrs, process_graph=proc)
    from src.hardware.analog_spec import Opcode
    assert any(ins.opcode == Opcode.NAND for ins in instructions)


@pytest.mark.xfail(reason="ProcessGraph build_from_expression requires full memory context")
def test_compile_pipeline_from_provenance():
    """``TapeCompiler.compile`` should route through SSA automatically."""
    pg = ProvenanceGraph()
    pg._nodes = [
        ProvNode(0, 'zeros', (2,), {}, 0),
        ProvNode(1, 'nand', (0, 0), {}, 1),
    ]
    pg._edges = [
        ProvEdge(0, 1, 0),
        ProvEdge(0, 1, 1),
    ]
    tc = TapeCompiler(pg, bit_width=1)
    _, instructions, _ = tc.compile()
    from src.hardware.analog_spec import Opcode
    assert any(ins.opcode == Opcode.NAND for ins in instructions)


@pytest.mark.xfail(reason="ProcessGraph build_from_expression requires full memory context")
def test_provenance_to_process_graph_roundtrip():
    """Helper should expose ProcessGraph utilities for provenance data."""
    pg = ProvenanceGraph()
    pg._nodes = [
        ProvNode(0, 'zeros', (), {}, 0),
        ProvNode(1, 'nand', (0, 0), {}, 1),
    ]
    pg._edges = [
        ProvEdge(0, 1, 0),
        ProvEdge(0, 1, 1),
    ]
    proc = provenance_to_process_graph(pg)
    # The ProcessGraph should mirror the provenance node count
    assert len(proc.G.nodes) == len(pg.nodes)


def test_reduce_cycles_to_mu_rewrites_backedges():
    pg = ProvenanceGraph()
    pg._nodes = [
        ProvNode(0, 'zeros', (), {}, 0),
        ProvNode(1, 'nand', (0, 0), {}, 1),
        ProvNode(2, 'nand', (1, 1), {}, 2),
    ]
    pg._edges = [
        ProvEdge(0, 1, 0),
        ProvEdge(0, 1, 1),
        ProvEdge(1, 2, 0),
        ProvEdge(1, 2, 1),
        ProvEdge(2, 1, 0),
    ]
    reduce_cycles_to_mu(pg)
    assert any(n.op == 'mu' for n in pg.nodes)
    assert not any(e.src_idx == 2 and e.dst_idx == 1 and e.arg_pos == 0 for e in pg.edges)


def test_loop_edges_lower_to_phi():
    """Cycles in a ProcessGraph become φ nodes during SSA lowering."""
    import networkx as nx

    class LoopPG:
        def __init__(self):
            self.G = nx.DiGraph()
            # Two node cycle: 0 -> 1 -> 0
            self.G.add_node(0, label='a', expr_obj=None, parents=[(1, 'arg0')], children=[(1, 'arg0')])
            self.G.add_node(1, label='b', expr_obj=None, parents=[(0, 'arg0')], children=[(0, 'arg0')])
            self.G.add_edge(0, 1)
            self.G.add_edge(1, 0)
            self.scheduler = self

        def compute_levels(self, method, order):
            return {0: 0, 1: 1}

    pg = LoopPG()
    instrs = process_graph_to_ssa_instrs(pg)
    assert any(i.op == 'phi' for i in instrs)


def test_concurrency_allocation_process_graph():
    import networkx as nx

    class MiniPG:
        def __init__(self):
            self.G = nx.DiGraph()
            self.G.add_node(0, label='zeros', expr_obj=None, parents=[], children=[(1, 'arg0'), (2, 'arg0')])
            self.G.add_node(1, label='nand', expr_obj=None, parents=[(0, 'arg0')], children=[])
            self.G.add_node(2, label='nand', expr_obj=None, parents=[(0, 'arg0')], children=[])
            self.G.add_edge(0, 1)
            self.G.add_edge(0, 2)
            self.scheduler = self
            self.mG = nx.DiGraph()

        # Scheduler API used by process_graph_to_ssa_instrs/compile_ssa
        def compute_levels(self, method, order, interference_mode='asap-maxslack'):
            self.proc_interference_graph = nx.Graph()
            self.proc_interference_graph.add_nodes_from([0, 1, 2])
            self.proc_interference_graph.add_edge(1, 2)
            return {0: 0, 1: 1, 2: 1}

    pg_proc = MiniPG()
    ssa_instrs = process_graph_to_ssa_instrs(pg_proc)
    tc = TapeCompiler(ProvenanceGraph(), bit_width=1)
    tc.compile_ssa(ssa_instrs, process_graph=pg_proc)
    # The two parallel NANDs should occupy different registers
    assert tc.memory_map[1] != tc.memory_map[2]


def test_process_graph_to_ssa_respects_schedule():
    """Ensure the ProcessGraph schedule dictates SSA emission order."""
    import networkx as nx

    class OrderPG:
        def __init__(self):
            self.G = nx.DiGraph()
            # Simple chain 0 -> 1 but scheduler will invert execution order
            self.G.add_node(0, label='a', expr_obj=None, parents=[], children=[(1, 'arg0')])
            self.G.add_node(1, label='b', expr_obj=None, parents=[(0, 'arg0')], children=[])
            self.G.add_edge(0, 1)
            self.scheduler = self

        def compute_levels(self, method, order):
            # Force node 1 to appear before node 0
            return {0: 1, 1: 0}

    pg = OrderPG()
    instrs = process_graph_to_ssa_instrs(pg)
    # First instruction should correspond to node 1 due to scheduling
    assert instrs[0].res.id == 1


def test_memory_nodes_allocate_data_space():
    """ProcessGraph memory nodes should be mapped to data addresses."""
    import networkx as nx

    class MemPG:
        def __init__(self):
            self.G = nx.DiGraph()
            self.G.add_node(0, label='zeros', expr_obj=None, parents=[], children=[])
            # Separate memory graph holding a single storage node
            self.mG = nx.DiGraph()
            self.mG.add_node(100)
            self.scheduler = self

        def compute_levels(self, method, order, interference_mode='asap-maxslack'):
            self.proc_interference_graph = nx.Graph()
            self.proc_interference_graph.add_nodes_from([0, 100])
            return {0: 0}

    pg_proc = MemPG()
    ssa_instrs = process_graph_to_ssa_instrs(pg_proc)
    tc = TapeCompiler(ProvenanceGraph(), bit_width=1)
    tc.compile_ssa(ssa_instrs, process_graph=pg_proc)
    assert 100 in tc.data_map
    assert 100 not in tc.memory_map
