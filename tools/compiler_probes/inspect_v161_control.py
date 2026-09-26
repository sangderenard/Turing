from __future__ import annotations

import pickle
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.compiler.control_source import (
    CallBlock,
    ConditionalBlock,
    LoopBlock,
    ParallelDeployment,
    ResourceScopeBlock,
    SequenceBlock,
    StateMachineTick,
    WhileBlock,
    enrich_represented_conditionals,
)
from src.compiler.glsl_deployment_strategy import (
    _ordinary_conditional_control_programs,
)


checkpoint = Path("build/patch_sequence_source_v147/pre-frame-link.pkl")
positional, _keywords = pickle.loads(checkpoint.read_bytes())
root = positional[0].deployment
shell = root.function_shells[16]
shell.prepare_graph_precompile(
    structural_ssa_only=True,
    progress=lambda message: print(message, flush=True),
)
control = shell.shell_control_program
candidates = _ordinary_conditional_control_programs(
    shell.process_graph,
    control.region_indices,
    shell.dispatch_subgraphs,
)
control, direct_receipts = enrich_represented_conditionals(
    control,
    candidates,
    first_synthetic_value_id=(
        max(
            int(data.get("value_id", node_id))
            for node_id, data in shell.process_graph.G.nodes(data=True)
        ) + 1
    ),
)


def walk(block, depth=0):
    prefix = "  " * depth
    if isinstance(block, (LoopBlock, WhileBlock)):
        print(
            prefix
            + f"{type(block).__name__} source={block.source_loop_node_id} "
            + f"carries={block.carried_aliases} results={block.result_ports}",
            flush=True,
        )
        if isinstance(block, WhileBlock):
            walk(block.condition, depth + 1)
        walk(block.body, depth + 1)
    elif isinstance(block, ConditionalBlock):
        print(
            prefix
            + f"Conditional source={block.source_node_id} "
            + f"aliases={block.carried_aliases}",
            flush=True,
        )
        walk(block.body, depth + 1)
        if block.orelse is not None:
            walk(block.orelse, depth + 1)
    elif isinstance(block, SequenceBlock):
        for child in block.blocks:
            walk(child, depth)
    elif isinstance(block, ResourceScopeBlock):
        walk(block.body, depth)
    elif isinstance(block, CallBlock):
        walk(block.callee, depth)
    elif isinstance(block, StateMachineTick):
        for _value, body in block.cases:
            walk(body, depth)
        if block.default is not None:
            walk(block.default, depth)
    elif isinstance(block, ParallelDeployment):
        for lane in block.lanes:
            walk(lane, depth)


print("CONTROL", flush=True)
walk(control.root)
print("DIRECT_RECEIPTS", direct_receipts, flush=True)
print(
    "RECEIPTS",
    shell.process_graph.G.graph.get("conditional_control_enrichment_receipts", ()),
    flush=True,
)
