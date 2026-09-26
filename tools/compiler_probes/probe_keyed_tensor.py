from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.glsl_deployment_strategy import _fold_callsite_structural_values
import src.compiler.glsl_deployment_strategy as strategy
from src.compiler.identity_concordance import identity_book
import src.compiler.tensor_ssa_lowering as tensor_lowering
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)


_original_build_plan = strategy._build_shell_hierarchy_plan


def _traced_build_plan(shell):
    graph = shell.process_graph.G
    if graph.graph.get("function_name") == "sample":
        print("PREPLAN")
        for node_id, data in graph.nodes(data=True):
            print("PN", node_id, data.get("op"), data.get("attributes"), data.get("tensor"))
    plan = _original_build_plan(shell)
    if graph.graph.get("function_name") == "sample":
        print("PLAN", plan)
    return plan


strategy._build_shell_hierarchy_plan = _traced_build_plan


def _types(module):
    rows = []
    for name in ("keyed_tensor_lookup__sample", "ssa_sequence_2_lookup"):
        fn = module.functions.get(name)
        if fn is None:
            continue
        rows.append((name, tuple((v.id, v.dtype, v.accounting.get("physical_dtype")) for v in fn.args)))
    return rows


_original_lower_tensor = tensor_lowering.lower_tensor_calls_to_repository_ssa
def _traced_lower_tensor(module, *args, **kwargs):
    print("TLBEFORE", _types(module))
    result = _original_lower_tensor(module, *args, **kwargs)
    print("TLAFTER", _types(module))
    return result
tensor_lowering.lower_tensor_calls_to_repository_ssa = _traced_lower_tensor

_original_propagate = tensor_lowering.propagate_repository_ssa_call_metadata
def _traced_propagate(module, *args, **kwargs):
    print("TPBEFORE", _types(module))
    result = _original_propagate(module, *args, **kwargs)
    print("TPAFTER", _types(module))
    return result
tensor_lowering.propagate_repository_ssa_call_metadata = _traced_propagate


source = """
from dataclasses import dataclass
from typing import Mapping
import torch

@dataclass(frozen=True)
class Reading:
    voltage_v: Mapping[str, torch.Tensor]

def sample(reading: Reading, key: str):
    selected = reading.voltage_v[key]
    return selected.real.sum()
"""

resolved = []
module, outputs, exports = lower_ast_source_to_ssa(
    source,
    "sample",
    name="keyed_tensor_lookup",
    extraction_contract=(
        Path(__file__).resolve().parents[2]
        / "extraction_contracts"
        / "program_extraction.yaml"
    ),
    progress=lambda message: print(message, flush=True),
    tensor_ssa_reference=c_backend_repository_ssa_reference(),
    resolved_process_graph_sink=resolved.append,
)

for entry in resolved[0].function_table:
    graph = getattr(getattr(entry, "graph", None), "G", None)
    if graph is None or graph.graph.get("function_name") != "sample":
        continue
    _fold_callsite_structural_values(entry.graph)
    print("GRAPH")
    print("ABI", graph.graph.get("parameter_record_abi"))
    print("PROGRAM_ABI", graph.graph.get("program_abi"))
    for node_id, data in graph.nodes(data=True):
        print(node_id, data.get("type"), data.get("op"), data.get("value_id"), data.get("attributes"), data.get("tensor"), data.get("parents"))

for name, function in module.functions.items():
    if name.startswith("keyed_tensor_lookup__sample") or name == "ssa_sequence_2_lookup":
        print("ROOT", name)
        print("ARGS")
        for value in function.args:
            print(value.id, value.dtype, value.shape, value.accounting)
        print("SEQUENCES")
        for sequence in module.sequence_tables.get(name, ()).sequences.values() if name in module.sequence_tables else ():
            print(sequence)
        print("TENSORS")
        tensor_table = module.tensor_tables.get(name)
        if tensor_table is not None:
            for tensor in tensor_table.tensors.values():
                print(tensor)
        print("INSTRUCTIONS")
        for block_name, block in function.blocks.items():
            for instruction in block.instrs:
                print(
                    block_name,
                    instruction.op,
                    [(value.id, value.dtype, value.shape, value.accounting) for value in instruction.args],
                    None if instruction.res is None else (
                        instruction.res.id,
                        instruction.res.dtype,
                        instruction.res.shape,
                        instruction.res.accounting,
                    ),
                    instruction.attributes,
                )

helper = module.functions["ssa_sequence_2_lookup"]
print("HMD", helper.metadata)
for index, value in enumerate(helper.args):
    print("HARG", index, value.id, value.dtype, value.shape, value.accounting)
page = identity_book(module).pages.get("tensor_shape_enrichment")
if page is not None:
    for row in (("ssa_sequence_2_lookup", 2, "dtype"),):
        print("HISTORY", row, page.history(row))
