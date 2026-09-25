from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm


source = """
from dataclasses import dataclass
from typing import Mapping
import torch

@dataclass(frozen=True)
class Reading:
    values: Mapping[str, torch.Tensor]

def sample(reading: Reading, key: str, index: int):
    selected = reading.values[key]
    return selected[:, index]
"""

resolved = []
module, _outputs, _exports = lower_ast_source_to_ssa(
    source,
    "sample",
    name="keyed_tensor_rank2",
    extraction_contract=(
        Path(__file__).resolve().parents[2]
        / "extraction_contracts"
        / "program_extraction.yaml"
    ),
    tensor_ssa_reference=c_backend_repository_ssa_reference(),
    progress=lambda _message: None,
    resolved_process_graph_sink=resolved.append,
)

for entry in resolved[0].function_table:
    graph = getattr(getattr(entry, "graph", None), "G", None)
    if graph is None or graph.graph.get("function_name") != "sample":
        continue
    for node_id, data in graph.nodes(data=True):
        print(
            "GRAPH", node_id, data.get("type"), data.get("op"),
            data.get("expr_obj"), data.get("tensor"), data.get("attributes"),
            data.get("parents"),
        )

for name, function in module.functions.items():
    if "planned_region" not in name and name != "keyed_tensor_rank2__sample":
        continue
    print(name)
    for argument in function.args:
        print("ARG", argument.id, argument.shape, argument.accounting)
    for block in function.blocks.values():
        for instruction in block.instrs:
            print(
                instruction.op,
                [(value.id, value.shape, value.accounting) for value in instruction.args],
                None if instruction.res is None else (
                    instruction.res.id,
                    instruction.res.shape,
                    instruction.res.accounting,
                ),
                instruction.attributes,
            )

emitted = emit_ssa_function_to_llvm(
    module,
    "keyed_tensor_rank2__sample",
    entry_name="keyed_tensor_rank2_sample",
)
print("LLVM_SHORTFALLS", emitted.shortfalls)
