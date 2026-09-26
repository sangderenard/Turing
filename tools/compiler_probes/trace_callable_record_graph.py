from pathlib import Path
import ast
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa


source = r'''
from dataclasses import dataclass
from typing import Callable

@dataclass
class Law:
    advance: Callable[[float], float]

def add_one(value):
    return value + 1.0

def root(value):
    law = Law(add_one)
    return law.advance(value)
'''

captured = []
lower_ast_source_to_ssa(
    source,
    "root",
    name="callable_record_trace",
    extraction_contract=ExtractionContract(
        Path(__file__).resolve().parents[2]
        / "extraction_contracts" / "program_extraction.yaml"
    ),
    resolved_process_graph_sink=captured.append,
    stop_after_compilation_unit_plan=True,
)

graph = captured[0]
for entry in graph.function_table:
    function_graph = entry.graph
    print(f"\nFUNCTION {entry.name} ref={entry.reference.address}")
    print("metadata", {
        key: value for key, value in function_graph.G.graph.items()
        if key in {
            "record_abi", "parameter_record_abi", "planner_specializations",
            "sequence_abi", "mapping_abi", "class_table", "program_abi",
            "sequence_record_abi", "parameter_sequence_record_abi",
        }
    })
    for node_id, data in function_graph.G.nodes(data=True):
        expr = data.get("expr_obj")
        text = ast.unparse(expr) if isinstance(expr, ast.AST) else None
        attributes = data.get("attributes") or {}
        if (
            entry.name in {"root", "run", "add_one"}
            or data.get("type") == "StaticReference"
            or "Law" in repr(attributes)
            or "function_ref" in repr(attributes)
        ):
            print(node_id, data.get("type"), data.get("op"), text,
                  "parents=", data.get("parents"),
                  "children=", data.get("children"),
                  "attributes=", attributes)
