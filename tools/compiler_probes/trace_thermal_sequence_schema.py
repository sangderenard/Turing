from pathlib import Path
import contextlib
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler import fortran_c_shell as shell_module


root = Path(__file__).resolve().parents[3]
turing = root / "turing"
source_path = root / "spectral-analyzer" / "electrical_tensor_network.py"
sys.path.insert(0, str(source_path.parent))
contract = ExtractionContract(
    turing / "extraction_contracts" / "program_extraction.yaml"
).with_sources([
    ("electrical_tensor_network", source_path),
])
OUT = sys.__stdout__
original = shell_module._field_slot_ops


def traced(graph_obj, *args, **kwargs):
    result = original(graph_obj, *args, **kwargs)
    if graph_obj.graph.get("function_name") == "set_thermal_temperatures":
        print("THERMAL_GRAPH_ABI", graph_obj.graph.get("program_abi"), file=OUT)
        print("THERMAL_IDENTITIES", graph_obj.graph.get("identity_table"), file=OUT)
        print("THERMAL_DECLARATIONS", result[8], file=OUT)
        print("THERMAL_INITIALIZATIONS", result[6], file=OUT)
        for node_id, data in graph_obj.nodes(data=True):
            if int(data.get("value_id", node_id)) == 3:
                print("THERMAL_VALUE_3", node_id, data, file=OUT)
    return result


shell_module._field_slot_ops = traced
try:
    with open(os.devnull, "w") as quiet, \
            contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
        try:
            lower_ast_source_to_ssa(
                source_path.read_text(encoding="utf-8"),
                "ElectricalDeviceRegistry.set_thermal_temperatures",
                name="thermal_sequence_schema_trace",
                extraction_contract=contract,
                runtime_closure_only=True,
            )
        except Exception as error:
            print(type(error).__name__, str(error), file=OUT)
finally:
    shell_module._field_slot_ops = original
