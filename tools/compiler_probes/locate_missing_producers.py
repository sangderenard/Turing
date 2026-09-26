import json
import pickle
from pathlib import Path


root = Path(__file__).resolve().parents[2] / "build" / "full_formal_diagnostic"
rows = json.loads((root / "formals.json").read_text())
with (root / "repository-ssa.pkl").open("rb") as stream:
    module = pickle.load(stream)

for row in rows:
    value_id = int(row["value_id"])
    if value_id in {208, 211, 212, 215, 252} and "step_with_dt_control_used" in row["function"]:
        continue
    owner = row["function"]
    print("\n", owner, value_id)
    prefix = owner + "__planned_region_"
    for name, function in module.functions.items():
        if name != owner and not name.startswith(prefix):
            continue
        for block_name, block in function.blocks.items():
            for index, instruction in enumerate(block.instrs):
                attrs = instruction.attributes or {}
                if (
                    instruction.res is not None and int(instruction.res.id) == value_id
                ) or attrs.get("source_output_id") == value_id or value_id in attrs.get("output_ids", ()):
                    print(name, block_name, index, instruction.op,
                          "args", [arg.id for arg in instruction.args],
                          "res", None if instruction.res is None else instruction.res.id,
                          "attrs", {key: attrs[key] for key in (
                              "source_output_id", "output_ids", "region_index",
                              "callee", "binding", "source_operation",
                          ) if key in attrs})
