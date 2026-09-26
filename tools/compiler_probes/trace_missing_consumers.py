import json
import pickle
from pathlib import Path


root = Path(__file__).resolve().parents[2] / "build" / "full_formal_diagnostic"
rows = json.loads((root / "formals.json").read_text())
with (root / "repository-ssa.pkl").open("rb") as stream:
    module = pickle.load(stream)

literal_ids = {208, 211, 212, 215, 252}
for row in rows:
    value_id = int(row["value_id"])
    if value_id in literal_ids and "step_with_dt_control_used" in row["function"]:
        continue
    function = module.functions[row["function"]]
    print("\n", row["function"], value_id)
    for block_name, block in function.blocks.items():
        for index, instruction in enumerate(block.instrs):
            positions = [
                position for position, argument in enumerate(instruction.args)
                if int(argument.id) == value_id
            ]
            if not positions:
                continue
            attributes = dict(instruction.attributes or {})
            print(" ", block_name, index, instruction.op, "positions", positions,
                  "res", None if instruction.res is None else instruction.res.id,
                  "attrs", {key: attributes[key] for key in (
                      "callee", "region_index", "source_output_id", "callsite_id",
                      "binding", "incoming_blocks", "target", "true_target",
                      "false_target", "output_ids", "feed_ids",
                  ) if key in attributes})
            callee = module.functions.get(str(attributes.get("callee") or ""))
            if callee is not None:
                for position in positions:
                    if position < len(callee.args):
                        formal = callee.args[position]
                        print("    callee-formal", position, formal.id, formal.dtype,
                              formal.accounting)
