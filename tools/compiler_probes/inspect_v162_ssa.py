from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

module, _outputs, _exports = pickle.loads(
    Path("build/patch_sequence_replay_v164/repository-ssa.pkl").read_bytes()
)
name = next(
    name for name in module.functions
    if "run_superstep__specialized" in name
    and "planned_region" not in name
)
function = module.functions[name]
print("FUNCTION", name, "args", len(function.args), "blocks", len(function.blocks))
for block_name, block in function.blocks.items():
    for instruction in block.instrs:
        attributes = instruction.attributes or {}
        if instruction.op == "Phi" and (
            attributes.get("binding") == "loop_carried"
            or attributes.get("initial_value_id") in {228, 274, 280, 282}
            or attributes.get("updated_value_id") in {274, 282, 286}
        ):
            print(
                "PHI",
                block_name,
                "res",
                None if instruction.res is None else instruction.res.id,
                "args",
                [value.id for value in instruction.args],
                "attrs",
                attributes,
            )
definitions = sum(
    1
    for block in function.blocks.values()
    for instruction in block.instrs
    if instruction.res is not None and instruction.res.id == 328
)
uses = sum(
    1
    for block in function.blocks.values()
    for instruction in block.instrs
    for argument in instruction.args
    if argument.id == 328
)
print("328 defs/uses", definitions, uses)
print("outputs", function.metadata.get("named_outputs"))
print("carried_ports", function.metadata.get("carried_port_values"))
