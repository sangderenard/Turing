import pickle
import sys
from pathlib import Path

path = Path(__file__).resolve().parents[2] / "build" / "full_formal_diagnostic" / "repository-ssa.pkl"
with path.open("rb") as stream:
    module = pickle.load(stream)
for needle in sys.argv[1:]:
    for name, function in module.functions.items():
        if needle not in name:
            continue
        print("FUNCTION", name, "ARGS", [(v.id, v.dtype) for v in function.args])
        for block_name, block in function.blocks.items():
            print(" BLOCK", block_name)
            for index, instruction in enumerate(block.instrs):
                print("  ", index, instruction.op,
                      [v.id for v in instruction.args],
                      None if instruction.res is None else instruction.res.id,
                      instruction.attributes)
