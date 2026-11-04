"""Print planned-region bodies of the saved SSA with constant operands (args: batch, region list).

Report only; edits nothing.  Run from the compressor-validation checkout with
PYTHONPATH set to it and the system Python: the saved SSA pickles in %TEMP% were
produced there and unpickle only against its compiler.  The game itself is read
from this repository.  See HANDOFF_MEOWDOKU_WASM_GAME_2026-11-04.md.
"""
import os
import pickle
import sys
from pathlib import Path

BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 25
REGIONS = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [9, 11, 12, 13, 14]
PKL = Path(os.environ["TEMP"]) / f"meowdoku_validation_{BATCH}_module.pkl"
with PKL.open("rb") as handle:
    loaded = pickle.load(handle)
module = loaded[0] if isinstance(loaded, tuple) else loaded
entry = next(n for n in module.functions if n.endswith("__meowdoku_step"))

consts = {}
for fname, function in module.functions.items():
    for block in function.blocks.values():
        for instr in block.instrs:
            if str(instr.op) == "Const" and getattr(instr, "res", None) is not None:
                attrs = dict(instr.attributes or {})
                consts[int(instr.res.id)] = attrs.get("value", attrs)


def describe(value):
    vid = int(value.id)
    shape = getattr(value, "shape", None)
    if vid in consts and (shape == () or shape is None or shape == (2,)):
        return f"%{vid}={consts[vid]!r}"
    return f"%{vid}{shape}"


for region in REGIONS:
    name = f"{entry}__planned_region_{region}"
    function = module.functions[name]
    print(f"--- region {region}: args={[describe(a) for a in function.args]}")
    for block in function.blocks.values():
        for instr in block.instrs:
            op = str(instr.op)
            if op == "Const":
                continue
            res = getattr(instr, "res", None)
            attrs = dict(instr.attributes or {})
            shown = {k: v for k, v in attrs.items() if k not in ("callee",)}
            shown = {k: (v if len(repr(v)) < 120 else repr(v)[:120] + "...") for k, v in shown.items()}
            head = f"%{int(res.id)}{getattr(res, 'shape', None)}:{getattr(res, 'dtype', None)}" if res is not None else "-"
            print(f"  {head} = {attrs.get('callee', op)}({', '.join(describe(a) for a in instr.args)})  {shown}")
