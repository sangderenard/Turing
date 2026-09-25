import pickle
from pathlib import Path

path = Path(__file__).resolve().parents[2] / "build" / "full_formal_diagnostic" / "repository-ssa.pkl"
with path.open("rb") as stream:
    module = pickle.load(stream)

for name, function in module.functions.items():
    if "step_with_dt_control_used" not in name and "run_superstep" not in name:
        continue
    print("FUNCTION", name)
    ledger = dict(function.metadata.get("authored_constant_values", ()))
    print("LEDGER", len(ledger), {key: value for key, value in ledger.items() if isinstance(value, str)})
    for value_id in (57, 83, 124, 126, 144, 167, 169, 174, 188, 207, 208, 211, 212, 215, 240, 248, 252, 273, 284, 406, 419, 420, 442, 450, 468, 476, 486, 514, 515, 516, 546, 557):
        formal = next((arg for arg in function.args if int(arg.id) == value_id), None)
        if formal is not None:
            print(value_id, formal.dtype, formal.accounting)
