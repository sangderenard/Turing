import pickle
from pathlib import Path

from src.compiler.fortran_c_shell import _recover_late_source_literals
from src.compiler.ssa_self_check import check_formal_parity


root = Path(__file__).resolve().parents[2] / "build" / "full_formal_diagnostic"
with (root / "repository-ssa.pkl").open("rb") as stream:
    module = pickle.load(stream)
with (root / "resolved-process-graph.pkl").open("rb") as stream:
    process_graph = pickle.load(stream)

entries = tuple(process_graph.function_table._entries.values())
recovered = []
for symbol, function in module.functions.items():
    matches = [
        entry for entry in entries
        if f"__{entry.name}__" in symbol or symbol.endswith(f"__{entry.name}")
    ]
    if not matches or "__planned_region_" in symbol:
        continue
    entry = max(matches, key=lambda candidate: len(candidate.name))
    rows = _recover_late_source_literals(function, entry.graph.G)
    recovered.extend((symbol, value_id, value) for value_id, value in rows)

print("RECOVERED", recovered)
findings = check_formal_parity(module)
print("FINDINGS", len(findings))
for finding in findings:
    print(finding.function, finding.detail)
