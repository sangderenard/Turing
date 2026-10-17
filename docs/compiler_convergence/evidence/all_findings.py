import glob
import json
import os
import pickle
import re
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, r"C:\dev\Powershell\turing")
from src.compiler.identity_concordance import CorrelationTable

root = glob.glob(
    r"C:\dev\Powershell\turing\build\compiler-partitions\products\provenance-core\*\growth\product\units"
)[0]
units = []
for path in sorted(glob.glob(os.path.join(root, "unit_*"))):
    pkl = os.path.join(path, "repository-ssa.pkl")
    if os.path.exists(pkl):
        units.append((int(os.path.basename(path).split("_")[1]), pkl))
print("units with a repository artifact:", len(units), flush=True)
pages = defaultdict(set)      # (kind, page-or-function, stage) -> units
counts = Counter()
t0 = time.time()
errors = 0
for index, pkl in units:
    try:
        module, _outputs, _exports = pickle.load(open(pkl, "rb"))
        findings = CorrelationTable.build(module).findings(module)
    except Exception as error:  # noqa: BLE001
        errors += 1
        print(f"  unit {index}: {type(error).__name__} {str(error)[:100]}", flush=True)
        continue
    for f in findings:
        stage = re.search(r"stage ([a-z_]+):", f.detail or "")
        fn = f.function
        if "__" in fn and f.kind not in ("unsourced-fact",):
            fn = "<function>"
        key = (f.kind, fn, stage.group(1) if stage else "")
        pages[key].add(index)
        counts[key] += 1
print(f"scanned {len(units) - errors} books in {time.time() - t0:.0f}s ({errors} unreadable)\n")
print(f"{'units':>5}  kind / page / stage")
for key, members in sorted(pages.items(), key=lambda kv: -len(kv[1])):
    print(f"{len(members):>5}  {key[0]:18} {key[1][:60]:60} {key[2]}")
