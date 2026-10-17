import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

root = glob.glob(
    r"C:\dev\Powershell\turing\build\compiler-partitions\products\provenance-core\*\growth\product\units"
)[0]
units = {}
for path in sorted(glob.glob(os.path.join(root, "unit_*"))):
    index = int(os.path.basename(path).split("_")[1])
    record = {"index": index, "status": None, "error": "", "stage": "", "names": [], "deps": [], "shortfalls": []}
    failure = os.path.join(path, "failure.json")
    unit_json = os.path.join(path, "unit.json")
    if os.path.exists(failure):
        d = json.load(open(failure, encoding="utf-8"))
        record["status"] = "failed"
        record["error"] = f"{d.get('error_type')}: {d.get('error')}"
        st = d.get("stage") or {}
        record["stage"] = (st.get("compiler") or {}).get("phase") or st.get("phase") or ""
        record["names"] = d.get("qualified_names") or []
        record["deps"] = (d.get("unit") or {}).get("dependency_units") or []
    elif os.path.exists(unit_json):
        d = json.load(open(unit_json, encoding="utf-8"))
        record["status"] = d.get("status") or "?"
        record["names"] = d.get("qualified_names") or []
        record["deps"] = (d.get("unit") or {}).get("dependency_units") or []
        for acc in d.get("repository_ssa_accounting") or []:
            for s in acc.get("shortfalls") or []:
                record["shortfalls"].append(str(s.get("kind")) + ":" + json.dumps(s.get("values"))[:120])
    units[index] = record


def signature(text):
    text = re.sub(r"0x[0-9a-fA-F]+", "0xN", text)
    text = re.sub(r"\b\d{4,}\b", "N", text)
    text = re.sub(r"\b\d+\b", "n", text)
    text = re.sub(r"'[^']{0,60}'", "'S'", text)
    return text[:170]


bad = {i for i, u in units.items() if u["status"] in ("failed", "partial")}
rows = Counter(u["status"] for u in units.values())
print("TOTAL", len(units), dict(rows))
own = {i for i in bad if not any(d in bad for d in units[i]["deps"])}
print("units with a failing/partial status:", len(bad), "| own-cause (no bad dependency):", len(own), "| inherit via dependency:", len(bad) - len(own))

sigs = defaultdict(list)
for i in sorted(bad):
    u = units[i]
    if u["status"] == "failed":
        key = ("FAILED", signature(u["error"]))
    else:
        key = ("PARTIAL", signature(";".join(sorted(set(re.sub(r':.*', '', s) for s in u["shortfalls"]))) or "no shortfall detail"))
    sigs[key].append(i)

print()
print(f"{'units':>5} {'own':>4}  kind     signature")
for (kind, sig), members in sorted(sigs.items(), key=lambda kv: -len(kv[1])):
    n_own = sum(1 for m in members if m in own)
    ex = units[members[0]]["names"][:1]
    print(f"{len(members):>5} {n_own:>4}  {kind:7}  {sig}")
    print(f"{'':>13}e.g. unit {members[0]} {ex}")
json.dump({str(k): v for k, v in units.items()}, open(r"C:\Users\alber\AppData\Local\Temp\claude\C--dev-Powershell\018de881-9b47-46f5-8217-2f88ebe7ad11\scratchpad\census_units.json", "w"))
