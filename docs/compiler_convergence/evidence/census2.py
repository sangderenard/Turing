import glob
import json
import os
import re
from collections import Counter, defaultdict

root = glob.glob(
    r"C:\dev\Powershell\turing\build\compiler-partitions\products\provenance-core\*\growth\product\units"
)[0]
U = {}
for path in sorted(glob.glob(os.path.join(root, "unit_*"))):
    i = int(os.path.basename(path).split("_")[1])
    rec = {"status": None, "error": "", "names": [], "deps": [], "short": []}
    f = os.path.join(path, "failure.json")
    j = os.path.join(path, "unit.json")
    if os.path.exists(f):
        d = json.load(open(f, encoding="utf-8"))
        rec.update(status="failed", error=f"{d.get('error_type')}: {d.get('error')}",
                   names=d.get("qualified_names") or [], deps=(d.get("unit") or {}).get("dependency_units") or [])
    elif os.path.exists(j):
        d = json.load(open(j, encoding="utf-8"))
        rec.update(status=d.get("status"), names=d.get("qualified_names") or [],
                   deps=(d.get("unit") or {}).get("dependency_units") or [])
        for acc in d.get("repository_ssa_accounting") or []:
            for s in acc.get("shortfalls") or []:
                rec["short"].append(str(s.get("kind")))
    U[i] = rec
json.dump({str(k): v for k, v in U.items()}, open(
    r"C:\Users\alber\AppData\Local\Temp\claude\C--dev-Powershell\018de881-9b47-46f5-8217-2f88ebe7ad11\scratchpad\census_units.json", "w"))
bad = {i for i, u in U.items() if u["status"] in ("failed", "partial")}
own = {i for i in bad if not any(d in bad for d in U[i]["deps"])}
reverse = defaultdict(set)
for i, u in U.items():
    for d in u["deps"]:
        reverse[d].add(i)


def closure(r):
    seen, st = set(), [r]
    while st:
        for n in reverse[st.pop()]:
            if n not in seen:
                seen.add(n)
                st.append(n)
    return seen


def key(u):
    e = u["error"]
    if "lacks complete provenance" in e:
        parts = []
        for m in re.finditer(r"Finding\(kind='([^']+)', function='([^']+)'", e):
            fn = re.sub(r"^(src_|_)?.*?__.*$", "<unit function>", m.group(2)) if "__" in m.group(2) else m.group(2)
            parts.append(f"{m.group(1)}:{fn}")
        stages = re.findall(r"stage ([a-z_]+):", e)
        return "PROVENANCE " + ",".join(sorted(set(parts))[:4]) + " | stages " + ",".join(sorted(set(stages)))
    if "native type predicate has no authoritative value_type_abi" in e:
        return "native type predicate has no authoritative value_type_abi"
    if "formatted glyph component" in e:
        return "glyph component has no resident producer"
    return re.sub(r"\b\d+\b", "n", re.sub(r"'[^']{0,50}'", "'S'", e))[:130]


groups = defaultdict(list)
for i in sorted(bad):
    u = U[i]
    k = key(u) if u["status"] == "failed" else "PARTIAL " + ",".join(sorted(set(u["short"])))[:120]
    groups[k].append(i)
print("TOTAL", len(U), dict(Counter(u["status"] for u in U.values())), "| own-cause", len(own), "| inherited", len(bad) - len(own))
for k, m in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:14]:
    n_own = len([x for x in m if x in own])
    unlock = set()
    for x in m:
        if x in own:
            unlock |= closure(x) & bad
    print(f"{len(m):>4} units ({n_own:>3} own, unlocks {len(unlock):>3} more)  {k[:150]}")
    print(f"{'':>8}e.g. {m[0]} {U[m[0]]['names'][:1]}")
rows = []
for i in own:
    rows.append((len(closure(i) & bad), i, U[i]["names"][:1], key(U[i]) if U[i]["status"] == "failed" else "PARTIAL"))
rows.sort(reverse=True)
print("\ntop roots by units unlocked:")
for n, i, nm, k in rows[:12]:
    print(f"{n:>5} unit {i:>3} {nm} {k[:100]}")
