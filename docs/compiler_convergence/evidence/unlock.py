import json
import re
from collections import defaultdict

U = {int(k): v for k, v in json.load(open(
    r"C:\Users\alber\AppData\Local\Temp\claude\C--dev-Powershell\018de881-9b47-46f5-8217-2f88ebe7ad11\scratchpad\census_units.json")).items()}
bad = {i for i, u in U.items() if u["status"] in ("failed", "partial")}
good = {i for i in U if i not in bad}
own = {i for i in bad if not any(d in bad for d in U[i]["deps"])}
reverse = defaultdict(set)
for i, u in U.items():
    for d in u["deps"]:
        reverse[d].add(i)


def dependents(root):
    seen, stack = set(), [root]
    while stack:
        for nxt in reverse[stack.pop()]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def label(u):
    if u["status"] == "failed":
        text = re.sub(r"\b\d{4,}\b", "N", u["error"])
        return text[:95]
    kinds = sorted({re.sub(r":.*", "", s) for s in u["shortfalls"]})
    return "PARTIAL " + ",".join(kinds)[:90]


rows = []
for i in sorted(own):
    deps = dependents(i)
    blocked = deps & bad
    rows.append((len(blocked), i, U[i]["names"][:1], label(U[i])))
rows.sort(reverse=True)
print("own-cause units:", len(own), "| bad units ever blocked by one:", len(set().union(*[dependents(i) & bad for i in own])))
print(f"{'unlocks':>7}  unit  name / cause")
for n, i, names, lab in rows[:22]:
    print(f"{n:>7}  {i:>4}  {names}\n{'':>15}{lab}")
print()
print("units unlocked by the top 5 / 10 / 20 own-cause fixes (union):")
for k in (5, 10, 20, len(rows)):
    union = set()
    for _n, i, _nm, _l in rows[:k]:
        union |= dependents(i) & bad
        union.add(i)
    print(f"  top {k:>3}: {len(union)} of {len(bad)} bad units reachable (they would still need their own subsequent layer)")
