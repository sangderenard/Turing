import glob
import json
import os
import re

root = glob.glob(
    r"C:\dev\Powershell\turing\build\compiler-partitions\products\provenance-core\*\growth\product\units"
)[0]


def load(i):
    return json.load(open(os.path.join(root, f"unit_{i:03d}", "failure.json"), encoding="utf-8"))


for i in (3, 29, 25, 71):
    d = load(i)
    e = d["error"]
    print(f"== unit {i} {d['qualified_names']} deps={d['unit']['dependency_units']}")
    for m in re.finditer(r"Finding\(kind='([^']+)', function='([^']+)', value_id=([^,]+), detail=(.{0,300})", e):
        print("  ", m.group(1), "|", m.group(2)[:80], "|", m.group(4)[:260])
    print("   findings in message:", len(re.findall(r"Finding\(", e)))
d = load(31)
t = d["traceback"].splitlines()
print("== unit 31 traceback tail")
print("\n".join(line[:200] for line in t[-14:]))
