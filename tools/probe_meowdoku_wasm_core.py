"""Emit the saved Meowdoku SSA through the direct Wasm core; report shortfall classes.

Run from the validation checkout with PYTHONPATH set to it (the pickle only
unpickles there). ``--backend PATH`` loads that ssa_wasm_backend.py file in
place of the checkout's own, so the canonical backend can be measured against
the same saved module without writing into the shared checkout:

    PYTHONPATH=$PWD python C:/dev/Powershell/turing/tools/probe_meowdoku_wasm_core.py 25 \
        --backend C:/dev/Powershell/turing/src/compiler/ssa_wasm_backend.py
"""
import collections
import importlib.util
import os
import pickle
import sys
import time
from pathlib import Path

args = [a for a in sys.argv[1:] if not a.startswith("--")]
BATCH = int(args[0]) if args else 25
backend_path = None
if "--backend" in sys.argv:
    backend_path = Path(sys.argv[sys.argv.index("--backend") + 1])

import src.compiler  # noqa: E402  (package context for relative imports)
if backend_path is not None:
    spec = importlib.util.spec_from_file_location(
        "src.compiler.ssa_wasm_backend", backend_path)
    backend = importlib.util.module_from_spec(spec)
    sys.modules["src.compiler.ssa_wasm_backend"] = backend
    spec.loader.exec_module(backend)
    print(f"[wasm-core] backend file: {backend_path}")
else:
    import src.compiler.ssa_wasm_backend as backend  # noqa: E402

PKL = Path(os.environ["TEMP"]) / f"meowdoku_validation_{BATCH}_module.pkl"
with PKL.open("rb") as handle:
    loaded = pickle.load(handle)
module = loaded[0] if isinstance(loaded, tuple) else loaded
entry = next(n for n in module.functions if n.endswith("__meowdoku_step"))

started = time.time()
artifact = backend.emit_ssa_module_to_wasm_core(module, entry)
print(f"[wasm-core] complete={artifact.complete} shortfalls={len(artifact.shortfalls)} "
      f"in {time.time() - started:.1f}s; parameters={len(artifact.parameters)} "
      f"outputs={len(artifact.output_order)} binary={len(artifact.binary)} bytes", flush=True)
classes = collections.defaultdict(list)
for item in artifact.shortfalls:
    classes[item.operation].append(item.reason)
for operation, reasons in sorted(classes.items(), key=lambda kv: -len(kv[1])):
    print(f"[wasm-core] {operation:14} x{len(reasons)}")
    seen = collections.Counter(r.split("%t")[0][:110] for r in reasons)
    for reason, count in seen.most_common(6):
        print(f"[wasm-core]     x{count:<3} {reason}")
