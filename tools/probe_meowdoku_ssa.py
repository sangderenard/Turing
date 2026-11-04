"""Repro: lower examples/meowdoku/program.py:meowdoku_step to SSA and the direct wasm core.

Report only; edits nothing.  ``python tools/probe_meowdoku_ssa.py [batch]`` from the
repo root under the system Python (turing/.venv lacks llvmlite and yaml).
"""
import faulthandler
import inspect
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(r"C:\dev\Powershell\turing")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))
faulthandler.dump_traceback_later(540, exit=False)

from src.common.tensors import AbstractTensor  # noqa: E402
try:
    import llvmlite  # noqa: F401
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )
    REFERENCE = c_backend_repository_ssa_reference
except ModuleNotFoundError as error:
    print(f'[probe] no tensor_ssa_reference: {error}', flush=True)
    REFERENCE = None
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa  # noqa: E402
from src.compiler.native_law_kernels import batch_contract  # noqa: E402
from src.compiler.ssa_wasm_backend import (  # noqa: E402
    emit_ssa_function_to_wasm, emit_ssa_module_to_wasm_core,
)
from meowdoku import program  # noqa: E402

BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 25
ENTRY = "meowdoku_step"
source = inspect.getsource(program.meowdoku_step)
names = program.MeowdokuPiece.argument_names

print(f"[probe] batch={BATCH} args={names}", flush=True)
started = time.perf_counter()
try:
    lowered = lower_ast_source_to_ssa(
        source, ENTRY,
        python_bindings={"AbstractTensor": AbstractTensor,
                         "PLACE_CAT": program.PLACE_CAT, "TOGGLE_X": program.TOGGLE_X},
        **({'tensor_ssa_reference': REFERENCE()} if REFERENCE else {}),
        name=f"{ENTRY}_batched", runtime_closure_only=True,
        extraction_contract=batch_contract(ENTRY, names, BATCH),
        progress_bars=False,
    )
except Exception:
    print(f"[probe] SSA LOWERING FAILED after {time.perf_counter() - started:.1f}s", flush=True)
    traceback.print_exc()
    sys.exit(1)
print(f"[probe] lowered in {time.perf_counter() - started:.1f}s", flush=True)
module = lowered[0] if isinstance(lowered, tuple) else lowered.module
print("[probe] functions:", list(module.functions)[:40], flush=True)
entry = next((n for n in module.functions if n.endswith(f"__{ENTRY}")), None) or ENTRY
function = module.functions[entry]
print(f"[probe] entry={entry} args={len(function.args)} blocks={list(function.blocks)}", flush=True)
print("[probe] named_outputs:", function.metadata.get("named_outputs"), flush=True)
count = sum(len(b.instrs) for b in function.blocks.values())
print(f"[probe] instructions in entry: {count}", flush=True)
calls = sorted({str(i.attributes.get('callee') or i.attributes.get('tensor_operation') or i.op)
                for b in function.blocks.values() for i in b.instrs
                if str(i.op) in ("Call", "call")})
print("[probe] calls:", calls, flush=True)

for label, emit in (("wasm core", lambda: emit_ssa_module_to_wasm_core(module, entry, entry_name=entry)),
                    ("wasm function", lambda: emit_ssa_function_to_wasm(module, entry))):
    try:
        artifact = emit()
        print(f"[probe] {label}: complete={artifact.complete}", flush=True)
        for s in artifact.shortfalls:
            print(f"[probe]   shortfall {s}", flush=True)
    except Exception:
        print(f"[probe] {label}: RAISED", flush=True)
        traceback.print_exc()
