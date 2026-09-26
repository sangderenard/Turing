from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from llvm_dt_system import lowered_system
from src.compiler.identity_concordance import concordance_report


started = time.time()
print("one-law b64 linked diagnostic: start", flush=True)
try:
    system = lowered_system(
        ["artifacts/llvm_pieces/voxel_air_step/b64/voxel_air_step.piece"],
        directory="build/tensorized_dt_one_b64_diagnostic",
    )
    print(concordance_report(system.module), flush=True)
    print(f"one-law b64 linked diagnostic: success after {time.time() - started:.2f}s", flush=True)
except BaseException:
    print(f"one-law b64 linked diagnostic: failed after {time.time() - started:.2f}s", flush=True)
    raise
