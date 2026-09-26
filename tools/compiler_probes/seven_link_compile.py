from __future__ import annotations

import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from llvm_dt_system import lowered_system
from src.compiler.identity_concordance import concordance_report


LAWS = (
    "voxel_air_step",
    "voxel_species_step",
    "aerosol_step",
    "droplet_step",
    "salt_solution_step",
    "surface_step",
    "pool_step",
)
PIECES = tuple(
    ROOT / "artifacts" / "llvm_pieces" / law / "b64" / f"{law}.piece"
    for law in LAWS
)


started = time.time()
print("seven-law b64 LINK compile: start", flush=True)
print("pieces:", flush=True)
for piece in PIECES:
    print(f"  {piece}", flush=True)

try:
    system = lowered_system(
        PIECES,
        directory=ROOT / "build" / "tensorized_dt_seven_b64",
        piece_mode="link",
    )
    print(concordance_report(system.module), flush=True)
    print(
        f"seven-law b64 LINK compile: success after "
        f"{time.time() - started:.2f}s",
        flush=True,
    )
except BaseException:
    print(
        f"seven-law b64 LINK compile: failed after "
        f"{time.time() - started:.2f}s",
        flush=True,
    )
    raise
