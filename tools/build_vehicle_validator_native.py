"""Build the complete native vehicle validator bundle: DLL, pool, viewer.

This is the checked-in seam that previously did not exist: earlier bundles
were compiled ad hoc, so the emitted deployment work never reached a linked
executable. The driver:

1. writes every compiler-emitted C section, shaders, manifest, and the
   turing_pool sources via ``write_native_vehicle_kernels`` under the
   selected work contract (``deploy`` requests O3 + deployment=auto, which
   outlines provable iteration lanes into native pool spans);
2. compiles all C sections plus ``turing_pool.c`` into
   ``vehicle_game_kernels.dll``;
3. compiles the threaded scientific viewer against that DLL;
4. records the build receipts (contract, optimization, pool linkage,
   dispatch markers found in the emitted C) into the manifest.

Every emitted deploy keeps its serial fallback in-source, so a build with
the pool is behaviorally safe and a build without it is merely undeployed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compiler.work_contract import active_contract, set_active_contract


def _zig(arguments: list[str]) -> None:
    command = [sys.executable, "-m", "ziglang", "cc", *arguments]
    for attempt in range(2):
        completed = subprocess.run(
            command, capture_output=True, text=True, check=False,
        )
        if completed.returncode == 0:
            return
        if "sub-compilation" not in (completed.stderr or ""):
            break
    raise RuntimeError(
        f"native compile failed ({completed.returncode}):\n"
        + (completed.stderr or "")[-4000:]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "build" / "vehicle_validator_native_deploy",
    )
    parser.add_argument(
        "--contract", default="deploy",
        choices=("prove", "develop", "deploy", "fast"),
        help="work contract; deploy = O3 + deployment=auto",
    )
    parser.add_argument("--rate-hz", type=int, default=1024)
    parser.add_argument(
        "--assembly-profile", default="default-car",
        choices=("default-car", "dually-axle"),
        help="specialize the validator input graph and tire batch",
    )
    parser.add_argument(
        "--optimization", default=None,
        choices=("O0", "O1", "O2", "O3"),
        help="override the contract's -O level (pooled dispatch is a "
        "source-level property and is unaffected; O0 turns the ~25-minute "
        "monolithic-TU compile into a couple of minutes)",
    )
    parser.add_argument(
        "--skip-kernels", action="store_true",
        help="reuse already-written C sections in the output directory",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    set_active_contract(args.contract)
    contract = active_contract()
    print(f"[contract] {contract.describe()}", flush=True)

    if not args.skip_kernels:
        from src.compiler.vehicle_native_deployment import (
            write_native_vehicle_kernels,
        )
        print("[1/4] emitting compiler-owned C sections", flush=True)
        write_native_vehicle_kernels(
            output,
            outer_rate_hz=args.rate_hz,
            assembly_profile=args.assembly_profile,
        )

    viewer_source = output / "vehicle_scientific_viewer.c"
    pool_source = output / "turing_pool.c"
    sections = sorted(
        path for path in output.glob("*.c")
        if path.name not in {viewer_source.name}
    )
    if pool_source not in sections and pool_source.is_file():
        sections.append(pool_source)
    if not sections:
        raise SystemExit(f"no C sections found in {output}")
    flags = [str(flag) for flag in contract.compiler_flags] or ["-O0"]
    if args.optimization is not None:
        flags = [
            f"-{args.optimization}",
            *(flag for flag in flags if not flag.startswith("-O")),
        ]
    dll_path = output / "vehicle_game_kernels.dll"
    print(
        f"[2/4] compiling {len(sections)} sections + pool -> {dll_path.name} "
        f"({' '.join(flags)})",
        flush=True,
    )
    _zig([
        "-shared", *flags,
        "-o", str(dll_path),
        *(str(path) for path in sections),
    ])

    viewer_path = output / "vehicle_scientific_viewer.exe"
    print(f"[3/4] compiling threaded viewer -> {viewer_path.name}", flush=True)
    _zig([
        *flags,
        "-o", str(viewer_path),
        str(viewer_source),
        str(dll_path),
        "-lopengl32", "-luser32", "-lkernel32",
    ])

    emitted_tick = (output / "balloon_tire_appendage_step.c")
    tick_text = (
        emitted_tick.read_text(encoding="utf-8")
        if emitted_tick.is_file() else ""
    )
    receipts = {
        "contract": contract.name,
        "assembly_profile": args.assembly_profile,
        "compiler_flags": flags,
        "pool_linked": pool_source.is_file(),
        "dispatch_markers": {
            "deploy_span_sites": tick_text.count("turing_pool_deploy_span("),
            "effect_lock_sites": tick_text.count("turing_pool_effect_lock();"),
        },
        "dll": dll_path.name,
        "viewer": viewer_path.name,
    }
    manifest_path = output / "vehicle_native.manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["native_build"] = receipts
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8",
        )
    print(f"[4/4] receipts: {json.dumps(receipts)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
