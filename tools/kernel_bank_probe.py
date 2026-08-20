"""Exercise the kernel bank end to end and print what it did.

    python tools/kernel_bank_probe.py                 # launch a few calls
    python tools/kernel_bank_probe.py --matrix        # prebuild the matrix
    python tools/kernel_bank_probe.py --inventory     # list the bank
    python tools/kernel_bank_probe.py --specialize    # try size-baked builds
    python tools/kernel_bank_probe.py --chart         # performance charts

Diagnostic probe, not a test suite: reports what routed where and whether
admission verification held, in the style of tools/compile_blas_probe.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.compiler.kernel_bank import (
    BankRefusal, LaunchCoordinator, open_blas_bank,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "build" / "kernel_bank")
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--specialize", action="store_true")
    parser.add_argument("--chart", action="store_true")
    args = parser.parse_args()

    bank = open_blas_bank(args.root)

    if args.chart:
        # The auto-collected performance chart: launch vs compute averages
        # per admitted variant, the evidence the deployment strategy calls
        # read. Collected during every build (matrix or on-demand); this
        # only renders what the manifests already hold.
        for name in bank.specs:
            rows = bank.performance_chart(name)
            if not rows:
                print(f"{name}: no profiled variants (build some: --matrix)")
                continue
            print(f"{name}:")
            for row in rows:
                specialized = str(row["specialized"] or "parametric")
                first = row["first_launch_seconds"]
                relaunch = row["relaunch_avg_seconds"]
                compute = row["compute_avg_seconds"]
                print(
                    f"  contract={row['contract']:<8} {specialized:<28} "
                    f"sizes={row['sizes']} "
                    f"first_launch={first * 1e6:8.1f} us  "
                    f"relaunch={relaunch * 1e6:8.1f} us  "
                    f"compute={compute * 1e6:8.1f} us"
                )
        return 0

    if args.inventory:
        for row in bank.inventory():
            verification = row.get("verification", {})
            status = "ADMITTED" if verification.get("admitted") else "REFUSED"
            specialized = str(row.get("specialized") or "-")
            print(f"{row['kernel']:<6} {row['key']} contract={row['contract']:<8} "
                  f"specialized={specialized:<26} {status} "
                  f"err={verification.get('worst_abs_error', '-')}")
        return 0

    if args.matrix:
        for name in bank.specs:
            for contract in (None, "fast"):
                try:
                    variant = bank.get(name, contract=contract)
                    print(f"built {name:<6} contract={contract or 'develop':<8} "
                          f"key={variant.key}")
                except BankRefusal as refusal:
                    print(f"REFUSED {name} contract={contract}: {refusal}")
        return 0

    if args.specialize:
        for name, sizes in (("dot", {"n": 512}), ("gemm", {"m": 8, "n": 8, "k": 8})):
            try:
                variant = bank.get(name, specialized=sizes)
                print(f"specialized {name} {sizes}: ADMITTED key={variant.key}")
            except BankRefusal as refusal:
                print(f"specialized {name} {sizes}: REFUSED: {refusal}")
        return 0

    # Default: launch a few real calls through the coordinator -- the
    # per-call, out-of-user-hands path. One of them uses a size the bank
    # already holds a specialized build for, so the routing difference is
    # visible in the log.
    coordinator = LaunchCoordinator(bank)
    rng = np.random.default_rng(3)

    n = 512
    x, y = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
    produced = coordinator.launch("dot", x=x, y=y, n=n)
    print(f"dot   n={n}: launched -> {produced:.6f} "
          f"(numpy {float(np.dot(x, y)):.6f})")

    m, n2, k = 16, 12, 20
    A = rng.uniform(-1, 1, m * k)
    B = rng.uniform(-1, 1, k * n2)
    C = rng.uniform(-1, 1, m * n2)
    produced = coordinator.launch(
        "gemm", A=A, B=B, C=C.copy(), alpha=1.1, beta=0.4,
        m=m, n=n2, k=k,
    )
    expected = 1.1 * (A.reshape(m, k) @ B.reshape(k, n2)) + 0.4 * C.reshape(m, n2)
    worst = float(np.max(np.abs(produced.reshape(m, n2) - expected)))
    print(f"gemm  {m}x{n2}x{k}: launched, worst |err| vs numpy = {worst:.3e}")

    print(f"routing log: {coordinator.log_path}")
    for line in coordinator.log_path.read_text(encoding="utf-8").splitlines()[-4:]:
        print("  ", line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
