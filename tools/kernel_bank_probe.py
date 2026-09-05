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
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.compiler.kernel_bank import (
    BankRefusal, LaunchCoordinator, open_blas_bank,
)


def _contract_value(text: str) -> str | None:
    value = str(text).strip().lower()
    return None if value in {"", "none", "develop"} else value


def _ladder_row(bank, name: str, contract: str | None, size: int,
                *, compile_missing: bool) -> dict:
    spec = bank.specs[name]
    specialized = {parameter: int(size) for parameter in spec.size_parameters}
    key = bank.variant_key(
        name, contract=contract, specialized=specialized,
    )
    status = "ADMITTED"
    reason = None
    try:
        bank.get(
            name, contract=contract, specialized=specialized,
            compile_missing=compile_missing,
        )
    except BankRefusal as error:
        status, reason = "REFUSED", str(error)
    except Exception as error:  # A ladder keeps diagnostic failures as rows.
        status = "ERROR"
        reason = f"{type(error).__name__}: {error}"

    manifest_path = bank.variant_directory(name, key) / "manifest.json"
    manifest = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            status = "ERROR"
            reason = f"unreadable manifest: {error}"
    verification = manifest.get("verification") or {}
    profile = manifest.get("profile") or {}
    compute = float(profile.get("compute_avg_seconds") or 0.0)
    flops = (
        2 * int(size) ** 3
        if name == "gemm" and len(spec.size_parameters) == 3 else None
    )
    return {
        "schema": "turing.kernel-bank-ladder.v1",
        "kernel": name,
        "key": key,
        "contract": contract or "develop",
        "specialized": specialized,
        "status": status,
        "reason": reason or verification.get("reason"),
        "compiler_fingerprint": (
            manifest.get("compiler_fingerprint") or bank._fingerprint
        ),
        "source_sha256": manifest.get("source_sha256") or hashlib.sha256(
            spec.source.encode("utf-8")
        ).hexdigest(),
        "access_signature": (
            manifest.get("access_signature") or list(spec.access_signature)
        ),
        "built_unix": manifest.get("built_unix"),
        "worst_abs_error": verification.get("worst_abs_error"),
        "first_launch_seconds": profile.get("first_launch_seconds"),
        "relaunch_median_seconds": profile.get("relaunch_avg_seconds"),
        "compute_median_seconds": profile.get("compute_avg_seconds"),
        "warm_samples": profile.get("warm_samples"),
        "cold_samples": profile.get("cold_samples"),
        "gflops": (flops / compute / 1.0e9 if flops and compute > 0 else None),
        "binding": manifest.get("binding"),
        "data_layout": manifest.get("data_layout"),
    }


def _print_ladder(rows: list[dict]) -> None:
    for row in rows:
        specialized = next(iter(row["specialized"].values()), "-")
        error = row.get("worst_abs_error")
        compute = row.get("compute_median_seconds")
        first = row.get("first_launch_seconds")
        relaunch = row.get("relaunch_median_seconds")
        print(
            f"{row['status']:<8} {row['kernel']:<6} tile={specialized:<5} "
            f"contract={row['contract']:<8} "
            f"err={error if error is not None else '-':<12} "
            f"first_us={first * 1e6 if first is not None else '-':<12} "
            f"relaunch_us={relaunch * 1e6 if relaunch is not None else '-':<12} "
            f"compute_us={compute * 1e6 if compute is not None else '-':<12} "
            f"GF/s={row.get('gflops') if row.get('gflops') is not None else '-'}"
        )
        if row.get("reason"):
            print(" " * 10 + str(row["reason"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "build" / "kernel_bank")
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--specialize", action="store_true")
    parser.add_argument("--chart", action="store_true")
    parser.add_argument(
        "--ladder", type=int, nargs="+", metavar="SIZE",
        help="build/profile square specializations, retaining every refusal",
    )
    parser.add_argument("--kernel", default="gemm")
    parser.add_argument("--contract", default="fast")
    parser.add_argument(
        "--rebuild-policy", choices=("missing", "fresh", "reuse-only"),
        default="missing",
        help="compile absent rows, require an empty root, or only read rows",
    )
    parser.add_argument("--output", type=Path, help="write ladder rows as JSON")
    args = parser.parse_args()

    if args.ladder and args.rebuild_policy == "fresh" and args.root.exists():
        if any(args.root.iterdir()):
            parser.error(
                "--rebuild-policy fresh requires an empty root; "
                "choose a new --root (existing artifacts are never deleted)"
            )

    bank = open_blas_bank(args.root)

    if args.ladder:
        if args.kernel not in bank.specs:
            parser.error(
                f"unknown kernel {args.kernel!r}; choices: "
                f"{', '.join(sorted(bank.specs))}"
            )
        contract = _contract_value(args.contract)
        rows = [
            _ladder_row(
                bank, args.kernel, contract, size,
                compile_missing=args.rebuild_policy != "reuse-only",
            )
            for size in args.ladder
        ]
        _print_ladder(rows)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(rows, indent=2), encoding="utf-8",
            )
            print(f"wrote {args.output}")
        return 0

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
