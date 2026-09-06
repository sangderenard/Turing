"""Build the outer native/Python/Web Turing mathematical library."""

from __future__ import annotations

import argparse
import functools
import http.server
import importlib.util
import os
from pathlib import Path
import sys
import webbrowser

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from src.compiler.kernel_bank import open_blas_bank
from src.compiler.mathematical_library_product import (
    build_mathematical_library_product,
)


def _shape(text: str) -> tuple[int, int, int]:
    values = tuple(map(int, text.lower().replace("×", "x").split("x")))
    if len(values) == 1:
        values *= 3
    if len(values) != 3 or min(values) <= 0:
        raise argparse.ArgumentTypeError("shape must be N or MxNxK")
    return values


def _verify(product) -> None:
    previous_bytecode_policy = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location(
            "built_turing_math", product.python_loader,
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        library = module.load(product.directory)
        checks = {}
        x = np.random.default_rng(113).standard_normal(23)
        y = np.random.default_rng(114).standard_normal(23)
        checks["scal"] = float(np.max(np.abs(library.blas.scal(x, 1.25) - 1.25 * x)))
        checks["axpy"] = float(np.max(np.abs(
            library.blas.axpy(x, y, 1.25) - (1.25 * x + y)
        )))
        checks["dot"] = abs(library.blas.dot(x, y) - float(x @ y))
        a_vector = np.random.default_rng(115).standard_normal((11, 23))
        checks["gemv"] = float(np.max(np.abs(
            library.blas.gemv(a_vector, x) - a_vector @ x
        )))
        rx, ry = library.blas.rot(x, y, 0.8, 0.6)
        checks["rot"] = max(
            float(np.max(np.abs(rx - (0.8 * x + 0.6 * y)))),
            float(np.max(np.abs(ry - (0.8 * y - 0.6 * x)))),
        )
        m, n, k = product.blas.manifest["shapes"][0]
        matrix_rng = np.random.default_rng(116)
        a = matrix_rng.standard_normal((m, k))
        b = matrix_rng.standard_normal((k, n))
        result = library.blas.gemm(a, b)
        checks["gemm"] = float(np.max(np.abs(result - a @ b)))
        worst = max(checks.values())
        if worst >= 1.0e-9:
            raise RuntimeError(
                f"outer Python/BLAS surface diverged: {checks!r}"
            )
        print(
            "verified math.blas methods: "
            + ", ".join(f"{name}={error:.1e}" for name, error in checks.items())
        )
        trig_input = np.asarray([0.25, 0.5, 0.75, 1.0])
        trig_forward_error = float(np.max(np.abs(
            library.trigonometry.sin(trig_input) - np.sin(trig_input)
        )))
        trig_gradient = library.trigonometry.vjp(
            "sin", np.ones_like(trig_input), value=trig_input,
        )["value"]
        trig_reverse_error = float(np.max(np.abs(
            trig_gradient - np.cos(trig_input)
        )))
        if max(trig_forward_error, trig_reverse_error) >= 1.0e-9:
            raise RuntimeError(
                "outer Python/trigonometry surface diverged: "
                f"forward={trig_forward_error!r}, reverse={trig_reverse_error!r}"
            )
        print(
            "verified math.trigonometry: "
            f"sin={trig_forward_error:.1e}, sin.vjp={trig_reverse_error:.1e}"
        )
        numpy_checks = {
            "scal": float(np.max(np.abs(library.numpy.blas.scal(x, 1.25) - 1.25 * x))),
            "axpy": float(np.max(np.abs(
                library.numpy.blas.axpy(x, y, 1.25) - (1.25 * x + y)
            ))),
            "dot": abs(library.numpy.blas.dot(x, y) - float(x @ y)),
            "gemv": float(np.max(np.abs(
                library.numpy.blas.gemv(a_vector, x) - a_vector @ x
            ))),
            "gemm": float(np.max(np.abs(library.numpy.blas.gemm(a, b) - a @ b))),
        }
        nrx, nry = library.numpy.blas.rot(x, y, 0.8, 0.6)
        numpy_checks["rot"] = max(
            float(np.max(np.abs(nrx - (0.8 * x + 0.6 * y)))),
            float(np.max(np.abs(nry - (0.8 * y - 0.6 * x)))),
        )
        if max(numpy_checks.values()) >= 1.0e-9:
            raise RuntimeError(
                f"standalone NumPy mathematical library diverged: {numpy_checks!r}"
            )
        print(
            "verified math.numpy.blas methods: "
            + ", ".join(
                f"{name}={error:.1e}" for name, error in numpy_checks.items()
            )
        )
    finally:
        if "library" in locals():
            library.close()
        sys.dont_write_bytecode = previous_bytecode_policy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gemm-shapes", nargs="+", type=_shape,
        default=((128, 128, 128), (256, 256, 256)),
    )
    parser.add_argument("--contract", default="fast")
    parser.add_argument("--cores", type=int, default=os.cpu_count())
    parser.add_argument("--bank", type=Path, default=ROOT / "build" / "kernel_bank")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "build" / "mathematical-library",
    )
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    product = build_mathematical_library_product(
        open_blas_bank(args.bank), args.gemm_shapes, args.output,
        contract=args.contract, cores=args.cores,
    )
    if not args.no_verify:
        _verify(product)
    print(f"Mathematical library: {product.directory}")
    print(f"  manifest : {product.manifest_path}")
    print(f"  Python   : {product.python_loader}")
    print(f"  NumPy    : {product.numpy_loader}")
    print(f"  Web      : {product.demo_path}")
    print(f"  product  : {product.manifest['product_id']}")
    if not args.serve:
        print("serve browser product: python -m tools.build_mathematical_library --serve")
        return 0
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(product.directory),
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://127.0.0.1:{args.port}/web/"
    print(f"Serving {url} (Ctrl+C to stop)")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
