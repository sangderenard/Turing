"""Build the durable native/Python/Web Turing BLAS server product.

Examples:

    python -m tools.build_blas_server
    python -m tools.build_blas_server --shapes 128 256 384x256x128 --serve
"""

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

from src.compiler.blas_server import build_blas_server
from src.compiler.kernel_bank import open_blas_bank


def _shape(text: str) -> tuple[int, int, int]:
    fields = text.lower().replace("×", "x").split("x")
    values = tuple(map(int, fields))
    if len(values) == 1:
        values *= 3
    if len(values) != 3 or min(values) <= 0:
        raise argparse.ArgumentTypeError("shape must be N or MxNxK")
    return values


def _verify(product) -> None:
    spec = importlib.util.spec_from_file_location(
        "built_turing_blas_server", product.python_loader,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    server = module.load(product.directory)
    try:
        shape = tuple(product.manifest["shapes"][0])
        m, n, k = shape
        rng = np.random.default_rng(73)
        a = rng.standard_normal((m, k))
        b = rng.standard_normal((k, n))
        c = rng.standard_normal((m, n))
        alpha, beta = 1.25, 0.5
        result = server.gemm(a, b, c=c, alpha=alpha, beta=beta)
        error = float(np.max(np.abs(result - (alpha * (a @ b) + beta * c))))
        if error >= 1.0e-9:
            raise RuntimeError(f"built Python/native surface diverged: {error:.3e}")
        print(f"verified Python -> DLL {shape}: worst |error| {error:.3e}")
    finally:
        server.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shapes", nargs="+", type=_shape,
        default=((128, 128, 128), (256, 256, 256)),
    )
    parser.add_argument("--contract", default="fast")
    parser.add_argument("--cores", type=int, default=os.cpu_count())
    parser.add_argument(
        "--bank", type=Path, default=ROOT / "build" / "kernel_bank",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "build" / "blas-server",
    )
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    product = build_blas_server(
        open_blas_bank(args.bank), args.shapes, args.output,
        contract=args.contract, cores=args.cores,
    )
    if not args.no_verify:
        _verify(product)
    print(f"BLAS server: {product.directory}")
    print(f"  manifest : {product.manifest_path}")
    print(f"  native   : {product.native_library}")
    print(f"  Python   : {product.python_loader}")
    print(f"  Web      : {product.demo_path}")
    print(f"  product  : {product.manifest['product_id']}")
    if not args.serve:
        print("serve WebGPU demo: python -m tools.build_blas_server --serve")
        return 0

    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler,
        directory=str(product.directory / "web"),
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://127.0.0.1:{args.port}/"
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
