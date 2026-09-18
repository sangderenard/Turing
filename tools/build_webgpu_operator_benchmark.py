"""Build and optionally serve the browser-native WebGPU operator benchmark.

Run:

    python -m tools.build_webgpu_operator_benchmark
    python -m tools.build_webgpu_operator_benchmark --serve
"""

from __future__ import annotations

import argparse
import functools
import http.server
import os
from pathlib import Path
import sys
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from src.compiler.webgpu_benchmark_bundle import write_webgpu_benchmark_bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "build" / "webgpu-operator-benchmark",
    )
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    bundle = write_webgpu_benchmark_bundle(args.output)
    print(f"WebGPU benchmark: {bundle.page_path}")
    print(
        f"Compiled {sum(k['kind'] == 'elementwise' for k in bundle.manifest['kernels'])} "
        f"AbstractTensor operations and "
        f"{sum(k['kind'] == 'gemm' for k in bundle.manifest['kernels'])} GEMM deployments"
    )
    print(f"Manifest: {bundle.manifest_path}")
    if not args.serve:
        print(
            "Serve it with: python -m tools.build_webgpu_operator_benchmark "
            "--serve"
        )
        return 0

    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler,
        directory=str(bundle.directory),
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
