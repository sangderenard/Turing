"""Receiver + decoder for the tip jar's QR wall check (tests/meowdoku_qrwall_page_check.mjs drives the page, this decodes what it posts).

    python tools/meowdoku_qr_decode.py --port 8961 --out <dir> [--lib <dir>]

The page POSTs (text/plain, no-cors) ``{"name": "head_on", "expected": "<uri>", "png": "data:image/png;base64,...", "extra": "..."}`` for each captured
view.  Each PNG is saved to ``--out`` and decoded with zxing-cpp (``zxingcpp``; ZXing-class, like phone scanners) and, failing that, OpenCV's
QRCodeDetector (``cv2``).  Neither is a project dependency: install them first, e.g. ``pip install opencv-python-headless zxing-cpp`` (``--lib <dir>``
names a ``pip --target`` directory).  The decoded text must equal ``expected``; the log prints MATCH / NO MATCH / UNREADABLE per view and never prints
the text itself (the jar's address is public but is not logged).  ``GET /`` returns the JSON list of results.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

RESULTS: list[dict] = []


def decode_png(data: bytes) -> tuple[str, str]:
    """(text, decoder name); ("", "") when nothing decodes."""
    import cv2
    import numpy as np

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return "", ""
    try:
        import zxingcpp
        for found in zxingcpp.read_barcodes(img):
            if found.text:
                return found.text, "zxing-cpp"
    except ImportError:
        pass
    det = cv2.QRCodeDetector()
    text, _, _ = det.detectAndDecode(img)
    if text:
        return text, "opencv"
    big = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    text, _, _ = det.detectAndDecode(big)
    return (text, "opencv") if text else ("", "")


def make_handler(out: Path):
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, body: bytes):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.end_headers()

        def do_GET(self):
            self._reply(json.dumps(RESULTS).encode())

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            png = base64.b64decode(body["png"].split(",", 1)[1])
            name = "".join(c for c in str(body.get("name", "view")) if c.isalnum() or c in "_-")
            (out / f"{name}.png").write_bytes(png)
            text, engine = decode_png(png)
            verdict = "UNREADABLE" if not text else "MATCH" if text == body.get("expected") else "NO MATCH"
            RESULTS.append({"name": name, "verdict": verdict, "decoder": engine, "bytes": len(png), "extra": body.get("extra")})
            print(f"{name}: {verdict} [{engine or 'none'}] ({len(png)} bytes png) {body.get('extra') or ''}", flush=True)
            self._reply(json.dumps(RESULTS[-1]).encode())

        def log_message(self, *args):
            pass

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8961)
    ap.add_argument("--out", default=".")
    ap.add_argument("--lib", default=None, help="a pip --target directory holding cv2 / zxingcpp")
    ap.add_argument("--file", default=None, help="decode one PNG file instead of serving (prints the text's length only)")
    args = ap.parse_args()
    if args.lib:
        sys.path.insert(0, args.lib)
    if args.file:
        text, engine = decode_png(Path(args.file).read_bytes())
        print("decoded" if text else "UNREADABLE", len(text), engine)
        return 0 if text else 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"decoder listening on {args.port}", flush=True)
    HTTPServer(("127.0.0.1", args.port), make_handler(out)).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
