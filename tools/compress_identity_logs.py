"""Compress existing identity-book ``*.log`` files to ``*.log.xz``.

Each file is streamed through ``lzma`` into a temporary name; the temporary
is then stream-decompressed and its SHA-256 compared with the original's
hash taken during the same pass.  Only on an exact match is the temporary
renamed to ``<name>.xz`` and the original deleted.  A file modified in the
last ``--min-age-minutes`` (default 10) is skipped: a compile may still be
writing it.  The preset is the compiler's own
(``TURING_IDENTITY_LOG_LZMA_PRESET``, default 3).

    python tools/compress_identity_logs.py DIRECTORY [--min-age-minutes N]
"""

from __future__ import annotations

import argparse
import hashlib
import lzma
import os
import sys
import time

_CHUNK = 1 << 20


def _preset() -> int:
    from src.compiler.identity_concordance import identity_log_lzma_preset

    return identity_log_lzma_preset()


def compress_verified(path: str, preset: int) -> tuple[int, int] | None:
    """Compress ``path`` to ``path + '.xz'`` with a verified round trip and
    delete the original.  Returns (original bytes, compressed bytes), or None
    when the round trip did not match (nothing deleted)."""
    final = path + ".xz"
    temporary = f"{final}.tmp{os.getpid()}"
    original = hashlib.sha256()
    size = 0
    try:
        with open(path, "rb") as source, lzma.open(
            temporary, "wb", preset=preset
        ) as target:
            while True:
                chunk = source.read(_CHUNK)
                if not chunk:
                    break
                original.update(chunk)
                size += len(chunk)
                target.write(chunk)
        restored = hashlib.sha256()
        restored_size = 0
        with lzma.open(temporary, "rb") as check:
            while True:
                chunk = check.read(_CHUNK)
                if not chunk:
                    break
                restored.update(chunk)
                restored_size += len(chunk)
        if (restored.digest() != original.digest()
                or restored_size != size
                or os.path.getsize(path) != size):
            return None
        packed = os.path.getsize(temporary)
        os.replace(temporary, final)
        temporary = None
        os.remove(path)
        return size, packed
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.remove(temporary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory")
    parser.add_argument("--min-age-minutes", type=float, default=10.0)
    args = parser.parse_args(argv)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    preset = _preset()
    now = time.time()
    before = after = done = 0
    for name in sorted(os.listdir(args.directory)):
        path = os.path.join(args.directory, name)
        if not name.endswith(".log") or not os.path.isfile(path):
            continue
        if now - os.path.getmtime(path) < args.min_age_minutes * 60:
            print(f"skip (recent): {name}", flush=True)
            continue
        if os.path.exists(path + ".xz"):
            print(f"skip (.xz exists): {name}", flush=True)
            continue
        result = compress_verified(path, preset)
        if result is None:
            print(f"MISMATCH, original kept: {name}", flush=True)
            continue
        size, packed = result
        before += size
        after += packed
        done += 1
        print(f"{name}: {size} -> {packed}", flush=True)
    print(f"{done} file(s): {before} -> {after} bytes", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
