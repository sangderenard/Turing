"""Bounded retention for regenerable compiler build products.

This module only removes directories whose names prove that they belong to a
known temporary/versioned compiler namespace.  Native piece archives, books,
manifests outside those directories, and arbitrary user-named build products
are never candidates.
"""

from __future__ import annotations

from pathlib import Path
from contextlib import contextmanager
import os
import re
import shutil
from typing import Callable, Iterable


_FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")
_ORBITAL_PROFILE_PRODUCT = re.compile(
    r"profile_rules_[1-9][0-9]*_(?:c|llvm)_[A-Za-z0-9_.-]+\Z"
)
_PRODUCT_LEASE = ".turing-product.lock"
_TEMPORARY_FILE = re.compile(
    r"\.tmp(?:[0-9]+|\.[0-9]+(?:\.[0-9]+)?)?\Z"
)
_TEMPORARY_DIRECTORY = re.compile(r"temporary-(?:tiles|cache|spill)\Z")


def _lock(stream, *, blocking: bool) -> bool:
    """Acquire one byte of the product lease without a side-channel PID."""

    stream.seek(0)
    if os.name == "nt":
        import msvcrt

        mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
        try:
            msvcrt.locking(stream.fileno(), mode, 1)
        except OSError:
            return False
        return True
    import fcntl

    flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
    try:
        fcntl.flock(stream.fileno(), flags)
    except OSError:
        return False
    return True


def _unlock(stream) -> None:
    stream.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@contextmanager
def compiler_product_lease(product: str | Path):
    """Hold the product's cross-process lease for its complete mutation."""

    root = Path(product).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lease = root / _PRODUCT_LEASE
    with lease.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        if not _lock(stream, blocking=False):
            raise RuntimeError(f"compiler product is active in another process: {root}")
        try:
            yield root
        finally:
            _unlock(stream)


def _product_is_leased(product: Path) -> bool:
    lease = product / _PRODUCT_LEASE
    if not lease.is_file():
        return False
    try:
        with lease.open("r+b") as stream:
            if not _lock(stream, blocking=False):
                return True
            _unlock(stream)
    except OSError:
        # An unreadable lease is not permission to delete its product.
        return True
    return False


def _prune_directories(
    parent: str | Path,
    *,
    accepts: Callable[[str], bool],
    protected: Iterable[str] = (),
    keep_newest: int = 0,
) -> tuple[Path, ...]:
    """Remove accepted direct children, never links or arbitrary paths."""

    root = Path(parent).resolve()
    if not root.is_dir():
        return ()
    protected_names = frozenset(map(str, protected))
    candidates = [
        child for child in root.iterdir()
        if child.is_dir()
        and not child.is_symlink()
        and child.name not in protected_names
        and accepts(child.name)
        and child.parent.resolve() == root
        and not _product_is_leased(child)
    ]
    candidates.sort(
        key=lambda path: (path.stat().st_mtime_ns, path.name), reverse=True,
    )
    removed = []
    for child in candidates[max(0, int(keep_newest)):]:
        try:
            shutil.rmtree(child)
        except OSError:
            # A loaded Windows DLL or a genuinely concurrent reader may hold
            # a stale product open. Retention must not become a new compiler
            # blocker; the still-present directory remains a candidate on the
            # next invocation.
            continue
        removed.append(child)
    return tuple(removed)


def prune_stale_fingerprint_products(
    parent: str | Path,
    active_fingerprint: str,
) -> tuple[Path, ...]:
    """Keep the active 64-hex product and remove older fingerprint products."""

    active = str(active_fingerprint)
    if _FINGERPRINT.fullmatch(active) is None:
        raise ValueError("active compiler product fingerprint must be 64 lowercase hex digits")
    return _prune_directories(
        parent,
        accepts=lambda name: _FINGERPRINT.fullmatch(name) is not None,
        protected=(active,),
    )


def prune_stale_orbital_profile_products(
    parent: str | Path,
    active_product: str | Path,
) -> tuple[Path, ...]:
    """Keep this orbital profile product and remove older profile products.

    The orbital driver uses :func:`tempfile.mkdtemp` under ``build`` and used
    to retain every compiler book, emitted module and native intermediate
    forever.  Admission remains deliberately narrower than a prefix check:
    only the driver's complete ``profile_rules_<count>_<backend>_<nonce>``
    namespace is removable, and the resolved active direct child is always
    protected.
    """

    root = Path(parent).resolve()
    active = Path(active_product).resolve()
    if active.parent != root:
        raise ValueError("active orbital profile product must be a direct child")
    if _ORBITAL_PROFILE_PRODUCT.fullmatch(active.name) is None:
        raise ValueError("active orbital profile product has an unknown name")
    return _prune_directories(
        root,
        accepts=lambda name: _ORBITAL_PROFILE_PRODUCT.fullmatch(name) is not None,
        protected=(active.name,),
    )


def prune_abandoned_product_temporaries(
    product: str | Path,
) -> tuple[Path, ...]:
    """Remove only incomplete atomic-write/tile artifacts in a leased product.

    Resumable graphs, scoped inputs, failure receipts, emitted sources and
    completed unit products are not caches and are deliberately untouched.
    The caller must hold :func:`compiler_product_lease`; this function proves
    that condition rather than racing an active compiler.
    """

    root = Path(product).resolve()
    if not root.is_dir() or not _product_is_leased(root):
        raise RuntimeError("temporary cleanup requires the active product lease")
    removed = []
    candidates = sorted(
        (
            path for path in root.rglob("*")
            if path.name != _PRODUCT_LEASE
            and (
                path.is_file() and _TEMPORARY_FILE.search(path.name)
                or path.is_dir() and not path.is_symlink()
                and _TEMPORARY_DIRECTORY.fullmatch(path.name) is not None
            )
        ),
        key=lambda path: len(path.parts), reverse=True,
    )
    for path in candidates:
        resolved = path.resolve()
        if root not in resolved.parents:
            continue
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        except FileNotFoundError:
            continue
        removed.append(path)
    return tuple(removed)


__all__ = [
    "compiler_product_lease",
    "prune_abandoned_product_temporaries",
    "prune_stale_fingerprint_products",
    "prune_stale_orbital_profile_products",
]
