"""Plain, permanent on-disk retention for ``DualIRShell`` objects.

This is deliberately not the checkpoint system (``aot_checkpoint.py``),
which is resumable and phase-aware, keyed off compile inputs so a later
identical call can skip work. Nothing here expires, nothing here is keyed
off inputs, and nothing currently reads it back automatically -- it exists
because ``DualIRShell`` is the one point every compile path (whatever
language, whatever frontend) is meant to converge on
(``GRAPH_DESCRIPTION_LAYER_SURVEY.md``). Writing every shell produced to one
plain, inspectable place now, before there is a concrete reason to read one
back, means a later bootstrap does not have to invent this from scratch --
it just has a directory of real shells to look at.
"""

from __future__ import annotations

import time
from pathlib import Path

from joblib.externals import cloudpickle

from .artifact_cache import repository_cache_root
from .dual_ir_shell import DualIRShell


def shell_archive_root() -> Path:
    return repository_cache_root() / "shells"


def save_shell(shell: DualIRShell, *, key: str | None = None) -> Path:
    """Write ``shell`` to the archive and return the file it wrote.

    Every write is a new file (the timestamp suffix), never an overwrite --
    retention, not a cache slot. ``key`` names the shell for a human
    scanning the directory; it does not identify it for lookup.
    """

    root = shell_archive_root()
    root.mkdir(parents=True, exist_ok=True)
    label = key or shell.name or "shell"
    path = root / f"{label}-{time.time_ns()}.pkl"
    with path.open("wb") as stream:
        cloudpickle.dump(shell, stream, protocol=5)
    return path


def load_shell(path: Path | str) -> DualIRShell:
    """Read back a shell written by ``save_shell``.

    The inverse operation, kept for symmetry -- nothing in the compile
    pipeline currently reads from this archive.
    """

    with Path(path).open("rb") as stream:
        return cloudpickle.load(stream)


__all__ = ["shell_archive_root", "save_shell", "load_shell"]
