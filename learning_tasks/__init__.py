from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace


def get_task(name: str) -> SimpleNamespace:
    """Return helpers for the named learning task.

    The returned namespace exposes ``pump_queue``, ``build_loss_composer`` and
    ``num_logits``.  Tasks are loaded lazily to keep optional dependencies
    isolated.
    """

    mod = import_module(f"learning_tasks.{name}_task")
    return SimpleNamespace(
        pump_queue=mod.pump_queue,
        build_loss_composer=getattr(mod, "build_loss_composer"),
        num_logits=getattr(mod, "NUM_LOGITS", 0),
    )
