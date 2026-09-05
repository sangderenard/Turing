"""Dynamic authority for recursively realizing authored tensor source.

The flag lives below the compiler so semantic algorithms and generated pack
installers can honor the same rule without importing compilation machinery.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import update_wrapper
from typing import Any, Callable


_AUTHORED_SOURCE_DEPTH: ContextVar[int] = ContextVar(
    "turing_authored_source_depth", default=0,
)
_AUTHORED_SOURCE_TARGETS: ContextVar[frozenset[str]] = ContextVar(
    "turing_authored_source_targets", default=frozenset(),
)


@contextmanager
def authored_source_realization(*, targets=()):
    """Make every nested standard-object call reveal authored source."""

    token = _AUTHORED_SOURCE_DEPTH.set(_AUTHORED_SOURCE_DEPTH.get() + 1)
    target_token = _AUTHORED_SOURCE_TARGETS.set(
        _AUTHORED_SOURCE_TARGETS.get() | frozenset(map(str, targets))
    )
    try:
        yield
    finally:
        _AUTHORED_SOURCE_TARGETS.reset(target_token)
        _AUTHORED_SOURCE_DEPTH.reset(token)


def realizing_authored_source() -> bool:
    """Whether the current call stack is recursively realizing source."""

    return _AUTHORED_SOURCE_DEPTH.get() > 0


def realizing_authored_target(identity: str) -> bool:
    """Whether one target-scoped deployment must expose its source now."""

    return str(identity) in _AUTHORED_SOURCE_TARGETS.get()


def deployed_with_authored_fallback(
    authored_callable: Callable[..., Any],
    deployed_callable: Callable[..., Any],
    *,
    identity: str | None = None,
    targeted: bool = False,
) -> Callable[..., Any]:
    """Install native behavior while retaining recursive source authority."""

    authored = getattr(
        authored_callable,
        "__turing_authored_source_callable__",
        authored_callable,
    )

    def installed(*args: Any, **kwargs: Any) -> Any:
        reveal_source = (
            realizing_authored_target(str(identity))
            if targeted and identity is not None
            else realizing_authored_source()
        )
        target = authored if reveal_source else deployed_callable
        return target(*args, **kwargs)

    update_wrapper(installed, authored)
    installed.__turing_authored_source_callable__ = authored
    installed.__turing_deployed_callable__ = deployed_callable
    return installed


def install_authored_deployment(
    owner: Any,
    name: str,
    deployed_callable: Callable[..., Any],
    *,
    identity: str | None = None,
    targeted: bool = False,
) -> Callable[..., Any]:
    """Replace one module/class attribute with a source-revealing deployment."""

    authored = getattr(owner, str(name))
    installed = deployed_with_authored_fallback(
        authored, deployed_callable,
        identity=identity,
        targeted=targeted,
    )
    setattr(owner, str(name), installed)
    return installed


__all__ = [
    "authored_source_realization",
    "deployed_with_authored_fallback",
    "install_authored_deployment",
    "realizing_authored_source",
    "realizing_authored_target",
]
