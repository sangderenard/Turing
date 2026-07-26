"""OpenGL context acquisition for the GLSL backend.

Policy: **borrow before you build.** A tensor backend has no business owning a
window. It should run on the context the host already has, and only create one
as a last resort for headless and test use.

That is the same conclusion two sibling projects reached independently:

* **pluck** (``spectral-analyzer/camera_software/ui_context.py``) implements
  ``OpenGLContextHost`` with a PARENT -> CURRENT -> OWNED preference order, where
  the owned fallback is a *separate hidden SDL window* so it coexists with a
  parent display instead of replacing it.
* **nodus** does not create contexts at all. Its ABI is
  ``gp_mem_backend_register_gl_context(ctx_id, native_ctx_ptr)`` — the frontend
  owns a context and *registers* it, and `GP_MEM_BACKEND_OPENGL` buffers are then
  addressed against that registration. So "use nodus's context system" means
  *be registerable*, not *ask nodus for a context*.

This module therefore does not invent a third policy. It defines a small provider
protocol, adapts pluck's host when it is importable, and keeps a minimal owned
fallback for standalone turing.

Acquisition order
-----------------
1. ``TURING_GL_CONTEXT`` env var naming a registered provider (explicit wins).
2. A context the host already made current — never replaced.
3. Registered providers, by descending priority.
4. pluck's ``OpenGLContextHost``, if importable.
5. An owned hidden SDL window (turing's own minimal fallback).

Set ``TURING_GL_NO_OWNED_CONTEXT=1`` to forbid steps 4-5, so a host that must
supply its own context gets a hard error instead of a surprise window.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

__all__ = [
    "GLContextUnavailable",
    "ContextSource",
    "GLContextLease",
    "register_context_provider",
    "unregister_context_provider",
    "registered_providers",
    "require_gl_context",
    "gl_context_info",
    "gl_context_lease",
    "release_gl_context",
    "nodus_registration_handle",
]

MIN_VERSION = (4, 3)  # compute shaders


class GLContextUnavailable(RuntimeError):
    """No usable OpenGL context. Never downgraded to a warning or a CPU fallback.

    Silent degradation is the failure mode this ecosystem has already been burned
    by twice (research/06_memory_substrate.md, research/12_substrate_blocker.md):
    a lookup miss returned 0/null and a broken substrate looked like a broken
    algorithm. A missing GPU must look like a missing GPU.
    """


class ContextSource(str, Enum):
    HOST_CURRENT = "host_current"
    PROVIDER = "provider"
    PLUCK = "pluck"
    OWNED_SDL = "owned_sdl"
    UNAVAILABLE = "unavailable"


@dataclass
class GLContextLease:
    """An acquired context plus how we got it and how (if at all) to give it back."""

    source: ContextSource = ContextSource.UNAVAILABLE
    info: dict[str, Any] = field(default_factory=dict)
    context: Any = None
    provider_name: str | None = None
    _release: Callable[[], None] | None = None
    _closed: bool = False

    @property
    def available(self) -> bool:
        return self.source is not ContextSource.UNAVAILABLE

    @property
    def owned(self) -> bool:
        """True when we created it, and are therefore responsible for closing it."""
        return self._release is not None

    def activate(self) -> None:
        make_current = getattr(self.context, "make_current", None)
        if callable(make_current):
            make_current()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._release is not None:
            try:
                self._release()
            except Exception:
                pass  # teardown races during interpreter shutdown are not errors

    def __enter__(self) -> "GLContextLease":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# provider registry -- how a host injects its own context
# ---------------------------------------------------------------------------

_providers: dict[str, tuple[int, Callable[[], Any]]] = {}


def register_context_provider(
    name: str, factory: Callable[[], Any], priority: int = 100
) -> None:
    """Register a context factory. Highest priority is tried first.

    ``factory()`` returns either a context object, or ``(context, release)``. It
    must leave the context **current** when it returns. This is the seam a nodus
    frontend or a pluck session uses to hand turing the context it already owns,
    without turing importing either repo.
    """
    _providers[name] = (int(priority), factory)


def unregister_context_provider(name: str) -> None:
    _providers.pop(name, None)


def registered_providers() -> list[str]:
    return [n for n, _ in sorted(_providers.items(), key=lambda kv: -kv[1][0])]


# ---------------------------------------------------------------------------
# probing
# ---------------------------------------------------------------------------

def _probe_current(min_version: tuple[int, int] = MIN_VERSION) -> dict[str, Any] | None:
    """Describe the current context, or None if absent/too old."""
    try:
        from OpenGL import GL
        major = int(GL.glGetIntegerv(GL.GL_MAJOR_VERSION))
        minor = int(GL.glGetIntegerv(GL.GL_MINOR_VERSION))
    except Exception:
        return None
    if (major, minor) < min_version:
        return None

    def _s(name: Any) -> str:
        try:
            raw = GL.glGetString(name)
            return raw.decode("utf-8", "replace") if raw else "?"
        except Exception:
            return "?"

    return {
        "major": major,
        "minor": minor,
        "vendor": _s(GL.GL_VENDOR),
        "renderer": _s(GL.GL_RENDERER),
        "version": _s(GL.GL_VERSION),
        "glsl": _s(GL.GL_SHADING_LANGUAGE_VERSION),
    }


# ---------------------------------------------------------------------------
# owned fallback: a separate hidden SDL window
#
# Deliberately NOT pygame.display.set_mode(): that replaces the process's single
# pygame display, so calling it from inside a host application would tear down
# the host's window. A private SDL window coexists. This mirrors pluck's
# _SDLOpenGLContext; when pluck is importable we use theirs instead of this.
# ---------------------------------------------------------------------------

class _OwnedSDLContext:
    SDL_WINDOWPOS_UNDEFINED = 0x1FFF0000
    SDL_WINDOW_OPENGL = 0x00000002
    SDL_WINDOW_HIDDEN = 0x00000008
    SDL_GL_CONTEXT_MAJOR_VERSION = 17
    SDL_GL_CONTEXT_MINOR_VERSION = 18

    def __init__(self, min_version: tuple[int, int] = MIN_VERSION) -> None:
        self._sdl = ctypes.CDLL(self._find_sdl2())
        self._configure_symbols()
        self._sdl.SDL_InitSubSystem(0x00000020)  # SDL_INIT_VIDEO
        major, minor = min_version
        self._sdl.SDL_GL_SetAttribute(self.SDL_GL_CONTEXT_MAJOR_VERSION, major)
        self._sdl.SDL_GL_SetAttribute(self.SDL_GL_CONTEXT_MINOR_VERSION, minor)
        self._window = self._sdl.SDL_CreateWindow(
            b"turing-glsl-backend",
            self.SDL_WINDOWPOS_UNDEFINED, self.SDL_WINDOWPOS_UNDEFINED,
            16, 16,
            self.SDL_WINDOW_OPENGL | self.SDL_WINDOW_HIDDEN,
        )
        if not self._window:
            raise GLContextUnavailable(self._error("SDL_CreateWindow"))
        self._context = self._sdl.SDL_GL_CreateContext(self._window)
        if not self._context:
            self._sdl.SDL_DestroyWindow(self._window)
            self._window = None
            raise GLContextUnavailable(self._error("SDL_GL_CreateContext"))
        self.make_current()

    @staticmethod
    def _find_sdl2() -> str:
        candidates: list[str | None] = []
        try:
            import pygame  # pygame ships an SDL2 DLL we can borrow
            base = os.path.dirname(pygame.__file__)
            candidates += [
                os.path.join(base, "SDL2.dll"),
                os.path.join(base, "libSDL2-2.0.so.0"),
                os.path.join(base, "libSDL2.dylib"),
            ]
        except Exception:
            pass
        candidates += [ctypes.util.find_library("SDL2-2.0"),
                       ctypes.util.find_library("SDL2")]
        for candidate in candidates:
            if candidate and (os.path.isfile(candidate)
                              or candidate == os.path.basename(candidate)):
                return candidate
        raise GLContextUnavailable(
            "SDL2 was not found, so no owned OpenGL context could be created. "
            "Install pygame (which bundles SDL2), or register a context provider."
        )

    def _configure_symbols(self) -> None:
        s = self._sdl
        s.SDL_GetError.restype = ctypes.c_char_p
        s.SDL_InitSubSystem.argtypes = [ctypes.c_uint32]
        s.SDL_InitSubSystem.restype = ctypes.c_int
        s.SDL_GL_SetAttribute.argtypes = [ctypes.c_int, ctypes.c_int]
        s.SDL_GL_SetAttribute.restype = ctypes.c_int
        s.SDL_CreateWindow.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
                                       ctypes.c_int, ctypes.c_int, ctypes.c_uint32]
        s.SDL_CreateWindow.restype = ctypes.c_void_p
        s.SDL_GL_CreateContext.argtypes = [ctypes.c_void_p]
        s.SDL_GL_CreateContext.restype = ctypes.c_void_p
        s.SDL_GL_MakeCurrent.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        s.SDL_GL_MakeCurrent.restype = ctypes.c_int
        s.SDL_GL_DeleteContext.argtypes = [ctypes.c_void_p]
        s.SDL_DestroyWindow.argtypes = [ctypes.c_void_p]

    def _error(self, operation: str) -> str:
        raw = self._sdl.SDL_GetError()
        detail = raw.decode("utf-8", "replace") if raw else "unknown SDL error"
        return f"{operation} failed: {detail}"

    @property
    def native_handle(self) -> int:
        """The native GL context pointer -- what nodus wants registered."""
        return int(self._context) if self._context else 0

    def make_current(self) -> None:
        if not self._window or not self._context:
            raise GLContextUnavailable("owned OpenGL context is closed")
        if self._sdl.SDL_GL_MakeCurrent(self._window, self._context) != 0:
            raise GLContextUnavailable(self._error("SDL_GL_MakeCurrent"))

    def close(self) -> None:
        if getattr(self, "_context", None):
            self._sdl.SDL_GL_DeleteContext(self._context)
            self._context = None
        if getattr(self, "_window", None):
            self._sdl.SDL_DestroyWindow(self._window)
            self._window = None


# ---------------------------------------------------------------------------
# pluck adapter
# ---------------------------------------------------------------------------

def _try_pluck(min_version: tuple[int, int]) -> tuple[Any, Callable[[], None]] | None:
    """Use pluck's OpenGLContextHost when we are running inside pluck.

    Import is late and guarded on purpose: turing must not gain a dependency on
    spectral-analyzer. pluck already depends on turing (dt_system), so a static
    import here would close a cycle. This only succeeds when pluck is already on
    ``sys.path``, i.e. when turing is running inside a pluck process.
    """
    try:
        from camera_software.ui_context import OpenGLContextHost  # type: ignore
        from camera_software.control_layout import OpenGLContextRequest  # type: ignore
    except Exception:
        return None
    try:
        host = OpenGLContextHost()
        request = OpenGLContextRequest(
            prefer_parent=True, create_if_missing=True, minimum_version=min_version
        )
        lease = host.acquire(request)
        if not getattr(lease, "available", False):
            return None
        lease.activate()
        return lease.context, lease.close
    except Exception:
        return None


# ---------------------------------------------------------------------------
# acquisition
# ---------------------------------------------------------------------------

_lease: GLContextLease | None = None


def _no_owned() -> bool:
    return os.environ.get("TURING_GL_NO_OWNED_CONTEXT", "").strip().lower() in (
        "1", "true", "yes"
    )


def _finish(source: ContextSource, context: Any,
            release: Callable[[], None] | None,
            provider_name: str | None,
            min_version: tuple[int, int]) -> GLContextLease:
    info = _probe_current(min_version)
    if info is None:
        if release is not None:
            try:
                release()
            except Exception:
                pass
        raise GLContextUnavailable(
            f"{source.value} produced a context, but it does not report OpenGL "
            f"{min_version[0]}.{min_version[1]}+, the minimum for compute shaders. "
            f"Check the driver, and that a hardware renderer (not a software "
            f"fallback such as llvmpipe) is selected."
        )
    info["source"] = source.value
    if provider_name:
        info["provider"] = provider_name
    return GLContextLease(source=source, info=info, context=context,
                          provider_name=provider_name, _release=release)


def require_gl_context(min_version: tuple[int, int] = MIN_VERSION) -> dict[str, Any]:
    """Ensure a compute-capable context is current, or raise. Idempotent."""
    global _lease
    if _lease is not None and _lease.available:
        return _lease.info

    try:
        import OpenGL  # noqa: F401
    except Exception as exc:
        raise GLContextUnavailable(
            "PyOpenGL is required for the GLSL backend. This backend does not "
            "fall back to CPU: install PyOpenGL or select another backend."
        ) from exc

    tried: list[str] = []

    # 1. explicit provider by name
    forced = os.environ.get("TURING_GL_CONTEXT", "").strip()
    if forced:
        entry = _providers.get(forced)
        if entry is None:
            raise GLContextUnavailable(
                f"TURING_GL_CONTEXT={forced!r} but no such provider is registered. "
                f"Registered: {registered_providers() or '(none)'}"
            )
        created = entry[1]()
        context, release = created if isinstance(created, tuple) else (created, None)
        _lease = _finish(ContextSource.PROVIDER, context, release, forced, min_version)
        return _lease.info

    # 2. a context the host already made current -- never replaced
    info = _probe_current(min_version)
    if info is not None:
        info["source"] = ContextSource.HOST_CURRENT.value
        _lease = GLContextLease(ContextSource.HOST_CURRENT, info, None)
        return _lease.info
    tried.append("host-current")

    # 3. registered providers, highest priority first
    for name, (_prio, factory) in sorted(_providers.items(), key=lambda kv: -kv[1][0]):
        try:
            created = factory()
        except Exception:
            tried.append(f"provider:{name}")
            continue
        context, release = created if isinstance(created, tuple) else (created, None)
        try:
            _lease = _finish(ContextSource.PROVIDER, context, release, name, min_version)
            return _lease.info
        except GLContextUnavailable:
            tried.append(f"provider:{name}")

    if _no_owned():
        raise GLContextUnavailable(
            "No host context and TURING_GL_NO_OWNED_CONTEXT is set, so no context "
            f"was created. Tried: {', '.join(tried)}. The host must make a "
            "compute-capable context current, or register a provider."
        )

    # 4. pluck's context host, if we are running inside pluck
    pluck = _try_pluck(min_version)
    if pluck is not None:
        context, release = pluck
        try:
            _lease = _finish(ContextSource.PLUCK, context, release, "pluck", min_version)
            return _lease.info
        except GLContextUnavailable:
            tried.append("pluck")
    else:
        tried.append("pluck")

    # 5. our own hidden SDL window
    try:
        owned = _OwnedSDLContext(min_version)
    except GLContextUnavailable:
        raise
    except Exception as exc:
        raise GLContextUnavailable(
            f"Could not create an OpenGL context. Tried: {', '.join(tried)}, "
            f"owned-sdl ({exc}). The GLSL backend requires a real compute-capable "
            f"context and will not run on CPU."
        ) from exc
    _lease = _finish(ContextSource.OWNED_SDL, owned, owned.close, None, min_version)
    return _lease.info


def gl_context_info() -> dict[str, Any] | None:
    return _lease.info if _lease is not None and _lease.available else None


def gl_context_lease() -> GLContextLease | None:
    return _lease


def release_gl_context() -> None:
    """Close an owned context. A borrowed host context is left untouched."""
    global _lease
    if _lease is not None:
        _lease.close()
    _lease = None


def nodus_registration_handle() -> int:
    """Native GL context pointer for ``gp_mem_backend_register_gl_context``.

    nodus does not create contexts; it registers one the frontend owns, then
    addresses ``GP_MEM_BACKEND_OPENGL`` buffers against that registration. This
    returns the value to hand it, or 0 when the active context is one we borrowed
    and whose native pointer we therefore do not hold.
    """
    if _lease is None or not _lease.available:
        return 0
    handle = getattr(_lease.context, "native_handle", 0)
    return int(handle) if handle else 0
