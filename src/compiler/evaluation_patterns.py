"""Lazy backend alternatives and measurements for logical compiled objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from time import perf_counter_ns
from typing import Any, Callable, Hashable, Iterable, Mapping

from .control_source import ControlTarget


@dataclass(frozen=True)
class EvaluationVariantKey:
    """One late code-selection choice for a logical object and input shape."""

    object_key: Hashable
    signature: Hashable
    shell_language: ControlTarget
    interior_backend: str


@dataclass
class EvaluationProfile:
    compile_ns: int | None = None
    samples_ns: list[int] = field(default_factory=list)
    status: str = "not_run"
    error: str | None = None

    @property
    def median_ns(self) -> int | None:
        return int(median(self.samples_ns)) if self.samples_ns else None


@dataclass
class LazyEvaluationVariant:
    factory: Callable[[], Callable[..., Any]]
    value: Callable[..., Any] | None = None
    profile: EvaluationProfile = field(default_factory=EvaluationProfile)

    def resolve(self) -> Callable[..., Any]:
        if self.value is None:
            started = perf_counter_ns()
            try:
                self.value = self.factory()
            except Exception as error:
                self.profile.compile_ns = perf_counter_ns() - started
                self.profile.status = "failed"
                self.profile.error = f"{type(error).__name__}: {error}"
                raise
            self.profile.compile_ns = perf_counter_ns() - started
            self.profile.status = "compiled"
        return self.value


class EvaluationPatternMap:
    """Multi-layer lazy map of available execution patterns.

    Variants are retained independently.  Selecting another language/backend
    only changes the active key; it does not discard compiled alternatives or
    their timing history.  A downstream policy network can consume
    ``profile_rows`` and call ``select`` without rebuilding this structure.
    """

    def __init__(self) -> None:
        self._objects: dict[
            Hashable,
            dict[
                Hashable,
                dict[
                    ControlTarget,
                    dict[str, LazyEvaluationVariant],
                ],
            ],
        ] = {}
        self._active: dict[tuple[Hashable, Hashable], EvaluationVariantKey] = {}

    def register(
        self,
        key: EvaluationVariantKey,
        factory: Callable[[], Callable[..., Any]],
    ) -> None:
        backends = (
            self._objects
            .setdefault(key.object_key, {})
            .setdefault(key.signature, {})
            .setdefault(key.shell_language, {})
        )
        if key.interior_backend in backends:
            raise ValueError(f"evaluation variant already registered: {key!r}")
        backends[key.interior_backend] = LazyEvaluationVariant(factory)

    def variant(self, key: EvaluationVariantKey) -> LazyEvaluationVariant:
        try:
            return self._objects[key.object_key][key.signature][
                key.shell_language
            ][key.interior_backend]
        except KeyError as error:
            raise KeyError(f"unknown evaluation variant {key!r}") from error

    def resolve(self, key: EvaluationVariantKey) -> Callable[..., Any]:
        return self.variant(key).resolve()

    def select(self, key: EvaluationVariantKey) -> Callable[..., Any]:
        value = self.resolve(key)
        self._active[(key.object_key, key.signature)] = key
        return value

    def active(
        self,
        object_key: Hashable,
        signature: Hashable,
    ) -> Callable[..., Any]:
        try:
            key = self._active[(object_key, signature)]
        except KeyError as error:
            raise KeyError(
                f"no active evaluation pattern for {(object_key, signature)!r}"
            ) from error
        return self.resolve(key)

    def profile(
        self,
        key: EvaluationVariantKey,
        args: Iterable[Any] = (),
        kwargs: Mapping[str, Any] | None = None,
        *,
        warmups: int = 1,
        repeats: int = 5,
        synchronize: Callable[[Any], None] | None = None,
    ) -> Any:
        function = self.resolve(key)
        positional = tuple(args)
        named = dict(kwargs or {})
        result = None
        for _ in range(max(0, int(warmups))):
            result = function(*positional, **named)
            if synchronize is not None:
                synchronize(result)
        for _ in range(max(1, int(repeats))):
            started = perf_counter_ns()
            result = function(*positional, **named)
            if synchronize is not None:
                synchronize(result)
            self.variant(key).profile.samples_ns.append(
                perf_counter_ns() - started
            )
        self.variant(key).profile.status = "passed"
        self.variant(key).profile.error = None
        return result

    def profile_attempt(self, key: EvaluationVariantKey, *args, **kwargs):
        """Profile one variant and retain failure details without aborting."""

        try:
            result = self.profile(key, *args, **kwargs)
        except Exception as error:
            profile = self.variant(key).profile
            profile.status = "failed"
            profile.error = f"{type(error).__name__}: {error}"
            return None
        return result

    def profile_rows(self) -> tuple[dict[str, Any], ...]:
        rows = []
        for object_key, signatures in self._objects.items():
            for signature, languages in signatures.items():
                for language, backends in languages.items():
                    for backend, variant in backends.items():
                        rows.append({
                            "object_key": object_key,
                            "signature": signature,
                            "shell_language": language.value,
                            "interior_backend": backend,
                            "compile_ns": variant.profile.compile_ns,
                            "median_ns": variant.profile.median_ns,
                            "samples_ns": tuple(variant.profile.samples_ns),
                            "status": variant.profile.status,
                            "error": variant.profile.error,
                        })
        return tuple(rows)


__all__ = [
    "EvaluationPatternMap",
    "EvaluationProfile",
    "EvaluationVariantKey",
    "LazyEvaluationVariant",
]
