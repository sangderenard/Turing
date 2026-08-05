"""Atomic, phase-aware resume checkpoints for long AOT compilations."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import threading
from typing import Any, Mapping

from joblib.externals import cloudpickle

from .artifact_cache import repository_cache_root


_CHECKPOINT_SCHEMA = "turing-aot-checkpoint-v1"


def _new_lock():
    return threading.Lock()


def _new_rlock():
    return threading.RLock()


def _restore_sympy_global_parameters(state: Mapping[str, Any]):
    """Return SymPy's process singleton with its thread-local state restored."""

    from sympy.core.parameters import global_parameters

    global_parameters.__dict__.update(dict(state))
    return global_parameters


class _CheckpointPickler(cloudpickle.CloudPickler):
    """Snapshot compiler state while recreating synchronization handles."""

    dispatch_table = cloudpickle.CloudPickler.dispatch_table.copy()
    dispatch_table[type(threading.Lock())] = lambda _lock: (_new_lock, ())
    dispatch_table[type(threading.RLock())] = lambda _lock: (_new_rlock, ())
    try:
        from sympy.core.parameters import global_parameters

        dispatch_table[type(global_parameters)] = lambda parameters: (
            _restore_sympy_global_parameters,
            (dict(parameters.__dict__),),
        )
    except ImportError:
        pass


def _stable_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, (tuple, list)):
        return [_stable_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(
            (_stable_value(item) for item in value),
            key=lambda item: json.dumps(item, sort_keys=True),
        )
    if isinstance(value, Mapping):
        return {
            str(key): _stable_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return {
                "ndarray": str(value.dtype),
                "shape": tuple(map(int, value.shape)),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
            }
    except ImportError:
        pass
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "repr": repr(value),
    }


def callable_digest(*values: Any) -> str:
    digest = hashlib.sha256()
    for value in values:
        try:
            source = inspect.getsource(value)
        except (OSError, TypeError):
            source = repr(value)
        digest.update(source.encode("utf-8", errors="backslashreplace"))
        digest.update(b"\0")
    return digest.hexdigest()


class AOTCheckpointStore:
    """Persist compiler-owned phase artifacts under one semantic input key."""

    def __init__(self, record: Mapping[str, Any], root: str | Path | None = None):
        canonical = json.dumps(
            {"schema": _CHECKPOINT_SCHEMA, **_stable_value(record)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.identity = hashlib.sha256(canonical).hexdigest()
        base = Path(root).expanduser().resolve() if root else repository_cache_root()
        self.directory = base / "aot-checkpoints" / self.identity
        self.last_load_status = "not-attempted"

    def _paths(self, phase: str) -> tuple[Path, Path]:
        return self.directory / f"{phase}.pkl", self.directory / f"{phase}.json"

    def load(self, phase: str, implementation: str) -> Any | None:
        payload_path, manifest_path = self._paths(phase)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected_manifest = {
                "schema": _CHECKPOINT_SCHEMA,
                "identity": self.identity,
                "phase": phase,
                "implementation": implementation,
            }
            if manifest != expected_manifest:
                differing_fields = ", ".join(
                    key
                    for key in expected_manifest
                    if manifest.get(key) != expected_manifest[key]
                )
                self.last_load_status = (
                    "miss: manifest mismatch"
                    + (f" ({differing_fields})" if differing_fields else "")
                )
                return None
            with payload_path.open("rb") as stream:
                value = cloudpickle.load(stream)
            self.last_load_status = "hit"
            return value
        except (OSError, ValueError, TypeError, EOFError, ImportError) as error:
            self.last_load_status = (
                f"miss: {type(error).__name__}: {error}"
            )
            return None

    def store(self, phase: str, implementation: str, value: Any) -> Path:
        payload_path, manifest_path = self._paths(phase)
        self.directory.mkdir(parents=True, exist_ok=True)
        discriminator = f"{os.getpid()}.tmp"
        payload_temporary = payload_path.with_suffix(f".pkl.{discriminator}")
        manifest_temporary = manifest_path.with_suffix(f".json.{discriminator}")
        try:
            with payload_temporary.open("wb") as stream:
                _CheckpointPickler(stream, protocol=5).dump(value)
            manifest = {
                "schema": _CHECKPOINT_SCHEMA,
                "identity": self.identity,
                "phase": phase,
                "implementation": implementation,
            }
            manifest_temporary.write_text(
                json.dumps(manifest, sort_keys=True, indent=2),
                encoding="utf-8",
            )
            os.replace(payload_temporary, payload_path)
            os.replace(manifest_temporary, manifest_path)
        finally:
            payload_temporary.unlink(missing_ok=True)
            manifest_temporary.unlink(missing_ok=True)
        return payload_path


__all__ = ["AOTCheckpointStore", "callable_digest"]
