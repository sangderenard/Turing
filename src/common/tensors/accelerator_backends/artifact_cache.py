"""Deterministic repository-local cache for accelerated compiler artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any, Mapping


_CACHE_SCHEMA = "turing-accelerated-artifact-v1"


def repository_cache_root() -> Path:
    configured = os.environ.get("TURING_ACCELERATOR_CACHE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    repository = Path(__file__).resolve().parents[4]
    return repository / ".turing-cache" / "accelerator_backends"


def implementation_digest(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in sorted((path.resolve() for path in paths), key=str):
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def artifact_identity(record: Mapping[str, Any]) -> str:
    payload = json.dumps(
        {
            "cache_schema": _CACHE_SCHEMA,
            **dict(record),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class CachedArtifact:
    target: str
    identity: str
    source: str
    manifest: Mapping[str, Any]
    hit: bool


class RepositoryArtifactCache:
    """Atomic text-artifact cache whose manifest is the validity contract."""

    def __init__(self, root: Path | None = None):
        self.root = (root or repository_cache_root()).resolve()

    def _paths(self, target: str, identity: str, suffix: str):
        directory = self.root / target
        return (
            directory / f"{identity}{suffix}",
            directory / f"{identity}.json",
        )

    def load(
        self,
        target: str,
        record: Mapping[str, Any],
        *,
        suffix: str,
    ) -> CachedArtifact | None:
        identity = artifact_identity(record)
        source_path, manifest_path = self._paths(target, identity, suffix)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            source = source_path.read_text(encoding="utf-8")
        except (OSError, ValueError, UnicodeError):
            return None
        if manifest.get("identity") != identity:
            return None
        if manifest.get("record") != dict(record):
            return None
        if manifest.get("source_sha256") != hashlib.sha256(
            source.encode("utf-8")
        ).hexdigest():
            return None
        return CachedArtifact(
            target=target,
            identity=identity,
            source=source,
            manifest=manifest,
            hit=True,
        )

    def store(
        self,
        target: str,
        record: Mapping[str, Any],
        source: str,
        *,
        suffix: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> CachedArtifact:
        identity = artifact_identity(record)
        source_path, manifest_path = self._paths(target, identity, suffix)
        source_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "cache_schema": _CACHE_SCHEMA,
            "identity": identity,
            "record": dict(record),
            "source_sha256": hashlib.sha256(
                source.encode("utf-8")
            ).hexdigest(),
            "metadata": dict(metadata or {}),
        }
        discriminator = f"{os.getpid()}.{threading.get_ident()}.tmp"
        source_temporary = source_path.with_suffix(
            source_path.suffix + f".{discriminator}"
        )
        manifest_temporary = manifest_path.with_suffix(
            manifest_path.suffix + f".{discriminator}"
        )
        source_temporary.write_text(source, encoding="utf-8")
        manifest_temporary.write_text(
            json.dumps(manifest, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        os.replace(source_temporary, source_path)
        os.replace(manifest_temporary, manifest_path)
        return CachedArtifact(
            target=target,
            identity=identity,
            source=source,
            manifest=manifest,
            hit=False,
        )


__all__ = [
    "CachedArtifact",
    "RepositoryArtifactCache",
    "artifact_identity",
    "implementation_digest",
    "repository_cache_root",
]
