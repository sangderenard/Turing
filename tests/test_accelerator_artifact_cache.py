from __future__ import annotations

from src.common.tensors.accelerator_backends.artifact_cache import (
    RepositoryArtifactCache,
    artifact_identity,
)


def test_artifact_identity_is_order_independent_and_semantic():
    assert artifact_identity({"case": "add", "options": {"fast": False}}) == (
        artifact_identity({"options": {"fast": False}, "case": "add"})
    )
    assert artifact_identity({"case": "add"}) != artifact_identity(
        {"case": "mul"}
    )


def test_repository_artifact_cache_validates_manifest_and_payload(tmp_path):
    cache = RepositoryArtifactCache(tmp_path)
    record = {"case": "grab-bag", "optimization": False}

    written = cache.store("llvm", record, "define void @run() { ret void }", suffix=".ll")
    loaded = cache.load("llvm", record, suffix=".ll")

    assert not written.hit
    assert loaded is not None
    assert loaded.hit
    assert loaded.identity == written.identity
    assert loaded.source == written.source

    source_path = tmp_path / "llvm" / f"{written.identity}.ll"
    source_path.write_text("corrupt", encoding="utf-8")
    assert cache.load("llvm", record, suffix=".ll") is None
