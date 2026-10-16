from pathlib import Path

import pytest

from src.compiler.build_product_retention import (
    compiler_product_lease,
    prune_abandoned_product_temporaries,
    prune_stale_orbital_profile_products,
)


def test_orbital_profile_retention_keeps_active_and_rejects_other_names(
    tmp_path: Path,
) -> None:
    active = tmp_path / "profile_rules_2_llvm_active"
    stale_c = tmp_path / "profile_rules_1_c_stale"
    stale_llvm = tmp_path / "profile_rules_9_llvm_stale"
    unrelated = tmp_path / "profile_rules_notes"
    for directory in (active, stale_c, stale_llvm, unrelated):
        directory.mkdir()

    removed = prune_stale_orbital_profile_products(tmp_path, active)

    assert set(removed) == {stale_c, stale_llvm}
    assert active.is_dir()
    assert unrelated.is_dir()


def test_orbital_profile_retention_requires_exact_direct_active_child(
    tmp_path: Path,
) -> None:
    nested = tmp_path / "nested" / "profile_rules_2_llvm_active"
    nested.mkdir(parents=True)

    with pytest.raises(ValueError, match="direct child"):
        prune_stale_orbital_profile_products(tmp_path, nested)

    unknown = tmp_path / "profile_rules_current"
    unknown.mkdir()
    with pytest.raises(ValueError, match="unknown name"):
        prune_stale_orbital_profile_products(tmp_path, unknown)


def test_retention_skips_another_process_product_lease(tmp_path: Path) -> None:
    active = tmp_path / "profile_rules_2_llvm_active"
    other = tmp_path / "profile_rules_3_c_other"
    active.mkdir()
    other.mkdir()

    with compiler_product_lease(other):
        removed = prune_stale_orbital_profile_products(tmp_path, active)

    assert removed == ()
    assert other.is_dir()


def test_active_product_cleanup_removes_only_abandoned_temporary_artifacts(
    tmp_path: Path,
) -> None:
    product = tmp_path / ("a" * 64)
    resumable = product / "growth" / "resolved-process-graph.pkl"
    temporary_file = product / "growth" / "manifest.json.tmp.12.3"
    temporary_tiles = product / "growth" / "temporary-tiles"
    resumable.parent.mkdir(parents=True)
    resumable.write_bytes(b"graph")
    temporary_file.write_bytes(b"partial")
    temporary_tiles.mkdir()
    (temporary_tiles / "tile").write_bytes(b"partial")

    with compiler_product_lease(product):
        removed = prune_abandoned_product_temporaries(product)

    assert set(removed) == {temporary_file, temporary_tiles}
    assert resumable.read_bytes() == b"graph"
