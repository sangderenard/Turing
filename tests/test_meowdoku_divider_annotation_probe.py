"""Isolate the divider annotation gate without changing live admission.

These are admission/adapter tests, not compiler-inference acceptance tests.
They use temporary state, the existing divider queue and the real Nodus request
adapter. No compiler, Nodus connection, shader link or rendering is performed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import divider_nodus_bridge as bridge  # noqa: E402
from meowdoku import source_divider as divider  # noqa: E402
from meowdoku.state_paths import StatePaths  # noqa: E402


@pytest.mark.parametrize("backend", ["wasm", "llvm"])
def test_test_only_bypass_preserves_source_and_exposes_next_gate(tmp_path, monkeypatch, backend):
    source = "def scale(x):\n    return x * 2.0\n"
    state = StatePaths(tmp_path / "state")
    db = divider.connect(state.accounts_db)
    original_check = divider._annotated
    try:
        ordinary = divider.submit(
            db, state, divider.intake_text(state, "ordinary.py", source),
            owner_key="annotation-test", backends=(backend,),
        )
        ordinary_piece = divider.pieces(db, ordinary)[0]
        assert ordinary_piece["annotated"] is False
        assert divider.claim_next(db, "probe", 30) is None

        # User-requested experiment only. No production option or live job is
        # changed, and no annotations, casts or inferred ABI are written.
        with monkeypatch.context() as patch:
            patch.setattr(divider, "_annotated", lambda node, in_class: True)
            probe = divider.submit(
                db, state, divider.intake_text(state, "probe.py", source),
                owner_key="annotation-test", backends=(backend,),
            )
        assert divider._annotated is original_check
        assert (state.jobs_dir / probe / "source" / "probe.py").read_text(encoding="utf-8") == source

        claim = divider.claim_next(db, "probe", 30)
        assert claim is not None and claim["job_id"] == probe
        assert claim["names"] == ["scale"]
        output = tmp_path / "compiler-products"
        request = bridge.request_for_piece(
            db, claim, jobs_dir=output, source_root=state.jobs_dir,
        )
        assert request is None
        row = dict(db.execute(
            "SELECT * FROM divider_attempts WHERE job_id=?", (probe,),
        ).fetchone())
        assert row["state"] == "refused"
        assert row["stage"] == "nodus-row-tool: request"
        if backend == "wasm":
            assert row["error_type"] == "BackendNotServed"
            assert "attempt backend is 'wasm'" in row["error"]
        else:
            assert row["error_type"] == "RequestNotExpressible"
            assert "formal 'x' is annotated None" in row["error"]
        assert not output.exists(), "adapter refusal must not write compiler products"
        assert row["product_sha256"] is None
        assert row["seconds"] is None, "no compiler was started"

        # Prove the scoped bypass cannot affect a later ordinary submission.
        after = divider.submit(
            db, state, divider.intake_text(state, "after.py", source),
            owner_key="annotation-test", backends=(backend,),
        )
        assert divider.pieces(db, after)[0]["annotated"] is False
        assert divider.claim_next(db, "probe", 30) is None
    finally:
        db.close()


def test_truthful_scalar_annotation_is_not_a_batched_tensor_abi(tmp_path):
    source = "def scale(x: float) -> float:\n    return x * 2.0\n"
    state = StatePaths(tmp_path / "state")
    db = divider.connect(state.accounts_db)
    try:
        job = divider.submit(
            db, state, divider.intake_text(state, "scalar.py", source),
            owner_key="annotation-test", backends=("llvm",),
        )
        assert divider.pieces(db, job)[0]["annotated"] is True
        claim = divider.claim_next(db, "probe", 30)
        assert claim is not None
        assert bridge.request_for_piece(
            db, claim, jobs_dir=tmp_path / "compiler-products", source_root=state.jobs_dir,
        ) is None
        row = dict(db.execute("SELECT * FROM divider_attempts WHERE job_id=?", (job,)).fetchone())
        assert row["error_type"] == "RequestNotExpressible"
        assert "formal 'x' is annotated 'float'" in row["error"]
        assert row["stage"] == "nodus-row-tool: request"
        assert not (tmp_path / "compiler-products").exists()
    finally:
        db.close()
