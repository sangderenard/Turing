"""examples/meowdoku/source_divider.py: safe intake, triage plan, queue semantics (fake runner), one real wasm attempt, one real
refusal, recomposition.  Temp state roots only."""
from __future__ import annotations

import json
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import lowering_library as lowering_mod  # noqa: E402
from meowdoku import source_divider as sd  # noqa: E402
from meowdoku.state_paths import StatePaths  # noqa: E402

THREE = '''
def b(x: float) -> float:
    return x * 2.0

def a(x: float) -> float:
    return b(x) + 1.0

def c(x):
    return x
'''
FLOAT = "def scale(x: float) -> float:\n    y = x * 2.0\n    return y + 1.0\n"


def _job(tmp_path, text, backends=("wasm",), name="m.py"):
    state = StatePaths(tmp_path / "state")
    db = sd.connect(state.accounts_db)
    job = sd.submit(db, state, sd.intake_text(state, name, text), owner_key="k", backends=backends)
    return state, db, job


def test_intake_zip_safety(tmp_path):
    z = tmp_path / "x.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("ok/a.py", "x = 1\n")
        zf.writestr("../evil.py", "x = 2\n")
        zf.writestr("/abs.py", "x = 3\n")
        zf.writestr("README.md", "hi")
        link = zipfile.ZipInfo("link.py")
        link.external_attr = (0o120777 << 16)
        zf.writestr(link, "a.py")
    state = StatePaths(tmp_path / "state")
    intake = sd.intake_zip(state, z)
    assert intake.kept == ["ok/a.py"]
    reasons = dict(intake.skipped)
    assert reasons["../evil.py"] == "unsafe-path" and reasons["/abs.py"] == "unsafe-path"
    assert reasons["README.md"] == "not-python" and reasons["link.py"] == "symlink"
    assert not (state.jobs_dir / intake.job_id / "evil.py").exists()
    try:
        sd.intake_git(state, "https://example.invalid/r.git")
        raise AssertionError("intake_git must refuse without allow_network")
    except sd.IntakeError:
        pass


def test_plan_order_and_annotation(tmp_path):
    state, db, job = _job(tmp_path, THREE)
    ps = sd.pieces(db, job)
    by = {p["names"][0]: p for p in ps}
    assert by["b"]["ord"] < by["a"]["ord"] and by["b"]["piece_id"] in by["a"]["deps"]
    assert by["a"]["annotated"] and by["b"]["annotated"] and not by["c"]["annotated"]
    plan_doc = json.loads((state.jobs_dir / job / "plan.json").read_text(encoding="utf-8"))
    assert plan_doc["kind"] == "triage"
    st = {r["piece_id"]: r["state"] for r in db.execute("SELECT piece_id,state FROM divider_attempts")}
    assert st[by["c"]["piece_id"]] == "needs-annotation" and st[by["a"]["piece_id"]] == "queued"


def test_queue_semantics_with_fake_runner(tmp_path):
    state, db, job = _job(tmp_path, THREE)
    first = sd.claim_next(db, "w1", 60)
    assert first["names"] == ["b"]
    assert sd.claim_next(db, "w2", 60) is None            # a waits for b: dependencies first, one lease at a time
    assert not sd.complete(db, job, first["piece_id"], "wasm", "w1", first["generation"] + 1)   # wrong generation
    assert not sd.complete(db, job, first["piece_id"], "wasm", "w2", first["generation"])       # wrong worker
    assert sd.refuse(db, job, first["piece_id"], "wasm", "w1", first["generation"], error_type="X", error="e", stage="emission")
    second = sd.claim_next(db, "w1", 60)                    # b was refused: a is still attempted, noted
    assert second["names"] == ["a"] and second["dep_refused"] == [first["piece_id"]]
    assert sd.reclaim_expired(db, time.time() + 120) == 1  # lease expired -> requeued
    assert not sd.complete(db, job, second["piece_id"], "wasm", "w1", second["generation"])     # stale generation
    third = sd.claim_next(db, "w3", 60)
    assert third["piece_id"] == second["piece_id"] and third["generation"] == second["generation"] + 1
    calls = []

    def fake(job_, piece, backend, out, timeout):
        calls.append(piece["names"][0])
        return {"state": "complete", "stage": "emission", "seconds": 0.0, "product_sha256": "ab" * 32, "entries": []}

    assert sd.complete(db, job, third["piece_id"], "wasm", "w3", third["generation"], stage="emission")
    assert sd.work(db, state, "w4", runner=fake) == 0 and calls == []     # c is needs-annotation: never attempted


def test_real_attempt_complete_and_recomposition(tmp_path):
    state, db, job = _job(tmp_path, FLOAT + "\ndef raw(x):\n    return x\n")
    t = time.time()
    assert sd.work(db, state, "w", max_attempts=1) == 1
    seconds = time.time() - t
    print(f"real wasm attempt wall seconds: {seconds:.2f}")
    row = db.execute("SELECT * FROM divider_attempts WHERE state='complete'").fetchone()
    assert row["product_sha256"] and row["api_path"].endswith(".api.yaml") and row["seconds"] > 0
    assert any(c["family"] == "divider" for c in lowering_mod.cores(state.accounts_db))
    low = lowering_mod.lowerings(state.accounts_db)
    assert low and low[0]["product_sha256"] == row["product_sha256"] and low[0]["verdict"] == "built"
    doc = json.loads(sd.write_recomposition(db, state, job).read_text(encoding="utf-8"))
    assert doc["schema"] == "turing.divider-recomposition.v1"
    fn = {f["name"]: f for f in doc["functions"]}
    assert not fn["scale"]["python"] and fn["scale"]["lowerings"][0]["sha256"] == row["product_sha256"]
    assert fn["raw"]["python"] and fn["raw"]["reason"] == "needs-annotation"


def test_real_refusal_recorded_with_stage_and_type(tmp_path):
    state, db, job = _job(tmp_path, "def broken(x: float) -> float:\n    return undefined_name(x)\n")
    assert sd.work(db, state, "w", max_attempts=1) == 1
    row = db.execute("SELECT * FROM divider_attempts").fetchone()
    print("refusal:", row["state"], row["stage"], row["error_type"], row["error"])
    assert row["state"] in ("refused", "error") and row["error_type"] and row["stage"] in ("ssa-lowering", "emission", "partition")
