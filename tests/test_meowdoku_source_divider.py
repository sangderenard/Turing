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
    # The enduring record: the same job as a dream document, part by part, read back through the compiler's own loader.
    from src.compiler.dream_document import load_dream_document
    path = sd.write_dream_record(db, state, job)
    assert path.name == "record.dream"
    record = load_dream_document(path)
    parts = {b.outputs: b for b in record.blocks if b.decorations.get("role") == "part"}
    scale, raw = parts[("scale",)], parts[("raw",)]
    assert scale.language == "python" and scale.payload == FLOAT.rstrip("\n")
    assert scale.decorations["cut"] == "exact lexical closure" and scale.decorations["first_line"] == "1"
    assert scale.decorations["verdict"] == "lowered (wasm)" and scale.decorations["attempt.wasm"] == "complete"
    assert raw.payload == "def raw(x):\n    return x" and raw.decorations["verdict"] == "stays python"
    assert raw.decorations["reason"] == "needs-annotation" and raw.decorations["annotated"] == "no"
    lowering = next(b for b in record.blocks if b.decorations.get("role") == "lowering")
    assert lowering.language == "wasm" and lowering.decorations["lowers"] == scale.identity
    assert lowering.content_hash == row["product_sha256"], "the product is carried byte for byte"
    assert lowering.payload.encode("latin-1")[:4] == b"\x00asm"
    stated = json.loads(record.block("record").payload)
    assert stated["schema"] == "turing.divider-dream-record.v1" and stated["job_id"] == job
    assert (stated["pieces"], stated["lowered_pieces"], stated["python_pieces"], stated["lowering_blocks"]) == (2, 1, 1, 1)
    assert [b.identity for b in record.blocks][-1] == "record" and record.card_graph()["cards"]
    assert sd.write_dream_record(db, state, job) == path, "writing it again replaces it"


CHAIN = """
import math

RATE = 2.0

def unused(x: float) -> float:
    return x - 1.0

def double(x: float) -> float:
    return x * RATE

def step(x: float) -> float:
    y = double(x) + 1.0
    return y
"""


def test_every_part_of_the_record_holds_its_own_closure(tmp_path):
    """No compile here: the cut is the compiler's own partition, and a part that calls another carries that other."""
    state, db, job = _job(tmp_path, CHAIN)
    from src.compiler.dream_document import load_dream_document
    record = load_dream_document(sd.write_dream_record(db, state, job))
    parts = {b.outputs[0]: b for b in record.blocks if b.decorations.get("role") == "part"}
    assert set(parts) == {"unused", "double", "step"}
    step = parts["step"]
    assert "def double" in step.payload and "RATE = 2.0" in step.payload and "def step" in step.payload
    assert "def unused" not in step.payload and "import math" not in step.payload, "only what the part needs"
    assert step.inputs == ("double",) and step.decorations["depends"] == parts["double"].identity
    assert "def step" not in parts["double"].payload and "RATE = 2.0" in parts["double"].payload
    for name, part in parts.items():
        namespace = {}
        exec(compile(part.payload, f"<{part.identity}>", "exec"), namespace)     # each part stands alone
        assert callable(namespace[name])
    assert step.payload.splitlines()[0] == "RATE = 2.0"
    assert step.decorations["first_line"] == "4", "the line the closure starts at in the source"


def test_the_goal_document_holds_each_units_own_text_and_its_links(tmp_path):
    """No compile here. The Goal document is the linked form: a unit names what it takes, it does not copy it."""
    from meowdoku import rest_api
    if rest_api.goal is None:
        import pytest
        pytest.skip("the SpeakToMe checkout has no speaktome_goal package")
    state, db, job = _job(tmp_path, CHAIN)
    path = sd.write_goal_document(db, state, job)
    assert path.name == "goal.goal"
    document = rest_api.goal.load(path)
    units = {u.entries()[0]: u for u in document.units}
    step, double = units["step"], units["double"]
    assert "def double" not in step.text and "def step" in step.text, "the unit holds its own definition only"
    assert "RATE" not in step.text and "RATE = 2.0" in double.text, "a constant stays with the unit that uses it"
    assert [dict(r) for r in step.link] == [{"name": "double", "unit": double.identity, "entry": "double", "how": "reference"}]
    assert double.link == () and units["unused"].link == ()
    assert [dict(r) for r in step.api] == [{"entry": "step", "name": "x", "role": "parameter", "type": "float"},
                                           {"entry": "step", "name": "return", "role": "result", "type": "float"}]
    assert all(u.state == "source" and u.build == () for u in document.units), "nothing was lowered in this test"
    assert document.achieved() == () and document.targets() == ()
    graph = {n["identity"]: n for n in rest_api.goal.as_graph(document)}
    root = f"goal:job-{job}"
    assert {"target": f"{root}/unit/{double.identity}", "relationship": "links-to"} in graph[f"{root}/unit/{step.identity}"]["dependencies"]


def test_a_unit_keeps_what_its_own_definitions_reach_and_nothing_else(tmp_path):
    from meowdoku import rest_api
    if rest_api.goal is None:
        import pytest
        pytest.skip("the SpeakToMe checkout has no speaktome_goal package")
    shared = CHAIN.replace("y = double(x) + 1.0", "y = double(x) + RATE").replace("RATE = 2.0", "BASE = 1.0\nRATE = BASE * 2.0")
    state, db, job = _job(tmp_path, shared)
    units = {u.entries()[0]: u for u in rest_api.goal.load(sd.write_goal_document(db, state, job)).units}
    for name in ("step", "double"):
        assert "RATE = BASE * 2.0" in units[name].text and "BASE = 1.0" in units[name].text, "a constant both use stays in each"
    assert "def double" not in units["step"].text and "import math" not in units["step"].text
    namespace = {"double": lambda x: x}                  # what the link provides, supplied by the reader
    exec(compile(units["step"].text, "<step>", "exec"), namespace)
    assert namespace["step"](3.0) == 5.0


def test_a_record_goes_back_in_and_its_parts_build_one_by_one(tmp_path):
    """Dream in, dream out: the second pass sees every part as its own file and reaches the same products."""
    from src.compiler.dream_document import DreamBlock, DreamDocument, load_dream_document, render_dream_document
    state, db, job = _job(tmp_path, CHAIN, backends=("wasm",))
    assert sd.work(db, state, "first") == 3
    first = load_dream_document(sd.write_dream_record(db, state, job))
    # With every unit lowered, the Goal document says the goal is achieved for that language, and how each is called.
    from meowdoku import rest_api
    if rest_api.goal is not None:
        goal_document = rest_api.goal.load(sd.write_goal_document(db, state, job))
        assert goal_document.achieved() == ("wasm",)
        assert [dict(r) for r in goal_document.targets()] == [{"target": "wasm", "units": "3", "lowered": "3", "achieved": "yes"}]
        built = {u.entries()[0]: u for u in goal_document.units}
        record_blocks = {b.identity: b.content_hash for b in first.blocks}
        for unit in built.values():
            row = dict(unit.build[0])
            assert record_blocks[row["record"]] == row["sha256"], "a build row names its block in the units record"
            assert unit.abi and unit.abi[0]["source"] == "x" and unit.abi[0]["target"] == "wasm"
        assert built["step"].link[0]["unit"] == built["double"].identity
        assert goal_document.facts["units_record"].startswith("record.dream sha256 ")
    # A block in another language rides along untouched.
    shader = DreamBlock(identity="colors", kind="shader", language="glsl", stage="compute", encoding="utf-8",
                        payload="void main() {}", outputs=("colors",))
    fed = tmp_path / "fed.dream"
    fed.write_bytes(render_dream_document(DreamDocument(first.blocks + (shader,))))
    intake = sd.intake_dream(state, fed)
    assert sorted(intake.kept) == ["piece-0.py", "piece-1.py", "piece-2.py"]
    reasons = dict(intake.skipped)
    assert reasons["colors"] == "carried unchanged (glsl)" and "earlier pass" in reasons["record"]
    second_job = sd.submit(db, state, intake, owner_key="t", backends=("wasm",))
    sd.work(db, state, "second")
    second = load_dream_document(sd.write_dream_record(db, state, second_job))

    def lowered(document):
        by_part = {b.identity: b for b in document.blocks if b.decorations.get("role") == "part"}
        found = {}
        for block in document.blocks:
            if block.decorations.get("role") == "lowering":
                found.setdefault(by_part[block.decorations["lowers"]].outputs[0], set()).add(block.content_hash)
        return found
    assert set(lowered(first)) == {"unused", "double", "step"}
    # A part's closure also holds the functions it calls, so the second pass divides further; the functions the
    # parts are FOR must all have built again, each from its own block alone, to the same bytes.
    again = lowered(second)
    for name, digests in lowered(first).items():
        assert digests <= again.get(name, set()), f"{name} did not rebuild to the same product from its own part"
    assert second.block("colors").payload == "void main() {}" and second.block("colors").stage == "compute"
    stated = json.loads(second.block("record").payload)
    assert stated["came_from_dream"] and stated["carried_blocks"] == ["colors"]


def test_real_refusal_recorded_with_stage_and_type(tmp_path):
    state, db, job = _job(tmp_path, "def broken(x: float) -> float:\n    return undefined_name(x)\n")
    assert sd.work(db, state, "w", max_attempts=1) == 1
    row = db.execute("SELECT * FROM divider_attempts").fetchone()
    print("refusal:", row["state"], row["stage"], row["error_type"], row["error"])
    assert row["state"] in ("refused", "error") and row["error_type"] and row["stage"] in ("ssa-lowering", "emission", "partition")
