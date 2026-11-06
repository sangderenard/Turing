"""The fountain: a DEM granular law, a compiler-chosen and compiler-generated compute program,
no per-id tables, and deterministic image verification."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from meowdoku import fountain as F  # noqa: E402
from meowdoku import fountain_build as B  # noqa: E402
from src.common.tensors import AbstractTensor  # noqa: E402

REST_STEPS = int(14.0 / F.DT)               # sim time by which a 256-id pile is at rest


def test_the_size_ladder_reaches_ten_million_without_padding():
    counts = [count for _tag, (count, _pad) in F.SIZES.items()]
    assert counts == [256, 4096, 16384, 65536, 262144, 1048576, 4194304, 10_000_000]
    assert all(pad == count for count, pad in F.SIZES.values())           # no power-of-two pad: the sort is a merge exchange


def test_every_per_id_constant_is_derived_from_the_id_so_mix_picks_the_species():
    """One eager step: ``rad`` is the species radius derived from the id; mix 0 leaves two
    species, mix 1 all four.  No per-id table is an argument of the law."""
    n = 256
    assert F.ARGUMENTS == F.STATE + ("pid", "ir", "ig", "ib") + F.UNIFORMS
    piece = F.make_piece(n)
    radii = {}
    for mix in (0.0, 1.0):
        out = F.run_frames(piece, F.initial_columns(n, mix=mix), 1)
        radii[mix] = set(np.unique(np.round(out["rad"], 6)))
    assert len(radii[0.0]) == 2 and len(radii[1.0]) == 4
    assert radii[0.0] <= radii[1.0]


@pytest.fixture(scope="module")
def native_piece():
    n = F.N_SMALL
    columns = F.initial_columns(n, mix=0.5)
    piece = F.make_piece(n)
    piece.compile(columns)
    return piece, columns


def test_the_pile_forms_and_settles_by_contact_forces(native_piece):
    piece, columns = native_piece
    final = F.run_frames(piece, columns, REST_STEPS)
    geo = F.basin(F.N_SMALL)
    assert not np.isnan(final["x"]).any()
    assert np.all(np.abs(final["x"]) <= geo["half"] + 1e-3) and float(final["y"].min()) > -1e-3
    assert float(np.hypot(final["vx"], final["vy"]).max()) < 0.2          # at rest
    top = float(np.percentile(final["y"], 99))
    fraction = float((np.pi * final["rad"] ** 2).sum() / (geo["width"] * top))
    assert 0.55 < fraction < 0.95
    x, y, r = final["x"], final["y"], final["rad"]
    gap = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :]) - (r[:, None] + r[None, :])
    np.fill_diagonal(gap, 1.0)
    assert float(gap.min()) > -0.6 * float(r.max())


def test_the_replay_is_bit_identical_and_the_settled_pile_shows_the_image(native_piece):
    """End to end, in the native lane: phase 1 settles; the image is sampled at the recorded rest
    positions (the page's rule, ``fit_extent`` + ``sample_raster``); phase 2 replays the identical
    simulation, every id lands where it did, and each id's colour is its image sample."""
    piece, columns = native_piece
    steps = REST_STEPS
    again = F.run_frames(piece, columns, 300)
    assert all(np.array_equal(again[k], F.run_frames(piece, columns, 300)[k]) for k in F.STATE)
    phase1 = F.run_frames(piece, columns, steps)
    u, v = F.fit_extent(phase1["x"], phase1["y"])
    raster = (np.random.default_rng(0).random((72, 96, 3)) * 255).astype(np.uint8)
    samples = F.sample_raster(raster, u, v)
    phase2 = dict(columns)
    phase2.update(F.uniforms(n=256, mode=0, use_image=1.0, mix=0.5))
    phase2["ir"], phase2["ig"], phase2["ib"] = samples[:, 0], samples[:, 1], samples[:, 2]
    final = F.run_frames(piece, phase2, steps)
    assert np.array_equal(final["x"], phase1["x"]) and np.array_equal(final["y"], phase1["y"])
    assert np.array_equal(final["cr"], samples[:, 0]) and np.array_equal(final["cg"], samples[:, 1])
    assert np.array_equal(final["cb"], samples[:, 2])


def test_hue_modes_trig_and_rotation_differ_and_stay_in_range(native_piece):
    piece, columns = native_piece
    a = F.run_frames(piece, columns, 2)
    b = F.run_frames(piece, F.initial_columns(F.N_SMALL, mix=0.5, mode=1), 2)
    assert not np.allclose(a["cr"], b["cr"])
    for out in (a, b):
        assert all(0.0 <= float(out[k].min()) and float(out[k].max()) <= 1.0 for k in ("cr", "cg", "cb"))


def test_the_compiler_chooses_per_size_and_the_program_carries_no_per_id_data():
    # 256 ids sits at the break-even (K_RUN = 16 candidates per row put it just over the model's bar):
    # the verdict is the classifier's, so the test holds only that it is the weakest of the ladder
    small = B.verdict_for("small")
    assert small.speedup < B.verdict_for("4k").speedup / 2
    for tag in ("4k", "64k", "1M", "10M"):
        assert B.verdict_for(tag).pays, tag
    compiled, _program, _outputs, ids = B.passes_for("10M")
    assert compiled.complete, compiled.shortfalls
    pad = F.SIZES["10M"][1]
    # N is baked into the dispatch structure: Knuth's merge-exchange stages + search/contact + the iota
    assert compiled.dispatches == len(F.merge_exchange_stages(F.SIZES["10M"][0])) + 2 + len(F.reduction_shifts(pad))
    assert compiled.groups[1] > 1 and compiled.groups[0] * compiled.groups[1] * 64 >= pad     # a 2D grid
    assert ids["pid"] not in compiled.feed_slots                       # the iota is the invocation index
    assert sorted(compiled.feed_slots) == sorted(ids[n] for n in F.STATE + ("ir", "ig", "ib"))
    assert "lane0[" in compiled.source and "f32(i)" in compiled.source
    # bytes per id (rematerialised sort temporaries hold no slot): the pile state and colours, plus the seven reduced
    # diagnostics, each carried through the reduction's stages in two buffers
    assert compiled.arena_elements * 4 / pad < 170
    # the arena is split across bindings of at most 256 MiB, none of the single-buffer limit's business
    assert compiled.bindings > 1 and compiled.slots_per_binding * pad * 4 <= 256 * 1024 * 1024
    assert compiled.bindings == -(-compiled.slots // compiled.slots_per_binding)


def test_the_wgsl_program_is_generated_from_the_authored_law():
    compiled, _program, _outputs, ids = B.passes_for("small")
    assert compiled.complete, compiled.shortfalls
    assert compiled.dispatches == len(F.merge_exchange_stages(256)) + 2 + len(F.reduction_shifts(256))
    assert "atan2(" in compiled.source and "tanh(" in compiled.source
    assert compiled.source.count("@binding(") == compiled.bindings + 1       # the lane bindings and the uniform head
    assert compiled.backend_identity_decisions[0].identity == "shader_float64_storage_to_float32"
    assert set(compiled.uniform_index) == {ids[n] for n in F.UNIFORMS}


def test_elementwise_lowering_names_what_it_cannot_express():
    from src.common.tensors.accelerator_backends.ssa_backend import SSATensorOperations, SSATensorProgram
    from src.compiler import ssa_webgpu_backend as webgpu
    from src.compiler.work_contract import PRESETS, scoped_work_contract

    with scoped_work_contract(PRESETS["fast"]):
        program = SSATensorProgram("reduces")
        x = SSATensorOperations.input(program, (8,))
        with program.activate():
            total = (x * 2.0).sum()
        artifact = webgpu.emit_tensor_program_module(program.module, "reduces", [total.data.value], count=8)
        passes = webgpu.emit_tensor_program_passes(program.module, "reduces", [total.data.value], count=8)
    for compiled in (artifact, passes):
        assert not compiled.complete
        assert any("sum_double" in item.format() and "not an elementwise" in item.format()
                   for item in compiled.shortfalls)


JS_SAMPLE = """
import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
const [dir] = process.argv.slice(2);
const {sampleRasterAt} = await import(pathToFileURL(`${dir}/compute.js`).href);
const spec = JSON.parse(fs.readFileSync(`${dir}/job.json`, 'utf8'));
const f32 = (name, n) => { const b = fs.readFileSync(`${dir}/${name}.bin`); return new Float32Array(b.buffer, b.byteOffset, n); };
const raster = {w: spec.w, h: spec.h, data: new Uint8ClampedArray(fs.readFileSync(`${dir}/raster.bin`))};
const {bytes, channels} = sampleRasterAt(raster, f32('x', spec.n), f32('y', spec.n), spec.n, ['ir', 'ig', 'ib']);
fs.writeFileSync(`${dir}/out.bin`, Buffer.concat([Buffer.from(bytes), Buffer.from(Float32Array.from(channels.flatMap((c) => Array.from(c))).buffer)]));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node runs the page's sampling code")
def test_the_pages_image_sampling_equals_pythons_sample_raster(tmp_path):
    """The file and camera controls both end in ``sampleRasterAt``: held against ``fit_extent`` +
    ``sample_raster`` on a random pile and a random raster."""
    rng = np.random.default_rng(3)
    n, w, h = 4000, 96, 72
    x = rng.normal(0, 0.6, n).astype(np.float32)
    y = np.abs(rng.normal(0.4, 0.2, n)).astype(np.float32)
    raster = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    rgba = np.concatenate([raster, np.full((h, w, 1), 255, np.uint8)], axis=2)
    rgba.tofile(tmp_path / "raster.bin"); x.tofile(tmp_path / "x.bin"); y.tofile(tmp_path / "y.bin")
    (tmp_path / "job.json").write_text(json.dumps({"n": n, "w": w, "h": h}))
    shutil.copyfile(ROOT / "examples" / "meowdoku" / "web" / "compute.js", tmp_path / "compute.js")
    shutil.copyfile(ROOT / "examples" / "meowdoku" / "web" / "recording.js", tmp_path / "recording.js")
    shutil.copyfile(ROOT / "examples" / "mobile_reference_app" / "web" / "turing-wasm-shell.js", tmp_path / "turing-wasm-shell.js")
    (tmp_path / "run.mjs").write_text(JS_SAMPLE)
    done = subprocess.run(["node", str(tmp_path / "run.mjs"), str(tmp_path)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-1500:]
    raw = (tmp_path / "out.bin").read_bytes()
    got_bytes = np.frombuffer(raw[: 3 * n], dtype=np.uint8).reshape(n, 3)
    got = np.frombuffer(raw[3 * n:], dtype="<f4").reshape(3, n).T
    u, v = F.fit_extent(x, y)
    expect = F.sample_raster(raster, u, v)
    assert np.array_equal(got_bytes, np.rint(expect * 255).astype(np.uint8))
    assert np.array_equal(got.astype(np.float64), expect)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is needed to load the wasm product")
def test_wasm_product_agrees_with_the_native_law_over_a_short_horizon(tmp_path, native_piece):
    from meowdoku import fountain_parity as P

    piece, columns = native_piece
    B.build(tmp_path, ("small",))
    ref = F.run_frames(piece, columns, 120)
    got = P.wasm_frames(columns, "small", tmp_path, 120, job="test_small")
    assert max(P.compare(ref, got)["max_abs"].values()) < 1e-6


def test_server_game_declares_a_compute_product_sizes_and_count_only_images():
    import base64

    game = F.FountainGame(size="small")
    before = game.state()["compute"]["epoch"]
    game.do({"type": "fountain_start", "mix": 0.25, "mode": 1})
    state = game.state()
    assert (state["phase"], state["mode"], state["n"]) == ("fill", 1, 256)
    assert state["board"] == "free" and state["entities"] == [] and state["input"] == "compute"
    assert state["compute"]["epoch"] == before + 1 and state["compute"]["uniforms"] == {k: float(v) for k, v in F.uniforms(n=256, mode=1, mix=0.25).items()}
    assert "layout" not in state["compute"]                              # nothing per-id is served
    ids = {item["id"]: item for item in state["controls_ui"]}
    assert [o[0] for o in ids["size"]["options"]] == list(F.SIZES) and ids["replay"]["enabled"] is False
    with pytest.raises(ValueError):
        game.do({"type": "fountain_replay"})
    game.do({"type": "fountain_image", "rgb": base64.b64encode(((np.arange(256 * 3) % 251).astype(np.uint8)).tobytes()).decode()})
    assert {i["id"]: i for i in game.state()["controls_ui"]}["replay"]["enabled"] is True
    game.do({"type": "fountain_replay"})
    assert game.state()["phase"] == "replay" and game.state()["compute"]["uniforms"]["use_image"] == 1.0
    game.do({"type": "fountain_size", "size": "10M"})
    assert game.state()["n"] == 10_000_000 and not game.state()["has_image"]
    game.do({"type": "fountain_image", "count": 10_000_000})              # too many to ship: only the count
    assert game.state()["has_image"]
    with pytest.raises(ValueError):
        game.do({"type": "fountain_image", "count": 5})


def _one_step(columns, n=256):
    piece = F.make_piece(n)
    return F.run_frames(piece, columns, 1)


def test_an_id_emitted_this_step_is_not_a_ghost_in_the_pile():
    """The step an id is emitted its stored state is still the parked zero state, i.e. a body at the
    floor's centre.  It must take part in no contact, or each emission kicks the grains resting there.
    Id 0 (emit time 0) is emitted at age 0; id 1 rests overlapping the origin, already placed."""
    columns = F.initial_columns(256, mix=1.0)
    columns["age"][1] = 1.0
    columns["x"][1], columns["y"][1] = 0.01, 0.0145
    out = _one_step(columns)
    assert abs(out["vy"][1] - (-F.G * F.DT)) < 1e-4 and abs(out["vx"][1]) < 1e-4


def test_slip_friction_stays_finite_for_any_slip_speed_and_coincident_ids_separate():
    """An emitted grain meets the plume at ~1.5 m/s, so slip / VREF reaches ~75 and a tanh built from
    exp(2x) overflows f32 (NaN) past ~44: the argument is clamped.  Two ids at one point are pushed
    apart along an id-fixed direction instead of dividing a zero separation."""
    t = AbstractTensor.get_tensor([0.0, 5.0, -5.0])
    fx, fy, ft = F._contact(t + 0.01, t * 0.0, t * 0.0 + 1.0, t, t, t * 0.0, t * 0.0 + 1.0, t * 0.0 + 1.0, t * 0.0 + 0.3)
    assert np.isfinite(np.asarray(ft)).all()
    assert abs(float(np.asarray(ft)[1]) + 0.3 * float(np.asarray(fy)[1])) < 1e-6      # saturated: |ft| = mu * fn
    columns = F.initial_columns(256, mix=1.0)
    columns["age"][2:4] = 1.0
    columns["x"][2:4], columns["y"][2:4] = 0.05, 0.5
    out = _one_step(columns)
    assert np.isfinite(out["vx"]).all() and np.isfinite(out["vy"]).all()
    assert out["vx"][2] * out["vx"][3] < 0                                            # equal and opposite, by id


def test_the_merge_exchange_network_sorts_every_length():
    rng = np.random.default_rng(7)
    for n in (2, 3, 5, 17, 100, 1000, 4097):
        cell = rng.integers(0, 40, n).astype(np.float64)
        pid = np.arange(n, dtype=np.float64)
        kc, ki = cell.copy(), pid.copy()
        for (p, d, r) in F.merge_exchange_stages(n):
            bs = np.floor(pid / p) - 2 * np.floor(pid / (2 * p))
            pj = np.clip(pid - d, 0, n - 1)
            bo = np.floor(pj / p) - 2 * np.floor(pj / (2 * p))
            lower = (np.abs(bs - r) < .5) * (pid + d < n)
            upper = (np.abs(bo - r) < .5) * (pid >= d) * (1 - lower)
            partner = (pid + d * (lower - upper)).astype(int)
            oc, oi = kc[partner], ki[partner]
            less = (kc < oc) * 1.0 + (np.abs(kc - oc) < .5) * (ki < oi)
            keep = (1 - lower - upper) + lower * less + upper * (1 - less)
            kc, ki = np.where(keep > .5, kc, oc), np.where(keep > .5, ki, oi)
        assert np.array_equal(ki, np.lexsort((pid, cell))), n


# ------------------------------------------------------------------ emission, conservation, mirror, settle, aspect
def _series(piece, columns, steps, keep=()):
    """Step the compiled law ``steps`` times; returns (final state, per-step diagnostics totals, and the per-step
    columns named in ``keep``)."""
    state = {name: np.array(columns[name], dtype=np.float64) for name in F.STATE}
    diag = {name: [] for name in F.DIAGNOSTICS}
    kept = {name: [] for name in keep}
    for _ in range(steps):
        produced = dict(zip(F.OUTPUTS, piece(*[state[k] if k in state else columns[k] for k in F.ARGUMENTS])))
        for name in F.STATE:
            state[name] = produced[name + "_next"]
        for name in F.DIAGNOSTICS:
            diag[name].append(float(produced[name][0]))
        for name in keep:
            kept[name].append(np.array(produced[name] if name in produced else state[name]))
    return state, {k: np.array(v) for k, v in diag.items()}, kept, {k: produced[k] for k in ("rad",)}


@pytest.fixture(scope="module")
def emission_run(native_piece):
    """The 256-id fountain from round zero through its whole emission, with every step's post-step state kept."""
    piece, columns = native_piece
    steps = int(F.EMIT_SPAN / F.DT) + 120
    state, diag, kept, extra = _series(piece, columns, steps, keep=F.STATE)
    return {"state": state, "diag": diag, "kept": kept, "rad": extra["rad"], "steps": steps, "columns": columns}


def _system_energy(x, y, vx, vy, w, rad, live):
    """Kinetic + rotational + gravitational + the contacts' elastic energy of the live grains (the springs' ``kn delta^2 / 2``)."""
    species = {round(s[1], 6): s for s in F.SPECIES}
    mass = np.array([species[round(float(r), 6)][2] * np.pi * r * r for r in rad])
    idx = np.where(live)[0]
    m, r, px, py = mass[idx], rad[idx], x[idx], y[idx]
    total = float((0.5 * m * (vx[idx] ** 2 + vy[idx] ** 2) + 0.25 * m * r * r * w[idx] ** 2 + m * F.G * py).sum())
    d = np.hypot(px[:, None] - px[None, :], py[:, None] - py[None, :])
    delta = np.clip(r[:, None] + r[None, :] - d, 0.0, None)
    np.fill_diagonal(delta, 0.0)
    meff = m[:, None] * m[None, :] / (m[:, None] + m[None, :])
    total += float(0.25 * (F.OMEGA ** 2 * meff * delta ** 2).sum())                    # each pair once
    for gap in (r - py, r - (px + emission_half), r - (emission_half - px)):
        total += float((0.5 * F.OMEGA ** 2 * m * np.clip(gap, 0.0, None) ** 2).sum())
    return total


emission_half = F.basin(F.N_SMALL)["half"]


def test_an_emitted_grain_never_spawns_inside_another_and_the_system_gains_no_energy(emission_run):
    """Measured on the plume: the step an id is placed at the nozzle it overlaps no live grain, so the contact springs
    inject no energy there; and the system's mechanical energy (kinetic + rotational + gravitational + elastic) never
    exceeds what the emissions have put in (each grain's launch energy): collisions and drag only take energy out.  So
    no height or speed is beyond what the launches allow."""
    run, n = emission_run, F.N_SMALL
    x, y, vx, vy, w = (np.array(run["kept"][k]) for k in ("x", "y", "vx", "vy", "w"))      # (steps, n)
    rad = run["rad"]
    live = ((np.abs(x) + np.abs(y)) >= 1e-30) & (vy > F.ARMED * 0.5)               # launched: not parked, not armed at the slot
    placed_at = np.where(live.any(axis=0), live.argmax(axis=0), -1)
    assert (placed_at >= 0).all()                                                    # every id was emitted
    deepest = 0.0
    for j in range(n):
        k = placed_at[j]
        others = np.where(live[k])[0]
        others = others[others != j]
        gap = np.hypot(x[k, others] - x[k, j], y[k, others] - y[k, j]) - (rad[others] + rad[j])
        deepest = max(deepest, float(-gap.min()) if gap.size else 0.0)
    assert deepest <= 3e-3, deepest                       # placed overlapping nothing (4k, measured: 2 of 4096 spawns by <= 1.8 mm; the baseline: half of them, up to 25 mm)
    species = {round(s[1], 6): s for s in F.SPECIES}
    mass = np.array([species[round(float(r), 6)][2] * np.pi * r * r for r in rad])
    launch = 0.5 * mass * (vx[placed_at, np.arange(n)] ** 2 + vy[placed_at, np.arange(n)] ** 2) \
        + mass * F.G * y[placed_at, np.arange(n)]
    worst = -np.inf
    for k in range(0, run["steps"], 30):
        e_in = float(launch[placed_at <= k].sum())
        worst = max(worst, _system_energy(x[k], y[k], vx[k], vy[k], w[k], rad, live[k]) - e_in)
    assert worst < 0.03 * float(launch.sum()), worst      # semi-implicit Euler's own drift, ~1% (4k, measured)
    ceiling = (y[placed_at, np.arange(n)] + vy[placed_at, np.arange(n)] ** 2 / (2.0 * F.G)).max()
    assert float(np.where(live, y, -np.inf).max()) <= 1.06 * ceiling           # the highest grain is within 6% of the highest launch ceiling (measured: 4% at 256, 2% at 4k; before: 42% over)


def test_an_id_waits_for_a_clear_nozzle_slot_and_goes_the_step_it_clears(native_piece):
    piece, columns = native_piece
    n = F.N_SMALL
    geo = F.basin(n)
    slot = [float(np.asarray(v)[0]) for v in F._nozzle(AbstractTensor.get_tensor(np.zeros(1)), geo["bar_half"], geo["nozzle_y"])]

    def after(state, steps=1):
        out = F.run_frames(piece, {**columns, **state}, steps)
        return {k: np.array(out[k]) for k in F.STATE}

    armed = after({k: np.array(columns[k]) for k in F.STATE})
    assert armed["vy"][0] == F.ARMED and abs(armed["x"][0] - slot[0]) < 1e-6 and abs(armed["y"][0] - slot[1]) < 1e-6   # due: armed AT its slot
    blocked = dict(armed)
    for k in ("age", "x", "y"):
        blocked[k] = armed[k].copy()
    blocked["age"][5] = 1.0                                                          # a live grain beside id 0's slot
    blocked["x"][5], blocked["y"][5] = slot[0] + 0.005, slot[1]
    held = after(blocked)
    assert held["vy"][0] == F.ARMED and abs(held["x"][0] - slot[0]) < 1e-6          # id 0 waits, unmoved
    clear = dict(blocked)
    clear["x"] = blocked["x"].copy()
    clear["y"] = blocked["y"].copy()
    clear["x"][5], clear["y"][5] = slot[0] + 0.3, slot[1] - 0.5
    went = after(clear)
    assert abs(went["x"][0] - slot[0]) < 1e-6 and abs(went["y"][0] - slot[1]) < 1e-6 and went["vy"][0] > 0.5


def test_every_id_is_in_exactly_one_class_every_step_and_nothing_leaves_the_walls(emission_run):
    diag, n = emission_run["diag"], F.N_SMALL
    total = diag["n_nan"] + diag["n_out"] + diag["n_parked"] + diag["n_live"]
    assert np.array_equal(total, np.full(len(total), float(n)))
    assert diag["n_nan"].max() == 0 and diag["n_out"].max() == 0
    assert diag["n_drop"].max() == 0                                                  # the neighbour walk dropped no contact
    assert diag["n_live"][-1] == n and diag["n_parked"][-1] == 0
    assert (np.diff(diag["n_live"]) >= 0).all()                                       # nothing returns to the parked class


def test_the_reduction_is_the_sum_of_the_columns(emission_run):
    """Element 0 of the reduced diagnostics is the sum (the max) of the per-id values, computed here from the state."""
    run = emission_run
    last = run["steps"] - 1
    x, y, vx, vy, w = (run["kept"][k][last] for k in ("x", "y", "vx", "vy", "w"))
    rad = run["rad"]
    species = {round(s[1], 6): s for s in F.SPECIES}
    mass = np.array([species[round(float(r), 6)][2] * np.pi * r * r for r in rad])
    ke = float((0.5 * mass * (vx ** 2 + vy ** 2) + 0.5 * (0.5 * mass * rad ** 2) * w ** 2).sum())
    assert abs(run["diag"]["ke"][last] - ke) < 1e-9 * max(ke, 1.0)
    assert abs(run["diag"]["vmax2"][last] - float((vx ** 2 + vy ** 2).max())) < 1e-12


def test_a_mirrored_post_emission_state_evolves_mirrored(emission_run, native_piece):
    """x -> -x, vx -> -vx, spin -> -spin with the same ids: the evolution is the mirror image within float noise.  A
    neighbour walk that drops contacts on one side (the right-hand cells of a row run) would break it."""
    piece, columns = native_piece
    run = emission_run
    state = {k: np.array(run["kept"][k][-1]) for k in F.STATE}
    mirrored = dict(state)
    mirrored["x"], mirrored["vx"], mirrored["w"] = -state["x"], -state["vx"], -state["w"]
    steps = 80
    a, da, _, _ = _series(piece, {**columns, **state}, steps)
    b, db, _, _ = _series(piece, {**columns, **mirrored}, steps)
    assert da["n_drop"].max() == 0 and db["n_drop"].max() == 0
    assert float(np.abs(a["x"] + b["x"]).max()) < 1e-9 and float(np.abs(a["y"] - b["y"]).max()) < 1e-9
    assert float(np.abs(a["vx"] + b["vx"]).max()) < 1e-8 and float(np.abs(a["w"] + b["w"]).max()) < 1e-8


def test_the_settle_step_comes_from_the_energy_and_differs_with_size_and_mix(native_piece):
    piece, columns = native_piece
    steps = int(24.0 / F.DT)
    found = {}
    for mix in (0.0, 1.0):
        cols = F.initial_columns(F.N_SMALL, mix=mix)
        _s, diag, _k, _e = _series(piece, cols, steps)
        found[mix] = F.settle_step(diag, F.N_SMALL, F.N_SMALL)
        assert found[mix] is not None and found[mix] % F.SETTLE["sample"] == 0
        assert found[mix] * F.DT > F.EMIT_SPAN                                          # never before everything is emitted
    assert found[0.0] != found[1.0]


def test_the_basin_stretches_to_the_screen_at_the_same_pile_area():
    ref = F.basin(4096)
    for aspect in (0.35, 0.46, 1.0, 2.2, 3.0):
        geo = F.basin(4096, aspect)
        view = geo["view"]
        assert abs((view[1] - view[0]) / (view[3] - view[2]) - aspect) < 1e-9
        assert geo["cols"] * F.CELL >= geo["width"] and geo["rows"] * F.CELL >= geo["nozzle_y"] + 0.1
    assert F.basin(4096, 100.0)["view"] == F.basin(4096, F.ASPECT_MAX)["view"]          # clamped
    assert F.basin(4096, None) == ref and F.basin(4096)["width"] == F.WIDTH_BASE
    # a narrow basin is tall and a wide one is flat: the pile (width x height) keeps its area
    tall, flat = F.basin(4096, 0.4), F.basin(4096, 2.5)
    assert tall["width"] < ref["width"] < flat["width"] and tall["nozzle_y"] > ref["nozzle_y"] > flat["nozzle_y"]


def test_the_basin_is_a_runtime_uniform_so_one_program_serves_every_aspect_and_fills_it(native_piece):
    """The same compiled piece, three basins: the pile stays inside each one's walls, nothing is lost and no contact is dropped."""
    piece, _columns = native_piece
    for aspect in (0.4, 1.0, 2.5):
        cols = F.initial_columns(F.N_SMALL, mix=0.5, aspect=aspect)
        geo = F.basin(F.N_SMALL, aspect)
        assert float(cols["half"]) == pytest.approx(geo["half"], rel=1e-6)
        state, diag, _k, _e = _series(piece, cols, 1500)
        live = diag["n_live"][-1]
        assert diag["n_nan"].max() == 0 and diag["n_out"].max() == 0 and live > 0
        placed = (np.abs(state["x"]) + np.abs(state["y"])) > 0
        assert float(np.abs(state["x"][placed]).max()) <= geo["half"] + 1e-3
        assert float(state["y"][placed].min()) > -1e-3


def test_the_server_basin_is_fixed_by_the_run_and_kept_by_its_replay():
    import base64

    game = F.FountainGame(size="small", aspect=2.0)
    first = game.state()["compute"]
    assert first["aspect"] == 2.0 and abs((first["view"][1] - first["view"][0]) / (first["view"][3] - first["view"][2]) - 2.0) < 1e-9
    width, height = F.board_size(first["view"])
    assert abs(width / height - 2.0) < 1e-9 and game.state()["width"] == width
    assert first["uniforms"]["half"] == pytest.approx(F.basin(256, 2.0)["half"], rel=1e-6)
    game.do({"type": "fountain_start", "aspect": 0.5})                                  # a new run takes the screen's aspect
    run = game.state()["compute"]
    assert run["aspect"] == 0.5 and run["uniforms"]["half"] < first["uniforms"]["half"]
    game.do({"type": "fountain_image", "rgb": base64.b64encode(bytes(256 * 3)).decode()})
    game.do({"type": "layout", "aspect": 3.0}) if hasattr(game, "layout") else None     # the screen rotates: nothing changes
    game.do({"type": "fountain_replay"})
    replay = game.state()["compute"]
    assert replay["view"] == run["view"] and replay["aspect"] == 0.5
    assert {k: v for k, v in replay["uniforms"].items() if k not in ("use_image",)} == \
        {k: v for k, v in run["uniforms"].items() if k not in ("use_image",)}


JS_SETTLE = """
import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
const [dir] = process.argv.slice(2);
const {SettleDetector} = await import(pathToFileURL(`${dir}/compute.js`).href);
const spec = JSON.parse(fs.readFileSync(`${dir}/series.json`, 'utf8'));
const detector = new SettleDetector(spec.params, spec.active, spec.pad);
let found = null;
for (let step = spec.params.sample; step <= spec.ke.length && found === null; step += spec.params.sample) {
  found = detector.push(step, {ke: spec.ke[step - 1], vmax2: spec.vmax2[step - 1], n_parked: spec.n_parked[step - 1]});
}
fs.writeFileSync(`${dir}/settle.json`, JSON.stringify({step: found}));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node runs the page's settle rule")
def test_the_pages_settle_rule_equals_pythons_on_a_native_series(tmp_path, emission_run):
    """The page ends phase 1 with ``SettleDetector`` (compute.js); ``fountain.settle_step`` is the same rule: held on the
    diagnostics of a native run (and on a series that quiets, built from it)."""
    diag = emission_run["diag"]
    quiet = {k: np.concatenate([v, np.full(4000, v[-1] if k != "ke" else 1e-9)]) for k, v in diag.items()}
    quiet["vmax2"][len(diag["ke"]):] = 1e-6
    shutil.copyfile(ROOT / "examples" / "meowdoku" / "web" / "compute.js", tmp_path / "compute.js")
    shutil.copyfile(ROOT / "examples" / "meowdoku" / "web" / "recording.js", tmp_path / "recording.js")
    shutil.copyfile(ROOT / "examples" / "mobile_reference_app" / "web" / "turing-wasm-shell.js", tmp_path / "turing-wasm-shell.js")
    (tmp_path / "run.mjs").write_text(JS_SETTLE)
    for series in (diag, quiet):
        (tmp_path / "series.json").write_text(json.dumps({
            "params": F.SETTLE, "active": F.N_SMALL, "pad": F.N_SMALL,
            "ke": series["ke"].tolist(), "vmax2": series["vmax2"].tolist(), "n_parked": series["n_parked"].tolist()}))
        done = subprocess.run(["node", str(tmp_path / "run.mjs"), str(tmp_path)], capture_output=True, text=True)
        assert done.returncode == 0, done.stderr[-1500:]
        assert json.loads((tmp_path / "settle.json").read_text())["step"] == F.settle_step(series, F.N_SMALL, F.N_SMALL)
    assert F.settle_step(quiet, F.N_SMALL, F.N_SMALL) is not None


# ---------------------------------------------------------- run modes: gravity, recording, playback
#: sha256 of the float64 bytes of STATE + (cr, cg, cb) after 700 steps of the 256-id law at mix 0.5, as the law ran BEFORE gravity
#: became a runtime vector (``G`` baked into ``vy``).  The default inputs must reproduce it bit for bit.
PRE_GRAVITY_DIGEST_700 = "576e50eb1463aa61fa3727fae820dadd018e01f587c0e54a46a70de49692643f"


def _digest(final) -> str:
    import hashlib
    return hashlib.sha256(b"".join(np.ascontiguousarray(final[k], dtype=np.float64).tobytes() for k in F.STATE + ("cr", "cg", "cb"))).hexdigest()


def test_gravity_is_a_runtime_input_the_default_is_the_old_law_and_a_tilt_changes_it_without_a_rebuild(native_piece):
    piece, columns = native_piece
    assert F.RUNTIME_INPUTS == ("grav_x", "grav_y", "lid", "rows") and set(F.RUNTIME_INPUTS) <= set(F.UNIFORMS)
    assert float(columns["grav_x"]) == 0.0 and float(columns["grav_y"]) == -F.G and float(columns["lid"]) == F.LID_OFF
    default = F.run_frames(piece, columns, 700)
    assert _digest(default) == PRE_GRAVITY_DIGEST_700                     # the default is bit for bit the law before the change
    tilted = dict(columns, grav_x=np.float64(0.5), grav_y=np.float64(-0.8660254))
    other = F.run_frames(piece, tilted, 700)                              # the same compiled piece: only the uniform changed
    assert _digest(other) != _digest(default)
    live = (np.abs(default["x"]) + np.abs(default["y"])) > 0
    assert float(other["x"][live].mean()) > float(default["x"][live].mean()) + 0.05          # the pile slid toward +x
    assert np.isfinite(other["x"]).all() and float(other["n_nan"][0]) == 0 and float(other["n_out"][0]) == 0


def test_the_lid_keeps_grains_in_the_basin_when_gravity_points_up(native_piece):
    piece, columns = native_piece
    geo = F.basin(F.N_SMALL)
    top = geo["rows"] * F.CELL
    up = dict(columns, grav_x=np.float64(0.0), grav_y=np.float64(F.G))
    free = F.run_frames(piece, dict(up), 700)
    closed = F.run_frames(piece, dict(up, lid=np.float64(top)), 700)
    placed = lambda f: (np.abs(f["x"]) + np.abs(f["y"])) > 0
    assert float(free["y"][placed(free)].max()) > top + 0.5               # open top: the spray leaves
    assert float(closed["y"][placed(closed)].max()) <= top + 1e-3         # closed: nothing passes the lid
    assert float(closed["n_nan"][0]) == 0


def test_the_server_declares_the_run_modes_and_the_tilt_inputs():
    game = F.FountainGame(size="small")
    state = game.state()
    assert state["compute"]["run"] == "continuous"                         # the default run never stops
    ids = {item["id"]: item for item in state["controls_ui"]}
    for name in ("rebirth", "rec_replay", "bounce", "continue"):
        assert ids[name]["kind"] == "button" and ids[name]["local"] == name.replace("rec_", "")
    assert ids["tilt"]["kind"] == "toggle" and ids["tilt"]["local"] == "tilt" and ids["tilt"]["value"] is True
    assert ids["run"]["value"] == "continuous" and [o[0] for o in ids["run"]["options"]] == ["continuous", "settle"]
    assert [o[0] for o in ids["speed"]["options"]] == [1, 2, 4, 0]          # 0: follow the newest frame
    tilt = state["compute"]["tilt"]                                          # what the page writes while the sensor drives gravity
    geo = F.basin(256)
    assert tilt["lid"] > geo["nozzle_y"] + 1.35 and tilt["rows"] * F.CELL >= tilt["lid"]
    before = state["compute"]["epoch"]
    game.do({"type": "fountain_run", "run": "settle"})
    assert game.state()["compute"]["run"] == "settle" and game.state()["compute"]["epoch"] == before + 1
    with pytest.raises(ValueError):
        game.do({"type": "fountain_run", "run": "sideways"})


JS_RECORDING = """
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
const [dir] = process.argv.slice(2);
const {Recording, Playback} = await import(pathToFileURL(`${dir}/recording.js`).href);

// ---- the window: a memory cap in bytes, oldest frame out first
const rec = new Recording({snapEvery: 6, frameBytes: 100, capBytes: 1000, keyBytes: 1e9, keyEvery: 60});
assert.equal(rec.capFrames, 10);
const evicted = [];
for (let k = 1; k <= 25; k++) evicted.push(...rec.push(k, {k}).frames.map((f) => f.k));
assert.deepEqual([rec.lo, rec.hi, rec.length], [16, 25, 10]);
assert.deepEqual(evicted, Array.from({length: 15}, (_, i) => i + 1));
assert.equal(rec.get(15), null); assert.equal(rec.get(16).k, 16); assert.equal(rec.get(25).k, 25); assert.equal(rec.get(26), null);
assert.throws(() => rec.push(27, {}));                 // frames come in order

// ---- playback: live follows at the pace given and never passes the newest; turbo is the newest
const play = new Playback();
play.set('live', rec); play.jump(16);
assert.equal(play.advance(3.5, rec), 19);
assert.equal(play.advance(100, rec), 25);
assert.equal(play.advance(0, rec, {turbo: true}), 25);
play.jump(16); assert.equal(play.advance(1, rec, {maxLead: 2}), 23);   // too far behind: skips ahead to within maxLead
// replay: the window's first frame to its last, then holds
play.set('replay', rec);
assert.equal(play.advance(0, rec), 16);
assert.equal(play.advance(4, rec), 20);
assert.equal(play.advance(100, rec), 25); assert.equal(play.ended, true); assert.equal(play.advance(5, rec), 25);
// bounce: forward then backward, for ever, inside the window
play.set('bounce', rec);
const seen = [];
for (let i = 0; i < 40; i++) seen.push(play.advance(1, rec));
assert.deepEqual(seen.slice(0, 11), [17, 18, 19, 20, 21, 22, 23, 24, 25, 24, 23]);
assert.ok(seen.every((k) => k >= 16 && k <= 25));
assert.ok(seen.includes(16) && seen.slice(18).includes(25));              // it turns at both ends
assert.equal(play.advance(37.5, rec) >= 16, true);                        // a long step still reflects into the window

// ---- continue is EXACT: a toy law with a runtime input, keyframes, and the input log
const SNAP = 6, KEY = 60;
const law = (s, g) => s * 1.0001 + g;                    // deterministic, input-dependent
const run = new Recording({snapEvery: SNAP, frameBytes: 8, capBytes: 8 * 200, keyBytes: 1e9, keyEvery: KEY});
let s = 1.5, g = -1, step = 0;
run.setInputs(0, {g});
const truth = new Map(), keys = new Map();
const inputAt = (st) => (st === 70 ? 0.3 : st === 133 ? -0.2 : st === 500 ? 0.9 : null);
for (let k = 1; k <= 160; k++) {
  const target = k * SNAP;
  for (; step < target; step++) {
    const change = inputAt(step);
    if (change !== null) { g = change; run.setInputs(step, {g}); }
    s = law(s, g);
  }
  truth.set(k, s);
  run.push(k, s);
  if (target % KEY === 0) { keys.set(target, s); run.addKey(target, s, 8); }
}
assert.equal(run.length, 160);
const replay = (f) => {
  const plan = run.plan(f);
  let state = plan.key ? plan.key.snap : 1.5, gg = plan.initial ? plan.initial.g : -1;
  for (const op of plan.ops) { if (op.set) gg = op.set.g; else for (let i = 0; i < op.steps; i++) state = law(state, gg); }
  return {state, plan};
};
for (const f of [1, 2, 11, 12, 23, 24, 60, 99, 100, 158, 160]) {
  const {state, plan} = replay(f);
  assert.equal(state, truth.get(f), `frame ${f}`);                         // bit for bit
  assert.ok(plan.start <= plan.target);
  if (f >= 10) assert.ok(plan.key !== null && plan.start % KEY === 0);      // from a keyframe, not from the start
}
// branching drops the future (frames, keys, inputs) and leaves the rest replayable
const dropped = run.truncateAfter(100);
assert.equal(run.hi, 100); assert.equal(dropped.frames.length, 60);
assert.ok(run.keys.every((k) => k.step <= 600));
assert.equal(replay(100).state, truth.get(100));

// ---- a long run under the cap: the window slides, keyframes older than the window's start are dropped, replay still exact
const slide = new Recording({snapEvery: SNAP, frameBytes: 8, capBytes: 8 * 30, keyBytes: 1e9, keyEvery: KEY});
let t = 1.5, gg = -1, st = 0; slide.setInputs(0, {g: gg});
const truth2 = new Map();
for (let k = 1; k <= 400; k++) {
  for (; st < k * SNAP; st++) { if (st % 97 === 0) { gg = ((st / 97) % 5) - 2; slide.setInputs(st, {g: gg}); } t = law(t, gg); }
  truth2.set(k, t); slide.push(k, t);
  if ((k * SNAP) % KEY === 0) slide.addKey(k * SNAP, t, 8);
}
assert.deepEqual([slide.lo, slide.hi], [371, 400]);
assert.ok(slide.keys.length <= 5 && slide.keys[0].step <= slide.lo * SNAP);
assert.ok(slide.inputs.length < 400 / 97 * 2);
for (const f of [371, 380, 400]) {
  const plan = slide.plan(f);
  let state = plan.key ? plan.key.snap : 1.5, g2 = plan.initial ? plan.initial.g : -1;
  for (const op of plan.ops) { if (op.set) g2 = op.set.g; else for (let i = 0; i < op.steps; i++) state = law(state, g2); }
  assert.equal(state, truth2.get(f), `slid frame ${f}`);
}
// thinning: keyframes over their byte budget keep the oldest and the newest
const thin = new Recording({snapEvery: SNAP, frameBytes: 8, capBytes: 8 * 1000, keyBytes: 50, keyEvery: KEY});
for (let k = 1; k <= 100; k++) { thin.push(k, k); if ((k * SNAP) % KEY === 0) thin.addKey(k * SNAP, k, 8); }
assert.ok(thin.keyBytesUsed <= 50 + 8 && thin.keys[0].step === 60 && thin.keys.at(-1).step === 600);
console.log('ok');
"""

JS_GRAVITY = """
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
const [dir] = process.argv.slice(2);
const {gravityInScreen, gravityFromAcceleration, GravitySensor} = await import(pathToFileURL(`${dir}/gravity.js`).href);
const near = (a, b) => assert.ok(Math.abs(a - b) < 1e-9, `${a} vs ${b}`);
let g = gravityInScreen(90, 0, 0); near(g[0], 0); near(g[1], -1);              // held upright: straight down the screen
g = gravityInScreen(0, 0, 0); near(g[0], 0); near(g[1], 0);                     // lying flat: no pull in the screen's plane
g = gravityInScreen(60, 0, 0); near(g[0], 0); near(g[1], -Math.sin(Math.PI / 3));      // tipped back 30 degrees
g = gravityInScreen(0, 30, 0); near(g[0], 0.5); near(g[1], 0);                  // flat, right edge down
g = gravityInScreen(0, -90, 90); near(g[0], 0); near(g[1], -1);                 // landscape (screen angle 90): still down the screen
g = gravityFromAcceleration(0, 9.81, 0, 0, true); near(g[0], 0); near(g[1], -1);    // Android upright reads +y; gravity is the other way
g = gravityFromAcceleration(0, -9.81, 0, 0, false); near(g[0], 0); near(g[1], -1); // iOS reads the opposite

// the sensor: nothing listens until the first tap on iOS; the tap asks permission; the off switch releases the gravity
const listeners = {}, calls = [];
const win = {DeviceOrientationEvent: {requestPermission: async () => 'granted'}, screen: {orientation: {angle: 0}},
             addEventListener: (type, fn) => { listeners[type] = fn; }};
const states = [];
const sensor = new GravitySensor({win, onGravity: (x, y) => calls.push([x, y]), onState: (s) => states.push(s)});
sensor.start();
assert.deepEqual(Object.keys(listeners), []);                                   // iOS: no events before a gesture
await sensor.grant();
assert.ok(listeners.deviceorientation); assert.equal(sensor.enabled, true);
listeners.deviceorientation({beta: 90, gamma: 0});
assert.equal(calls.length, 1); near(calls[0][1], -1); assert.equal(sensor.state, 'on');
listeners.deviceorientation({beta: null, gamma: null});                          // a desktop's empty event is ignored
assert.equal(calls.length, 1);
sensor.enabled = false;
assert.deepEqual(calls.at(-1), [null, null]); assert.equal(sensor.state, 'off');
const denied = new GravitySensor({win: {...win, DeviceOrientationEvent: {requestPermission: async () => 'denied'}, addEventListener: () => { throw new Error('must not listen'); }}, onGravity: () => {}});
assert.equal(await denied.grant(), 'denied');
// elsewhere (no permission API) it listens at once and is on by default
const plain = {addEventListener: (type, fn) => { listeners['plain' + type] = fn; }, screen: {orientation: {angle: 0}}};
const other = new GravitySensor({win: plain, onGravity: (x, y) => calls.push([x, y])});
assert.equal(other.enabled, true); other.start(); assert.ok(listeners.plaindeviceorientation);
console.log('ok');
"""


def _run_node(tmp_path, script: str, *modules: str) -> str:
    for name in modules:
        shutil.copyfile(ROOT / "examples" / "meowdoku" / "web" / name, tmp_path / name)
    (tmp_path / "run.mjs").write_text(script, encoding="utf-8")
    done = subprocess.run(["node", str(tmp_path / "run.mjs"), str(tmp_path)], capture_output=True, text=True)
    assert done.returncode == 0, (done.stdout + done.stderr)[-2500:]
    return done.stdout


@pytest.mark.skipif(shutil.which("node") is None, reason="node runs the page's recording logic")
def test_the_recording_window_playback_and_continue_are_exact(tmp_path):
    """The page's recording (a memory-capped sliding window with keyframes and an input log), its Replay / Bounce /
    live cursor, and the Continue plan: a frame's state rebuilt from a keyframe plus the logged inputs equals the run."""
    assert _run_node(tmp_path, JS_RECORDING, "recording.js").strip() == "ok"


@pytest.mark.skipif(shutil.which("node") is None, reason="node runs the page's gravity code")
def test_the_phone_orientation_gives_the_screen_plane_gravity_and_the_sensor_asks_on_a_tap(tmp_path):
    assert _run_node(tmp_path, JS_GRAVITY, "gravity.js").strip() == "ok"
