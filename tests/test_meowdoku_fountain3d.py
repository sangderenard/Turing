"""The fountain: a 3-D (slab chamber) DEM granular law, a compiler-chosen and compiler-generated compute program,
no per-id tables, runtime tunables, and deterministic image verification.

The state is 3-D (x y z, v, a spin vector): the DEFAULT chamber is a thin slab (about one grain diameter deep), whose pile is
the 2-D one of before (no bit identity with that law is required: the layout changed)."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from meowdoku import fountain3d as F  # noqa: E402
from meowdoku import fountain3d_build as B  # noqa: E402
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
    assert float(np.sqrt(final["vx"] ** 2 + final["vy"] ** 2 + final["vz"] ** 2).max()) < 0.2          # at rest
    assert float(np.abs(final["z"]).max()) <= float(columns["dh"])                                   # inside the plexiglass
    x, y, z, r = final["x"], final["y"], final["z"], final["rad"]
    gap = np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2 + (z[:, None] - z[None, :]) ** 2) - (r[:, None] + r[None, :])
    np.fill_diagonal(gap, 1.0)
    assert float(gap.min()) > -0.6 * float(r.max())                 # no two spheres overlap deeply: a volume, not a sheet


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
    assert all(np.array_equal(final[k], phase1[k]) for k in ("x", "y", "z"))
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
    assert compiled.arena_elements * 4 / pad < 205                  # 51 slots, 204 B/id (the orbeez radius is one state column: 196 before)
    # the arena is split across bindings of at most 256 MiB, none of the single-buffer limit's business
    assert compiled.bindings > 1 and compiled.bindings + 1 <= 8       # the binding COUNT budget holds (7 stacked lanes + the head)
    assert compiled.bindings == -(-compiled.slots // compiled.slots_per_binding)
    assert compiled.slots_per_binding * pad * 4 <= 320 * 1024 * 1024    # 51 slots at 10M: 8 per lane, 320 MB (280 MB before the orbeez radius)
    assert compiled.slots == F.ARENA_SLOTS and F.binding_bytes(pad) == compiled.slots_per_binding * pad * 4   # the volume adds no per-id column; the orbeez radius adds one state column (2 slots)


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
    shutil.copyfile(ROOT / "examples" / "meowdoku" / "web" / "particle_styles.js", tmp_path / "particle_styles.js")
    shutil.copyfile(ROOT / "examples" / "meowdoku" / "web" / "noise_filter.js", tmp_path / "noise_filter.js")
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
    zero, one = t * 0.0, t * 0.0 + 1.0
    # a contact with the normal along y and a slip of t along x: the friction force is -mu * fn along x once saturated
    force, torque = F._contact(t * 0.0 + 0.01, one, (zero, one, zero), (t, zero, zero), (zero, zero, zero), (zero, zero, zero),
                               one, one, zero + 0.3, zero + 0.7, zero, zero + 0.01, zero + 0.005, zero + 0.01, zero, zero)
    fx, fy = np.asarray(force[0]), np.asarray(force[1])
    assert all(np.isfinite(np.asarray(v)).all() for v in (*force, *torque))
    assert abs(float(fx[1]) + 0.3 * float(fy[1])) < 1e-6      # saturated: |ft| = mu * fn
    columns = F.initial_columns(256, mix=1.0)
    columns["age"][2:4] = 1.0
    columns["x"][2:4], columns["y"][2:4] = 0.05, 0.5
    out = _one_step(columns)
    assert np.isfinite(out["vx"]).all() and np.isfinite(out["vy"]).all() and np.isfinite(out["vz"]).all()
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


def _mass(rad):
    species = {round(s[1], 6): s for s in F.SPECIES}
    return np.array([species[round(float(r), 6)][2] * 4.0 / 3.0 * np.pi * r ** 3 for r in rad])


def _system_energy(x, y, z, vx, vy, vz, w2, rad, live):
    """Kinetic + rotational + gravitational + the contacts' elastic energy of the live grains (the springs' ``kn delta^2 / 2``)."""
    mass = _mass(rad)
    idx = np.where(live)[0]
    m, r, px, py, pz = mass[idx], rad[idx], x[idx], y[idx], z[idx]
    total = float((0.5 * m * (vx[idx] ** 2 + vy[idx] ** 2 + vz[idx] ** 2) + 0.2 * m * r * r * w2[idx] + m * F.G * py).sum())
    d = np.sqrt((px[:, None] - px[None, :]) ** 2 + (py[:, None] - py[None, :]) ** 2 + (pz[:, None] - pz[None, :]) ** 2)
    delta = np.clip(r[:, None] + r[None, :] - d, 0.0, None)
    np.fill_diagonal(delta, 0.0)
    meff = m[:, None] * m[None, :] / (m[:, None] + m[None, :])
    total += float(0.25 * (F.OMEGA ** 2 * meff * delta ** 2).sum())                    # each pair once (linear spring: hertz = 0)
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
    x, y, z, vx, vy, vz, wx, wy, wz = (np.array(run["kept"][k]) for k in ("x", "y", "z", "vx", "vy", "vz", "wx", "wy", "wz"))      # (steps, n)
    w2 = wx ** 2 + wy ** 2 + wz ** 2
    rad = run["rad"]
    live = ((np.abs(x) + np.abs(y)) >= 1e-30) & (vy > F.ARMED * 0.5)               # launched: not parked, not armed at the slot
    placed_at = np.where(live.any(axis=0), live.argmax(axis=0), -1)
    assert (placed_at >= 0).all()                                                    # every id was emitted
    deepest = 0.0
    for j in range(n):
        k = placed_at[j]
        others = np.where(live[k])[0]
        others = others[others != j]
        gap = np.sqrt((x[k, others] - x[k, j]) ** 2 + (y[k, others] - y[k, j]) ** 2 + (z[k, others] - z[k, j]) ** 2) - (rad[others] + rad[j])
        deepest = max(deepest, float(-gap.min()) if gap.size else 0.0)
    assert deepest <= 3e-3, deepest                       # placed overlapping nothing (4k, measured: 2 of 4096 spawns by <= 1.8 mm; the baseline: half of them, up to 25 mm)
    mass = _mass(rad)
    launch = 0.5 * mass * (vx[placed_at, np.arange(n)] ** 2 + vy[placed_at, np.arange(n)] ** 2 + vz[placed_at, np.arange(n)] ** 2) \
        + mass * F.G * y[placed_at, np.arange(n)]
    worst = -np.inf
    for k in range(0, run["steps"], 30):
        e_in = float(launch[placed_at <= k].sum())
        worst = max(worst, _system_energy(x[k], y[k], z[k], vx[k], vy[k], vz[k], w2[k], rad, live[k]) - e_in)
    assert worst < 0.03 * float(launch.sum()), worst      # semi-implicit Euler's own drift, ~1% (4k, measured)
    ceiling = (y[placed_at, np.arange(n)] + vy[placed_at, np.arange(n)] ** 2 / (2.0 * F.G)).max()
    assert float(np.where(live, y, -np.inf).max()) <= 1.06 * ceiling           # the highest grain is within 6% of the highest launch ceiling (measured: 4% at 256, 2% at 4k; before: 42% over)


def test_an_id_waits_for_a_clear_nozzle_slot_and_goes_the_step_it_clears(native_piece):
    piece, columns = native_piece
    n = F.N_SMALL
    u = {k: float(v) for k, v in columns.items() if k in F.UNIFORMS}
    slot = [float(np.asarray(v)[0]) for v in F._nozzle(
        AbstractTensor.get_tensor(np.zeros(1)), (u["nozzle_x"], u["nozzle_y"], u["nozzle_z"]), (u["e1x"], u["e1y"], u["e1z"]),
        (u["e2x"], u["e2y"], u["e2z"]), (u["in_dx"], u["in_dy"], u["in_dz"]), u["bar_half"], u["bar_z"])]

    def after(state, steps=1):
        out = F.run_frames(piece, {**columns, **state}, steps)
        return {k: np.array(out[k]) for k in F.STATE}

    armed = after({k: np.array(columns[k]) for k in F.STATE})
    assert armed["vy"][0] == F.ARMED and abs(armed["x"][0] - slot[0]) < 1e-6 and abs(armed["y"][0] - slot[1]) < 1e-6   # due: armed AT its slot
    assert abs(armed["z"][0] - slot[2]) < 1e-6
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
    vx, vy, vz, wx, wy, wz = (run["kept"][k][last] for k in ("vx", "vy", "vz", "wx", "wy", "wz"))
    rad = run["rad"]
    mass = _mass(rad)
    ke = float((0.5 * mass * (vx ** 2 + vy ** 2 + vz ** 2) + 0.5 * (0.4 * mass * rad ** 2) * (wx ** 2 + wy ** 2 + wz ** 2)).sum())
    assert abs(run["diag"]["ke"][last] - ke) < 1e-7 * max(ke, 1e-12)      # the species densities are float32 uniforms (1.6 -> 1.60000002): 1e-8 apart from the table's
    assert abs(run["diag"]["vmax2"][last] - float((vx ** 2 + vy ** 2 + vz ** 2).max())) < 1e-12


def test_a_mirrored_post_emission_state_evolves_mirrored(emission_run, native_piece):
    """x -> -x, vx -> -vx, spin (a pseudovector: wy, wz flip) with the same ids: the evolution is the mirror image within float noise.  A
    neighbour walk that drops contacts on one side (the right-hand cells of a row run) would break it."""
    piece, columns = native_piece
    run = emission_run
    state = {k: np.array(run["kept"][k][-1]) for k in F.STATE}
    mirrored = dict(state)
    mirrored["x"], mirrored["vx"], mirrored["wy"], mirrored["wz"] = -state["x"], -state["vx"], -state["wy"], -state["wz"]
    steps = 80
    a, da, _, _ = _series(piece, {**columns, **state}, steps)
    b, db, _, _ = _series(piece, {**columns, **mirrored}, steps)
    assert da["n_drop"].max() == 0 and db["n_drop"].max() == 0
    assert float(np.abs(a["x"] + b["x"]).max()) < 1e-9 and float(np.abs(a["y"] - b["y"]).max()) < 1e-9
    assert float(np.abs(a["vx"] + b["vx"]).max()) < 1e-8 and float(np.abs(a["wz"] + b["wz"]).max()) < 1e-8


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
    shutil.copyfile(ROOT / "examples" / "meowdoku" / "web" / "particle_styles.js", tmp_path / "particle_styles.js")
    shutil.copyfile(ROOT / "examples" / "meowdoku" / "web" / "noise_filter.js", tmp_path / "noise_filter.js")
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
def _digest(final) -> str:
    import hashlib
    return hashlib.sha256(b"".join(np.ascontiguousarray(final[k], dtype=np.float64).tobytes() for k in F.STATE + ("cr", "cg", "cb"))).hexdigest()


def test_gravity_is_a_runtime_input_the_default_is_the_old_law_and_a_tilt_changes_it_without_a_rebuild(native_piece):
    piece, columns = native_piece
    assert {"grav_x", "grav_y", "grav_z", "lid", "rows"} <= set(F.RUNTIME_INPUTS) and set(F.RUNTIME_INPUTS) <= set(F.UNIFORMS)
    assert float(columns["grav_x"]) == 0.0 and float(columns["grav_y"]) == -F.G and float(columns["grav_z"]) == 0.0
    assert float(columns["lid"]) == F.LID_OFF
    default = F.run_frames(piece, columns, 700)
    assert F.run_frames(piece, columns, 700)["x"].tolist() == default["x"].tolist()          # deterministic
    tilted = dict(columns, grav_x=np.float64(0.5), grav_y=np.float64(-0.8660254))
    other = F.run_frames(piece, tilted, 700)                              # the same compiled piece: only the uniform changed
    assert _digest(other) != _digest(default)
    live = (np.abs(default["x"]) + np.abs(default["y"])) > 0
    assert float(other["x"][live].mean()) > float(default["x"][live].mean()) + 0.05          # the pile slid toward +x
    assert np.isfinite(other["x"]).all() and float(other["n_nan"][0]) == 0 and float(other["n_out"][0]) == 0


def test_colour_rate_is_a_runtime_uniform_that_changes_only_the_colours(native_piece):
    piece, columns = native_piece
    assert "strata" in F.UNIFORMS and F.STRATA in F.STRATA_CHOICES
    strata = dict(columns, mode=np.float64(2.0))
    slow = F.run_frames(piece, dict(strata, strata=np.float64(1.0)), 400)
    fast = F.run_frames(piece, dict(strata, strata=np.float64(4.0)), 400)   # the same compiled piece
    for k in F.STATE:                                                       # the physics is untouched
        assert np.array_equal(slow[k], fast[k]), k
    live = (np.abs(slow["x"]) + np.abs(slow["y"])) > 0
    colour = lambda f: np.stack([f["cr"], f["cg"], f["cb"]], axis=1)[live]
    assert not np.array_equal(colour(slow), colour(fast))                   # the hue turns at the chosen rate
    # slower = fewer colour changes between neighbours in emission order (wider bands in place)
    order = np.argsort(columns["pid"][live]) if columns["pid"].shape == live.shape else None
    jumps = lambda c: float(np.abs(np.diff(c[order], axis=0)).sum())
    assert jumps(colour(slow)) < jumps(colour(fast))


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
    assert ids["orbit"]["kind"] == "toggle" and ids["orbit"]["local"] == "orbit" and ids["orbit"]["value"] is False
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
const {gravityInScreen, gravityVector, gravityFromAcceleration, GravitySensor, layoutTransform} = await import(pathToFileURL(`${dir}/gravity.js`).href);
const near = (a, b) => assert.ok(Math.abs(a - b) < 1e-9, `${a} vs ${b}`);
let g = gravityInScreen(90, 0, 0); near(g[0], 0); near(g[1], -1);              // held upright: straight down the screen
g = gravityInScreen(0, 0, 0); near(g[0], 0); near(g[1], 0);                     // lying flat: no pull in the screen's plane
g = gravityInScreen(60, 0, 0); near(g[0], 0); near(g[1], -Math.sin(Math.PI / 3));      // tipped back 30 degrees
g = gravityInScreen(0, 30, 0); near(g[0], 0.5); near(g[1], 0);                  // flat, right edge down
g = gravityInScreen(0, -90, 90); near(g[0], 0); near(g[1], -1);                 // landscape (screen angle 90): still down the screen
g = gravityVector(0, 0, 0); near(g[0], 0); near(g[1], 0); near(g[2], -1);           // lying flat face up: the pull is INTO the slab (against the back wall)
g = gravityVector(90, 0, 0); near(g[1], -1); near(g[2], 0);
g = gravityFromAcceleration(0, 9.81, 0, 0, true); near(g[0], 0); near(g[1], -1);    // Android upright reads +y; gravity is the other way
g = gravityFromAcceleration(0, -9.81, 0, 0, false); near(g[0], 0); near(g[1], -1); // iOS reads the opposite

// the sensor: nothing listens until the first tap on iOS; the tap asks permission; the off switch releases the gravity
const listeners = {}, calls = [];
const win = {DeviceOrientationEvent: {requestPermission: async () => 'granted'}, screen: {orientation: {angle: 0}},
             addEventListener: (type, fn) => { listeners[type] = fn; }};
const states = [];
const sensor = new GravitySensor({win, onGravity: (x, y, z) => calls.push([x, y]), onState: (s) => states.push(s)});
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
// iOS: BOTH permission requests are made inside the tap.  A request issued after an await has lost the tap's activation
// and throws (it used to fail the whole grant, so the switch had to be cycled before gravity followed the phone).
let active = true;
const strict = (answer) => () => { if (!active) throw new Error('NotAllowedError: no user activation'); queueMicrotask(() => { active = false; }); return Promise.resolve(answer); };
const L2 = {};
const ios = new GravitySensor({win: {DeviceOrientationEvent: {requestPermission: strict('granted')}, DeviceMotionEvent: {requestPermission: strict('granted')},
                                     screen: {orientation: {angle: 0}}, addEventListener: (type, fn) => { L2[type] = fn; }}, onGravity: () => {}});
assert.equal(await ios.grant(), 'waiting'); assert.ok(L2.deviceorientation);
// the layout lock: a turn of the screen is undone, and gravity is read in the locked frame
assert.equal(layoutTransform(0, 0, 390, 844), null);
assert.equal(layoutTransform(0, 90, 390, 844), 'translate(0px, 390px) rotate(-90deg)');
assert.equal(layoutTransform(0, 270, 390, 844), 'translate(844px, 0px) rotate(90deg)');
assert.equal(layoutTransform(90, 0, 844, 390), 'translate(390px, 0px) rotate(90deg)');
const held = []; const L3 = {};
const locked = new GravitySensor({win: {screen: {orientation: {angle: 90}}, addEventListener: (type, fn) => { L3[type] = fn; }}, onGravity: (x, y) => held.push([x, y])});
locked.lockedAngle = 0; locked.start();
L3.deviceorientation({beta: 90, gamma: 0});            // upright in the locked (portrait) frame, whatever the screen says
near(held[0][0], 0); near(held[0][1], -1);
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


JS_STYLES = """
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
const [dir] = process.argv.slice(2);
const {STYLES, STYLE_IDS, DEFAULT_STYLE, chooseStyle, styleAvailability} = await import(pathToFileURL(`${dir}/particle_styles.js`).href);
const full = {floatRender: 'float', maxPointSize: 256}, half = {floatRender: 'half', maxPointSize: 256}, none = {floatRender: null, maxPointSize: 256};
assert.equal(DEFAULT_STYLE, 'points'); assert.equal(STYLE_IDS[0], 'points'); assert.equal(new Set(STYLE_IDS).size, STYLE_IDS.length);
// the default and unknown names are always points; an ordinary device runs every style
assert.deepEqual(chooseStyle('points', null), {style: 'points', requested: 'points', reason: ''});
assert.equal(chooseStyle('no-such-style', full).style, 'points');
for (const id of STYLE_IDS) assert.equal(chooseStyle(id, full).style, id);
// no float render targets: the float styles fall back to points and say why; sand loses only its ambient occlusion; stamp and glow need none
for (const id of ['liquid', 'heat']) {
  const c = chooseStyle(id, none);
  assert.equal(c.style, 'points'); assert.equal(c.requested, id); assert.match(c.reason, /float/);
}
for (const id of ['sand', 'glow', 'stamp']) assert.equal(chooseStyle(id, none).style, id);
assert.match(chooseStyle('sand', none).reason, /ambient/);
// half float runs them, with the precision noted
assert.equal(chooseStyle('liquid', half).style, 'liquid'); assert.match(chooseStyle('liquid', half).reason, /half/);
// tiny point sprites (some phones): every sprite style falls back
assert.equal(chooseStyle('glow', {floatRender: 'float', maxPointSize: 1}).style, 'points');
// a style that failed at run time is not retried
assert.equal(chooseStyle('liquid', full, {liquid: 'incomplete target'}).style, 'points');
assert.match(chooseStyle('liquid', full, {liquid: 'incomplete target'}).reason, /incomplete target/);
assert.equal(styleAvailability('liquid', null).ok, false);
console.log(JSON.stringify(STYLES.map((s) => [s.id, s.label])));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node runs the page's particle style registry")
def test_the_particle_styles_registry_picks_a_style_the_device_can_run_and_matches_the_server(tmp_path):
    """Render-only looks (web/particle_styles.js): the registry, what a device can run, the fallback to points -- and the
    server's `look` control declares exactly the page's styles, in the page's order."""
    import json
    listed = json.loads(_run_node(tmp_path, JS_STYLES, "particle_styles.js").strip().splitlines()[-1])
    ids = {item["id"]: item for item in F.FountainGame(size="small").state()["controls_ui"]}
    look = ids["pstyle"]
    assert look["kind"] == "select" and look["local"] == "particle_style" and look["value"] == "points"
    assert look["options"] == listed


JS_NOISE = """
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
const [dir] = process.argv.slice(2);
const N = await import(pathToFileURL(`${dir}/noise_filter.js`).href);
const near = (a, b, tol = 0.02) => assert.ok(Math.abs(a - b) <= tol, `${a} vs ${b}`);
// variance ladders (fine -> coarse) of known spectra: band energy v_k = s * 2^(k (beta - 2))
const raw = (beta, s = 0.01) => Array.from({length: N.GAPS}, (_, k) => s * 2 ** (k * (beta - 2)));
const ladder = raw;
for (const beta of [0, 0.5, 1, 1.5, 2]) near(N.fitBeta(ladder(beta)), beta);
near(N.fitBeta(ladder(3)), 3); near(N.fitBeta(ladder(4)), 4, 1e-3);                 // an edge reads 3, a ramp 4 (clamped)
assert.equal(N.fitBeta(ladder(5)), 4); assert.equal(N.fitBeta(ladder(-3)), -1);       // clamped both ways
near(N.fitBeta(ladder(1, 1e-3)), 1);                                                     // scale-free in the amplitude
// a lone band: all the variance at the finest scale reads as white or steeper, at the coarsest as steep; neither is NaN
const lone = (k) => [0, 0, 0, 0, 0].map((_, i) => (i === k ? 0.01 : 0));
assert.ok(N.fitBeta(lone(0)) < N.WHITE_PINK); assert.ok(N.fitBeta(lone(4)) > N.PINK_BROWN);
// the noise share: speckle high, coarse structure low
near(N.noiseShare(raw(0)), 0.984, 0.002); near(N.noiseShare(raw(1)), 0.903, 0.002); near(N.noiseShare(raw(2)), 0.6, 0.002);
near(N.noiseShare(raw(3)), 0.226, 0.002); near(N.noiseShare(raw(4)), 0.062, 0.002);
assert.equal(N.noiseShare([0, 0, 0, 0, 0]), 0);
// the classes
const c = (v, lum = 0.2) => N.classify(v, lum).cls;
assert.equal(c(ladder(0)), 'white'); assert.equal(c(ladder(0), 0.95), 'grey');
assert.equal(c(ladder(1)), 'pink'); assert.equal(c(ladder(1), 0.95), 'pink');            // grey is a white-noise subclass
assert.equal(c(ladder(2)), 'brown'); assert.equal(c(ladder(3)), 'brown');
assert.equal(c(ladder(0, 1e-6)), 'solid'); assert.equal(c([0, 0, 0, 0, 0]), 'solid');
// the boundaries sit where the constants say
assert.equal(c(ladder(N.WHITE_PINK - 0.05)), 'white'); assert.equal(c(ladder(N.WHITE_PINK + 0.05)), 'pink');
assert.equal(c(ladder(N.PINK_BROWN - 0.05)), 'pink'); assert.equal(c(ladder(N.PINK_BROWN + 0.05)), 'brown');
// noise vs structure: white speckle is replaced, an edge or a strata ramp is kept, at the default threshold
assert.equal(N.classify(ladder(0), 0.2).noise, true); assert.equal(N.classify(ladder(3), 0.2).noise, false); assert.equal(N.classify(ladder(4), 0.2).noise, false);
assert.ok(N.noiseWeight(0.98, 0.5) > 0.99 && N.noiseWeight(0.23, 0.5) < 0.01);
assert.ok(N.noiseWeight(0.6, 0.3) > N.noiseWeight(0.6, 0.7));                              // a higher threshold keeps more
assert.ok(N.solidWeight(0) === 0 && N.solidWeight(1) === 1);
// the palette: the fixed colour at beta 0 / 1 / 2, mixed between, grey replacing white as the luminance share rises
const eq = (a, b) => a.forEach((x, i) => near(x, b[i], 1e-9));
eq(N.paletteFor(0, 0), N.PALETTE.white); eq(N.paletteFor(0, 1), N.PALETTE.grey); eq(N.paletteFor(1, 0.3), N.PALETTE.pink); eq(N.paletteFor(2, 0.3), N.PALETTE.brown);
eq(N.paletteFor(-1, 0), N.PALETTE.white); eq(N.paletteFor(4, 0), N.PALETTE.brown);
eq(N.paletteFor(1.5, 0), N.PALETTE.pink.map((x, i) => (x + N.PALETTE.brown[i]) / 2));
// replacement: mean mode blends only noise; spectral mode also tints solid regions with the solid colour
const mean = {mode: 'mean', strength: 0.8, threshold: 0.5}, spec = {mode: 'spectral', strength: 0.8, threshold: 0.5};
near(N.replacement(ladder(0), 0.2, mean).alpha, 0.8); assert.equal(N.replacement(ladder(0), 0.2, mean).rgb, null);
near(N.replacement(ladder(3), 0.2, mean).alpha, 0); near(N.replacement(ladder(0, 1e-7), 0.2, mean).alpha, 0);
near(N.replacement(ladder(3), 0.2, spec).alpha, 0);
near(N.replacement(ladder(0, 1e-7), 0.2, spec).alpha, 0.8); eq(N.replacement(ladder(0, 1e-7), 0.2, spec).rgb, N.PALETTE.solid);
N.replacement(ladder(1), 0.2, spec).rgb.forEach((x, i) => near(x, N.PALETTE.pink[i], 0.01));
// availability: off with a reason without float or half-float render targets
const full = {floatRender: 'float', maxPointSize: 256}, half = {floatRender: 'half', maxPointSize: 256}, none = {floatRender: null, maxPointSize: 256};
assert.deepEqual(N.chooseNoise(false, none), {on: false, reason: ''});
assert.equal(N.chooseNoise(true, full).on, true); assert.equal(N.chooseNoise(true, half).on, true); assert.match(N.chooseNoise(true, half).reason, /half/);
assert.equal(N.chooseNoise(true, none).on, false); assert.match(N.chooseNoise(true, none).reason, /float/);
assert.equal(N.chooseNoise(true, null).on, false);
assert.equal(N.chooseNoise(true, full, 'incomplete target').on, false); assert.match(N.chooseNoise(true, full, 'incomplete target').reason, /incomplete target/);
console.log(JSON.stringify(N.NOISE_MODES));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node runs the page's noise filter logic")
def test_the_noise_filters_slope_fit_and_classes_read_synthetic_variance_ladders_and_match_the_server(tmp_path):
    """Render-only spectral noise filter (web/noise_filter.js): the log-variance slope fit reads white 0 / pink 1 / brown 2
    from band-energy ladders (edge 3, ramp 4), the classes and palette, the speckle-vs-structure weight, the fallback
    without float render targets -- and the server's controls declare the page's local noise controls and modes."""
    import json
    listed = json.loads(_run_node(tmp_path, JS_NOISE, "noise_filter.js", "particle_styles.js").strip().splitlines()[-1])
    ids = {item["id"]: item for item in F.FountainGame(size="small").state()["controls_ui"]}
    assert ids["nf"]["kind"] == "toggle" and ids["nf"]["local"] == "noise_on" and ids["nf"]["value"] is False
    assert ids["nfmode"]["local"] == "noise_mode" and ids["nfmode"]["options"] == listed and ids["nfmode"]["value"] == "mean"
    for key, local in (("nfstrength", "noise_strength"), ("nfthr", "noise_threshold")):
        assert ids[key]["kind"] == "range" and ids[key]["local"] == local and ids[key]["min"] < ids[key]["value"] < ids[key]["max"]


# ------------------------------------------------------------- 3-D chamber, tunables, fluids, polygon, inlet
def _step_count(piece, cols, steps):
    return F.run_frames(piece, cols, steps)


def _live(f):
    return (np.abs(f["x"]) + np.abs(f["y"]) + np.abs(f["z"])) > 0


def test_the_default_chamber_is_a_box_nine_grains_deep_and_the_pile_settles_in_depth(native_piece):
    """Depth is a runtime uniform; the default is about nine grain diameters, the inlet spans it, and the settled pile is spread through it
    (the z spread of the grains is many diameters), no grain leaves through the front or back glass, and every id is counted exactly."""
    piece, columns = native_piece
    dh = float(columns["dh"])
    assert 8.0 * 2.0 * 0.0105 <= 2.0 * dh <= 10.0 * 2.0 * 0.0105 + 1e-9 and float(columns["bar_z"]) > 0.0
    assert float(columns["lays"]) == np.ceil(2.0 * dh / float(columns["cell"]) - 1e-9) and float(columns["lays"]) >= 6
    out = F.run_frames(piece, columns, REST_STEPS)
    live = _live(out)
    diameter = 2.0 * float(out["rad"][live].mean())
    assert float(np.ptp(out["z"][live])) > 5.0 * diameter                              # the pile is deep: many diameters of z spread
    assert float(np.abs(out["z"][live]).max()) <= dh + 1e-6                            # nothing through the front or back glass
    n = F.N_SMALL
    assert float(out["n_nan"][0] + out["n_out"][0] + out["n_parked"][0] + out["n_live"][0]) == n
    assert float(out["n_nan"][0]) == 0 and float(out["n_out"][0]) == 0 and float(out["n_live"][0]) == n


def test_the_z_grid_is_part_of_the_cell_index_and_the_key_stays_exact_in_float32():
    for tag, (count, _pad) in F.SIZES.items():
        u = F.uniforms(n=count)
        cells = float(u["cols"]) * (float(u["rows"]) + F.ROWS_MARGIN) * float(u["lays"])
        assert cells < 2.0 ** 24, tag                                                   # the sort key is a float32 whole number
        assert float(u["lays"]) * float(u["cell"]) >= 2.0 * float(u["dh"]) - 1e-6, tag     # the layers cover the depth
    assert float(F.uniforms(n=4096)["lays"]) >= 6                                      # the default is a volume
    assert 2.0 * float(F.uniforms(n=10_000_000)["dh"]) < F.DEPTH_DEFAULT                 # 10M is capped honestly (its grid cannot index more layers)
    assert F.depth_cap(10_000_000) < F.DEPTH_DEFAULT <= F.depth_cap(F.SIZES["4M"][0])
    assert float(F.uniforms(n=256, params={"depth": F.DEPTH_MAX})["lays"]) > 50        # up to a box as deep as the chamber is wide
    game = F.FountainGame(size="10M")
    ids = {item["id"]: item for item in game.state()["controls_ui"]}
    assert ids["depth"]["max"] < F.DEPTH_DEFAULT and "320 MB" in dict(ids["size"]["options"])["10M"]    # the size control says what 10M needs
    assert F.binding_bytes(10_000_000) > F.BINDING_CLASS_LIMIT >= F.binding_bytes(F.SIZES["4M"][0])


def test_a_deeper_box_uses_its_depth_and_the_plexiglass_holds_it(native_piece):
    piece, _columns = native_piece
    cols = F.initial_columns(F.N_SMALL, mix=0.5, params={"depth": 0.4})
    assert float(cols["bar_z"]) > 0.0
    out = F.run_frames(piece, cols, 2500)
    live = _live(out)
    dh = float(cols["dh"])
    assert float(np.abs(out["z"][live]).max()) > 0.5 * dh and float(np.abs(out["z"][live]).max()) <= dh + 1e-6
    assert float(out["n_nan"][0]) == 0 and float(out["n_out"][0]) == 0


def test_a_phone_lying_flat_pulls_the_grains_into_the_slab_against_the_back_wall(native_piece):
    piece, _columns = native_piece
    cols = F.initial_columns(F.N_SMALL, mix=0.5, params={"depth": 0.1})
    cols.update(grav_x=np.float64(0.0), grav_y=np.float64(0.0), grav_z=np.float64(-F.G))
    out = F.run_frames(piece, cols, 3000)
    live = _live(out)
    dh = float(cols["dh"])
    assert float(out["z"][live].mean()) < -0.5 * dh                       # lying against the back wall (-z)
    assert float(np.abs(out["z"][live]).max()) <= dh + 1e-6


def test_the_polygon_outline_confines_the_pile(native_piece):
    piece, _columns = native_piece
    for shape in ("funnel", "hexagon", "ramp"):
        cols = F.initial_columns(F.N_SMALL, mix=0.5, shape=shape)
        geo = F.basin(F.N_SMALL)
        verts = F.shape_vertices(shape, geo)
        out = F.run_frames(piece, cols, 2500)
        live = _live(out)
        assert float(out["n_nan"][0]) == 0 and float(out["n_out"][0]) == 0, shape
        assert float(out["y"][live].min()) > -1e-3 and float(np.abs(out["x"][live]).max()) <= geo["half"] + 1e-3
    funnel = F.run_frames(piece, F.initial_columns(F.N_SMALL, mix=0.5, shape="funnel"), 3000)
    box = F.run_frames(piece, F.initial_columns(F.N_SMALL, mix=0.5), 3000)
    # the funnel narrows to the bottom: its lowest grains sit in a narrower span than the box's
    low = lambda f: float(np.ptp(f["x"][_live(f) & (f["y"] < 0.05)]))
    assert low(funnel) < low(box)


def test_the_inlet_is_a_tube_with_a_runtime_direction_and_position(native_piece):
    piece, _columns = native_piece
    spec = F.inlet_spec("side", F.basin(F.N_SMALL), F.resolve_params())
    d = np.array([spec["in_dx"], spec["in_dy"], spec["in_dz"]])
    e1 = np.array([spec["e1x"], spec["e1y"], spec["e1z"]])
    e2 = np.array([spec["e2x"], spec["e2y"], spec["e2z"]])
    assert abs(np.linalg.norm(d) - 1) < 1e-9 and abs(d @ e1) < 1e-9 and abs(d @ e2) < 1e-9 and abs(e1 @ e2) < 1e-9
    cols = F.initial_columns(F.N_SMALL, mix=0.5, inlet="jet")                       # a colinear stream straight up
    assert float(cols["spread"]) == 0.0
    out = _series(piece, cols, 120, keep=("x", "vx", "vy", "vz"))
    kept = out[2]
    vx, vy = np.array(kept["vx"]), np.array(kept["vy"])
    shot = vy[-1] > 0.5
    assert shot.any() and float(np.abs(vx[-1][shot]).max()) < 0.3 and float(np.abs(np.array(kept["vz"])[-1][shot]).max()) < 1e-6   # colinear (a fan reaches +-1)


def test_the_fluid_buoyancy_and_viscosity_slow_a_settling_grain(native_piece):
    piece, columns = native_piece
    base = dict(columns, mode=np.float64(2.0))
    speeds = {}
    for name, (rho, mu) in F.FLUIDS.items():
        cols = dict(base, fl_rho=np.float64(rho), fl_mu=np.float64(mu))
        # one free grain dropped high in the open (no neighbours): its speed after a second of fall
        cols = {k: (np.array(v) if np.ndim(v) else v) for k, v in cols.items()}
        for k in ("x", "y", "z", "vx", "vy", "vz", "wx", "wy", "wz"):
            cols[k] = np.zeros(F.N_SMALL)
        cols["age"] = np.full(F.N_SMALL, 1.0e3)
        cols["x"][0], cols["y"][0] = 0.0, 1.0
        out = F.run_frames(piece, cols, 360)
        speeds[name] = float(-out["vy"][0])
    assert speeds["air"] > speeds["water"] > speeds["glycerine"] > 0.0           # denser, more viscous: slower
    assert speeds["air"] > 0.8 * F.G and speeds["glycerine"] < 0.2                # air: ~free fall; glycerine: near terminal speed


def test_stiffness_is_clamped_to_the_stability_bound_and_the_neighbour_cap_counts_what_it_drops(native_piece):
    piece, columns = native_piece
    for hertz in (0.0, 1.0):
        u = F.uniforms(n=256, params={"omega": 1.0e4, "hertz": hertz})
        assert float(u["omega"]) <= F.OMEGA_MAX / (1.0 + 2.0 * hertz) ** 0.5 + 1e-3
        assert float(u["omega"]) * F.DT * (1.0 + 2.0 * hertz) ** 0.5 < 0.6
    stiff = F.initial_columns(F.N_SMALL, mix=0.0, params={"omega": 1.0e4, "hertz": 1.0, "coh": 1.0, "fl_mu": 5.0e-4})
    out = F.run_frames(piece, stiff, 3000)
    assert np.isfinite(out["x"]).all() and float(out["n_nan"][0]) == 0 and float(out["n_out"][0]) == 0
    capped = F.run_frames(piece, F.initial_columns(F.N_SMALL, mix=0.5, params={"k_run": 4}), 2500)
    assert float(capped["n_drop"][0]) > 0                                         # a small cap says so
    assert float(F.run_frames(piece, columns, 2500)["n_drop"][0]) == 0            # the default cap drops nothing


def test_species_roughness_irregularity_and_restitution_are_runtime_tunables():
    u = F.uniforms(n=256, params={"irr0": 0.2, "roll0": 0.1, "e0": 0.9, "r0": 0.012})
    assert float(u["irr0"]) == pytest.approx(0.2, abs=1e-6) and float(u["roll0"]) == pytest.approx(0.1, abs=1e-6)
    assert float(u["zeta0"]) == pytest.approx(F.zeta_of(0.9), abs=1e-6) and F.restitution_of(F.zeta_of(0.9)) == pytest.approx(0.9)
    assert F.restitution_of(F.ZETA) == pytest.approx(F.PARAMS["e0"][4]) and float(F.uniforms(n=256)["zeta0"]) == pytest.approx(F.ZETA, abs=1e-6)
    assert float(u["cell"]) == pytest.approx(2.0 * 0.012 * 1.2, abs=1e-6)       # 2 * max(radius * (1 + irr)): the widened grain species is now the largest
    assert float(F.uniforms(n=256)["cell"]) == pytest.approx(F.CELL, abs=1e-6)
    cols = F.initial_columns(256, mix=0.0, params={"irr0": 0.2})
    out = F.run_frames(F.make_piece(256), cols, 1)
    base = F.run_frames(F.make_piece(256), F.initial_columns(256, mix=0.0), 1)
    assert len(np.unique(np.round(out["rad"], 7))) > len(np.unique(np.round(base["rad"], 7)))          # one species, many sizes
    assert float(out["rad"].min()) >= 0.0090 * 0.8 - 1e-9


def test_the_server_declares_the_grouped_tunables_and_live_ones_do_not_restart_the_run():
    game = F.FountainGame(size="small")
    ids = {c["id"]: c for c in game.state()["controls_ui"]}
    assert {c["group"] for c in ids.values()} >= {"Run", "Grains", "Physics", "Fluid", "Chamber"}
    for key in ("r0", "rho1", "mu2", "e3", "roll0", "irr1", "wt2", "omega", "hertz", "coh", "spin_drag", "k_run", "nozzle_speed",
                "rate", "fl_rho", "fl_mu", "depth", "spread"):
        assert ids[key]["kind"] == "range" and ids[key]["action"] == "fountain_param" and ids[key]["name"] == key
    assert ids["fluid"]["kind"] == "select" and [o[0] for o in ids["fluid"]["options"]][:3] == ["air", "water", "glycerine"]
    assert ids["shape"]["kind"] == "select" and ids["inlet"]["kind"] == "select" and ids["k_run_drop"]["kind"] == "readout"
    before = game.state()["compute"]["epoch"]
    game.do({"type": "fountain_param", "name": "coh", "value": 1.5})              # live: the same run, new uniform
    state = game.state()["compute"]
    assert state["epoch"] == before and state["uniforms"]["coh"] == 1.5 and "coh" in F.RUNTIME_INPUTS
    game.do({"type": "fountain_param", "name": "omega", "value": 1.0e5})          # clamped to its range, and to the CFL bound in the law's input
    assert game.state()["compute"]["uniforms"]["omega"] <= F.OMEGA_MAX
    game.do({"type": "fountain_choice", "name": "fluid", "value": "glycerine"})
    u = game.state()["compute"]["uniforms"]
    assert u["fl_rho"] == pytest.approx(F.FLUIDS["glycerine"][0], rel=1e-6) and u["fl_mu"] == pytest.approx(F.FLUIDS["glycerine"][1], rel=1e-6)
    assert game.state()["compute"]["epoch"] == before
    game.do({"type": "fountain_param", "name": "depth", "value": 0.05})           # a restart class: a new run
    assert game.state()["compute"]["epoch"] == before + 1 and game.state()["compute"]["uniforms"]["dh"] == pytest.approx(0.025, rel=1e-6)
    game.do({"type": "fountain_choice", "name": "shape", "value": "funnel"})
    assert game.state()["compute"]["epoch"] == before + 2
    game.do({"type": "fountain_choice", "name": "inlet", "value": "pour"})
    assert game.state()["compute"]["uniforms"]["in_dy"] == -1.0 and game.state()["compute"]["uniforms"]["spread"] == 0.0
    for bad in ({"type": "fountain_param", "name": "nope", "value": 1}, {"type": "fountain_choice", "name": "shape", "value": "torus"}):
        with pytest.raises(ValueError):
            game.do(bad)
    game.do({"type": "fountain_start", "mix": 1.0})                               # the poly mix is the macro that sets the shares
    assert game.state()["compute"]["uniforms"]["wt2"] == pytest.approx(F.POLY_WEIGHT[2], rel=1e-6)


def _dust(piece_cols, fluid, temp, steps, *, mix=0.0):
    """256 grains spread out so none touch (a lattice), at rest, gravity off, in a fluid at kT = ``temp``."""
    rho, mu = F.FLUIDS[fluid]
    cols = F.initial_columns(F.N_SMALL, mix=mix, params={"fl_rho": rho, "fl_mu": 5.0e-4, "temp": temp, "depth": F.DEPTH_MAX})
    cols.update(grav_x=np.float64(0.0), grav_y=np.float64(0.0), grav_z=np.float64(0.0))
    ix, iy = np.meshgrid(np.arange(16), np.arange(16))
    cols["x"] = (ix.ravel() - 7.5) * 0.026
    cols["y"] = 0.15 + iy.ravel() * 0.04
    cols["age"] = np.full(F.N_SMALL, 1.0e3)
    return cols


def test_temperature_is_a_langevin_thermostat_whose_measured_kT_settles_near_the_set_one(native_piece):
    """Gravity off, small grains in glycerine (a strong drag, so a short relaxation), the measured kT = ke / (3 n_live)
    settles near the set kT; at kT = 0 nothing moves and the run is bit-identical to itself (the noise terms are exact zeros)."""
    piece, columns = native_piece
    assert "temp" in F.UNIFORMS and "temp" in F.RUNTIME_INPUTS and float(columns["temp"]) == 0.0
    cold = F.run_frames(piece, _dust(columns, "glycerine", 0.0, 0), 400)
    again = F.run_frames(piece, _dust(columns, "glycerine", 0.0, 0), 400)
    assert all(np.array_equal(cold[k], again[k]) for k in F.STATE)
    assert float(np.abs(cold["vx"]).max()) == 0.0 and float(np.abs(cold["wz"]).max()) == 0.0            # kT = 0: exactly still
    kT = 1.0e-9
    cols = _dust(columns, "glycerine", kT, 0)
    series = _series(piece, cols, 1800)[1]
    measured = series["ke"] / (3.0 * series["n_live"])
    hot = float(measured[600:].mean())                                      # after many relaxation times (gamma >= 12 /s, a few steps apiece)
    assert 0.9 * kT < hot < 1.1 * kT, (hot, kT)
    assert series["n_nan"].max() == 0 and series["n_out"].max() == 0
    replay = _series(piece, cols, 600)[1]                                   # the noise is a hash of (id, step, component): a replay repeats it
    assert np.array_equal(replay["ke"], series["ke"][:600])
    assert F.measured_temperature({"ke": series["ke"][-1], "n_live": series["n_live"][-1]}) == pytest.approx(float(measured[-1]))


def test_the_server_declares_temperature_on_a_log_slider_in_the_physics_group():
    ids = {c["id"]: c for c in F.FountainGame(size="small").state()["controls_ui"]}
    assert ids["temp"]["group"] == "Physics" and ids["temp"]["scale"] == "log" and ids["temp"]["min"] == 0.0 and ids["temp"]["log_min"] > 0
    game = F.FountainGame(size="small")
    before = game.state()["compute"]["epoch"]
    game.do({"type": "fountain_param", "name": "temp", "value": 2.0e-7})
    assert game.state()["compute"]["uniforms"]["temp"] == pytest.approx(2.0e-7, rel=1e-6) and game.state()["compute"]["epoch"] == before


def test_size_distributions_lognormal_and_bidisperse_are_hashed_per_id_and_off_by_default():
    n = F.N_SMALL

    def radii(**params):
        return F.run_frames(F.make_piece(n), F.initial_columns(n, mix=0.0, params=params), 1)["rad"]

    base = radii()
    assert len(np.unique(np.round(base, 7))) == 2                                   # grain and bead: no spread, ratio 1
    assert F.uniforms(n=n)["bi_ratio"] == 1.0 and F.uniforms(n=n)["lognormal"] == 0.0
    uni = radii(irr0=0.2, irr1=0.2)
    assert len(np.unique(np.round(uni, 7))) > 20 and float(uni.min()) >= 0.0090 * 0.8 - 1e-9 and float(uni.max()) <= 0.0105 * 1.2 + 1e-9
    logn = radii(irr0=0.2, irr1=0.2, lognormal=1.0)
    assert len(np.unique(np.round(logn, 7))) > 20 and float(logn.max()) <= 0.0105 * np.exp(0.6) + 1e-9
    assert not np.array_equal(uni, logn) and np.array_equal(logn, radii(irr0=0.2, irr1=0.2, lognormal=1.0))     # a hash: repeatable
    assert abs(float(np.mean(np.log(logn / np.where(logn > 0, base, 1.0)))) + 0.02) < 0.03       # lognormal: log-radius has mean -sigma^2/2
    bi = radii(bi_ratio=2.0, bi_frac=0.3)
    large, small = bi > base * 1.2, bi < base / 1.2
    assert abs(float(large.mean()) - 0.3) < 0.08 and (large | small).all()                         # two sizes only
    assert float(np.max(bi[large] / base[large])) == pytest.approx(2.0 ** 0.5, rel=1e-6) and float(np.max(base[small] / bi[small])) == pytest.approx(2.0 ** 0.5, rel=1e-6)
    assert float(F.uniforms(n=n, params={"bi_ratio": 2.0})["cell"]) == pytest.approx(F.CELL * 2.0 ** 0.5, rel=1e-6)      # the grid fits the largest


def _turning_case(piece, omega, steps, domega=0.0, at=(0.08, 0.3)):
    """One grain at rest in the case frame, gravity and fluid off, the case turning about (0, 0.3)."""
    n = F.N_SMALL
    cols = F.initial_columns(n, mix=0.0, params={"fl_mu": 0.0, "fl_rho": 0.0})
    cols.update(grav_x=np.float64(0.0), grav_y=np.float64(0.0), grav_z=np.float64(0.0), case_omega=np.float64(omega),
                case_domega=np.float64(domega), pivot_x=np.float64(0.0), pivot_y=np.float64(0.3), age=np.full(n, -1.0e9))
    cols["age"][0] = 1.0e3
    cols["x"][0], cols["y"][0] = at
    return F.run_frames(piece, cols, steps)


def _rotating_frame_reference(r0, omega, domega, steps):
    """RK4 of  v' = -2 w x v + w^2 r - dw x r  (r from the pivot) with small substeps."""
    r, v = np.array(r0, float), np.zeros(2)

    def acc(r, v):
        return np.array([2 * omega * v[1] + omega * omega * r[0] + domega * r[1], -2 * omega * v[0] + omega * omega * r[1] - domega * r[0]])
    h = F.DT / 20.0
    for _ in range(steps * 20):
        k1r, k1v = v, acc(r, v)
        k2r, k2v = v + 0.5 * h * k1v, acc(r + 0.5 * h * k1r, v + 0.5 * h * k1v)
        k3r, k3v = v + 0.5 * h * k2v, acc(r + 0.5 * h * k2r, v + 0.5 * h * k2v)
        k4r, k4v = v + h * k3v, acc(r + h * k3r, v + h * k3v)
        r, v = r + h / 6 * (k1r + 2 * k2r + 2 * k3r + k4r), v + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return r


def test_a_turning_case_drags_its_contents_in_3d_too(native_piece):
    """The same case-frame law in the slab: outward, bent sideways, matching the rotating-frame equation; still at omega = 0."""
    piece, _columns = native_piece
    assert {"case_omega", "case_domega"} <= set(F.RUNTIME_INPUTS) and {"pivot_x", "pivot_y"} <= set(F.UNIFORMS)
    still = _turning_case(piece, 0.0, 120)
    assert float(still["x"][0]) == 0.08 and float(still["y"][0]) == 0.3 and float(still["vx"][0]) == 0.0
    steps, omega = 120, 3.0
    out = _turning_case(piece, omega, steps)
    rel = np.array([float(out["x"][0]), float(out["y"][0]) - 0.3])
    ref = _rotating_frame_reference((0.08, 0.0), omega, 0.0, steps)
    assert rel[0] > 0.1 and rel[1] < -1e-3
    assert np.hypot(*(rel - ref)) < 0.03 * np.hypot(*(ref - np.array([0.08, 0.0]))) + 1e-4, (rel, ref)
    eul = _turning_case(piece, 0.0, steps, domega=4.0)
    ref_e = _rotating_frame_reference((0.08, 0.0), 0.0, 4.0, steps)
    assert np.hypot(float(eul["x"][0]) - ref_e[0], float(eul["y"][0]) - 0.3 - ref_e[1]) < 0.03 * np.hypot(*(ref_e - np.array([0.08, 0.0]))) + 1e-4
    assert float(out["n_nan"][0]) == 0 and float(out["n_out"][0]) == 0


def test_every_shape_is_inscribed_in_the_visible_window_closed_when_a_polygon_and_the_inlet_is_inside():
    for n in (256, 4096):
        geo = F.basin(n)
        x0, x1, y0, y1 = geo["view"]
        box = F.shape_vertices("box", geo)
        assert box == [(-geo["half"], geo["nozzle_y"] + 2.5), (-geo["half"], 0.0), (geo["half"], 0.0), (geo["half"], geo["nozzle_y"] + 2.5)]    # the old basin
        for shape in ("funnel", "hexagon", "ramp"):
            verts = F.shape_vertices(shape, geo)
            xs, ys = [v[0] for v in verts], [v[1] for v in verts]
            assert min(xs) >= x0 + 2.0 * geo["cell"] and max(xs) <= x1 - 2.0 * geo["cell"] and min(ys) >= 0.0 and max(ys) <= y1 - 2.0 * geo["cell"], shape      # inside the window, both axes, with a margin
            assert max(xs) == pytest.approx(-min(xs)) and max(xs) > x1 - 4.0 * geo["cell"] and max(ys) > 0.9 * (y1 - 3.0 * geo["cell"]) - 1e-9           # centred, as large as fits
            assert verts[0] == verts[-1] and len(verts) <= F.NV                                                        # every non-box shape is closed: its own ceiling
            u = F.uniforms(n=n, shape=shape)
            xl, xr = F.chord(verts, float(u["nozzle_y"]))
            assert xl < float(u["nozzle_x"]) - float(u["bar_half"]) and float(u["nozzle_x"]) + float(u["bar_half"]) < xr    # the inlet bar is inside the chord
            assert float(u["nozzle_y"]) < max(ys)


# ------------------------------------------------------------------------------- SDF chambers (true 3-D solids)
def _eager_step(law, cols, n):
    tensors = [AbstractTensor.get_tensor(np.asarray(cols[k], dtype=np.float64)) for k in F.ARGUMENTS]
    out = law(*tensors)
    got = {nm: np.broadcast_to(np.asarray(v.numpy(), dtype=np.float64), (n,)).copy() for nm, v in zip(F.OUTPUTS, out)}
    nxt = dict(cols)
    for s in F.STATE:
        nxt[s] = got[s + "_next"]
    return nxt, got


def _thrown(n, uni, seed=0, box=((-0.15, 0.15), (0.012, 0.09))):
    """``n`` live grains packed into a small region with random velocities and spin (contact-rich from the first step)."""
    rng = np.random.default_rng(seed)
    cols = {k: np.zeros(n) for k in F.STATE + ("ir", "ig", "ib")}
    cols["pid"] = np.arange(n, dtype=np.float64)
    cols.update({k: float(v) for k, v in uni.items()})
    dh = float(cols["dh"])
    cols["x"], cols["y"] = rng.uniform(*box[0], n), rng.uniform(*box[1], n)
    cols["z"] = rng.uniform(-dh + 0.012, dh - 0.012, n)
    for k in ("vx", "vy", "vz"):
        cols[k] = rng.uniform(-1.0, 1.0, n)
    for k in ("wx", "wy", "wz"):
        cols[k] = rng.uniform(-3.0, 3.0, n)
    cols["age"] = np.full(n, 100.0)
    return cols


def test_sdf_off_is_the_old_law_bit_for_bit_over_contact_rich_steps():
    """``sdf_on = 0`` multiplies the SDF contact by an exact zero: the law with the SDF walls equals the law compiled without them, every state and
    every output, over steps in which the grains collide with each other and every wall (eager NumPy lane; the native 3-D compile is too long to run here).
    (The same run was also held equal to the law at the commit before the SDF walls: 40 steps, 64 ids.)"""
    n = 24
    uni = F.uniforms(n=n)
    assert float(uni["sdf_on"]) == 0.0 and all(float(uni[f"q{k}kind"]) == 0.0 for k in range(F.SDF_SLOTS))
    off, base = F.make_law(n, n, True), F.make_law(n, n, False)
    a, b = _thrown(n, uni), _thrown(n, uni)
    for _ in range(12):
        a, ga = _eager_step(off, a, n)
        b, gb = _eager_step(base, b, n)
        assert all(np.array_equal(ga[k], gb[k]) for k in F.OUTPUTS)
    assert float(np.abs(a["vy"]).max()) > 0.0 and ga["n_out"][0] == 0                          # the grains moved and none left


def test_sdf_gradients_are_analytic_and_the_distances_exact_where_the_solids_are_simple():
    rng = np.random.default_rng(1)
    for shape in F.SDF_SHAPES:
        u = F.uniforms(n=4096, shape=shape)
        _c, w, top = F.window_fit(shape, F.basin(4096, shape=shape))
        dh = float(u["dh"])
        xs, ys, zs = rng.uniform(-w, w, 1500), rng.uniform(0.0, top, 1500), rng.uniform(-dh, dh, 1500)
        d, nx, ny, nz = F.chamber_distance(u, xs, ys, zs)
        h = 2.0e-3
        fd = []
        for ax in range(3):
            e = [0.0, 0.0, 0.0]
            e[ax] = h
            fd.append((F.chamber_distance(u, xs + e[0], ys + e[1], zs + e[2])[0] - F.chamber_distance(u, xs - e[0], ys - e[1], zs - e[2])[0]) / (2.0 * h))
        err = np.sqrt((nx - fd[0]) ** 2 + (ny - fd[1]) ** 2 + (nz - fd[2]) ** 2)
        near = np.abs(d) < 0.3
        assert near.any() and (err[near] < 6.0e-3).mean() > 0.95, shape                      # the rest are the kinks / medial surfaces the difference straddles
        assert (np.abs(np.sqrt(nx ** 2 + ny ** 2 + nz ** 2) - 1.0) < 2.0e-3).mean() > 0.999, shape       # a unit gradient everywhere (bar a measure-zero sliver at a face)
    # exact: the ball and the tetrahedron (inside) against their closed forms
    u = F.uniforms(n=4096, shape="sdf_sphere")
    rad, cy = float(u["q3p0"]), float(u["q3cy"])
    p = rng.uniform(-0.5, 0.5, (3, 200)) * rad
    assert np.abs(F.chamber_distance(u, p[0], cy + p[1], p[2])[0] - (rad - np.sqrt((p ** 2).sum(0)))).max() < 1.0e-6
    u = F.uniforms(n=4096, shape="sdf_tetra")
    rad, cy, cz = float(u["q9p0"]), float(u["q9cy"]), float(u["q9cz"])
    pts = np.array([[0.0, 0.0, 0.0], [0.1, 0.2, 0.05], [-0.1, -0.1, 0.1]]) * rad
    d = F.chamber_distance(u, pts[:, 0], cy + pts[:, 1], cz + pts[:, 2])[0]
    normals = np.array(F._TET_NORMALS)
    assert np.abs(d - (rad / 3.0 - (pts @ normals.T).max(1))).max() < 1.0e-6                # inside: the distance to the nearest face plane (the inradius at the centre)
    assert abs(float(F.chamber_distance(u, [0.0], [cy], [cz])[0][0]) - rad / 3.0) < 1.0e-6


def test_sdf_presets_are_inscribed_in_the_window_and_the_depth_and_the_inlet_is_inside_the_fluid():
    for n in (256, 4096, 65536):
        for shape in F.SDF_SHAPES:
            c, w, top = F.window_fit(shape, F.basin(n, shape=shape))
            u = F.uniforms(n=n, shape=shape)
            dh = float(u["dh"])
            assert float(u["sdf_on"]) == 1.0 and dh <= 0.5 * F.DEPTH_MAX + 1.0e-9
            # the fluid (d > 0), sampled over the window and the slab (and a margin around them), lies inside |x| <= w, 0 <= y <= top and the depth
            xs, ys, zs = np.meshgrid(np.linspace(-w - 0.1, w + 0.1, 41), np.linspace(-0.05, top + 0.1, 61), np.linspace(-dh, dh, 15), indexing="ij")
            d = F.chamber_distance(u, xs.ravel(), ys.ravel(), zs.ravel())[0]
            fx, fy, fz = (a.ravel()[d > 0.0] for a in (xs, ys, zs))
            assert fx.size > 0, shape
            assert np.abs(fx).max() <= w + 1.0e-6 and fy.min() >= -1.0e-6 and fy.max() <= top + 1.0e-6, (shape, n)          # inscribed in the window
            assert shape == "sdf_toy" or np.abs(fz).max() <= dh - F.Z_MARGIN_CELLS * c + 1.0e-6, (shape, n)                   # and in the depth, a margin from the glass
            # the inlet: the whole source volume is deeper than one grain diameter (> 2 r) inside the fluid
            px, py, pz = F.source_points({k: float(v) for k, v in u.items()}, float(u["bar_half"]), float(u["bar_z"]))
            assert float(F.chamber_distance(u, px, py, pz)[0].min()) > (c if n >= 4096 else 0.0), (shape, n)       # (a 256-id window is too small for a source clear by a diameter: it is at least inside)
            assert float(u["rows"]) * c >= top                                                                                  # the grid covers the whole chamber
    # the polygon presets and the box are untouched: sdf off, no slots
    for shape in ("box", "funnel", "hexagon", "ramp"):
        assert float(F.uniforms(n=256, shape=shape)["sdf_on"]) == 0.0


def test_the_sdf_chamber_mesh_is_a_deterministic_compact_watertight_surface_the_page_can_draw():
    from collections import Counter
    from meowdoku import fountain_game as G
    game = G.FountainGame(size="small")
    assert game.state()["compute"]["chamber_mesh"] is None                                   # the box and the polygons are drawn from their vertices
    for shape in F.SDF_SHAPES:
        game = G.FountainGame(size="4k")
        game.do({"type": "fountain_choice", "name": "shape", "value": shape})
        mesh = game.state()["compute"]["chamber_mesh"]
        G._MESHES.clear()
        again = G.chamber_mesh(shape, game.n, game.aspect, game.params)
        assert again == mesh, shape                                                           # deterministic (same bytes)
        assert mesh["nt"] <= max(G.MESH_TRIANGLES, 10000) and len(json.dumps(mesh)) < 200_000, shape      # compact
        verts, tris = G.unpack_mesh(mesh)
        assert len(verts) == mesh["nv"] and len(tris) == mesh["nt"] and tris.max() < len(verts)
        u = {k: float(v) for k, v in game.uniforms().items()}
        d = F.chamber_distance(u, verts[:, 0], verts[:, 1], verts[:, 2])[0]
        assert np.abs(d).mean() < 0.01 and np.abs(d).max() < 0.1, shape                      # every vertex lies on the zero surface, to the grid (a coarse step cuts a cube corner by a few hundredths)
        assert np.abs(verts[:, 2]).max() <= u["dh"] + 1.0e-3                                  # inside the slab
        if shape != "sdf_toy":                                                                # a true solid is a closed surface: every edge has two triangles
            edges = Counter()
            for a, b, c in tris:
                for p, q in ((a, b), (b, c), (c, a)):
                    edges[(min(p, q), max(p, q))] += 1
            assert set(edges.values()) == {2}, shape
        else:                                                                                 # the toy is the extruded contour: its walls span the whole depth
            assert abs(verts[:, 2].min() + u["dh"]) < 1.0e-3 and abs(verts[:, 2].max() - u["dh"]) < 1.0e-3


def test_the_server_offers_the_sdf_chambers_and_builds_each_to_the_depth_its_solid_needs():
    from meowdoku import fountain_game as G
    game = G.FountainGame(size="4k")
    ids = {c["id"]: c for c in game.controls_ui()}
    assert [k for k, _label in ids["shape"]["options"]] == [k for k, _label in F.CHAMBER_CHOICES] + [G.GLYPH_SHAPE]
    depths = {}
    for shape in F.SDF_SHAPES:
        game.do({"type": "fountain_choice", "name": "shape", "value": shape})
        st = game.state()
        depths[shape] = st["compute"]["depth"]
        assert st["compute"]["uniforms"]["sdf_on"] == 1.0 and st["compute"]["chamber_mesh"] is not None
        assert {c["id"]: c for c in st["controls_ui"]}["depth"]["value"] == pytest.approx(depths[shape], abs=1e-6)
    assert depths["sdf_sphere"] > 10 * depths["sdf_toy"] and depths["sdf_toy"] == pytest.approx(F.DEPTH_DEFAULT, abs=1e-6)
    game.do({"type": "fountain_param", "name": "depth", "value": 0.4})                         # a depth the page sets is kept: the sphere shrinks to fit it
    assert game.state()["compute"]["depth"] == pytest.approx(0.4, abs=1e-6)
    game.do({"type": "fountain_choice", "name": "shape", "value": "funnel"})                  # back to a polygon: the polygons' default depth
    assert game.state()["compute"]["depth"] == pytest.approx(F.DEPTH_DEFAULT, abs=1e-6) and game.state()["compute"]["chamber_mesh"] is None


def _throw_at_walls(shape, n=24, steps=60, speed=2.5, where=None, seed=0):
    """Grains placed in the fluid 0.06 - 0.30 from a wall and thrown at it at ``speed`` (fan-jet speeds are ~1.6) for ``steps`` steps (eager NumPy lane);
    returns (the largest n_out over the steps, the worst distance a grain centre went beyond a wall, the grain radius)."""
    rng = np.random.default_rng(seed)
    uf = {k: float(v) for k, v in F.uniforms(n=4096, shape=shape).items()}
    _c, w, top = F.window_fit(shape, F.basin(4096, shape=shape))
    pts = []
    while len(pts) < n:
        p = np.array([rng.uniform(-w, w), rng.uniform(0.0, top), rng.uniform(-uf["dh"] + 0.02, uf["dh"] - 0.02)])
        d, nx, ny, nz = (float(v[0]) for v in F.chamber_distance(uf, [p[0]], [p[1]], [p[2]]))
        if 0.06 < d < 0.30 and (where is None or where(p, uf)):
            pts.append((p, -np.array([nx, ny, nz]) * speed + rng.normal(size=3) * 0.3))
    cols = {k: np.zeros(n) for k in F.STATE + ("ir", "ig", "ib")}
    cols["pid"] = np.arange(n, dtype=np.float64)
    cols.update(uf)
    for i, (p, v) in enumerate(pts):
        cols["x"][i], cols["y"][i], cols["z"][i] = p
        cols["vx"][i], cols["vy"][i], cols["vz"][i] = v
    cols["age"] = np.full(n, 100.0)
    law = F.make_law(n, n)
    worst, out = 0.0, 0.0
    for _ in range(steps):
        cols, got = _eager_step(law, cols, n)
        d = F.chamber_distance(uf, cols["x"], cols["y"], cols["z"])[0]
        worst, out = max(worst, float((-d).max())), max(out, float(got["n_out"][0]))
    return out, worst, float(got["rad"].mean())


def test_grains_thrown_at_the_pegs_the_ramps_and_the_torus_inner_wall_do_not_escape():
    """Thrown at the walls faster than the fan jet fires (eager lane; the 64-id, 150-step run over every chamber is in the commit's report):
    none leaves (n_out = 0), and no grain centre goes more than a radius beyond a wall."""
    top = F.window_fit("sdf_toy", F.basin(4096, shape="sdf_toy"))[2]
    out, worst, rad = _throw_at_walls("sdf_toy", where=lambda p, u: 0.05 * top <= p[1] <= 0.30 * top, seed=1)          # the peg field's band
    assert out == 0.0 and worst < rad
    out, worst, rad = _throw_at_walls("sdf_toy", where=lambda p, u: p[1] > 0.7 * top, seed=2)                         # the ramps' band
    assert out == 0.0 and worst < rad
    u = F.uniforms(n=4096, shape="sdf_torus")
    big, cy = float(u["q8p0"]), float(u["q8cy"])
    out, worst, rad = _throw_at_walls("sdf_torus", where=lambda p, uf: np.hypot(p[0], p[1] - cy) < big - 0.02, seed=3)  # the inner half of the tube: at the INNER wall
    assert out == 0.0 and worst < rad


# ------------------------------------------------------------------------------- ORBEEZ (grains that swell with hydration)
#: A SHA-256 (first 16 hex digits) of every output but ``rs_next`` over 12 contact-rich steps (``_thrown``, 24 ids, eager NumPy), recorded from the law
#: at the commit BEFORE the orbeez (state without ``rs``): with ``swell`` = 0 the new law must reproduce it to the bit.
PRE_ORBEEZ_DIGEST = {0.5: "54cef27ed35d84ec", 1.0: "62e1ff22b3c31ab6"}


def test_without_swelling_the_law_is_the_pre_orbeez_law_bit_for_bit_over_contact_rich_steps():
    """swell = 0 (the default): every orbeez term is an exact zero and the radius is the species radius, so every output the old law had equals the
    old law's to the bit (the digests were recorded from the law before the radius became state), over steps in which the grains collide with each other
    and every wall.  Eager NumPy lane (the native 3-D compile is too long to run here)."""
    import hashlib
    n = 24
    for mix, digest in PRE_ORBEEZ_DIGEST.items():
        uni = F.uniforms(n=n, mix=mix)
        assert all(float(uni[f"swell{s}"]) == 0.0 and float(uni[f"swmax{s}"]) == 1.0 for s in range(4))
        law = F.make_law(n, n, True)
        cols = _thrown(n, uni)
        h = hashlib.sha256()
        overlapped = False
        for _ in range(12):
            gap = np.sqrt(sum((cols[k][:, None] - cols[k][None, :]) ** 2 for k in "xyz")) - 0.018
            np.fill_diagonal(gap, 1.0)
            overlapped = overlapped or float(gap.min()) < 0.0
            cols, got = _eager_step(law, cols, n)
            for k in F.OUTPUTS:
                if k != "rs_next":
                    h.update(np.ascontiguousarray(got[k], dtype=np.float64).tobytes())
        assert h.hexdigest()[:16] == digest, mix
        assert overlapped and float(np.abs(got["vy_next"]).max()) > 0.0 and got["n_out"][0] == 0
        assert np.array_equal(cols["rs"], np.zeros(n))                           # with no swelling the radius state is never touched (an unset rs is the species radius)


def _orbeez_columns(n, fluid, params=None, mix=0.0):
    rho, mu = F.FLUIDS[fluid] if isinstance(fluid, str) else fluid
    cols = F.initial_columns(n, mix=mix, params={"fl_rho": rho, "fl_mu": mu, **(params or {})})
    cols.update(grav_x=np.float64(0.0), grav_y=np.float64(0.0), grav_z=np.float64(0.0), age=np.full(n, -1.0e9))
    return cols


def _species_ids(law, n):
    """The ids of species 0 and 1 (mix 0), read off one eager step of an unplaced state."""
    _c, got = _eager_step(law, dict(F.initial_columns(n, mix=0.0)), n)
    s0 = [int(i) for i in np.where(np.abs(got["rad"] - F.SPECIES[0][1]) < 1e-9)[0]]
    s1 = [int(i) for i in np.where(np.abs(got["rad"] - F.SPECIES[1][1]) < 1e-9)[0]]
    return s0, s1


_PHI = (1.0 + 5.0 ** 0.5) / 2.0
_ICOSAHEDRON = [tuple(np.roll((0.0, a, b * _PHI), k)) for a in (-1.0, 1.0) for b in (-1.0, 1.0) for k in range(3)]       # the 12 close-packed directions
_OCTAHEDRON = [tuple(s * float(k == m) for m in range(3)) for k in range(3) for s in (-1.0, 1.0)]                       # 6 of them


def _grow_scene(law, n, fluid, rate=0.3, swmax=1.5):
    """Species-0 grains: A free, B6 touching six inert species-1 grains (half of the 12 that close a sphere), B12 closed in by twelve; all others parked
    (not emitted), gravity off.  One step.  Returns the state after it, the step's outputs, the three ids and the ring ids."""
    s0, s1 = _species_ids(law, n)
    cols = _orbeez_columns(n, fluid, {"swell0": rate, "swmax0": swmax})
    a, b6, b12 = s0[0], s0[1], s0[2]
    gap = F.SPECIES[0][1] + F.SPECIES[1][1] - 0.0004                       # overlapping a little: touching
    place = {a: (-0.12, 0.4, 0.0), b6: (0.0, 0.4, 0.0), b12: (0.12, 0.4, 0.0)}
    ring, k = [], 0
    for centre, dirs in ((b6, _OCTAHEDRON), (b12, _ICOSAHEDRON)):
        for d in dirs:
            d = np.asarray(d, dtype=np.float64)
            place[s1[k]] = tuple(np.asarray(place[centre]) + gap * d / np.linalg.norm(d))
            ring.append(s1[k])
            k += 1
    for i, (px, py, pz) in place.items():
        cols["x"][i], cols["y"][i], cols["z"][i], cols["age"][i] = px, py, pz, 1.0e3
        cols["rs"][i] = F.SPECIES[0][1] if i in (a, b6, b12) else F.SPECIES[1][1]            # placed grains, at their species radius
    cols, got = _eager_step(law, cols, n)
    return cols, got, (a, b6, b12), ring


def test_orbeez_growth_share_is_the_exposed_surface_times_the_solvent_the_fluid_gives():
    """ONE step in water: a free grain grows at the full rate (share 1), one touching 6 of the 12 contacts that close a sphere at half of it, one
    closed in by 12 not at all.  In air nothing grows; in glycerine (viscosity-limited uptake) a free grain grows at half the rate.  Counters stay exact.
    (Eager NumPy lane.)"""
    n, rate = F.N_SMALL, 0.3
    law = F.make_law(n, n, False)
    r0 = F.SPECIES[0][1]
    share = lambda cols, i: (float(cols["rs"][i]) - r0) / (rate * r0 * F.DT)
    cols, got, (a, b6, b12), ring = _grow_scene(law, n, "water")
    assert F.FULL_COORDINATION == 12.0
    assert share(cols, a) == pytest.approx(1.0, abs=1e-3) and share(cols, b6) == pytest.approx(0.5, abs=1e-3) and abs(share(cols, b12)) < 1e-3
    assert all(float(cols["rs"][i]) == pytest.approx(F.SPECIES[1][1], rel=1e-6) for i in ring)           # a species that does not swell keeps its radius
    assert float(got["rad"][a]) == pytest.approx(r0, abs=1e-12)               # the drawn radius is the one the step used (the state is a step of growth ahead)
    assert float(got["n_nan"][0]) == 0 and float(got["n_out"][0]) == 0
    assert float(got["n_live"][0] + got["n_parked"][0] + got["n_out"][0] + got["n_nan"][0]) == n         # conservation: every id in exactly one class
    dry, _got, (a, b6, b12), _ring = _grow_scene(law, n, "air")
    assert all(float(dry["rs"][i]) == pytest.approx(r0, rel=1e-6) for i in (a, b6, b12))                # air: nothing swells
    glycerine, _got, (a, _b6, _b12), _ring = _grow_scene(law, n, "glycerine")
    assert share(glycerine, a) == pytest.approx(0.5, abs=1e-3)


def test_a_free_orbeez_grows_linearly_to_swmax_and_its_mass_follows_r_cubed():
    """A free grain in water grows linearly, 1 + swell t (checked over 0.1 s = 36 steps: the eager lane takes ~1 s a step; the 1 s run of 360 steps is the
    same recurrence), stops at swmax x its radius, and its mass is density x 4/3 pi r^3: a coasting grain at twice the radius carries 8x the kinetic energy."""
    n, rate = 8, 0.5
    law = F.make_law(n, n, False)
    _c, got = _eager_step(law, dict(F.initial_columns(n, mix=0.0)), n)
    r0 = float(got["rad"][0])
    cols = _orbeez_columns(n, "water", {"swell0": rate, "swmax0": 1.04, "swell1": rate, "swmax1": 1.04})
    cols["x"][0], cols["y"][0], cols["age"][0], cols["rs"][0] = 0.0, 0.4, 1.0e3, r0
    capped, uncapped = cols, dict(cols, swmax0=np.float64(3.0), swmax1=np.float64(3.0))
    steps = 36
    for _ in range(steps):
        uncapped, got_u = _eager_step(law, uncapped, n)
        capped, _got_c = _eager_step(law, capped, n)
    assert float(uncapped["rs"][0]) / r0 == pytest.approx(1.0 + rate * steps * F.DT, abs=1e-9)         # 1.05 after 0.1 s
    assert float(capped["rs"][0]) == pytest.approx(1.04 * r0, rel=1e-6)                                # the cap: swmax x the initial radius
    assert float(got_u["n_nan"][0]) == 0 and float(got_u["n_live"][0] + got_u["n_parked"][0] + got_u["n_out"][0]) == n
    energy = {}
    for scale in (1.0, 2.0):                                                                           # no fluid, no gravity: the speed is unchanged by a step
        c = _orbeez_columns(n, (0.0, 0.0))
        c["x"][0], c["y"][0], c["age"][0], c["vx"][0], c["rs"][0] = 0.0, 0.4, 1.0e3, 0.1, scale * r0
        _c, g = _eager_step(law, c, n)
        energy[scale] = float(g["ke"][0])
    assert energy[2.0] / energy[1.0] == pytest.approx(8.0, rel=1e-12)


def test_the_grid_cell_covers_the_largest_radius_an_orbeez_can_reach_and_the_game_offers_the_controls():
    base = F.uniforms(n=256)
    swell = F.uniforms(n=256, params={"swell3": 0.2, "swmax3": 3.0})
    assert float(base["cell"]) == pytest.approx(F.CELL, abs=1e-9)
    assert float(swell["cell"]) == pytest.approx(2.0 * 3.0 * F.SPECIES[3][1], rel=1e-6)           # 2 x swmax x the pearl radius
    assert float(F.uniforms(n=256, params={"swell3": 0.0, "swmax3": 3.0})["cell"]) == float(base["cell"])    # no swelling: swmax does not size the grid
    assert float(swell["bar_z"]) == float(base["bar_z"])                                           # the inlet is for grains as placed
    assert all(float(base[f"swell{s}"]) == 0.0 and float(base[f"swmax{s}"]) == 1.0 for s in range(4))
    assert "rs" in F.STATE and "rs_next" in F.OUTPUTS and F.ARGUMENTS == F.STATE + ("pid", "ir", "ig", "ib") + F.UNIFORMS
    game = F.FountainGame(size="small")
    ids = {c["id"]: c for c in game.state()["controls_ui"]}
    for s in range(4):
        assert ids[f"swell{s}"]["group"] == "Grains" and ids[f"swell{s}"]["max"] == F.SWELL_RATE_MAX and ids[f"swell{s}"]["species"] == s
        assert ids[f"swmax{s}"]["group"] == "Grains" and ids[f"swmax{s}"]["min"] == 1.0 and ids[f"swmax{s}"]["max"] == F.SWELL_MAX_RATIO
    epoch = game.state()["compute"]["epoch"]
    game.do({"type": "fountain_param", "name": "swell1", "value": 0.2})                       # the grid is sized from it: a new run
    state = game.state()["compute"]
    assert state["epoch"] == epoch + 1 and state["uniforms"]["swell1"] == pytest.approx(0.2, rel=1e-6)
    game.do({"type": "fountain_choice", "name": "fluid", "value": "water"})                    # the fluid is the solvent: live, no restart
    assert game.state()["compute"]["epoch"] == epoch + 1 and game.state()["compute"]["uniforms"]["fl_rho"] == pytest.approx(F.FLUIDS["water"][0], rel=1e-6)
    assert F.ARENA_SLOTS == 51 and F.binding_bytes(10_000_000) == 8 * 10_000_000 * 4 and F.binding_bytes(4096) == 51 * 4096 * 4


# ------------------------------------------------------------------------------- SDF presets keep their DESIGNED height
def test_sdf_presets_keep_their_designed_proportions_and_the_view_follows_the_chamber():
    """An SDF solid is not normalised to the view: its box is designed (``sdf_design``: scaled by the id count and the grain cell), the screen aspect
    does not stretch it, and the view, grid and depth derive from its bounding box plus a margin of three cells."""
    for shape in F.SDF_SHAPES:
        for n in (256, 4096, 65536, 1048576):
            geo = F.basin(n, shape=shape)
            c, w, top = F.window_fit(shape, geo)
            assert geo["view"] == (-(w + 3 * c), w + 3 * c, geo["view"][2], top + 3 * c)                  # the view is the box plus the margin
            assert float(F.uniforms(n=n, shape=shape)["rows"]) * c >= top and geo["half"] >= w
            wide, tall = F.basin(n, 3.0, shape=shape), F.basin(n, 0.35, shape=shape)
            assert wide["view"] == tall["view"] == geo["view"] and wide["cols"] == geo["cols"]               # the screen's aspect does not reshape it
            cells = geo["cols"] * (geo["rows"] + F.ROWS_MARGIN) * max(1, int(round(F.depth_cap(n, c, None, shape) / c)))
            assert cells < 2.0 ** 24, (shape, n)                                                              # the f32 cell index stays exact
            u = F.uniforms(n=n, shape=shape)
            assert float(u["cols"]) * (float(u["rows"]) + F.ROWS_MARGIN) * float(u["lays"]) < 2.0 ** 24 and 2.0 * float(u["dh"]) <= F.depth_cap(n, c, None, shape) + 1e-6
    c = F.CELL
    assert F.sdf_design("sdf_sphere", 1.0, c)[2] == pytest.approx(2.0 * F.sdf_design("sdf_sphere", 1.0, c)[1])      # as tall as wide
    assert F.sdf_design("sdf_hourglass", 1.0, c)[2] == pytest.approx(F.HOURGLASS_ASPECT * F.sdf_design("sdf_hourglass", 1.0, c)[1])
    # The 2-D and 3-D toy use the same bottom-up geometry, including split ramps and the wheel.
    from meowdoku import toy_design
    for n in (256, 4096):
        for cell in (F.CELL, 2.0 * F.CELL, 4.0 * F.CELL):
            _c, w, top = F.sdf_design("sdf_toy", math.sqrt(n / F.N_BASE), cell)
            parts = toy_design.toy_parts(w, cell, [species[1] for species in F.SPECIES])
            assert top == pytest.approx(parts["top"])
            assert parts["pegs"]["rows"] >= toy_design.PEG_ROWS
            assert parts["pegs"]["top"] < parts["yc"] < parts["wheel"]["y"] < parts["ramps"][0]["high"][1] < top
    small = F.sdf_design("sdf_toy", 0.25, F.CELL)
    assert small[2] / small[1] > F.sdf_design("sdf_toy", 1.0, F.CELL)[2] / F.sdf_design("sdf_toy", 1.0, F.CELL)[1]     # a small chamber is relatively taller
    # the polygons and the box are untouched by the shape argument
    for shape in ("box", "funnel", "hexagon", "ramp"):
        assert F.basin(4096, 1.7, shape=shape) == F.basin(4096, 1.7)


# ------------------------------------------------------------- the hinged waterwheel (body variant of the 3-D law; eager NumPy lane, ~1 s a step)
class _WheelWorld:
    def __init__(self, n=16, **uniform_overrides):
        self.n = n
        self.law = F.make_law(n, n, False, True)
        self.cols = F.initial_columns(n, mix=0.0, shape="waterwheel")
        self.cols["age"][:] = -1.0e9                                        # every id parked (never emitted); grains are placed by hand
        for k, v in uniform_overrides.items():
            self.cols[k] = np.float64(v)

    def place(self, grains):
        """{id: (x, y, z, vx, vy)}: live grains."""
        for i, (x, y, z, vx, vy) in grains.items():
            self.cols["age"][i], self.cols["x"][i], self.cols["y"][i], self.cols["z"][i], self.cols["vx"][i], self.cols["vy"][i] = 1.0e3, x, y, z, vx, vy

    def steps(self, steps, each=None):
        """Step ``steps`` times; returns the state, the per-step (body_th, body_om, body_tq) and the last outputs."""
        names = F.STATE + F.BODY_STATE
        state = {k: np.array(self.cols[k], dtype=np.float64) for k in names}
        series, out = [], {}
        for step in range(steps):
            tensors = [AbstractTensor.get_tensor(np.asarray(state[k] if k in state else self.cols[k], dtype=np.float64)) for k in F.BODY_ARGUMENTS]
            produced = self.law(*tensors)
            out = {nm: np.broadcast_to(np.asarray(v.numpy(), dtype=np.float64), (self.n,)).copy() for nm, v in zip(F.BODY_OUTPUTS, produced)}
            state = {k: out[k + "_next"] for k in names}
            series.append((float(out["body_th"][0]), float(out["body_om"][0]), float(out["body_tq"][0])))
            if each is not None:
                each(step, self.cols, out)
        return state, np.array(series), out


def test_the_default_3d_law_has_no_body_columns_and_the_wheel_variant_adds_exactly_them():
    assert F.BODY_ARGUMENTS == F.STATE + F.BODY_STATE + F.CONSTANTS + F.UNIFORMS + F.BODY_UNIFORMS
    assert not set(F.BODY_STATE + F.BODY_UNIFORMS) & set(F.ARGUMENTS) and len(F.BODY_ARGUMENTS) == len(set(F.BODY_ARGUMENTS))
    assert F.BODY_OUTPUTS == tuple(f"{s}_next" for s in F.STATE + F.BODY_STATE) + ("cr", "cg", "cb", "rad", "shown") + F.DIAGNOSTICS + F.BODY_DIAGNOSTICS
    assert "bth" not in F.initial_columns(16) and "bth" in F.initial_columns(16, shape="waterwheel")
    assert F.OUTPUTS == tuple(f"{s}_next" for s in F.STATE) + ("cr", "cg", "cb", "rad", "shown") + F.DIAGNOSTICS            # the default law's signature is unchanged
    assert "waterwheel" in dict(F.CHAMBER_CHOICES)
    assert B.VARIANTS["wheel"] == ("fountain3d_wheel", False) and "10M" not in B.WHEEL_SIZES and "4k" in B.WHEEL_SIZES
    assert B.layout("wheel")["state"] == F.STATE + F.BODY_STATE and B.layout("flat")["state"] == F.STATE
    u = F.uniforms(n=256, shape="waterwheel")
    assert float(u["sdf_on"]) == 0.0 and float(u["body_np"]) == 8.0 and float(u["body_i"]) > 0.0 and float(u["bar_half"]) == pytest.approx(F.WHEEL_NOZZLE_BAR * 0.25, rel=1e-5)


def test_the_3d_waterwheel_spins_the_way_the_stream_hits_it():
    for side, sign in ((+1.0, -1.0), (-1.0, +1.0)):                     # a stream on the right half turns it clockwise, on the left counterclockwise
        world = _WheelWorld(16)
        bx, by, reach = (float(world.cols[k]) for k in ("body_x", "body_y", "body_pl"))
        world.place({i: (bx + side * (0.45 * reach + 0.012 * (i % 2)), by + reach + 0.02 + 0.025 * i, 0.0, 0.0, -2.5) for i in range(6)})
        _state, series, out = world.steps(45)
        assert sign * series[-1, 1] > 0.05, (side, series[-1])
        assert np.isfinite(series).all() and float(out["n_nan"][0]) == 0


def test_a_grain_hitting_a_paddle_changes_the_3d_wheel_by_impulse_times_arm_over_inertia():
    world = _WheelWorld(16, grav_x=0.0, grav_y=0.0, body_gam=0.0)
    bx, by, reach, inertia = (float(world.cols[k]) for k in ("body_x", "body_y", "body_pl", "body_i"))
    world.place({0: (bx + 0.6 * reach, by + 0.05, 0.0, 0.0, -2.0)})
    state = {k: np.array(world.cols[k], dtype=np.float64) for k in F.STATE + F.BODY_STATE}

    def angular_momentum(st, rad):
        mass = [s[2] for s in F.SPECIES if abs(s[1] - rad) < 1.0e-9][0] * (4.0 / 3.0) * np.pi * rad ** 3
        return mass * ((st["x"][0] - bx) * st["vy"][0] - (st["y"][0] - by) * st["vx"][0]) + 0.4 * mass * rad * rad * st["wz"][0]
    first = {nm: np.asarray(v.numpy(), dtype=np.float64) for nm, v in zip(F.BODY_OUTPUTS, world.law(*[
        AbstractTensor.get_tensor(np.asarray(state[k] if k in state else world.cols[k], dtype=np.float64)) for k in F.BODY_ARGUMENTS]))}
    rad = float(np.broadcast_to(first["rad"], (16,))[0])
    before = angular_momentum(state, rad)
    final, _series, _ = world.steps(30)                       # the hit is over by step 10; the grain is then clear of the wheel
    after = angular_momentum(final, rad)
    delta_wheel = inertia * final["bom"][0]
    assert abs(delta_wheel) > 0.2 * abs(before)                                 # the paddle did take it
    assert abs(delta_wheel + (after - before)) < 0.06 * abs(before), (delta_wheel, after - before)     # the pair's angular momentum about the hinge axis is conserved (to the air drag)


def test_a_3d_motor_torque_drives_the_wheel_to_the_semi_implicit_terminal_speed_and_body_tau_adds_its_impulse():
    gam, alpha = 100.0, 40.0
    world = _WheelWorld(8, body_alpha=alpha, body_gam=gam)
    _s, series, _ = world.steps(70)
    # applied torque first, then the hinge impulse (engine_toy's order): omega = alpha (1 - gam DT) / gam, which is alpha / gam to O(gam DT)
    assert abs(series[-1, 1] - alpha * (1.0 - gam * F.DT) / gam) < 1.0e-6 and abs(series[-1, 1] - series[-2, 1]) < 1.0e-6
    free = _WheelWorld(8, body_alpha=0.0, body_gam=gam)                           # free: it stays at rest
    assert np.all(free.steps(3)[1][:, 1] == 0.0)
    driven = _WheelWorld(8, body_gam=0.0, body_tau=0.002)                         # an absolute torque from a machine-sim coupling: omega = tau / I dt
    om = driven.steps(1)[1][0, 1]
    assert om == pytest.approx(0.002 / float(driven.cols["body_i"]) * F.DT, rel=1e-6)


def test_the_3d_wheel_replay_is_bit_identical_and_every_id_carries_the_same_body_state():
    world = _WheelWorld(16)
    bx, by, reach = (float(world.cols[k]) for k in ("body_x", "body_y", "body_pl"))
    world.place({i: (bx + 0.4 * reach, by + reach + 0.02 + 0.025 * i, 0.0, 0.0, -2.0) for i in range(6)})
    one = world.steps(35)
    two = world.steps(35)
    assert all(np.array_equal(one[0][k], two[0][k]) for k in one[0]) and np.array_equal(one[1], two[1])
    assert np.all(one[0]["bth"] == one[0]["bth"][0]) and np.all(one[0]["bom"] == one[0]["bom"][0])


def test_the_game_offers_the_waterwheel_with_its_wheel_controls_manifest_and_body_spec():
    from meowdoku import fountain_game as G
    game = G.FountainGame(size="small")
    assert game.state()["compute"]["body"] is None and game.state()["compute"]["manifest"] == "fountain3d_flat.manifest.json"
    game.do({"type": "fountain_choice", "name": "shape", "value": "waterwheel"})
    st = game.state()
    ids = {c["id"]: c for c in st["controls_ui"]}
    assert st["compute"]["manifest"] == "fountain3d_wheel.manifest.json" and st["compute"]["uniforms"]["sdf_on"] == 0.0
    assert ids["wheel_n"]["group"] == "Wheel" and ids["wheel_om"]["diag"] == "body_om" and ids["wheel_alpha"]["action"] == "fountain_param"
    body = st["compute"]["body"]
    assert body["paddles"] == 8 and body["angle"] == "body_th" and body["torque_input"] == "body_tau" and body["dh"] == pytest.approx(0.5 * st["compute"]["depth"], rel=1e-6)
    epoch = st["compute"]["epoch"]
    game.do({"type": "fountain_param", "name": "wheel_alpha", "value": 5.0})              # a runtime input: the same run
    assert game.state()["compute"]["epoch"] == epoch and game.state()["compute"]["uniforms"]["body_alpha"] == 5.0
    game.do({"type": "fountain_param", "name": "wheel_n", "value": 6.0})                  # what the body IS: a new run
    assert game.state()["compute"]["epoch"] == epoch + 1 and game.state()["compute"]["body"]["paddles"] == 6
    assert {"body_tau", "body_alpha", "body_gam", "body_i"} <= set(B.layout("wheel")["runtime_inputs"])


# ------------------------------------------------------------------------------------------------ glyph start (WORKLIST 4b)
from meowdoku import glyph_start as GS  # noqa: E402


def _glyph_overlap(start):
    """The deepest overlap of any two grains, in the smallest radius (negative: every pair has a gap)."""
    from scipy.spatial import cKDTree
    pairs = cKDTree(start.positions).query_pairs(3.0 * float(start.radii.max()), output_type="ndarray")
    if not len(pairs):
        return -1.0
    d = np.linalg.norm(start.positions[pairs[:, 0]] - start.positions[pairs[:, 1]], axis=1)
    return float((start.radii[pairs[:, 0]] + start.radii[pairs[:, 1]] - d).max() / start.radii.min())


@pytest.mark.parametrize("crystal", [False, True])
def test_glyph_start_places_exactly_n_grains_inside_the_glyph_volumes_without_overlap_on_every_shippable_size(crystal):
    for tag in ("small", "4k", "16k", "64k"):
        n = F.SIZES[tag][0]
        for mix in (0.0, 0.5):
            start = GS.build("MEOW", n, mix=mix, crystal=crystal)
            assert len(start.positions) == n and len(start.radii) == n                       # the count is N
            assert start.inside_glyphs(start.positions).all()                                # every centre is in a glyph's volume (holes count, |z| < depth/2)
            assert _glyph_overlap(start) < 0.0, (tag, mix)                                   # no two grains overlap at t = 0 (the lattice is built with a gap)
            assert start.phi <= start.phi_target + 1e-9 and 0.25 < start.phi                 # the achieved packing is reported, never above the target
            u = {k: float(v) for k, v in start.uniforms(mix=mix).items()}
            d = F.chamber_distance(u, *start.positions.T)[0]
            assert (d > start.radii).all()                                                   # inside the closed box with room


def test_glyph_start_is_a_pure_function_of_the_text_n_mix_and_option():
    a = GS.build("MEOW", 4096, crystal=False)
    GS._BUILDS.clear()
    b = GS.build("meow", 4096, crystal=False)                                                # upper-cased: the same text
    assert np.array_equal(a.positions, b.positions) and a.payload() == b.payload()
    assert not np.array_equal(GS.build("CAT", 4096).positions, a.positions)
    assert not np.array_equal(GS.build("MEOW", 4096, crystal=True).positions, a.positions)
    assert GS.normalize_text("  ") == GS.DEFAULT_TEXT and len(GS.normalize_text("x" * 99)) == GS.MAX_CHARS


def test_glyph_outlines_are_the_exports_even_odd_contours_and_holes_are_empty():
    o = GS.glyph("O")
    assert not o.inside(np.zeros((1, 2)))[0]                                                 # the counter of the O is outside
    assert o.inside(np.array([[max(float(c[:, 0].max()) for c in o.contours) - 0.01, 0.0]]))[0] and o.area > 0.0
    assert o.depth == pytest.approx(0.28 * GS.HEIGHT)
    try:                                                                                      # the export's front cap is triangulated from the same contours
        from meowdoku import phong_tiles, tiles
        tiles.SPECTRAL_ROOT = GS.spectral_root()                                              # (a worktree has the sibling checkout further up than tiles looks)
        meshes = {t: phong_tiles.token_mesh(t, "DejaVu Sans", "bold", depth_ratio=0.28, outline_subdivisions=1, cap_grid=8) for t in "OM"}
    except RuntimeError:
        pytest.skip("spectral-analyzer is not beside this checkout")
    for token, (verts, normals) in meshes.items():
        front = verts[normals[:, 2] > 0.99]
        cross = np.cross(front[:, 1] - front[:, 0], front[:, 2] - front[:, 0])[:, 2]
        cap = 0.5 * float(np.abs(cross).sum())
        # the cap is the outline's area where it has no concave slivers (the O: equal); the export's cap_grid=8 filter drops thin triangles of a concave
        # outline (the M), so there it is smaller, never larger
        assert cap <= GS.glyph(token).area * (1 + 2e-3)
        if token == "O":
            assert cap == pytest.approx(o.area, rel=2e-3)


def test_glyph_start_columns_are_placed_at_rest_with_nothing_to_emit_and_the_payload_round_trips():
    n = 256
    start = GS.build("MEOW", n, mix=0.0)
    cols = start.columns(n, mix=0.0)
    assert float(cols["emit_rate"]) == 0.0 and float(cols["sdf_on"]) == 1.0 and float(cols["lid"]) == F.LID_OFF
    assert (cols["vx"] == 0).all() and (cols["vy"] == 0).all() and (cols["wz"] == 0).all()
    got = GS.decode_payload(start.payload(), n)                                              # the page's decode (compute.js decodeStart) is this formula
    for c, name in enumerate("xyz"):
        assert np.array_equal(got[name].astype(np.float64), cols[name]) and np.array_equal(cols[name], start.positions[:, c])
    assert np.array_equal(cols["pid"], np.arange(n))


@pytest.mark.skipif(shutil.which("node") is None, reason="node runs the page's decode")
def test_the_pages_start_decode_equals_pythons(tmp_path):
    start = GS.build("MEOW", 256, mix=0.0)
    (tmp_path / "start.json").write_text(json.dumps(start.payload()))
    for name in ("compute.js", "recording.js", "particle_styles.js", "noise_filter.js"):
        shutil.copyfile(ROOT / "examples" / "meowdoku" / "web" / name, tmp_path / name)
    shutil.copyfile(ROOT / "examples" / "mobile_reference_app" / "web" / "turing-wasm-shell.js", tmp_path / "turing-wasm-shell.js")
    (tmp_path / "run.mjs").write_text("""
import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
const [dir] = process.argv.slice(2);
const {decodeStart} = await import(pathToFileURL(`${dir}/compute.js`).href);
const out = decodeStart(JSON.parse(fs.readFileSync(`${dir}/start.json`, 'utf8')), 256);
fs.writeFileSync(`${dir}/out.bin`, Buffer.concat(['x', 'y', 'z'].map((k) => Buffer.from(out[k].buffer))));
""")
    done = subprocess.run(["node", str(tmp_path / "run.mjs"), str(tmp_path)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-1500:]
    got = np.frombuffer((tmp_path / "out.bin").read_bytes(), dtype="<f4").reshape(3, 256).T.astype(np.float64)
    assert np.array_equal(got, start.positions)


def test_the_law_reads_a_glyph_start_as_live_grains_with_the_radii_the_host_packed():
    """One eager step: every id is live (none parked, none emitted late), nothing is outside the closed box, the law's own radius of each id is the
    one the packing used, and the grains are at rest to within one step of gravity (no polygon wall or inlet touches them)."""
    n = 256
    start = GS.build("MEOW", n, mix=0.0)
    cols = start.columns(n, mix=0.0)
    _nxt, got = _eager_step(F.make_law(n, n), cols, n)
    assert (got["n_live"][0], got["n_parked"][0], got["n_out"][0], got["n_nan"][0]) == (n, 0, 0, 0)
    assert np.abs(got["rad"] - start.radii).max() < 1e-6 * start.radii.max()
    assert float(np.sqrt(got["vmax2"][0])) < 0.05


def test_the_server_offers_the_glyph_start_with_its_text_control_and_ships_the_placed_grains():
    from meowdoku import fountain_game as G
    game = G.FountainGame(size="small")
    assert game.state()["compute"]["start"] is None
    game.do({"type": "fountain_choice", "name": "shape", "value": G.GLYPH_SHAPE})
    state = game.state()
    ids = {c["id"]: c for c in state["controls_ui"]}
    assert ids["glyph_text"]["kind"] == "text" and ids["glyph_text"]["value"] == "MEOW" and ids["glyph_text"]["action"] == "fountain_glyph"
    assert "inlet" not in ids and ids["glyph_crystal"]["kind"] == "toggle"
    c = state["compute"]
    assert c["manifest"] == "fountain3d.manifest.json" and c["chamber_mesh"] is not None and c["uniforms"]["emit_rate"] == 0.0
    assert c["start"]["n"] == 256 and c["uniforms"]["sdf_on"] == 1.0
    mesh_v, mesh_t = G.unpack_mesh(c["chamber_mesh"])
    assert len(mesh_v) == 8 and len(mesh_t) == 12                                           # the closed box, drawn through the housing's glass
    before = c["epoch"]
    game.do({"type": "fountain_glyph", "text": "cat"})                                       # changing the text restarts with a different start
    after = game.state()["compute"]
    assert after["epoch"] == before + 1 and after["start"]["xyz"] != c["start"]["xyz"] and game.glyph_text == "CAT"
    game.do({"type": "fountain_size", "size": "256k"})                                       # more ids than the state ships: the box fountain, said on the status line
    big = game.state()
    assert big["compute"]["start"] is None and "choose a smaller size" in big["status"]



def test_under_gravity_the_letters_collapse_into_a_pile_inside_the_closed_box_and_nothing_leaves_it():
    """Eager NumPy lane (the native 3-D compile is out of reach for a test): 64 grains spell "MI" in a closed box and fall under the default gravity.
    The string's top comes down, the centre of mass drops, every grain stays in the box (n_out = 0, all live), and the pile is a pile (the column heights shrink)."""
    n, steps = 64, 150
    start = GS.build("MI", n, mix=0.0)
    cols = start.columns(n, mix=0.0)
    law = F.make_law(n, n)
    top0, com0 = float(cols["y"].max()), float(cols["y"].mean())
    worst_out = 0.0
    for _ in range(steps):
        cols, got = _eager_step(law, cols, n)
        worst_out = max(worst_out, float(got["n_out"][0]))
    assert worst_out == 0.0 and (got["n_live"][0], got["n_parked"][0], got["n_nan"][0]) == (n, 0, 0)
    assert float(cols["y"].max()) < 0.75 * top0 and float(cols["y"].mean()) < 0.7 * com0           # the letters came down
    u = {k: float(v) for k, v in start.uniforms(mix=0.0).items()}
    assert (F.chamber_distance(u, cols["x"], cols["y"], cols["z"])[0] > -0.2 * float(got["rad"].max())).all()   # inside the closed box



def test_the_armed_queue_is_bounded_so_a_blocked_inlet_cannot_pile_ids_past_the_neighbour_walk():
    """As the 2-D law's: with every id due at once, only the ids the schedule emits in ``QUEUE_STEPS`` steps arm; the rest stay parked."""
    n = 1024
    columns = F.initial_columns(n, shape="sdf_toy")
    piece = F.make_piece(n)
    state = {k: np.zeros(n) for k in F.STATE}
    state["age"] = np.full(n, 100.0)
    produced = dict(zip(F.OUTPUTS, piece(*[state[k] if k in state else columns[k] for k in F.ARGUMENTS])))
    queue = max(1, int(n * F.QUEUE_STEPS * F.DT / F.EMIT_SPAN))
    armed = np.asarray(produced["vy_next"]) < F.ARMED * 0.5
    assert 0 < armed.sum() <= queue, (int(armed.sum()), queue)
    assert armed[:queue].all() and not armed[queue:].any()
