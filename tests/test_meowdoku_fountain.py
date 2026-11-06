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
    final = F.run_frames(piece, columns, int(F.SETTLE_SECONDS / F.DT))
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
    steps = int(F.SETTLE_SECONDS / F.DT)
    again = F.run_frames(piece, columns, 300)
    assert all(np.array_equal(again[k], F.run_frames(piece, columns, 300)[k]) for k in F.STATE)
    phase1 = F.run_frames(piece, columns, steps)
    u, v = F.fit_extent(phase1["x"], phase1["y"])
    raster = (np.random.default_rng(0).random((72, 96, 3)) * 255).astype(np.uint8)
    samples = F.sample_raster(raster, u, v)
    phase2 = dict(columns)
    phase2.update(F.uniforms(mode=0, use_image=1.0, mix=0.5))
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
    assert compiled.dispatches == len(F.merge_exchange_stages(F.SIZES["10M"][0])) + 2
    assert compiled.groups[1] > 1 and compiled.groups[0] * compiled.groups[1] * 64 >= pad     # a 2D grid
    assert ids["pid"] not in compiled.feed_slots                       # the iota is the invocation index
    assert sorted(compiled.feed_slots) == sorted(ids[n] for n in F.STATE + ("ir", "ig", "ib"))
    assert "lane0[" in compiled.source and "f32(i)" in compiled.source
    assert compiled.arena_elements * 4 / pad < 100                     # bytes per id (rematerialised sort temporaries hold no slot)
    # the arena is split across bindings of at most 256 MiB, none of the single-buffer limit's business
    assert compiled.bindings > 1 and compiled.slots_per_binding * pad * 4 <= 256 * 1024 * 1024
    assert compiled.bindings == -(-compiled.slots // compiled.slots_per_binding)


def test_the_wgsl_program_is_generated_from_the_authored_law():
    compiled, _program, _outputs, ids = B.passes_for("small")
    assert compiled.complete, compiled.shortfalls
    assert compiled.dispatches == len(F.merge_exchange_stages(256)) + 2
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
    assert state["compute"]["epoch"] == before + 1 and state["compute"]["uniforms"] == {"mode": 1.0, "use_image": 0.0, "mix": 0.25}
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
