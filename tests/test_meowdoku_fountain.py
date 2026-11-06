"""The fountain: a DEM granular law, a compiler-chosen and compiler-generated compute program."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from meowdoku import fountain as F  # noqa: E402
from meowdoku import fountain_build as B  # noqa: E402


def test_layout_is_deterministic_and_mix_picks_the_species():
    n = 256
    mono = F.layout(n, seed=3, mix=0.0)
    poly = F.layout(n, seed=3, mix=1.0)
    assert set(np.unique(mono["species"])) <= {0, 1}               # mix 0: grains and beads only
    assert set(np.unique(poly["species"])) == {0, 1, 2, 3}         # mix 1: all four
    again = F.layout(n, seed=3, mix=1.0)
    assert all(np.array_equal(poly[k], again[k]) for k in poly)
    for s in range(4):
        sel = poly["species"] == s
        assert np.unique(poly["radius"][sel]).size == 1 and np.unique(poly["mass"][sel]).size == 1
    assert len({float(poly["mass"][poly["species"] == s][0]) for s in range(4)}) == 4
    assert np.array_equal(poly["pid"], np.arange(n))
    # an emitter, not slots: ids start parked at the nozzle bar
    assert np.all(np.abs(poly["x"] - F.NOZZLE[0]) <= 0.8 + 1e-6) and np.all(poly["vx"] == 0)


@pytest.fixture(scope="module")
def native_piece():
    n = F.N_SMALL
    columns = F.initial_columns(n, seed=5, mix=0.5)
    piece = F.make_piece(n)
    piece.compile(columns)
    return piece, columns


def test_the_pile_forms_and_settles_by_contact_forces(native_piece):
    piece, columns = native_piece
    steps = int(F.SETTLE_SECONDS / F.DT)
    final = F.run_frames(piece, columns, steps)
    geo = F.basin(F.N_SMALL)
    assert not np.isnan(final["x"]).any()
    assert np.all(np.abs(final["x"]) <= geo["half"] + 1e-3) and float(final["y"].min()) > -1e-3
    assert float(np.hypot(final["vx"], final["vy"]).max()) < 0.2          # at rest
    # packing fraction of the settled pile (disks), by its 99th-percentile top
    top = float(np.percentile(final["y"], 99))
    fraction = float((np.pi * columns["radius"] ** 2).sum() / (geo["width"] * top))
    assert 0.55 < fraction < 0.95
    # no particle sits inside another by more than a fraction of a radius: contacts push apart
    x, y, r = final["x"], final["y"], columns["radius"]
    gap = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :]) - (r[:, None] + r[None, :])
    np.fill_diagonal(gap, 1.0)
    assert float(gap.min()) > -0.6 * float(r.max())


def test_the_replay_is_bit_identical_in_the_native_lane(native_piece):
    piece, columns = native_piece
    a = F.run_frames(piece, columns, 900)
    b = F.run_frames(piece, columns, 900)
    assert all(np.array_equal(a[k], b[k]) for k in F.STATE)


def test_phase_two_final_image_equals_the_samples_at_the_rest_positions(native_piece):
    piece, columns = native_piece
    steps = int(F.SETTLE_SECONDS / F.DT)
    phase1 = F.run_frames(piece, columns, steps)
    u, v = F.fit_extent(phase1["x"], phase1["y"])
    rng = np.random.default_rng(0)
    raster = (rng.random((72, 96, 3)) * 255).astype(np.uint8)
    samples = F.sample_raster(raster, u, v)
    phase2 = dict(columns)
    phase2.update(F.uniforms(mode=0, use_image=1.0))
    phase2["ir"], phase2["ig"], phase2["ib"] = samples[:, 0], samples[:, 1], samples[:, 2]
    final = F.run_frames(piece, phase2, steps)
    # the identical simulation: every id comes to rest where it did in phase 1
    for k in ("x", "y"):
        assert np.array_equal(final[k], phase1[k])
    assert np.array_equal(final["cr"], samples[:, 0])
    assert np.array_equal(final["cg"], samples[:, 1])
    assert np.array_equal(final["cb"], samples[:, 2])


def test_hue_modes_trig_and_rotation_differ_and_stay_in_range(native_piece):
    piece, columns = native_piece
    a = F.run_frames(piece, columns, 2)
    b = F.run_frames(piece, F.initial_columns(F.N_SMALL, seed=5, mix=0.5, mode=1), 2)
    assert not np.allclose(a["cr"], b["cr"])
    for out in (a, b):
        assert all(0.0 <= float(out[k].min()) and float(out[k].max()) <= 1.0 for k in ("cr", "cg", "cb"))


def test_the_compiler_chooses_webgpu_compute_at_the_game_size_and_not_at_a_tiny_one():
    big, tiny = B.verdict_for(F.N_FULL), B.verdict_for(64)
    assert big.pays and not tiny.pays
    assert big.data_parallel_extent == F.N_FULL
    assert big.speedup > 2.0 > tiny.speedup


def test_the_wgsl_program_is_generated_from_the_authored_law():
    compiled, _program, _outputs, ids = B.passes_for(256)
    assert compiled.complete, compiled.shortfalls
    # the bitonic sort is a chain of dispatches (one per compare-exchange stage); the search
    # and the contact loop gather from finished values and fuse into the last one
    assert compiled.dispatches == 8 * 9 // 2 + 2
    assert "atan2(" in compiled.source and "tanh(" in compiled.source and "arena[" in compiled.source
    assert compiled.source.count("@binding(") == 1                   # one arena
    assert sorted(compiled.feed_slots) == sorted(ids[name] for name in F.ARGUMENTS)
    assert compiled.slots < 60                                        # slots are reused
    assert compiled.backend_identity_decisions[0].identity == "shader_float64_storage_to_float32"


def test_elementwise_lowering_names_what_it_cannot_express():
    from src.common.tensors.accelerator_backends.ssa_backend import SSATensorOperations, SSATensorProgram
    from src.compiler import ssa_webgpu_backend as webgpu
    from src.compiler.work_contract import PRESETS, scoped_work_contract

    with scoped_work_contract(PRESETS["fast"]):
        program = SSATensorProgram("reduces")
        x = SSATensorOperations.input(program, (8,))
        with program.activate():
            total = (x * 2.0).sum()
        artifact = webgpu.emit_tensor_program_module(
            program.module, "reduces", [total.data.value], count=8)
        passes = webgpu.emit_tensor_program_passes(
            program.module, "reduces", [total.data.value], count=8)
    for compiled in (artifact, passes):
        assert not compiled.complete
        assert any("sum_double" in item.format() and "not an elementwise" in item.format()
                   for item in compiled.shortfalls)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is needed to load the wasm product")
def test_wasm_product_agrees_with_the_native_law_over_a_short_horizon(tmp_path, native_piece):
    from meowdoku import fountain_parity as P

    piece, columns = native_piece
    B.build(tmp_path, ("small",), sweep=False)
    ref = F.run_frames(piece, columns, 120)
    got = P.wasm_frames(columns, "small", tmp_path, 120, job="test_small")
    worst = P.compare(ref, got)
    assert max(worst["max_abs"].values()) < 1e-6


def test_server_game_declares_a_compute_product_and_controls_without_streaming_particles():
    import base64

    game = F.FountainGame(n=64)
    before = game.state()["compute"]["epoch"]
    game.do({"type": "fountain_start", "seed": 11, "mix": 0.25, "mode": 1})
    state = game.state()
    assert (state["phase"], state["seed"], state["mode"], state["n"]) == ("fill", 11, 1, 64)
    assert state["board"] == "free" and state["entities"] == [] and state["input"] == "compute"
    assert state["compute"]["epoch"] == before + 1
    assert state["compute"]["uniforms"] == {"mode": 1.0, "use_image": 0.0}
    assert "n=64&seed=11&mix=0.25" in state["compute"]["layout"]
    ids = {item["id"]: item for item in state["controls_ui"]}
    assert ids["replay"]["enabled"] is False and ids["image"]["kind"] == "file"
    with pytest.raises(ValueError):
        game.do({"type": "fountain_replay"})                      # no image yet
    rgb = (np.arange(64 * 3) % 251).astype(np.uint8)
    game.do({"type": "fountain_image", "rgb": base64.b64encode(rgb.tobytes()).decode()})
    assert {i["id"]: i for i in game.state()["controls_ui"]}["replay"]["enabled"] is True
    game.do({"type": "fountain_replay"})
    state = game.state()
    assert state["phase"] == "replay" and state["compute"]["uniforms"]["use_image"] == 1.0
    assert state["compute"]["epoch"] == before + 2
    game.do({"type": "fountain_size", "size": "small"})
    assert game.state()["n"] == F.N_SMALL and not game.state()["has_image"]
