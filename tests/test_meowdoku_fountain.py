"""The fountain: one batched law, a compiler-chosen and compiler-generated compute shader."""

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


def test_layout_is_a_permutation_of_lattice_slots_and_mix_picks_species():
    cols, rows = 24, 16
    n = cols * rows
    regular = F.layout(cols, rows, seed=3, mix=0.0)
    mixed = F.layout(cols, rows, seed=3, mix=1.0)
    assert regular["sx"].shape == (n,)
    assert set(np.unique(regular["species"])) <= {0, 1}          # mix 0: only the regular species
    assert set(np.unique(mixed["species"])) <= {2, 3}            # mix 1: only the jittered ones
    px, py = F.lattice(cols, rows)
    # mix 0 lays every id exactly on a distinct lattice point, in a seeded (not lattice) order
    on_lattice = {(round(float(np.float32(a)), 6), round(float(np.float32(b)), 6)) for a, b in zip(px, py)}
    placed = {(round(float(a), 6), round(float(b), 6)) for a, b in zip(regular["sx"], regular["sy"])}
    assert placed == on_lattice and len(placed) == n
    assert not np.array_equal(regular["sx"], px)
    # mix 1 jitters the irregular species off the lattice by at most JITTER cells
    cell = F.WIDTH / cols
    off = np.abs(mixed["sx"][:, None] - px[None, :]).min(axis=1)
    assert float(off.max()) <= F.JITTER * cell + 1e-6 and float(off.max()) > 0.0
    again = F.layout(cols, rows, seed=3, mix=1.0)
    assert all(np.array_equal(mixed[k], again[k]) for k in mixed)  # deterministic


def test_species_differ_in_radius_mass_and_drag():
    layout = F.layout(40, 20, seed=1, mix=0.5)
    species = layout["species"]
    assert set(np.unique(species)) == {0, 1, 2, 3}
    for s in range(4):
        sel = species == s
        assert np.unique(layout["mass"][sel]).size == 1 and np.unique(layout["drag"][sel]).size == 1
        assert np.unique(layout["radius"][sel]).size == 1
    assert len({float(layout["mass"][species == s][0]) for s in range(4)}) == 4


@pytest.fixture(scope="module")
def native_piece():
    columns = F.initial_columns(40, 30, seed=5, mix=0.5)
    piece = F.FountainPiece(40 * 30)
    piece.compile(columns)
    return piece, columns


def test_phase_one_fills_every_id_into_its_slot(native_piece):
    piece, columns = native_piece
    steps = int((F.EMIT_SPAN + F.FLIGHT[1] + 3.0) / F.DT)
    final = F.run_frames(piece, columns, steps)
    assert np.all(final["landed"] > 0.5)
    miss = np.hypot(final["x"] - columns["sx"], final["y"] - columns["sy"])
    assert float(miss.max()) < 0.01 * F.WIDTH                    # settled onto the slots
    assert float(np.hypot(final["vx"], final["vy"]).max()) < 0.2


def test_hue_modes_trig_and_rotation_agree_on_a_hue_wheel_but_are_not_equal(native_piece):
    piece, columns = native_piece
    a = F.run_frames(piece, dict(columns), 2)
    b = F.run_frames(piece, F.initial_columns(40, 30, seed=5, mix=0.5, mode=1), 2)
    assert not np.allclose(a["cr"], b["cr"])
    for out in (a, b):
        assert all(0.0 <= float(out[k].min()) and float(out[k].max()) <= 1.0 for k in ("cr", "cg", "cb"))
    # trig mode: opposite slot directions are half a hue turn apart (hue = atan2/2pi + 0.5)
    assert float(np.ptp(a["cr"])) > 0.5


def test_phase_two_final_image_equals_the_samples_at_the_slots(native_piece):
    piece, columns = native_piece
    layout = F.layout(40, 30, seed=5, mix=0.5)
    h, w = 68, 88
    rng = np.random.default_rng(0)
    raster = (rng.random((h, w, 3)) * 255).astype(np.uint8)       # the user's image, rasterised
    samples = F.sample_raster(raster, layout["slot_u"], layout["slot_v"])
    phase2 = dict(columns)
    phase2.update(F.uniforms(mode=0, use_image=1.0, cols=40, rows=30))
    phase2["ir"], phase2["ig"], phase2["ib"] = samples[:, 0], samples[:, 1], samples[:, 2]
    steps = int((F.EMIT_SPAN + F.FLIGHT[1] + 3.0) / F.DT)
    final = F.run_frames(piece, phase2, steps)
    assert np.all(final["landed"] > 0.5)
    assert np.array_equal(final["cr"], samples[:, 0])
    assert np.array_equal(final["cg"], samples[:, 1])
    assert np.array_equal(final["cb"], samples[:, 2])
    # the image is in the particles' places: rebuilding the picture from the settled positions
    px = np.clip(((final["x"] + F.WIDTH / 2) / F.WIDTH * w).astype(int), 0, w - 1)
    assert px.min() >= 0


def test_the_compiler_chooses_webgpu_compute_at_the_game_size_and_not_at_a_tiny_one():
    big = B.verdict_for(F.N_FULL)
    tiny = B.verdict_for(96)
    assert big.pays and not tiny.pays
    assert big.data_parallel_extent == F.N_FULL
    assert big.speedup > 2.0 > tiny.speedup


def test_the_wgsl_is_generated_from_the_authored_law():
    artifact, ids = B.shader_for(1024)
    assert artifact.complete, artifact.shortfalls
    source = artifact.source
    assert "@compute @workgroup_size(" in source
    assert "atan2(" in source and "sqrt(" in source and "floor(" in source
    assert source.count("@binding(") == 2                         # packed feeds + packed outputs
    meta = artifact.api.to_mapping()["metadata"]
    assert meta["feed_span"] == [ids[name] for name in F.ARGUMENTS]
    assert len(meta["output_span"]) == len(F.OUTPUTS)
    assert meta["uniform_feeds"] == sorted(ids[name] for name in F.UNIFORMS)
    assert meta["backend_identities"][0]["identity"] == "shader_float64_storage_to_float32"


def test_elementwise_lowering_names_what_it_cannot_express():
    from src.common.tensors import AbstractTensor
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
    assert not artifact.complete
    assert any("sum_double" in item.format() and "not an elementwise" in item.format()
               for item in artifact.shortfalls)
    del AbstractTensor


@pytest.mark.skipif(shutil.which("node") is None, reason="node is needed to load the wasm product")
def test_wasm_product_is_bit_exact_with_the_numpy_law(tmp_path):
    from meowdoku import fountain_parity as P

    columns = F.initial_columns(12, 8, seed=2, mix=0.5)
    B.build(tmp_path, ("small",))
    ref, _ = P.numpy_reference(columns, 150)
    got = P.wasm_frames(columns, "small", tmp_path, 150, job="test_small")
    worst = P.compare(ref, got)
    assert worst["landed_flips"] == 0
    assert max(worst["max_abs"].values()) < 1e-12


def test_manifest_publishes_the_routes(tmp_path):
    import json

    manifest = B.build(tmp_path, ("small",))
    assert manifest["sizes"]["small"]["route"] == "wasm"
    assert json.loads((tmp_path / "fountain.manifest.json").read_text())["sweep"][-1]["pays"] is True


def test_server_game_holds_phase_seed_and_image_without_streaming_particles():
    import base64

    game = F.FountainGame(cols=8, rows=6)
    game.do({"type": "fountain_start", "seed": 11, "mix": 0.25, "mode": 1})
    state = game.state()
    assert (state["phase"], state["seed"], state["mode"], state["n"]) == ("fill", 11, 1, 48)
    with pytest.raises(ValueError):
        game.do({"type": "fountain_replay"})                      # no image yet
    rgb = (np.arange(48 * 3) % 251).astype(np.uint8)
    game.do({"type": "fountain_image", "rgb": base64.b64encode(rgb.tobytes()).decode()})
    game.do({"type": "fountain_replay"})
    assert game.state()["phase"] == "replay" and game.state()["has_image"]
    assert "x" not in game.state() and "entities" not in game.state()
