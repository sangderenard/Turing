import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest

import src.hardware.nand_wave as nw


ENERGY_THRESH = 0.01


def test_parallel_xor_copy():
    lane = 5
    x = nw.generate_bit_wave(lane)
    y = np.zeros(nw.FRAME_SAMPLES, dtype="f4")
    out = nw.nand_wave(x, y, mode="parallel", lane_mask=1 << lane, energy_thresh=ENERGY_THRESH)
    corr = np.corrcoef(out, x)[0, 1]
    assert corr == pytest.approx(1.0, rel=2e-2)


def test_parallel_nor_generates_fresh_one():
    lane = 2
    x = np.zeros(nw.FRAME_SAMPLES, dtype="f4")
    y = np.zeros_like(x)
    out = nw.nand_wave(x, y, mode="parallel", lane_mask=1 << lane, energy_thresh=ENERGY_THRESH)
    expected = nw.generate_bit_wave(lane)
    corr = np.corrcoef(out, expected)[0, 1]
    assert corr == pytest.approx(1.0, rel=1e-2)


def test_parallel_both_on_silence():
    lane = 4
    x = nw.generate_bit_wave(lane)
    y = nw.generate_bit_wave(lane)
    out = nw.nand_wave(x, y, mode="parallel", lane_mask=1 << lane, energy_thresh=ENERGY_THRESH)
    assert np.max(np.abs(out)) < 1e-6


def test_dominant_xor_envelope_replay():
    src_lane = 3
    target_lane = 7
    x = nw.generate_bit_wave(src_lane)
    y = np.zeros(nw.FRAME_SAMPLES, dtype="f4")
    out = nw.nand_wave(x, y, mode="dominant", target_lane=target_lane, energy_thresh=ENERGY_THRESH)
    env = nw.extract_lane(x, src_lane)
    expected = nw.replay_envelope(env, target_lane)
    corr = np.corrcoef(out, expected)[0, 1]
    assert corr == pytest.approx(1.0, rel=1e-2)


def test_dominant_nor_generates_fresh_one():
    target_lane = 1
    x = np.zeros(nw.FRAME_SAMPLES, dtype="f4")
    y = np.zeros_like(x)
    out = nw.nand_wave(x, y, mode="dominant", target_lane=target_lane, energy_thresh=ENERGY_THRESH)
    expected = nw.generate_bit_wave(target_lane)
    corr = np.corrcoef(out, expected)[0, 1]
    assert corr == pytest.approx(1.0, rel=1e-2)


def test_dominant_both_on_silence():
    lane = 6
    x = nw.generate_bit_wave(lane)
    y = nw.generate_bit_wave(lane)
    out = nw.nand_wave(x, y, mode="dominant", target_lane=lane, energy_thresh=ENERGY_THRESH)
    assert np.max(np.abs(out)) < 1e-6
