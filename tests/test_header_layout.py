import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.hardware.analog_spec import (
    BiosHeader,
    BIOS_HEADER_STRUCT,
    MAGIC_ID,
    header_frames,
    pack_bios_header,
    unpack_bios_header,
    LANES,
)
from src.turing_machine.tape_map import TapeMap, create_register_tapes


def make_header():
    return BiosHeader(
        calib_fast_ms=1.0,
        calib_read_ms=2.0,
        drift_ms=0.1,
        inputs=[0, 1],
        outputs=[2],
        instr_start_addr=1234,
    )


def bits_to_bytes(frames):
    bits = [b for frame in frames for b in frame]
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for bit in bits[i : i + 8]:
            byte = (byte << 1) | bit
        out.append(byte)
    return bytes(out)


def test_header_pack_roundtrip():
    h = make_header()
    packed = pack_bios_header(h)
    assert len(packed) == BIOS_HEADER_STRUCT.size
    unpacked = unpack_bios_header(packed)
    assert unpacked.calib_fast_ms == pytest.approx(h.calib_fast_ms)
    assert unpacked.inputs == h.inputs
    assert unpacked.outputs == h.outputs
    assert unpacked.instr_start_addr == h.instr_start_addr


def test_header_serialization_bits():
    h = make_header()
    frames = header_frames(h)
    assert frames and all(len(frame) == LANES for frame in frames)
    reconstructed = bits_to_bytes(frames)[: len(MAGIC_ID)]
    assert reconstructed == MAGIC_ID


def test_tape_map_layout():
    h = make_header()
    tmap = TapeMap(h, instruction_frames=10)
    assert tmap.instr_start == len(header_frames(h))
    assert tmap.data_start == tmap.instr_start + 10


def test_tape_map_bios_roundtrip():
    h = make_header()
    tmap = TapeMap(h, instruction_frames=0)
    frames = tmap.encode_bios()
    decoded = TapeMap.decode_bios(frames)
    assert decoded.instr_start_addr == h.instr_start_addr


def test_register_tapes_independent():
    """Registers are pure data tapes with no BIOS or instruction table."""
    regs = create_register_tapes()
    assert set(regs.keys()) == {0, 1, 2}
    for r in regs.values():
        assert r.instr_start == 0
        assert r.data_start == 0
        assert r.is_register

