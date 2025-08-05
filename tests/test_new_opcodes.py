import os, sys
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

import numpy as np
import pytest

pytestmark = pytest.mark.operators

from src.hardware.analog_spec import (
    BiosHeader,
    Opcode,
    generate_bit_wave,
    header_frames,
    dominant_tone,
)
from src.hardware.cassette_tape import CassetteTapeBackend
from src.turing_machine.tape_machine import TapeMachine
from src.turing_machine.tape_map import TapeMap


def make_machine():
    tape = CassetteTapeBackend(tape_length=128)
    machine = TapeMachine(tape, bit_width=1)
    machine.data_registers = {0: 0, 1: 10, 2: 20, 3: 30}
    return machine, tape


def test_seek_opcode_moves_head_and_register():
    machine, tape = make_machine()
    machine._execute(Opcode.SEEK, dest=0, reg_a=0, reg_b=0, param=5)
    assert machine.data_registers[0] == 5
    assert machine.transport._cursor == 5
    tape.close()


def test_read_opcode_copies_frames():
    machine, tape = make_machine()
    src = [generate_bit_wave(1, 0), generate_bit_wave(1, 1)]
    machine.transport[:2] = src
    machine._execute(Opcode.READ, dest=1, reg_a=0, reg_b=0, param=2)
    out = machine.transport[10:12]
    assert dominant_tone(out[0]).bin == dominant_tone(src[0]).bin
    assert dominant_tone(out[1]).bin == dominant_tone(src[1]).bin
    tape.close()


def test_write_opcode_copies_from_reg_b():
    machine, tape = make_machine()
    src = [generate_bit_wave(1, 2), generate_bit_wave(1, 3)]
    machine.transport[20:22] = src
    machine._execute(Opcode.WRITE, dest=0, reg_a=0, reg_b=2, param=2)
    out = machine.transport[0:2]
    assert dominant_tone(out[0]).bin == dominant_tone(src[0]).bin
    assert dominant_tone(out[1]).bin == dominant_tone(src[1]).bin
    tape.close()


def encode_instruction(word: int, frame_idx: int, tape: CassetteTapeBackend) -> None:
    for lane in range(16):
        bit = (word >> (15 - lane)) & 1
        tape.write_bit(0, lane, frame_idx, bit)


def test_run_halts_on_halt_opcode():
    tape = CassetteTapeBackend(tape_length=256)
    machine = TapeMachine(tape, bit_width=1)

    bios = BiosHeader(1.0, 1.0, 0.0, [], [], 0)
    frames = header_frames(bios)
    for idx, frame in enumerate(frames):
        for lane, bit in enumerate(frame):
            tape.write_bit(0, lane, idx, bit)

    tmap = TapeMap(bios, instruction_frames=2)
    halt_word = Opcode.HALT.value << 12
    nand_word = Opcode.NAND.value << 12
    encode_instruction(halt_word, tmap.instr_start, tape)
    encode_instruction(nand_word, tmap.instr_start + 1, tape)

    machine.run(2)
    assert machine.halted is True
    assert machine.instruction_pointer == tmap.instr_start + 1
    tape.close()
