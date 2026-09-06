import os, sys
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

import numpy as np
import pytest

pytestmark = pytest.mark.operators

from src.hardware.analog_spec import (
    BiosHeader,
    InstructionWord,
    Opcode,
    generate_bit_wave,
    header_frames,
    dominant_tone,
    pack_instruction_word,
    unpack_instruction_word,
)
from src.compiler.tape_compiler import TapeCompiler
from src.hardware.cassette_tape import CassetteTapeBackend
from src.turing_machine.tape_machine import TapeMachine
from src.turing_machine.tape_map import TapeMap


def make_machine():
    tape = CassetteTapeBackend(
        tape_length=128,
        time_scale_factor=0.0,
        play_audio=False,
    )
    machine = TapeMachine(tape, bit_width=1)
    machine.data_registers = {0: 0, 1: 10, 2: 20, 3: 30}
    return machine, tape


def test_instruction_codec_preserves_registers_and_parameter():
    instruction = InstructionWord(
        Opcode.MU, reg_a=1, reg_b=2, dest=3, param=63,
    )

    word = pack_instruction_word(instruction)
    frame, = TapeCompiler.binarize_instructions([instruction])

    assert sum(bit << (15 - index) for index, bit in enumerate(frame)) == word
    assert unpack_instruction_word(word) == instruction


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


def test_load_store_use_fixed_width_spill_slots_after_registers():
    tape = CassetteTapeBackend(tape_length=128, time_scale_factor=0.0)
    machine = TapeMachine(tape, bit_width=4)
    machine.tape_map = TapeMap(bios=None, instruction_frames=0)
    machine.data_registers = {0: 0, 1: 4, 2: 8}
    source = [generate_bit_wave(1, 0) for _ in range(4)]
    machine.transport[0:4] = source

    machine._execute(Opcode.STORE, dest=0, reg_a=0, reg_b=0, param=2)
    machine._execute(Opcode.LOAD, dest=1, reg_a=0, reg_b=0, param=2)

    out = machine.transport[4:8]
    assert all(dominant_tone(frame).amp > 0.0 for frame in out)
    spill_start = (3 + 2) * 4
    assert all((0, 0, spill_start + index) in tape._tape_frames for index in range(4))
    tape.close()


def encode_instruction(word: int, frame_idx: int, tape: CassetteTapeBackend) -> None:
    for lane in range(16):
        bit = (word >> (15 - lane)) & 1
        tape.write_bit(0, lane, frame_idx, bit)


def test_run_halts_on_halt_opcode():
    tape = CassetteTapeBackend(
        tape_length=256,
        time_scale_factor=0.0,
        play_audio=False,
    )
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
