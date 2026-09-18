# tape_machine.py
"""
The core execution engine for the survival computer.

The TapeMachine class simulates a physical processor that fetches, decodes,
and executes a binary instruction stream directly from a cassette tape. It operates
on analog waveforms, using the functions defined in analog_spec to perform
computations.
"""
from __future__ import annotations

from typing import Dict, List, Sequence
import os
import numpy as np

from ..hardware.analog_spec import (
    Opcode,
    apply_operator,
    mu,
    nand_wave,
    unpack_instruction_word,
)
from ..hardware.constants import (
    BIAS_AMP,
    BIT_FRAME_MS,
    FRAME_SAMPLES,
    LANES,
    REGISTERS,
    WRITE_BIAS,
)
from ..hardware.cassette_tape import CassetteTapeBackend
from .tape_map import TapeMap
from .tape_transport import TapeTransport
from .tape_visualizer import TapeVisualizer

class TapeMachine:
    """
    A simulated machine that reads instructions from a tape and executes them
    using analog wave-based operations.
    """
    def __init__(self, cassette: CassetteTapeBackend, bit_width: int, register_mode: bool = False):
        self.cassette = cassette
        self.bit_width = bit_width
        self.transport = TapeTransport(cassette, register_mode=register_mode)
        self.tape_map: TapeMap | None = None
        self.instruction_pointer = 0
        self.data_registers: Dict[int, int] = {}
        self.halted = False
        self.register_mode = register_mode
        self.visualizer: TapeVisualizer | None = None
        if os.environ.get("TAPE_VIZ"):
            self.visualizer = TapeVisualizer(self)

    def _boot(self, instruction_count: int) -> None:
        """Read BIOS and initialise register map."""

        print("TapeMachine: Booting...")
        bios_frames: List[List[int]] = []
        for i in range(TapeMap.get_bios_frame_count()):
            frame = [self.cassette.read_bit(0, lane, i) for lane in range(LANES)]
            bios_frames.append(frame)
        bios = TapeMap.decode_bios(bios_frames)
        self.tape_map = TapeMap(bios, instruction_frames=instruction_count)

        self.instruction_pointer = self.tape_map.instr_start
        print(
            f"TapeMachine: Boot successful. Instruction pointer set to frame {self.instruction_pointer}."
        )

        for i in range(16):
            self.data_registers[i] = self.tape_map.data_start + (i * self.bit_width)

    def _fetch_decode(self) -> tuple[Opcode, int, int, int, int]:
        """Fetch and decode one 16‑bit instruction."""

        bits = [self.cassette.read_bit(0, lane, self.instruction_pointer) for lane in range(16)]
        self.instruction_pointer += 1
        word = 0
        for bit in bits:
            word = (word << 1) | bit

        instruction = unpack_instruction_word(word)
        return (
            instruction.opcode,
            instruction.dest,
            instruction.reg_a,
            instruction.reg_b,
            instruction.param,
        )

    def _execute(
        self, opcode: Opcode, dest: int, reg_a: int, reg_b: int, param: int
    ) -> None:
        """Execute a single instruction via ``analog_spec`` operators."""

        addr_dest = self.data_registers[dest]
        addr_a = self.data_registers[reg_a]
        addr_b = self.data_registers[reg_b]

        if opcode == Opcode.SEEK:
            target = addr_a + param
            distance = abs(target - self.transport._cursor)
            apply_operator(opcode, [], [], distance)
            self.transport.seek(target)
            self.data_registers[dest] = target
            return

        if opcode == Opcode.HALT:
            apply_operator(opcode, [], [], param)
            self.halted = True
            return

        if opcode == Opcode.READ:
            frames = self.transport[addr_a : addr_a + param]
            out = apply_operator(opcode, frames, [], param)
            self.transport[addr_dest : addr_dest + len(out)] = out
            return

        if opcode == Opcode.WRITE:
            frames = self.transport[addr_b : addr_b + param]
            out = apply_operator(opcode, [], frames, param)
            self.transport[addr_dest : addr_dest + len(out)] = out
            return

        if opcode in {Opcode.LOAD, Opcode.STORE}:
            if self.tape_map is None:  # pragma: no cover - boot invariant
                raise RuntimeError("tape machine is not booted")
            spill_addr = (
                self.tape_map.data_start
                + (REGISTERS + param) * self.bit_width
            )
            if spill_addr + self.bit_width > len(self.transport):
                raise IndexError("spill slot exceeds the physical tape")
            time_axis = np.linspace(
                0,
                BIT_FRAME_MS / 1000.0,
                FRAME_SAMPLES,
                endpoint=False,
            )
            write_bias = BIAS_AMP * np.sin(
                2 * np.pi * WRITE_BIAS * time_axis
            )

            def rebias(frames):
                # Frames read from tape already contain one write-bias carrier.
                # Remove it before the destination head applies a fresh one.
                return [
                    (frame - write_bias).astype("f4") for frame in frames
                ]

            if opcode == Opcode.LOAD:
                frames = self.transport[
                    spill_addr : spill_addr + self.bit_width
                ]
                self.transport[
                    addr_dest : addr_dest + self.bit_width
                ] = rebias(frames)
            else:
                frames = self.transport[addr_a : addr_a + self.bit_width]
                self.transport[
                    spill_addr : spill_addr + self.bit_width
                ] = rebias(frames)
            return

        wave_a = self.transport[addr_a : addr_a + self.bit_width]
        wave_b = self.transport[addr_b : addr_b + self.bit_width]

        if opcode == Opcode.MU:
            sel_idx = param & 0xF
            sel_addr = self.data_registers.get(sel_idx, addr_b)
            sel = self.transport[sel_addr : sel_addr + self.bit_width]
            out = mu(wave_a, wave_b, sel)
        elif opcode == Opcode.NAND:
            # Main-tape registers are scalar bit streams on the transport
            # lane.  The general NAND operator can process all spectral lanes
            # in parallel, but doing so here invents 31 unrelated bits and
            # eventually normalises away the actual register signal.
            lane_mask = 1 << self.transport.lane
            out = [
                nand_wave(left, right, lane_mask=lane_mask)
                for left, right in zip(wave_a, wave_b)
            ]
        else:
            out = apply_operator(opcode, wave_a, wave_b, param)

        self.transport[addr_dest : addr_dest + len(out)] = out

    def run(self, instruction_count: int) -> None:
        """Boot the machine then execute ``instruction_count`` instructions."""

        self._boot(instruction_count)
        print("TapeMachine: Starting execution loop...")
        executed = 0
        while executed < instruction_count and not self.halted:
            print(
                f"  Executing instruction {executed + 1}/{instruction_count}",
                end="\r",
            )
            opcode, dest, reg_a, reg_b, param = self._fetch_decode()
            self._execute(opcode, dest, reg_a, reg_b, param)
            if self.visualizer:
                self.visualizer.draw()
            executed += 1
        print()
        if self.halted:
            print("Execution halted.")
        else:
            print("Execution finished.")

    # ------------------------------------------------------------------
    def queue_register_ops(self, ops: Sequence[int]) -> None:
        """Forward operator codes to the underlying transport when in register mode."""

        if not self.register_mode:
            raise RuntimeError("machine not initialised for register mode")
        self.transport.queue_operators(ops)

