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

from ..hardware.analog_spec import (
    LANES,
    Opcode,
    apply_operator,
    mu,
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

        opcode_val = (word >> 12) & 0xF
        reg_a = (word >> 10) & 0x3
        reg_b = (word >> 8) & 0x3
        dest = (word >> 6) & 0x3
        param = word & 0x3F
        return Opcode(opcode_val), dest, reg_a, reg_b, param

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

        wave_a = self.transport[addr_a : addr_a + self.bit_width]
        wave_b = self.transport[addr_b : addr_b + self.bit_width]

        if opcode == Opcode.MU:
            sel_idx = param & 0xF
            sel_addr = self.data_registers.get(sel_idx, addr_b)
            sel = self.transport[sel_addr : sel_addr + self.bit_width]
            out = mu(wave_a, wave_b, sel)
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

