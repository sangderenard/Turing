# tape_machine.py
"""
The core execution engine for the survival computer.

The TapeMachine class simulates a physical processor that fetches, decodes,
and executes a binary instruction stream directly from a cassette tape. It operates
on analog waveforms, using the functions defined in analog_spec to perform
computations.
"""
from __future__ import annotations

import time
from typing import Dict, List

from analog_spec import Opcode, nand_wave, sigma_L, sigma_R, concat, slice_frames, mu, length, zeros, generate_bit_wave
from cassette_tape import CassetteTapeBackend
from tape_map import TapeMap
from tape_transport import TapeTransport

class TapeMachine:
    """
    A simulated machine that reads instructions from a tape and executes them
    using analog wave-based operations.
    """
    def __init__(self, cassette: CassetteTapeBackend, bit_width: int):
        self.cassette = cassette
        self.bit_width = bit_width
        self.transport = TapeTransport(cassette)
        self.tape_map: TapeMap | None = None
        self.instruction_pointer = 0
        self.data_registers: Dict[int, int] = {} # Maps register index to tape address

    def _boot(self):
        """Reads the BIOS from the tape to configure the machine."""
        print("TapeMachine: Booting...")
        # Decode BIOS from the start of the tape
        bios_frames = self.transport[0:TapeMap.get_bios_frame_count()]
        self.tape_map = TapeMap.decode_bios(bios_frames)
        
        # Set the instruction pointer to the start of the code section
        self.instruction_pointer = self.tape_map.instr_start
        print(f"TapeMachine: Boot successful. Instruction pointer set to frame {self.instruction_pointer}.")

        # Map logical registers (0-15) to physical tape addresses in the data section
        for i in range(16):
            self.data_registers[i] = self.tape_map.data_start + (i * self.bit_width)

    def _fetch_decode(self) -> tuple[Opcode, int, int, int]:
        """Fetches and decodes one 16-bit instruction from the tape."""
        instr_frame_bits = self.transport[self.instruction_pointer]
        self.instruction_pointer += 1

        # Convert bit list to integer
        word = 0
        for bit in instr_frame_bits:
            word = (word << 1) | bit
        
        # Decode fields from the 16-bit word
        opcode_val = (word >> 12) & 0xF
        dest = (word >> 8) & 0xF
        reg_a = (word >> 4) & 0xF
        reg_b_or_param = word & 0xF
        
        return Opcode(opcode_val), dest, reg_a, reg_b_or_param

    def _execute(self, opcode: Opcode, dest: int, reg_a: int, reg_b_or_param: int):
        """Executes a single instruction using analog operators."""
        # Get physical tape addresses for the logical registers
        addr_dest = self.data_registers[dest]
        addr_a = self.data_registers[reg_a]
        addr_b = self.data_registers[reg_b_or_param] # Assumes it's a register for now

        # --- This is the new core logic, dispatching to analog_spec functions ---
        
        if opcode == Opcode.NAND:
            # Read the full wave forms for the operands
            wave_a = self.transport[addr_a : addr_a + self.bit_width]
            wave_b = self.transport[addr_b : addr_b + self.bit_width]
            
            # Perform the NAND operation on each corresponding frame
            result_waves = [nand_wave(wa, wb) for wa, wb in zip(wave_a, wave_b)]

            # Write the resulting waveforms back to the destination
            self.transport[addr_dest : addr_dest + self.bit_width] = result_waves

        elif opcode == Opcode.ZEROS:
            # Generate zero-energy frames
            zero_waves = zeros(self.bit_width)
            self.transport[addr_dest : addr_dest + self.bit_width] = zero_waves

        # ... Other opcodes would be implemented here using the same pattern ...
        # read waves -> call analog_spec function -> write result waves
        
        else:
            # Placeholder for unimplemented opcodes
            print(f"Warning: Opcode {opcode.name} not yet implemented in TapeMachine executor.")
            time.sleep(0.01) # Simulate taking time

    def run(self, instruction_count: int):
        """Runs the boot sequence and then the fetch-decode-execute cycle."""
        self._boot()
        
        print("TapeMachine: Starting execution loop...")
        for i in range(instruction_count):
            print(f"  Executing instruction {i+1}/{instruction_count}", end='\r')
            opcode, dest, reg_a, reg_b = self._fetch_decode()
            self._execute(opcode, dest, reg_a, reg_b)
        
        print("\nExecution finished.")

