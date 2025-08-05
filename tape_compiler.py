# tape_compiler.py
"""
Compiles a ProvenanceGraph into a tape-based Intermediate Representation (IR).

This component translates the abstract, data-flow graph from turing_provenance
into a linear sequence of binary machine code instructions that can be executed
by the TapeMachine. It handles memory allocation for all variables and assembles
a complete tape image including a BIOS, the instruction stream, and a data section.
"""
from __future__ import annotations

import struct
from typing import Dict, List, Tuple

from analog_spec import BiosHeader, InstructionWord, Opcode, LANES
from tape_map import TapeMap
from turing_provenance import ProvenanceGraph, ProvNode

# Type aliases for clarity
TapeAddress = int
ObjectID = int
MemoryMap = Dict[ObjectID, int] # Maps graph object ID to a "register" index (0-15)
InstructionStream = List[InstructionWord]

class TapeCompiler:
    """
    Translates a ProvenanceGraph into a binary instruction stream and a
    corresponding memory layout for the tape.
    """
    def __init__(self, graph: ProvenanceGraph, bit_width: int):
        self.graph = graph
        self.bit_width = bit_width
        self.memory_map: MemoryMap = {}
        self._next_reg_idx = 0

    def _allocate_register(self, obj_id: ObjectID) -> int:
        """Assigns a unique register index (0-15) to a graph object ID."""
        if obj_id not in self.memory_map:
            if self._next_reg_idx >= 16:
                # In a real compiler, this would spill to memory. Here, we error out.
                raise MemoryError("Out of registers (max 16 supported)")
            self.memory_map[obj_id] = self._next_reg_idx
            self._next_reg_idx += 1
        return self.memory_map[obj_id]

    def _op_to_opcode(self, op_name: str) -> Opcode:
        """Maps the string operator name from the graph to an Opcode enum."""
        opcode_map = {
            "nand": Opcode.NAND,
            "sigma_L": Opcode.SIGL,
            "sigma_R": Opcode.SIGR,
            "concat": Opcode.CONCAT,
            "slice": Opcode.SLICE,
            "mu": Opcode.MU,
            "length": Opcode.LENGTH,
            "zeros": Opcode.ZEROS,
        }
        if op_name not in opcode_map:
            raise ValueError(f"Unknown graph operation: {op_name}")
        return opcode_map[op_name]

    def compile(self) -> Tuple[TapeMap, InstructionStream]:
        """
        Compiles the graph into a tape map and a stream of instructions.
        """
        print("Compiler Step 1: Allocating registers for all graph objects...")
        # First pass: Allocate a "register" for every output object in the graph
        for node in self.graph.nodes:
            self._allocate_register(node.out_obj_id)

        print("Compiler Step 2: Generating instruction stream from graph...")
        instructions: InstructionStream = []
        for node in self.graph.nodes:
            opcode = self._op_to_opcode(node.op)
            
            # Assign registers for inputs and outputs
            dest_reg = self._allocate_register(node.out_obj_id)
            
            # Handle operands based on operation type
            reg_a = 0
            reg_b_or_param = 0

            if node.args:
                reg_a = self.memory_map.get(node.args[0], 0)
            if len(node.args) > 1:
                # If the arg is an integer, it's a parameter (like for shifts)
                if isinstance(node.args[1], int):
                    reg_b_or_param = node.args[1]
                else: # Otherwise it's a register
                    reg_b_or_param = self.memory_map.get(node.args[1], 0)
            if len(node.args) > 2:
                # Special handling for mu, which has a third register argument
                # We can pack it into the param field for this simple ISA
                if opcode == Opcode.MU:
                    # This is a simplification; a real ISA would need more space
                    # Here we'll just use reg_b for the selector
                    reg_b_or_param = self.memory_map.get(node.args[2], 0)


            instr = InstructionWord(
                opcode=opcode,
                dest=dest_reg,
                reg_a=reg_a,
                reg_b=reg_b_or_param,
                param=reg_b_or_param # For clarity, param and reg_b are the same field
            )
            instructions.append(instr)

        # Create a default BIOS header
        bios = BiosHeader(
            calib_fast_ms=10.0,
            calib_read_ms=50.0,
            drift_ms=1.0,
            inputs=[],
            outputs=[],
            instr_start_addr=0 # This will be set by TapeMap
        )

        # The TapeMap will calculate the final layout
        tape_map = TapeMap(bios, instruction_frames=len(instructions))
        tape_map.bios.instr_start_addr = tape_map.instr_start
        
        print(f"Compilation successful. Generated {len(instructions)} instructions.")
        return tape_map, instructions

    @staticmethod
    def binarize_instructions(instructions: InstructionStream) -> List[List[int]]:
        """
        Converts the instruction stream into a list of 16-bit frames,
        where each frame is a list of 16 bits.
        """
        bit_frames = []
        for instr in instructions:
            # Pack into a 16-bit integer: OOOODDDDAAAABBBB
            word = (
                (instr.opcode.value & 0xF) << 12 |
                (instr.dest & 0xF) << 8 |
                (instr.reg_a & 0xF) << 4 |
                (instr.param & 0xF)
            )
            
            # Unpack into a list of 16 bits (MSB first)
            bits = [(word >> (15 - i)) & 1 for i in range(16)]
            bit_frames.append(bits)
        return bit_frames
