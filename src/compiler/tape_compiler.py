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
import numpy as np

from ..hardware.analog_spec import BiosHeader, InstructionWord, Opcode
from ..hardware.constants import LANES, REGISTERS
from ..turing_machine.tape_map import TapeMap
from ..turing_machine.turing_provenance import ProvenanceGraph, ProvNode
from ..transmogrifier.ssa import Instr

# Type aliases for clarity
TapeAddress = int
ObjectID = int
MemoryMap = Dict[ObjectID, int]  # Maps object IDs to register indices
InstructionStream = List[InstructionWord]

class TapeCompiler:
    """
    Translates a ProvenanceGraph into a binary instruction stream and a
    corresponding memory layout for the tape.
    """
    def __init__(self, graph: ProvenanceGraph, bit_width: int):
        self.graph = graph
        self.bit_width = bit_width
        # ``memory_map`` tracks values resident in the limited register file.
        # ``data_map`` records values that live in the tape's data section when
        # register pressure forces us to spill.
        self.memory_map: MemoryMap = {}
        self.data_map: Dict[ObjectID, TapeAddress] = {}
        self._next_reg_idx = 0

    def _allocate_register(self, obj_id: ObjectID) -> int:
        """Assign a register to ``obj_id`` or spill it to tape if none remain."""
        if obj_id not in self.memory_map:
            if self._next_reg_idx >= REGISTERS:
                # Register file is exhausted – cache the value in data space.
                addr = self.data_map.setdefault(obj_id, len(self.data_map))
                # Returning zero reserves R0 as a staging area for later loads.
                return 0
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

    def compile_ssa(
        self,
        ssa_instrs: List[Instr],
        process_graph: "ProcessGraph" | None = None,
        schedule_mode: str = "alap",
    ) -> Tuple[TapeMap, InstructionStream, List[List[np.ndarray]]]:
        """Compile a sequence of SSA instructions into tape format.

        If a :class:`ProcessGraph` is supplied it is first scheduled using the
        requested mode (``"alap"`` by default).  The resulting concurrency map
        guides register allocation so temporaries only consume registers while
        live. φ-instructions reuse the register of their first operand and do
        not emit machine words. All other instructions map 1:1 to the minimal
        Opcode set defined in :mod:`analog_spec`.
        """

        self.memory_map = {}
        self.data_map = {}
        self._next_reg_idx = 0

        if process_graph is not None:
            import networkx as nx

            # ``compute_levels`` performs scheduling *and* interference analysis.
            # Its side effects populate ``proc_interference_graph`` and memory
            # graphs which we then colour to assign registers.  Calling separate
            # interference helpers would double work and risk diverging from the
            # ProcessGraph's own view of concurrency.
            process_graph.compute_levels(
                method=schedule_mode, order="dependency", interference_mode="asap-maxslack"
            )

            ig = getattr(process_graph, "proc_interference_graph", None)
            if ig is None:
                ig = nx.Graph()
                ig.add_nodes_from(process_graph.G.nodes)

            colouring = nx.algorithms.coloring.greedy_color(ig, strategy="largest_first")
            mem_nodes = set(getattr(getattr(process_graph, "mG", None), "nodes", []))
            for nid, colour in colouring.items():
                if nid in mem_nodes or colour >= REGISTERS:
                    self.data_map.setdefault(nid, len(self.data_map))
                else:
                    self.memory_map[nid] = colour
            if self.memory_map:
                self._next_reg_idx = min(max(self.memory_map.values()) + 1, REGISTERS)

        instructions: InstructionStream = []

        def _ensure(val_id: int) -> int:
            if val_id < 0:
                return 0
            if val_id in self.data_map:
                return 0  # will require explicit LOAD/STORE in a later pass
            if val_id not in self.memory_map:
                return self._allocate_register(val_id)
            return self.memory_map[val_id]

        for inst in ssa_instrs:
            if inst.op == "phi":
                # Reuse the first operand's register for the φ-result
                reg = _ensure(inst.args[0].id)
                self.memory_map[inst.res.id] = reg
                continue

            opcode = self._op_to_opcode(inst.op)
            dest_reg = _ensure(inst.res.id)
            reg_a = _ensure(inst.args[0].id) if inst.args else 0
            reg_b = _ensure(inst.args[1].id) if len(inst.args) > 1 else 0

            instructions.append(
                InstructionWord(
                    opcode=opcode,
                    dest=dest_reg,
                    reg_a=reg_a,
                    reg_b=reg_b,
                    param=reg_b,
                )
            )

        # Build BIOS and PCM frames identical to ``compile``
        bios = BiosHeader(
            calib_fast_ms=10.0,
            calib_read_ms=50.0,
            drift_ms=1.0,
            inputs=[],
            outputs=[],
            instr_start_addr=0,
        )

        tape_map = TapeMap(bios, instruction_frames=len(instructions))
        tape_map.bios.instr_start_addr = tape_map.instr_start

        from ..hardware.analog_spec import generate_bit_wave

        bios_bit_frames = tape_map.encode_bios()
        bios_pcm_frames: List[List[np.ndarray]] = []
        for frame in bios_bit_frames:
            pcm_lanes = [generate_bit_wave(bit, lane) for lane, bit in enumerate(frame)]
            bios_pcm_frames.append(pcm_lanes)

        instr_bit_frames = self.binarize_instructions(instructions)
        instr_pcm_frames: List[List[np.ndarray]] = []
        for frame in instr_bit_frames:
            pcm_lanes = [generate_bit_wave(bit, lane) for lane, bit in enumerate(frame)]
            instr_pcm_frames.append(pcm_lanes)

        tape_pcm = bios_pcm_frames + instr_pcm_frames
        return tape_map, instructions, tape_pcm

    def compile(self) -> Tuple[TapeMap, InstructionStream, List[List[np.ndarray]]]:
        """Compile ``self.graph`` by funnelling it through SSA first.

        Historically ``TapeCompiler`` duplicated logic for ProvenanceGraph and
        ProcessGraph inputs, emitting machine words directly from whichever
        structure it was handed.  That made optimisation difficult and
        discouraged experimentation – any new transformation had to be re
        implemented twice.  The modern pipeline always converts to SSA and then
        reuses :meth:`compile_ssa` for the final lowering step.  This keeps the
        book‑keeping in one place and gives the SSA stage maximal freedom to
        optimise and schedule without worrying about tape details.
        """

        from .process_graph_helper import provenance_to_process_graph, reduce_cycles_to_mu
        from .ssa_builder import process_graph_to_ssa_instrs

        # Normalise the provenance graph by rewriting feedback edges into
        # explicit ``mu`` nodes so subsequent scheduling operates on an acyclic
        # structure.
        reduce_cycles_to_mu(self.graph)
        proc = provenance_to_process_graph(self.graph)
        instrs = process_graph_to_ssa_instrs(proc)
        return self.compile_ssa(instrs, process_graph=proc)

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
