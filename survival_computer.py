# survival_computer.py
"""
Simulates a physical tape-based "survival computer".

1. Records a mathematical calculation into a ProvenanceGraph.
2. "Compiles" the graph by allocating space on a virtual tape for all variables.
3. Initializes the tape with any required constants.
4. Executes the graph by translating each logical primitive into a sequence of
   physical tape operations (read, write, move_head).
5. The CassetteTapeBackend generates a high-fidelity audio simulation of this
   entire physical process.
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, Any

# Core components
from bitops_translator import BitOpsTranslator
from cassette_tape import CassetteTapeBackend
from turing_provenance import ProvenanceGraph, ProvNode

# Type alias for clarity
TapeAddress = int
ObjectID = int
MemoryMap = Dict[ObjectID, TapeAddress]

class GraphExecutor:
    """
    Executes a ProvenanceGraph on a physical tape machine simulator.
    """
    def __init__(self, graph: ProvenanceGraph, cassette: CassetteTapeBackend, bit_width: int):
        self.graph = graph
        self.cassette = cassette
        self.bit_width = bit_width
        self.tape_len = cassette.tape_length
        self.memory_map: MemoryMap = {}
        self._tape_cursor = 0 # Next available tape address

    def _allocate_memory(self, obj_id: ObjectID) -> TapeAddress:
        """Allocates a block of memory on the tape for a given object ID."""
        if obj_id not in self.memory_map:
            if self._tape_cursor + self.bit_width > self.tape_len:
                raise MemoryError("Out of tape memory.")
            
            self.memory_map[obj_id] = self._tape_cursor
            self._tape_cursor += self.bit_width
        return self.memory_map[obj_id]

    def _execute_node(self, node: ProvNode):
        """Translates a single graph node into physical tape operations."""
        op = node.op
        
        # Get addresses for inputs and allocate space for the output
        input_addrs = [self.memory_map.get(arg_id) for arg_id in node.args]
        output_addr = self._allocate_memory(node.out_obj_id)

        # A small delay to make operations audibly distinct
        time.sleep(0.001)

        # --- Translate each primitive into physical R/W/Move operations ---
        if op == "zeros":
            for i in range(self.bit_width):
                self.cassette.write_bit(output_addr + i, 0)
        
        elif op == "nand":
            addr_x, addr_y = input_addrs
            for i in range(self.bit_width):
                bit_x = self.cassette.read_bit(addr_x + i)
                bit_y = self.cassette.read_bit(addr_y + i)
                result = 1 - (bit_x & bit_y)
                self.cassette.write_bit(output_addr + i, result)
        
        elif op == "mu":
            addr_x, addr_y, addr_sel = input_addrs
            for i in range(self.bit_width):
                bit_x = self.cassette.read_bit(addr_x + i)
                bit_y = self.cassette.read_bit(addr_y + i)
                sel = self.cassette.read_bit(addr_sel + i)
                result = bit_y if sel else bit_x
                self.cassette.write_bit(output_addr + i, result)

        elif op == "concat":
            # This is a simplified model: we assume concat inputs are smaller
            # and we just copy them sequentially. A full implementation would be more complex.
            # For this demo, we handle the common case of single-bit concat.
            addr_x, addr_y = input_addrs
            # Naive implementation: this would require dynamic length handling.
            # For now, we simulate it as a conceptual operation.
            self.cassette.execute_instruction() # Placeholder sound

        # Other primitives like slice, sigma_L/R would require similar translation
        # into sequences of read, write, and move operations. For brevity,
        # we generate a generic sound for them in this demonstration.
        else:
            self.cassette.execute_instruction()

    def run(self):
        """Compiles and runs the entire graph."""
        print("Compiling graph: Allocating memory on tape...")
        # First pass: Allocate memory for all node outputs
        for node in self.graph.nodes:
            self._allocate_memory(node.out_obj_id)

        print(f"Execution phase: Processing {len(self.graph.nodes)} nodes...")
        # Second pass: Execute the nodes in order
        for i, node in enumerate(self.graph.nodes):
            print(f"  Executing node {i+1}/{len(self.graph.nodes)}: {node.op}", end='\r')
            self._execute_node(node)
        print("\nExecution finished.")


def main():
    BIT_WIDTH = 4
    TAPE_LEN = 4096

    # --- 1. RECORD ---
    print("Step 1: Recording computation into a ProvenanceGraph...")
    translator = BitOpsTranslator(bit_width=BIT_WIDTH)
    a, b = 5, 3
    print(f"Tracing the operation: {a} * {b}")
    translator.bit_mul(a, b)
    graph = translator.graph
    print(f"Trace complete. Graph has {len(graph.nodes)} primitive nodes.")

    # --- 2. INITIALIZE HARDWARE & EXECUTOR ---
    print("\nStep 2: Initializing tape machine and executor...")
    cassette = CassetteTapeBackend(
        tape_length=TAPE_LEN,
        analogue_mode=True,
        frame_ms=5.0
    )
    executor = GraphExecutor(graph, cassette, BIT_WIDTH)

    # --- 3. COMPILE & EXECUTE ---
    print("\nStep 3: Starting survival computer simulation.")
    print("This will be noisy! Listen for motor, read, and write sounds.")
    
    start_time = time.time()
    executor.run()
    end_time = time.time()

    time.sleep(1.0) # Let final sounds play out

    print("\n--- Simulation Complete ---")
    print(f"Simulated execution in {end_time - start_time:.2f} seconds.")
    print(f"A total of {cassette._cursor} audio frames were generated.")

    # You can now export the full audio IR for analysis
    # lanes, motor, eq = cassette.export_ir()
    # print(f"Exported IR shape: {lanes.shape}")

    cassette.close()
    print("Simulator shut down.")

if __name__ == "__main__":
    main()