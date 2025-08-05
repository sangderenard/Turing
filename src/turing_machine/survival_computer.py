# survival_computer.py
"""
Simulates a physical tape-based "survival computer" using a compile-and-run model.

1. Records a mathematical calculation into a ProvenanceGraph.
2. Compiles the graph into a binary instruction stream and tape layout (Tape IR).
3. Primes a virtual cassette tape with this compiled program (BIOS, code, data).
4. A simulated TapeMachine boots up, reads the instructions from the tape, and
   executes them using physically-modeled analog operators.
5. The CassetteTapeBackend generates a high-fidelity audio simulation of this
   entire physical process.
"""
from __future__ import annotations

import time

# Core components
from ..compiler.bitops_translator import BitOpsTranslator
from ..hardware.cassette_tape import CassetteTapeBackend
from ..compiler.tape_compiler import TapeCompiler
from .tape_machine import TapeMachine


def prime_tape_with_program(cassette: CassetteTapeBackend, tape_map, instructions_binary):
    """Write BIOS and instruction frames directly onto the cassette."""

    print("Priming tape with compiled program...")
    bios_frames = tape_map.encode_bios()
    for i, frame in enumerate(bios_frames):
        for lane, bit in enumerate(frame):
            cassette.write_bit(0, lane, tape_map.bios_start + i, bit)

    for i, frame in enumerate(instructions_binary):
        for lane, bit in enumerate(frame):
            cassette.write_bit(0, lane, tape_map.instr_start + i, bit)

    print("Tape priming complete.")


def main():
    BIT_WIDTH = 32
    TAPE_LEN = 8192 # Increased tape length for code and data

    # --- 1. RECORD & COMPILE ---
    print("Step 1: Recording computation into a ProvenanceGraph...")
    translator = BitOpsTranslator(bit_width=BIT_WIDTH)
    a, b = 5, 3
    print(f"Tracing the operation: {a} * {b}")
    translator.bit_mul(a, b)
    graph = translator.graph
    print(f"Trace complete. Graph has {len(graph.nodes)} primitive nodes.")

    print("\nStep 2: Compiling graph into Tape IR...")
    compiler = TapeCompiler(graph, BIT_WIDTH)
    tape_map, instructions = compiler.compile()
    
    # Binarize the instructions into frames of bits
    instructions_binary = TapeCompiler.binarize_instructions(instructions)

    # --- 2. INITIALIZE HARDWARE & "FLASH" THE TAPE ---
    print("\nStep 3: Initializing tape machine...")
    # NOTE: The invalid parameters are now removed.
    cassette = CassetteTapeBackend(tape_length=TAPE_LEN)
    
    # This is a conceptual step. We are "priming" the tape with our program.
    # A full implementation would need a robust method on the cassette to handle this.
    prime_tape_with_program(cassette, tape_map, instructions_binary)

    # --- 3. EXECUTE FROM TAPE ---
    print("\nStep 4: Starting survival computer simulation.")
    print("This will be noisy! The machine will now boot and run the program from the tape.")
    
    machine = TapeMachine(cassette, BIT_WIDTH)
    
    start_time = time.time()
    # Tell the machine to run the number of instructions we compiled
    machine.run(instruction_count=len(instructions))
    end_time = time.time()

    time.sleep(1.0) # Let final sounds play out

    print("\n--- Simulation Complete ---")
    print(f"Simulated execution in {end_time - start_time:.2f} seconds.")
    print(f"A total of {cassette._audio_cursor} audio frames were generated.")

    cassette.close()
    print("Simulator shut down.")

if __name__ == "__main__":
    main()
