# Turing

This project simulates a cassette-driven "survival computer" using a compile and run flow.

## Compiling a program

The quickest way to build and execute a demo program is to run the driver script:

```bash
$ python -m src.turing_machine.survival_computer
Step 1: Recording computation into a ProvenanceGraph...
Tracing the operation: 5 * 3
Trace complete. Graph has 7 primitive nodes.

Step 2: Compiling graph into Tape IR...

Step 3: Initializing tape machine...
Priming tape with compiled program...
Tape priming complete.

Step 4: Starting survival computer simulation.
This will be noisy! The machine will now boot and run the program from the tape.
TapeMachine: Booting...
TapeMachine: Boot successful. Instruction pointer set to frame 42.
TapeMachine: Starting execution loop...
  Executing instruction 1/6
  Executing instruction 2/6
  Executing instruction 3/6
  Executing instruction 4/6
  Executing instruction 5/6
  Executing instruction 6/6

Execution finished.

--- Simulation Complete ---
Simulated execution in 0.42 seconds.
A total of 999 audio frames were generated.
Simulator shut down.
```

## Priming a tape and executing with `TapeMachine`

`survival_computer.py` uses a helper to prime the cassette with the compiled program. You can perform the same steps manually:

```bash
$ python - <<'PY'
from src.turing_machine.survival_computer import prime_tape_with_program
from src.compiler.bitops_translator import BitOpsTranslator
from src.compiler.tape_compiler import TapeCompiler
from src.hardware.cassette_tape import CassetteTapeBackend
from src.turing_machine.tape_machine import TapeMachine

BIT_WIDTH = 32
TAPE_LEN = 8192

translator = BitOpsTranslator(bit_width=BIT_WIDTH)
translator.bit_mul(5, 3)
compiler = TapeCompiler(translator.graph, BIT_WIDTH)
tape_map, instructions = compiler.compile()
frames = TapeCompiler.binarize_instructions(instructions)

cassette = CassetteTapeBackend(tape_length=TAPE_LEN)
prime_tape_with_program(cassette, tape_map, frames)

machine = TapeMachine(cassette, BIT_WIDTH)
machine.run(instruction_count=len(instructions))

cassette.close()
PY
Priming tape with compiled program...
Tape priming complete.
TapeMachine: Booting...
TapeMachine: Boot successful. Instruction pointer set to frame 42.
TapeMachine: Starting execution loop...
  Executing instruction 1/6
  Executing instruction 2/6
  Executing instruction 3/6
  Executing instruction 4/6
  Executing instruction 5/6
  Executing instruction 6/6

Execution finished.
```

These commands prime a virtual tape, boot the `TapeMachine`, and run the program end-to-end, emitting PCM audio as it goes.

## Reel animation demo

For a visual illustration of tape motion, a small Pygame program renders
cassette or reel-to-reel spools that scale with tape usage and draw the
tape path through a read/write head.

```bash
python -m src.reel_demo --mode cassette  # Cassette layout
python -m src.reel_demo --mode reel      # Reel-to-reel layout
```

Press <kbd>Space</kbd> to toggle play and <kbd>W</kbd> to toggle the red write indicator.
