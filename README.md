# Novel Idea:

Title idea:
Data-Driven Riemannian Metrics for Convolutional Layers: Learning Geometry from Latent Manifolds

Abstract (sketch):
We introduce a new class of convolutional neural networks where the Riemannian metric tensor of the feature space is learned directly from the data manifold. Unlike prior manifold CNNs that fix geometry a priori, our approach dynamically adapts the Laplace–Beltrami operator to the latent structure. We demonstrate stable convergence to below 1e-6 training error within 235 epochs on an 8-class dataset and strong generalization on 100-class classification. This shows that data-driven metric learning can serve as both a geometric prior and a numerical preconditioner, unifying geometric deep learning with learned differential geometry.


# Turing

This project simulates a cassette-driven "survival computer" using a compile and run flow.

## Table of contents

- [Compiling a program](#compiling-a-program)
- [Priming a tape and executing with TapeMachine](#priming-a-tape-and-executing-with-tapemachine)
- [Reel animation demo](#reel-animation-demo)
- [Live cassette demo](#live-cassette-demo)
- [Modules & capabilities](#modules--capabilities)
- [Tests](#tests)
- [Full module index](MODULES.md)

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
python -m src.visualizations.reel_demo --mode cassette  # Cassette layout
python -m src.visualizations.reel_demo --mode reel      # Reel-to-reel layout
```

Press Space to toggle play and W to toggle the red write indicator.

## Live cassette demo

To watch the full compile/write/run cycle with synchronized audio and reel
animation, run:

```bash
python -m src.visualizations.live_cassette_demo
```

The window shows tape motion in real time as the simulator writes the program
to a blank cassette and then executes it.

## Modules & capabilities

Below is a current inventory of the repository’s Python modules, grouped by package, to reflect the full breadth of capability.

### Top-level

- `backend.py` — entry point that runs tests and a NAND demo tape
- `generated_ssa_helpers.py` — generated helper table for SSA
- `count_loc.py` — line-of-code counter utility

### src/common

- `types.py` — shared types and aliases

### src/compiler

- `__init__.py`
- `bitops.py` — primitive bitwise ops on graphs
- `bitops_translator.py` — translates bit-ops into provenance/SSA
- `process_graph_helper.py` — utility helpers for process graphs
- `ssa_builder.py` — SSA construction utilities
- `tape_compiler.py` — compiles graphs into tape instruction streams

### src/hardware

- `__init__.py`
- `analog_spec.py` — analog operator specifications (see ANALOG_SPEC.md)
- `analog_helpers.py` — helpers for analog simulation
- `cassette_adapter.py` — legacy façade for cassette operations
- `cassette_tape.py` — high‑fidelity cassette backend
- `constants.py` — hardware constants
- `lane_tuner.py` — frequency lane tuning and helpers
- `nand_wave.py` — NAND tone synthesis/decoding

### src/common/tensors

- `abstraction.py` — shared `AbstractTensor` interface and backend registry
- `pure_backend.py` — reference backend built on nested Python lists
- `numpy_backend.py`, `torch_backend.py`, `jax_backend.py` — NumPy, PyTorch, and JAX implementations
- `accelerator_backends/` — experimental C, Rust, and GLSL kernels
- `models/` — higher level tensor utilities

### src/turing_machine

- `__init__.py`
- `docstrings_math.py` — docstring-backed math helpers
- `loop_structure.py` — loop constructs for the machine
- `survival_computer.py` — high-level compile/prime/run driver
- `tape_head.py`, `tape_transport.py` — transport mechanics
- `tape_machine.py` — execution loop
- `tape_map.py` — mapping between instructions and frames
- `tape_visualizer.py` — visualizations helpers
- `turing.py`, `turing_ssa.py`, `turing_provenance.py` — core logic and provenance

### src/bitbitbuffer

- `__init__.py`, `bitbitbuffer.py`
- helpers/: `__init__.py`, `utils.py`, `testbench.py`, `rawspan.py`, `pidbuffer.py`, `data_access.py`, `cell_proposal.py`, `bitbitslice.py`, `bitbititem.py`, `bitbitindex.py`, `bitbitindexer.py`, `bitstream_search.py`

### src/cells

- `__init__.py`, `cell_consts.py`, `cell_walls.py`, `simulator.py`
- `bath/`
- `cellsim/`
  - `__init__.py`, `constants.py`
  - api/: `saline.py`
  - bath/: `reservoir.py`
  - chemistry/: `crn.py`, `electrochem.py`
  - core/: `units.py`, `numerics.py`, `geometry.py`, `checks.py`
  - data/: `state.py`, `species.py`, `proposals.py`
  - engine/: `saline.py`
  - membranes/: `membrane.py`, `gates.py`
  - mechanics/: `provider.py`, `softbody0d.py`, `tension.py`
  - transport/: `pumps.py`, `kedem_katchalsky.py`, `ghk.py`
  - placement/: `bitbuffer.py`, `sync.py`
  - organelles/: `inner_loop.py`
  - viz/: `ascii.py`
  - examples/: `demo_sim.py`
- `simulator_methods/`: `cell_mask.py`, `data_io.py`, `injection.py`, `lcm.py`, `logutil.py`, `minimize.py`, `quanta_map_and_dump_cells.py`, `visualization.py`
- `softbody/`
  - `__init__.py`
  - engine/: `__init__.py`, `xpbd_core.py`, `params.py`, `mesh.py`, `hierarchy.py`, `fields.py`, `coupling.py`, `constraints.py`, `collisions.py`
  - geometry/: `primitives.py`, `geodesic.py`
  - bridge/: `__init__.py`, `state_io.py`
  - resources/: `field_library.py`
  - demo/: `__init__.py`, `run_ascii_demo.py`, `numpy_sim_coordinator.py`
    - Geometry visualization is handled by the shared `src/opengl_render` package.

### src/transmogrifier

- `__init__.py`
- `dec.py` — decoding utilities
- `ilpscheduler.py` — ILP-based scheduling experiments
- `operator_defs.py` — operator definitions
- `orbital.py`, `orbital_transfer.py` — orbital optimization models
- `physics.py` — physics helpers
- `ssa.py`, `ssa_registry.py` — SSA and registry
- `solver_types.py` — solver/data types

#### src/transmogrifier/graph

- `__init__.py`
- `graph_express2.py`, `graph_express2printing.py`, `graph_express2_tests.py`
- `graph_express_chalkboard_problem.py`
- `memory_graph/`
  - `__init__.py`, `memory_graph.py`
  - `helpers/`: `bt_graph_header.py`, `deque3_d.py`, `edge_entry.py`, `graph_search.py`, `mask_consolidation.py`, `meta_graph_edge.py`, `networkx_emulation.py`, `node_entry.py`, `node_region.py`, `set_micrograin_entry.py`, `struct_view.py`, `bit_tensor_memory.py`, `bit_tensor_memory_dag_helper.py`, `bit_tensor_memory_units.py`

### src/visualizations

- `live_cassette_demo.py`
- `reel_demo.py`, `reel_demo_shell.py`, `reel_math.py`

### tests

- End-to-end and unit tests under `tests/` covering: analog fidelity, lane tuning, NAND wave, compiler and SSA, tape machine, cellsim (API/engine/physics/transport), bitbitbuffer utilities, visualization scaling, soft-body contacts, memory graph dynamics, and more.

For a clickable, hyperlinked index, see MODULES.md.

## Tests

- Run the whole suite with `pytest` from repo root. The suite includes:
  - Turing machine core: `test_turing_reference.py`, `test_turing_ssa.py`, loops, new opcodes
  - Hardware/analog: `test_nand_wave.py`, `test_lane_tuner.py`, header layout
  - Visuals: `test_visualization_no_pygame.py`, `test_visualization_scaling.py`
  - bitbitbuffer: slicing/indexing/search and PID buffer
  - Cellsim: API/engine, pressure/mass/charge, membranes/collisions, organelle exclusion
  - Transmogrifier graphs and memory graph dynamics

This is a large project and a full `pytest` run can take a while. Each run records
its total runtime in `pytest_run_times.log` (kept to the last 50 entries) so you can
track how long recent test sessions have taken.

### Focused subsets

- dt system fast: run adaptive-dt controller and graph smoke tests quickly
  using markers: `pytest -q -m "dt and fast"`
- dt graph only: `pytest -q -m dt_graph`
- explicit files: `pytest -q tests/test_dt_controller.py tests/test_dt_graph.py`
<!-- No-op change to trigger PR -->
