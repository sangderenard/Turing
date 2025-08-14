# Module Index

A hyperlinked index of modules and capabilities in this repository.

## Top-level

- [backend.py](backend.py) — test runner and NAND demo tape
- [generated_ssa_helpers.py](generated_ssa_helpers.py) — generated SSA helpers
- [count_loc.py](count_loc.py) — LOC utility

## src/common

- [src/common/types.py](src/common/types.py)

## src/compiler

- [src/compiler/__init__.py](src/compiler/__init__.py)
- [src/compiler/bitops.py](src/compiler/bitops.py)
- [src/compiler/bitops_translator.py](src/compiler/bitops_translator.py)
- [src/compiler/process_graph_helper.py](src/compiler/process_graph_helper.py)
- [src/compiler/ssa_builder.py](src/compiler/ssa_builder.py)
- [src/compiler/tape_compiler.py](src/compiler/tape_compiler.py)

## src/hardware

- [src/hardware/__init__.py](src/hardware/__init__.py)
- [src/hardware/analog_spec.py](src/hardware/analog_spec.py)
- [src/hardware/analog_helpers.py](src/hardware/analog_helpers.py)
- [src/hardware/cassette_adapter.py](src/hardware/cassette_adapter.py)
- [src/hardware/cassette_tape.py](src/hardware/cassette_tape.py)
- [src/hardware/constants.py](src/hardware/constants.py)
- [src/hardware/lane_tuner.py](src/hardware/lane_tuner.py)
- [src/hardware/nand_wave.py](src/hardware/nand_wave.py)

## src/turing_machine

- [src/turing_machine/__init__.py](src/turing_machine/__init__.py)
- [src/turing_machine/docstrings_math.py](src/turing_machine/docstrings_math.py)
- [src/turing_machine/loop_structure.py](src/turing_machine/loop_structure.py)
- [src/turing_machine/survival_computer.py](src/turing_machine/survival_computer.py)
- [src/turing_machine/tape_head.py](src/turing_machine/tape_head.py)
- [src/turing_machine/tape_transport.py](src/turing_machine/tape_transport.py)
- [src/turing_machine/tape_machine.py](src/turing_machine/tape_machine.py)
- [src/turing_machine/tape_map.py](src/turing_machine/tape_map.py)
- [src/turing_machine/tape_visualizer.py](src/turing_machine/tape_visualizer.py)
- [src/turing_machine/turing.py](src/turing_machine/turing.py)
- [src/turing_machine/turing_ssa.py](src/turing_machine/turing_ssa.py)
- [src/turing_machine/turing_provenance.py](src/turing_machine/turing_provenance.py)

## src/transmogrifier (core)

- [src/transmogrifier/__init__.py](src/transmogrifier/__init__.py)
- [src/transmogrifier/dec.py](src/transmogrifier/dec.py)
- [src/transmogrifier/ilpscheduler.py](src/transmogrifier/ilpscheduler.py)
- [src/transmogrifier/operator_defs.py](src/transmogrifier/operator_defs.py)
- [src/transmogrifier/orbital.py](src/transmogrifier/orbital.py)
- [src/transmogrifier/orbital_transfer.py](src/transmogrifier/orbital_transfer.py)
- [src/transmogrifier/physics.py](src/transmogrifier/physics.py)
- [src/transmogrifier/ssa.py](src/transmogrifier/ssa.py)
- [src/transmogrifier/ssa_registry.py](src/transmogrifier/ssa_registry.py)
- [src/transmogrifier/solver_types.py](src/transmogrifier/solver_types.py)

### src/transmogrifier/graph

- [src/transmogrifier/graph/__init__.py](src/transmogrifier/graph/__init__.py)
- [src/transmogrifier/graph/graph_express2.py](src/transmogrifier/graph/graph_express2.py)
- [src/transmogrifier/graph/graph_express2printing.py](src/transmogrifier/graph/graph_express2printing.py)
- [src/transmogrifier/graph/graph_express2_tests.py](src/transmogrifier/graph/graph_express2_tests.py)
- [src/transmogrifier/graph/graph_express_chalkboard_problem.py](src/transmogrifier/graph/graph_express_chalkboard_problem.py)
- [src/transmogrifier/graph/memory_graph/__init__.py](src/transmogrifier/graph/memory_graph/__init__.py)
- [src/transmogrifier/graph/memory_graph/memory_graph.py](src/transmogrifier/graph/memory_graph/memory_graph.py)
- helpers
  - [src/transmogrifier/graph/memory_graph/helpers/__init__.py](src/transmogrifier/graph/memory_graph/helpers/__init__.py)
  - [src/transmogrifier/graph/memory_graph/helpers/bt_graph_header.py](src/transmogrifier/graph/memory_graph/helpers/bt_graph_header.py)
  - [src/transmogrifier/graph/memory_graph/helpers/deque3_d.py](src/transmogrifier/graph/memory_graph/helpers/deque3_d.py)
  - [src/transmogrifier/graph/memory_graph/helpers/edge_entry.py](src/transmogrifier/graph/memory_graph/helpers/edge_entry.py)
  - [src/transmogrifier/graph/memory_graph/helpers/graph_search.py](src/transmogrifier/graph/memory_graph/helpers/graph_search.py)
  - [src/transmogrifier/graph/memory_graph/helpers/mask_consolidation.py](src/transmogrifier/graph/memory_graph/helpers/mask_consolidation.py)
  - [src/transmogrifier/graph/memory_graph/helpers/meta_graph_edge.py](src/transmogrifier/graph/memory_graph/helpers/meta_graph_edge.py)
  - [src/transmogrifier/graph/memory_graph/helpers/networkx_emulation.py](src/transmogrifier/graph/memory_graph/helpers/networkx_emulation.py)
  - [src/transmogrifier/graph/memory_graph/helpers/node_entry.py](src/transmogrifier/graph/memory_graph/helpers/node_entry.py)
  - [src/transmogrifier/graph/memory_graph/helpers/node_region.py](src/transmogrifier/graph/memory_graph/helpers/node_region.py)
  - [src/transmogrifier/graph/memory_graph/helpers/set_micrograin_entry.py](src/transmogrifier/graph/memory_graph/helpers/set_micrograin_entry.py)
  - [src/transmogrifier/graph/memory_graph/helpers/struct_view.py](src/transmogrifier/graph/memory_graph/helpers/struct_view.py)
  - [src/transmogrifier/graph/memory_graph/helpers/bit_tensor_memory.py](src/transmogrifier/graph/memory_graph/helpers/bit_tensor_memory.py)
  - [src/transmogrifier/graph/memory_graph/helpers/bit_tensor_memory_dag_helper.py](src/transmogrifier/graph/memory_graph/helpers/bit_tensor_memory_dag_helper.py)
  - [src/transmogrifier/graph/memory_graph/helpers/bit_tensor_memory_units.py](src/transmogrifier/graph/memory_graph/helpers/bit_tensor_memory_units.py)

### src/transmogrifier/bitbitbuffer

- [src/transmogrifier/bitbitbuffer/__init__.py](src/transmogrifier/bitbitbuffer/__init__.py)
- [src/transmogrifier/bitbitbuffer/bitbitbuffer.py](src/transmogrifier/bitbitbuffer/bitbitbuffer.py)
- helpers
  - [src/transmogrifier/bitbitbuffer/helpers/__init__.py](src/transmogrifier/bitbitbuffer/helpers/__init__.py)
  - [src/transmogrifier/bitbitbuffer/helpers/utils.py](src/transmogrifier/bitbitbuffer/helpers/utils.py)
  - [src/transmogrifier/bitbitbuffer/helpers/testbench.py](src/transmogrifier/bitbitbuffer/helpers/testbench.py)
  - [src/transmogrifier/bitbitbuffer/helpers/rawspan.py](src/transmogrifier/bitbitbuffer/helpers/rawspan.py)
  - [src/transmogrifier/bitbitbuffer/helpers/pidbuffer.py](src/transmogrifier/bitbitbuffer/helpers/pidbuffer.py)
  - [src/transmogrifier/bitbitbuffer/helpers/data_access.py](src/transmogrifier/bitbitbuffer/helpers/data_access.py)
  - [src/transmogrifier/bitbitbuffer/helpers/cell_proposal.py](src/transmogrifier/bitbitbuffer/helpers/cell_proposal.py)
  - [src/transmogrifier/bitbitbuffer/helpers/bitbitslice.py](src/transmogrifier/bitbitbuffer/helpers/bitbitslice.py)
  - [src/transmogrifier/bitbitbuffer/helpers/bitbititem.py](src/transmogrifier/bitbitbuffer/helpers/bitbititem.py)
  - [src/transmogrifier/bitbitbuffer/helpers/bitbitindex.py](src/transmogrifier/bitbitbuffer/helpers/bitbitindex.py)
  - [src/transmogrifier/bitbitbuffer/helpers/bitbitindexer.py](src/transmogrifier/bitbitbuffer/helpers/bitbitindexer.py)
  - [src/transmogrifier/bitbitbuffer/helpers/bitstream_search.py](src/transmogrifier/bitbitbuffer/helpers/bitstream_search.py)

### src/transmogrifier/cells

- [src/transmogrifier/cells/__init__.py](src/transmogrifier/cells/__init__.py)
- [src/transmogrifier/cells/cell_consts.py](src/transmogrifier/cells/cell_consts.py)
- [src/transmogrifier/cells/cell_walls.py](src/transmogrifier/cells/cell_walls.py)
- [src/transmogrifier/cells/simulator.py](src/transmogrifier/cells/simulator.py)
- cellsim
  - [src/transmogrifier/cells/cellsim/__init__.py](src/transmogrifier/cells/cellsim/__init__.py)
  - [src/transmogrifier/cells/cellsim/constants.py](src/transmogrifier/cells/cellsim/constants.py)
  - api: [src/transmogrifier/cells/cellsim/api/saline.py](src/transmogrifier/cells/cellsim/api/saline.py)
  - bath: [src/transmogrifier/cells/cellsim/bath/reservoir.py](src/transmogrifier/cells/cellsim/bath/reservoir.py)
  - chemistry: [src/transmogrifier/cells/cellsim/chemistry/crn.py](src/transmogrifier/cells/cellsim/chemistry/crn.py), [electrochem.py](src/transmogrifier/cells/cellsim/chemistry/electrochem.py)
  - core: [units.py](src/transmogrifier/cells/cellsim/core/units.py), [numerics.py](src/transmogrifier/cells/cellsim/core/numerics.py), [geometry.py](src/transmogrifier/cells/cellsim/core/geometry.py), [checks.py](src/transmogrifier/cells/cellsim/core/checks.py)
  - data: [state.py](src/transmogrifier/cells/cellsim/data/state.py), [species.py](src/transmogrifier/cells/cellsim/data/species.py), [proposals.py](src/transmogrifier/cells/cellsim/data/proposals.py)
  - engine: [saline.py](src/transmogrifier/cells/cellsim/engine/saline.py)
  - membranes: [membrane.py](src/transmogrifier/cells/cellsim/membranes/membrane.py), [gates.py](src/transmogrifier/cells/cellsim/membranes/gates.py)
  - mechanics: [provider.py](src/transmogrifier/cells/cellsim/mechanics/provider.py), [softbody0d.py](src/transmogrifier/cells/cellsim/mechanics/softbody0d.py), [tension.py](src/transmogrifier/cells/cellsim/mechanics/tension.py)
  - transport: [pumps.py](src/transmogrifier/cells/cellsim/transport/pumps.py), [kedem_katchalsky.py](src/transmogrifier/cells/cellsim/transport/kedem_katchalsky.py), [ghk.py](src/transmogrifier/cells/cellsim/transport/ghk.py)
  - placement: [bitbuffer.py](src/transmogrifier/cells/cellsim/placement/bitbuffer.py), [sync.py](src/transmogrifier/cells/cellsim/placement/sync.py)
  - organelles: [inner_loop.py](src/transmogrifier/cells/cellsim/organelles/inner_loop.py)
  - viz: [ascii.py](src/transmogrifier/cells/cellsim/viz/ascii.py)
  - examples: [demo_sim.py](src/transmogrifier/cells/cellsim/examples/demo_sim.py)

### src/transmogrifier/softbody

- [src/transmogrifier/softbody/__init__.py](src/transmogrifier/softbody/__init__.py)
- engine: [__init__.py](src/transmogrifier/softbody/engine/__init__.py), [xpbd_core.py](src/transmogrifier/softbody/engine/xpbd_core.py), [params.py](src/transmogrifier/softbody/engine/params.py), [mesh.py](src/transmogrifier/softbody/engine/mesh.py), [hierarchy.py](src/transmogrifier/softbody/engine/hierarchy.py), [fields.py](src/transmogrifier/softbody/engine/fields.py), [coupling.py](src/transmogrifier/softbody/engine/coupling.py), [constraints.py](src/transmogrifier/softbody/engine/constraints.py), [collisions.py](src/transmogrifier/softbody/engine/collisions.py)
- geometry: [primitives.py](src/transmogrifier/softbody/geometry/primitives.py), [geodesic.py](src/transmogrifier/softbody/geometry/geodesic.py)
- bridge: [__init__.py](src/transmogrifier/softbody/bridge/__init__.py), [state_io.py](src/transmogrifier/softbody/bridge/state_io.py)
- resources: [field_library.py](src/transmogrifier/softbody/resources/field_library.py)
- demo: [__init__.py](src/transmogrifier/softbody/demo/__init__.py), [run_ascii_demo.py](src/transmogrifier/softbody/demo/run_ascii_demo.py), [run_numpy_demo.py](src/transmogrifier/softbody/demo/run_numpy_demo.py), [run_opengl_demo.py](src/transmogrifier/softbody/demo/run_opengl_demo.py)

## src/visualizations

- [src/visualizations/live_cassette_demo.py](src/visualizations/live_cassette_demo.py)
- [src/visualizations/reel_demo.py](src/visualizations/reel_demo.py)
- [src/visualizations/reel_demo_shell.py](src/visualizations/reel_demo_shell.py)
- [src/visualizations/reel_math.py](src/visualizations/reel_math.py)

## tests (highlights)

See the `tests/` directory for full coverage. Highlights include:

- Turing machine core: `test_turing_reference.py`, `test_turing_ssa.py`, loops/new opcodes
- Hardware/analog: `test_nand_wave.py`, lane tuner, header layout
- Visualizations: no-pygame and scaling
- bitbitbuffer: indexing/slicing/search, pidbuffer
- Cellsim: API/engine/physics/transport, membranes/collisions, organelle exclusion
- Transmogrifier graphs and memory graph dynamics
