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

## src/transmogrifier

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

## src/bitbitbuffer

- [src/bitbitbuffer/__init__.py](src/bitbitbuffer/__init__.py)
- [src/bitbitbuffer/bitbitbuffer.py](src/bitbitbuffer/bitbitbuffer.py)
- helpers
  - [src/bitbitbuffer/helpers/__init__.py](src/bitbitbuffer/helpers/__init__.py)
  - [src/bitbitbuffer/helpers/utils.py](src/bitbitbuffer/helpers/utils.py)
  - [src/bitbitbuffer/helpers/testbench.py](src/bitbitbuffer/helpers/testbench.py)
  - [src/bitbitbuffer/helpers/rawspan.py](src/bitbitbuffer/helpers/rawspan.py)
  - [src/bitbitbuffer/helpers/pidbuffer.py](src/bitbitbuffer/helpers/pidbuffer.py)
  - [src/bitbitbuffer/helpers/data_access.py](src/bitbitbuffer/helpers/data_access.py)
  - [src/bitbitbuffer/helpers/cell_proposal.py](src/bitbitbuffer/helpers/cell_proposal.py)
  - [src/bitbitbuffer/helpers/bitbitslice.py](src/bitbitbuffer/helpers/bitbitslice.py)
  - [src/bitbitbuffer/helpers/bitbititem.py](src/bitbitbuffer/helpers/bitbititem.py)
  - [src/bitbitbuffer/helpers/bitbitindex.py](src/bitbitbuffer/helpers/bitbitindex.py)
  - [src/bitbitbuffer/helpers/bitbitindexer.py](src/bitbitbuffer/helpers/bitbitindexer.py)
  - [src/bitbitbuffer/helpers/bitstream_search.py](src/bitbitbuffer/helpers/bitstream_search.py)

## src/cells

- [src/cells/__init__.py](src/cells/__init__.py)
- [src/cells/cell_consts.py](src/cells/cell_consts.py)
- [src/cells/cell_walls.py](src/cells/cell_walls.py)
- [src/cells/simulator.py](src/cells/simulator.py)
- bath
  - [src/cells/bath/__init__.py](src/cells/bath/__init__.py)
  - [src/cells/bath/adapter.py](src/cells/bath/adapter.py)
  - [src/cells/bath/discrete_fluid.py](src/cells/bath/discrete_fluid.py)
  - [src/cells/bath/fluid.py](src/cells/bath/fluid.py)
  - [src/cells/bath/voxel_fluid.py](src/cells/bath/voxel_fluid.py)
- cellsim
  - [src/cells/cellsim/__init__.py](src/cells/cellsim/__init__.py)
  - [src/cells/cellsim/constants.py](src/cells/cellsim/constants.py)
  - api: [src/cells/cellsim/api/saline.py](src/cells/cellsim/api/saline.py)
  - bath: [src/cells/cellsim/bath/reservoir.py](src/cells/cellsim/bath/reservoir.py)
  - chemistry: [src/cells/cellsim/chemistry/crn.py](src/cells/cellsim/chemistry/crn.py), [electrochem.py](src/cells/cellsim/chemistry/electrochem.py)
  - core: [units.py](src/cells/cellsim/core/units.py), [numerics.py](src/cells/cellsim/core/numerics.py), [geometry.py](src/cells/cellsim/core/geometry.py), [checks.py](src/cells/cellsim/core/checks.py)
  - data: [state.py](src/cells/cellsim/data/state.py), [species.py](src/cells/cellsim/data/species.py), [proposals.py](src/cells/cellsim/data/proposals.py)
  - engine: [saline.py](src/cells/cellsim/engine/saline.py)
  - membranes: [membrane.py](src/cells/cellsim/membranes/membrane.py), [gates.py](src/cells/cellsim/membranes/gates.py)
  - mechanics: [provider.py](src/cells/cellsim/mechanics/provider.py), [softbody0d.py](src/cells/cellsim/mechanics/softbody0d.py), [tension.py](src/cells/cellsim/mechanics/tension.py)
  - transport: [pumps.py](src/cells/cellsim/transport/pumps.py), [kedem_katchalsky.py](src/cells/cellsim/transport/kedem_katchalsky.py), [ghk.py](src/cells/cellsim/transport/ghk.py)
  - placement: [bitbuffer.py](src/cells/cellsim/placement/bitbuffer.py), [sync.py](src/cells/cellsim/placement/sync.py)
  - organelles: [inner_loop.py](src/cells/cellsim/organelles/inner_loop.py)
  - viz: [ascii.py](src/cells/cellsim/viz/ascii.py)
  - examples: [demo_sim.py](src/cells/cellsim/examples/demo_sim.py)
- simulator_methods
  - [src/cells/simulator_methods/cell_mask.py](src/cells/simulator_methods/cell_mask.py)
  - [src/cells/simulator_methods/data_io.py](src/cells/simulator_methods/data_io.py)
  - [src/cells/simulator_methods/injection.py](src/cells/simulator_methods/injection.py)
  - [src/cells/simulator_methods/lcm.py](src/cells/simulator_methods/lcm.py)
  - [src/cells/simulator_methods/logutil.py](src/cells/simulator_methods/logutil.py)
  - [src/cells/simulator_methods/minimize.py](src/cells/simulator_methods/minimize.py)
  - [src/cells/simulator_methods/quanta_map_and_dump_cells.py](src/cells/simulator_methods/quanta_map_and_dump_cells.py)
  - [src/cells/simulator_methods/visualization.py](src/cells/simulator_methods/visualization.py)
- softbody
  - [src/cells/softbody/__init__.py](src/cells/softbody/__init__.py)
  - engine: [src/cells/softbody/engine/__init__.py](src/cells/softbody/engine/__init__.py), [xpbd_core.py](src/cells/softbody/engine/xpbd_core.py), [params.py](src/cells/softbody/engine/params.py), [mesh.py](src/cells/softbody/engine/mesh.py), [hierarchy.py](src/cells/softbody/engine/hierarchy.py), [fields.py](src/cells/softbody/engine/fields.py), [coupling.py](src/cells/softbody/engine/coupling.py), [constraints.py](src/cells/softbody/engine/constraints.py), [collisions.py](src/cells/softbody/engine/collisions.py)
  - geometry: [primitives.py](src/cells/softbody/geometry/primitives.py), [geodesic.py](src/cells/softbody/geometry/geodesic.py)
  - bridge: [__init__.py](src/cells/softbody/bridge/__init__.py), [state_io.py](src/cells/softbody/bridge/state_io.py)
  - resources: [field_library.py](src/cells/softbody/resources/field_library.py)
  - demo: [__init__.py](src/cells/softbody/demo/__init__.py), [run_ascii_demo.py](src/cells/softbody/demo/run_ascii_demo.py), [numpy_sim_coordinator.py](src/cells/softbody/demo/numpy_sim_coordinator.py)

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
