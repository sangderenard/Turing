# Arena line-driver prototype

The driver consumes data-only JSON with schema `turing.arena-line-driver.v1`.
It does not load user JavaScript, Wasm, URLs, or expressions. The trusted
`graph_policy_forward` in `examples/meowdoku/pilot_law.py` is compiled through
`graph_wasm.build_pilot` and loaded by the existing `TuringGraphProgram`.

## Packet

- `network.architecture`: `graph-15-5-3-tanh-gates-v1`.
- `network.weights`: each of GW1, Gb1, GM1, GW2, Gb2, GM2 has `shape` and
  flattened row-major `data`, matching the existing pilot weight-file format.
- Shapes: GW1/GM1 `[15,5]`, Gb1 `[1,5]`, GW2/GM2 `[5,3]`, Gb2 `[1,3]`.
- Masks GM1/GM2 are 0 or 1. Other weights are finite numbers within +/-100.
- `circuits`: 1..8 records with distinct `id`, `color` (#RRGGBB), and `segments`.
- Each segment has `support` (exact registered collider identity), `from`
  and `to` (world `[x,z]` in metres). Segments are connected and ordered.
- Maximum upload: 256 KiB, 512 segments total, 16,384 projected points total.
  Unknown fields, unsupported architectures, malformed shapes, nonfinite
  values, missing supports, disconnected support heights, and oversized
  sampling work are refused before running a replacement network.

## Ground and reference plane

Each segment is linearly interpolated at intervals no larger than 0.25m in x/z,
then every sample's y is projected using its explicitly named support. Sampled
terrain uses the game's existing `sampleDeclaredSurface`. A named axis-aligned
solid uses its top. Rotated supports are rejected until the native oriented
surface contract is available. No highest-ground query chooses the layer.
A bridge deck and the ground beneath it must have different support identities.
Absent/quarantined bridge slabs are rejected, not silently replaced.

The sensor is the vehicle chassis-local x/z plane at its origin. Its basis
comes from `rotateVehiclePresentationVector`, including yaw, pitch and roll.
The network input is four local xyz offsets to the selected line, at path
arc-length lookaheads 0, 1, 3 and 6 metres, each divided by 10; then chassis-local
forward speed / 10, lateral speed / 10, and yaw rate in rad/s. The y coordinate
is retained as the signed height relative to the plane. Progress searches only
the current connected section, not all segments across crossing roads.

The five hidden gates and three output neurons use the EXISTING graph network:
`gates=(tanh(x @ (GW1*GM1)+Gb1)+1)/2`,
`action=tanh(gates @ (GW2*GM2)+Gb2)`.
Outputs are steering, forward throttle, brake. Actuator bounds are [-1,1],
[0,0.5], [0,1]. Reverse is deliberately not in v1. The next original game frame
feeds those controls through `updateActiveVehicle` to the native vehicle worker.
There is no vehicle pose write, alternate physics clock, or waypoint teleport.

## UI and safety

Load test circuits or upload a packet, select a circuit, mount the truck and
bring it near the circuit's start, then press Start. The UI uses the existing
game input-focus request. It never starts an uploaded route automatically.
Steering/throttle/brake input immediately takes over. Stop, invalid inference,
leaving game controls, and hiding the document stop the network and brake.
A completed circuit brakes; Start is required for another lap. Loading or
starting rechecks supports. Surface edits during a run require stop and reload;
this prototype does not certify continuous collision-free drivable routes.

Colored lines are additive render mesh paint, not new collision surfaces.
The two included blue and amber oval circuits lie side by side on the same
registered arena terrain. They are test markings, not proven navigable roads.
The complimentary five-hidden-neuron weights are HAND-SEEDED, UNCALIBRATED,
and not trained. Successful compilation/parity is not evidence of driving skill.

## Reproduction and staging

Use the system Python installation (the repo .venv currently lacks llvmlite).
Set `TURING_WASM_CLANG` to the installed clang executable if Zig 0.16 rejects
LLVM input with `Unexpected`. This setting only selects the compiler driver in
`llvm_training_runtime._compile_artifact_to_wasm`; emitted IR and ABI are unchanged.

Build: `python tools/build_arena_line_policy.py DESTINATION MATCHED_MODEL_JSON`.
Stage: `tools.add_arena_line_driver.patch_arena_line_driver` takes exact original
HTML bytes plus their SHA256, returns patched bytes and a reverse-byte receipt.
Copy assets beside the HTML under `arena-line-driver/`: `arena_line_driver.js`,
`turing-wasm-shell.js`, `pilot.wasm`, `pilot.abi.json`, `starter-circuits.json`.
No network assets are downloaded from uploaded data. The adapter preserves the
original model, physics worker and existing navigation. No live deployment made
by this task. The HTML adapter, UI and actual driving still need browser QA.
