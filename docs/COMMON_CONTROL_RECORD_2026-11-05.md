# A common control record for the dt system and its shells

Date: 2026-11-05. Survey and proposal; nothing here is implemented yet.

The question: one control object that a Python program simply hands to the
dt system each round, which the compiled program publishes as the control
hookups each shell (web page, native process, node harness, mobile page)
binds its devices to. This note records how controls reach compiled programs
today, lane by lane, and proposes the record that unifies them.

## 1. How controls reach a compiled program today (observed)

### The car game (Living Data Map, `abstract_ui_div_map.py` + `abstract_ui_vehicles.py`)

- The neutral model declares semantic actions and the native inputs that may
  express them: `ViewportControlBinding("move-forward", ("keyboard:KeyW",
  "gamepad:left-y-negative"))` and friends in
  `abstract_ui_viewports.DEFAULT_VIEWPORT_CONTROL_BINDINGS`; the
  `ViewportControlPolicy` adds capture, activation and rates;
  `ControlFocusPolicy` routes devices between `game`, `projected-pointer` and
  `dialogue`; `DeviceMonitor.from_bindings` derives the lamp panel from the
  same bindings. All of these are Python records serialized into the page.
- The page's input adapter (`viewportInputValue(action, gamepad)`,
  div_map 6507) reads keys and gamepad per semantic action and composes the
  vehicle's controls by hand: `throttle = forward * (running ? 1.18 : 1)`,
  `steering = strafe`, `brake = jumping` (div_map 8415-8426), then posts
  `{type: "vehicle-control", throttle, steering, brake}` to the physics
  worker (div_map 7807).
- The worker (`state_loop_deployment.py` 2212) parks it in
  `body.pendingControls`, commits it to `body.controls` at the next tick
  (1645), and the GPU adapter copies it into the compiled vehicle's feed by
  POSITION: `vehicle_feed[vi['throttle']] = controls[0u]`,
  `brake = controls[2u]`, `forward_gear_ratio = controls[3u]` ...
  (vehicles 4918-4930), against a hand-kept `control_abi` list of 36 names
  (vehicles 5010). The vehicle's own `control_mapping` says which semantic
  action feeds which control: `{"move-forward/backward": "throttle",
  "strafe-left/right": "steering", "run": "boost", "jump": "handbrake"}`
  (vehicles 9140).
- The compiled vehicle itself takes `throttle`, `brake`, `drive_direction`,
  `engine_enabled` ... as plain law inputs (`_symbols()`, vehicles 1796);
  the law never sees a device.

So the car game already has three of the four pieces: a semantic action
vocabulary with device bindings (Python records), a per-game mapping from
actions to law inputs, and law inputs that are ordinary feeds. What it lacks
is the record in between: the controls travel as an ad-hoc JS object and a
positional float array that two hand-written adapters must agree on.

### The native process shell (`python_native_shell.py` + `shell_io.py`)

- IR declares the facilities it needs (`ShellIOManifest`: requests such as
  `keyboard`, `pointer`, `display_double_buffer`; `ShellIOBinding(resource,
  entry_point, parameter)` binds a compiled API parameter to a shell
  resource). The resource vocabulary is `pointer.x`, `pointer.y`,
  `pointer.buttons`, `keyboard.bits`, `clock.seconds`, `display.unit_x/y`,
  `option.<name>`, plus system ports.
- `plan_shell_stack` picks the wrappers that provide the requested
  capabilities (`WEB_JAVASCRIPT_SHELL`, `NATIVE_PROCESS_SHELL`), and the
  generated pygame launcher fills each bound parameter from the live device
  every frame (`_resource`, native shell 461-480; key bits and pointer from
  pygame events, 630-645).

This is the most declarative control hookup in the tree: the program says
which parameter is the pointer, and whichever shell wraps it supplies the
pointer. Its vocabulary is device-level, not semantic.

### The dt system (`examples/llvm_dt_system.py`)

- `dt_system_over(state, targets, controller, round_dt, dt_initial, dx)` is
  the lowered unit's entry and its signature is fixed. `controller` is the
  `STController`, the dt controller, not a game controller.
- Game control reaches the laws as STATE: Meowdoku writes its `intent`
  column into `state.intent` before each round (`examples/meowdoku/
  session.py`), and the law reads it like any other column. That is the only
  mechanism the dt system has for input today, and it already compiles:
  every state field is a ProgramABI slot (`dt_system_contract`, the
  `PieceState` record) and the emission posts one `program_abi_field_slot`
  row per slot, which `NativeSystem.state_field_ids()` reads on every lane
  (C, LLVM, and since f63e112a the Wasm core).

### The mobile reference page and the node harness

- `turing-wasm-shell.js` invokes the core by NAME from the published
  contract (`program.abi.json`); the mobile page feeds `touch_x`, `touch_y`
  per frame. The node harness feeds by value id. Neither knows what a
  parameter means.

## 2. The gap, in one sentence

Every lane has a way to deliver device values to named parameters, and the
car game has a semantic action vocabulary, but there is no declared record
that says "these parameters are the program's controls, here is each one's
meaning, dtype and storage, and here is which semantic action fills it", so
each shell grows its own adapter and the car game ships a positional array.

## 3. Proposal: a ProgramABI control record

Use what is already there at every hop. The new thing is one record and one
rule.

### 3.1 The record

A control record is an ordinary ProgramABI record (`records` in the
extraction contract, like `Targets`, `STController`, `PieceState`): named
fields with dtype, storage and shape. Its fields are the program's control
inputs in the program's own words, declared once:

```yaml
records:
  MeowdokuControls:
    identity: examples.meowdoku.controls.MeowdokuControls
    fields:
      selected_cell:    {storage: scalar, dtype: int64,   mutable: false}
      primary_action:   {storage: scalar, dtype: bool,    mutable: false}
      secondary_action: {storage: scalar, dtype: bool,    mutable: false}
```

Each field carries one extra, declared attribute: `control: <semantic
action>` (`primary-action`, `secondary-action`, `move-forward`, `look`,
`pointer-cell` ...) from the viewport binding vocabulary, or a device
resource (`pointer.x`) from the shell IO vocabulary. That attribute is the
hookup. It is posted with the slot row, so it is a book fact a shell reads,
never a JS object a page remembers.

### 3.2 How it enters the dt system (Python)

The dt system's input mechanism is the state, and `dt_system_over`'s
signature is fixed, so the controls ride on the state the way `rollback`
and `allow_increase_mid_round` already do: `dt_system_contract(entry,
columns, batch, ..., controls=MeowdokuControls)` adds the record's fields
to `PieceState`, marked with their `control` attribute. In Python the game
hands the record to the round and nothing else:

```python
state.controls = MeowdokuControls(selected_cell=12, primary_action=True)
dt_system(pieces, columns, rounds=1, ..., state=state)
```

and the law derives what it needs from the fields (`intent` becomes a
function of `selected_cell` and the two actions inside `meowdoku_step`,
instead of a column the session fabricates). The eager lane and the
compiled lane then take the same object.

### 3.3 How it leaves the compiler

Nothing new: the emission already posts one `program_abi_field_slot` row per
PieceState field on every lane. The row gains the `control` attribute. The
host-facing contract (`core_contract` for Wasm, `layout.json` for native,
the `.api.yaml` for the fused lanes) lists, per parameter: name, dtype,
count, offset, and now `control`. That list IS the controller hookup table.

### 3.4 How each shell binds it

- Web page (`wasm_html_shell`, `site_bundle`, the Meowdoku page): the
  viewport's `ViewportControlBinding`s already map semantic actions to
  devices; `viewportInputValue(action)` already evaluates them. The adapter
  becomes generic: for each contract parameter with a `control`, write
  `viewportInputValue(control)` into that slot before the round. No
  per-game `throttle = forward * ...` glue; a scaled or composed control is a
  law input computed by the law from the semantic fields.
- Native process (`python_native_shell` + `shell_io`): a `control` that
  names a shell resource binds as a `ShellIOBinding` today; a semantic
  action binds through the same `ViewportControlBinding` table evaluated
  against pygame keys and the joystick, which is a small resolver beside
  `_resource`. The manifest requests `keyboard`/`pointer`/`system_devices`
  from the record, so `plan_shell_stack` picks a shell that has them.
- Node harness and tests: feed the control fields by name from a Python
  `MeowdokuControls` instance, which is exactly how the parity tests already
  drive a round.
- The car game: its `control_abi` list becomes this record (36 fields with
  their `control` attributes), the `vehicle-control` message carries the
  record, and the GPU adapter's `controls[0u]`/`controls[2u]` become
  by-name slot writes from the contract. `control_mapping` stops being a
  side table because each field names its action.

### 3.5 Why this is not new machinery

- Records, bindings and slot rows are the ProgramABI as it stands.
- The semantic action vocabulary and its device bindings are the viewport
  records as they stand.
- The device resources are the shell IO vocabulary as it stands.
- The dt system is used as is: controls are state fields, saved and
  restored with the state (so undo in Meowdoku restores the control too).

The one rule added: a field's `control` attribute is the only place a
program says how a shell fills it. Any shell that cannot honour a named
control refuses by name at bind time (a page without a gamepad still binds
keyboard inputs for the same action; a headless harness with no device
refuses a required control it was not fed).

## 4. First step

Meowdoku is the right first consumer: three fields, one page, the Python
session as the reference, and the shader page already renders Python state.

1. Declare `MeowdokuControls` and pass it through `dt_system_contract` as
   PieceState fields with `control` attributes.
2. Move intent derivation into `meowdoku_step` (from `selected_cell`,
   `primary_action`, `secondary_action`); the session hands the record.
3. Post the `control` attribute on the slot rows and surface it in
   `core_contract` and `layout.json`.
4. Make the page adapter generic: slot writes from
   `viewportInputValue(control)`; `pointer-cell` resolves the pointer to a
   board cell from the viewport bounds and the declared `n`.
5. Gate: the Python session and the page produce identical state for the
   same control sequence (the existing parity test shape), on the Wasm lane
   once the compiler items in the Meowdoku handoff land, and on the C/LLVM
   lanes before that.

Then the car game's `control_abi` migrates onto the same record, replacing
the positional `controls[]` array and the hand-composed throttle.

## 5. Open points for the owner

- Name. `controller` is taken by the dt controller (`STController`) in the
  contract bindings; the proposal uses `controls` for the input record.
- Scalar vs. span. Per-player controls are scalars; a batched world (many
  cars) wants spans of batch length like every other state column. The
  record form supports both; the first step uses scalars.
- Edge-triggered actions (`primary_action`) are booleans the shell sets for
  one round and clears; the law must not have to debounce. The dt round is
  the natural quantum, and Meowdoku already works that way with `intent`.
- The car game's vehicle law reads `throttle` scaled and sign-folded by the
  page; moving that composition into the law changes nothing numerically
  only if the same expression is authored there. That is a law edit and
  needs its own parity check.

## 6. Step 1 status (2026-11-05, evening): Meowdoku carries the record

Done, tests green (13 Meowdoku + 4 core-product):

- `examples/meowdoku/program.py`: `CONTROLS` (column -> semantic action:
  `select_row: pointer-cell-row`, `select_col: pointer-cell-col`,
  `primary_action: primary-action`, `secondary_action: secondary-action`),
  the `MeowdokuControls` record, and `meowdoku_step` deriving the move from
  those columns (`intent` is gone). The controls are ordinary piece
  arguments, so `column_names_of` makes them state columns and the state
  saves and restores them with everything else (undo restores the control).
- `examples/meowdoku/session.py`: the Python game does one thing with
  input, `hand(MeowdokuControls(...))`; `place`/`toggle_x` are spellings of
  it. `session.controls` reads the record back from the state.
- `extraction_contract.ProgramABIField.control`: the declared hookup, a
  string naming a semantic action or shell resource, allowed on scalar and
  span fields, carried in the receipt.
- `dt_system_contract(..., controls=CONTROLS)` stamps it on the PieceState
  fields and refuses a name that is not a column; `lowered_system(...,
  controls=)` passes it through and `NativeSystem.layout()["controls"]`
  joins each control to its slot (buffer index, value id).
- `wasm_core_product.core_contract(..., controls=)` publishes `control` on
  the parameter descriptor (by parameter name or by the root formal's
  `program_abi_field`) and refuses a control naming no input.
- `turing-wasm-shell.js`: `TuringWasmProgram.controlFeeds(sample)` fills
  every control parameter from `sample(action, parameter)` and refuses one
  the shell cannot answer; `invokeWithControls` is one round from the
  devices; `TuringControlSampler` evaluates the viewport binding vocabulary
  (keyboard codes, pointer buttons, gamepad sticks and buttons, the same
  rule as the Living Data Map's `viewportInputValue`) with page resolvers
  for geometry-bound actions and `pulse`/`settle` for edge-triggered ones.
  Verified in node: a held `KeyW` fills `move-forward` into the drift law's
  `v` and the round lands `x + dt`.
- The control-driven law lowers on canonical in 8.9 s to complete SSA (15
  root formals); the direct Wasm core shows the same two compiler-side
  items as before plus their cascades (Meowdoku handoff, Finding 3).

**Compiler-lane ask (one hop, their file).** The `control` attribute is on
the ProgramABI field declaration and in every host contract, but it is not
yet on the `program_abi_field_slot` row itself, because that row is posted
from the root formal's ACCOUNTING (`program_abi_storage`,
`program_abi_field_written` ...), and that accounting is stamped in
`fortran_c_shell.py` (12 sites, e.g. 40120, 40297, 40716, 41374), which is
under active work. The ask: where `program_abi_storage` is stamped from the
record field, also stamp `program_abi_control` from `ProgramABIField.
control`; `post_program_abi_field_slots` then carries it on `ProgramAbiSlot`
(one new field) and `NativeSystem.layout` / `core_contract` read it from the
row instead of the host-side join. Until then the host join is DERIVED from
the same declaration the row is derived from, keyed by (parameter, field).

Not done: the car game's `control_abi` migration (its 36 controls onto the
record, replacing the positional `controls[]` array), and the end-to-end
page round for Meowdoku, which waits on the compiler items.

## 7. Step 2 status (2026-11-05, night): the car game's controls are the record

- `abstract_ui_vehicles.VEHICLE_CONTROL_RECORD`: the 36 control slots,
  declared once, each `(name, control)`: `throttle: move-forward/backward`,
  `steering: strafe-left/right`, `brake: jump`; the other 33 are
  worker-derived state (transmission, locks, knuckle angles, body shell,
  tire pressure, outrigger wrench) and say so (`source: worker-state`).
  `VEHICLE_CONTROL_INDEX` is its slot order; `vehicle_control_record_data()`
  publishes it as rows on the program's graph adapters (`control_record`)
  and the vehicle model; `control_abi` is now the record's names and
  `control_mapping` a view of its semantic actions (plus the page's `run:
  boost` scaling, which stays declared until that composition moves into
  the law).
- Every WGSL read of the resident `controls` buffer is derived from the
  record: the assembly adapter's 25 feeds and the front-share/center-lock
  select by `VEHICLE_CONTROL_INDEX[...]`, the terrain kernel's body-shell
  gate and lane-indexed knuckle angles by two new placeholders filled from
  the record. No hand-written `controls[Nu]` remains; the module asserts the
  four knuckle angles are one contiguous run.
- The worker (`state_loop_deployment.emit_javascript_physics_worker`)
  builds the buffer BY NAME and in the published `control_abi` order
  (`residentVehicleControlValues(body, gpu)`), refusing a name it has no
  value for instead of keeping its own positional array.
- Proof of no behaviour change: the record-derived indices equal HEAD's
  literal indices for all 25 assembly feeds and the two select reads
  (checked against `git show HEAD:`); the rendered worker passes `node
  --check`; `tests/test_vehicle_control_record.py`: 4 passed.
- Not run: `test_gpu_terrain_contact_...` renders the full WebGPU program
  model and stops in `compile_sympy_equations` on current main (the
  2026-11-03 `post_derived_node_identity` regression, see the Meowdoku
  handoff), before any of this is reached.

Remaining for the car game: the page's `vehicle-control` message still
carries only the three shell controls by name (correct, since the rest are
worker state), and the page still composes `throttle = forward * (running ?
1.18 : 1)` itself. Binding the three through `TuringControlSampler` and
moving the boost composition into the law are the next two edits, each with
its own parity check.

## 8. Step 3 status (2026-11-05, night): the car page binds its controls through the record

- One grammar for a hookup, `abstract_ui_viewports.control_terms`: a
  single action (`jump`) or an axis of signed actions
  (`+move-forward -move-backward`), summed by the shell. The vehicle record
  spells its axes in it: `throttle: +move-forward -move-backward`,
  `steering: +strafe-right -strafe-left`, `brake: jump`.
- The Living Data Map samples the active vehicle's `control_record`
  generically (`vehicleControlValue(control, gamepad)` over
  `viewportInputValue`, with the on-screen joystick as one more device for
  the four movement actions) and hands throttle, steering and brake to
  `updateActiveVehicle`; the hand-composed `forward/strafe/jumping` triple
  for the vehicle is gone. The page's run-to-boost throttle scaling stays
  where the record declares it until it moves into the law.
- `TuringControlSampler.sample` reads the same grammar (`sampleAction` per
  term, summed), so a core page binds an axis control without any page
  code; verified in node: `KeyW` gives +1, `KeyS` gives -1 for
  `+move-forward -move-backward`.
- Checks: the whole 590 kB page script (the raw-string constant the
  function was inserted into) passes `node --check`; 5 record tests and 4
  core-product tests pass. The full page is not rendered here because the
  vehicle model's build stops in `compile_sympy_equations` on current main.

Left for the car game: the boost composition into the law, with its parity
check, and the worker-state slots staying worker-owned by design.
