# AbstractUI hotbar and control focus

Device capture and interaction focus are different authorities. A viewport may
retain ownership of keyboard, pointer, and gamepad devices while routing their
output to one of three contexts:

- `game` drives the actor and camera;
- `projected-pointer` publishes pointer motion in document coordinates and
  permits ordinary page interaction;
- `dialogue` temporarily owns response input and resumes the previous context
  when released.

Secondary action switches game and projected-pointer contexts. Dialogue focus
has higher priority and cannot be switched away by that action. Browser pointer
lock is one native realization of capture, not the neutral meaning of capture;
it may be released while logical viewport ownership remains in projected mode.

The hotbar is an editable view of inventory slots 1 through 10. Its physical
bindings are `Digit1` through `Digit9`, followed by `Digit0` for slot 10.
Numeric selection is consumed only while the viewport owns focus and no
dialogue is active. Slot selection changes inventory's active-tool reference
and emits an observable action edge.

Game focus binds either Shift key to the semantic `run` action and Space to
`jump`. Run multiplies manual planar movement by the control policy's
`run_multiplier` (2.0 by default). Jump is edge-triggered, cancels an active
auto-locate route, and submits a 3.6 m/s vertical velocity to the compiled
physics loop only while the actor is grounded. Holding Space cannot repeatedly
relaunch the actor. Outside game focus Space retains its normal document and
accessibility behavior.

The initial inventory contains one `Form tool` in slot 1. Both the inventory
record and hotbar slot point to the same item identity; the hotbar does not own
a duplicate tool. Empty slots remain explicit so later edits can place, move,
or remove items without changing the keyboard ABI.

## Tool hooks and aesthetic dialogue

A tool is an AbstractUI object, not merely an inventory flag. It owns semantic
hooks mapping `primary-action` and `secondary-action` to operations and
destinations. Native pointer and gamepad buttons route through the active
inventory tool before a backend performs those operations.

The first Form tool maps primary action to its model-authored aesthetic
dialogue and secondary action to focus-context switching. The dialogue claims
dialogue focus and exposes face color, wall color, wall height, wall thickness,
and corner radius plus Verdant, Warm, and Stone presets. Changes update the
focused geometry identity, rebuild the shared scene mesh, and republish DOM
wall/appearance variables. Done or Escape releases dialogue focus and resumes
the previous game or projected-pointer context.

## Returning to compiled defaults

The placement panel includes a `Return to defaults` button. It removes only
the current world's scoped AbstractUI edit cookie and its matching
local-storage fallback, clears the in-memory dirty set, and reloads the page.
The regenerated model then starts from its compiler-authored geometry,
appearance, placement stock, and physics parameters. Unrelated cookies and
other worlds' saved edits are not touched.
