# WebAssembly class-memory coordinator

Implemented and runtime-verified 2026-08-01. This is the v3 successor to the
host-scheduled v2 deployment recorded in `WASM_CLASS_GRAPH_HANDOFF.md`.

## Contract

A lowered class is represented by one `turing.class-memory-inventory.v1`
descriptor:

- one resident memory block;
- indexed field slots whose values are byte addresses in that memory;
- an ordered method-card inventory;
- each method card's imported function, input field slots, and output field
  slots;
- one range coordinator with the ABI
  `run_range(count, field_slot_table, start, end)`.

`end` is exclusive. A full automatic run calls the coordinator once over the
whole method inventory. A debugging shell can call adjacent ranges and hold a
latch between them. JavaScript therefore performs no per-card calls during a
normal staged run; per-card host crossings exist only when the user explicitly
enables boundary breakpoints.

This is intentionally class-shaped rather than Mandelbrot-shaped. Future OOP
lowering can produce the same memory, fields, and method inventory descriptor.
The Mandelbrot program is simply its first published instance.

## One source-level shell, two runtimes

`wasm_class_coordinator.build_coordinator_control` constructs the schedule as
the existing target-neutral `ControlProgram`: a Python-authored `LoopBlock`
containing a `StateMachineTick` over the method inventory. The same object is:

1. rendered and compiled by `render_python_shell` / `compile_python_shell` for
   ordinary non-browser process coordination;
2. lowered by `lower_control_program_to_ssa`, including the loop induction and
   region calls;
3. emitted as a root WebAssembly module which imports every method card and
   the class memory.

The Python coordinator accepts `(memory, inventory, count, start, end)` and
calls `inventory.call(index, memory, count)`. The WebAssembly coordinator
accepts the corresponding resident field-slot table and loads each method's
pointer arguments from it. The deployment order is not reimplemented in
JavaScript.

## Published v3 modes

The generated homepage initially selects no execution mode and keeps Run
disabled. A user must choose one of:

- Mono / contiguous;
- 200 lowered operations per method card (14 cards);
- 400 lowered operations per method card (7 cards);
- 800 lowered operations per method card (4 cards).

The choice itself downloads nothing. Mono is fetched on its first run. A
staged choice fetches its versioned cards and coordinator on its first run,
instantiates them once against one memory, and caches the resulting class
instance. Source files remain completely unloaded until their individual
Download buttons are clicked.

The optional boundary-breakpoint checkbox turns each card boundary into a
shell latch. In that diagnostic mode the shell invokes one-card coordinator
ranges and waits for Release latch. With the option off, the shell makes one
call into WebAssembly for the complete inventory.

Artifacts are additive under:

```text
site/v3/wasm/render_contiguous.wasm
site/v3/wasm/size-200/
site/v3/wasm/size-400/
site/v3/wasm/size-800/
```

Each size directory contains its method cards, coordinator `.wasm`, the exact
generated Python shell, a readable WAT outline, and `class-inventory.json`.
Language sources are separately versioned under `site/v3/source/render/`.

## Memory behavior and honest limitation

Every method imports the same `env.memory`; tensor payloads are never copied
through JavaScript at a seam. The inventory table contains offsets, not tensor
data. Fan-out consumers reuse the producer field address.

The current topological partition still assigns a resident full-domain array
to every live region output. Shared memory removes transfers and the WASM
coordinator removes host call overhead, but neither optimization removes those
boundary materializations. Larger card sizes reduce the number of boundaries;
Mono removes them entirely. Future lifetime/alias analysis can reuse field
storage without changing this class ABI.

## Verification

The focused compiler suite passes 84 tests. A Node/WebAssembly run instantiated
the actual generated v3 files, invoked each size's coordinator over its full
range, and compared all three outputs against the contiguous module at the same
inputs. The 200-, 400-, and 800-operation variants matched Mono exactly.

The dedicated coordinator regression also verifies latched range behavior: the
first range produces the expected resident seam and the second range consumes
that same memory field to produce the public result.

Only the reduced Mandelbrot/Julia math-and-colour toy is compiled. AVI/JPEG
encoding remains outside this build.
