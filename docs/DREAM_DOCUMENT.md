# Dream documents

`turing.dream-document.v1` is a byte-scanned polyglot program container. A
single `.dream` file may contain Python, JavaScript, GLSL compute, GLSL
fragment, or future language arenas without requiring the complete file to be
valid source in any one language.

## Blocks

Ordinary language arena:

```text
/*@turing.segment.v1
id = setup
language = python
encoding = utf-8
inputs = configuration
outputs = state
@end*/
...payload...
/*@turing.end*/
```

Shader arena:

```text
/*@turing.shader.v1
id = update
language = glsl
stage = compute
inputs = state
outputs = colors
abi = turing.shader-component.v1
@end*/
...shader source...
/*@turing.end*/
```

A shader block is intrinsically an in-place device deployment. It does not
need a separate language-switch or launch directive. The runtime raises the
GPU-active indicator when entering the block and clears it on departure.

Parallel deployment is structure rather than locking syntax:

```text
/*@turing.parallel.v1
id = frame
members = cpu-step, gpu-colors
join = frame-barrier
@end*/
```

Members are submitted together and joined. The runtime does not manufacture a
shared-state lock; blocks communicate through declared ports or deliberately
share an arena supplied by their host.

## Loading

The byte scanner validates restricted-ASCII headers, declared encodings,
unique identities, optional SHA-256 hashes, parallel membership, and complete
framing before invoking a language parser. It projects every executable block
to a `turing.card-graph.v1` card with a content-hash cache key, sequential read
path, resident boundary connections, parallel deployment records, and block
metadata. `DreamRuntime` is the read head over that graph.

Inspect the initial simulator:

```text
python -m src.compiler.dream_document examples/reversible_chip_simulator.dream --inspect
```

Reference-run its Python blocks and simulated shader deployments:

```text
python -m src.compiler.dream_document examples/reversible_chip_simulator.dream --run-reference
```

A live OpenGL host replaces only the shader-deployer callback; it does not
rewrite the document or its graph.

## Deployment IR and interior display ownership

`DreamDocument.lower_to_ssa()` lowers every block to an explicit SSA call and
publishes `SSADeploymentRegion` records in the module deployment table. A
parallel sentinel becomes an `independent_lanes` region with a barrier join.
Every shader becomes a `device_dispatch` region whose schedule names its
language and stage. The loader and shell therefore consume a compiler-owned
dispatch catalogue rather than interpreting comments at execution time.

A fragment block may promise:

```text
display-owner = program-interior
context = webgl2
```

Exactly one ordinary block must then declare `role = display-controller`. The
standard HTML shell allocates its canvas, selects the promised context, and
constructs the input/ABI liaison. It does not compile a display shader or own
the animation loop. Instead it calls the interior controller with the complete
context and embedded shader sources; the controller must return
`ownsDisplay: true` or the handoff fails closed.

Emit that launchable shell with:

```text
python -m src.compiler.dream_document examples/reversible_chip_simulator.dream \
  --emit-shell build/reversible_chip_live.html
```

The simulator document's Python arena contains the binary-machine program and
shell-tick controls. Its subject bytes remain runtime machine input rather than
cards. The HTML liaison exposes `machineSnapshots`, a newest-generation bus for
complete `TMSNAP01` buffers, plus an optional compiled `machineProgram` host.
The interior display subscribes to that bus, uploads the real register banks
and either an RGBA8 subject framebuffer or UTF-8 terminal when present, and
does not manufacture colored placeholder textures. The compiled descriptor declares `subject-binary` as a
file system port whose byte and length fields bind `load_subject`. HTML renders
that port as a file picker and exposes it through `TuringSystemPorts`; the
compiled program registers its handler without requiring a Python interpreter.

An existing system tape can seed the page with a real cold-resumed observation:

```text
python -m src.compiler.dream_document examples/reversible_chip_simulator.dream \
  --emit-shell build/reversible_chip_state.html --machine-tape cmd-machine.tape.jsonl
```

The emitter validates tape dependencies while rebuilding the executor, then
embeds and publishes its newest `TMSNAP01` buffer on page load. This makes a
self-contained examination artifact. It does not claim that the browser can
continue the guest until a compiled machine-program host is attached.

## Writing a document

`render_dream_document(document) -> bytes` is the exact inverse of `parse_dream_document`:
`parse_dream_document(render_dream_document(d)) == d` for every document it accepts. It refuses, and never alters,
anything the scanner would not read back unchanged: a payload containing the closing sentinel, a header value that is
not one restricted-ASCII line, a decoration named like a reserved key, an empty identity or language. Every block is
written with its `sha256` sentinel. A payload ending in a carriage return is closed with CRLF so the scanner drops the
pair together and keeps the byte. Any bytes can be carried as a block under `encoding = latin-1`
(`tests/test_dream_document_render.py`).

## The part-by-part compiler's record

The source divider (`examples/meowdoku/source_divider.py`) takes a dream document in and writes one out.

This record is the COMPILED UNITS stage. It is not the Goal document the owner named on 2026-11-11: the units are
each buildable alone, and they have not had their linking done (a part that calls another carries that other's text
inside its own closure). See the SpeakToMe repository, `AGENTS/conceptual_flags/Goal_Document.md`.

The Goal document is written beside it as `<job>/goal.goal` by `write_goal_document`, in the format
`speaktome.goal-document.v1` (SpeakToMe `speaktome_goal/`, manual page `goal(5)`). There a unit holds only its own
text; what it takes from another unit is a row of its `link` table; `api`, `build` and `abi` tables say what it
offers, which lowerings exist and how they are called; each `build` row names the block of this record that holds
the product's bytes. The record keeps the bytes; the Goal document keeps the joins.

- In: `intake_dream`. Each python block becomes one source file and is divided on its own. Blocks in other languages
  are not divided; they are carried into the record unchanged. Blocks an earlier pass produced (`role = lowering` or
  `role = job-record`) are dropped, so a record can be fed back in.
- Out: `write_dream_record` -> `<job>/record.dream`.
  - One python block per piece (`role = part`), dependencies first. Its payload is the exact lexical closure of the
    piece, the cut `partition_authored_source` gives and `run_attempt` compiles, with the blank lines before and after
    it dropped (`first_line` records where it began). Nothing in a block refers to another block's text, so each part
    can be built by itself. `outputs` are the functions the piece is for; `inputs` are those of the pieces it depends
    on, whose definitions are inside the closure.
  - After a piece, one block per completed lowering (`role = lowering`, `lowers = <piece>`), in the language of the
    backend (`javascript`, `c`, `llvm`, `fortran`, `wasm`), holding the product byte for byte; its content hash is the
    product's hash. A wasm product is carried under latin-1.
  - What was tried and refused is on the piece's header: `attempt.<backend>`, `verdict`, `reason`.
  - A last `record` block (`language = json`, `role = job-record`) states the job as data.
- The file is checked by reading it back with `load_dream_document` before it replaces the previous one.

Measured (`tests/test_meowdoku_source_divider.py`): each part of a three-function chain executes standing alone; a
record given back to the divider, with every part now its own block, rebuilt each function to the same product bytes;
a shader block added to the input came out unchanged.

Limits: pieces are cut within one block, so a python block that calls a function defined in ANOTHER input block is not
resolved (the divider follows no cross-file imports yet). A parallel group whose members were divided is dropped and
named in the record block. `compile_sections` knows `python`, `javascript`, `glsl`, `wgsl`, `sympy` and `cpp`; the
lowering blocks in `c`, `llvm`, `fortran` and `wasm` are carried as source, not interpreted. The web intake page does
not accept a dream document yet; the command line does (`submit --dream FILE`).
