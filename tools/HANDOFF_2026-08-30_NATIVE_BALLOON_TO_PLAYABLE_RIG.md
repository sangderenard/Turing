# Native balloon compiler to playable validator rig handoff

**Date:** 2026-08-30  
**Repository:** `C:\dev\Powershell\turing`  
**Immediate objective:** make the authored Python balloon-tire validator run as
a correct Python-free native executable, then host that exact compiled material
as a persistent car-building rig inside the game world.

## Read this first

The native program now **compiles, links, launches, loads its material buffers,
and executes one frame without Python**. It is not yet numerically correct and
must not be described as playable. The first frame currently corrupts the tire
state to non-finite values and leaves the public output buffer zero.

The remaining first corruptor is known. Two planned regions receive linked
multi-output results as structural aggregates and treat those aggregates as
flat `double *` tensor storage. They must instead receive the already
materialized tensor projections. This is shared SSA tuple legalization, not a
scheduler, heap, shader-width, or physics-stability problem.

Do not widen the work until this exact boundary is repaired and a native frame
is finite.

## User intent and non-negotiable architecture

- The source is ordinary Python written against `AbstractTensor`; the compiler
  must make the executable. Do not reimplement the validator by hand in C.
- Use the canonical repository-SSA path. Do not use the deprecated fused AOT
  product route to force a pass.
- Do not fuse the whole program. Tensor math may retain its normal local fusion
  and identity reductions.
- Do not add a scheduler. The compiler and its explicit frame/heap already own
  scheduling and storage.
- Tensors may live in the compiler's explicit non-recursive frame arena. They
  must not become multi-megabyte automatic locals on the Windows native thread
  stack.
- Preserve the buffer-oriented material ABI. Do not invent a second heap or
  silently replace caller buffers with backend-private storage.
- C, Fortran, LLVM, Wasm, and graphics channels must converge on the same
  material contract. LLVM and Fortran are lead ABI references where the C
  backend is incomplete.
- GLSL/WebGPU conversion belongs at an explicit graphics-channel boundary. It
  must not change the CPU tensor arena to 32-bit storage.
- The eventual native shell may use C++ for SDL/OpenGL/UI ownership, but the
  compiled math remains producer-neutral and may mix C, Fortran, and LLVM
  sections.
- The existing recursive Living Data Map game site is the web shell. Do not
  replace it with a generic new game page.

## Supported entrypoints

Canonical lowering:

```python
from src.compiler.vehicle_python_compilation import lower_balloon_tire_python_ssa
```

Canonical native build:

```powershell
python tools/build_balloon_tire_native.py `
  --output build/balloon-tire-native `
  --backend c --batch-size 8 --frames 1
```

The public convenience entry is:

```python
compile_balloon_tire_python_native(directory, batch_size=8, backend="c")
```

`compile_balloon_tire_python_aot` remains a compatibility adapter for callers
that explicitly inspect the historical planning product. It is not the native
product route.

## Exact current artifact state

Current executable:

```text
C:\dev\Powershell\turing\build\balloon-tire-native\balloon_tire_native_c.exe
```

It launches successfully and reports:

```text
entry=balloon_tire_native_c frames=1 buffers=10
```

The host is Python-free. It owns a `void **buffers` material table, reads
`initial-state.bin`, invokes the compiled entry, and writes
`final-outputs.bin`. Paths are resolved beside the executable. A relative-path
duplication in `CStandaloneExecutable.run()` was fixed by storing an absolute
build destination.

One-frame numerical characterization after the latest build:

```text
inputs               changed=0      nonfinite=0
state                changed=23808  nonfinite=12289
output               changed=0      nonfinite=0
wheel_input_indices  changed=0      nonfinite=0
rest                 changed=0      nonfinite=0
face_vertices        changed=0      nonfinite=0
face_rest            changed=0      nonfinite=0
face_scatter         changed=0      nonfinite=0
laplacian             changed=0      nonfinite=0
bead_mask            changed=0      nonfinite=0
```

All public buffers are physically float64 on this extraction contract,
including semantic topology indices and masks. Gather/index helpers perform
their explicit integer conversions. The semantic dtype remains available in
SSA metadata.

## Confirmed fixes in the current working tree

### Standalone C product

`src/compiler/ssa_c_backend.py` now provides:

- `CStandaloneExecutable`;
- `CModuleArtifact.compile_standalone()`;
- a Python-free pointer-table host;
- sibling initial/final state files;
- authored-feed authority for public buffer sizes;
- static placement for large interim compiler temporaries and callouts so the
  Windows thread stack is not used as a tensor arena.

The static declarations are a proof/interim placement. The game shell should
move compiler-owned workspace into a per-instance arena so multiple cars and
rigs do not share temporary state.

### Persistent assignment publication

Repository `index_set_double` represents functional subscript assignment. The
C emitter now recognizes `SubscriptStore` and lowers it through
`index_assign_double` into the resident source arena. This repaired the earlier
failure where final-region state/output update chains ended in temporaries and
were never published.

Do not replace this with whole-buffer copies or a scheduler.

### Physical tensor dtype across call edges

Semantic boolean masks `%1280`, `%1540`, and `%1708` were emitted as
`int32_t[]` and then consumed by `*_double` helpers. The lowering already
carried `physical_dtype: float64`; the C emitter discarded it.

The emitter now:

- honors `SSAValue.accounting["physical_dtype"]` for buffer declarations;
- solves physical element-type equality across pointer call edges and planned
  caller/callee output slots;
- lets concrete repository helper formals constrain internal temporary types;
- retains integer scalar controls and genuine integer shape/index arrays;
- reports a `buffer_abi` shortfall on irreconcilable strongest constraints.

The generated C now declares the affected masks and broadcast temporaries as
`double`. The NaN count did not change, proving this was a real ABI defect but
not the remaining first corruptor.

### Native stack versus compiler frame

The earlier Windows stack overflow was caused by roughly tens of megabytes of
automatic C arrays, not by the repository's explicit non-recursive stack plan.
Keep this distinction explicit:

```text
Windows thread call stack: small control locals only
Compiler frame/workspace: tensor arenas and persistent material buffers
```

No Windows semantic adapter is presently required. The final shell needs a
per-instance workspace owner, not a different tensor model.

## Confirmed current root cause

Compile the generated C with warnings enabled or inspect these calls:

```text
planned_region_4(aggregate2320, ...)
planned_region_5(aggregate2327, ...)
```

The caller materializes these as pointer tables:

```c
double *aggregate2320[] = { ...three real gas tensor addresses... };
double *aggregate2327[] = { ...real membrane tensor addresses... };
```

But the region signatures currently say:

```c
planned_region_4(double *v467, ...)
planned_region_5(double *v718, ...)
```

Region 4 then loads pointer bits as scalar doubles. Region 5 does the same and
passes uninitialized `tmp1836`/`tmp1841` arrays into reductions. This is the
first demonstrated source of the non-finite state.

AddressSanitizer and UndefinedBehaviorSanitizer complete the frame without a
reported memory-boundary violation. The failure is incorrect structural
interpretation and missing publication, not a heap overflow.

## Correct next repair: shared aggregate-adapter legalization

Do not cast `double **` to `double *`, flatten the pointer table, synthesize
numeric tuple elements, or create C-only copies.

The desired SSA transformation is:

```text
producer aggregate call
  -> explicit real projections already materialized in caller
  -> consumer region's aggregate formal expanded to those projections
  -> projection-only loads removed from consumer
  -> pass-through consumer outputs aliased to producer projections
  -> only genuinely computed consumer outputs retained
```

For region 4, all selected outputs are pass-through gas tensors. The adapter
can collapse completely after its uses are rebound.

For region 5, most selected membrane outputs are pass-through tensors, while
`%1838` and `%1843` are new reductions. Expand the real membrane tensors into
the region inputs, remove pass-through outputs, and keep those two computed
outputs.

Useful existing shared analysis:

- `src/compiler/ssa_aggregate_abi.py`
- `propagate_repository_ssa_call_metadata()` in
  `src/compiler/tensor_ssa_lowering.py`
- the linked-call/result-correlation logic near the aggregate comments in
  `src/compiler/fortran_c_shell.py`
- aggregate diagnostics in `tools/TRANSLATION_DEBUGGING.md`

Preserve object identity and `ssa_storage_alias`; integer SSA ids can collide
across caller/callee/planner domains. Never replace every numerically equal id.

Add focused tests covering both adapter forms before rebuilding the vehicle:

1. a projection-only aggregate adapter disappears or aliases real projections;
2. an adapter with pass-through projections plus a new reduction receives real
   tensor operands and retains only the new output;
3. C/LLVM/Fortran observe the same legalized call record;
4. no synthetic aggregate formal enters the public/root material frame.

## Tests currently known green

```powershell
python -m pytest tests/test_ssa_c_aggregate_constants.py -q
```

Result: `16 passed`.

Focused Fortran coverage previously run:

```powershell
python -m pytest tests/test_ssa_fortran_and_optimizing_llvm.py -q `
  -k "pointer_array_stack_group or generic_index_addresses or dynamic_rank_two_contract or raw_pointer_formal or resolves_callee_dynamic_extents"
```

Result: `5 passed`.

Fortran is not complete. It still needs the same structural aggregate feed
legalization plus mutation-through-GEP/output publication cleanup. Do not claim
Fortran parity from the focused result.

After aggregate legalization, rebuild once and check in this order:

1. generated region signatures contain no aggregate pointer-table warnings;
2. one frame returns normally;
3. every public buffer is finite;
4. state changes and remains physically bounded;
5. output changes from zero;
6. compare C with the eager Python reference;
7. run 60 frames only after the one-frame differential passes;
8. run focused C, Fortran/LLVM, assignment, storage, deployment, and GLSL
   suites;
9. run `git diff --check`;
10. update `tools/TRANSLATION_DEBUGGING.md`, hazards, and an experience report.

## Do not repeat these detours

- Do not add a scheduler. No scheduling defect has been demonstrated.
- Do not invent a heap. Use the existing explicit material/workspace boundary.
- Do not blame GLSL 32-bit storage. No shader runs in the failing native tick.
- Do not globally coerce every bool/int tensor to double. Honor the explicit
  physical dtype and call-edge ABI; shape/index buffers remain integer.
- Do not infer public buffer sizes from a larger internal storage view. The
  authored feed/extraction receipt owns the public size.
- Do not treat a late unknown helper or concat as proof that an operator body
  is absent. Follow source call -> linked call -> projections -> shared ABI ->
  backend emission.
- Do not run a multi-minute vehicle rebuild between already-known local fixes.
  Finish the bounded shared repair and focused tests first.
- Do not edit or clean unrelated dirty-tree files. This workspace contains a
  large amount of concurrent user work.

## From finite validator tick to a playable game

The shortest product sequence is:

1. legalize the aggregate adapters and prove finite/equivalent tire frames;
2. publish a manifest naming every public buffer, physical dtype, shape, count,
   and authored source field;
3. replace interim global static compiler workspace with one explicit
   per-instance workspace owned by the shell;
4. host a car material frame in the SDL/OpenGL world tick;
5. pass fixed `dt`/`subdt`, controls, and world contacts through the same ABI;
6. render the compiled car with the native GLSL channel and make it drivable;
7. host a second material frame as the validator rig;
8. let the rig accept material balls, validate/build incrementally, project
   line-shader parts, transfer installed parts into solid/Phong rendering, and
   release the completed frame as a live `Car` world object;
9. expose the same material/section manifest to Wasm/WebGPU in the existing
   Living Data Map site.

The rig lifecycle should remain explicit:

```text
idle
-> accepting material
-> validating/building
-> line projection and actuator installation
-> solid-part ownership transfer
-> completed car release
```

The world should initially own the free/player car, the hidden original car,
and a rig already holding enough material for a backup build. Damage can make a
car unusable; crafting produces a new instance rather than repairing a visual
facsimile.

Broader game/validator design is recorded in:

```text
docs/VEHICLE_VALIDATOR_TO_GAME_HANDOFF.md
```

Treat that document as product context. Treat this dated document as the
current compiler/runtime frontier.

## Files most relevant to the next agent

- `src/compiler/ssa_aggregate_abi.py`
- `src/compiler/tensor_ssa_lowering.py`
- `src/compiler/fortran_c_shell.py`
- `src/compiler/ssa_c_backend.py`
- `src/compiler/ssa_fortran_backend.py`
- `src/compiler/ssa_llvm_backend.py`
- `src/compiler/vehicle_python_compilation.py`
- `src/compiler/vehicle_balloon_tire_program.py`
- `tests/test_ssa_c_aggregate_constants.py`
- `tests/test_ssa_fortran_and_optimizing_llvm.py`
- `tests/test_tensor_ssa_call_metadata.py`
- `tools/build_balloon_tire_native.py`
- `tools/TRANSLATION_DEBUGGING.md`
- `docs/VEHICLE_VALIDATOR_TO_GAME_HANDOFF.md`

## Final status sentence

The repository can now produce and launch a standalone native C executable
from the authored Python balloon program, but the first tire frame is not yet
valid because two planned consumers still interpret linked tuple containers as
numeric tensors; expand those consumers onto the real projections in shared
SSA, prove one finite equivalent frame, and only then advance to the
per-instance game shell and in-world car-building rig.
