# Handoff: system crash during investigation, and where the real work stands

Read this file first. It exists because a background command this
session likely destabilised the host machine. The cause is diagnosed
below from safe, read-only evidence -- **it was not reproduced on
purpose, and should not be reproduced on purpose.**

---

## What almost certainly happened

Near the end of the previous session, while chasing a state-ordering bug
(see "The real investigation" below), I ran this, as a background task,
to test a hypothesis:

```python
module, outputs, exports = lower_ast_source_to_ssa(
    SYMBOLIC_FLUID_DT_SOURCE,
    'symbolic_fluid_frame',
    python_bindings={'Metrics': Metrics, 'run_superstep': run_superstep},
    linked_process_graphs={'symbolic_fluid_step': symbolic.process_graph},
    name='symbolic_fluid_control',
    runtime_closure_only=True,
    extraction_contract=None,   # <-- the dangerous part
)
```

`extraction_contract=None` removes the extraction contract entirely.
That is not "no restrictions" in a benign sense -- it removes the ONLY
thing that keeps native-code decompilation switched off.

**The evidence, all found by reading, not by re-running anything:**

1. `extraction_contracts/program_extraction.yaml` (the contract this
   compiler normally uses) contains, explicitly:

   ```yaml
   # Machine decompilation is forbidden unless an exact rule opts in.
   limits:
     machine_decompilation:
       enabled_by_default: false
       max_functions: 0
       max_total_bytes: 0
       max_dependency_depth: 0
   ```

   The file's own comment on the `limits` section: *"Zero means
   unbounded. Reachability is bounded by the exhaustive action table
   above, not by arbitrary program-size truncation."* -- i.e. if
   decompilation ever starts, NOTHING bounds how much it will do. The
   only thing stopping it from starting is `enabled_by_default: false`,
   enforced by the contract.

2. `src/transmogrifier/graph/graph_express2.py::_expand_unresolved_ast_parents`
   (called from `build_from_ast`, which is what the real compile path
   uses) has this in its own docstring, about how it treats a callee
   with no Python source available:

   > "so a source-less builtin can be decompiled rather than becoming an
   > unexplained callee token later in reduction"

   That is the x86 read-head machinery
   (`src/compiler/x86_tensor_read_head.py`,
   `src/compiler/binary_ingestion.py`, `src/compiler/pe_recompilation.py`)
   -- a real PE/machine-code decoder, meant for the "recognize native
   code without lowering it" feature, walking actual compiled binaries.

3. With `extraction_contract=None`, `parent_include=None` reaches this
   code with nothing gating it. `symbolic_fluid_frame`'s reachable
   closure, through `run_superstep`/`step_with_dt_control_used`
   (`src/common/dt_system/dt_controller.py`), pulls in a much larger,
   deeper set of calls than the isolated single-function capture that
   works correctly elsewhere in this codebase -- almost certainly
   including compiled/native code with no Python source (NumPy's own C
   extension, or a JIT/DLL artifact resident from earlier compiles this
   session). Any one of those, unresolved, with no contract to reject or
   bound it, is a plausible trigger for the decompiler to start walking
   real machine code with `max_total_bytes: 0` (unbounded).

4. The background task's own output file contained only `[exited with
   code 0]` -- no stdout at all, not even the script's own final
   `print(...)` statement that should have run on a clean return. That
   is consistent with the process being killed externally (OOM, or a
   host-level intervention) rather than genuinely completing with
   nothing to say.

**This is a real risk in this codebase, not a fluke.** The contract
exists specifically because unrestricted dependency pursuit can reach
native code, and the decompiler has no default bound. Passing
`extraction_contract=None` on anything beyond a deliberately tiny,
fully-isolated snippet removes that safety, on purpose or not.

### Do not do this

- **Never call `lower_ast_source_to_ssa`, `ProcessGraph.build_from_ast`,
  or anything that reaches `_expand_unresolved_ast_parents` with
  `extraction_contract=None` / `parent_include=None`** on a program with
  any real reachable closure. If a contract-free test is genuinely
  needed, it must run in an isolated subprocess with a hard memory/time
  cap (a Windows Job Object, or `resource.setrlimit` under WSL/Linux),
  never inline in the main session.
- If the underlying question is "does the extraction contract's pruning
  hide or distort a dependency edge", the safe way to ask it is to copy
  `program_extraction.yaml`, adjust only what's needed (e.g. widen
  `roots`), and **keep `machine_decompilation.enabled_by_default:
  false`**. That tests the same thing without removing the guard.
- One earlier, smaller-scope check this session (`build_from_ast` over
  just `symbolic_fluid_advance`'s own source, also without a contract)
  completed cleanly and did not visibly destabilise anything. That does
  not make it safe -- it most plausibly means that narrower closure
  never happened to reach anything source-less. Treat both as unsafe in
  general; the size of the closure is what decides whether the risk
  fires, and that is not something to gamble on again.

### For a fresh agent picking this up

Check `git status` and `git log` first. If mid-session state (background
tasks, running builds) looks abandoned or inconsistent, that is expected
-- the host may have been rebooted. Recovering to a known-good state
(the last real commit) is safe; anything claiming to be "in progress"
from an interrupted session should be re-verified, not trusted.

---

## The real investigation: where it stood, and it is NOT invalidated

The crash happened while testing one hypothesis about a real, separately
well-evidenced bug. That bug and everything found about it up to the
crash is genuine, safely obtained, and still correct. Full detail is in
`tools/DIFFERENTIAL_PHASES.md` (append-only log, read it in full) and
`tools/TRANSLATION_DEBUGGING.md`; this is the short version.

**Current branch:** `codex/recursive-reduction-bridge`. Latest real commit
before this handoff: `c090228` ("Persist the graph-level root-cause
finding to the phases document"). Everything below that commit is safe,
tested, and durable.

**The bug:** the whole-program native executable
(`tools/build_fluid_c_shell.py`) builds and runs, but the compiled
`symbolic_fluid_advance` function calls its internal region for
`state.height = state.next_height + 0.0` (the authored commit,
line 67 of `SYMBOLIC_FLUID_DT_SOURCE`) BEFORE the region that computes
`next_height`'s real values (the per-cell loop, lines ~50-59). The commit
reads a not-yet-computed buffer and the whole simulator's state output
comes back zero. Confirmed with LLVM independently of Fortran -- both
backends inherit this from the same upstream capture, so the bug is not
backend-specific.

**Root cause, found and confirmed safely** (this part used
`graph.build_from_ast(tree, resolve_unresolved_parents=True)` over the
authored source directly -- no contract was passed here either, which in
hindsight was still a risk even though it happened not to fire; a future
repeat of even this check should use the contract): the ProcessGraph node
for `state.next_height` at the commit statement has, as its ONLY parent,
the `Name('state')` node. There is no edge to the loop's write
(`state.next_height[row, column] = height_next`) at all. A topological
scheduler downstream (`reduce_scheduled_shader_regions`, verified
correct by reading it) can only be as correct as the edges it is given;
this edge is simply missing at ingestion. This is not specific to
`height` -- `momentum_x`, `momentum_y`, and `tracer` follow the identical
authored pattern at the following three lines and almost certainly carry
the identical missing edge.

**What the fix needs to be:** when a record field is written through
element-wise subscript assignment inside a loop (`obj.field[i, j] = ...`)
and later read as a whole value (`obj.field`), the read must carry a
dependency edge on the write(s) that mutate it -- not merely on the
object it is fetched from. This should be general, in whatever code
handles `Attribute`/`GetAttr` resolution against a mutable record field,
not a special case for `height`.

**What was in progress, interrupted by the crash:** testing whether the
extraction contract (`program_extraction.yaml`, used by the whole-program
capture but not by the isolated per-function capture that works
correctly) is where this specific edge gets lost, versus it being missing
even in the contract-free general path. **Do not resume this exact test
as written.** If it's still worth answering, do it the safe way described
above (a modified copy of the real contract, not `None`), or -- more
directly and without touching extraction/native-decompilation machinery
at all -- read `graph_express2.py`'s and `topological_reducer.py`'s
handling of `Subscript` writes to an `Attribute` target, and check
whether ANY code path ever adds a dependency edge from such a write to a
later bare read of the same attribute. That answers the same question by
reading code rather than by running a wide, unguarded capture.

**Also still open, unrelated to the crash, recorded earlier this
session:**
- The `viscosity`/`tracer_diffusivity` formal-vs-gather defect
  (`symbol_provenance.py` still reports it) -- diffusion runs ~5000x too
  strong.
- The id-authority defect (non-deterministic/collision-prone SSA node
  ids) -- isolated on branch `id-identity-sweep`, not merged. Needed for
  reproducible builds and the eventual web IDE, not blocking the
  ordering bug above.

**Diagnostic tools built this session, all safe, read-only, no native
execution:** `tools/trace_fortran_alias.py` (follows a named buffer
through a whole-program Fortran call graph by position, not by trusting
hand-parsed text), `tools/symbol_provenance.py`, `tools/bisect_emission.py`,
`tools/differential_matrix.py`, `tools/translation_graph.py` and
`tools/build_tracer_site.py` (the spectral provenance tracer). All still
work as documented; none of them touch extraction contracts or native
decompilation.
