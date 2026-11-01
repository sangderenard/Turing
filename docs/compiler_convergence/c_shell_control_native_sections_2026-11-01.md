# Opt-in C control around native compute sections

LLVM and Fortran can now explicitly delegate orchestration to the existing C
materialization while retaining selected numerical functions in their requested
backend. This is AOT object/IR linking into a standalone executable, not JIT,
an in-process C executor, or a Python implementation of threading.

```python
from src.compiler.c_shell_sections import CShellControl
from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm

approval = CShellControl(compute_functions=(exact_compute_ssa_function_name,))
artifact = emit_ssa_function_to_llvm(
    module, exact_root_ssa_function_name, c_shell_control=approval)
executable = artifact.compile_standalone(build_directory, feeds_by_value_id)
```

For Fortran, use its existing module emitter:

```python
from src.compiler.ssa_fortran_backend import emit_module

approval = CShellControl(
    compute_functions=(exact_compute_ssa_function_name,),
    directory=section_build_directory)
artifact = emit_module(
    module, name='mixed_program', entrypoint=exact_root_ssa_function_name,
    c_shell_control=approval)
executable = artifact.compile_standalone(build_directory, feeds_by_value_id)
```

Without the option, neither emitter changes its backend selection. With the
option the return value is explicitly a **CModuleArtifact**, not a purported
complete LLVMFunctionArtifact or FortranModule. The native section names must
be exact reachable non-root SSA functions. In current ingestion, numerical
functions often have existing `__planned_region_...` bodies; use those actual
function-table identities. No source AST or spelling-based body recreation
occurs here. Boolean permission alone is refused because it does not identify
which portions must remain native.

## Division of responsibility

| Portion | Current implementation | Recorded boundary |
| --- | --- | --- |
| Thread creation/start/join, worker contexts/captures | Existing C thread table and generated trampolines | Existing source/SSA/C threading receipts |
| Exception records, optional checked operators, try/handler/finally/cleanup CFG | Existing SSA-to-C materialization and native error channel | Existing exception/control/operator receipts |
| Mutexes, Conditions, latches, fences, barriers and completion | Existing admitted C operations/compositions | Existing synchronization selection; same scope, progress and lifetime restrictions |
| Explicit numerical section | Requested LLVM or Fortran backend | `C_SHELL_SECTION_DEMAND`, then `C_SHELL_SECTION` |
| C call into the section | Existing linked-piece buffer ABI | C call emission derives from that exact section cell |
| LLVM section source/build | Existing LLVM emitter and C artifact linker | LLVM module emission -> section cell -> linked PIECE_FILE -> compile command |
| Fortran section source/build | Existing Fortran emitter and packed C ABI generator, then AOT object compilation | Exact source/command/object-hash revisions in `C_SHELL_SECTION_BUILD`; object hashes verified again by C linker |

The source dual IR and original SSA remain intact. Selection does not replace
functions with invented empty bodies, alter their call edges, or serialize
communicating workers. The current C-only control operations divide deployment
at existing function calls. Their control bodies are emitted by C while a
selected numerical callee is emitted by its native backend. Comments in the
C shim name that native backend. There is no new language-level control opcode
or duplicated synchronization runtime.

## Initial admitted section ABI

This first implementation accepts scalar numerical leaf functions containing
Const/Add/Sub/Mul/Neg/Fma/Abs/Min/Max, unconditional branches and return.
Existing type/target emission must also accept the section. Fortran sections
currently require float64 scalars and an available AOT Fortran compiler plus
explicit build directory. This subset avoids silently introducing Fortran
runtime-library, array-descriptor or exceptional-call dependencies.

Arrays, aggregate/reference layouts, arbitrary calls and other operations in
a selected section are refused. The C orchestration retains its established
supported vocabulary; this option does not make unsupported C operations work.
A compute call inside protected Python try still needs the existing explicit
failure-return ingestion contract. The verified combination calls a declared
float-returning compute function before the protected division, inside the
same worker loop. The division's error is caught, and finally performs the
barrier rendezvous. It does not claim caught exceptions originating inside a
native compute section.

Fortran module emission through this option compiles its selected objects at
selection time. Default Fortran source emission still requires no compiler.
The temporary verification toolchain is not a new repository dependency.

## Next backend refinements

These are proposed boundaries, not implemented capabilities:

| Refinement | Required preservation |
| --- | --- |
| Split within an existing SSA function | Concorded subgraph outlining, exact live-ins/live-outs, memory effects, exceptional and cleanup exits; never AST reconstruction |
| Array/native aggregate sections | Published layout, extent, alias and ownership transport; respect C row-major versus Fortran column-major ABI conversion |
| Native compute failures inside try | Explicit failure-return/exceptional continuation ABI end to end, including worker packets and cleanup |
| LLVM multi-lane/loop dispatch | Use existing Deploy/Join permissions and scheduling contracts; SIMD/pool choice remains target-specific |
| Fortran loop dispatch/coarray implementations | Explicit mapping of image/team/context, scope, progress, storage ownership and join semantics; coarray synchronization is not an automatic substitute for a host mutex or local barrier |
| Move C-supported control into a native backend | Select the equivalent native operation/composition per capability; preserve the original concordance identity and error/lifetime behavior |

The [backend operation reference](backend_threading_operation_reference_2026-11-01.md)
remains the common porting table. A C-wrapper capability is distinct from native
LLVM or Fortran operator support. C's supported synchronization scope subset
and all existing refusal policies continue to apply.

## Native execution and observed thread counts

The current standalone source test uses a worker loop with 1000 authored
barrier phases, protected integer division, handler and finally, and a native
scalar compute callee outside the protected call boundary. Both successful and
zero divisors produce the expected outputs with either LLVM or Fortran compute,
under ASAP and ALAP. On Linux the test observes the child process's `/proc`
task directory and requires exactly two native threads at peak: main plus its
one authored worker. This is not a Python thread runner or a generated-source
substring check. Observation results are saved beside each executable as
`thread-observation.json`.

`tests/test_source_native_thread_counts.py` also verifies two authored workers
plus main (three OS tasks), and thread-free source selected for passive pool
execution. The latter uses the general C emitter's `deployment='auto'` work
contract and `deployment_cores=2`: one parked pool worker plus the participating
caller. The same source under `serial` observes one OS task. Both ASAP and ALAP
are exercised, and a shared-read loop is rejected for this automatic proof and
executes serially. Native Fortran/LLVM lane dispatch still remains distinct
from this C-host deployment capability.
