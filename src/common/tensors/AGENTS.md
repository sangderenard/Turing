# Tensor Backends

## Quick Setup

Refer to `../ENV_SETUP_OPTIONS.md` for setup instructions.

This directory hosts the implementations of tensor operations for different numerical libraries.

### Backend Policy

- The **NumPy backend** is the default and canonical implementation.
- Backend resolution loops must check for `"numpy"` before `"torch"`, ensuring NumPy is always preferred when available.
- **Do not import, install, or rely on PyTorch (`torch`)**. The dependency is too heavy for this project.
- If an operation is missing, implement it in the NumPy backend rather than reaching for PyTorch or other large frameworks.

**Important:** Each backend must implement `_apply_operator` from `AbstractTensor`. This single method handles all arithmetic primitives. Avoid creating additional bespoke operator helpers – Python's magic methods already route standard arithmetic through `_apply_operator`.

Follow the repository coding standards and run targeted `pytest` tests after modifications.

## Development Ethos

`AbstractTensor` mirrors a subset of the PyTorch API while accommodating idioms
from NumPy and plain Python lists.  Operators are overloaded to hide backend
specifics, letting contributors work with whichever library they know best.
When behaviours diverge between libraries, **NumPy is the authoritative
reference**.

Implementation priority follows this order:
1. Get the abstract interface fully specified.
2. Maintain feature parity across **NumPy** and pure Python backends.
3. Address backend-specific gaps without introducing heavy dependencies.
4. Expand the C backend only after the above are complete unless a simple stub
   can be filled quickly.

---

## Agents: you will not out-design what is already here

Read this before touching anything.

### You will not do a better job than the systems already here

This repository holds complex, finished systems: a whole-program source
compiler with one sanctioned entry, a managed-dt runtime with its own
save/restore, an extended-precision tensor type, symbolic laws that are the
numerical truth. Each was built deliberately, over a long time, against
measured failures. An agent arriving cold cannot see those measurements and
cannot out-design what they produced.

So:

- **Never "find a better way."** If the task names a mechanism, use that
  mechanism as it is spelled. If the task can be done with an existing
  system, use the existing system. Do not write a stand-in, a wrapper that
  duplicates it, a "simpler" version, a convention of your own, or a
  tolerance that papers over a result you did not understand.
- **You were not given axes of freedom.** A closed instruction is a closed
  instruction. When you notice yourself choosing between designs, stop: the
  choice is not yours, and the correct answer is usually "the one that is
  already here."
- **Do not add your own machinery next to a system to avoid learning it.**
  Learning it is the job. Writing around it is the failure.
- **When something seems wrong, the system is right until proven otherwise.**
  Read more before editing. Compare against the more developed lane (LLVM and
  Fortran are more developed than C). Measure before claiming.
- **When you are unsure, stop and ask one precise question.** Do not run
  probes, tests or edits while the question is open. Do not restate the
  instruction back in your own words as if it were a plan.
- **"Keep it what it was" means revert to the working state**, not layer
  another change on top.

### Concretely

- The dt system (`turing/src/common/dt_system`) is used AS IS. Engines
  register their columns as parameters; the dt system saves and restores
  them. No runner, table, snapshot, or rebind convention of your own.
- The source compiler's public entry is
  `src.compiler.fortran_c_shell.lower_ast_source_to_ssa`. A global-scope
  program becomes compilable by wrapping it in one function that takes the
  columns; you do not pick an inner function and call it "the entry".
- `AbstractTensor` and `Precision` are the numerical substrate. Precision
  enters at the AbstractTensor stage by promoting operands
  (`Precision.of`); you do not reimplement it in SymPy or validate around it.
- SymPy laws are the truth. Identities that remove cancellation are
  welcome; anything that changes the physics is not.

If you cannot do the task with what is here, say so. Do not ship a
substitute.
