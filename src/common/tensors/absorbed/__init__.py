"""Machine-translated code AbstractTensor has taken in but not yet vouched for.

Everything in this package arrived by **auto-port**: foreign source (numpy,
torch, plain Python) compiled through this tree's own pipeline
(``lower_ast_source_to_ssa`` -> SSA -> ``materialize_ir_module`` with
``tensor_vocabulary=True``) and re-emitted as AbstractTensor Python.  Nothing
here was written by hand, and that is the point: a hand-port is a claim about
what the source meant, while an auto-port is a *translation the compiler is
accountable for*, reproducible by re-running it.

Why a separate space rather than the main tree
----------------------------------------------
Absorbed code is real and usable, but it has not had the review that
``abstraction.py`` or ``abstract_convolution/`` code has had.  Mixing the two
would quietly launder machine output into reviewed API.  Keeping it here means
a reader can always tell which they are looking at, and means this package can
accumulate a great deal of translated material without that accumulation
costing the reviewed tree anything.

The contract every module here meets
------------------------------------
1. **Provenance is recorded, not remembered.** Each module carries an
   :class:`~.provenance.Absorption` naming the repository, file and symbol it
   came from, and the entry point it was compiled through.  A translation
   whose origin is not stated is not absorbable.
2. **It executes and it agrees with its source.** Absorption requires a test
   that runs the translated function and compares it against the original
   implementation on real input.  "It compiled" is not evidence -- this tree
   has repeatedly found silent-success paths where every stage reported ok and
   the program computed nothing.
3. **The emitted source is kept verbatim.** Modules here are not tidied,
   renamed, or refactored after generation.  Editing translated output by hand
   turns it into unreviewed hand-written code wearing a machine's provenance,
   which is worse than either.  Improvements belong in the compiler, and the
   module is regenerated.
4. **Widened signatures are expected.** A slice or an unresolved bound becomes
   a real parameter of the translated function, so its signature can be wider
   than the authored one.  That is the program parameterised rather than
   damaged, and the authored parameter list stays recoverable from the SSA
   metadata.

Promotion out of here
---------------------
A module graduates into the reviewed tree when someone has actually read it
and it has earned a place in the catalogue -- not automatically, and not
merely because its tests pass.  Until then it is importable from here and
honest about what it is.
"""
from __future__ import annotations

from .provenance import Absorption

__all__ = ["Absorption"]
