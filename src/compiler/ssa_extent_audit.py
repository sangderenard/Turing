"""Do a merge's arms agree with the merge about how many elements there are?

A Phi says "the value is whichever arm we arrived on".  That is only true when
every arm denotes the same amount of storage as the result.  Nothing in the
backends enforces it: the C emitter carries a span as a bare ``double *`` and
compares ``buffer_type`` strings, so a pointer to ONE double and a pointer to
sixty-four are the same type to it, and a disagreement passes as an ordinary
address assignment.  The consumer then indexes whatever it was promised.

Both directions of the disagreement are real and they fail differently:

* the result keeps FEWER elements than an arm supplies -- a truncation.  This
  is the dangerous one, because nothing refuses it.  The emitter writes
  ``t = *((double *)(buffer))``, which is a legal read of element zero, and the
  other elements are simply gone.
* the result promises MORE than an arm supplies -- the case
  ``ssa_c_backend`` already refuses with "scalar incoming value needs an
  explicit broadcast".  That refusal is correct and this audit does not argue
  with it; it finds the same shape of fault everywhere else, including the
  places where matching C types let it slip past.

The authority for "how many" is deliberately NOT ``.shape`` alone.  These
arrays are sized at run time, so a rank-1 record field routinely carries an
empty static shape while its accounting states ``program_abi_rank: 1`` --
``_declared_span_rank`` is the same authority the Fortran backend declares
from, and it is what this asks.  Rank is therefore the primary signal and is
always available; a static element count is compared additionally, and only
when both sides actually have one.

This exists because a shape can be attached to an SSA identity after the fact.
``ssa_c_backend._publication_element_count`` already says so in as many words
-- "even if a stale call-site view shape has been attached to the same SSA
identity" -- and works around it for one operand.  An audit says how many
other identities are carrying a shape that disagrees with their own uses,
which is the difference between one bad accumulator and a class of them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.compiler import id_space
from src.compiler.ssa_llvm_backend import _declared_span_rank

# What a disagreement means for the program, as three named outcomes rather
# than a signed difference a reader has to interpret.
TRUNCATION = "truncation"
"""The merge keeps fewer elements than this arm supplies; the rest are lost."""

BROADCAST_WANTED = "broadcast-wanted"
"""The merge promises more than this arm supplies; a splat is missing."""

MISMATCH = "mismatch"
"""Both sides are plural and neither contains the other.  Not a broadcast in
either direction -- 8 elements into 64 is a bug, and saying so keeps it out of
the pile of things a broadcast would silently 'fix'."""


@dataclass(frozen=True)
class ExtentDisagreement:
    """One merge arm that does not denote what its merge denotes."""

    function: str
    block: str
    phi_id: int
    position: int
    arm_id: int
    kind: str
    phi_rank: int
    arm_rank: int
    phi_elements: int | None
    arm_elements: int | None

    def describe(self) -> str:
        """One line, with ids in the form the concordance prints them.

        ``id_space.describe`` rather than the raw integer: a composed id is
        nineteen digits and two of them differing is not something a reader
        can see.  ``minted#1000006`` is.
        """
        def extent(rank: int, elements: int | None) -> str:
            return f"rank {rank}" + (
                f", {elements} elements" if elements is not None
                else ", extent from accounting"
            )

        return (
            f"{self.kind}: {self.function}[{self.block}] "
            f"phi {id_space.describe(self.phi_id)} "
            f"({extent(self.phi_rank, self.phi_elements)}) "
            f"<- arm {self.position} {id_space.describe(self.arm_id)} "
            f"({extent(self.arm_rank, self.arm_elements)})"
        )


def _static_elements(value) -> int | None:
    """The element count when it is knowable statically, else None.

    Three cases, and conflating the last two is what made the first version of
    this audit cry wolf:

    * a non-empty static shape counts itself, whatever its rank -- a ``(1, 1,
      1)`` bool is ONE element, exactly as a ``()`` bool is.
    * rank 0 is one element.  An empty shape with no declared rank is a
      scalar, not an unknown; reporting it as unknown pushed the verdict onto
      the rank comparison, which then called a harmless ``(1, 1, 1) -> ()``
      merge of a loop condition a truncation.
    * rank above zero with no static shape is genuinely unknown, because these
      arrays are sized at run time and the extent lives in the artifact's
      extents vector.  None says so rather than inventing a number.
    """
    shape = tuple(getattr(value, "shape", ()) or ())
    if shape:
        return int(math.prod(int(axis) for axis in shape))
    if _declared_span_rank(value) == 0:
        return 1
    return None


def _classify(phi_rank, arm_rank, phi_elements, arm_elements) -> str | None:
    """Which outcome this pair is, or None when the arms agree.

    The COUNT decides wherever both counts are known, and rank is only the
    fallback for a dynamic extent.  That order matters: rank disagreement on
    its own is not a fault, because a loop condition shaped ``(1, 1, 1)``
    merging into a ``()`` condition loses nothing, and judging it by rank
    reported three faults in a program that has one.  What can never be
    harmless is a count that shrinks.

    Two values whose extents are both dynamic and whose ranks agree are
    treated as agreeing: nothing here knows otherwise, and a finding that
    cannot be acted on is worse than no finding.
    """
    if phi_elements is None or arm_elements is None:
        if phi_rank != arm_rank:
            return TRUNCATION if phi_rank < arm_rank else BROADCAST_WANTED
        return None
    if phi_elements == arm_elements:
        return None
    if arm_elements == 1:
        return BROADCAST_WANTED
    if phi_elements == 1:
        return TRUNCATION
    return MISMATCH


def audit_function(name: str, function) -> tuple[ExtentDisagreement, ...]:
    """Every merge arm in one function that disagrees with its merge."""
    found: list[ExtentDisagreement] = []
    for block_name, block in function.blocks.items():
        for instruction in block.instrs:
            if str(instruction.op) not in {"Phi", "phi"}:
                continue
            result = instruction.res
            if result is None:
                continue
            phi_rank = _declared_span_rank(result)
            phi_elements = _static_elements(result)
            for position, argument in enumerate(instruction.args or ()):
                arm_rank = _declared_span_rank(argument)
                arm_elements = _static_elements(argument)
                kind = _classify(phi_rank, arm_rank, phi_elements, arm_elements)
                if kind is None:
                    continue
                found.append(ExtentDisagreement(
                    function=name,
                    block=str(block_name),
                    phi_id=int(result.id),
                    position=position,
                    arm_id=int(argument.id),
                    kind=kind,
                    phi_rank=phi_rank,
                    arm_rank=arm_rank,
                    phi_elements=phi_elements,
                    arm_elements=arm_elements,
                ))
    return tuple(found)


def audit_module(module) -> tuple[ExtentDisagreement, ...]:
    """Every merge arm in the module that disagrees with its merge."""
    found: list[ExtentDisagreement] = []
    for name, function in module.functions.items():
        found.extend(audit_function(str(name), function))
    return tuple(found)


def report(disagreements) -> str:
    """The findings grouped by outcome, worst first.

    Truncation leads because it is the one nothing refuses: a program carrying
    it compiles, runs, and is wrong, where a missing broadcast at least stops
    the build.
    """
    disagreements = tuple(disagreements)
    if not disagreements:
        return "extent audit: every merge arm agrees with its merge"
    order = (TRUNCATION, MISMATCH, BROADCAST_WANTED)
    lines = [f"extent audit: {len(disagreements)} disagreeing merge arms"]
    for kind in order:
        rows = [item for item in disagreements if item.kind == kind]
        if not rows:
            continue
        lines.append(f"  {kind}: {len(rows)}")
        for item in rows:
            lines.append(f"    {item.describe()}")
    other = [item for item in disagreements if item.kind not in order]
    for item in other:
        lines.append(f"    {item.describe()}")
    return "\n".join(lines)


__all__ = [
    "BROADCAST_WANTED",
    "ExtentDisagreement",
    "MISMATCH",
    "TRUNCATION",
    "audit_function",
    "audit_module",
    "report",
]
