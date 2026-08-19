"""Static invariants the SSA product can check about itself.

The diagnose tool's stages (shortfalls, well-formedness, aliasing,
observability, influence) all cleared while three real miscompilations went
through -- the frozen second carried value, the formal that is also a region
output, the id()-scale formal. Each of those is visible in the emitted
``IRModule`` itself, without executing anything, once you know what to look
for. This module is that knowledge as code.

Every check returns findings rather than raising, because a self-check that
stops at the first finding hides the second -- and reports what it does NOT
prove, in the tradition of ``tools/diagnose_translation.py``: a clean result
here is necessary, never sufficient.

One gap is recorded rather than papered over. The frozen-carried-value defect
is NOT statically decidable from the SSA alone: a formal consumed inside a
loop body is either a legitimate loop-invariant input (``dt`` in the fluid
step) or a carried value the planner dropped (``m`` in an Adam-shaped loop),
and the SSA carries no record of which the AUTHOR meant. The authored carried
set lives upstream, in the graph's ``loop_carried_bindings``, and dies before
reaching function metadata. ``suspicious_loop_invariant_formals`` therefore
reports CANDIDATES for the runtime frozen-reference comparison (see the
decision tree's silent-miscompilation signatures), and the honest fix is for
the lowering to stamp the carried intent into ``Function.metadata`` so this
becomes decidable here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator


# Real value ids are monotonic counters. An id at this scale is a memory
# address that leaked in -- ``fortran_c_shell`` already calls id()-carrying
# arguments "dead code by definition" -- and one in a SIGNATURE additionally
# displaces the positional correlation every ABI consumer relies on.
ID_SCALE_THRESHOLD = 10**9


@dataclass(frozen=True)
class Finding:
    """One violated invariant, located and explained."""

    check: str
    function: str
    detail: str

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"[{self.check}] {self.function}: {self.detail}"


def _functions(module: Any) -> Iterator[tuple[str, Any]]:
    yield from (getattr(module, "functions", {}) or {}).items()


def _instructions(function: Any) -> Iterator[Any]:
    for block in (getattr(function, "blocks", {}) or {}).values():
        yield from block.instrs


def check_formal_parity(module: Any) -> list[Finding]:
    """Every formal is either named or accounted for; none appears from nowhere.

    The signature IS the ABI. A function whose formals outnumber its
    ``parameter_names`` has grown a parameter no caller can name or fill --
    the visible symptom of the formal/region-output collision, where a value
    consumed before the region that produces it is quietly promoted to a
    formal so it is never unbound.

    Does not prove the named formals are correctly ordered or typed.
    """

    findings: list[Finding] = []
    for name, function in _functions(module):
        formals = [int(argument.id) for argument in getattr(function, "args", ())]
        metadata = getattr(function, "metadata", None) or {}
        named = {int(value) for _label, value in metadata.get("parameter_names") or ()}
        # Carried ports legitimately rename a parameter (the name follows the
        # phi); follow the accounting back so those do not read as unnamed.
        for carried in (metadata.get("carried_port_values") or {}).values():
            accounting = getattr(carried, "accounting", None) or {}
            if int(getattr(carried, "id", -1)) in named:
                source = accounting.get("source_value_id")
                if source is not None:
                    named.add(int(source))
        unnamed = [value for value in formals if value not in named]
        if named and unnamed:
            findings.append(Finding(
                "formal_parity", str(name),
                f"{len(formals)} formals but only {len(named)} named; "
                f"unnamed value ids {unnamed} -- no caller can know what to "
                "pass. If one is also in a region's output_ids, this is the "
                "formal/region-output collision.",
            ))
    return findings


def check_dead_storage_formals(module: Any) -> list[Finding]:
    """No formal is a frame-storage slot that the function never touches.

    The caller-provides-workspace convention (the LAPACK WORK-array move)
    only works DECLARED. A storage formal is legitimate ABI when it appears
    in ``Function.metadata["storage_formals"]`` with its dtype and shape --
    the shell leases the top of the chain from the heap, the materializer
    allocates it at entry. Undeclared storage accounting on a formal, or a
    declaration naming a value that is not a formal, is parity drift.

    Does not prove a declared formal is correctly threaded at call sites.
    """

    findings: list[Finding] = []
    for name, function in _functions(module):
        metadata = getattr(function, "metadata", None) or {}
        declared = {
            int(entry["value_id"])
            for entry in metadata.get("storage_formals") or ()
        }
        for argument in getattr(function, "args", ()):
            accounting = getattr(argument, "accounting", None) or {}
            marker = (
                accounting.get("linked_call_frame_storage")
                or accounting.get("returned_record_storage")
            )
            if marker and int(argument.id) not in declared:
                findings.append(Finding(
                    "dead_storage_formal", str(name),
                    f"formal {int(argument.id)} is frame storage for "
                    f"{marker!r} but is not declared in the function's "
                    "storage_formals metadata; an undeclared workspace is a "
                    "parameter no caller can name, size, or fill",
                ))
        for value_id in declared:
            if value_id not in {
                int(argument.id) for argument in getattr(function, "args", ())
            }:
                findings.append(Finding(
                    "dead_storage_formal", str(name),
                    f"storage_formals declares value {value_id}, which is not "
                    "among the function's formals; the declaration and the "
                    "signature have drifted apart",
                ))
    return findings


def check_id_scale(module: Any) -> list[Finding]:
    """No value id is a memory address.

    Catches the poisoned-allocator class at the product boundary: a formal,
    result or operand with an id()-scale id means some allocator's base was
    seeded from an object identity rather than the monotonic counter.

    Does not prove ids are dense or gap-free, only that none is absurd.
    """

    findings: list[Finding] = []
    for name, function in _functions(module):
        suspicious: set[int] = set()
        for argument in getattr(function, "args", ()):
            if int(argument.id) > ID_SCALE_THRESHOLD:
                suspicious.add(int(argument.id))
        for instruction in _instructions(function):
            for operand in instruction.args:
                if int(operand.id) > ID_SCALE_THRESHOLD:
                    suspicious.add(int(operand.id))
            if instruction.res is not None and int(instruction.res.id) > ID_SCALE_THRESHOLD:
                suspicious.add(int(instruction.res.id))
        if suspicious:
            findings.append(Finding(
                "id_scale", str(name),
                f"value ids at memory-address scale: {sorted(suspicious)[:4]} "
                "-- an allocator base was seeded from id() rather than the "
                "monotonic counter",
            ))
    return findings


def check_output_contract_agreement(module: Any) -> list[Finding]:
    """Every region's callers agree on what it publishes.

    The region calling convention is only half-located: a region function
    states no outputs of its own, so the ``output_ids`` tuple on each Call is
    the only record of the contract. Two callers disagreeing means the region
    would return a different aggregate depending on who asked.

    Does not prove the agreed contract is the RIGHT one.
    """

    contracts: dict[str, tuple[int, ...]] = {}
    findings: list[Finding] = []
    for name, function in _functions(module):
        for instruction in _instructions(function):
            if str(instruction.op) not in {"Call", "call"}:
                continue
            attributes = instruction.attributes or {}
            callee = str(attributes.get("callee") or "")
            declared = attributes.get("output_ids")
            if not callee or declared is None:
                continue
            outputs = tuple(int(each) for each in declared)
            existing = contracts.setdefault(callee, outputs)
            if existing != outputs:
                findings.append(Finding(
                    "output_contract", str(name),
                    f"call to {callee!r} projects {outputs}, but another call "
                    f"site projects {existing}",
                ))
    return findings


def suspicious_loop_invariant_formals(module: Any) -> list[Finding]:
    """Formals consumed inside a loop body, alongside at least one carried phi.

    CANDIDATES ONLY -- this is the recorded gap, not a verdict. Such a formal
    is either a legitimate loop-invariant input or a value the author carried
    and the planner dropped (the frozen-second-carried-value miscompilation),
    and the SSA does not record which was meant. Confirm with the runtime
    frozen-reference comparison from the decision tree: recompute with the
    candidate held constant, and a bit-exact match convicts it.

    Becomes decidable here the day the lowering stamps the authored carried
    set into ``Function.metadata``.
    """

    findings: list[Finding] = []
    for name, function in _functions(module):
        blocks = getattr(function, "blocks", {}) or {}
        header = blocks.get("loop_header")
        body = blocks.get("loop_body")
        if header is None or body is None:
            continue
        carried_phis = [
            instruction for instruction in header.instrs
            if str(instruction.op) in {"Phi", "phi"}
            and (instruction.attributes or {}).get("binding") == "loop_carried"
        ]
        if not carried_phis:
            continue
        formals = {int(argument.id) for argument in getattr(function, "args", ())}
        consumed_formals = sorted({
            int(operand.id)
            for instruction in body.instrs
            for operand in instruction.args
            if int(operand.id) in formals
        })
        if consumed_formals:
            findings.append(Finding(
                "loop_invariant_formal", str(name),
                f"formals {consumed_formals} are consumed inside a loop body "
                f"that carries {len(carried_phis)} phi(s). Legitimate for a "
                "loop-invariant input; the frozen-carried miscompilation for "
                "a value the author reassigns in the loop. Confirm with the "
                "frozen-reference comparison -- the SSA cannot say which.",
            ))
    return findings


def run_all(module: Any) -> list[Finding]:
    """Every decisive check; the candidate reporter is separate on purpose.

    ``suspicious_loop_invariant_formals`` is excluded because its findings are
    not violations -- mixing candidates into a list of convictions teaches
    callers to ignore the list.
    """

    return [
        *check_formal_parity(module),
        *check_dead_storage_formals(module),
        *check_id_scale(module),
        *check_output_contract_agreement(module),
    ]


__all__ = [
    "ID_SCALE_THRESHOLD",
    "Finding",
    "check_dead_storage_formals",
    "check_formal_parity",
    "check_id_scale",
    "check_output_contract_agreement",
    "run_all",
    "suspicious_loop_invariant_formals",
]
