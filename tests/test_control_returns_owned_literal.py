"""A returned literal is the control function's own output, not a region's.

``z = 0; t = x + z; return t, z``: the planner keeps every region's
rematerialised copy of a canonical literal private, so no region publishes
``z``.  ``finish`` used to drop it from the Ret silently.  The call-frame
projection pass then found the callee's private ``Const`` and appended the
id to the region call's declared ``output_ids`` -- after the callee's Ret
was already fixed -- and LLVM emission refused:

    the call selects 6 aggregate position(s) but the callee produces 5
    output(s) ... declared=(6, 8, 10, 12, 14, 4)

The declared list is the planner's sorted outputs with the literal's id
appended last, which is the exact shape the vehicle body reported (144
declared against 143 produced, ending on the literal's id 349).

Now the control function resolves such an output as the provisional formal
that ``_materialize_control_constants`` turns into its own ``Const``, the
same way a folded ``flag = True; break`` is resolved on a break edge.
"""
from pathlib import Path
import sys
import warnings

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.tensors import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.native_law_kernels import batch_contract
from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm

ARGS = ("p0", "p1", "p2", "p3")

CASES = {
    # the literal is consumed by name inside the region and returned
    "consumed_by_name": (
        "def tick(p0, p1, p2, p3):\n"
        "    z = 0\n"
        "    t0 = p0 * p1\n"
        "    t1 = t0 + z\n"
        "    return t0, t1, z\n"
    ),
    # the literal is returned; a separate inline ``0`` is consumed. Measured:
    # occurrences do NOT pool -- ``z`` is one canonical id, the inline ``0``
    # another, and the region owns only the latter. So this is the
    # returned-only shape plus a private region literal of the same value.
    "consumed_inline": (
        "def tick(p0, p1, p2, p3):\n"
        "    z = 0\n"
        "    t0 = p0 * p1\n"
        "    t1 = t0 + 0\n"
        "    return t0, t1, z\n"
    ),
    # the literal is only returned; this shape was never broken and must
    # stay clean
    "returned_only": (
        "def tick(p0, p1, p2, p3):\n"
        "    z = 0\n"
        "    t0 = p0 * p1\n"
        "    t1 = t0 + p2\n"
        "    return t0, t1, z\n"
    ),
}


def _lower(source):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lowered = lower_ast_source_to_ssa(
            source, "tick",
            python_bindings={"AbstractTensor": AbstractTensor},
            tensor_ssa_reference=c_backend_repository_ssa_reference(),
            name="c_batched", runtime_closure_only=True,
            extraction_contract=batch_contract("tick", ARGS, 8),
        )
    return lowered[0]


def _instructions(function):
    return [
        instruction
        for block in function.blocks.values()
        for instruction in block.instrs
    ]


@pytest.mark.parametrize("case", sorted(CASES))
def test_region_call_declares_exactly_what_the_callee_returns(case):
    module = _lower(CASES[case])
    entry = module.functions["c_batched__tick"]
    calls = [
        instruction for instruction in _instructions(entry)
        if instruction.op in {"Call", "call"}
        and instruction.attributes.get("region_index") is not None
    ]
    assert len(calls) == 1
    callee = module.functions[str(calls[0].attributes["callee"])]
    callee_ret = next(
        instruction for instruction in _instructions(callee)
        if instruction.op in {"Ret", "ret"}
    )
    declared = tuple(calls[0].attributes["output_ids"])
    assert len(declared) == len(callee_ret.args), (case, declared)

    # the entry owns the literal: its own Const, returned by name
    named = dict(entry.metadata["named_outputs"])
    assert "z" in named, (case, named)
    literal = next(
        instruction for instruction in _instructions(entry)
        if instruction.res is not None
        and int(instruction.res.id) == int(named["z"])
    )
    assert str(literal.op) == "Const" and literal.attributes["value"] == 0
    assert int(named["z"]) not in declared
    entry_ret = next(
        instruction for instruction in _instructions(entry)
        if instruction.op in {"Ret", "ret"}
    )
    assert int(named["z"]) in {int(value.id) for value in entry_ret.args}
    assert int(named["z"]) not in {int(value.id) for value in entry.args}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        artifact = emit_ssa_function_to_llvm(module, "c_batched__tick")
    assert artifact.complete, [s.reason for s in artifact.shortfalls]
