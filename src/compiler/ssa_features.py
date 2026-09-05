"""Repository SSA algorithms linked when a numerical program calls them."""

from __future__ import annotations

from typing import Mapping

from ..common.tensors.accelerator_backends.llvm_repository_ssa import (
    import_llvm_to_repository_ssa,
)
from ..transmogrifier.ssa import Function
from ..transmogrifier.ssa_registry import Handler


XOROSHIRO128SS_FILL = "xoroshiro128ss_fill_double"

RANDOM_SSA_MODULE = r"""
define void @xoroshiro128ss_fill_double(
    ptr %out, i32 %n, i64 %seed0, i64 %seed1) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %next.i, %body ]
  %s0 = phi i64 [ %seed0, %entry ], [ %next.s0, %body ]
  %s1 = phi i64 [ %seed1, %entry ], [ %next.s1, %body ]
  %continue = icmp slt i32 %i, %n
  br i1 %continue, label %body, label %exit

body:
  %star5 = mul i64 %s0, 5
  %result.left = shl i64 %star5, 7
  %result.right = lshr i64 %star5, 57
  %result.rot = or i64 %result.left, %result.right
  %result = mul i64 %result.rot, 9
  %mantissa = lshr i64 %result, 11
  %mantissa.float = uitofp i64 %mantissa to double
  %sample = fdiv double %mantissa.float, 9.007199254740992e+15
  %index = zext i32 %i to i64
  %destination = getelementptr inbounds double, ptr %out, i64 %index
  store double %sample, ptr %destination, align 8

  %mixed = xor i64 %s1, %s0
  %s0.left = shl i64 %s0, 24
  %s0.right = lshr i64 %s0, 40
  %s0.rot = or i64 %s0.left, %s0.right
  %mixed.left16 = shl i64 %mixed, 16
  %s0.with.mixed = xor i64 %s0.rot, %mixed
  %next.s0 = xor i64 %s0.with.mixed, %mixed.left16
  %mixed.left37 = shl i64 %mixed, 37
  %mixed.right27 = lshr i64 %mixed, 27
  %next.s1 = or i64 %mixed.left37, %mixed.right27
  %next.i = add i32 %i, 1
  br label %loop

exit:
  ret void
}
"""

_FEATURE_MODULES = {XOROSHIRO128SS_FILL: RANDOM_SSA_MODULE}


def link_required_ssa_features(
    functions: Mapping[str, Function],
) -> dict[str, Function]:
    """Return ``functions`` plus definitions for referenced SSA features."""

    linked = dict(functions)
    callees = {
        str(instruction.attributes.get("callee"))
        for function in linked.values()
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op == Handler.Call.value
        and instruction.attributes.get("callee") is not None
    }
    for callee in sorted(callees & _FEATURE_MODULES.keys()):
        imported = import_llvm_to_repository_ssa(
            _FEATURE_MODULES[callee],
            include_declarations=False,
        )
        if not imported.complete:
            raise ValueError(imported.shortfall_report())
        for name, function in imported.module.functions.items():
            linked.setdefault(name, function)
    return linked


__all__ = [
    "RANDOM_SSA_MODULE",
    "XOROSHIRO128SS_FILL",
    "link_required_ssa_features",
]