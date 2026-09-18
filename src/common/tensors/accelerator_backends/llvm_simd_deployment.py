"""Scale-1 SSA deployment frames lowered to host-native LLVM SIMD.

This is intentionally a small first consumer: a legal reduction frame becomes
one native vector cohort plus a scalar tail.  The ordinary scalar loop remains
both the fallback and the differential reference.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Iterable

from ....compiler.deployment_frame import DeploymentJoinMode
from ....compiler.llvm_optimizing_pipeline import (
    OptimizingJITProgram,
    host_preferred_vector_width,
    tuned_host_profile,
)
from ....transmogrifier.ssa import SSADeploymentRegion


def host_simd_width_bits() -> int:
    """Return a safe floating-point vector width for this exact host."""
    preferred = host_preferred_vector_width()
    if preferred is not None:
        return int(preferred)
    from llvmlite import binding as llvm

    features = llvm.get_host_cpu_features()
    if bool(features.get("avx512f", False)):
        return 512
    # Floating-point AVX does not require AVX2, which concerns integer lanes.
    if bool(features.get("avx", False)):
        return 256
    if bool(features.get("sse2", False)):
        return 128
    return 64


def host_simd_lane_width(element_bits: int = 64) -> int:
    return max(1, host_simd_width_bits() // int(element_bits))


def _validate(region: SSADeploymentRegion) -> None:
    if region.scale != 1:
        raise ValueError("LLVM SIMD currently implements deployment scale 1")
    if region.join.mode is not DeploymentJoinMode.REDUCE:
        raise ValueError("LLVM SIMD requires a reduction join")
    if region.join.reduction_operator != "Add":
        raise ValueError("first LLVM SIMD reduction supports existing Add only")
    if not region.join.allow_reassociation:
        raise ValueError("floating Add reduction requires explicit reassociation")
    if region.carried_aliases:
        raise ValueError("SIMD deployment cannot carry loop aliases")


def _reduction_ir(lanes: int) -> str:
    if lanes < 1:
        raise ValueError("lane width must be positive")
    if lanes == 1:
        return """define double @deploy_reduce_add(ptr %data, i64 %n) {
entry:
  %empty = icmp eq i64 %n, 0
  br i1 %empty, label %exit.empty, label %body
body:
  %i = phi i64 [0, %entry], [%next.i, %body]
  %sum = phi double [0.0, %entry], [%next, %body]
  %ptr = getelementptr double, ptr %data, i64 %i
  %item = load double, ptr %ptr, align 8
  %next = fadd double %sum, %item
  %next.i = add i64 %i, 1
  %more = icmp ult i64 %next.i, %n
  br i1 %more, label %body, label %exit
exit:
  ret double %next
exit.empty:
  ret double 0.0
}
"""
    vector_setup = ""
    vector_phi = ""
    vector_body = ""
    vector_join = "  %vector.total = fadd fast double 0.0, 0.0\n"
    if lanes > 1:
        zero = "<" + ", ".join("double 0.0" for _ in range(lanes)) + ">"
        vector_setup = f"  %vector.limit = sub i64 %n, {lanes - 1}\n"
        vector_phi = f"  %vacc = phi <{lanes} x double> [{zero}, %entry], [%vnext, %vector.body]\n"
        vector_body = f"""  %vptr = getelementptr double, ptr %data, i64 %vi
  %chunk = load <{lanes} x double>, ptr %vptr, align 8
  %vnext = fadd fast <{lanes} x double> %vacc, %chunk
  %vnext.i = add i64 %vi, {lanes}
  %vmore = icmp ult i64 %vnext.i, %vector.limit
  br i1 %vmore, label %vector.body, label %vector.join
"""
        parts = [f"  %lane{index} = extractelement <{lanes} x double> %vnext, i32 {index}" for index in range(lanes)]
        total = "%lane0"
        for index in range(1, lanes):
            name = f"%partial{index}"
            parts.append(f"  {name} = fadd fast double {total}, %lane{index}")
            total = name
        vector_join = "\n".join(parts) + f"\n  %vector.total = fadd fast double {total}, 0.0\n"
    vector_condition = (
        f"%has.vector = icmp uge i64 %n, {lanes}" if lanes > 1
        else "%has.vector = icmp eq i1 false, true"
    )
    vector_index = "%vnext.i" if lanes > 1 else "0"
    vector_value = "%vector.total" if lanes > 1 else "0.0"
    return f"""define double @deploy_reduce_add(ptr %data, i64 %n) {{
entry:
{vector_setup}  {vector_condition}
  br i1 %has.vector, label %vector.body, label %scalar.preheader

vector.body:
  %vi = phi i64 [0, %entry], [%vnext.i, %vector.body]
{vector_phi}{vector_body}
vector.join:
{vector_join}  br label %scalar.preheader

scalar.preheader:
  %si0 = phi i64 [0, %entry], [{vector_index}, %vector.join]
  %sum0 = phi double [0.0, %entry], [{vector_value}, %vector.join]
  %done0 = icmp uge i64 %si0, %n
  br i1 %done0, label %exit, label %scalar.body

scalar.body:
  %si = phi i64 [%si0, %scalar.preheader], [%snext.i, %scalar.body]
  %sum = phi double [%sum0, %scalar.preheader], [%snext, %scalar.body]
  %sptr = getelementptr double, ptr %data, i64 %si
  %item = load double, ptr %sptr, align 8
  %snext = fadd fast double %sum, %item
  %snext.i = add i64 %si, 1
  %more = icmp ult i64 %snext.i, %n
  br i1 %more, label %scalar.body, label %exit

exit:
  %result = phi double [%sum0, %scalar.preheader], [%snext, %scalar.body]
  ret double %result
}}
"""


@dataclass
class LLVMScale1Reduction:
    region: SSADeploymentRegion
    lane_width: int
    vector_bits: int
    llvm_ir: str
    program: OptimizingJITProgram

    @property
    def assembly(self) -> str:
        return self.program.result.assembly

    @property
    def vectorized(self) -> bool:
        return self.lane_width > 1 and self.program.result.vectorized

    def execute(self, values: Iterable[float]) -> float:
        materialized = tuple(float(value) for value in values)
        storage = (ctypes.c_double * max(1, len(materialized)))(*materialized)
        function = ctypes.CFUNCTYPE(
            ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.c_uint64
        )(self.program.address("deploy_reduce_add"))
        return float(function(storage, len(materialized)))

    @staticmethod
    def execute_serial(values: Iterable[float]) -> float:
        total = 0.0
        for value in values:
            total += float(value)
        return total


def compile_scale1_reduction(
    region: SSADeploymentRegion, *, force_vector_bits: int | None = None
) -> LLVMScale1Reduction:
    """Compile one legal deployment frame, retaining a scalar tail/fallback."""
    _validate(region)
    width = host_simd_width_bits() if force_vector_bits is None else int(force_vector_bits)
    if width < 64 or width % 64:
        raise ValueError("float64 vector width must be a positive multiple of 64")
    lanes = width // 64
    llvm_ir = _reduction_ir(lanes)
    profile = tuned_host_profile(
        fast_math=True,
        prefer_vector_width=(width if lanes > 1 else None),
    )
    return LLVMScale1Reduction(
        region=region,
        lane_width=lanes,
        vector_bits=width,
        llvm_ir=llvm_ir,
        program=OptimizingJITProgram(llvm_ir, profile=profile),
    )


__all__ = [
    "LLVMScale1Reduction",
    "compile_scale1_reduction",
    "host_simd_lane_width",
    "host_simd_width_bits",
]
