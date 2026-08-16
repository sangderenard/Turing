"""Explicit correspondences between C tensor kernels and LLVM SSA.

This module is intentionally a translation table, not another tensor runtime.
The C functions in ``c_backend/ctensor_ops.c`` remain the authored algorithmic
source.  Their LLVM definitions below are handwritten, inspectable SSA
equivalents that can later be imported into Turing's repository SSA and
legalized through BitOps.

The first translated portion currently includes:

* scalar binary dispatch and its elementwise loop;
* scalar unary dispatch and its elementwise loop;
* fill;
* matrix multiplication;
* conditional and Boolean-mask kernels;
* C backend value casts;
* flat sum and arithmetic-sequence construction.

Tensor layout kernels, reductions, scans, and indexed mutation will join the
same registry incrementally.  ProcessGraph rewriting and BitOps legalization
are deliberately outside this module.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import re
import struct
from functools import lru_cache
from typing import Any, Iterable, Mapping


_C_BACKEND_DIR = Path(__file__).with_name("c_backend")
_C_SOURCE_PATH = _C_BACKEND_DIR / "ctensor_ops.c"
_C_HEADER_PATH = _C_BACKEND_DIR / "ctensor_ops.h"
_C_BACKEND_PATH = Path(__file__).with_name("c_backend.py")

# ``slice`` is emitted by the numerical precompiler after Python indexing has
# already been normalized.  It is intentionally not a public AbstractTensor
# method name, but it still needs the same real C -> LLVM algorithm registry.
PRECOMPILE_INTERNAL_OPERATORS = frozenset({"slice", "index_set"})

# Explicit output-buffer positions in the authored low-level SSA signatures.
# Scalar-returning helpers (binary_value, sum_double, count_true_double) carry
# a repository-SSA ``return_value`` instead and therefore do not appear here.
C_SSA_OUTPUT_ARGUMENTS = {
    "fill_double": 0,
    "binary_double": 2,
    "binary_scalar_double": 2,
    "unary_double": 1,
    "matmul_double": 2,
    "where_double": 3,
    "broadcast_double": 1,
    "reduce_dim_double": 1,
    "transpose_double": 1,
    "cumsum_dim_double": 1,
    "stack_double": 5,
    "cat_double": 6,
    "pad_double_nd": 1,
    "slice_copy_double": 1,
    "index_select_double": 1,
    "index_assign_double": 0,
    "index_set_double": 1,
    "unfold2d_double": 1,
    "fold2d_double": 1,
    "sign_double": 1,
    "mask_select_double": 2,
    "increment_mask_double": 0,
    "cast_double_to_int_values": 1,
    "cast_double_to_float_values": 1,
    "cast_double_to_double_values": 1,
    "cast_double_to_bool_values": 1,
    "create_arange": 3,
}

# LLVM opaque pointers erase pointee types.  Repository SSA and its Fortran
# ABI still need to distinguish integer metadata vectors from double tensor
# buffers, so retain that finite part of the authored C signatures here.
C_SSA_I32_POINTER_ARGUMENTS = {
    "pad_double_nd": (2, 3, 4),
    "slice_copy_double": (2,),
    "index_select_double": (2, 5),
    "index_assign_double": (1, 3, 4),
    "index_set_double": (2, 4, 5),
    "cumsum_dim_double": (2,),
    "reduce_dim_double": (2,),
    "transpose_double": (2, 3),
    "broadcast_double": (2, 4),
    "stack_double": (2,),
    "cat_double": (1, 3),
}


# These values are consumed by the handwritten switch instructions below.
# validate_c_opcode_alignment() compares them with the authoritative C enum so
# an inserted C opcode cannot silently change the meaning of the LLVM module.
C_TENSOR_OPCODE_ORDER = (
    "ADD",
    "SUB",
    "MUL",
    "DIV",
    "POW",
    "MOD",
    "FLOORDIV",
    "SQRT",
    "EXP",
    "LOG",
    "NEG",
    "ABS",
    "ROUND",
    "TRUNC",
    "FLOOR",
    "CEIL",
    "ISFINITE",
    "ISNAN",
    "ISINF",
    "LOGICAL_NOT",
    "LT",
    "LE",
    "GT",
    "GE",
    "EQ",
    "NE",
    "MAXIMUM",
    "MINIMUM",
    "TANH",
    "SIN",
    "COS",
    "TAN",
    "ASIN",
    "ACOS",
    "ATAN",
    "SINH",
    "COSH",
    "ASINH",
    "ACOSH",
    "ATANH",
    "SIGN",
    "INVERT",
    "BITAND",
    "BITOR",
    "BITXOR",
    "SHL",
    "SHR",
    "LOGICAL_AND",
    "LOGICAL_OR",
)

C_SSA_EXTERNAL_PRIMITIVES = frozenset({
    "acos", "acosh", "asin", "asinh", "atan", "atanh", "cos", "cosh",
    "exp", "llvm.ceil.f64", "llvm.fabs.f64", "llvm.fcmp.ord",
    "llvm.fcmp.uno", "llvm.floor.f64", "llvm.round.f64", "llvm.sqrt.f64",
    "llvm.trunc.f64", "llvm.memcpy.p0.p0.i64", "llvm.memset.p0.i64",
    "log", "pow", "sin", "sinh", "tan", "tanh",
    "llvm.maxnum.f64", "llvm.minnum.f64", "llvm.pow.f64",
})


# Classification covers names discovered from ctensor_ops.c.  Validation below
# requires exact set equality, so adding or removing a C function forces this
# inventory to be reviewed.  These are classifications, not replacement
# operation names.
_C_FUNCTION_ROLES = {
    "mean_dim_callback": "private_callback",
    "topk_dim_callback": "private_callback",
    "log_softmax_dim_callback": "private_callback",
    "fill_double": "tensor_kernel",
    "binary_value": "scalar_helper",
    "binary_double": "tensor_kernel",
    "binary_scalar_double": "tensor_kernel",
    "matmul_double": "tensor_kernel",
    "unary_double": "tensor_kernel",
    "batched_matmul_indexed_double": "tensor_kernel",
    "is_unary_op": "scalar_helper",
    "ctensor_execute_primitive_program_slots": "execution_driver",
    "ctensor_execute_primitive_program": "execution_driver",
    "reduce_dim_double": "tensor_kernel",
    "transpose_double": "tensor_kernel",
    "where_double": "tensor_kernel",
    "broadcast_double": "tensor_kernel",
    "cumsum_dim_double": "tensor_kernel",
    "argreduce_dim_double": "tensor_kernel",
    "repeat_interleave_double": "tensor_kernel",
    "tile_double": "tensor_kernel",
    "index_select_double": "tensor_kernel",
    "slice_copy_double": "tensor_kernel",
    "index_assign_double": "tensor_kernel",
    "index_set_double": "tensor_kernel",
    "unfold2d_double": "tensor_kernel",
    "fold2d_double": "tensor_kernel",
    "sign_double": "tensor_kernel",
    "count_true_double": "tensor_kernel",
    "mask_select_double": "tensor_kernel",
    "increment_mask_double": "tensor_kernel",
    "cast_double_to_int_values": "tensor_kernel",
    "cast_double_to_float_values": "tensor_kernel",
    "cast_double_to_double_values": "tensor_kernel",
    "cast_double_to_bool_values": "tensor_kernel",
    "log_softmax_1d": "tensor_kernel",
    "log_softmax_callback": "private_callback",
    "pad_double_nd": "tensor_kernel",
    "mean_dim": "tensor_kernel",
    "gather_pairs_2d": "tensor_kernel",
    "sum_double": "tensor_kernel",
    "create_arange": "tensor_kernel",
    "topk_double": "tensor_kernel",
    "topk_double_dim": "tensor_kernel",
    "log_softmax_dim": "tensor_kernel",
    "for_each_cell_along_dim": "traversal_helper",
    "stack_double": "tensor_kernel",
    "cat_double": "tensor_kernel",
}


# This is LLVM SSA written as LLVM SSA, not generated from Python expression
# nodes.  Symbols intentionally mirror the corresponding C function names.
LLVM_SSA_MODULE = r"""
; Turing C tensor computational core, manually represented in LLVM SSA.
source_filename = "ctensor_ops.manual-ssa"

declare double @llvm.sqrt.f64(double)
declare double @llvm.fabs.f64(double)
declare double @llvm.round.f64(double)
declare double @llvm.trunc.f64(double)
declare double @llvm.floor.f64(double)
declare double @llvm.ceil.f64(double)
declare double @llvm.maxnum.f64(double, double)
declare double @llvm.minnum.f64(double, double)
declare double @llvm.pow.f64(double, double)
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)
declare double @pow(double, double)
declare double @exp(double)
declare double @log(double)
declare double @tanh(double)
declare double @sin(double)
declare double @cos(double)
declare double @tan(double)
declare double @asin(double)
declare double @acos(double)
declare double @atan(double)
declare double @sinh(double)
declare double @cosh(double)
declare double @asinh(double)
declare double @acosh(double)
declare double @atanh(double)

define void @fill_double(ptr %out, double %value, i32 %n) {
entry:
  br label %loop.header

loop.header:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop.body ]
  %continue = icmp slt i32 %i, %n
  br i1 %continue, label %loop.body, label %exit

loop.body:
  %i64 = sext i32 %i to i64
  %dst = getelementptr inbounds double, ptr %out, i64 %i64
  store double %value, ptr %dst, align 8
  %i.next = add nsw i32 %i, 1
  br label %loop.header

exit:
  ret void
}

define void @pad_double_nd(
    ptr %input, ptr %output, ptr %shape, ptr %new_shape,
    ptr %left_pad, i32 %dims, double %value) {
entry:
  br label %output.product.header

output.product.header:
  %output.dimension = phi i32 [ 0, %entry ], [ %output.dimension.next, %output.product.body ]
  %total.out = phi i32 [ 1, %entry ], [ %total.out.next, %output.product.body ]
  %output.product.continue = icmp slt i32 %output.dimension, %dims
  br i1 %output.product.continue, label %output.product.body, label %fill.header

output.product.body:
  %output.dimension64 = sext i32 %output.dimension to i64
  %output.extent.ptr = getelementptr inbounds i32, ptr %new_shape, i64 %output.dimension64
  %output.extent = load i32, ptr %output.extent.ptr, align 4
  %total.out.next = mul nsw i32 %total.out, %output.extent
  %output.dimension.next = add nsw i32 %output.dimension, 1
  br label %output.product.header

fill.header:
  %fill.index = phi i32 [ 0, %output.product.header ], [ %fill.index.next, %fill.body ]
  %fill.continue = icmp slt i32 %fill.index, %total.out
  br i1 %fill.continue, label %fill.body, label %input.product.header

fill.body:
  %fill.index64 = sext i32 %fill.index to i64
  %fill.ptr = getelementptr inbounds double, ptr %output, i64 %fill.index64
  store double %value, ptr %fill.ptr, align 8
  %fill.index.next = add nsw i32 %fill.index, 1
  br label %fill.header

input.product.header:
  %input.dimension = phi i32 [ 0, %fill.header ], [ %input.dimension.next, %input.product.body ]
  %input.size = phi i32 [ 1, %fill.header ], [ %input.size.next, %input.product.body ]
  %input.product.continue = icmp slt i32 %input.dimension, %dims
  br i1 %input.product.continue, label %input.product.body, label %copy.header

input.product.body:
  %input.dimension64 = sext i32 %input.dimension to i64
  %input.extent.ptr = getelementptr inbounds i32, ptr %shape, i64 %input.dimension64
  %input.extent = load i32, ptr %input.extent.ptr, align 4
  %input.size.next = mul nsw i32 %input.size, %input.extent
  %input.dimension.next = add nsw i32 %input.dimension, 1
  br label %input.product.header

copy.header:
  %input.index = phi i32 [ 0, %input.product.header ], [ %input.index.next, %copy.store ]
  %copy.continue = icmp slt i32 %input.index, %input.size
  br i1 %copy.continue, label %coordinate.entry, label %exit

coordinate.entry:
  %last.dimension = sub nsw i32 %dims, 1
  br label %coordinate.header

coordinate.header:
  %dimension = phi i32 [ %last.dimension, %coordinate.entry ], [ %dimension.next, %coordinate.body ]
  %remaining = phi i32 [ %input.index, %coordinate.entry ], [ %remaining.next, %coordinate.body ]
  %output.index = phi i32 [ 0, %coordinate.entry ], [ %output.index.next, %coordinate.body ]
  %output.stride = phi i32 [ 1, %coordinate.entry ], [ %output.stride.next, %coordinate.body ]
  %coordinate.continue = icmp sge i32 %dimension, 0
  br i1 %coordinate.continue, label %coordinate.body, label %copy.store

coordinate.body:
  %dimension64 = sext i32 %dimension to i64
  %shape.ptr = getelementptr inbounds i32, ptr %shape, i64 %dimension64
  %shape.value = load i32, ptr %shape.ptr, align 4
  %coordinate = srem i32 %remaining, %shape.value
  %remaining.next = sdiv i32 %remaining, %shape.value
  %left.ptr = getelementptr inbounds i32, ptr %left_pad, i64 %dimension64
  %left.value = load i32, ptr %left.ptr, align 4
  %padded.coordinate = add nsw i32 %coordinate, %left.value
  %coordinate.offset = mul nsw i32 %padded.coordinate, %output.stride
  %output.index.next = add nsw i32 %output.index, %coordinate.offset
  %new.shape.ptr = getelementptr inbounds i32, ptr %new_shape, i64 %dimension64
  %new.shape.value = load i32, ptr %new.shape.ptr, align 4
  %output.stride.next = mul nsw i32 %output.stride, %new.shape.value
  %dimension.next = sub nsw i32 %dimension, 1
  br label %coordinate.header

copy.store:
  %input.index64 = sext i32 %input.index to i64
  %output.index64 = sext i32 %output.index to i64
  %input.ptr = getelementptr inbounds double, ptr %input, i64 %input.index64
  %output.ptr = getelementptr inbounds double, ptr %output, i64 %output.index64
  %input.value = load double, ptr %input.ptr, align 8
  store double %input.value, ptr %output.ptr, align 8
  %input.index.next = add nsw i32 %input.index, 1
  br label %copy.header

exit:
  ret void
}

define internal double @binary_value(double %a, double %b, i32 %op) {
entry:
  switch i32 %op, label %invalid [
    i32 0, label %add
    i32 1, label %sub
    i32 2, label %mul
    i32 3, label %div
    i32 4, label %pow
    i32 5, label %mod
    i32 6, label %floordiv
    i32 20, label %lt
    i32 21, label %le
    i32 22, label %gt
    i32 23, label %ge
    i32 24, label %eq
    i32 25, label %ne
    i32 26, label %maximum
    i32 27, label %minimum
    i32 42, label %bitand
    i32 43, label %bitor
    i32 44, label %bitxor
    i32 45, label %shl
    i32 46, label %shr
    i32 47, label %logical_and
    i32 48, label %logical_or
  ]

add:
  %add.value = fadd double %a, %b
  ret double %add.value
sub:
  %sub.value = fsub double %a, %b
  ret double %sub.value
mul:
  %mul.value = fmul double %a, %b
  ret double %mul.value
div:
  %div.value = fdiv double %a, %b
  ret double %div.value
pow:
  %pow.value = call double @pow(double %a, double %b)
  ret double %pow.value
mod:
  %mod.quotient = fdiv double %a, %b
  %mod.floor = call double @llvm.floor.f64(double %mod.quotient)
  %mod.product = fmul double %mod.floor, %b
  %mod.value = fsub double %a, %mod.product
  ret double %mod.value
floordiv:
  %floordiv.quotient = fdiv double %a, %b
  %floordiv.value = call double @llvm.floor.f64(double %floordiv.quotient)
  ret double %floordiv.value
lt:
  %lt.bit = fcmp olt double %a, %b
  %lt.value = uitofp i1 %lt.bit to double
  ret double %lt.value
le:
  %le.bit = fcmp ole double %a, %b
  %le.value = uitofp i1 %le.bit to double
  ret double %le.value
gt:
  %gt.bit = fcmp ogt double %a, %b
  %gt.value = uitofp i1 %gt.bit to double
  ret double %gt.value
ge:
  %ge.bit = fcmp oge double %a, %b
  %ge.value = uitofp i1 %ge.bit to double
  ret double %ge.value
eq:
  %eq.bit = fcmp oeq double %a, %b
  %eq.value = uitofp i1 %eq.bit to double
  ret double %eq.value
ne:
  %ne.bit = fcmp une double %a, %b
  %ne.value = uitofp i1 %ne.bit to double
  ret double %ne.value
maximum:
  %maximum.bit = fcmp ogt double %a, %b
  %maximum.value = select i1 %maximum.bit, double %a, double %b
  ret double %maximum.value
minimum:
  %minimum.bit = fcmp olt double %a, %b
  %minimum.value = select i1 %minimum.bit, double %a, double %b
  ret double %minimum.value
bitand:
  %bitand.a = fptosi double %a to i64
  %bitand.b = fptosi double %b to i64
  %bitand.int = and i64 %bitand.a, %bitand.b
  %bitand.value = sitofp i64 %bitand.int to double
  ret double %bitand.value
bitor:
  %bitor.a = fptosi double %a to i64
  %bitor.b = fptosi double %b to i64
  %bitor.int = or i64 %bitor.a, %bitor.b
  %bitor.value = sitofp i64 %bitor.int to double
  ret double %bitor.value
bitxor:
  %bitxor.a = fptosi double %a to i64
  %bitxor.b = fptosi double %b to i64
  %bitxor.int = xor i64 %bitxor.a, %bitxor.b
  %bitxor.value = sitofp i64 %bitxor.int to double
  ret double %bitxor.value
shl:
  %shl.a = fptosi double %a to i64
  %shl.b.raw = fptosi double %b to i64
  %shl.b = and i64 %shl.b.raw, 63
  %shl.int = shl i64 %shl.a, %shl.b
  %shl.value = sitofp i64 %shl.int to double
  ret double %shl.value
shr:
  %shr.a = fptosi double %a to i64
  %shr.b.raw = fptosi double %b to i64
  %shr.b = and i64 %shr.b.raw, 63
  %shr.int = ashr i64 %shr.a, %shr.b
  %shr.value = sitofp i64 %shr.int to double
  ret double %shr.value
logical_and:
  %logical.and.a = fcmp une double %a, 0.000000e+00
  %logical.and.b = fcmp une double %b, 0.000000e+00
  %logical.and.bit = and i1 %logical.and.a, %logical.and.b
  %logical.and.value = uitofp i1 %logical.and.bit to double
  ret double %logical.and.value
logical_or:
  %logical.or.a = fcmp une double %a, 0.000000e+00
  %logical.or.b = fcmp une double %b, 0.000000e+00
  %logical.or.bit = or i1 %logical.or.a, %logical.or.b
  %logical.or.value = uitofp i1 %logical.or.bit to double
  ret double %logical.or.value
invalid:
  ret double 0x7FF8000000000000
}

define void @binary_double(ptr %a, ptr %b, ptr %out, i32 %n, i32 %op) {
entry:
  br label %loop.header

loop.header:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop.body ]
  %continue = icmp slt i32 %i, %n
  br i1 %continue, label %loop.body, label %exit

loop.body:
  %i64 = sext i32 %i to i64
  %a.ptr = getelementptr inbounds double, ptr %a, i64 %i64
  %b.ptr = getelementptr inbounds double, ptr %b, i64 %i64
  %out.ptr = getelementptr inbounds double, ptr %out, i64 %i64
  %a.value = load double, ptr %a.ptr, align 8
  %b.value = load double, ptr %b.ptr, align 8
  %result = call double @binary_value(double %a.value, double %b.value, i32 %op)
  store double %result, ptr %out.ptr, align 8
  %i.next = add nsw i32 %i, 1
  br label %loop.header

exit:
  ret void
}

define void @binary_scalar_double(
    ptr %a, double %b, ptr %out, i32 %n, i32 %op, i32 %reverse) {
entry:
  %is.reverse = icmp ne i32 %reverse, 0
  br label %loop.header

loop.header:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop.body ]
  %continue = icmp slt i32 %i, %n
  br i1 %continue, label %loop.body, label %exit

loop.body:
  %i64 = sext i32 %i to i64
  %a.ptr = getelementptr inbounds double, ptr %a, i64 %i64
  %out.ptr = getelementptr inbounds double, ptr %out, i64 %i64
  %a.value = load double, ptr %a.ptr, align 8
  %left = select i1 %is.reverse, double %b, double %a.value
  %right = select i1 %is.reverse, double %a.value, double %b
  %result = call double @binary_value(double %left, double %right, i32 %op)
  store double %result, ptr %out.ptr, align 8
  %i.next = add nsw i32 %i, 1
  br label %loop.header

exit:
  ret void
}

define void @unary_double(ptr %a, ptr %out, i32 %n, i32 %op) {
entry:
  br label %loop.header

loop.header:
  %i = phi i32 [ 0, %entry ], [ %i.next, %store ]
  %continue = icmp slt i32 %i, %n
  br i1 %continue, label %loop.body, label %exit

loop.body:
  %i64 = sext i32 %i to i64
  %a.ptr = getelementptr inbounds double, ptr %a, i64 %i64
  %out.ptr = getelementptr inbounds double, ptr %out, i64 %i64
  %value = load double, ptr %a.ptr, align 8
  switch i32 %op, label %identity [
    i32 7, label %sqrt
    i32 8, label %exp
    i32 9, label %log
    i32 10, label %neg
    i32 11, label %abs
    i32 12, label %round
    i32 13, label %trunc
    i32 14, label %floor
    i32 15, label %ceil
    i32 16, label %isfinite
    i32 17, label %isnan
    i32 18, label %isinf
    i32 19, label %logical_not
    i32 28, label %tanh
    i32 29, label %sin
    i32 30, label %cos
    i32 31, label %tan
    i32 32, label %asin
    i32 33, label %acos
    i32 34, label %atan
    i32 35, label %sinh
    i32 36, label %cosh
    i32 37, label %asinh
    i32 38, label %acosh
    i32 39, label %atanh
    i32 40, label %sign
    i32 41, label %invert
  ]

sqrt:
  %sqrt.value = call double @llvm.sqrt.f64(double %value)
  br label %store
exp:
  %exp.value = call double @exp(double %value)
  br label %store
log:
  %log.value = call double @log(double %value)
  br label %store
neg:
  %neg.value = fneg double %value
  br label %store
abs:
  %abs.value = call double @llvm.fabs.f64(double %value)
  br label %store
round:
  %round.value = call double @llvm.round.f64(double %value)
  br label %store
trunc:
  %trunc.value = call double @llvm.trunc.f64(double %value)
  br label %store
floor:
  %floor.value = call double @llvm.floor.f64(double %value)
  br label %store
ceil:
  %ceil.value = call double @llvm.ceil.f64(double %value)
  br label %store
isfinite:
  %finite.ordered = fcmp ord double %value, %value
  %finite.abs = call double @llvm.fabs.f64(double %value)
  %finite.not.inf = fcmp one double %finite.abs, 0x7FF0000000000000
  %finite.bit = and i1 %finite.ordered, %finite.not.inf
  %finite.value = uitofp i1 %finite.bit to double
  br label %store
isnan:
  %nan.bit = fcmp uno double %value, %value
  %nan.value = uitofp i1 %nan.bit to double
  br label %store
isinf:
  %inf.abs = call double @llvm.fabs.f64(double %value)
  %inf.bit = fcmp oeq double %inf.abs, 0x7FF0000000000000
  %inf.value = uitofp i1 %inf.bit to double
  br label %store
logical_not:
  %not.bit = fcmp oeq double %value, 0.000000e+00
  %not.value = uitofp i1 %not.bit to double
  br label %store
tanh:
  %tanh.value = call double @tanh(double %value)
  br label %store
sin:
  %sin.value = call double @sin(double %value)
  br label %store
cos:
  %cos.value = call double @cos(double %value)
  br label %store
tan:
  %tan.value = call double @tan(double %value)
  br label %store
asin:
  %asin.value = call double @asin(double %value)
  br label %store
acos:
  %acos.value = call double @acos(double %value)
  br label %store
atan:
  %atan.value = call double @atan(double %value)
  br label %store
sinh:
  %sinh.value = call double @sinh(double %value)
  br label %store
cosh:
  %cosh.value = call double @cosh(double %value)
  br label %store
asinh:
  %asinh.value = call double @asinh(double %value)
  br label %store
acosh:
  %acosh.value = call double @acosh(double %value)
  br label %store
atanh:
  %atanh.value = call double @atanh(double %value)
  br label %store
sign:
  %sign.positive = fcmp ogt double %value, 0.000000e+00
  %sign.negative = fcmp olt double %value, 0.000000e+00
  %sign.positive.value = uitofp i1 %sign.positive to double
  %sign.negative.value = uitofp i1 %sign.negative to double
  %sign.value = fsub double %sign.positive.value, %sign.negative.value
  br label %store
invert:
  %invert.int = fptosi double %value to i64
  %invert.flipped = xor i64 %invert.int, -1
  %invert.value = sitofp i64 %invert.flipped to double
  br label %store
identity:
  br label %store

store:
  %result = phi double
    [ %sqrt.value, %sqrt ],
    [ %exp.value, %exp ],
    [ %log.value, %log ],
    [ %neg.value, %neg ],
    [ %abs.value, %abs ],
    [ %round.value, %round ],
    [ %trunc.value, %trunc ],
    [ %floor.value, %floor ],
    [ %ceil.value, %ceil ],
    [ %finite.value, %isfinite ],
    [ %nan.value, %isnan ],
    [ %inf.value, %isinf ],
    [ %not.value, %logical_not ],
    [ %tanh.value, %tanh ],
    [ %sin.value, %sin ],
    [ %cos.value, %cos ],
    [ %tan.value, %tan ],
    [ %asin.value, %asin ],
    [ %acos.value, %acos ],
    [ %atan.value, %atan ],
    [ %sinh.value, %sinh ],
    [ %cosh.value, %cosh ],
    [ %asinh.value, %asinh ],
    [ %acosh.value, %acosh ],
    [ %atanh.value, %atanh ],
    [ %sign.value, %sign ],
    [ %invert.value, %invert ],
    [ %value, %identity ]
  store double %result, ptr %out.ptr, align 8
  %i.next = add nsw i32 %i, 1
  br label %loop.header

exit:
  ret void
}

define void @where_double(
    ptr %condition, ptr %x, ptr %y, ptr %out, i32 %n) {
entry:
  br label %loop.header

loop.header:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop.body ]
  %continue = icmp slt i32 %i, %n
  br i1 %continue, label %loop.body, label %exit

loop.body:
  %i64 = sext i32 %i to i64
  %condition.ptr = getelementptr inbounds double, ptr %condition, i64 %i64
  %x.ptr = getelementptr inbounds double, ptr %x, i64 %i64
  %y.ptr = getelementptr inbounds double, ptr %y, i64 %i64
  %out.ptr = getelementptr inbounds double, ptr %out, i64 %i64
  %condition.value = load double, ptr %condition.ptr, align 8
  %x.value = load double, ptr %x.ptr, align 8
  %y.value = load double, ptr %y.ptr, align 8
  %condition.bit = fcmp une double %condition.value, 0.000000e+00
  %result = select i1 %condition.bit, double %x.value, double %y.value
  store double %result, ptr %out.ptr, align 8
  %i.next = add nsw i32 %i, 1
  br label %loop.header

exit:
  ret void
}

define void @broadcast_double(
    ptr %input, ptr %output, ptr %input_shape, i32 %input_ndim,
    ptr %output_shape, i32 %output_ndim) {
entry:
  br label %total.header

total.header:
  %total.axis = phi i32 [ 0, %entry ], [ %total.axis.next, %total.body ]
  %total = phi i32 [ 1, %entry ], [ %total.next, %total.body ]
  %total.continue = icmp slt i32 %total.axis, %output_ndim
  br i1 %total.continue, label %total.body, label %flat.header

total.body:
  %total.axis64 = sext i32 %total.axis to i64
  %total.extent.ptr = getelementptr inbounds i32, ptr %output_shape, i64 %total.axis64
  %total.extent = load i32, ptr %total.extent.ptr, align 4
  %total.next = mul nsw i32 %total, %total.extent
  %total.axis.next = add nsw i32 %total.axis, 1
  br label %total.header

flat.header:
  %flat = phi i32 [ 0, %total.header ], [ %flat.next, %flat.latch ]
  %flat.continue = icmp slt i32 %flat, %total
  br i1 %flat.continue, label %axis.header, label %exit

axis.header:
  %axis = phi i32 [ 0, %flat.header ], [ %axis.next, %axis.latch ]
  %remaining = phi i32 [ %flat, %flat.header ], [ %remaining.next, %axis.latch ]
  %input.flat = phi i32 [ 0, %flat.header ], [ %input.flat.next, %axis.latch ]
  %axis.continue = icmp slt i32 %axis, %output_ndim
  br i1 %axis.continue, label %output.stride.entry, label %store

output.stride.entry:
  %output.later.start = add nsw i32 %axis, 1
  br label %output.stride.header

output.stride.header:
  %output.later = phi i32 [ %output.later.start, %output.stride.entry ], [ %output.later.next, %output.stride.body ]
  %output.stride = phi i32 [ 1, %output.stride.entry ], [ %output.stride.next, %output.stride.body ]
  %output.later.continue = icmp slt i32 %output.later, %output_ndim
  br i1 %output.later.continue, label %output.stride.body, label %coordinate

output.stride.body:
  %output.later64 = sext i32 %output.later to i64
  %output.extent.ptr = getelementptr inbounds i32, ptr %output_shape, i64 %output.later64
  %output.extent = load i32, ptr %output.extent.ptr, align 4
  %output.stride.next = mul nsw i32 %output.stride, %output.extent
  %output.later.next = add nsw i32 %output.later, 1
  br label %output.stride.header

coordinate:
  %coordinate.value = sdiv i32 %remaining, %output.stride
  %remaining.next = srem i32 %remaining, %output.stride
  %offset = sub nsw i32 %output_ndim, %input_ndim
  %input.axis = sub nsw i32 %axis, %offset
  %input.axis.valid = icmp sge i32 %input.axis, 0
  br i1 %input.axis.valid, label %input.extent.block, label %axis.noinput

input.extent.block:
  %input.axis64 = sext i32 %input.axis to i64
  %input.extent.ptr = getelementptr inbounds i32, ptr %input_shape, i64 %input.axis64
  %input.extent = load i32, ptr %input.extent.ptr, align 4
  %input.broadcast = icmp eq i32 %input.extent, 1
  br i1 %input.broadcast, label %axis.noinput, label %input.stride.entry

input.stride.entry:
  %input.later.start = add nsw i32 %input.axis, 1
  br label %input.stride.header

input.stride.header:
  %input.later = phi i32 [ %input.later.start, %input.stride.entry ], [ %input.later.next, %input.stride.body ]
  %input.stride = phi i32 [ 1, %input.stride.entry ], [ %input.stride.next, %input.stride.body ]
  %input.later.continue = icmp slt i32 %input.later, %input_ndim
  br i1 %input.later.continue, label %input.stride.body, label %axis.input

input.stride.body:
  %input.later64 = sext i32 %input.later to i64
  %input.later.extent.ptr = getelementptr inbounds i32, ptr %input_shape, i64 %input.later64
  %input.later.extent = load i32, ptr %input.later.extent.ptr, align 4
  %input.stride.next = mul nsw i32 %input.stride, %input.later.extent
  %input.later.next = add nsw i32 %input.later, 1
  br label %input.stride.header

axis.input:
  %input.contribution = mul nsw i32 %coordinate.value, %input.stride
  %input.flat.with = add nsw i32 %input.flat, %input.contribution
  br label %axis.latch

axis.noinput:
  br label %axis.latch

axis.latch:
  %input.flat.next = phi i32 [ %input.flat, %axis.noinput ], [ %input.flat.with, %axis.input ]
  %axis.next = add nsw i32 %axis, 1
  br label %axis.header

store:
  %input.flat64 = sext i32 %input.flat to i64
  %flat64 = sext i32 %flat to i64
  %input.ptr = getelementptr inbounds double, ptr %input, i64 %input.flat64
  %output.ptr = getelementptr inbounds double, ptr %output, i64 %flat64
  %value = load double, ptr %input.ptr, align 8
  store double %value, ptr %output.ptr, align 8
  br label %flat.latch

flat.latch:
  %flat.next = add nsw i32 %flat, 1
  br label %flat.header

exit:
  ret void
}

define void @slice_copy_double(
    ptr %input, ptr %output, ptr %shape, i32 %ndim, i32 %dim,
    i32 %start, i32 %step, i32 %count) {
entry:
  br label %before.header

before.header:
  %before.axis = phi i32 [ 0, %entry ], [ %before.axis.next, %before.body ]
  %before = phi i32 [ 1, %entry ], [ %before.next, %before.body ]
  %before.continue = icmp slt i32 %before.axis, %dim
  br i1 %before.continue, label %before.body, label %after.init

before.body:
  %before.axis64 = sext i32 %before.axis to i64
  %before.shape.ptr = getelementptr inbounds i32, ptr %shape, i64 %before.axis64
  %before.extent = load i32, ptr %before.shape.ptr, align 4
  %before.next = mul nsw i32 %before, %before.extent
  %before.axis.next = add nsw i32 %before.axis, 1
  br label %before.header

after.init:
  %after.first = add nsw i32 %dim, 1
  br label %after.header

after.header:
  %after.axis = phi i32 [ %after.first, %after.init ], [ %after.axis.next, %after.body ]
  %after = phi i32 [ 1, %after.init ], [ %after.next, %after.body ]
  %after.continue = icmp slt i32 %after.axis, %ndim
  br i1 %after.continue, label %after.body, label %copy.init

after.body:
  %after.axis64 = sext i32 %after.axis to i64
  %after.shape.ptr = getelementptr inbounds i32, ptr %shape, i64 %after.axis64
  %after.extent = load i32, ptr %after.shape.ptr, align 4
  %after.next = mul nsw i32 %after, %after.extent
  %after.axis.next = add nsw i32 %after.axis, 1
  br label %after.header

copy.init:
  %dim64 = sext i32 %dim to i64
  %source.count.ptr = getelementptr inbounds i32, ptr %shape, i64 %dim64
  %source.count = load i32, ptr %source.count.ptr, align 4
  br label %batch.header

batch.header:
  %batch = phi i32 [ 0, %copy.init ], [ %batch.next, %batch.latch ]
  %batch.continue = icmp slt i32 %batch, %before
  br i1 %batch.continue, label %item.header, label %exit

item.header:
  %item = phi i32 [ 0, %batch.header ], [ %item.next, %item.latch ]
  %item.continue = icmp slt i32 %item, %count
  br i1 %item.continue, label %element.header, label %batch.latch

element.header:
  %element = phi i32 [ 0, %item.header ], [ %element.next, %element.body ]
  %element.continue = icmp slt i32 %element, %after
  br i1 %element.continue, label %element.body, label %item.latch

element.body:
  %output.batch.base = mul nsw i32 %batch, %count
  %output.item.base = add nsw i32 %output.batch.base, %item
  %output.outer = mul nsw i32 %output.item.base, %after
  %output.index = add nsw i32 %output.outer, %element
  %source.item.step = mul nsw i32 %item, %step
  %source.item = add nsw i32 %start, %source.item.step
  %source.batch.base = mul nsw i32 %batch, %source.count
  %source.item.base = add nsw i32 %source.batch.base, %source.item
  %source.outer = mul nsw i32 %source.item.base, %after
  %source.index = add nsw i32 %source.outer, %element
  %source.index64 = sext i32 %source.index to i64
  %output.index64 = sext i32 %output.index to i64
  %source.ptr = getelementptr inbounds double, ptr %input, i64 %source.index64
  %output.ptr = getelementptr inbounds double, ptr %output, i64 %output.index64
  %value = load double, ptr %source.ptr, align 8
  store double %value, ptr %output.ptr, align 8
  %element.next = add nsw i32 %element, 1
  br label %element.header

item.latch:
  %item.next = add nsw i32 %item, 1
  br label %item.header

batch.latch:
  %batch.next = add nsw i32 %batch, 1
  br label %batch.header

exit:
  ret void
}

define void @index_select_double(
    ptr %input, ptr %output, ptr %shape, i32 %ndim, i32 %dim,
    ptr %indices, i32 %index.count) {
entry:
  br label %before.header

before.header:
  %before.axis = phi i32 [ 0, %entry ], [ %before.axis.next, %before.body ]
  %before = phi i32 [ 1, %entry ], [ %before.next, %before.body ]
  %before.continue = icmp slt i32 %before.axis, %dim
  br i1 %before.continue, label %before.body, label %after.init

before.body:
  %before.axis64 = sext i32 %before.axis to i64
  %before.shape.ptr = getelementptr inbounds i32, ptr %shape, i64 %before.axis64
  %before.extent = load i32, ptr %before.shape.ptr, align 4
  %before.next = mul nsw i32 %before, %before.extent
  %before.axis.next = add nsw i32 %before.axis, 1
  br label %before.header

after.init:
  %after.first = add nsw i32 %dim, 1
  br label %after.header

after.header:
  %after.axis = phi i32 [ %after.first, %after.init ], [ %after.axis.next, %after.body ]
  %after = phi i32 [ 1, %after.init ], [ %after.next, %after.body ]
  %after.continue = icmp slt i32 %after.axis, %ndim
  br i1 %after.continue, label %after.body, label %copy.init

after.body:
  %after.axis64 = sext i32 %after.axis to i64
  %after.shape.ptr = getelementptr inbounds i32, ptr %shape, i64 %after.axis64
  %after.extent = load i32, ptr %after.shape.ptr, align 4
  %after.next = mul nsw i32 %after, %after.extent
  %after.axis.next = add nsw i32 %after.axis, 1
  br label %after.header

copy.init:
  %dim64 = sext i32 %dim to i64
  %source.count.ptr = getelementptr inbounds i32, ptr %shape, i64 %dim64
  %source.count = load i32, ptr %source.count.ptr, align 4
  br label %batch.header

batch.header:
  %batch = phi i32 [ 0, %copy.init ], [ %batch.next, %batch.latch ]
  %batch.continue = icmp slt i32 %batch, %before
  br i1 %batch.continue, label %item.header, label %exit

item.header:
  %item = phi i32 [ 0, %batch.header ], [ %item.next, %item.latch ]
  %item.continue = icmp slt i32 %item, %index.count
  br i1 %item.continue, label %item.load, label %batch.latch

item.load:
  %item64 = sext i32 %item to i64
  %index.ptr = getelementptr inbounds i32, ptr %indices, i64 %item64
  %source.item = load i32, ptr %index.ptr, align 4
  br label %element.header

element.header:
  %element = phi i32 [ 0, %item.load ], [ %element.next, %element.body ]
  %element.continue = icmp slt i32 %element, %after
  br i1 %element.continue, label %element.body, label %item.latch

element.body:
  %output.batch.base = mul nsw i32 %batch, %index.count
  %output.item.base = add nsw i32 %output.batch.base, %item
  %output.outer = mul nsw i32 %output.item.base, %after
  %output.index = add nsw i32 %output.outer, %element
  %source.batch.base = mul nsw i32 %batch, %source.count
  %source.item.base = add nsw i32 %source.batch.base, %source.item
  %source.outer = mul nsw i32 %source.item.base, %after
  %source.index = add nsw i32 %source.outer, %element
  %source.index64 = sext i32 %source.index to i64
  %output.index64 = sext i32 %output.index to i64
  %source.ptr = getelementptr inbounds double, ptr %input, i64 %source.index64
  %output.ptr = getelementptr inbounds double, ptr %output, i64 %output.index64
  %value = load double, ptr %source.ptr, align 8
  store double %value, ptr %output.ptr, align 8
  %element.next = add nsw i32 %element, 1
  br label %element.header

item.latch:
  %item.next = add nsw i32 %item, 1
  br label %item.header

batch.latch:
  %batch.next = add nsw i32 %batch, 1
  br label %batch.header

exit:
  ret void
}

define void @index_assign_double(
    ptr %target, ptr %shape, i32 %ndim, ptr %axis.offsets,
    ptr %axis.indices, ptr %values, i32 %value.count) {
entry:
  %scalar = icmp eq i32 %ndim, 0
  br i1 %scalar, label %scalar.body, label %count.header

scalar.body:
  %scalar.value = load double, ptr %values, align 8
  store double %scalar.value, ptr %target, align 8
  br label %exit

count.header:
  %count.axis = phi i32 [ 0, %entry ], [ %count.axis.next, %count.body ]
  %selected.count = phi i32 [ 1, %entry ], [ %selected.count.next, %count.body ]
  %count.continue = icmp slt i32 %count.axis, %ndim
  br i1 %count.continue, label %count.body, label %flat.header

count.body:
  %count.axis64 = sext i32 %count.axis to i64
  %count.next.axis = add nsw i32 %count.axis, 1
  %count.next.axis64 = sext i32 %count.next.axis to i64
  %count.begin.ptr = getelementptr inbounds i32, ptr %axis.offsets, i64 %count.axis64
  %count.end.ptr = getelementptr inbounds i32, ptr %axis.offsets, i64 %count.next.axis64
  %count.begin = load i32, ptr %count.begin.ptr, align 4
  %count.end = load i32, ptr %count.end.ptr, align 4
  %axis.count = sub nsw i32 %count.end, %count.begin
  %selected.count.next = mul nsw i32 %selected.count, %axis.count
  %count.axis.next = add nsw i32 %count.axis, 1
  br label %count.header

flat.header:
  %flat = phi i32 [ 0, %count.header ], [ %flat.next, %flat.latch ]
  %flat.continue = icmp slt i32 %flat, %selected.count
  br i1 %flat.continue, label %axis.header, label %exit

axis.header:
  %axis = phi i32 [ 0, %flat.header ], [ %axis.next, %axis.latch ]
  %target.flat = phi i32 [ 0, %flat.header ], [ %target.flat.next, %axis.latch ]
  %axis.continue = icmp slt i32 %axis, %ndim
  br i1 %axis.continue, label %axis.body, label %flat.store

axis.body:
  %axis64 = sext i32 %axis to i64
  %axis.next.raw = add nsw i32 %axis, 1
  %axis.next64 = sext i32 %axis.next.raw to i64
  %axis.begin.ptr = getelementptr inbounds i32, ptr %axis.offsets, i64 %axis64
  %axis.end.ptr = getelementptr inbounds i32, ptr %axis.offsets, i64 %axis.next64
  %axis.begin = load i32, ptr %axis.begin.ptr, align 4
  %axis.end = load i32, ptr %axis.end.ptr, align 4
  %current.axis.count = sub nsw i32 %axis.end, %axis.begin
  br label %later.header

later.header:
  %later = phi i32 [ %axis.next.raw, %axis.body ], [ %later.next, %later.body ]
  %selection.stride = phi i32 [ 1, %axis.body ], [ %selection.stride.next, %later.body ]
  %target.stride = phi i32 [ 1, %axis.body ], [ %target.stride.next, %later.body ]
  %later.continue = icmp slt i32 %later, %ndim
  br i1 %later.continue, label %later.body, label %axis.coordinate

later.body:
  %later64 = sext i32 %later to i64
  %later.next.raw = add nsw i32 %later, 1
  %later.next64 = sext i32 %later.next.raw to i64
  %later.begin.ptr = getelementptr inbounds i32, ptr %axis.offsets, i64 %later64
  %later.end.ptr = getelementptr inbounds i32, ptr %axis.offsets, i64 %later.next64
  %later.begin = load i32, ptr %later.begin.ptr, align 4
  %later.end = load i32, ptr %later.end.ptr, align 4
  %later.axis.count = sub nsw i32 %later.end, %later.begin
  %selection.stride.next = mul nsw i32 %selection.stride, %later.axis.count
  %shape.ptr = getelementptr inbounds i32, ptr %shape, i64 %later64
  %shape.extent = load i32, ptr %shape.ptr, align 4
  %target.stride.next = mul nsw i32 %target.stride, %shape.extent
  %later.next = add nsw i32 %later, 1
  br label %later.header

axis.coordinate:
  %coordinate.quotient = sdiv i32 %flat, %selection.stride
  %coordinate = srem i32 %coordinate.quotient, %current.axis.count
  %selected.offset = add nsw i32 %axis.begin, %coordinate
  %selected.offset64 = sext i32 %selected.offset to i64
  %selected.ptr = getelementptr inbounds i32, ptr %axis.indices, i64 %selected.offset64
  %selected = load i32, ptr %selected.ptr, align 4
  %target.contribution = mul nsw i32 %selected, %target.stride
  %target.flat.computed = add nsw i32 %target.flat, %target.contribution
  br label %axis.latch

axis.latch:
  %target.flat.next = phi i32 [ %target.flat.computed, %axis.coordinate ]
  %axis.next = add nsw i32 %axis, 1
  br label %axis.header

flat.store:
  %broadcast = icmp eq i32 %value.count, 1
  %value.index = select i1 %broadcast, i32 0, i32 %flat
  %value.index64 = sext i32 %value.index to i64
  %target.index64 = sext i32 %target.flat to i64
  %value.ptr = getelementptr inbounds double, ptr %values, i64 %value.index64
  %target.ptr = getelementptr inbounds double, ptr %target, i64 %target.index64
  %value = load double, ptr %value.ptr, align 8
  store double %value, ptr %target.ptr, align 8
  br label %flat.latch

flat.latch:
  %flat.next = add nsw i32 %flat, 1
  br label %flat.header

exit:
  ret void
}

define void @index_set_double(
    ptr %input, ptr %output, ptr %shape, i32 %ndim,
    ptr %axis.offsets, ptr %axis.indices, ptr %values, i32 %value.count) {
entry:
  br label %count.header

count.header:
  %axis = phi i32 [ 0, %entry ], [ %axis.next, %count.body ]
  %element.count = phi i32 [ 1, %entry ], [ %element.count.next, %count.body ]
  %continue = icmp slt i32 %axis, %ndim
  br i1 %continue, label %count.body, label %copy

count.body:
  %axis64 = sext i32 %axis to i64
  %shape.ptr = getelementptr inbounds i32, ptr %shape, i64 %axis64
  %extent = load i32, ptr %shape.ptr, align 4
  %element.count.next = mul nsw i32 %element.count, %extent
  %axis.next = add nsw i32 %axis, 1
  br label %count.header

copy:
  %byte.count32 = mul nsw i32 %element.count, 8
  %byte.count = sext i32 %byte.count32 to i64
  call void @llvm.memcpy.p0.p0.i64(
      ptr %output, ptr %input, i64 %byte.count, i1 false)
  call void @index_assign_double(
      ptr %output, ptr %shape, i32 %ndim,
      ptr %axis.offsets, ptr %axis.indices,
      ptr %values, i32 %value.count)
  ret void
}

define void @unfold2d_double(
    ptr %input, ptr %output,
    i32 %n, i32 %c, i32 %h, i32 %w,
    i32 %kh.count, i32 %kw.count,
    i32 %sh, i32 %sw, i32 %ph, i32 %pw, i32 %dh, i32 %dw) {
entry:
  %kh.minus = sub nsw i32 %kh.count, 1
  %kw.minus = sub nsw i32 %kw.count, 1
  %effective.h.raw = mul nsw i32 %kh.minus, %dh
  %effective.w.raw = mul nsw i32 %kw.minus, %dw
  %effective.h = add nsw i32 %effective.h.raw, 1
  %effective.w = add nsw i32 %effective.w.raw, 1
  %twoph = mul nsw i32 %ph, 2
  %twopw = mul nsw i32 %pw, 2
  %padded.h = add nsw i32 %h, %twoph
  %padded.w = add nsw i32 %w, %twopw
  %numerator.h = sub nsw i32 %padded.h, %effective.h
  %numerator.w = sub nsw i32 %padded.w, %effective.w
  %oh.raw = sdiv i32 %numerator.h, %sh
  %ow.raw = sdiv i32 %numerator.w, %sw
  %oh.count = add nsw i32 %oh.raw, 1
  %ow.count = add nsw i32 %ow.raw, 1
  %count.0 = mul nsw i32 %n, %c
  %count.1 = mul nsw i32 %count.0, %kh.count
  %count.2 = mul nsw i32 %count.1, %kw.count
  %count.3 = mul nsw i32 %count.2, %oh.count
  %count = mul nsw i32 %count.3, %ow.count
  br label %loop.header

loop.header:
  %flat = phi i32 [ 0, %entry ], [ %flat.next, %loop.latch ]
  %continue = icmp slt i32 %flat, %count
  br i1 %continue, label %decode, label %exit

decode:
  %ow = srem i32 %flat, %ow.count
  %q.ow = sdiv i32 %flat, %ow.count
  %oh = srem i32 %q.ow, %oh.count
  %q.oh = sdiv i32 %q.ow, %oh.count
  %kw = srem i32 %q.oh, %kw.count
  %q.kw = sdiv i32 %q.oh, %kw.count
  %kh = srem i32 %q.kw, %kh.count
  %q.kh = sdiv i32 %q.kw, %kh.count
  %channel = srem i32 %q.kh, %c
  %batch = sdiv i32 %q.kh, %c
  %ih.base = mul nsw i32 %oh, %sh
  %iw.base = mul nsw i32 %ow, %sw
  %ih.unpadded = sub nsw i32 %ih.base, %ph
  %iw.unpadded = sub nsw i32 %iw.base, %pw
  %kh.offset = mul nsw i32 %kh, %dh
  %kw.offset = mul nsw i32 %kw, %dw
  %ih = add nsw i32 %ih.unpadded, %kh.offset
  %iw = add nsw i32 %iw.unpadded, %kw.offset
  %ih.low = icmp sge i32 %ih, 0
  %ih.high = icmp slt i32 %ih, %h
  %iw.low = icmp sge i32 %iw, 0
  %iw.high = icmp slt i32 %iw, %w
  %h.valid = and i1 %ih.low, %ih.high
  %w.valid = and i1 %iw.low, %iw.high
  %valid = and i1 %h.valid, %w.valid
  br i1 %valid, label %load, label %zero

load:
  %input.bc = mul nsw i32 %batch, %c
  %input.channel = add nsw i32 %input.bc, %channel
  %input.row.base = mul nsw i32 %input.channel, %h
  %input.row = add nsw i32 %input.row.base, %ih
  %input.flat.base = mul nsw i32 %input.row, %w
  %input.flat = add nsw i32 %input.flat.base, %iw
  %input.flat64 = sext i32 %input.flat to i64
  %input.ptr = getelementptr inbounds double, ptr %input, i64 %input.flat64
  %loaded = load double, ptr %input.ptr, align 8
  br label %store

zero:
  br label %store

store:
  %value = phi double [ %loaded, %load ], [ 0.000000e+00, %zero ]
  %flat64 = sext i32 %flat to i64
  %output.ptr = getelementptr inbounds double, ptr %output, i64 %flat64
  store double %value, ptr %output.ptr, align 8
  br label %loop.latch

loop.latch:
  %flat.next = add nsw i32 %flat, 1
  br label %loop.header

exit:
  ret void
}

define void @fold2d_double(
    ptr %columns, ptr %output,
    i32 %n, i32 %c, i32 %h, i32 %w,
    i32 %kh.count, i32 %kw.count,
    i32 %sh, i32 %sw, i32 %ph, i32 %pw, i32 %dh, i32 %dw) {
entry:
  %output.count.0 = mul nsw i32 %n, %c
  %output.count.1 = mul nsw i32 %output.count.0, %h
  %output.count = mul nsw i32 %output.count.1, %w
  %byte.count32 = mul nsw i32 %output.count, 8
  %byte.count = sext i32 %byte.count32 to i64
  call void @llvm.memset.p0.i64(ptr %output, i8 0, i64 %byte.count, i1 false)
  %kh.minus = sub nsw i32 %kh.count, 1
  %kw.minus = sub nsw i32 %kw.count, 1
  %effective.h.raw = mul nsw i32 %kh.minus, %dh
  %effective.w.raw = mul nsw i32 %kw.minus, %dw
  %effective.h = add nsw i32 %effective.h.raw, 1
  %effective.w = add nsw i32 %effective.w.raw, 1
  %twoph = mul nsw i32 %ph, 2
  %twopw = mul nsw i32 %pw, 2
  %padded.h = add nsw i32 %h, %twoph
  %padded.w = add nsw i32 %w, %twopw
  %numerator.h = sub nsw i32 %padded.h, %effective.h
  %numerator.w = sub nsw i32 %padded.w, %effective.w
  %oh.raw = sdiv i32 %numerator.h, %sh
  %ow.raw = sdiv i32 %numerator.w, %sw
  %oh.count = add nsw i32 %oh.raw, 1
  %ow.count = add nsw i32 %ow.raw, 1
  %count.0 = mul nsw i32 %n, %c
  %count.1 = mul nsw i32 %count.0, %kh.count
  %count.2 = mul nsw i32 %count.1, %kw.count
  %count.3 = mul nsw i32 %count.2, %oh.count
  %count = mul nsw i32 %count.3, %ow.count
  br label %loop.header

loop.header:
  %flat = phi i32 [ 0, %entry ], [ %flat.next, %loop.latch ]
  %continue = icmp slt i32 %flat, %count
  br i1 %continue, label %decode, label %exit

decode:
  %ow = srem i32 %flat, %ow.count
  %q.ow = sdiv i32 %flat, %ow.count
  %oh = srem i32 %q.ow, %oh.count
  %q.oh = sdiv i32 %q.ow, %oh.count
  %kw = srem i32 %q.oh, %kw.count
  %q.kw = sdiv i32 %q.oh, %kw.count
  %kh = srem i32 %q.kw, %kh.count
  %q.kh = sdiv i32 %q.kw, %kh.count
  %channel = srem i32 %q.kh, %c
  %batch = sdiv i32 %q.kh, %c
  %ih.base = mul nsw i32 %oh, %sh
  %iw.base = mul nsw i32 %ow, %sw
  %ih.unpadded = sub nsw i32 %ih.base, %ph
  %iw.unpadded = sub nsw i32 %iw.base, %pw
  %kh.offset = mul nsw i32 %kh, %dh
  %kw.offset = mul nsw i32 %kw, %dw
  %ih = add nsw i32 %ih.unpadded, %kh.offset
  %iw = add nsw i32 %iw.unpadded, %kw.offset
  %ih.low = icmp sge i32 %ih, 0
  %ih.high = icmp slt i32 %ih, %h
  %iw.low = icmp sge i32 %iw, 0
  %iw.high = icmp slt i32 %iw, %w
  %h.valid = and i1 %ih.low, %ih.high
  %w.valid = and i1 %iw.low, %iw.high
  %valid = and i1 %h.valid, %w.valid
  br i1 %valid, label %accumulate, label %loop.latch

accumulate:
  %flat64 = sext i32 %flat to i64
  %column.ptr = getelementptr inbounds double, ptr %columns, i64 %flat64
  %column.value = load double, ptr %column.ptr, align 8
  %output.bc = mul nsw i32 %batch, %c
  %output.channel = add nsw i32 %output.bc, %channel
  %output.row.base = mul nsw i32 %output.channel, %h
  %output.row = add nsw i32 %output.row.base, %ih
  %output.flat.base = mul nsw i32 %output.row, %w
  %output.flat = add nsw i32 %output.flat.base, %iw
  %output.flat64 = sext i32 %output.flat to i64
  %output.ptr = getelementptr inbounds double, ptr %output, i64 %output.flat64
  %previous = load double, ptr %output.ptr, align 8
  %updated = fadd double %previous, %column.value
  store double %updated, ptr %output.ptr, align 8
  br label %loop.latch

loop.latch:
  %flat.next = add nsw i32 %flat, 1
  br label %loop.header

exit:
  ret void
}

define void @sign_double(ptr %input, ptr %output, i32 %n) {
entry:
  br label %loop.header

loop.header:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop.body ]
  %continue = icmp slt i32 %i, %n
  br i1 %continue, label %loop.body, label %exit

loop.body:
  %i64 = sext i32 %i to i64
  %input.ptr = getelementptr inbounds double, ptr %input, i64 %i64
  %output.ptr = getelementptr inbounds double, ptr %output, i64 %i64
  %value = load double, ptr %input.ptr, align 8
  %positive = fcmp ogt double %value, 0.000000e+00
  %negative = fcmp olt double %value, 0.000000e+00
  %nonpositive = select i1 %negative, double -1.000000e+00, double 0.000000e+00
  %result = select i1 %positive, double 1.000000e+00, double %nonpositive
  store double %result, ptr %output.ptr, align 8
  %i.next = add nsw i32 %i, 1
  br label %loop.header

exit:
  ret void
}

define i32 @count_true_double(ptr %mask, i32 %n) {
entry:
  br label %loop.header

loop.header:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop.body ]
  %count = phi i32 [ 0, %entry ], [ %count.next, %loop.body ]
  %continue = icmp slt i32 %i, %n
  br i1 %continue, label %loop.body, label %exit

loop.body:
  %i64 = sext i32 %i to i64
  %mask.ptr = getelementptr inbounds double, ptr %mask, i64 %i64
  %mask.value = load double, ptr %mask.ptr, align 8
  %selected = fcmp une double %mask.value, 0.000000e+00
  %increment = zext i1 %selected to i32
  %count.next = add nsw i32 %count, %increment
  %i.next = add nsw i32 %i, 1
  br label %loop.header

exit:
  ret i32 %count
}

define void @mask_select_double(ptr %input, ptr %mask, ptr %output, i32 %n) {
entry:
  br label %loop.header

loop.header:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop.latch ]
  %target = phi i32 [ 0, %entry ], [ %target.next, %loop.latch ]
  %continue = icmp slt i32 %i, %n
  br i1 %continue, label %loop.body, label %exit

loop.body:
  %i64 = sext i32 %i to i64
  %mask.ptr = getelementptr inbounds double, ptr %mask, i64 %i64
  %mask.value = load double, ptr %mask.ptr, align 8
  %selected = fcmp une double %mask.value, 0.000000e+00
  br i1 %selected, label %copy, label %skip

copy:
  %input.ptr = getelementptr inbounds double, ptr %input, i64 %i64
  %input.value = load double, ptr %input.ptr, align 8
  %target64 = sext i32 %target to i64
  %output.ptr = getelementptr inbounds double, ptr %output, i64 %target64
  store double %input.value, ptr %output.ptr, align 8
  %target.incremented = add nsw i32 %target, 1
  br label %loop.latch

skip:
  br label %loop.latch

loop.latch:
  %target.next = phi i32
    [ %target.incremented, %copy ],
    [ %target, %skip ]
  %i.next = add nsw i32 %i, 1
  br label %loop.header

exit:
  ret void
}

define void @increment_mask_double(ptr %input, ptr %mask, i32 %n) {
entry:
  br label %loop.header

loop.header:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop.latch ]
  %continue = icmp slt i32 %i, %n
  br i1 %continue, label %loop.body, label %exit

loop.body:
  %i64 = sext i32 %i to i64
  %mask.ptr = getelementptr inbounds double, ptr %mask, i64 %i64
  %mask.value = load double, ptr %mask.ptr, align 8
  %selected = fcmp une double %mask.value, 0.000000e+00
  br i1 %selected, label %increment, label %loop.latch

increment:
  %input.ptr = getelementptr inbounds double, ptr %input, i64 %i64
  %input.value = load double, ptr %input.ptr, align 8
  %incremented = fadd double %input.value, 1.000000e+00
  store double %incremented, ptr %input.ptr, align 8
  br label %loop.latch

loop.latch:
  %i.next = add nsw i32 %i, 1
  br label %loop.header

exit:
  ret void
}

define void @cast_double_to_int_values(ptr %a, ptr %out, i32 %n) {
entry:
  br label %loop.header

loop.header:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop.body ]
  %continue = icmp slt i32 %i, %n
  br i1 %continue, label %loop.body, label %exit

loop.body:
  %i64 = sext i32 %i to i64
  %a.ptr = getelementptr inbounds double, ptr %a, i64 %i64
  %out.ptr = getelementptr inbounds double, ptr %out, i64 %i64
  %value = load double, ptr %a.ptr, align 8
  %integer = fptosi double %value to i64
  %result = sitofp i64 %integer to double
  store double %result, ptr %out.ptr, align 8
  %i.next = add nsw i32 %i, 1
  br label %loop.header

exit:
  ret void
}

define void @cast_double_to_float_values(ptr %a, ptr %out, i32 %n) {
entry:
  br label %loop.header

loop.header:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop.body ]
  %continue = icmp slt i32 %i, %n
  br i1 %continue, label %loop.body, label %exit

loop.body:
  %i64 = sext i32 %i to i64
  %a.ptr = getelementptr inbounds double, ptr %a, i64 %i64
  %out.ptr = getelementptr inbounds double, ptr %out, i64 %i64
  %value = load double, ptr %a.ptr, align 8
  %single = fptrunc double %value to float
  %result = fpext float %single to double
  store double %result, ptr %out.ptr, align 8
  %i.next = add nsw i32 %i, 1
  br label %loop.header

exit:
  ret void
}

define void @cast_double_to_double_values(ptr %a, ptr %out, i32 %n) {
entry:
  br label %loop.header

loop.header:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop.body ]
  %continue = icmp slt i32 %i, %n
  br i1 %continue, label %loop.body, label %exit

loop.body:
  %i64 = sext i32 %i to i64
  %a.ptr = getelementptr inbounds double, ptr %a, i64 %i64
  %out.ptr = getelementptr inbounds double, ptr %out, i64 %i64
  %value = load double, ptr %a.ptr, align 8
  store double %value, ptr %out.ptr, align 8
  %i.next = add nsw i32 %i, 1
  br label %loop.header

exit:
  ret void
}

define void @cast_double_to_bool_values(ptr %a, ptr %out, i32 %n) {
entry:
  br label %loop.header

loop.header:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop.body ]
  %continue = icmp slt i32 %i, %n
  br i1 %continue, label %loop.body, label %exit

loop.body:
  %i64 = sext i32 %i to i64
  %a.ptr = getelementptr inbounds double, ptr %a, i64 %i64
  %out.ptr = getelementptr inbounds double, ptr %out, i64 %i64
  %value = load double, ptr %a.ptr, align 8
  %nonzero = fcmp une double %value, 0.0
  %result = uitofp i1 %nonzero to double
  store double %result, ptr %out.ptr, align 8
  %i.next = add nsw i32 %i, 1
  br label %loop.header

exit:
  ret void
}

define double @sum_double(ptr %a, i32 %n) {
entry:
  br label %loop.header

loop.header:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop.body ]
  %sum = phi double [ 0.000000e+00, %entry ], [ %sum.next, %loop.body ]
  %continue = icmp slt i32 %i, %n
  br i1 %continue, label %loop.body, label %exit

loop.body:
  %i64 = sext i32 %i to i64
  %a.ptr = getelementptr inbounds double, ptr %a, i64 %i64
  %value = load double, ptr %a.ptr, align 8
  %sum.next = fadd double %sum, %value
  %i.next = add nsw i32 %i, 1
  br label %loop.header

exit:
  ret double %sum
}

define void @create_arange(double %start, double %step, i32 %n, ptr %out) {
entry:
  br label %loop.header

loop.header:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop.body ]
  %continue = icmp slt i32 %i, %n
  br i1 %continue, label %loop.body, label %exit

loop.body:
  %i.double = sitofp i32 %i to double
  %offset = fmul double %i.double, %step
  %value = fadd double %start, %offset
  %i64 = sext i32 %i to i64
  %out.ptr = getelementptr inbounds double, ptr %out, i64 %i64
  store double %value, ptr %out.ptr, align 8
  %i.next = add nsw i32 %i, 1
  br label %loop.header

exit:
  ret void
}

define void @matmul_double(
    ptr %a, ptr %b, ptr %out, i32 %m, i32 %n, i32 %p) {
entry:
  br label %i.header

i.header:
  %i = phi i32 [ 0, %entry ], [ %i.next, %i.latch ]
  %i.continue = icmp slt i32 %i, %m
  br i1 %i.continue, label %j.header, label %exit

j.header:
  %j = phi i32 [ 0, %i.header ], [ %j.next, %j.latch ]
  %j.continue = icmp slt i32 %j, %p
  br i1 %j.continue, label %k.header, label %i.latch

k.header:
  %k = phi i32 [ 0, %j.header ], [ %k.next, %k.body ]
  %sum = phi double [ 0.000000e+00, %j.header ], [ %sum.next, %k.body ]
  %k.continue = icmp slt i32 %k, %n
  br i1 %k.continue, label %k.body, label %store

k.body:
  %a.row = mul nsw i32 %i, %n
  %a.index = add nsw i32 %a.row, %k
  %b.row = mul nsw i32 %k, %p
  %b.index = add nsw i32 %b.row, %j
  %a.index64 = sext i32 %a.index to i64
  %b.index64 = sext i32 %b.index to i64
  %a.ptr = getelementptr inbounds double, ptr %a, i64 %a.index64
  %b.ptr = getelementptr inbounds double, ptr %b, i64 %b.index64
  %a.value = load double, ptr %a.ptr, align 8
  %b.value = load double, ptr %b.ptr, align 8
  %product = fmul double %a.value, %b.value
  %sum.next = fadd double %sum, %product
  %k.next = add nsw i32 %k, 1
  br label %k.header

store:
  %out.row = mul nsw i32 %i, %p
  %out.index = add nsw i32 %out.row, %j
  %out.index64 = sext i32 %out.index to i64
  %out.ptr = getelementptr inbounds double, ptr %out, i64 %out.index64
  store double %sum, ptr %out.ptr, align 8
  br label %j.latch

j.latch:
  %j.next = add nsw i32 %j, 1
  br label %j.header

i.latch:
  %i.next = add nsw i32 %i, 1
  br label %i.header

exit:
  ret void
}

define void @cumsum_dim_double(
    ptr %input, ptr %output, ptr %shape, i32 %ndim, i32 %dim) {
entry:
  br label %before.header

before.header:
  %before.axis = phi i32 [ 0, %entry ], [ %before.axis.next, %before.body ]
  %before = phi i32 [ 1, %entry ], [ %before.next, %before.body ]
  %before.continue = icmp slt i32 %before.axis, %dim
  br i1 %before.continue, label %before.body, label %after.entry

before.body:
  %before.axis64 = sext i32 %before.axis to i64
  %before.shape.ptr = getelementptr inbounds i32, ptr %shape, i64 %before.axis64
  %before.extent = load i32, ptr %before.shape.ptr, align 4
  %before.next = mul nsw i32 %before, %before.extent
  %before.axis.next = add nsw i32 %before.axis, 1
  br label %before.header

after.entry:
  %after.start = add nsw i32 %dim, 1
  br label %after.header

after.header:
  %after.axis = phi i32 [ %after.start, %after.entry ], [ %after.axis.next, %after.body ]
  %after = phi i32 [ 1, %after.entry ], [ %after.next, %after.body ]
  %after.continue = icmp slt i32 %after.axis, %ndim
  br i1 %after.continue, label %after.body, label %dimensions.ready

after.body:
  %after.axis64 = sext i32 %after.axis to i64
  %after.shape.ptr = getelementptr inbounds i32, ptr %shape, i64 %after.axis64
  %after.extent = load i32, ptr %after.shape.ptr, align 4
  %after.next = mul nsw i32 %after, %after.extent
  %after.axis.next = add nsw i32 %after.axis, 1
  br label %after.header

dimensions.ready:
  %dim64 = sext i32 %dim to i64
  %count.ptr = getelementptr inbounds i32, ptr %shape, i64 %dim64
  %count = load i32, ptr %count.ptr, align 4
  br label %b.header

b.header:
  %b = phi i32 [ 0, %dimensions.ready ], [ %b.next, %b.latch ]
  %b.continue = icmp slt i32 %b, %before
  br i1 %b.continue, label %tail.header, label %exit

tail.header:
  %tail = phi i32 [ 0, %b.header ], [ %tail.next, %tail.latch ]
  %tail.continue = icmp slt i32 %tail, %after
  br i1 %tail.continue, label %d.header, label %b.latch

d.header:
  %d = phi i32 [ 0, %tail.header ], [ %d.next, %d.body ]
  %accum = phi double [ 0.000000e+00, %tail.header ], [ %accum.next, %d.body ]
  %d.continue = icmp slt i32 %d, %count
  br i1 %d.continue, label %d.body, label %tail.latch

d.body:
  %b.count = mul nsw i32 %b, %count
  %bd = add nsw i32 %b.count, %d
  %bd.after = mul nsw i32 %bd, %after
  %index = add nsw i32 %bd.after, %tail
  %index64 = sext i32 %index to i64
  %input.ptr = getelementptr inbounds double, ptr %input, i64 %index64
  %output.ptr = getelementptr inbounds double, ptr %output, i64 %index64
  %value = load double, ptr %input.ptr, align 8
  %accum.next = fadd double %accum, %value
  store double %accum.next, ptr %output.ptr, align 8
  %d.next = add nsw i32 %d, 1
  br label %d.header

tail.latch:
  %tail.next = add nsw i32 %tail, 1
  br label %tail.header

b.latch:
  %b.next = add nsw i32 %b, 1
  br label %b.header

exit:
  ret void
}

define void @reduce_dim_double(
    ptr %input, ptr %output, ptr %shape, i32 %ndim, i32 %dim, i32 %op) {
entry:
  br label %before.header

before.header:
  %before.axis = phi i32 [ 0, %entry ], [ %before.axis.next, %before.body ]
  %before = phi i32 [ 1, %entry ], [ %before.next, %before.body ]
  %before.continue = icmp slt i32 %before.axis, %dim
  br i1 %before.continue, label %before.body, label %after.entry

before.body:
  %before.axis64 = sext i32 %before.axis to i64
  %before.shape.ptr = getelementptr inbounds i32, ptr %shape, i64 %before.axis64
  %before.extent = load i32, ptr %before.shape.ptr, align 4
  %before.next = mul nsw i32 %before, %before.extent
  %before.axis.next = add nsw i32 %before.axis, 1
  br label %before.header

after.entry:
  %after.start = add nsw i32 %dim, 1
  br label %after.header

after.header:
  %after.axis = phi i32 [ %after.start, %after.entry ], [ %after.axis.next, %after.body ]
  %after = phi i32 [ 1, %after.entry ], [ %after.next, %after.body ]
  %after.continue = icmp slt i32 %after.axis, %ndim
  br i1 %after.continue, label %after.body, label %dimensions.ready

after.body:
  %after.axis64 = sext i32 %after.axis to i64
  %after.shape.ptr = getelementptr inbounds i32, ptr %shape, i64 %after.axis64
  %after.extent = load i32, ptr %after.shape.ptr, align 4
  %after.next = mul nsw i32 %after, %after.extent
  %after.axis.next = add nsw i32 %after.axis, 1
  br label %after.header

dimensions.ready:
  %dim64 = sext i32 %dim to i64
  %count.ptr = getelementptr inbounds i32, ptr %shape, i64 %dim64
  %count = load i32, ptr %count.ptr, align 4
  %op.prod = icmp eq i32 %op, 1
  %op.min = icmp eq i32 %op, 2
  %op.max = icmp eq i32 %op, 3
  %op.all = icmp eq i32 %op, 5
  %initial.prod = select i1 %op.prod, double 1.000000e+00, double 0.000000e+00
  %initial.min = select i1 %op.min, double 0x7FF0000000000000, double %initial.prod
  %initial.max = select i1 %op.max, double 0xFFF0000000000000, double %initial.min
  %initial = select i1 %op.all, double 1.000000e+00, double %initial.max
  br label %b.header

b.header:
  %b = phi i32 [ 0, %dimensions.ready ], [ %b.next, %b.latch ]
  %b.continue = icmp slt i32 %b, %before
  br i1 %b.continue, label %tail.header, label %exit

tail.header:
  %tail = phi i32 [ 0, %b.header ], [ %tail.next, %tail.latch ]
  %tail.continue = icmp slt i32 %tail, %after
  br i1 %tail.continue, label %d.header, label %b.latch

d.header:
  %d = phi i32 [ 0, %tail.header ], [ %d.next, %d.body ]
  %accum = phi double [ %initial, %tail.header ], [ %accum.next, %d.body ]
  %d.continue = icmp slt i32 %d, %count
  br i1 %d.continue, label %d.body, label %store

d.body:
  %b.count = mul nsw i32 %b, %count
  %bd = add nsw i32 %b.count, %d
  %bd.after = mul nsw i32 %bd, %after
  %index = add nsw i32 %bd.after, %tail
  %index64 = sext i32 %index to i64
  %input.ptr = getelementptr inbounds double, ptr %input, i64 %index64
  %value = load double, ptr %input.ptr, align 8
  %sum.value = fadd double %accum, %value
  %product.value = fmul double %accum, %value
  %min.bit = fcmp olt double %value, %accum
  %min.value = select i1 %min.bit, double %value, double %accum
  %max.bit = fcmp ogt double %value, %accum
  %max.value = select i1 %max.bit, double %value, double %accum
  %truth = fcmp une double %value, 0.000000e+00
  %any.value = select i1 %truth, double 1.000000e+00, double %accum
  %all.value = select i1 %truth, double %accum, double 0.000000e+00
  %is.sum = icmp eq i32 %op, 0
  %is.prod = icmp eq i32 %op, 1
  %is.min = icmp eq i32 %op, 2
  %is.max = icmp eq i32 %op, 3
  %is.any = icmp eq i32 %op, 4
  %selected.sum = select i1 %is.sum, double %sum.value, double %accum
  %selected.prod = select i1 %is.prod, double %product.value, double %selected.sum
  %selected.min = select i1 %is.min, double %min.value, double %selected.prod
  %selected.max = select i1 %is.max, double %max.value, double %selected.min
  %selected.any = select i1 %is.any, double %any.value, double %selected.max
  %accum.next = select i1 %op.all, double %all.value, double %selected.any
  %d.next = add nsw i32 %d, 1
  br label %d.header

store:
  %output.base = mul nsw i32 %b, %after
  %output.index = add nsw i32 %output.base, %tail
  %output.index64 = sext i32 %output.index to i64
  %output.ptr = getelementptr inbounds double, ptr %output, i64 %output.index64
  store double %accum, ptr %output.ptr, align 8
  br label %tail.latch

tail.latch:
  %tail.next = add nsw i32 %tail, 1
  br label %tail.header

b.latch:
  %b.next = add nsw i32 %b, 1
  br label %b.header

exit:
  ret void
}

define void @transpose_double(
    ptr %input, ptr %output, ptr %shape, ptr %axes, i32 %ndim) {
entry:
  br label %total.header

total.header:
  %total.axis = phi i32 [ 0, %entry ], [ %total.axis.next, %total.body ]
  %total = phi i32 [ 1, %entry ], [ %total.next, %total.body ]
  %total.continue = icmp slt i32 %total.axis, %ndim
  br i1 %total.continue, label %total.body, label %flat.header

total.body:
  %total.axis64 = sext i32 %total.axis to i64
  %total.shape.ptr = getelementptr inbounds i32, ptr %shape, i64 %total.axis64
  %total.extent = load i32, ptr %total.shape.ptr, align 4
  %total.next = mul nsw i32 %total, %total.extent
  %total.axis.next = add nsw i32 %total.axis, 1
  br label %total.header

flat.header:
  %flat = phi i32 [ 0, %total.header ], [ %flat.next, %flat.latch ]
  %flat.continue = icmp slt i32 %flat, %total
  br i1 %flat.continue, label %axis.header, label %exit

axis.header:
  %out.axis = phi i32 [ 0, %flat.header ], [ %out.axis.next, %axis.latch ]
  %remaining = phi i32 [ %flat, %flat.header ], [ %remaining.next, %axis.latch ]
  %input.flat = phi i32 [ 0, %flat.header ], [ %input.flat.next, %axis.latch ]
  %axis.continue = icmp slt i32 %out.axis, %ndim
  br i1 %axis.continue, label %out.stride.entry, label %store

out.stride.entry:
  %out.stride.start = add nsw i32 %out.axis, 1
  br label %out.stride.header

out.stride.header:
  %out.stride.axis = phi i32 [ %out.stride.start, %out.stride.entry ], [ %out.stride.axis.next, %out.stride.body ]
  %out.stride = phi i32 [ 1, %out.stride.entry ], [ %out.stride.next, %out.stride.body ]
  %out.stride.continue = icmp slt i32 %out.stride.axis, %ndim
  br i1 %out.stride.continue, label %out.stride.body, label %coordinate

out.stride.body:
  %out.stride.axis64 = sext i32 %out.stride.axis to i64
  %out.axis.map.ptr = getelementptr inbounds i32, ptr %axes, i64 %out.stride.axis64
  %out.axis.map = load i32, ptr %out.axis.map.ptr, align 4
  %out.axis.map64 = sext i32 %out.axis.map to i64
  %out.extent.ptr = getelementptr inbounds i32, ptr %shape, i64 %out.axis.map64
  %out.extent = load i32, ptr %out.extent.ptr, align 4
  %out.stride.next = mul nsw i32 %out.stride, %out.extent
  %out.stride.axis.next = add nsw i32 %out.stride.axis, 1
  br label %out.stride.header

coordinate:
  %coordinate.value = sdiv i32 %remaining, %out.stride
  %remaining.next = srem i32 %remaining, %out.stride
  %out.axis64 = sext i32 %out.axis to i64
  %input.axis.ptr = getelementptr inbounds i32, ptr %axes, i64 %out.axis64
  %input.axis = load i32, ptr %input.axis.ptr, align 4
  %input.stride.start = add nsw i32 %input.axis, 1
  br label %input.stride.header

input.stride.header:
  %input.stride.axis = phi i32 [ %input.stride.start, %coordinate ], [ %input.stride.axis.next, %input.stride.body ]
  %input.stride = phi i32 [ 1, %coordinate ], [ %input.stride.next, %input.stride.body ]
  %input.stride.continue = icmp slt i32 %input.stride.axis, %ndim
  br i1 %input.stride.continue, label %input.stride.body, label %axis.latch

input.stride.body:
  %input.stride.axis64 = sext i32 %input.stride.axis to i64
  %input.extent.ptr = getelementptr inbounds i32, ptr %shape, i64 %input.stride.axis64
  %input.extent = load i32, ptr %input.extent.ptr, align 4
  %input.stride.next = mul nsw i32 %input.stride, %input.extent
  %input.stride.axis.next = add nsw i32 %input.stride.axis, 1
  br label %input.stride.header

axis.latch:
  %coordinate.offset = mul nsw i32 %coordinate.value, %input.stride
  %input.flat.next = add nsw i32 %input.flat, %coordinate.offset
  %out.axis.next = add nsw i32 %out.axis, 1
  br label %axis.header

store:
  %input.flat64 = sext i32 %input.flat to i64
  %flat64 = sext i32 %flat to i64
  %input.ptr = getelementptr inbounds double, ptr %input, i64 %input.flat64
  %output.ptr = getelementptr inbounds double, ptr %output, i64 %flat64
  %value = load double, ptr %input.ptr, align 8
  store double %value, ptr %output.ptr, align 8
  br label %flat.latch

flat.latch:
  %flat.next = add nsw i32 %flat, 1
  br label %flat.header

exit:
  ret void
}

define void @stack_double(
    ptr %tensors, i32 %num.tensors, ptr %shape, i32 %ndim,
    i32 %dim, ptr %output) {
entry:
  br label %before.header

before.header:
  %before.axis = phi i32 [ 0, %entry ], [ %before.axis.next, %before.body ]
  %before = phi i32 [ 1, %entry ], [ %before.next, %before.body ]
  %before.continue = icmp slt i32 %before.axis, %dim
  br i1 %before.continue, label %before.body, label %after.entry

before.body:
  %before.axis64 = sext i32 %before.axis to i64
  %before.ptr = getelementptr inbounds i32, ptr %shape, i64 %before.axis64
  %before.extent = load i32, ptr %before.ptr, align 4
  %before.next = mul nsw i32 %before, %before.extent
  %before.axis.next = add nsw i32 %before.axis, 1
  br label %before.header

after.entry:
  br label %after.header

after.header:
  %after.axis = phi i32 [ %dim, %after.entry ], [ %after.axis.next, %after.body ]
  %after = phi i32 [ 1, %after.entry ], [ %after.next, %after.body ]
  %after.continue = icmp slt i32 %after.axis, %ndim
  br i1 %after.continue, label %after.body, label %b.header

after.body:
  %after.axis64 = sext i32 %after.axis to i64
  %after.ptr = getelementptr inbounds i32, ptr %shape, i64 %after.axis64
  %after.extent = load i32, ptr %after.ptr, align 4
  %after.next = mul nsw i32 %after, %after.extent
  %after.axis.next = add nsw i32 %after.axis, 1
  br label %after.header

b.header:
  %b = phi i32 [ 0, %after.header ], [ %b.next, %b.latch ]
  %b.continue = icmp slt i32 %b, %before
  br i1 %b.continue, label %tensor.header, label %exit

tensor.header:
  %tensor = phi i32 [ 0, %b.header ], [ %tensor.next, %tensor.latch ]
  %tensor.continue = icmp slt i32 %tensor, %num.tensors
  br i1 %tensor.continue, label %element.header, label %b.latch

element.header:
  %element = phi i32 [ 0, %tensor.header ], [ %element.next, %element.body ]
  %element.continue = icmp slt i32 %element, %after
  br i1 %element.continue, label %element.body, label %tensor.latch

element.body:
  %tensor64 = sext i32 %tensor to i64
  %tensor.ptr.ptr = getelementptr inbounds ptr, ptr %tensors, i64 %tensor64
  %tensor.ptr = load ptr, ptr %tensor.ptr.ptr, align 8
  %source.base = mul nsw i32 %b, %after
  %source.index = add nsw i32 %source.base, %element
  %b.tensor.count = mul nsw i32 %b, %num.tensors
  %output.group = add nsw i32 %b.tensor.count, %tensor
  %output.base = mul nsw i32 %output.group, %after
  %output.index = add nsw i32 %output.base, %element
  %source.index64 = sext i32 %source.index to i64
  %output.index64 = sext i32 %output.index to i64
  %source.ptr = getelementptr inbounds double, ptr %tensor.ptr, i64 %source.index64
  %output.ptr = getelementptr inbounds double, ptr %output, i64 %output.index64
  %value = load double, ptr %source.ptr, align 8
  store double %value, ptr %output.ptr, align 8
  %element.next = add nsw i32 %element, 1
  br label %element.header

tensor.latch:
  %tensor.next = add nsw i32 %tensor, 1
  br label %tensor.header

b.latch:
  %b.next = add nsw i32 %b, 1
  br label %b.header

exit:
  ret void
}

define void @cat_double(
    ptr %tensors, ptr %dim.sizes, i32 %num.tensors, ptr %shape,
    i32 %ndim, i32 %dim, ptr %output) {
entry:
  br label %before.header

before.header:
  %before.axis = phi i32 [ 0, %entry ], [ %before.axis.next, %before.body ]
  %before = phi i32 [ 1, %entry ], [ %before.next, %before.body ]
  %before.continue = icmp slt i32 %before.axis, %dim
  br i1 %before.continue, label %before.body, label %after.entry

before.body:
  %before.axis64 = sext i32 %before.axis to i64
  %before.ptr = getelementptr inbounds i32, ptr %shape, i64 %before.axis64
  %before.extent = load i32, ptr %before.ptr, align 4
  %before.next = mul nsw i32 %before, %before.extent
  %before.axis.next = add nsw i32 %before.axis, 1
  br label %before.header

after.entry:
  %after.start = add nsw i32 %dim, 1
  br label %after.header

after.header:
  %after.axis = phi i32 [ %after.start, %after.entry ], [ %after.axis.next, %after.body ]
  %after = phi i32 [ 1, %after.entry ], [ %after.next, %after.body ]
  %after.continue = icmp slt i32 %after.axis, %ndim
  br i1 %after.continue, label %after.body, label %total.header

after.body:
  %after.axis64 = sext i32 %after.axis to i64
  %after.ptr = getelementptr inbounds i32, ptr %shape, i64 %after.axis64
  %after.extent = load i32, ptr %after.ptr, align 4
  %after.next = mul nsw i32 %after, %after.extent
  %after.axis.next = add nsw i32 %after.axis, 1
  br label %after.header

total.header:
  %total.tensor = phi i32 [ 0, %after.header ], [ %total.tensor.next, %total.body ]
  %total.dim = phi i32 [ 0, %after.header ], [ %total.dim.next, %total.body ]
  %total.continue = icmp slt i32 %total.tensor, %num.tensors
  br i1 %total.continue, label %total.body, label %b.header

total.body:
  %total.tensor64 = sext i32 %total.tensor to i64
  %total.size.ptr = getelementptr inbounds i32, ptr %dim.sizes, i64 %total.tensor64
  %total.size = load i32, ptr %total.size.ptr, align 4
  %total.dim.next = add nsw i32 %total.dim, %total.size
  %total.tensor.next = add nsw i32 %total.tensor, 1
  br label %total.header

b.header:
  %b = phi i32 [ 0, %total.header ], [ %b.next, %b.latch ]
  %b.continue = icmp slt i32 %b, %before
  br i1 %b.continue, label %tensor.header, label %exit

tensor.header:
  %tensor = phi i32 [ 0, %b.header ], [ %tensor.next, %tensor.latch ]
  %destination.offset = phi i32 [ 0, %b.header ], [ %destination.offset.next, %tensor.latch ]
  %tensor.continue = icmp slt i32 %tensor, %num.tensors
  br i1 %tensor.continue, label %tensor.body, label %b.latch

tensor.body:
  %tensor64 = sext i32 %tensor to i64
  %size.ptr = getelementptr inbounds i32, ptr %dim.sizes, i64 %tensor64
  %dim.size = load i32, ptr %size.ptr, align 4
  %copy.count = mul nsw i32 %dim.size, %after
  br label %element.header

element.header:
  %element = phi i32 [ 0, %tensor.body ], [ %element.next, %element.body ]
  %element.continue = icmp slt i32 %element, %copy.count
  br i1 %element.continue, label %element.body, label %tensor.latch

element.body:
  %tensor.ptr.ptr = getelementptr inbounds ptr, ptr %tensors, i64 %tensor64
  %tensor.ptr = load ptr, ptr %tensor.ptr.ptr, align 8
  %source.base = mul nsw i32 %b, %copy.count
  %source.index = add nsw i32 %source.base, %element
  %output.stride = mul nsw i32 %total.dim, %after
  %output.base = mul nsw i32 %b, %output.stride
  %output.with.offset = add nsw i32 %output.base, %destination.offset
  %output.index = add nsw i32 %output.with.offset, %element
  %source.index64 = sext i32 %source.index to i64
  %output.index64 = sext i32 %output.index to i64
  %source.ptr = getelementptr inbounds double, ptr %tensor.ptr, i64 %source.index64
  %output.ptr = getelementptr inbounds double, ptr %output, i64 %output.index64
  %value = load double, ptr %source.ptr, align 8
  store double %value, ptr %output.ptr, align 8
  %element.next = add nsw i32 %element, 1
  br label %element.header

tensor.latch:
  %destination.offset.next = add nsw i32 %destination.offset, %copy.count
  %tensor.next = add nsw i32 %tensor, 1
  br label %tensor.header

b.latch:
  %b.next = add nsw i32 %b, 1
  br label %b.header

exit:
  ret void
}
"""


@dataclass(frozen=True)
class CBackendLLVMSSA:
    """One inspectable C-kernel-to-LLVM-SSA correspondence."""

    c_symbol: str
    llvm_symbol: str
    abstract_tensor_operations: tuple[str, ...]
    role: str

    @property
    def c_source(self) -> str:
        return extract_c_function(self.c_symbol)

    @property
    def llvm_source(self) -> str:
        return extract_llvm_function(self.llvm_symbol)


@dataclass(frozen=True)
class CBackendFunction:
    """One function definition discovered from the real C source."""

    symbol: str
    linkage: str
    role: str
    translated: bool


@dataclass(frozen=True)
class TapeLLVMShortfall:
    result_id: int
    operation: str
    reason: str


@dataclass(frozen=True)
class TapeLLVMModule:
    """Direct LLVM module plus the live Python identities at its ABI."""

    llvm_ir: str
    feed_ids: tuple[int, ...]
    output_ids: Mapping[str, int]
    workspace_sizes: tuple[int, ...]
    shortfalls: tuple[TapeLLVMShortfall, ...]
    trig_solver: str = "lut"
    trig_epsilon: float | None = None

    @property
    def complete(self) -> bool:
        return not self.shortfalls


TRANSLATIONS = (
    CBackendLLVMSSA(
        "fill_double",
        "fill_double",
        ("full", "zeros"),
        "construction fill loop",
    ),
    CBackendLLVMSSA(
        "binary_value",
        "binary_value",
        (
            "add",
            "sub",
            "mul",
            "truediv",
            "pow",
            "mod",
            "floordiv",
            "less",
            "less_equal",
            "greater",
            "greater_equal",
            "equal",
            "not_equal",
            "maximum",
            "minimum",
            "bitand",
            "bitor",
            "bitxor",
            "shl",
            "shr",
            "logical_and",
            "logical_or",
        ),
        "scalar binary semantics",
    ),
    CBackendLLVMSSA(
        "binary_double",
        "binary_double",
        (
            "add",
            "sub",
            "mul",
            "truediv",
            "pow",
            "mod",
            "floordiv",
            "less",
            "less_equal",
            "greater",
            "greater_equal",
            "equal",
            "not_equal",
            "maximum",
            "minimum",
            "bitand",
            "bitor",
            "bitxor",
            "shl",
            "shr",
            "logical_and",
            "logical_or",
        ),
        "elementwise binary loop",
    ),
    CBackendLLVMSSA(
        "binary_scalar_double",
        "binary_scalar_double",
        (
            "add",
            "sub",
            "mul",
            "truediv",
            "pow",
            "mod",
            "floordiv",
            "less",
            "less_equal",
            "greater",
            "greater_equal",
            "equal",
            "not_equal",
            "maximum",
            "minimum",
            "bitand",
            "bitor",
            "bitxor",
            "shl",
            "shr",
            "logical_and",
            "logical_or",
        ),
        "elementwise tensor-scalar loop",
    ),
    CBackendLLVMSSA(
        "unary_double",
        "unary_double",
        (
            "sqrt",
            "exp",
            "log",
            "neg",
            "abs",
            "round",
            "trunc",
            "floor",
            "ceil",
            "isfinite",
            "isnan",
            "isinf",
            "logical_not",
            "tanh",
            "sin",
            "cos",
            "tan",
            "asin",
            "acos",
            "atan",
            "sinh",
            "cosh",
            "asinh",
            "acosh",
            "atanh",
            "sign",
            "invert",
        ),
        "elementwise unary loop and signal-call frontier",
    ),
    CBackendLLVMSSA(
        "where_double",
        "where_double",
        ("where",),
        "elementwise conditional selection",
    ),
    CBackendLLVMSSA(
        "broadcast_double",
        "broadcast_double",
        ("broadcast_to", "expand"),
        "row-major singleton-axis broadcast materialization",
    ),
    CBackendLLVMSSA(
        "reduce_dim_double",
        "reduce_dim_double",
        ("sum", "prod", "min", "max", "any", "all"),
        "arbitrary-rank single-dimension reduction",
    ),
    CBackendLLVMSSA(
        "transpose_double",
        "transpose_double",
        ("permute", "transpose", "swapaxes"),
        "arbitrary-rank axis permutation",
    ),
    CBackendLLVMSSA(
        "cumsum_dim_double",
        "cumsum_dim_double",
        ("cumsum",),
        "single-dimension inclusive prefix sum",
    ),
    CBackendLLVMSSA(
        "stack_double",
        "stack_double",
        ("stack",),
        "inserted-axis tensor materialization",
    ),
    CBackendLLVMSSA(
        "cat_double",
        "cat_double",
        ("cat", "concat", "concatenate"),
        "existing-axis tensor concatenation",
    ),
    CBackendLLVMSSA(
        "pad_double_nd",
        "pad_double_nd",
        ("pad",),
        "arbitrary-rank tensor padding",
    ),
    CBackendLLVMSSA(
        "slice_copy_double",
        "slice_copy_double",
        ("slice",),
        "strided single-axis slice materialization",
    ),
    CBackendLLVMSSA(
        "index_select_double",
        "index_select_double",
        ("slice", "gather"),
        "single-axis indexed selection",
    ),
    CBackendLLVMSSA(
        "index_assign_double",
        "index_assign_double",
        ("scatter",),
        "arbitrary-rank indexed mutation",
    ),
    CBackendLLVMSSA(
        "index_set_double",
        "index_set_double",
        ("index_set",),
        "functional arbitrary-rank indexed assignment",
    ),
    CBackendLLVMSSA(
        "unfold2d_double",
        "unfold2d_double",
        ("unfold2d",),
        "two-dimensional image-to-column transform",
    ),
    CBackendLLVMSSA(
        "fold2d_double",
        "fold2d_double",
        ("fold2d",),
        "two-dimensional overlap-accumulating column-to-image transform",
    ),
    CBackendLLVMSSA(
        "sign_double",
        "sign_double",
        ("sign",),
        "elementwise three-way sign selection",
    ),
    CBackendLLVMSSA(
        "count_true_double",
        "count_true_double",
        ("boolean_mask_select",),
        "boolean-mask output extent",
    ),
    CBackendLLVMSSA(
        "mask_select_double",
        "mask_select_double",
        ("boolean_mask_select",),
        "boolean-mask compaction",
    ),
    CBackendLLVMSSA(
        "increment_mask_double",
        "increment_mask_double",
        ("increment_at_indices",),
        "masked in-place increment",
    ),
    CBackendLLVMSSA(
        "cast_double_to_int_values",
        "cast_double_to_int_values",
        ("int", "long", "long_cast", "to_dtype"),
        "double-storage integer-value cast",
    ),
    CBackendLLVMSSA(
        "cast_double_to_float_values",
        "cast_double_to_float_values",
        ("float", "to_dtype"),
        "double-storage single-precision-value cast",
    ),
    CBackendLLVMSSA(
        "cast_double_to_double_values",
        "cast_double_to_double_values",
        ("double", "to_dtype"),
        "double-storage double-value cast (copying identity)",
    ),
    CBackendLLVMSSA(
        "cast_double_to_bool_values",
        "cast_double_to_bool_values",
        ("bool", "to_dtype"),
        "double-storage boolean-value cast (nonzero -> 1)",
    ),
    CBackendLLVMSSA(
        "sum_double",
        "sum_double",
        ("sum",),
        "flat scalar reduction",
    ),
    CBackendLLVMSSA(
        "create_arange",
        "create_arange",
        ("arange",),
        "arithmetic sequence construction",
    ),
    CBackendLLVMSSA(
        "matmul_double",
        "matmul_double",
        ("matmul",),
        "three-loop matrix multiplication",
    ),
)


def _extract_braced_definition(text: str, match: re.Match[str]) -> str:
    start = match.start()
    opening = text.find("{", match.start(), match.end())
    if opening < 0:
        raise ValueError("definition matcher did not include an opening brace")
    depth = 0
    for index in range(opening, len(text)):
        character = text[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    raise ValueError("unterminated braced definition")


def extract_c_function(symbol: str) -> str:
    """Return the authored C definition for ``symbol``."""

    text = _C_SOURCE_PATH.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"(?ms)^[ \t]*(?:static[ \t]+)?"
        rf"(?:void|double|int)[ \t]+{re.escape(symbol)}[ \t]*"
        rf"\([^;]*?\)[ \t]*\{{"
    )
    match = pattern.search(text)
    if match is None:
        raise KeyError(f"C tensor kernel {symbol!r} has no function definition")
    return _extract_braced_definition(text, match)


def discover_c_backend_functions() -> tuple[CBackendFunction, ...]:
    """Discover and classify every function body in ``ctensor_ops.c``."""

    text = _C_SOURCE_PATH.read_text(encoding="utf-8")
    pattern = re.compile(
        r"(?ms)^[ \t]*(static[ \t]+)?"
        r"(?:void|double|int)[ \t]+([A-Za-z_]\w*)[ \t]*"
        r"\([^;]*?\)[ \t]*\{"
    )
    translated = {entry.c_symbol for entry in TRANSLATIONS}
    functions = []
    for match in pattern.finditer(text):
        symbol = match.group(2)
        try:
            role = _C_FUNCTION_ROLES[symbol]
        except KeyError as error:
            raise ValueError(
                f"C tensor function {symbol!r} is absent from the inventory"
            ) from error
        functions.append(
            CBackendFunction(
                symbol=symbol,
                linkage="private" if match.group(1) else "exported",
                role=role,
                translated=symbol in translated,
            )
        )
    discovered = {function.symbol for function in functions}
    stale = set(_C_FUNCTION_ROLES) - discovered
    if stale:
        raise ValueError(
            f"C tensor inventory names functions absent from source: {sorted(stale)}"
        )
    return tuple(functions)


@lru_cache(maxsize=1)
def _c_backend_operator_codes() -> tuple[dict[str, str], dict[str, str]]:
    """Read the real ``_apply_operator__`` dispatch dictionaries."""

    tree = ast.parse(_C_BACKEND_PATH.read_text(encoding="utf-8"))
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_apply_operator__"
    )
    tables: dict[str, dict[str, str]] = {}
    for node in method.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if (
            not isinstance(target, ast.Name)
            or target.id not in {"binary_codes", "unary_codes"}
            or not isinstance(node.value, ast.Dict)
        ):
            continue
        table = {}
        for key, value in zip(node.value.keys, node.value.values):
            if (
                not isinstance(key, ast.Constant)
                or not isinstance(key.value, str)
                or not isinstance(value, ast.Attribute)
                or not isinstance(value.value, ast.Name)
                or value.value.id != "C"
                or not value.attr.startswith("CT_OP_")
            ):
                raise ValueError(
                    f"{target.id} is not a literal C opcode dictionary"
                )
            table[key.value] = value.attr.removeprefix("CT_OP_")
        tables[target.id] = table
    if set(tables) != {"binary_codes", "unary_codes"}:
        raise ValueError("C backend operator dispatch dictionaries were not found")
    return tables["binary_codes"], tables["unary_codes"]


def _element_count(value: Any) -> int:
    count = 1
    for extent in tuple(getattr(value, "shape", ())):
        count *= int(extent)
    return count


def _llvm_double(value: Any) -> str:
    bits = struct.unpack(">Q", struct.pack(">d", float(value)))[0]
    return f"0x{bits:016X}"


def _flatten_host_values(value: Any):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _flatten_host_values(item)
    else:
        yield value


# Span-memory initialisation constructors and their implicit fill scalar.
# ``None`` requires an explicit ``fill_value`` parameter (``full``/``fill``).
_TAPE_FILL_DEFAULTS: Mapping[str, float | None] = {
    "fill": None,
    "zeros": 0.0,
    "zeros_like": 0.0,
    "empty": 0.0,
    "empty_like": 0.0,
    "ones": 1.0,
    "ones_like": 1.0,
    "full": None,
    "full_like": None,
}


def lower_abstract_tensor_tape_to_llvm_ssa(
    tape: Any,
    outputs: Mapping[str, Any] | Any,
    *,
    function_name: str = "abstract_tensor_tape",
    trig_solver: str = "lut",
    trig_epsilon: float | None = None,
) -> TapeLLVMModule:
    """Lower a recorded AbstractTensor tape directly to real C-kernel LLVM.

    This first direct path covers the C backend's actual elementwise dispatch,
    rank-two ``matmul_double``, and recorded ``tensor_from_list`` constants.
    It emits calls to functions present in :data:`TRANSLATIONS`; it never emits
    an unqualified or invented fallback call.
    """

    if not isinstance(outputs, Mapping):
        outputs = {"result": outputs}
    outputs = dict(outputs)
    if not outputs:
        raise ValueError("direct tape lowering requires at least one output")
    nodes = dict(getattr(tape, "_nodes", {}))
    if not nodes:
        raise ValueError("direct tape lowering requires a recorded tape")

    required: set[int] = set()

    def require(value: Any) -> None:
        identity = id(value)
        if identity in required:
            return
        node = nodes.get(identity)
        if node is None:
            return
        required.add(identity)
        for operand in node.ctx.get("inputs", ()):
            require(operand)

    for output in outputs.values():
        require(output)

    live_nodes = [
        node
        for result_id, node in nodes.items()
        if result_id in required
    ]
    produced_ids = {
        id(node.ctx.get("result"))
        for node in live_nodes
    }
    feeds: list[Any] = []
    feed_seen: set[int] = set()
    for node in live_nodes:
        for operand in node.ctx.get("inputs", ()):
            identity = id(operand)
            if (
                identity not in produced_ids
                and identity not in feed_seen
                and hasattr(operand, "shape")
            ):
                feeds.append(operand)
                feed_seen.add(identity)

    arguments = [
        *(f"ptr %feed{index}" for index in range(len(feeds))),
        *(f"ptr %output{index}" for index in range(len(outputs))),
    ]
    feed_pointer = {
        id(value): f"%feed{index}" for index, value in enumerate(feeds)
    }
    output_pointer = {
        id(value): f"%output{index}"
        for index, value in enumerate(outputs.values())
    }
    globals_: list[str] = []
    pointers: dict[int, str] = dict(feed_pointer)
    entry: list[str] = []
    workspace_sizes: list[int] = []
    shortfalls: list[TapeLLVMShortfall] = []

    for constant_index, node in enumerate(
        node for node in live_nodes if str(node.op) == "tensor_from_list"
    ):
        result = node.ctx["result"]
        identity = id(result)
        values = tuple(_flatten_host_values(node.ctx["params"]["data"]))
        count = len(values)
        elements = ", ".join(
            f"double {_llvm_double(value)}" for value in values
        )
        symbol = f"@tape.constant.{constant_index}"
        globals_.append(
            f"{symbol} = private unnamed_addr constant "
            f"[{count} x double] [{elements}], align 8"
        )
        pointers[identity] = symbol

    from ..fused_ir import ELEMENTWISE_ALIASES

    for node_index, node in enumerate(live_nodes):
        operation = ELEMENTWISE_ALIASES.get(str(node.op), str(node.op))
        result = node.ctx["result"]
        result_id = id(result)
        if operation == "tensor_from_list":
            continue
        if result_id in output_pointer:
            destination = output_pointer[result_id]
        else:
            count = _element_count(result)
            destination = f"%workspace{len(workspace_sizes)}"
            workspace_sizes.append(count)
            arguments.append(f"ptr {destination}")
        pointers[result_id] = destination

    binary_codes, unary_codes = _c_backend_operator_codes()
    opcode_index = {
        name: index for index, name in enumerate(C_TENSOR_OPCODE_ORDER)
    }
    translated_symbols = {entry.c_symbol for entry in TRANSLATIONS}

    for node_index, node in enumerate(live_nodes):
        operation = ELEMENTWISE_ALIASES.get(str(node.op), str(node.op))
        result = node.ctx["result"]
        result_id = id(result)
        if operation == "tensor_from_list":
            continue
        operands = tuple(node.ctx.get("inputs", ()))
        missing = [
            id(operand) for operand in operands if id(operand) not in pointers
        ]
        if missing:
            shortfalls.append(
                TapeLLVMShortfall(
                    result_id,
                    operation,
                    f"operands have no LLVM storage: {missing}",
                )
            )
            continue
        destination = pointers[result_id]
        result_count = _element_count(result)

        if operation in _TAPE_FILL_DEFAULTS and "fill_double" in translated_symbols:
            parameters = dict(node.ctx.get("params") or {})
            default_value = _TAPE_FILL_DEFAULTS[operation]
            fill_value = parameters.get(
                "fill_value", parameters.get("value", default_value)
            )
            if fill_value is None:
                shortfalls.append(
                    TapeLLVMShortfall(
                        result_id,
                        operation,
                        "span initialisation requires an explicit fill_value",
                    )
                )
                continue
            fill_value = float(fill_value)
            if fill_value == 0.0:
                # Zero-fill is the calloc case: a bytewise memset is the
                # definite lowering for zero-initialised span memory.
                byte_count = result_count * 8
                entry.append(
                    f"  call void @llvm.memset.p0.i64(ptr {destination}, i8 0, "
                    f"i64 {byte_count}, i1 false)"
                )
            else:
                entry.append(
                    f"  call void @fill_double(ptr {destination}, "
                    f"double {_llvm_double(fill_value)}, i32 {result_count})"
                )
            continue

        if operation == "arange" and "create_arange" in translated_symbols:
            parameters = dict(node.ctx.get("params") or {})
            start = parameters.get("start", 0)
            step = parameters.get("step", 1)
            entry.append(
                f"  call void @create_arange("
                f"double {_llvm_double(start)}, double {_llvm_double(step)}, "
                f"i32 {result_count}, ptr {destination})"
            )
            continue

        if operation == "sum" and {
            "sum_double",
            "reduce_dim_double",
        } <= translated_symbols:
            parameters = dict(node.ctx.get("params") or {})
            if len(operands) != 1:
                shortfalls.append(
                    TapeLLVMShortfall(
                        result_id,
                        operation,
                        "sum lowering requires one tensor operand",
                    )
                )
                continue
            axis = parameters.get("axis")
            if axis is not None:
                source = operands[0]
                shape = tuple(int(size) for size in source.shape)
                dim = int(axis) % len(shape)
                symbol = f"@tape.shape.{node_index}"
                globals_.append(
                    f"{symbol} = private constant [{len(shape)} x i32] ["
                    + ", ".join(f"i32 {size}" for size in shape)
                    + "]"
                )
                entry.append(
                    f"  call void @reduce_dim_double("
                    f"ptr {pointers[id(source)]}, ptr {destination}, "
                    f"ptr {symbol}, i32 {len(shape)}, i32 {dim}, i32 0)"
                )
                continue
            if result_count != 1:
                shortfalls.append(
                    TapeLLVMShortfall(
                        result_id,
                        operation,
                        "flat sum must produce one value",
                    )
                )
                continue
            scalar = f"%sum{node_index}"
            entry.append(
                f"  {scalar} = call double @sum_double("
                f"ptr {pointers[id(operands[0])]}, "
                f"i32 {_element_count(operands[0])})"
            )
            entry.append(
                f"  store double {scalar}, ptr {destination}, align 8"
            )
            continue

        if operation in {"reshape", "view"}:
            if len(operands) != 1 or _element_count(operands[0]) != result_count:
                shortfalls.append(
                    TapeLLVMShortfall(
                        result_id,
                        operation,
                        "reshape/view is not a compatible zero-copy layout",
                    )
                )
                continue
            source_pointer = pointers[id(operands[0])]
            if result_id in output_pointer:
                entry.append(
                    f"  call void @llvm.memcpy.p0.p0.i64("
                    f"ptr {destination}, ptr {source_pointer}, "
                    f"i64 {result_count * 8}, i1 false)"
                )
            else:
                pointers[result_id] = source_pointer
            continue

        if operation == "permute" and "transpose_double" in translated_symbols:
            if len(operands) != 1:
                shortfalls.append(
                    TapeLLVMShortfall(
                        result_id, operation, "permute requires one operand"
                    )
                )
                continue
            source = operands[0]
            shape = tuple(int(size) for size in source.shape)
            parameters = dict(node.ctx.get("params") or {})
            axes = tuple(
                int(axis)
                for axis in parameters.get("perm", parameters.get("dims", ()))
            )
            if len(axes) != len(shape):
                shortfalls.append(
                    TapeLLVMShortfall(
                        result_id, operation, "permute axes do not match rank"
                    )
                )
                continue
            shape_symbol = f"@tape.shape.{node_index}"
            axes_symbol = f"@tape.axes.{node_index}"
            globals_.extend(
                (
                    f"{shape_symbol} = private constant [{len(shape)} x i32] ["
                    + ", ".join(f"i32 {size}" for size in shape)
                    + "]",
                    f"{axes_symbol} = private constant [{len(axes)} x i32] ["
                    + ", ".join(f"i32 {axis}" for axis in axes)
                    + "]",
                )
            )
            entry.append(
                f"  call void @transpose_double("
                f"ptr {pointers[id(source)]}, ptr {destination}, "
                f"ptr {shape_symbol}, ptr {axes_symbol}, i32 {len(shape)})"
            )
            continue

        if operation == "cumsum" and "cumsum_dim_double" in translated_symbols:
            if len(operands) != 1:
                shortfalls.append(
                    TapeLLVMShortfall(
                        result_id, operation, "cumsum requires one operand"
                    )
                )
                continue
            source = operands[0]
            shape = tuple(int(size) for size in source.shape)
            dim = int((node.ctx.get("params") or {}).get("dim", 0)) % len(shape)
            shape_symbol = f"@tape.shape.{node_index}"
            globals_.append(
                f"{shape_symbol} = private constant [{len(shape)} x i32] ["
                + ", ".join(f"i32 {size}" for size in shape)
                + "]"
            )
            entry.append(
                f"  call void @cumsum_dim_double("
                f"ptr {pointers[id(source)]}, ptr {destination}, "
                f"ptr {shape_symbol}, i32 {len(shape)}, i32 {dim})"
            )
            continue

        if operation == "stack" and "stack_double" in translated_symbols:
            if not operands:
                shortfalls.append(
                    TapeLLVMShortfall(
                        result_id, operation, "stack requires tensor operands"
                    )
                )
                continue
            shape = tuple(int(size) for size in operands[0].shape)
            dim = int((node.ctx.get("params") or {}).get("dim", 0)) % (
                len(shape) + 1
            )
            shape_symbol = f"@tape.shape.{node_index}"
            globals_.append(
                f"{shape_symbol} = private constant [{len(shape)} x i32] ["
                + ", ".join(f"i32 {size}" for size in shape)
                + "]"
            )
            array = f"%stack.inputs.{node_index}"
            entry.append(
                f"  {array} = alloca [{len(operands)} x ptr], align 8"
            )
            for input_index, operand in enumerate(operands):
                slot = f"%stack.slot.{node_index}.{input_index}"
                entry.extend(
                    (
                        f"  {slot} = getelementptr inbounds "
                        f"[{len(operands)} x ptr], ptr {array}, "
                        f"i64 0, i64 {input_index}",
                        f"  store ptr {pointers[id(operand)]}, ptr {slot}, align 8",
                    )
                )
            entry.append(
                f"  call void @stack_double("
                f"ptr {array}, i32 {len(operands)}, ptr {shape_symbol}, "
                f"i32 {len(shape)}, i32 {dim}, ptr {destination})"
            )
            continue

        if operation in {"cat", "concat"} and "cat_double" in translated_symbols:
            if not operands:
                shortfalls.append(
                    TapeLLVMShortfall(
                        result_id, operation, "cat requires tensor operands"
                    )
                )
                continue
            shape = tuple(int(size) for size in operands[0].shape)
            dim = int((node.ctx.get("params") or {}).get("dim", 0)) % len(shape)
            sizes = tuple(int(value.shape[dim]) for value in operands)
            shape_symbol = f"@tape.shape.{node_index}"
            sizes_symbol = f"@tape.sizes.{node_index}"
            globals_.extend(
                (
                    f"{shape_symbol} = private constant [{len(shape)} x i32] ["
                    + ", ".join(f"i32 {size}" for size in shape)
                    + "]",
                    f"{sizes_symbol} = private constant [{len(sizes)} x i32] ["
                    + ", ".join(f"i32 {size}" for size in sizes)
                    + "]",
                )
            )
            array = f"%cat.inputs.{node_index}"
            entry.append(
                f"  {array} = alloca [{len(operands)} x ptr], align 8"
            )
            for input_index, operand in enumerate(operands):
                slot = f"%cat.slot.{node_index}.{input_index}"
                entry.extend(
                    (
                        f"  {slot} = getelementptr inbounds "
                        f"[{len(operands)} x ptr], ptr {array}, "
                        f"i64 0, i64 {input_index}",
                        f"  store ptr {pointers[id(operand)]}, ptr {slot}, align 8",
                    )
                )
            entry.append(
                f"  call void @cat_double("
                f"ptr {array}, ptr {sizes_symbol}, i32 {len(operands)}, "
                f"ptr {shape_symbol}, i32 {len(shape)}, i32 {dim}, "
                f"ptr {destination})"
            )
            continue

        if operation == "where" and "where_double" in translated_symbols:
            if len(operands) != 3 or any(
                _element_count(operand) != result_count for operand in operands
            ):
                shortfalls.append(
                    TapeLLVMShortfall(
                        result_id,
                        operation,
                        "where_double direct binding requires three equal extents",
                    )
                )
                continue
            condition, if_true, if_false = operands
            entry.append(
                f"  call void @where_double("
                f"ptr {pointers[id(condition)]}, "
                f"ptr {pointers[id(if_true)]}, "
                f"ptr {pointers[id(if_false)]}, "
                f"ptr {destination}, i32 {result_count})"
            )
            continue

        if operation in unary_codes and "unary_double" in translated_symbols:
            if len(operands) != 1:
                shortfalls.append(
                    TapeLLVMShortfall(
                        result_id, operation, "unary C operation has wrong arity"
                    )
                )
                continue
            opcode = opcode_index[unary_codes[operation]]
            entry.append(
                f"  call void @unary_double(ptr {pointers[id(operands[0])]}, "
                f"ptr {destination}, i32 {result_count}, i32 {opcode})"
            )
            continue

        if operation in binary_codes and {
            "binary_double",
            "binary_scalar_double",
        } <= translated_symbols:
            if len(operands) != 2:
                shortfalls.append(
                    TapeLLVMShortfall(
                        result_id, operation, "binary C operation has wrong arity"
                    )
                )
                continue
            left, right = operands
            left_count = _element_count(left)
            right_count = _element_count(right)
            opcode = opcode_index[binary_codes[operation]]
            if left_count == result_count and right_count == result_count:
                entry.append(
                    f"  call void @binary_double("
                    f"ptr {pointers[id(left)]}, ptr {pointers[id(right)]}, "
                    f"ptr {destination}, i32 {result_count}, i32 {opcode})"
                )
            elif left_count == result_count and right_count == 1:
                scalar = f"%scalar{node_index}"
                entry.append(
                    f"  {scalar} = load double, ptr {pointers[id(right)]}, align 8"
                )
                entry.append(
                    f"  call void @binary_scalar_double("
                    f"ptr {pointers[id(left)]}, double {scalar}, "
                    f"ptr {destination}, i32 {result_count}, i32 {opcode}, i32 0)"
                )
            elif left_count == 1 and right_count == result_count:
                scalar = f"%scalar{node_index}"
                entry.append(
                    f"  {scalar} = load double, ptr {pointers[id(left)]}, align 8"
                )
                entry.append(
                    f"  call void @binary_scalar_double("
                    f"ptr {pointers[id(right)]}, double {scalar}, "
                    f"ptr {destination}, i32 {result_count}, i32 {opcode}, i32 1)"
                )
            else:
                shortfalls.append(
                    TapeLLVMShortfall(
                        result_id,
                        operation,
                        "C elementwise kernel only supports equal extents or "
                        "one-element scalar broadcasting",
                    )
                )
            continue

        if operation in {"matmul", "rmatmul", "imatmul"}:
            if "matmul_double" not in translated_symbols or len(operands) != 2:
                shortfalls.append(
                    TapeLLVMShortfall(
                        result_id, operation, "matmul C translation unavailable"
                    )
                )
                continue
            left, right = operands
            left_shape = tuple(left.shape)
            right_shape = tuple(right.shape)
            if len(left_shape) != 2 or len(right_shape) != 2:
                shortfalls.append(
                    TapeLLVMShortfall(
                        result_id,
                        operation,
                        "matmul_double direct lowering currently requires rank two",
                    )
                )
                continue
            if operation == "rmatmul":
                left, right = right, left
                left_shape, right_shape = right_shape, left_shape
            m, n = left_shape
            n2, p = right_shape
            if n != n2:
                shortfalls.append(
                    TapeLLVMShortfall(
                        result_id, operation, "matmul dimensions do not agree"
                    )
                )
                continue
            entry.append(
                f"  call void @matmul_double("
                f"ptr {pointers[id(left)]}, ptr {pointers[id(right)]}, "
                f"ptr {destination}, i32 {m}, i32 {n}, i32 {p})"
            )
            continue

        shortfalls.append(
            TapeLLVMShortfall(
                result_id,
                operation,
                "operation has no direct binding to a translated C function",
            )
        )

    wrapper = [
        *globals_,
        "",
        f"define void @{function_name}({', '.join(arguments)}) {{",
        "entry:",
        *entry,
        "  ret void",
        "}",
    ]
    llvm_ir = LLVM_SSA_MODULE.rstrip() + "\n\n" + "\n".join(wrapper) + "\n"
    selected_epsilon = None
    if trig_solver:
        from .llvm_signal_math import link_llvm_trig_solver

        llvm_ir, signal = link_llvm_trig_solver(
            llvm_ir,
            trig_solver,
            epsilon=trig_epsilon,
        )
        assert signal is not None
        selected_epsilon = signal.epsilon
    if not shortfalls:
        from llvmlite import binding as llvm

        module = llvm.parse_assembly(llvm_ir)
        module.verify()
    return TapeLLVMModule(
        llvm_ir=llvm_ir,
        feed_ids=tuple(id(value) for value in feeds),
        output_ids={name: id(value) for name, value in outputs.items()},
        workspace_sizes=tuple(workspace_sizes),
        shortfalls=tuple(shortfalls),
        trig_solver=str(trig_solver),
        trig_epsilon=selected_epsilon,
    )


def extract_llvm_function(symbol: str) -> str:
    """Return the handwritten LLVM definition for ``symbol``."""

    pattern = re.compile(
        rf"(?ms)^define(?:[ \t]+internal)?[ \t]+"
        rf"(?:void|double|i32)[ \t]+@{re.escape(symbol)}"
        rf"\([^{{]*\)[ \t]*\{{"
    )
    match = pattern.search(LLVM_SSA_MODULE)
    if match is None:
        raise KeyError(f"LLVM SSA symbol {symbol!r} has no function definition")
    return _extract_braced_definition(LLVM_SSA_MODULE, match)


def extract_llvm_declaration(symbol: str) -> str:
    """Return the canonical external declaration for ``symbol``."""

    pattern = re.compile(
        rf"(?m)^declare[^\n@]*@{re.escape(symbol)}\([^\n]*$"
    )
    match = pattern.search(LLVM_SSA_MODULE)
    if match is None:
        raise KeyError(f"LLVM SSA symbol {symbol!r} has no declaration")
    return match.group(0)


def _header_opcode_order() -> tuple[str, ...]:
    text = _C_HEADER_PATH.read_text(encoding="utf-8")
    match = re.search(
        r"(?ms)typedef[ \t]+enum[ \t]+CTensorOp[ \t]*\{(.*?)\}[ \t]*CTensorOp",
        text,
    )
    if match is None:
        raise ValueError("ctensor_ops.h does not define CTensorOp")
    names = re.findall(r"\bCT_OP_([A-Z0-9_]+)\b", match.group(1))
    return tuple(name for name in names if name != "COUNT")


def validate_c_opcode_alignment() -> None:
    """Reject handwritten SSA when its switch numbers drift from C."""

    actual = _header_opcode_order()
    if actual != C_TENSOR_OPCODE_ORDER:
        raise ValueError(
            "handwritten LLVM opcode order differs from CTensorOp: "
            f"llvm={C_TENSOR_OPCODE_ORDER!r}, c={actual!r}"
        )


def verify_llvm_ssa() -> None:
    """Parse and verify the handwritten module with LLVM itself."""

    from llvmlite import binding as llvm

    module = llvm.parse_assembly(LLVM_SSA_MODULE)
    module.verify()


def translations_for_operation(operation: str) -> tuple[CBackendLLVMSSA, ...]:
    """Return every C/LLVM layer participating in ``operation``."""

    return tuple(
        translation
        for translation in TRANSLATIONS
        if operation in translation.abstract_tensor_operations
    )


def covered_operations() -> frozenset[str]:
    """Operations with an authored C-kernel/LLVM-SSA correspondence."""

    return frozenset(
        operation
        for translation in TRANSLATIONS
        for operation in translation.abstract_tensor_operations
    )


@lru_cache(maxsize=1)
def c_backend_repository_ssa_reference():
    """Return the C computational core as one repository-SSA code reference.

    The C functions and handwritten LLVM are provenance.  The returned object
    contains only Turing repository SSA and canonical operation→entrypoint
    references; it cannot execute or dispatch tensor operations.
    """

    from ....transmogrifier.tensor_ssa_reference import (
        SSATensorCodeReference,
        SSATensorOperationReference,
    )
    from .llvm_repository_ssa import import_llvm_to_repository_ssa

    imported = import_llvm_to_repository_ssa(
        LLVM_SSA_MODULE,
        include_declarations=False,
    )
    if not imported.complete:
        raise ValueError(imported.shortfall_report())
    for symbol, output_index in C_SSA_OUTPUT_ARGUMENTS.items():
        function = imported.module.functions[symbol]
        output = function.args[int(output_index)]
        function.metadata["named_outputs"] = (("out", int(output.id)),)
        function.metadata["ssa_output_argument"] = int(output_index)
    for symbol, argument_indices in C_SSA_I32_POINTER_ARGUMENTS.items():
        function = imported.module.functions[symbol]
        for argument_index in argument_indices:
            function.args[int(argument_index)].dtype = "int32"
    entrypoints_by_op: dict[str, list[str]] = {}
    for translation in TRANSLATIONS:
        for operation in translation.abstract_tensor_operations:
            entrypoints_by_op.setdefault(str(operation), []).append(
                str(translation.llvm_symbol)
            )
    operations = {
        operation: SSATensorOperationReference(
            operation,
            tuple(dict.fromkeys(entrypoints)),
        )
        for operation, entrypoints in entrypoints_by_op.items()
    }
    return SSATensorCodeReference(
        "c-backend-computational-core",
        imported.module,
        operations,
        "c_backend_llvm_ssa.LLVM_SSA_MODULE",
        C_SSA_EXTERNAL_PRIMITIVES,
    )


def c_dispatch_operations() -> frozenset[str]:
    """Operations in the real C backend's scalar opcode dictionaries.

    The dictionaries inside ``CAbstractTensor._apply_operator__`` are the
    source of truth.  Reading them through the existing AST inventory keeps
    compiler capability reports from importing/initializing the C runtime or
    maintaining another copy of the finite operator list.
    """

    binary, unary = _c_backend_operator_codes()
    return frozenset(binary) | frozenset(unary)


def c_tensor_opcode(operation: str) -> tuple[str, int] | None:
    """Return ``(kind, fixed opcode)`` for one scalar C tensor operation."""

    binary, unary = _c_backend_operator_codes()
    name = str(operation)
    if name in binary:
        opcode_name = binary[name]
        kind = "binary"
    elif name in unary:
        opcode_name = unary[name]
        kind = "unary"
    else:
        return None
    return kind, C_TENSOR_OPCODE_ORDER.index(opcode_name)


def validate_translation_table(
    translations: Iterable[CBackendLLVMSSA] = TRANSLATIONS,
) -> None:
    """Validate source presence, LLVM presence, opcode alignment, and IR."""

    translations = tuple(translations)
    if not translations:
        raise ValueError("the C-to-LLVM SSA table must not be empty")
    functions = discover_c_backend_functions()
    discovered_symbols = {function.symbol for function in functions}
    seen_pairs: set[tuple[str, str]] = set()
    for translation in translations:
        pair = (translation.c_symbol, translation.llvm_symbol)
        if pair in seen_pairs:
            raise ValueError(f"duplicate C/LLVM translation pair: {pair!r}")
        seen_pairs.add(pair)
        if not translation.abstract_tensor_operations:
            raise ValueError(f"{translation.c_symbol} has no operation mapping")
        if translation.c_symbol not in discovered_symbols:
            raise ValueError(
                f"translation names nonexistent C symbol {translation.c_symbol!r}"
            )
        if translation.llvm_symbol != translation.c_symbol:
            raise ValueError(
                "C and LLVM symbols must remain identical: "
                f"{translation.c_symbol!r} != {translation.llvm_symbol!r}"
            )
        translation.c_source
        translation.llvm_source
    validate_c_opcode_alignment()
    verify_llvm_ssa()


__all__ = [
    "CBackendLLVMSSA",
    "CBackendFunction",
    "TapeLLVMModule",
    "TapeLLVMShortfall",
    "C_TENSOR_OPCODE_ORDER",
    "C_SSA_OUTPUT_ARGUMENTS",
    "C_SSA_I32_POINTER_ARGUMENTS",
    "C_SSA_EXTERNAL_PRIMITIVES",
    "LLVM_SSA_MODULE",
    "PRECOMPILE_INTERNAL_OPERATORS",
    "TRANSLATIONS",
    "c_dispatch_operations",
    "c_tensor_opcode",
    "covered_operations",
    "c_backend_repository_ssa_reference",
    "discover_c_backend_functions",
    "extract_c_function",
    "extract_llvm_declaration",
    "extract_llvm_function",
    "lower_abstract_tensor_tape_to_llvm_ssa",
    "translations_for_operation",
    "validate_c_opcode_alignment",
    "validate_translation_table",
    "verify_llvm_ssa",
]
