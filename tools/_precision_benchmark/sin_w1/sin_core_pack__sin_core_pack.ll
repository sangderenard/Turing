source_filename = "turing.ssa-llvm.sin_core_pack__sin_core_pack"

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-windows-gnu"

define void @__ssa_sin_core_pack__sin_core_pack__planned_region_0(ptr noalias %arg.0, ptr %arg.1, ptr noalias %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr %arg.11) {
entry:
  %value.14 = alloca double, i64 1, align 8
  %value.15 = alloca double, i64 1, align 8
  %value.16 = alloca double, i64 1, align 8
  %value.17 = alloca double, i64 1, align 8
  %value.18 = alloca double, i64 1, align 8
  %value.19 = alloca double, i64 1, align 8
  %value.20 = alloca double, i64 1, align 8
  %value.21 = alloca double, i64 1, align 8
  %value.22 = alloca double, i64 1, align 8
  %value.23 = alloca double, i64 1, align 8
  %value.24 = alloca double, i64 1, align 8
  %value.25 = alloca double, i64 1, align 8
  %value.26 = alloca double, i64 1, align 8
  %value.27 = alloca double, i64 1, align 8
  %value.28 = alloca double, i64 1, align 8
  %value.29 = alloca double, i64 1, align 8
  %value.30 = alloca double, i64 1, align 8
  %value.31 = alloca double, i64 1, align 8
  %value.32 = alloca double, i64 1, align 8
  %load.0.40.0 = load i32, ptr %arg.1, align 4
  %address.0.40 = getelementptr double, ptr %arg.0, i32 %load.0.40.0
  %pinned.load.1.14 = load double, ptr %address.0.40, align 8
  store double %pinned.load.1.14, ptr %value.14, align 8
  %load.2.15.0 = load double, ptr %value.14, align 8
  %scalar.2.15 = fmul double %load.2.15.0, %load.2.15.0
  store double %scalar.2.15, ptr %value.15, align 8
  %load.3.16.0 = load double, ptr %arg.2, align 8
  %scalar.3.16 = fmul double %load.3.16.0, %scalar.2.15
  store double %scalar.3.16, ptr %value.16, align 8
  %load.4.17.0 = load double, ptr %arg.3, align 8
  %scalar.4.17 = fadd double %load.4.17.0, %scalar.3.16
  store double %scalar.4.17, ptr %value.17, align 8
  %scalar.5.18 = fmul double %scalar.2.15, %scalar.4.17
  store double %scalar.5.18, ptr %value.18, align 8
  %load.6.19.0 = load double, ptr %arg.4, align 8
  %scalar.6.19 = fadd double %load.6.19.0, %scalar.5.18
  store double %scalar.6.19, ptr %value.19, align 8
  %scalar.7.20 = fmul double %scalar.2.15, %scalar.6.19
  store double %scalar.7.20, ptr %value.20, align 8
  %load.8.21.0 = load double, ptr %arg.5, align 8
  %scalar.8.21 = fadd double %load.8.21.0, %scalar.7.20
  store double %scalar.8.21, ptr %value.21, align 8
  %scalar.9.22 = fmul double %scalar.2.15, %scalar.8.21
  store double %scalar.9.22, ptr %value.22, align 8
  %load.10.23.0 = load double, ptr %arg.6, align 8
  %scalar.10.23 = fadd double %load.10.23.0, %scalar.9.22
  store double %scalar.10.23, ptr %value.23, align 8
  %scalar.11.24 = fmul double %scalar.2.15, %scalar.10.23
  store double %scalar.11.24, ptr %value.24, align 8
  %load.12.25.0 = load double, ptr %arg.7, align 8
  %scalar.12.25 = fadd double %load.12.25.0, %scalar.11.24
  store double %scalar.12.25, ptr %value.25, align 8
  %scalar.13.26 = fmul double %scalar.2.15, %scalar.12.25
  store double %scalar.13.26, ptr %value.26, align 8
  %load.14.27.0 = load double, ptr %arg.8, align 8
  %scalar.14.27 = fadd double %load.14.27.0, %scalar.13.26
  store double %scalar.14.27, ptr %value.27, align 8
  %scalar.15.28 = fmul double %scalar.2.15, %scalar.14.27
  store double %scalar.15.28, ptr %value.28, align 8
  %load.16.29.0 = load double, ptr %arg.9, align 8
  %scalar.16.29 = fadd double %load.16.29.0, %scalar.15.28
  store double %scalar.16.29, ptr %value.29, align 8
  %scalar.17.30 = fmul double %scalar.2.15, %scalar.16.29
  store double %scalar.17.30, ptr %value.30, align 8
  %load.18.31.0 = load double, ptr %arg.10, align 8
  %scalar.18.31 = fadd double %load.18.31.0, %scalar.17.30
  store double %scalar.18.31, ptr %value.31, align 8
  %scalar.19.32 = fmul double %load.2.15.0, %scalar.18.31
  store double %scalar.19.32, ptr %value.32, align 8
  %address.20.41 = getelementptr double, ptr %arg.11, i32 %load.0.40.0
  store double %scalar.19.32, ptr %address.20.41, align 8
  ret void
}

define void @__ssa_sin_core_pack__sin_core_pack(ptr noalias %arg.0, ptr noalias %arg.1, ptr %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr noalias %arg.11, ptr %out.0) {
entry:
  %value.35 = alloca i64, i64 1, align 8
  %value.36 = alloca i64, i64 1, align 8
  %value.38 = alloca i64, i64 1, align 8
  %value.39 = alloca i1, i64 1, align 8
  store i64 0, ptr %value.35, align 8
  store i64 1, ptr %value.36, align 8
  br label %loop_header
loop_header:
  %phi.37 = phi ptr [ %value.35, %entry ], [ %value.38, %loop_latch ]
  %load.4.39.0 = load i32, ptr %phi.37, align 4
  %load.4.39.1 = load i32, ptr %arg.0, align 4
  %scalar.4.39 = icmp slt i32 %load.4.39.0, %load.4.39.1
  store i1 %scalar.4.39, ptr %value.39, align 1
  br i1 %scalar.4.39, label %loop_body, label %loop_exit
loop_body:
  call void @__ssa_sin_core_pack__sin_core_pack__planned_region_0(ptr %arg.1, ptr %phi.37, ptr %arg.11, ptr %arg.10, ptr %arg.9, ptr %arg.8, ptr %arg.7, ptr %arg.6, ptr %arg.5, ptr %arg.4, ptr %arg.3, ptr %arg.2)
  br label %loop_latch
loop_latch:
  %load.8.38.0 = load i32, ptr %phi.37, align 4
  %load.8.38.1 = load i64, ptr %value.36, align 8
  %convert.8.38.1 = trunc i64 %load.8.38.1 to i32
  %scalar.8.38 = add i32 %load.8.38.0, %convert.8.38.1
  %declared.8.38 = sext i32 %scalar.8.38 to i64
  store i64 %declared.8.38, ptr %value.38, align 8
  br label %loop_header
loop_exit:
  %return.load.0.23 = load double, ptr %arg.2, align 8
  store double %return.load.0.23, ptr %out.0, align 8
  ret void
}

define void @sin_core_pack__sin_core_pack(ptr %buffers, ptr %extents) {
entry:
  %public.addr.0 = getelementptr ptr, ptr %buffers, i64 0
  %public.0 = load ptr, ptr %public.addr.0, align 8
  %public.addr.1 = getelementptr ptr, ptr %buffers, i64 1
  %public.1 = load ptr, ptr %public.addr.1, align 8
  %public.addr.2 = getelementptr ptr, ptr %buffers, i64 2
  %public.2 = load ptr, ptr %public.addr.2, align 8
  %public.addr.3 = getelementptr ptr, ptr %buffers, i64 3
  %public.3 = load ptr, ptr %public.addr.3, align 8
  %public.addr.4 = getelementptr ptr, ptr %buffers, i64 4
  %public.4 = load ptr, ptr %public.addr.4, align 8
  %public.addr.5 = getelementptr ptr, ptr %buffers, i64 5
  %public.5 = load ptr, ptr %public.addr.5, align 8
  %public.addr.6 = getelementptr ptr, ptr %buffers, i64 6
  %public.6 = load ptr, ptr %public.addr.6, align 8
  %public.addr.7 = getelementptr ptr, ptr %buffers, i64 7
  %public.7 = load ptr, ptr %public.addr.7, align 8
  %public.addr.8 = getelementptr ptr, ptr %buffers, i64 8
  %public.8 = load ptr, ptr %public.addr.8, align 8
  %public.addr.9 = getelementptr ptr, ptr %buffers, i64 9
  %public.9 = load ptr, ptr %public.addr.9, align 8
  %public.addr.10 = getelementptr ptr, ptr %buffers, i64 10
  %public.10 = load ptr, ptr %public.addr.10, align 8
  %public.addr.11 = getelementptr ptr, ptr %buffers, i64 11
  %public.11 = load ptr, ptr %public.addr.11, align 8
  call void @__ssa_sin_core_pack__sin_core_pack(ptr %public.0, ptr %public.1, ptr %public.2, ptr %public.3, ptr %public.4, ptr %public.5, ptr %public.6, ptr %public.7, ptr %public.8, ptr %public.9, ptr %public.10, ptr %public.11, ptr %public.2)
  ret void
}
