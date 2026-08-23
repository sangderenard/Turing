source_filename = "turing.ssa-llvm.expm1_core_pack__expm1_core_pack"

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-windows-gnu"

define void @__ssa_expm1_core_pack__expm1_core_pack__planned_region_0(ptr noalias %arg.0, ptr %arg.1, ptr %out.0) {
entry:
  %load.0.31.0 = load i32, ptr %arg.1, align 4
  %address.0.31 = getelementptr double, ptr %arg.0, i32 %load.0.31.0
  %pinned.load.1.19 = load double, ptr %address.0.31, align 8
  store double %pinned.load.1.19, ptr %out.0, align 8
  ret void
}

define void @__ssa_expm1_core_pack__expm1_core_pack__planned_region_1(ptr %arg.0, ptr %arg.1, ptr %arg.2) {
entry:
  %load.0.32.0 = load i32, ptr %arg.1, align 4
  %address.0.32 = getelementptr double, ptr %arg.0, i32 %load.0.32.0
  %load.store.1.v = load double, ptr %arg.2, align 8
  store double %load.store.1.v, ptr %address.0.32, align 8
  ret void
}

define void @__ssa_expm1_core_pack__expm1_core_pack(ptr noalias %arg.0, ptr noalias %arg.1, ptr %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr noalias %arg.11, ptr noalias %arg.12, ptr noalias %arg.13, ptr noalias %arg.14, ptr noalias %arg.15, ptr noalias %arg.16, ptr %out.0) {
entry:
  %value.23 = alloca i64, i64 1, align 8
  %value.24 = alloca i64, i64 1, align 8
  %value.29 = alloca i64, i64 1, align 8
  %value.26 = alloca i64, i64 1, align 8
  %value.27 = alloca i1, i64 1, align 8
  %value.19 = alloca double, i64 1, align 8
  %value.20 = alloca double, i64 1, align 8
  store i64 0, ptr %value.23, align 8
  store i64 1, ptr %value.24, align 8
  store i64 0, ptr %value.29, align 8
  br label %loop_header
loop_header:
  %phi.25 = phi ptr [ %value.23, %entry ], [ %value.26, %loop_latch ]
  %load.5.27.0 = load i32, ptr %phi.25, align 4
  %load.5.27.1 = load i32, ptr %arg.0, align 4
  %scalar.5.27 = icmp slt i32 %load.5.27.0, %load.5.27.1
  store i1 %scalar.5.27, ptr %value.27, align 1
  br i1 %scalar.5.27, label %loop_body, label %loop_exit
loop_body:
  call void @__ssa_expm1_core_pack__expm1_core_pack__planned_region_0(ptr %arg.1, ptr %phi.25, ptr %value.19)
  call void @__ssa_expm1_core_pack__expm1_core(ptr %arg.3, ptr %arg.4, ptr %arg.5, ptr %arg.6, ptr %arg.7, ptr %arg.8, ptr %arg.9, ptr %arg.10, ptr %arg.11, ptr %arg.12, ptr %arg.13, ptr %arg.14, ptr %arg.15, ptr %arg.16, ptr %value.19, ptr %value.19, ptr %value.20)
  call void @__ssa_expm1_core_pack__expm1_core_pack__planned_region_1(ptr %arg.2, ptr %phi.25, ptr %value.20)
  br label %loop_latch
loop_latch:
  %load.13.26.0 = load i32, ptr %phi.25, align 4
  %load.13.26.1 = load i64, ptr %value.24, align 8
  %convert.13.26.1 = trunc i64 %load.13.26.1 to i32
  %scalar.13.26 = add i32 %load.13.26.0, %convert.13.26.1
  %declared.13.26 = sext i32 %scalar.13.26 to i64
  store i64 %declared.13.26, ptr %value.26, align 8
  br label %loop_header
loop_exit:
  %return.load.0.26 = load double, ptr %arg.2, align 8
  store double %return.load.0.26, ptr %out.0, align 8
  ret void
}

define void @__ssa_expm1_core_pack__expm1_core__planned_region_0(ptr noalias %arg.0, ptr %arg.1, ptr noalias %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr noalias %arg.11, ptr noalias %arg.12, ptr noalias %arg.13, ptr noalias %arg.14, ptr %arg.15, ptr %out.0, ptr %out.1, ptr %out.2, ptr %out.3, ptr %out.4, ptr %out.5, ptr %out.6, ptr %out.7, ptr %out.8, ptr %out.9, ptr %out.10, ptr %out.11, ptr %out.12, ptr %out.13, ptr %out.14, ptr %out.15, ptr %out.16, ptr %out.17, ptr %out.18, ptr %out.19, ptr %out.20, ptr %out.21, ptr %out.22, ptr %out.23, ptr %out.24, ptr %out.25, ptr %out.26) {
entry:
  %load.0.16.0 = load double, ptr %arg.0, align 8
  %load.0.16.1 = load double, ptr %arg.1, align 8
  %scalar.0.16 = fmul double %load.0.16.0, %load.0.16.1
  store double %scalar.0.16, ptr %out.1, align 8
  %load.1.17.0 = load double, ptr %arg.2, align 8
  %scalar.1.17 = fadd double %load.1.17.0, %scalar.0.16
  store double %scalar.1.17, ptr %out.2, align 8
  %scalar.2.18 = fmul double %load.0.16.1, %scalar.1.17
  store double %scalar.2.18, ptr %out.3, align 8
  %load.3.19.0 = load double, ptr %arg.3, align 8
  %scalar.3.19 = fadd double %load.3.19.0, %scalar.2.18
  store double %scalar.3.19, ptr %out.4, align 8
  %scalar.4.20 = fmul double %load.0.16.1, %scalar.3.19
  store double %scalar.4.20, ptr %out.5, align 8
  %load.5.21.0 = load double, ptr %arg.4, align 8
  %scalar.5.21 = fadd double %load.5.21.0, %scalar.4.20
  store double %scalar.5.21, ptr %out.6, align 8
  %scalar.6.22 = fmul double %load.0.16.1, %scalar.5.21
  store double %scalar.6.22, ptr %out.7, align 8
  %load.7.23.0 = load double, ptr %arg.5, align 8
  %scalar.7.23 = fadd double %load.7.23.0, %scalar.6.22
  store double %scalar.7.23, ptr %out.8, align 8
  %scalar.8.24 = fmul double %load.0.16.1, %scalar.7.23
  store double %scalar.8.24, ptr %out.9, align 8
  %load.9.25.0 = load double, ptr %arg.6, align 8
  %scalar.9.25 = fadd double %load.9.25.0, %scalar.8.24
  store double %scalar.9.25, ptr %out.10, align 8
  %scalar.10.26 = fmul double %load.0.16.1, %scalar.9.25
  store double %scalar.10.26, ptr %out.11, align 8
  %load.11.27.0 = load double, ptr %arg.7, align 8
  %scalar.11.27 = fadd double %load.11.27.0, %scalar.10.26
  store double %scalar.11.27, ptr %out.12, align 8
  %scalar.12.28 = fmul double %load.0.16.1, %scalar.11.27
  store double %scalar.12.28, ptr %out.13, align 8
  %load.13.29.0 = load double, ptr %arg.8, align 8
  %scalar.13.29 = fadd double %load.13.29.0, %scalar.12.28
  store double %scalar.13.29, ptr %out.14, align 8
  %scalar.14.30 = fmul double %load.0.16.1, %scalar.13.29
  store double %scalar.14.30, ptr %out.15, align 8
  %load.15.31.0 = load double, ptr %arg.9, align 8
  %scalar.15.31 = fadd double %load.15.31.0, %scalar.14.30
  store double %scalar.15.31, ptr %out.16, align 8
  %scalar.16.32 = fmul double %load.0.16.1, %scalar.15.31
  store double %scalar.16.32, ptr %out.17, align 8
  %load.17.33.0 = load double, ptr %arg.10, align 8
  %scalar.17.33 = fadd double %load.17.33.0, %scalar.16.32
  store double %scalar.17.33, ptr %out.18, align 8
  %scalar.18.34 = fmul double %load.0.16.1, %scalar.17.33
  store double %scalar.18.34, ptr %out.19, align 8
  %load.19.35.0 = load double, ptr %arg.11, align 8
  %scalar.19.35 = fadd double %load.19.35.0, %scalar.18.34
  store double %scalar.19.35, ptr %out.20, align 8
  %scalar.20.36 = fmul double %load.0.16.1, %scalar.19.35
  store double %scalar.20.36, ptr %out.21, align 8
  %load.21.37.0 = load double, ptr %arg.12, align 8
  %scalar.21.37 = fadd double %load.21.37.0, %scalar.20.36
  store double %scalar.21.37, ptr %out.22, align 8
  %scalar.22.38 = fmul double %load.0.16.1, %scalar.21.37
  store double %scalar.22.38, ptr %out.23, align 8
  %load.23.39.0 = load double, ptr %arg.13, align 8
  %scalar.23.39 = fadd double %load.23.39.0, %scalar.22.38
  store double %scalar.23.39, ptr %out.24, align 8
  %scalar.24.40 = fmul double %load.0.16.1, %scalar.23.39
  store double %scalar.24.40, ptr %out.25, align 8
  %load.25.41.0 = load double, ptr %arg.14, align 8
  %scalar.25.41 = fadd double %load.25.41.0, %scalar.24.40
  store double %scalar.25.41, ptr %out.26, align 8
  %load.26.42.0 = load double, ptr %arg.15, align 8
  %scalar.26.42 = fmul double %load.26.42.0, %scalar.25.41
  store double %scalar.26.42, ptr %out.0, align 8
  ret void
}

define void @__ssa_expm1_core_pack__expm1_core(ptr noalias %arg.0, ptr noalias %arg.1, ptr noalias %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr noalias %arg.11, ptr noalias %arg.12, ptr noalias %arg.13, ptr %arg.14, ptr %arg.15, ptr %out.0) {
entry:
  %value.96 = alloca i32, i64 1, align 8
  %value.94 = alloca i32, i64 1, align 8
  %value.92 = alloca i32, i64 1, align 8
  %value.90 = alloca i32, i64 1, align 8
  %value.88 = alloca i32, i64 1, align 8
  %value.86 = alloca i32, i64 1, align 8
  %value.84 = alloca i32, i64 1, align 8
  %value.82 = alloca i32, i64 1, align 8
  %value.80 = alloca i32, i64 1, align 8
  %value.78 = alloca i32, i64 1, align 8
  %value.76 = alloca i32, i64 1, align 8
  %value.74 = alloca i32, i64 1, align 8
  %value.72 = alloca i32, i64 1, align 8
  %value.70 = alloca i32, i64 1, align 8
  %value.68 = alloca i32, i64 1, align 8
  %value.66 = alloca i32, i64 1, align 8
  %value.64 = alloca i32, i64 1, align 8
  %value.62 = alloca i32, i64 1, align 8
  %value.60 = alloca i32, i64 1, align 8
  %value.58 = alloca i32, i64 1, align 8
  %value.56 = alloca i32, i64 1, align 8
  %value.54 = alloca i32, i64 1, align 8
  %value.52 = alloca i32, i64 1, align 8
  %value.50 = alloca i32, i64 1, align 8
  %value.48 = alloca i32, i64 1, align 8
  %value.46 = alloca i32, i64 1, align 8
  %value.44 = alloca i64, i64 1, align 8
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
  %value.33 = alloca double, i64 1, align 8
  %value.34 = alloca double, i64 1, align 8
  %value.35 = alloca double, i64 1, align 8
  %value.36 = alloca double, i64 1, align 8
  %value.37 = alloca double, i64 1, align 8
  %value.38 = alloca double, i64 1, align 8
  %value.39 = alloca double, i64 1, align 8
  %value.40 = alloca double, i64 1, align 8
  %value.41 = alloca double, i64 1, align 8
  store i32 26, ptr %value.96, align 4
  store i32 25, ptr %value.94, align 4
  store i32 24, ptr %value.92, align 4
  store i32 23, ptr %value.90, align 4
  store i32 22, ptr %value.88, align 4
  store i32 21, ptr %value.86, align 4
  store i32 20, ptr %value.84, align 4
  store i32 19, ptr %value.82, align 4
  store i32 18, ptr %value.80, align 4
  store i32 17, ptr %value.78, align 4
  store i32 16, ptr %value.76, align 4
  store i32 15, ptr %value.74, align 4
  store i32 14, ptr %value.72, align 4
  store i32 13, ptr %value.70, align 4
  store i32 12, ptr %value.68, align 4
  store i32 11, ptr %value.66, align 4
  store i32 10, ptr %value.64, align 4
  store i32 9, ptr %value.62, align 4
  store i32 8, ptr %value.60, align 4
  store i32 7, ptr %value.58, align 4
  store i32 6, ptr %value.56, align 4
  store i32 5, ptr %value.54, align 4
  store i32 4, ptr %value.52, align 4
  store i32 3, ptr %value.50, align 4
  store i32 2, ptr %value.48, align 4
  store i32 1, ptr %value.46, align 4
  store i64 0, ptr %value.44, align 8
  call void @__ssa_expm1_core_pack__expm1_core__planned_region_0(ptr %arg.5, ptr %arg.14, ptr %arg.4, ptr %arg.3, ptr %arg.2, ptr %arg.13, ptr %arg.12, ptr %arg.11, ptr %arg.10, ptr %arg.9, ptr %arg.8, ptr %arg.7, ptr %arg.6, ptr %arg.1, ptr %arg.0, ptr %arg.15, ptr %out.0, ptr %value.16, ptr %value.17, ptr %value.18, ptr %value.19, ptr %value.20, ptr %value.21, ptr %value.22, ptr %value.23, ptr %value.24, ptr %value.25, ptr %value.26, ptr %value.27, ptr %value.28, ptr %value.29, ptr %value.30, ptr %value.31, ptr %value.32, ptr %value.33, ptr %value.34, ptr %value.35, ptr %value.36, ptr %value.37, ptr %value.38, ptr %value.39, ptr %value.40, ptr %value.41)
  ret void
}

define void @expm1_core_pack__expm1_core_pack(ptr %buffers, ptr %extents) {
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
  %public.addr.12 = getelementptr ptr, ptr %buffers, i64 12
  %public.12 = load ptr, ptr %public.addr.12, align 8
  %public.addr.13 = getelementptr ptr, ptr %buffers, i64 13
  %public.13 = load ptr, ptr %public.addr.13, align 8
  %public.addr.14 = getelementptr ptr, ptr %buffers, i64 14
  %public.14 = load ptr, ptr %public.addr.14, align 8
  %public.addr.15 = getelementptr ptr, ptr %buffers, i64 15
  %public.15 = load ptr, ptr %public.addr.15, align 8
  %public.addr.16 = getelementptr ptr, ptr %buffers, i64 16
  %public.16 = load ptr, ptr %public.addr.16, align 8
  call void @__ssa_expm1_core_pack__expm1_core_pack(ptr %public.0, ptr %public.1, ptr %public.2, ptr %public.3, ptr %public.4, ptr %public.5, ptr %public.6, ptr %public.7, ptr %public.8, ptr %public.9, ptr %public.10, ptr %public.11, ptr %public.12, ptr %public.13, ptr %public.14, ptr %public.15, ptr %public.16, ptr %public.2)
  ret void
}
