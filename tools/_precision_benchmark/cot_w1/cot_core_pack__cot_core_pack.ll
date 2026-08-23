source_filename = "turing.ssa-llvm.cot_core_pack__cot_core_pack"

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-windows-gnu"

define void @__ssa_cot_core_pack__cot_core_pack__planned_region_0(ptr noalias %arg.0, ptr %arg.1, ptr %out.0, ptr %out.1) {
entry:
  %load.0.29.0 = load i32, ptr %arg.1, align 4
  %address.0.29 = getelementptr double, ptr %arg.0, i32 %load.0.29.0
  %pinned.load.1.16 = load double, ptr %address.0.29, align 8
  store double %pinned.load.1.16, ptr %out.1, align 8
  %load.2.17.0 = load double, ptr %out.1, align 8
  %scalar.2.17 = fmul double %load.2.17.0, %load.2.17.0
  store double %scalar.2.17, ptr %out.0, align 8
  ret void
}

define void @__ssa_cot_core_pack__cot_core_pack__planned_region_1(ptr %arg.0, ptr %arg.1, ptr %arg.2) {
entry:
  %load.0.30.0 = load i32, ptr %arg.1, align 4
  %address.0.30 = getelementptr double, ptr %arg.0, i32 %load.0.30.0
  %load.store.1.v = load double, ptr %arg.2, align 8
  store double %load.store.1.v, ptr %address.0.30, align 8
  ret void
}

define void @__ssa_cot_core_pack__cot_core_pack(ptr noalias %arg.0, ptr noalias %arg.1, ptr %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr noalias %arg.11, ptr noalias %arg.12, ptr noalias %arg.13, ptr %out.0) {
entry:
  %value.21 = alloca i64, i64 1, align 8
  %value.22 = alloca i64, i64 1, align 8
  %value.29 = alloca i32, i64 1, align 8
  %value.27 = alloca i64, i64 1, align 8
  %value.24 = alloca i64, i64 1, align 8
  %value.25 = alloca i1, i64 1, align 8
  %value.17 = alloca double, i64 1, align 8
  %value.16 = alloca double, i64 1, align 8
  %value.18 = alloca double, i64 1, align 8
  store i64 0, ptr %value.21, align 8
  store i64 1, ptr %value.22, align 8
  store i32 1, ptr %value.29, align 4
  store i64 0, ptr %value.27, align 8
  br label %loop_header
loop_header:
  %phi.23 = phi ptr [ %value.21, %entry ], [ %value.24, %loop_latch ]
  %load.6.25.0 = load i32, ptr %phi.23, align 4
  %load.6.25.1 = load i32, ptr %arg.0, align 4
  %scalar.6.25 = icmp slt i32 %load.6.25.0, %load.6.25.1
  store i1 %scalar.6.25, ptr %value.25, align 1
  br i1 %scalar.6.25, label %loop_body, label %loop_exit
loop_body:
  call void @__ssa_cot_core_pack__cot_core_pack__planned_region_0(ptr %arg.1, ptr %phi.23, ptr %value.17, ptr %value.16)
  call void @__ssa_cot_core_pack__cot_core(ptr %arg.3, ptr %arg.4, ptr %arg.5, ptr %arg.6, ptr %arg.7, ptr %arg.8, ptr %arg.9, ptr %arg.10, ptr %arg.11, ptr %arg.12, ptr %arg.13, ptr %value.17, ptr %value.18)
  call void @__ssa_cot_core_pack__cot_core_pack__planned_region_1(ptr %arg.2, ptr %phi.23, ptr %value.18)
  br label %loop_latch
loop_latch:
  %load.16.24.0 = load i32, ptr %phi.23, align 4
  %load.16.24.1 = load i64, ptr %value.22, align 8
  %convert.16.24.1 = trunc i64 %load.16.24.1 to i32
  %scalar.16.24 = add i32 %load.16.24.0, %convert.16.24.1
  %declared.16.24 = sext i32 %scalar.16.24 to i64
  store i64 %declared.16.24, ptr %value.24, align 8
  br label %loop_header
loop_exit:
  %return.load.0.27 = load double, ptr %arg.2, align 8
  store double %return.load.0.27, ptr %out.0, align 8
  ret void
}

define void @__ssa_cot_core_pack__cot_core__planned_region_0(ptr noalias %arg.0, ptr %arg.1, ptr noalias %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr noalias %arg.11, ptr %out.0, ptr %out.1, ptr %out.2, ptr %out.3, ptr %out.4, ptr %out.5, ptr %out.6, ptr %out.7, ptr %out.8, ptr %out.9, ptr %out.10, ptr %out.11, ptr %out.12, ptr %out.13, ptr %out.14, ptr %out.15, ptr %out.16, ptr %out.17, ptr %out.18, ptr %out.19) {
entry:
  %load.0.12.0 = load double, ptr %arg.0, align 8
  %load.0.12.1 = load double, ptr %arg.1, align 8
  %scalar.0.12 = fmul double %load.0.12.0, %load.0.12.1
  store double %scalar.0.12, ptr %out.1, align 8
  %load.1.13.0 = load double, ptr %arg.2, align 8
  %scalar.1.13 = fadd double %load.1.13.0, %scalar.0.12
  store double %scalar.1.13, ptr %out.2, align 8
  %scalar.2.14 = fmul double %load.0.12.1, %scalar.1.13
  store double %scalar.2.14, ptr %out.3, align 8
  %load.3.15.0 = load double, ptr %arg.3, align 8
  %scalar.3.15 = fadd double %load.3.15.0, %scalar.2.14
  store double %scalar.3.15, ptr %out.4, align 8
  %scalar.4.16 = fmul double %load.0.12.1, %scalar.3.15
  store double %scalar.4.16, ptr %out.5, align 8
  %load.5.17.0 = load double, ptr %arg.4, align 8
  %scalar.5.17 = fadd double %load.5.17.0, %scalar.4.16
  store double %scalar.5.17, ptr %out.6, align 8
  %scalar.6.18 = fmul double %load.0.12.1, %scalar.5.17
  store double %scalar.6.18, ptr %out.7, align 8
  %load.7.19.0 = load double, ptr %arg.5, align 8
  %scalar.7.19 = fadd double %load.7.19.0, %scalar.6.18
  store double %scalar.7.19, ptr %out.8, align 8
  %scalar.8.20 = fmul double %load.0.12.1, %scalar.7.19
  store double %scalar.8.20, ptr %out.9, align 8
  %load.9.21.0 = load double, ptr %arg.6, align 8
  %scalar.9.21 = fadd double %load.9.21.0, %scalar.8.20
  store double %scalar.9.21, ptr %out.10, align 8
  %scalar.10.22 = fmul double %load.0.12.1, %scalar.9.21
  store double %scalar.10.22, ptr %out.11, align 8
  %load.11.23.0 = load double, ptr %arg.7, align 8
  %scalar.11.23 = fadd double %load.11.23.0, %scalar.10.22
  store double %scalar.11.23, ptr %out.12, align 8
  %scalar.12.24 = fmul double %load.0.12.1, %scalar.11.23
  store double %scalar.12.24, ptr %out.13, align 8
  %load.13.25.0 = load double, ptr %arg.8, align 8
  %scalar.13.25 = fadd double %load.13.25.0, %scalar.12.24
  store double %scalar.13.25, ptr %out.14, align 8
  %scalar.14.26 = fmul double %load.0.12.1, %scalar.13.25
  store double %scalar.14.26, ptr %out.15, align 8
  %load.15.27.0 = load double, ptr %arg.9, align 8
  %scalar.15.27 = fadd double %load.15.27.0, %scalar.14.26
  store double %scalar.15.27, ptr %out.16, align 8
  %scalar.16.28 = fmul double %load.0.12.1, %scalar.15.27
  store double %scalar.16.28, ptr %out.17, align 8
  %load.17.29.0 = load double, ptr %arg.10, align 8
  %scalar.17.29 = fadd double %load.17.29.0, %scalar.16.28
  store double %scalar.17.29, ptr %out.18, align 8
  %scalar.18.30 = fmul double %load.0.12.1, %scalar.17.29
  store double %scalar.18.30, ptr %out.19, align 8
  %load.19.31.0 = load double, ptr %arg.11, align 8
  %scalar.19.31 = fadd double %load.19.31.0, %scalar.18.30
  store double %scalar.19.31, ptr %out.0, align 8
  ret void
}

define void @__ssa_cot_core_pack__cot_core(ptr noalias %arg.0, ptr noalias %arg.1, ptr noalias %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr %arg.11, ptr %out.0) {
entry:
  %value.71 = alloca i32, i64 1, align 8
  %value.69 = alloca i32, i64 1, align 8
  %value.67 = alloca i32, i64 1, align 8
  %value.65 = alloca i32, i64 1, align 8
  %value.63 = alloca i32, i64 1, align 8
  %value.61 = alloca i32, i64 1, align 8
  %value.59 = alloca i32, i64 1, align 8
  %value.57 = alloca i32, i64 1, align 8
  %value.55 = alloca i32, i64 1, align 8
  %value.53 = alloca i32, i64 1, align 8
  %value.51 = alloca i32, i64 1, align 8
  %value.49 = alloca i32, i64 1, align 8
  %value.47 = alloca i32, i64 1, align 8
  %value.45 = alloca i32, i64 1, align 8
  %value.43 = alloca i32, i64 1, align 8
  %value.41 = alloca i32, i64 1, align 8
  %value.39 = alloca i32, i64 1, align 8
  %value.37 = alloca i32, i64 1, align 8
  %value.35 = alloca i32, i64 1, align 8
  %value.33 = alloca i64, i64 1, align 8
  %value.12 = alloca double, i64 1, align 8
  %value.13 = alloca double, i64 1, align 8
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
  store i32 19, ptr %value.71, align 4
  store i32 18, ptr %value.69, align 4
  store i32 17, ptr %value.67, align 4
  store i32 16, ptr %value.65, align 4
  store i32 15, ptr %value.63, align 4
  store i32 14, ptr %value.61, align 4
  store i32 13, ptr %value.59, align 4
  store i32 12, ptr %value.57, align 4
  store i32 11, ptr %value.55, align 4
  store i32 10, ptr %value.53, align 4
  store i32 9, ptr %value.51, align 4
  store i32 8, ptr %value.49, align 4
  store i32 7, ptr %value.47, align 4
  store i32 6, ptr %value.45, align 4
  store i32 5, ptr %value.43, align 4
  store i32 4, ptr %value.41, align 4
  store i32 3, ptr %value.39, align 4
  store i32 2, ptr %value.37, align 4
  store i32 1, ptr %value.35, align 4
  store i64 0, ptr %value.33, align 8
  call void @__ssa_cot_core_pack__cot_core__planned_region_0(ptr %arg.2, ptr %arg.11, ptr %arg.10, ptr %arg.9, ptr %arg.8, ptr %arg.7, ptr %arg.6, ptr %arg.5, ptr %arg.4, ptr %arg.3, ptr %arg.1, ptr %arg.0, ptr %out.0, ptr %value.12, ptr %value.13, ptr %value.14, ptr %value.15, ptr %value.16, ptr %value.17, ptr %value.18, ptr %value.19, ptr %value.20, ptr %value.21, ptr %value.22, ptr %value.23, ptr %value.24, ptr %value.25, ptr %value.26, ptr %value.27, ptr %value.28, ptr %value.29, ptr %value.30)
  ret void
}

define void @cot_core_pack__cot_core_pack(ptr %buffers, ptr %extents) {
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
  call void @__ssa_cot_core_pack__cot_core_pack(ptr %public.0, ptr %public.1, ptr %public.2, ptr %public.3, ptr %public.4, ptr %public.5, ptr %public.6, ptr %public.7, ptr %public.8, ptr %public.9, ptr %public.10, ptr %public.11, ptr %public.12, ptr %public.13, ptr %public.2)
  ret void
}
