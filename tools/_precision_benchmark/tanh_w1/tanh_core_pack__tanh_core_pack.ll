source_filename = "turing.ssa-llvm.tanh_core_pack__tanh_core_pack"

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-windows-gnu"

define void @__ssa_tanh_core_pack__tanh_core_pack__planned_region_0(ptr noalias %arg.0, ptr %arg.1, ptr %out.0, ptr %out.1) {
entry:
  %load.0.38.0 = load i32, ptr %arg.1, align 4
  %address.0.38 = getelementptr double, ptr %arg.0, i32 %load.0.38.0
  %pinned.load.1.23 = load double, ptr %address.0.38, align 8
  store double %pinned.load.1.23, ptr %out.0, align 8
  %load.2.24.0 = load double, ptr %out.0, align 8
  %scalar.2.24 = fmul double %load.2.24.0, %load.2.24.0
  store double %scalar.2.24, ptr %out.1, align 8
  ret void
}

define void @__ssa_tanh_core_pack__tanh_core_pack__planned_region_1(ptr %arg.0, ptr %arg.1, ptr %arg.2) {
entry:
  %load.0.39.0 = load i32, ptr %arg.1, align 4
  %address.0.39 = getelementptr double, ptr %arg.0, i32 %load.0.39.0
  %load.store.1.v = load double, ptr %arg.2, align 8
  store double %load.store.1.v, ptr %address.0.39, align 8
  ret void
}

define void @__ssa_tanh_core_pack__tanh_core_pack(ptr noalias %arg.0, ptr noalias %arg.1, ptr %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr noalias %arg.11, ptr noalias %arg.12, ptr noalias %arg.13, ptr noalias %arg.14, ptr noalias %arg.15, ptr noalias %arg.16, ptr noalias %arg.17, ptr noalias %arg.18, ptr noalias %arg.19, ptr noalias %arg.20, ptr %out.0) {
entry:
  %value.28 = alloca i64, i64 1, align 8
  %value.29 = alloca i64, i64 1, align 8
  %value.34 = alloca i64, i64 1, align 8
  %value.36 = alloca i64, i64 1, align 8
  %value.31 = alloca i64, i64 1, align 8
  %value.32 = alloca i1, i64 1, align 8
  %value.23 = alloca double, i64 1, align 8
  %value.24 = alloca double, i64 1, align 8
  %value.25 = alloca double, i64 1, align 8
  store i64 0, ptr %value.28, align 8
  store i64 1, ptr %value.29, align 8
  store i64 0, ptr %value.34, align 8
  store i64 1, ptr %value.36, align 8
  br label %loop_header
loop_header:
  %phi.30 = phi ptr [ %value.28, %entry ], [ %value.31, %loop_latch ]
  %load.6.32.0 = load i32, ptr %phi.30, align 4
  %load.6.32.1 = load i32, ptr %arg.0, align 4
  %scalar.6.32 = icmp slt i32 %load.6.32.0, %load.6.32.1
  store i1 %scalar.6.32, ptr %value.32, align 1
  br i1 %scalar.6.32, label %loop_body, label %loop_exit
loop_body:
  call void @__ssa_tanh_core_pack__tanh_core_pack__planned_region_0(ptr %arg.1, ptr %phi.30, ptr %value.23, ptr %value.24)
  call void @__ssa_tanh_core_pack__tanh_core(ptr %arg.3, ptr %arg.4, ptr %arg.5, ptr %arg.6, ptr %arg.7, ptr %arg.8, ptr %arg.9, ptr %arg.10, ptr %arg.11, ptr %arg.12, ptr %arg.13, ptr %arg.14, ptr %arg.15, ptr %arg.16, ptr %arg.17, ptr %arg.18, ptr %arg.19, ptr %arg.20, ptr %value.24, ptr %value.23, ptr %value.25)
  call void @__ssa_tanh_core_pack__tanh_core_pack__planned_region_1(ptr %arg.2, ptr %phi.30, ptr %value.25)
  br label %loop_latch
loop_latch:
  %load.16.31.0 = load i32, ptr %phi.30, align 4
  %load.16.31.1 = load i64, ptr %value.29, align 8
  %convert.16.31.1 = trunc i64 %load.16.31.1 to i32
  %scalar.16.31 = add i32 %load.16.31.0, %convert.16.31.1
  %declared.16.31 = sext i32 %scalar.16.31 to i64
  store i64 %declared.16.31, ptr %value.31, align 8
  br label %loop_header
loop_exit:
  %return.load.0.27 = load double, ptr %arg.2, align 8
  store double %return.load.0.27, ptr %out.0, align 8
  ret void
}

define void @__ssa_tanh_core_pack__tanh_core__planned_region_0(ptr noalias %arg.0, ptr %arg.1, ptr noalias %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr noalias %arg.11, ptr noalias %arg.12, ptr noalias %arg.13, ptr noalias %arg.14, ptr noalias %arg.15, ptr noalias %arg.16, ptr noalias %arg.17, ptr noalias %arg.18, ptr %arg.19, ptr %out.0, ptr %out.1, ptr %out.2, ptr %out.3, ptr %out.4, ptr %out.5, ptr %out.6, ptr %out.7, ptr %out.8, ptr %out.9, ptr %out.10, ptr %out.11, ptr %out.12, ptr %out.13, ptr %out.14, ptr %out.15, ptr %out.16, ptr %out.17, ptr %out.18, ptr %out.19, ptr %out.20, ptr %out.21, ptr %out.22, ptr %out.23, ptr %out.24, ptr %out.25, ptr %out.26, ptr %out.27, ptr %out.28, ptr %out.29, ptr %out.30, ptr %out.31, ptr %out.32, ptr %out.33, ptr %out.34) {
entry:
  %load.0.20.0 = load double, ptr %arg.0, align 8
  %load.0.20.1 = load double, ptr %arg.1, align 8
  %scalar.0.20 = fmul double %load.0.20.0, %load.0.20.1
  store double %scalar.0.20, ptr %out.1, align 8
  %load.1.21.0 = load double, ptr %arg.2, align 8
  %scalar.1.21 = fadd double %load.1.21.0, %scalar.0.20
  store double %scalar.1.21, ptr %out.2, align 8
  %scalar.2.22 = fmul double %load.0.20.1, %scalar.1.21
  store double %scalar.2.22, ptr %out.3, align 8
  %load.3.23.0 = load double, ptr %arg.3, align 8
  %scalar.3.23 = fadd double %load.3.23.0, %scalar.2.22
  store double %scalar.3.23, ptr %out.4, align 8
  %scalar.4.24 = fmul double %load.0.20.1, %scalar.3.23
  store double %scalar.4.24, ptr %out.5, align 8
  %load.5.25.0 = load double, ptr %arg.4, align 8
  %scalar.5.25 = fadd double %load.5.25.0, %scalar.4.24
  store double %scalar.5.25, ptr %out.6, align 8
  %scalar.6.26 = fmul double %load.0.20.1, %scalar.5.25
  store double %scalar.6.26, ptr %out.7, align 8
  %load.7.27.0 = load double, ptr %arg.5, align 8
  %scalar.7.27 = fadd double %load.7.27.0, %scalar.6.26
  store double %scalar.7.27, ptr %out.8, align 8
  %scalar.8.28 = fmul double %load.0.20.1, %scalar.7.27
  store double %scalar.8.28, ptr %out.9, align 8
  %load.9.29.0 = load double, ptr %arg.6, align 8
  %scalar.9.29 = fadd double %load.9.29.0, %scalar.8.28
  store double %scalar.9.29, ptr %out.10, align 8
  %scalar.10.30 = fmul double %load.0.20.1, %scalar.9.29
  store double %scalar.10.30, ptr %out.11, align 8
  %load.11.31.0 = load double, ptr %arg.7, align 8
  %scalar.11.31 = fadd double %load.11.31.0, %scalar.10.30
  store double %scalar.11.31, ptr %out.12, align 8
  %scalar.12.32 = fmul double %load.0.20.1, %scalar.11.31
  store double %scalar.12.32, ptr %out.13, align 8
  %load.13.33.0 = load double, ptr %arg.8, align 8
  %scalar.13.33 = fadd double %load.13.33.0, %scalar.12.32
  store double %scalar.13.33, ptr %out.14, align 8
  %scalar.14.34 = fmul double %load.0.20.1, %scalar.13.33
  store double %scalar.14.34, ptr %out.15, align 8
  %load.15.35.0 = load double, ptr %arg.9, align 8
  %scalar.15.35 = fadd double %load.15.35.0, %scalar.14.34
  store double %scalar.15.35, ptr %out.16, align 8
  %scalar.16.36 = fmul double %load.0.20.1, %scalar.15.35
  store double %scalar.16.36, ptr %out.17, align 8
  %load.17.37.0 = load double, ptr %arg.10, align 8
  %scalar.17.37 = fadd double %load.17.37.0, %scalar.16.36
  store double %scalar.17.37, ptr %out.18, align 8
  %scalar.18.38 = fmul double %load.0.20.1, %scalar.17.37
  store double %scalar.18.38, ptr %out.19, align 8
  %load.19.39.0 = load double, ptr %arg.11, align 8
  %scalar.19.39 = fadd double %load.19.39.0, %scalar.18.38
  store double %scalar.19.39, ptr %out.20, align 8
  %scalar.20.40 = fmul double %load.0.20.1, %scalar.19.39
  store double %scalar.20.40, ptr %out.21, align 8
  %load.21.41.0 = load double, ptr %arg.12, align 8
  %scalar.21.41 = fadd double %load.21.41.0, %scalar.20.40
  store double %scalar.21.41, ptr %out.22, align 8
  %scalar.22.42 = fmul double %load.0.20.1, %scalar.21.41
  store double %scalar.22.42, ptr %out.23, align 8
  %load.23.43.0 = load double, ptr %arg.13, align 8
  %scalar.23.43 = fadd double %load.23.43.0, %scalar.22.42
  store double %scalar.23.43, ptr %out.24, align 8
  %scalar.24.44 = fmul double %load.0.20.1, %scalar.23.43
  store double %scalar.24.44, ptr %out.25, align 8
  %load.25.45.0 = load double, ptr %arg.14, align 8
  %scalar.25.45 = fadd double %load.25.45.0, %scalar.24.44
  store double %scalar.25.45, ptr %out.26, align 8
  %scalar.26.46 = fmul double %load.0.20.1, %scalar.25.45
  store double %scalar.26.46, ptr %out.27, align 8
  %load.27.47.0 = load double, ptr %arg.15, align 8
  %scalar.27.47 = fadd double %load.27.47.0, %scalar.26.46
  store double %scalar.27.47, ptr %out.28, align 8
  %scalar.28.48 = fmul double %load.0.20.1, %scalar.27.47
  store double %scalar.28.48, ptr %out.29, align 8
  %load.29.49.0 = load double, ptr %arg.16, align 8
  %scalar.29.49 = fadd double %load.29.49.0, %scalar.28.48
  store double %scalar.29.49, ptr %out.30, align 8
  %scalar.30.50 = fmul double %load.0.20.1, %scalar.29.49
  store double %scalar.30.50, ptr %out.31, align 8
  %load.31.51.0 = load double, ptr %arg.17, align 8
  %scalar.31.51 = fadd double %load.31.51.0, %scalar.30.50
  store double %scalar.31.51, ptr %out.32, align 8
  %scalar.32.52 = fmul double %load.0.20.1, %scalar.31.51
  store double %scalar.32.52, ptr %out.33, align 8
  %load.33.53.0 = load double, ptr %arg.18, align 8
  %scalar.33.53 = fadd double %load.33.53.0, %scalar.32.52
  store double %scalar.33.53, ptr %out.34, align 8
  %load.34.54.0 = load double, ptr %arg.19, align 8
  %scalar.34.54 = fmul double %load.34.54.0, %scalar.33.53
  store double %scalar.34.54, ptr %out.0, align 8
  ret void
}

define void @__ssa_tanh_core_pack__tanh_core(ptr noalias %arg.0, ptr noalias %arg.1, ptr noalias %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr noalias %arg.11, ptr noalias %arg.12, ptr noalias %arg.13, ptr noalias %arg.14, ptr noalias %arg.15, ptr noalias %arg.16, ptr noalias %arg.17, ptr %arg.18, ptr %arg.19, ptr %out.0) {
entry:
  %value.124 = alloca i32, i64 1, align 8
  %value.122 = alloca i32, i64 1, align 8
  %value.120 = alloca i32, i64 1, align 8
  %value.118 = alloca i32, i64 1, align 8
  %value.116 = alloca i32, i64 1, align 8
  %value.114 = alloca i32, i64 1, align 8
  %value.112 = alloca i32, i64 1, align 8
  %value.110 = alloca i32, i64 1, align 8
  %value.108 = alloca i32, i64 1, align 8
  %value.106 = alloca i32, i64 1, align 8
  %value.104 = alloca i32, i64 1, align 8
  %value.102 = alloca i32, i64 1, align 8
  %value.100 = alloca i32, i64 1, align 8
  %value.98 = alloca i32, i64 1, align 8
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
  %value.56 = alloca i64, i64 1, align 8
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
  %value.42 = alloca double, i64 1, align 8
  %value.43 = alloca double, i64 1, align 8
  %value.44 = alloca double, i64 1, align 8
  %value.45 = alloca double, i64 1, align 8
  %value.46 = alloca double, i64 1, align 8
  %value.47 = alloca double, i64 1, align 8
  %value.48 = alloca double, i64 1, align 8
  %value.49 = alloca double, i64 1, align 8
  %value.50 = alloca double, i64 1, align 8
  %value.51 = alloca double, i64 1, align 8
  %value.52 = alloca double, i64 1, align 8
  %value.53 = alloca double, i64 1, align 8
  store i32 34, ptr %value.124, align 4
  store i32 33, ptr %value.122, align 4
  store i32 32, ptr %value.120, align 4
  store i32 31, ptr %value.118, align 4
  store i32 30, ptr %value.116, align 4
  store i32 29, ptr %value.114, align 4
  store i32 28, ptr %value.112, align 4
  store i32 27, ptr %value.110, align 4
  store i32 26, ptr %value.108, align 4
  store i32 25, ptr %value.106, align 4
  store i32 24, ptr %value.104, align 4
  store i32 23, ptr %value.102, align 4
  store i32 22, ptr %value.100, align 4
  store i32 21, ptr %value.98, align 4
  store i32 20, ptr %value.96, align 4
  store i32 19, ptr %value.94, align 4
  store i32 18, ptr %value.92, align 4
  store i32 17, ptr %value.90, align 4
  store i32 16, ptr %value.88, align 4
  store i32 15, ptr %value.86, align 4
  store i32 14, ptr %value.84, align 4
  store i32 13, ptr %value.82, align 4
  store i32 12, ptr %value.80, align 4
  store i32 11, ptr %value.78, align 4
  store i32 10, ptr %value.76, align 4
  store i32 9, ptr %value.74, align 4
  store i32 8, ptr %value.72, align 4
  store i32 7, ptr %value.70, align 4
  store i32 6, ptr %value.68, align 4
  store i32 5, ptr %value.66, align 4
  store i32 4, ptr %value.64, align 4
  store i32 3, ptr %value.62, align 4
  store i32 2, ptr %value.60, align 4
  store i32 1, ptr %value.58, align 4
  store i64 0, ptr %value.56, align 8
  call void @__ssa_tanh_core_pack__tanh_core__planned_region_0(ptr %arg.9, ptr %arg.18, ptr %arg.8, ptr %arg.7, ptr %arg.6, ptr %arg.5, ptr %arg.4, ptr %arg.3, ptr %arg.2, ptr %arg.17, ptr %arg.16, ptr %arg.15, ptr %arg.14, ptr %arg.13, ptr %arg.12, ptr %arg.11, ptr %arg.10, ptr %arg.1, ptr %arg.0, ptr %arg.19, ptr %out.0, ptr %value.20, ptr %value.21, ptr %value.22, ptr %value.23, ptr %value.24, ptr %value.25, ptr %value.26, ptr %value.27, ptr %value.28, ptr %value.29, ptr %value.30, ptr %value.31, ptr %value.32, ptr %value.33, ptr %value.34, ptr %value.35, ptr %value.36, ptr %value.37, ptr %value.38, ptr %value.39, ptr %value.40, ptr %value.41, ptr %value.42, ptr %value.43, ptr %value.44, ptr %value.45, ptr %value.46, ptr %value.47, ptr %value.48, ptr %value.49, ptr %value.50, ptr %value.51, ptr %value.52, ptr %value.53)
  ret void
}

define void @tanh_core_pack__tanh_core_pack(ptr %buffers, ptr %extents) {
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
  %public.addr.17 = getelementptr ptr, ptr %buffers, i64 17
  %public.17 = load ptr, ptr %public.addr.17, align 8
  %public.addr.18 = getelementptr ptr, ptr %buffers, i64 18
  %public.18 = load ptr, ptr %public.addr.18, align 8
  %public.addr.19 = getelementptr ptr, ptr %buffers, i64 19
  %public.19 = load ptr, ptr %public.addr.19, align 8
  %public.addr.20 = getelementptr ptr, ptr %buffers, i64 20
  %public.20 = load ptr, ptr %public.addr.20, align 8
  call void @__ssa_tanh_core_pack__tanh_core_pack(ptr %public.0, ptr %public.1, ptr %public.2, ptr %public.3, ptr %public.4, ptr %public.5, ptr %public.6, ptr %public.7, ptr %public.8, ptr %public.9, ptr %public.10, ptr %public.11, ptr %public.12, ptr %public.13, ptr %public.14, ptr %public.15, ptr %public.16, ptr %public.17, ptr %public.18, ptr %public.19, ptr %public.20, ptr %public.2)
  ret void
}
