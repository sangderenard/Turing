source_filename = "turing.ssa-llvm.atanh_core_pack__atanh_core_pack"

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-windows-gnu"

define void @__ssa_atanh_core_pack__atanh_core_pack__planned_region_0(ptr noalias %arg.0, ptr %arg.1, ptr %out.0, ptr %out.1) {
entry:
  %load.0.46.0 = load i32, ptr %arg.1, align 4
  %address.0.46 = getelementptr double, ptr %arg.0, i32 %load.0.46.0
  %pinned.load.1.31 = load double, ptr %address.0.46, align 8
  store double %pinned.load.1.31, ptr %out.0, align 8
  %load.2.32.0 = load double, ptr %out.0, align 8
  %scalar.2.32 = fmul double %load.2.32.0, %load.2.32.0
  store double %scalar.2.32, ptr %out.1, align 8
  ret void
}

define void @__ssa_atanh_core_pack__atanh_core_pack__planned_region_1(ptr %arg.0, ptr %arg.1, ptr %arg.2) {
entry:
  %load.0.47.0 = load i32, ptr %arg.1, align 4
  %address.0.47 = getelementptr double, ptr %arg.0, i32 %load.0.47.0
  %load.store.1.v = load double, ptr %arg.2, align 8
  store double %load.store.1.v, ptr %address.0.47, align 8
  ret void
}

define void @__ssa_atanh_core_pack__atanh_core_pack(ptr noalias %arg.0, ptr noalias %arg.1, ptr %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr noalias %arg.11, ptr noalias %arg.12, ptr noalias %arg.13, ptr noalias %arg.14, ptr noalias %arg.15, ptr noalias %arg.16, ptr noalias %arg.17, ptr noalias %arg.18, ptr noalias %arg.19, ptr noalias %arg.20, ptr noalias %arg.21, ptr noalias %arg.22, ptr noalias %arg.23, ptr noalias %arg.24, ptr noalias %arg.25, ptr noalias %arg.26, ptr noalias %arg.27, ptr noalias %arg.28, ptr %out.0) {
entry:
  %value.36 = alloca i64, i64 1, align 8
  %value.37 = alloca i64, i64 1, align 8
  %value.42 = alloca i64, i64 1, align 8
  %value.44 = alloca i64, i64 1, align 8
  %value.39 = alloca i64, i64 1, align 8
  %value.40 = alloca i1, i64 1, align 8
  %value.31 = alloca double, i64 1, align 8
  %value.32 = alloca double, i64 1, align 8
  %value.33 = alloca double, i64 1, align 8
  store i64 0, ptr %value.36, align 8
  store i64 1, ptr %value.37, align 8
  store i64 0, ptr %value.42, align 8
  store i64 1, ptr %value.44, align 8
  br label %loop_header
loop_header:
  %phi.38 = phi ptr [ %value.36, %entry ], [ %value.39, %loop_latch ]
  %load.6.40.0 = load i32, ptr %phi.38, align 4
  %load.6.40.1 = load i32, ptr %arg.0, align 4
  %scalar.6.40 = icmp slt i32 %load.6.40.0, %load.6.40.1
  store i1 %scalar.6.40, ptr %value.40, align 1
  br i1 %scalar.6.40, label %loop_body, label %loop_exit
loop_body:
  call void @__ssa_atanh_core_pack__atanh_core_pack__planned_region_0(ptr %arg.1, ptr %phi.38, ptr %value.31, ptr %value.32)
  call void @__ssa_atanh_core_pack__atanh_core(ptr %arg.3, ptr %arg.4, ptr %arg.5, ptr %arg.6, ptr %arg.7, ptr %arg.8, ptr %arg.9, ptr %arg.10, ptr %arg.11, ptr %arg.12, ptr %arg.13, ptr %arg.14, ptr %arg.15, ptr %arg.16, ptr %arg.17, ptr %arg.18, ptr %arg.19, ptr %arg.20, ptr %arg.21, ptr %arg.22, ptr %arg.23, ptr %arg.24, ptr %arg.25, ptr %arg.26, ptr %arg.27, ptr %arg.28, ptr %value.32, ptr %value.31, ptr %value.33)
  call void @__ssa_atanh_core_pack__atanh_core_pack__planned_region_1(ptr %arg.2, ptr %phi.38, ptr %value.33)
  br label %loop_latch
loop_latch:
  %load.16.39.0 = load i32, ptr %phi.38, align 4
  %load.16.39.1 = load i64, ptr %value.37, align 8
  %convert.16.39.1 = trunc i64 %load.16.39.1 to i32
  %scalar.16.39 = add i32 %load.16.39.0, %convert.16.39.1
  %declared.16.39 = sext i32 %scalar.16.39 to i64
  store i64 %declared.16.39, ptr %value.39, align 8
  br label %loop_header
loop_exit:
  %return.load.0.27 = load double, ptr %arg.2, align 8
  store double %return.load.0.27, ptr %out.0, align 8
  ret void
}

define void @__ssa_atanh_core_pack__atanh_core__planned_region_0(ptr noalias %arg.0, ptr %arg.1, ptr noalias %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr noalias %arg.11, ptr noalias %arg.12, ptr noalias %arg.13, ptr noalias %arg.14, ptr noalias %arg.15, ptr noalias %arg.16, ptr noalias %arg.17, ptr noalias %arg.18, ptr noalias %arg.19, ptr noalias %arg.20, ptr noalias %arg.21, ptr noalias %arg.22, ptr noalias %arg.23, ptr noalias %arg.24, ptr noalias %arg.25, ptr noalias %arg.26, ptr %arg.27, ptr %out.0, ptr %out.1, ptr %out.2, ptr %out.3, ptr %out.4, ptr %out.5, ptr %out.6, ptr %out.7, ptr %out.8, ptr %out.9, ptr %out.10, ptr %out.11, ptr %out.12, ptr %out.13, ptr %out.14, ptr %out.15, ptr %out.16, ptr %out.17, ptr %out.18, ptr %out.19, ptr %out.20, ptr %out.21, ptr %out.22, ptr %out.23, ptr %out.24, ptr %out.25, ptr %out.26, ptr %out.27, ptr %out.28, ptr %out.29, ptr %out.30, ptr %out.31, ptr %out.32, ptr %out.33, ptr %out.34, ptr %out.35, ptr %out.36, ptr %out.37, ptr %out.38, ptr %out.39, ptr %out.40, ptr %out.41, ptr %out.42, ptr %out.43, ptr %out.44, ptr %out.45, ptr %out.46, ptr %out.47, ptr %out.48, ptr %out.49, ptr %out.50) {
entry:
  %load.0.28.0 = load double, ptr %arg.0, align 8
  %load.0.28.1 = load double, ptr %arg.1, align 8
  %scalar.0.28 = fmul double %load.0.28.0, %load.0.28.1
  store double %scalar.0.28, ptr %out.1, align 8
  %load.1.29.0 = load double, ptr %arg.2, align 8
  %scalar.1.29 = fadd double %load.1.29.0, %scalar.0.28
  store double %scalar.1.29, ptr %out.2, align 8
  %scalar.2.30 = fmul double %load.0.28.1, %scalar.1.29
  store double %scalar.2.30, ptr %out.3, align 8
  %load.3.31.0 = load double, ptr %arg.3, align 8
  %scalar.3.31 = fadd double %load.3.31.0, %scalar.2.30
  store double %scalar.3.31, ptr %out.4, align 8
  %scalar.4.32 = fmul double %load.0.28.1, %scalar.3.31
  store double %scalar.4.32, ptr %out.5, align 8
  %load.5.33.0 = load double, ptr %arg.4, align 8
  %scalar.5.33 = fadd double %load.5.33.0, %scalar.4.32
  store double %scalar.5.33, ptr %out.6, align 8
  %scalar.6.34 = fmul double %load.0.28.1, %scalar.5.33
  store double %scalar.6.34, ptr %out.7, align 8
  %load.7.35.0 = load double, ptr %arg.5, align 8
  %scalar.7.35 = fadd double %load.7.35.0, %scalar.6.34
  store double %scalar.7.35, ptr %out.8, align 8
  %scalar.8.36 = fmul double %load.0.28.1, %scalar.7.35
  store double %scalar.8.36, ptr %out.9, align 8
  %load.9.37.0 = load double, ptr %arg.6, align 8
  %scalar.9.37 = fadd double %load.9.37.0, %scalar.8.36
  store double %scalar.9.37, ptr %out.10, align 8
  %scalar.10.38 = fmul double %load.0.28.1, %scalar.9.37
  store double %scalar.10.38, ptr %out.11, align 8
  %load.11.39.0 = load double, ptr %arg.7, align 8
  %scalar.11.39 = fadd double %load.11.39.0, %scalar.10.38
  store double %scalar.11.39, ptr %out.12, align 8
  %scalar.12.40 = fmul double %load.0.28.1, %scalar.11.39
  store double %scalar.12.40, ptr %out.13, align 8
  %load.13.41.0 = load double, ptr %arg.8, align 8
  %scalar.13.41 = fadd double %load.13.41.0, %scalar.12.40
  store double %scalar.13.41, ptr %out.14, align 8
  %scalar.14.42 = fmul double %load.0.28.1, %scalar.13.41
  store double %scalar.14.42, ptr %out.15, align 8
  %load.15.43.0 = load double, ptr %arg.9, align 8
  %scalar.15.43 = fadd double %load.15.43.0, %scalar.14.42
  store double %scalar.15.43, ptr %out.16, align 8
  %scalar.16.44 = fmul double %load.0.28.1, %scalar.15.43
  store double %scalar.16.44, ptr %out.17, align 8
  %load.17.45.0 = load double, ptr %arg.10, align 8
  %scalar.17.45 = fadd double %load.17.45.0, %scalar.16.44
  store double %scalar.17.45, ptr %out.18, align 8
  %scalar.18.46 = fmul double %load.0.28.1, %scalar.17.45
  store double %scalar.18.46, ptr %out.19, align 8
  %load.19.47.0 = load double, ptr %arg.11, align 8
  %scalar.19.47 = fadd double %load.19.47.0, %scalar.18.46
  store double %scalar.19.47, ptr %out.20, align 8
  %scalar.20.48 = fmul double %load.0.28.1, %scalar.19.47
  store double %scalar.20.48, ptr %out.21, align 8
  %load.21.49.0 = load double, ptr %arg.12, align 8
  %scalar.21.49 = fadd double %load.21.49.0, %scalar.20.48
  store double %scalar.21.49, ptr %out.22, align 8
  %scalar.22.50 = fmul double %load.0.28.1, %scalar.21.49
  store double %scalar.22.50, ptr %out.23, align 8
  %load.23.51.0 = load double, ptr %arg.13, align 8
  %scalar.23.51 = fadd double %load.23.51.0, %scalar.22.50
  store double %scalar.23.51, ptr %out.24, align 8
  %scalar.24.52 = fmul double %load.0.28.1, %scalar.23.51
  store double %scalar.24.52, ptr %out.25, align 8
  %load.25.53.0 = load double, ptr %arg.14, align 8
  %scalar.25.53 = fadd double %load.25.53.0, %scalar.24.52
  store double %scalar.25.53, ptr %out.26, align 8
  %scalar.26.54 = fmul double %load.0.28.1, %scalar.25.53
  store double %scalar.26.54, ptr %out.27, align 8
  %load.27.55.0 = load double, ptr %arg.15, align 8
  %scalar.27.55 = fadd double %load.27.55.0, %scalar.26.54
  store double %scalar.27.55, ptr %out.28, align 8
  %scalar.28.56 = fmul double %load.0.28.1, %scalar.27.55
  store double %scalar.28.56, ptr %out.29, align 8
  %load.29.57.0 = load double, ptr %arg.16, align 8
  %scalar.29.57 = fadd double %load.29.57.0, %scalar.28.56
  store double %scalar.29.57, ptr %out.30, align 8
  %scalar.30.58 = fmul double %load.0.28.1, %scalar.29.57
  store double %scalar.30.58, ptr %out.31, align 8
  %load.31.59.0 = load double, ptr %arg.17, align 8
  %scalar.31.59 = fadd double %load.31.59.0, %scalar.30.58
  store double %scalar.31.59, ptr %out.32, align 8
  %scalar.32.60 = fmul double %load.0.28.1, %scalar.31.59
  store double %scalar.32.60, ptr %out.33, align 8
  %load.33.61.0 = load double, ptr %arg.18, align 8
  %scalar.33.61 = fadd double %load.33.61.0, %scalar.32.60
  store double %scalar.33.61, ptr %out.34, align 8
  %scalar.34.62 = fmul double %load.0.28.1, %scalar.33.61
  store double %scalar.34.62, ptr %out.35, align 8
  %load.35.63.0 = load double, ptr %arg.19, align 8
  %scalar.35.63 = fadd double %load.35.63.0, %scalar.34.62
  store double %scalar.35.63, ptr %out.36, align 8
  %scalar.36.64 = fmul double %load.0.28.1, %scalar.35.63
  store double %scalar.36.64, ptr %out.37, align 8
  %load.37.65.0 = load double, ptr %arg.20, align 8
  %scalar.37.65 = fadd double %load.37.65.0, %scalar.36.64
  store double %scalar.37.65, ptr %out.38, align 8
  %scalar.38.66 = fmul double %load.0.28.1, %scalar.37.65
  store double %scalar.38.66, ptr %out.39, align 8
  %load.39.67.0 = load double, ptr %arg.21, align 8
  %scalar.39.67 = fadd double %load.39.67.0, %scalar.38.66
  store double %scalar.39.67, ptr %out.40, align 8
  %scalar.40.68 = fmul double %load.0.28.1, %scalar.39.67
  store double %scalar.40.68, ptr %out.41, align 8
  %load.41.69.0 = load double, ptr %arg.22, align 8
  %scalar.41.69 = fadd double %load.41.69.0, %scalar.40.68
  store double %scalar.41.69, ptr %out.42, align 8
  %scalar.42.70 = fmul double %load.0.28.1, %scalar.41.69
  store double %scalar.42.70, ptr %out.43, align 8
  %load.43.71.0 = load double, ptr %arg.23, align 8
  %scalar.43.71 = fadd double %load.43.71.0, %scalar.42.70
  store double %scalar.43.71, ptr %out.44, align 8
  %scalar.44.72 = fmul double %load.0.28.1, %scalar.43.71
  store double %scalar.44.72, ptr %out.45, align 8
  %load.45.73.0 = load double, ptr %arg.24, align 8
  %scalar.45.73 = fadd double %load.45.73.0, %scalar.44.72
  store double %scalar.45.73, ptr %out.46, align 8
  %scalar.46.74 = fmul double %load.0.28.1, %scalar.45.73
  store double %scalar.46.74, ptr %out.47, align 8
  %load.47.75.0 = load double, ptr %arg.25, align 8
  %scalar.47.75 = fadd double %load.47.75.0, %scalar.46.74
  store double %scalar.47.75, ptr %out.48, align 8
  %scalar.48.76 = fmul double %load.0.28.1, %scalar.47.75
  store double %scalar.48.76, ptr %out.49, align 8
  %load.49.77.0 = load double, ptr %arg.26, align 8
  %scalar.49.77 = fadd double %load.49.77.0, %scalar.48.76
  store double %scalar.49.77, ptr %out.50, align 8
  %load.50.78.0 = load double, ptr %arg.27, align 8
  %scalar.50.78 = fmul double %load.50.78.0, %scalar.49.77
  store double %scalar.50.78, ptr %out.0, align 8
  ret void
}

define void @__ssa_atanh_core_pack__atanh_core(ptr noalias %arg.0, ptr noalias %arg.1, ptr noalias %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr noalias %arg.11, ptr noalias %arg.12, ptr noalias %arg.13, ptr noalias %arg.14, ptr noalias %arg.15, ptr noalias %arg.16, ptr noalias %arg.17, ptr noalias %arg.18, ptr noalias %arg.19, ptr noalias %arg.20, ptr noalias %arg.21, ptr noalias %arg.22, ptr noalias %arg.23, ptr noalias %arg.24, ptr noalias %arg.25, ptr %arg.26, ptr %arg.27, ptr %out.0) {
entry:
  %value.180 = alloca i32, i64 1, align 8
  %value.178 = alloca i32, i64 1, align 8
  %value.176 = alloca i32, i64 1, align 8
  %value.174 = alloca i32, i64 1, align 8
  %value.172 = alloca i32, i64 1, align 8
  %value.170 = alloca i32, i64 1, align 8
  %value.168 = alloca i32, i64 1, align 8
  %value.166 = alloca i32, i64 1, align 8
  %value.164 = alloca i32, i64 1, align 8
  %value.162 = alloca i32, i64 1, align 8
  %value.160 = alloca i32, i64 1, align 8
  %value.158 = alloca i32, i64 1, align 8
  %value.156 = alloca i32, i64 1, align 8
  %value.154 = alloca i32, i64 1, align 8
  %value.152 = alloca i32, i64 1, align 8
  %value.150 = alloca i32, i64 1, align 8
  %value.148 = alloca i32, i64 1, align 8
  %value.146 = alloca i32, i64 1, align 8
  %value.144 = alloca i32, i64 1, align 8
  %value.142 = alloca i32, i64 1, align 8
  %value.140 = alloca i32, i64 1, align 8
  %value.138 = alloca i32, i64 1, align 8
  %value.136 = alloca i32, i64 1, align 8
  %value.134 = alloca i32, i64 1, align 8
  %value.132 = alloca i32, i64 1, align 8
  %value.130 = alloca i32, i64 1, align 8
  %value.128 = alloca i32, i64 1, align 8
  %value.126 = alloca i32, i64 1, align 8
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
  %value.80 = alloca i64, i64 1, align 8
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
  %value.54 = alloca double, i64 1, align 8
  %value.55 = alloca double, i64 1, align 8
  %value.56 = alloca double, i64 1, align 8
  %value.57 = alloca double, i64 1, align 8
  %value.58 = alloca double, i64 1, align 8
  %value.59 = alloca double, i64 1, align 8
  %value.60 = alloca double, i64 1, align 8
  %value.61 = alloca double, i64 1, align 8
  %value.62 = alloca double, i64 1, align 8
  %value.63 = alloca double, i64 1, align 8
  %value.64 = alloca double, i64 1, align 8
  %value.65 = alloca double, i64 1, align 8
  %value.66 = alloca double, i64 1, align 8
  %value.67 = alloca double, i64 1, align 8
  %value.68 = alloca double, i64 1, align 8
  %value.69 = alloca double, i64 1, align 8
  %value.70 = alloca double, i64 1, align 8
  %value.71 = alloca double, i64 1, align 8
  %value.72 = alloca double, i64 1, align 8
  %value.73 = alloca double, i64 1, align 8
  %value.74 = alloca double, i64 1, align 8
  %value.75 = alloca double, i64 1, align 8
  %value.76 = alloca double, i64 1, align 8
  %value.77 = alloca double, i64 1, align 8
  store i32 50, ptr %value.180, align 4
  store i32 49, ptr %value.178, align 4
  store i32 48, ptr %value.176, align 4
  store i32 47, ptr %value.174, align 4
  store i32 46, ptr %value.172, align 4
  store i32 45, ptr %value.170, align 4
  store i32 44, ptr %value.168, align 4
  store i32 43, ptr %value.166, align 4
  store i32 42, ptr %value.164, align 4
  store i32 41, ptr %value.162, align 4
  store i32 40, ptr %value.160, align 4
  store i32 39, ptr %value.158, align 4
  store i32 38, ptr %value.156, align 4
  store i32 37, ptr %value.154, align 4
  store i32 36, ptr %value.152, align 4
  store i32 35, ptr %value.150, align 4
  store i32 34, ptr %value.148, align 4
  store i32 33, ptr %value.146, align 4
  store i32 32, ptr %value.144, align 4
  store i32 31, ptr %value.142, align 4
  store i32 30, ptr %value.140, align 4
  store i32 29, ptr %value.138, align 4
  store i32 28, ptr %value.136, align 4
  store i32 27, ptr %value.134, align 4
  store i32 26, ptr %value.132, align 4
  store i32 25, ptr %value.130, align 4
  store i32 24, ptr %value.128, align 4
  store i32 23, ptr %value.126, align 4
  store i32 22, ptr %value.124, align 4
  store i32 21, ptr %value.122, align 4
  store i32 20, ptr %value.120, align 4
  store i32 19, ptr %value.118, align 4
  store i32 18, ptr %value.116, align 4
  store i32 17, ptr %value.114, align 4
  store i32 16, ptr %value.112, align 4
  store i32 15, ptr %value.110, align 4
  store i32 14, ptr %value.108, align 4
  store i32 13, ptr %value.106, align 4
  store i32 12, ptr %value.104, align 4
  store i32 11, ptr %value.102, align 4
  store i32 10, ptr %value.100, align 4
  store i32 9, ptr %value.98, align 4
  store i32 8, ptr %value.96, align 4
  store i32 7, ptr %value.94, align 4
  store i32 6, ptr %value.92, align 4
  store i32 5, ptr %value.90, align 4
  store i32 4, ptr %value.88, align 4
  store i32 3, ptr %value.86, align 4
  store i32 2, ptr %value.84, align 4
  store i32 1, ptr %value.82, align 4
  store i64 0, ptr %value.80, align 8
  call void @__ssa_atanh_core_pack__atanh_core__planned_region_0(ptr %arg.18, ptr %arg.26, ptr %arg.17, ptr %arg.16, ptr %arg.15, ptr %arg.14, ptr %arg.13, ptr %arg.11, ptr %arg.10, ptr %arg.9, ptr %arg.8, ptr %arg.7, ptr %arg.6, ptr %arg.5, ptr %arg.4, ptr %arg.3, ptr %arg.2, ptr %arg.25, ptr %arg.24, ptr %arg.23, ptr %arg.22, ptr %arg.21, ptr %arg.20, ptr %arg.19, ptr %arg.12, ptr %arg.1, ptr %arg.0, ptr %arg.27, ptr %out.0, ptr %value.28, ptr %value.29, ptr %value.30, ptr %value.31, ptr %value.32, ptr %value.33, ptr %value.34, ptr %value.35, ptr %value.36, ptr %value.37, ptr %value.38, ptr %value.39, ptr %value.40, ptr %value.41, ptr %value.42, ptr %value.43, ptr %value.44, ptr %value.45, ptr %value.46, ptr %value.47, ptr %value.48, ptr %value.49, ptr %value.50, ptr %value.51, ptr %value.52, ptr %value.53, ptr %value.54, ptr %value.55, ptr %value.56, ptr %value.57, ptr %value.58, ptr %value.59, ptr %value.60, ptr %value.61, ptr %value.62, ptr %value.63, ptr %value.64, ptr %value.65, ptr %value.66, ptr %value.67, ptr %value.68, ptr %value.69, ptr %value.70, ptr %value.71, ptr %value.72, ptr %value.73, ptr %value.74, ptr %value.75, ptr %value.76, ptr %value.77)
  ret void
}

define void @atanh_core_pack__atanh_core_pack(ptr %buffers, ptr %extents) {
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
  %public.addr.21 = getelementptr ptr, ptr %buffers, i64 21
  %public.21 = load ptr, ptr %public.addr.21, align 8
  %public.addr.22 = getelementptr ptr, ptr %buffers, i64 22
  %public.22 = load ptr, ptr %public.addr.22, align 8
  %public.addr.23 = getelementptr ptr, ptr %buffers, i64 23
  %public.23 = load ptr, ptr %public.addr.23, align 8
  %public.addr.24 = getelementptr ptr, ptr %buffers, i64 24
  %public.24 = load ptr, ptr %public.addr.24, align 8
  %public.addr.25 = getelementptr ptr, ptr %buffers, i64 25
  %public.25 = load ptr, ptr %public.addr.25, align 8
  %public.addr.26 = getelementptr ptr, ptr %buffers, i64 26
  %public.26 = load ptr, ptr %public.addr.26, align 8
  %public.addr.27 = getelementptr ptr, ptr %buffers, i64 27
  %public.27 = load ptr, ptr %public.addr.27, align 8
  %public.addr.28 = getelementptr ptr, ptr %buffers, i64 28
  %public.28 = load ptr, ptr %public.addr.28, align 8
  call void @__ssa_atanh_core_pack__atanh_core_pack(ptr %public.0, ptr %public.1, ptr %public.2, ptr %public.3, ptr %public.4, ptr %public.5, ptr %public.6, ptr %public.7, ptr %public.8, ptr %public.9, ptr %public.10, ptr %public.11, ptr %public.12, ptr %public.13, ptr %public.14, ptr %public.15, ptr %public.16, ptr %public.17, ptr %public.18, ptr %public.19, ptr %public.20, ptr %public.21, ptr %public.22, ptr %public.23, ptr %public.24, ptr %public.25, ptr %public.26, ptr %public.27, ptr %public.28, ptr %public.2)
  ret void
}
