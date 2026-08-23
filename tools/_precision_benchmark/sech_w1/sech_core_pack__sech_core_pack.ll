source_filename = "turing.ssa-llvm.sech_core_pack__sech_core_pack"

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-windows-gnu"

define void @__ssa_sech_core_pack__sech_core_pack__planned_region_0(ptr noalias %arg.0, ptr %arg.1, ptr %out.0, ptr %out.1) {
entry:
  %load.0.63.0 = load i32, ptr %arg.1, align 4
  %address.0.63 = getelementptr double, ptr %arg.0, i32 %load.0.63.0
  %pinned.load.1.50 = load double, ptr %address.0.63, align 8
  store double %pinned.load.1.50, ptr %out.1, align 8
  %load.2.51.0 = load double, ptr %out.1, align 8
  %scalar.2.51 = fmul double %load.2.51.0, %load.2.51.0
  store double %scalar.2.51, ptr %out.0, align 8
  ret void
}

define void @__ssa_sech_core_pack__sech_core_pack__planned_region_1(ptr %arg.0, ptr %arg.1, ptr %arg.2) {
entry:
  %load.0.64.0 = load i32, ptr %arg.1, align 4
  %address.0.64 = getelementptr double, ptr %arg.0, i32 %load.0.64.0
  %load.store.1.v = load double, ptr %arg.2, align 8
  store double %load.store.1.v, ptr %address.0.64, align 8
  ret void
}

define void @__ssa_sech_core_pack__sech_core_pack(ptr noalias %arg.0, ptr noalias %arg.1, ptr %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr noalias %arg.11, ptr noalias %arg.12, ptr noalias %arg.13, ptr noalias %arg.14, ptr noalias %arg.15, ptr noalias %arg.16, ptr noalias %arg.17, ptr noalias %arg.18, ptr noalias %arg.19, ptr noalias %arg.20, ptr noalias %arg.21, ptr noalias %arg.22, ptr noalias %arg.23, ptr noalias %arg.24, ptr noalias %arg.25, ptr noalias %arg.26, ptr noalias %arg.27, ptr noalias %arg.28, ptr noalias %arg.29, ptr noalias %arg.30, ptr noalias %arg.31, ptr noalias %arg.32, ptr noalias %arg.33, ptr noalias %arg.34, ptr noalias %arg.35, ptr noalias %arg.36, ptr noalias %arg.37, ptr noalias %arg.38, ptr noalias %arg.39, ptr noalias %arg.40, ptr noalias %arg.41, ptr noalias %arg.42, ptr noalias %arg.43, ptr noalias %arg.44, ptr noalias %arg.45, ptr noalias %arg.46, ptr noalias %arg.47, ptr %out.0) {
entry:
  %value.55 = alloca i64, i64 1, align 8
  %value.56 = alloca i64, i64 1, align 8
  %value.63 = alloca i32, i64 1, align 8
  %value.61 = alloca i64, i64 1, align 8
  %value.58 = alloca i64, i64 1, align 8
  %value.59 = alloca i1, i64 1, align 8
  %value.51 = alloca double, i64 1, align 8
  %value.50 = alloca double, i64 1, align 8
  %value.52 = alloca double, i64 1, align 8
  store i64 0, ptr %value.55, align 8
  store i64 1, ptr %value.56, align 8
  store i32 1, ptr %value.63, align 4
  store i64 0, ptr %value.61, align 8
  br label %loop_header
loop_header:
  %phi.57 = phi ptr [ %value.55, %entry ], [ %value.58, %loop_latch ]
  %load.6.59.0 = load i32, ptr %phi.57, align 4
  %load.6.59.1 = load i32, ptr %arg.0, align 4
  %scalar.6.59 = icmp slt i32 %load.6.59.0, %load.6.59.1
  store i1 %scalar.6.59, ptr %value.59, align 1
  br i1 %scalar.6.59, label %loop_body, label %loop_exit
loop_body:
  call void @__ssa_sech_core_pack__sech_core_pack__planned_region_0(ptr %arg.1, ptr %phi.57, ptr %value.51, ptr %value.50)
  call void @__ssa_sech_core_pack__sech_core(ptr %arg.3, ptr %arg.4, ptr %arg.5, ptr %arg.6, ptr %arg.7, ptr %arg.8, ptr %arg.9, ptr %arg.10, ptr %arg.11, ptr %arg.12, ptr %arg.13, ptr %arg.14, ptr %arg.15, ptr %arg.16, ptr %arg.17, ptr %arg.18, ptr %arg.19, ptr %arg.20, ptr %arg.21, ptr %arg.22, ptr %arg.23, ptr %arg.24, ptr %arg.25, ptr %arg.26, ptr %arg.27, ptr %arg.28, ptr %arg.29, ptr %arg.30, ptr %arg.31, ptr %arg.32, ptr %arg.33, ptr %arg.34, ptr %arg.35, ptr %arg.36, ptr %arg.37, ptr %arg.38, ptr %arg.39, ptr %arg.40, ptr %arg.41, ptr %arg.42, ptr %arg.43, ptr %arg.44, ptr %arg.45, ptr %arg.46, ptr %arg.47, ptr %value.51, ptr %value.52)
  call void @__ssa_sech_core_pack__sech_core_pack__planned_region_1(ptr %arg.2, ptr %phi.57, ptr %value.52)
  br label %loop_latch
loop_latch:
  %load.16.58.0 = load i32, ptr %phi.57, align 4
  %load.16.58.1 = load i64, ptr %value.56, align 8
  %convert.16.58.1 = trunc i64 %load.16.58.1 to i32
  %scalar.16.58 = add i32 %load.16.58.0, %convert.16.58.1
  %declared.16.58 = sext i32 %scalar.16.58 to i64
  store i64 %declared.16.58, ptr %value.58, align 8
  br label %loop_header
loop_exit:
  %return.load.0.27 = load double, ptr %arg.2, align 8
  store double %return.load.0.27, ptr %out.0, align 8
  ret void
}

define void @__ssa_sech_core_pack__sech_core__planned_region_0(ptr noalias %arg.0, ptr %arg.1, ptr noalias %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr noalias %arg.11, ptr noalias %arg.12, ptr noalias %arg.13, ptr noalias %arg.14, ptr noalias %arg.15, ptr noalias %arg.16, ptr noalias %arg.17, ptr noalias %arg.18, ptr noalias %arg.19, ptr noalias %arg.20, ptr noalias %arg.21, ptr noalias %arg.22, ptr noalias %arg.23, ptr noalias %arg.24, ptr noalias %arg.25, ptr noalias %arg.26, ptr noalias %arg.27, ptr noalias %arg.28, ptr noalias %arg.29, ptr noalias %arg.30, ptr noalias %arg.31, ptr noalias %arg.32, ptr noalias %arg.33, ptr noalias %arg.34, ptr noalias %arg.35, ptr noalias %arg.36, ptr noalias %arg.37, ptr noalias %arg.38, ptr noalias %arg.39, ptr noalias %arg.40, ptr noalias %arg.41, ptr noalias %arg.42, ptr noalias %arg.43, ptr noalias %arg.44, ptr noalias %arg.45, ptr %out.0, ptr %out.1, ptr %out.2, ptr %out.3, ptr %out.4, ptr %out.5, ptr %out.6, ptr %out.7, ptr %out.8, ptr %out.9, ptr %out.10, ptr %out.11, ptr %out.12, ptr %out.13, ptr %out.14, ptr %out.15, ptr %out.16, ptr %out.17, ptr %out.18, ptr %out.19, ptr %out.20, ptr %out.21, ptr %out.22, ptr %out.23, ptr %out.24, ptr %out.25, ptr %out.26, ptr %out.27, ptr %out.28, ptr %out.29, ptr %out.30, ptr %out.31, ptr %out.32, ptr %out.33, ptr %out.34, ptr %out.35, ptr %out.36, ptr %out.37, ptr %out.38, ptr %out.39, ptr %out.40, ptr %out.41, ptr %out.42, ptr %out.43, ptr %out.44, ptr %out.45, ptr %out.46, ptr %out.47, ptr %out.48, ptr %out.49, ptr %out.50, ptr %out.51, ptr %out.52, ptr %out.53, ptr %out.54, ptr %out.55, ptr %out.56, ptr %out.57, ptr %out.58, ptr %out.59, ptr %out.60, ptr %out.61, ptr %out.62, ptr %out.63, ptr %out.64, ptr %out.65, ptr %out.66, ptr %out.67, ptr %out.68, ptr %out.69, ptr %out.70, ptr %out.71, ptr %out.72, ptr %out.73, ptr %out.74, ptr %out.75, ptr %out.76, ptr %out.77, ptr %out.78, ptr %out.79, ptr %out.80, ptr %out.81, ptr %out.82, ptr %out.83, ptr %out.84, ptr %out.85, ptr %out.86, ptr %out.87) {
entry:
  %load.0.46.0 = load double, ptr %arg.0, align 8
  %load.0.46.1 = load double, ptr %arg.1, align 8
  %scalar.0.46 = fmul double %load.0.46.0, %load.0.46.1
  store double %scalar.0.46, ptr %out.1, align 8
  %load.1.47.0 = load double, ptr %arg.2, align 8
  %scalar.1.47 = fadd double %load.1.47.0, %scalar.0.46
  store double %scalar.1.47, ptr %out.2, align 8
  %scalar.2.48 = fmul double %load.0.46.1, %scalar.1.47
  store double %scalar.2.48, ptr %out.3, align 8
  %load.3.49.0 = load double, ptr %arg.3, align 8
  %scalar.3.49 = fadd double %load.3.49.0, %scalar.2.48
  store double %scalar.3.49, ptr %out.4, align 8
  %scalar.4.50 = fmul double %load.0.46.1, %scalar.3.49
  store double %scalar.4.50, ptr %out.5, align 8
  %load.5.51.0 = load double, ptr %arg.4, align 8
  %scalar.5.51 = fadd double %load.5.51.0, %scalar.4.50
  store double %scalar.5.51, ptr %out.6, align 8
  %scalar.6.52 = fmul double %load.0.46.1, %scalar.5.51
  store double %scalar.6.52, ptr %out.7, align 8
  %load.7.53.0 = load double, ptr %arg.5, align 8
  %scalar.7.53 = fadd double %load.7.53.0, %scalar.6.52
  store double %scalar.7.53, ptr %out.8, align 8
  %scalar.8.54 = fmul double %load.0.46.1, %scalar.7.53
  store double %scalar.8.54, ptr %out.9, align 8
  %load.9.55.0 = load double, ptr %arg.6, align 8
  %scalar.9.55 = fadd double %load.9.55.0, %scalar.8.54
  store double %scalar.9.55, ptr %out.10, align 8
  %scalar.10.56 = fmul double %load.0.46.1, %scalar.9.55
  store double %scalar.10.56, ptr %out.11, align 8
  %load.11.57.0 = load double, ptr %arg.7, align 8
  %scalar.11.57 = fadd double %load.11.57.0, %scalar.10.56
  store double %scalar.11.57, ptr %out.12, align 8
  %scalar.12.58 = fmul double %load.0.46.1, %scalar.11.57
  store double %scalar.12.58, ptr %out.13, align 8
  %load.13.59.0 = load double, ptr %arg.8, align 8
  %scalar.13.59 = fadd double %load.13.59.0, %scalar.12.58
  store double %scalar.13.59, ptr %out.14, align 8
  %scalar.14.60 = fmul double %load.0.46.1, %scalar.13.59
  store double %scalar.14.60, ptr %out.15, align 8
  %load.15.61.0 = load double, ptr %arg.9, align 8
  %scalar.15.61 = fadd double %load.15.61.0, %scalar.14.60
  store double %scalar.15.61, ptr %out.16, align 8
  %scalar.16.62 = fmul double %load.0.46.1, %scalar.15.61
  store double %scalar.16.62, ptr %out.17, align 8
  %load.17.63.0 = load double, ptr %arg.10, align 8
  %scalar.17.63 = fadd double %load.17.63.0, %scalar.16.62
  store double %scalar.17.63, ptr %out.18, align 8
  %scalar.18.64 = fmul double %load.0.46.1, %scalar.17.63
  store double %scalar.18.64, ptr %out.19, align 8
  %load.19.65.0 = load double, ptr %arg.11, align 8
  %scalar.19.65 = fadd double %load.19.65.0, %scalar.18.64
  store double %scalar.19.65, ptr %out.20, align 8
  %scalar.20.66 = fmul double %load.0.46.1, %scalar.19.65
  store double %scalar.20.66, ptr %out.21, align 8
  %load.21.67.0 = load double, ptr %arg.12, align 8
  %scalar.21.67 = fadd double %load.21.67.0, %scalar.20.66
  store double %scalar.21.67, ptr %out.22, align 8
  %scalar.22.68 = fmul double %load.0.46.1, %scalar.21.67
  store double %scalar.22.68, ptr %out.23, align 8
  %load.23.69.0 = load double, ptr %arg.13, align 8
  %scalar.23.69 = fadd double %load.23.69.0, %scalar.22.68
  store double %scalar.23.69, ptr %out.24, align 8
  %scalar.24.70 = fmul double %load.0.46.1, %scalar.23.69
  store double %scalar.24.70, ptr %out.25, align 8
  %load.25.71.0 = load double, ptr %arg.14, align 8
  %scalar.25.71 = fadd double %load.25.71.0, %scalar.24.70
  store double %scalar.25.71, ptr %out.26, align 8
  %scalar.26.72 = fmul double %load.0.46.1, %scalar.25.71
  store double %scalar.26.72, ptr %out.27, align 8
  %load.27.73.0 = load double, ptr %arg.15, align 8
  %scalar.27.73 = fadd double %load.27.73.0, %scalar.26.72
  store double %scalar.27.73, ptr %out.28, align 8
  %scalar.28.74 = fmul double %load.0.46.1, %scalar.27.73
  store double %scalar.28.74, ptr %out.29, align 8
  %load.29.75.0 = load double, ptr %arg.16, align 8
  %scalar.29.75 = fadd double %load.29.75.0, %scalar.28.74
  store double %scalar.29.75, ptr %out.30, align 8
  %scalar.30.76 = fmul double %load.0.46.1, %scalar.29.75
  store double %scalar.30.76, ptr %out.31, align 8
  %load.31.77.0 = load double, ptr %arg.17, align 8
  %scalar.31.77 = fadd double %load.31.77.0, %scalar.30.76
  store double %scalar.31.77, ptr %out.32, align 8
  %scalar.32.78 = fmul double %load.0.46.1, %scalar.31.77
  store double %scalar.32.78, ptr %out.33, align 8
  %load.33.79.0 = load double, ptr %arg.18, align 8
  %scalar.33.79 = fadd double %load.33.79.0, %scalar.32.78
  store double %scalar.33.79, ptr %out.34, align 8
  %scalar.34.80 = fmul double %load.0.46.1, %scalar.33.79
  store double %scalar.34.80, ptr %out.35, align 8
  %load.35.81.0 = load double, ptr %arg.19, align 8
  %scalar.35.81 = fadd double %load.35.81.0, %scalar.34.80
  store double %scalar.35.81, ptr %out.36, align 8
  %scalar.36.82 = fmul double %load.0.46.1, %scalar.35.81
  store double %scalar.36.82, ptr %out.37, align 8
  %load.37.83.0 = load double, ptr %arg.20, align 8
  %scalar.37.83 = fadd double %load.37.83.0, %scalar.36.82
  store double %scalar.37.83, ptr %out.38, align 8
  %scalar.38.84 = fmul double %load.0.46.1, %scalar.37.83
  store double %scalar.38.84, ptr %out.39, align 8
  %load.39.85.0 = load double, ptr %arg.21, align 8
  %scalar.39.85 = fadd double %load.39.85.0, %scalar.38.84
  store double %scalar.39.85, ptr %out.40, align 8
  %scalar.40.86 = fmul double %load.0.46.1, %scalar.39.85
  store double %scalar.40.86, ptr %out.41, align 8
  %load.41.87.0 = load double, ptr %arg.22, align 8
  %scalar.41.87 = fadd double %load.41.87.0, %scalar.40.86
  store double %scalar.41.87, ptr %out.42, align 8
  %scalar.42.88 = fmul double %load.0.46.1, %scalar.41.87
  store double %scalar.42.88, ptr %out.43, align 8
  %load.43.89.0 = load double, ptr %arg.23, align 8
  %scalar.43.89 = fadd double %load.43.89.0, %scalar.42.88
  store double %scalar.43.89, ptr %out.44, align 8
  %scalar.44.90 = fmul double %load.0.46.1, %scalar.43.89
  store double %scalar.44.90, ptr %out.45, align 8
  %load.45.91.0 = load double, ptr %arg.24, align 8
  %scalar.45.91 = fadd double %load.45.91.0, %scalar.44.90
  store double %scalar.45.91, ptr %out.46, align 8
  %scalar.46.92 = fmul double %load.0.46.1, %scalar.45.91
  store double %scalar.46.92, ptr %out.47, align 8
  %load.47.93.0 = load double, ptr %arg.25, align 8
  %scalar.47.93 = fadd double %load.47.93.0, %scalar.46.92
  store double %scalar.47.93, ptr %out.48, align 8
  %scalar.48.94 = fmul double %load.0.46.1, %scalar.47.93
  store double %scalar.48.94, ptr %out.49, align 8
  %load.49.95.0 = load double, ptr %arg.26, align 8
  %scalar.49.95 = fadd double %load.49.95.0, %scalar.48.94
  store double %scalar.49.95, ptr %out.50, align 8
  %scalar.50.96 = fmul double %load.0.46.1, %scalar.49.95
  store double %scalar.50.96, ptr %out.51, align 8
  %load.51.97.0 = load double, ptr %arg.27, align 8
  %scalar.51.97 = fadd double %load.51.97.0, %scalar.50.96
  store double %scalar.51.97, ptr %out.52, align 8
  %scalar.52.98 = fmul double %load.0.46.1, %scalar.51.97
  store double %scalar.52.98, ptr %out.53, align 8
  %load.53.99.0 = load double, ptr %arg.28, align 8
  %scalar.53.99 = fadd double %load.53.99.0, %scalar.52.98
  store double %scalar.53.99, ptr %out.54, align 8
  %scalar.54.100 = fmul double %load.0.46.1, %scalar.53.99
  store double %scalar.54.100, ptr %out.55, align 8
  %load.55.101.0 = load double, ptr %arg.29, align 8
  %scalar.55.101 = fadd double %load.55.101.0, %scalar.54.100
  store double %scalar.55.101, ptr %out.56, align 8
  %scalar.56.102 = fmul double %load.0.46.1, %scalar.55.101
  store double %scalar.56.102, ptr %out.57, align 8
  %load.57.103.0 = load double, ptr %arg.30, align 8
  %scalar.57.103 = fadd double %load.57.103.0, %scalar.56.102
  store double %scalar.57.103, ptr %out.58, align 8
  %scalar.58.104 = fmul double %load.0.46.1, %scalar.57.103
  store double %scalar.58.104, ptr %out.59, align 8
  %load.59.105.0 = load double, ptr %arg.31, align 8
  %scalar.59.105 = fadd double %load.59.105.0, %scalar.58.104
  store double %scalar.59.105, ptr %out.60, align 8
  %scalar.60.106 = fmul double %load.0.46.1, %scalar.59.105
  store double %scalar.60.106, ptr %out.61, align 8
  %load.61.107.0 = load double, ptr %arg.32, align 8
  %scalar.61.107 = fadd double %load.61.107.0, %scalar.60.106
  store double %scalar.61.107, ptr %out.62, align 8
  %scalar.62.108 = fmul double %load.0.46.1, %scalar.61.107
  store double %scalar.62.108, ptr %out.63, align 8
  %load.63.109.0 = load double, ptr %arg.33, align 8
  %scalar.63.109 = fadd double %load.63.109.0, %scalar.62.108
  store double %scalar.63.109, ptr %out.64, align 8
  %scalar.64.110 = fmul double %load.0.46.1, %scalar.63.109
  store double %scalar.64.110, ptr %out.65, align 8
  %load.65.111.0 = load double, ptr %arg.34, align 8
  %scalar.65.111 = fadd double %load.65.111.0, %scalar.64.110
  store double %scalar.65.111, ptr %out.66, align 8
  %scalar.66.112 = fmul double %load.0.46.1, %scalar.65.111
  store double %scalar.66.112, ptr %out.67, align 8
  %load.67.113.0 = load double, ptr %arg.35, align 8
  %scalar.67.113 = fadd double %load.67.113.0, %scalar.66.112
  store double %scalar.67.113, ptr %out.68, align 8
  %scalar.68.114 = fmul double %load.0.46.1, %scalar.67.113
  store double %scalar.68.114, ptr %out.69, align 8
  %load.69.115.0 = load double, ptr %arg.36, align 8
  %scalar.69.115 = fadd double %load.69.115.0, %scalar.68.114
  store double %scalar.69.115, ptr %out.70, align 8
  %scalar.70.116 = fmul double %load.0.46.1, %scalar.69.115
  store double %scalar.70.116, ptr %out.71, align 8
  %load.71.117.0 = load double, ptr %arg.37, align 8
  %scalar.71.117 = fadd double %load.71.117.0, %scalar.70.116
  store double %scalar.71.117, ptr %out.72, align 8
  %scalar.72.118 = fmul double %load.0.46.1, %scalar.71.117
  store double %scalar.72.118, ptr %out.73, align 8
  %load.73.119.0 = load double, ptr %arg.38, align 8
  %scalar.73.119 = fadd double %load.73.119.0, %scalar.72.118
  store double %scalar.73.119, ptr %out.74, align 8
  %scalar.74.120 = fmul double %load.0.46.1, %scalar.73.119
  store double %scalar.74.120, ptr %out.75, align 8
  %load.75.121.0 = load double, ptr %arg.39, align 8
  %scalar.75.121 = fadd double %load.75.121.0, %scalar.74.120
  store double %scalar.75.121, ptr %out.76, align 8
  %scalar.76.122 = fmul double %load.0.46.1, %scalar.75.121
  store double %scalar.76.122, ptr %out.77, align 8
  %load.77.123.0 = load double, ptr %arg.40, align 8
  %scalar.77.123 = fadd double %load.77.123.0, %scalar.76.122
  store double %scalar.77.123, ptr %out.78, align 8
  %scalar.78.124 = fmul double %load.0.46.1, %scalar.77.123
  store double %scalar.78.124, ptr %out.79, align 8
  %load.79.125.0 = load double, ptr %arg.41, align 8
  %scalar.79.125 = fadd double %load.79.125.0, %scalar.78.124
  store double %scalar.79.125, ptr %out.80, align 8
  %scalar.80.126 = fmul double %load.0.46.1, %scalar.79.125
  store double %scalar.80.126, ptr %out.81, align 8
  %load.81.127.0 = load double, ptr %arg.42, align 8
  %scalar.81.127 = fadd double %load.81.127.0, %scalar.80.126
  store double %scalar.81.127, ptr %out.82, align 8
  %scalar.82.128 = fmul double %load.0.46.1, %scalar.81.127
  store double %scalar.82.128, ptr %out.83, align 8
  %load.83.129.0 = load double, ptr %arg.43, align 8
  %scalar.83.129 = fadd double %load.83.129.0, %scalar.82.128
  store double %scalar.83.129, ptr %out.84, align 8
  %scalar.84.130 = fmul double %load.0.46.1, %scalar.83.129
  store double %scalar.84.130, ptr %out.85, align 8
  %load.85.131.0 = load double, ptr %arg.44, align 8
  %scalar.85.131 = fadd double %load.85.131.0, %scalar.84.130
  store double %scalar.85.131, ptr %out.86, align 8
  %scalar.86.132 = fmul double %load.0.46.1, %scalar.85.131
  store double %scalar.86.132, ptr %out.87, align 8
  %load.87.133.0 = load double, ptr %arg.45, align 8
  %scalar.87.133 = fadd double %load.87.133.0, %scalar.86.132
  store double %scalar.87.133, ptr %out.0, align 8
  ret void
}

define void @__ssa_sech_core_pack__sech_core(ptr noalias %arg.0, ptr noalias %arg.1, ptr noalias %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr noalias %arg.11, ptr noalias %arg.12, ptr noalias %arg.13, ptr noalias %arg.14, ptr noalias %arg.15, ptr noalias %arg.16, ptr noalias %arg.17, ptr noalias %arg.18, ptr noalias %arg.19, ptr noalias %arg.20, ptr noalias %arg.21, ptr noalias %arg.22, ptr noalias %arg.23, ptr noalias %arg.24, ptr noalias %arg.25, ptr noalias %arg.26, ptr noalias %arg.27, ptr noalias %arg.28, ptr noalias %arg.29, ptr noalias %arg.30, ptr noalias %arg.31, ptr noalias %arg.32, ptr noalias %arg.33, ptr noalias %arg.34, ptr noalias %arg.35, ptr noalias %arg.36, ptr noalias %arg.37, ptr noalias %arg.38, ptr noalias %arg.39, ptr noalias %arg.40, ptr noalias %arg.41, ptr noalias %arg.42, ptr noalias %arg.43, ptr noalias %arg.44, ptr %arg.45, ptr %out.0) {
entry:
  %value.309 = alloca i32, i64 1, align 8
  %value.307 = alloca i32, i64 1, align 8
  %value.305 = alloca i32, i64 1, align 8
  %value.303 = alloca i32, i64 1, align 8
  %value.301 = alloca i32, i64 1, align 8
  %value.299 = alloca i32, i64 1, align 8
  %value.297 = alloca i32, i64 1, align 8
  %value.295 = alloca i32, i64 1, align 8
  %value.293 = alloca i32, i64 1, align 8
  %value.291 = alloca i32, i64 1, align 8
  %value.289 = alloca i32, i64 1, align 8
  %value.287 = alloca i32, i64 1, align 8
  %value.285 = alloca i32, i64 1, align 8
  %value.283 = alloca i32, i64 1, align 8
  %value.281 = alloca i32, i64 1, align 8
  %value.279 = alloca i32, i64 1, align 8
  %value.277 = alloca i32, i64 1, align 8
  %value.275 = alloca i32, i64 1, align 8
  %value.273 = alloca i32, i64 1, align 8
  %value.271 = alloca i32, i64 1, align 8
  %value.269 = alloca i32, i64 1, align 8
  %value.267 = alloca i32, i64 1, align 8
  %value.265 = alloca i32, i64 1, align 8
  %value.263 = alloca i32, i64 1, align 8
  %value.261 = alloca i32, i64 1, align 8
  %value.259 = alloca i32, i64 1, align 8
  %value.257 = alloca i32, i64 1, align 8
  %value.255 = alloca i32, i64 1, align 8
  %value.253 = alloca i32, i64 1, align 8
  %value.251 = alloca i32, i64 1, align 8
  %value.249 = alloca i32, i64 1, align 8
  %value.247 = alloca i32, i64 1, align 8
  %value.245 = alloca i32, i64 1, align 8
  %value.243 = alloca i32, i64 1, align 8
  %value.241 = alloca i32, i64 1, align 8
  %value.239 = alloca i32, i64 1, align 8
  %value.237 = alloca i32, i64 1, align 8
  %value.235 = alloca i32, i64 1, align 8
  %value.233 = alloca i32, i64 1, align 8
  %value.231 = alloca i32, i64 1, align 8
  %value.229 = alloca i32, i64 1, align 8
  %value.227 = alloca i32, i64 1, align 8
  %value.225 = alloca i32, i64 1, align 8
  %value.223 = alloca i32, i64 1, align 8
  %value.221 = alloca i32, i64 1, align 8
  %value.219 = alloca i32, i64 1, align 8
  %value.217 = alloca i32, i64 1, align 8
  %value.215 = alloca i32, i64 1, align 8
  %value.213 = alloca i32, i64 1, align 8
  %value.211 = alloca i32, i64 1, align 8
  %value.209 = alloca i32, i64 1, align 8
  %value.207 = alloca i32, i64 1, align 8
  %value.205 = alloca i32, i64 1, align 8
  %value.203 = alloca i32, i64 1, align 8
  %value.201 = alloca i32, i64 1, align 8
  %value.199 = alloca i32, i64 1, align 8
  %value.197 = alloca i32, i64 1, align 8
  %value.195 = alloca i32, i64 1, align 8
  %value.193 = alloca i32, i64 1, align 8
  %value.191 = alloca i32, i64 1, align 8
  %value.189 = alloca i32, i64 1, align 8
  %value.187 = alloca i32, i64 1, align 8
  %value.185 = alloca i32, i64 1, align 8
  %value.183 = alloca i32, i64 1, align 8
  %value.181 = alloca i32, i64 1, align 8
  %value.179 = alloca i32, i64 1, align 8
  %value.177 = alloca i32, i64 1, align 8
  %value.175 = alloca i32, i64 1, align 8
  %value.173 = alloca i32, i64 1, align 8
  %value.171 = alloca i32, i64 1, align 8
  %value.169 = alloca i32, i64 1, align 8
  %value.167 = alloca i32, i64 1, align 8
  %value.165 = alloca i32, i64 1, align 8
  %value.163 = alloca i32, i64 1, align 8
  %value.161 = alloca i32, i64 1, align 8
  %value.159 = alloca i32, i64 1, align 8
  %value.157 = alloca i32, i64 1, align 8
  %value.155 = alloca i32, i64 1, align 8
  %value.153 = alloca i32, i64 1, align 8
  %value.151 = alloca i32, i64 1, align 8
  %value.149 = alloca i32, i64 1, align 8
  %value.147 = alloca i32, i64 1, align 8
  %value.145 = alloca i32, i64 1, align 8
  %value.143 = alloca i32, i64 1, align 8
  %value.141 = alloca i32, i64 1, align 8
  %value.139 = alloca i32, i64 1, align 8
  %value.137 = alloca i32, i64 1, align 8
  %value.135 = alloca i64, i64 1, align 8
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
  %value.78 = alloca double, i64 1, align 8
  %value.79 = alloca double, i64 1, align 8
  %value.80 = alloca double, i64 1, align 8
  %value.81 = alloca double, i64 1, align 8
  %value.82 = alloca double, i64 1, align 8
  %value.83 = alloca double, i64 1, align 8
  %value.84 = alloca double, i64 1, align 8
  %value.85 = alloca double, i64 1, align 8
  %value.86 = alloca double, i64 1, align 8
  %value.87 = alloca double, i64 1, align 8
  %value.88 = alloca double, i64 1, align 8
  %value.89 = alloca double, i64 1, align 8
  %value.90 = alloca double, i64 1, align 8
  %value.91 = alloca double, i64 1, align 8
  %value.92 = alloca double, i64 1, align 8
  %value.93 = alloca double, i64 1, align 8
  %value.94 = alloca double, i64 1, align 8
  %value.95 = alloca double, i64 1, align 8
  %value.96 = alloca double, i64 1, align 8
  %value.97 = alloca double, i64 1, align 8
  %value.98 = alloca double, i64 1, align 8
  %value.99 = alloca double, i64 1, align 8
  %value.100 = alloca double, i64 1, align 8
  %value.101 = alloca double, i64 1, align 8
  %value.102 = alloca double, i64 1, align 8
  %value.103 = alloca double, i64 1, align 8
  %value.104 = alloca double, i64 1, align 8
  %value.105 = alloca double, i64 1, align 8
  %value.106 = alloca double, i64 1, align 8
  %value.107 = alloca double, i64 1, align 8
  %value.108 = alloca double, i64 1, align 8
  %value.109 = alloca double, i64 1, align 8
  %value.110 = alloca double, i64 1, align 8
  %value.111 = alloca double, i64 1, align 8
  %value.112 = alloca double, i64 1, align 8
  %value.113 = alloca double, i64 1, align 8
  %value.114 = alloca double, i64 1, align 8
  %value.115 = alloca double, i64 1, align 8
  %value.116 = alloca double, i64 1, align 8
  %value.117 = alloca double, i64 1, align 8
  %value.118 = alloca double, i64 1, align 8
  %value.119 = alloca double, i64 1, align 8
  %value.120 = alloca double, i64 1, align 8
  %value.121 = alloca double, i64 1, align 8
  %value.122 = alloca double, i64 1, align 8
  %value.123 = alloca double, i64 1, align 8
  %value.124 = alloca double, i64 1, align 8
  %value.125 = alloca double, i64 1, align 8
  %value.126 = alloca double, i64 1, align 8
  %value.127 = alloca double, i64 1, align 8
  %value.128 = alloca double, i64 1, align 8
  %value.129 = alloca double, i64 1, align 8
  %value.130 = alloca double, i64 1, align 8
  %value.131 = alloca double, i64 1, align 8
  %value.132 = alloca double, i64 1, align 8
  store i32 87, ptr %value.309, align 4
  store i32 86, ptr %value.307, align 4
  store i32 85, ptr %value.305, align 4
  store i32 84, ptr %value.303, align 4
  store i32 83, ptr %value.301, align 4
  store i32 82, ptr %value.299, align 4
  store i32 81, ptr %value.297, align 4
  store i32 80, ptr %value.295, align 4
  store i32 79, ptr %value.293, align 4
  store i32 78, ptr %value.291, align 4
  store i32 77, ptr %value.289, align 4
  store i32 76, ptr %value.287, align 4
  store i32 75, ptr %value.285, align 4
  store i32 74, ptr %value.283, align 4
  store i32 73, ptr %value.281, align 4
  store i32 72, ptr %value.279, align 4
  store i32 71, ptr %value.277, align 4
  store i32 70, ptr %value.275, align 4
  store i32 69, ptr %value.273, align 4
  store i32 68, ptr %value.271, align 4
  store i32 67, ptr %value.269, align 4
  store i32 66, ptr %value.267, align 4
  store i32 65, ptr %value.265, align 4
  store i32 64, ptr %value.263, align 4
  store i32 63, ptr %value.261, align 4
  store i32 62, ptr %value.259, align 4
  store i32 61, ptr %value.257, align 4
  store i32 60, ptr %value.255, align 4
  store i32 59, ptr %value.253, align 4
  store i32 58, ptr %value.251, align 4
  store i32 57, ptr %value.249, align 4
  store i32 56, ptr %value.247, align 4
  store i32 55, ptr %value.245, align 4
  store i32 54, ptr %value.243, align 4
  store i32 53, ptr %value.241, align 4
  store i32 52, ptr %value.239, align 4
  store i32 51, ptr %value.237, align 4
  store i32 50, ptr %value.235, align 4
  store i32 49, ptr %value.233, align 4
  store i32 48, ptr %value.231, align 4
  store i32 47, ptr %value.229, align 4
  store i32 46, ptr %value.227, align 4
  store i32 45, ptr %value.225, align 4
  store i32 44, ptr %value.223, align 4
  store i32 43, ptr %value.221, align 4
  store i32 42, ptr %value.219, align 4
  store i32 41, ptr %value.217, align 4
  store i32 40, ptr %value.215, align 4
  store i32 39, ptr %value.213, align 4
  store i32 38, ptr %value.211, align 4
  store i32 37, ptr %value.209, align 4
  store i32 36, ptr %value.207, align 4
  store i32 35, ptr %value.205, align 4
  store i32 34, ptr %value.203, align 4
  store i32 33, ptr %value.201, align 4
  store i32 32, ptr %value.199, align 4
  store i32 31, ptr %value.197, align 4
  store i32 30, ptr %value.195, align 4
  store i32 29, ptr %value.193, align 4
  store i32 28, ptr %value.191, align 4
  store i32 27, ptr %value.189, align 4
  store i32 26, ptr %value.187, align 4
  store i32 25, ptr %value.185, align 4
  store i32 24, ptr %value.183, align 4
  store i32 23, ptr %value.181, align 4
  store i32 22, ptr %value.179, align 4
  store i32 21, ptr %value.177, align 4
  store i32 20, ptr %value.175, align 4
  store i32 19, ptr %value.173, align 4
  store i32 18, ptr %value.171, align 4
  store i32 17, ptr %value.169, align 4
  store i32 16, ptr %value.167, align 4
  store i32 15, ptr %value.165, align 4
  store i32 14, ptr %value.163, align 4
  store i32 13, ptr %value.161, align 4
  store i32 12, ptr %value.159, align 4
  store i32 11, ptr %value.157, align 4
  store i32 10, ptr %value.155, align 4
  store i32 9, ptr %value.153, align 4
  store i32 8, ptr %value.151, align 4
  store i32 7, ptr %value.149, align 4
  store i32 6, ptr %value.147, align 4
  store i32 5, ptr %value.145, align 4
  store i32 4, ptr %value.143, align 4
  store i32 3, ptr %value.141, align 4
  store i32 2, ptr %value.139, align 4
  store i32 1, ptr %value.137, align 4
  store i64 0, ptr %value.135, align 8
  call void @__ssa_sech_core_pack__sech_core__planned_region_0(ptr %arg.39, ptr %arg.45, ptr %arg.38, ptr %arg.37, ptr %arg.36, ptr %arg.35, ptr %arg.33, ptr %arg.32, ptr %arg.31, ptr %arg.30, ptr %arg.29, ptr %arg.28, ptr %arg.27, ptr %arg.26, ptr %arg.25, ptr %arg.24, ptr %arg.22, ptr %arg.21, ptr %arg.20, ptr %arg.19, ptr %arg.18, ptr %arg.17, ptr %arg.16, ptr %arg.15, ptr %arg.14, ptr %arg.13, ptr %arg.11, ptr %arg.10, ptr %arg.9, ptr %arg.8, ptr %arg.7, ptr %arg.6, ptr %arg.5, ptr %arg.4, ptr %arg.3, ptr %arg.2, ptr %arg.44, ptr %arg.43, ptr %arg.42, ptr %arg.41, ptr %arg.40, ptr %arg.34, ptr %arg.23, ptr %arg.12, ptr %arg.1, ptr %arg.0, ptr %out.0, ptr %value.46, ptr %value.47, ptr %value.48, ptr %value.49, ptr %value.50, ptr %value.51, ptr %value.52, ptr %value.53, ptr %value.54, ptr %value.55, ptr %value.56, ptr %value.57, ptr %value.58, ptr %value.59, ptr %value.60, ptr %value.61, ptr %value.62, ptr %value.63, ptr %value.64, ptr %value.65, ptr %value.66, ptr %value.67, ptr %value.68, ptr %value.69, ptr %value.70, ptr %value.71, ptr %value.72, ptr %value.73, ptr %value.74, ptr %value.75, ptr %value.76, ptr %value.77, ptr %value.78, ptr %value.79, ptr %value.80, ptr %value.81, ptr %value.82, ptr %value.83, ptr %value.84, ptr %value.85, ptr %value.86, ptr %value.87, ptr %value.88, ptr %value.89, ptr %value.90, ptr %value.91, ptr %value.92, ptr %value.93, ptr %value.94, ptr %value.95, ptr %value.96, ptr %value.97, ptr %value.98, ptr %value.99, ptr %value.100, ptr %value.101, ptr %value.102, ptr %value.103, ptr %value.104, ptr %value.105, ptr %value.106, ptr %value.107, ptr %value.108, ptr %value.109, ptr %value.110, ptr %value.111, ptr %value.112, ptr %value.113, ptr %value.114, ptr %value.115, ptr %value.116, ptr %value.117, ptr %value.118, ptr %value.119, ptr %value.120, ptr %value.121, ptr %value.122, ptr %value.123, ptr %value.124, ptr %value.125, ptr %value.126, ptr %value.127, ptr %value.128, ptr %value.129, ptr %value.130, ptr %value.131, ptr %value.132)
  ret void
}

define void @sech_core_pack__sech_core_pack(ptr %buffers, ptr %extents) {
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
  %public.addr.29 = getelementptr ptr, ptr %buffers, i64 29
  %public.29 = load ptr, ptr %public.addr.29, align 8
  %public.addr.30 = getelementptr ptr, ptr %buffers, i64 30
  %public.30 = load ptr, ptr %public.addr.30, align 8
  %public.addr.31 = getelementptr ptr, ptr %buffers, i64 31
  %public.31 = load ptr, ptr %public.addr.31, align 8
  %public.addr.32 = getelementptr ptr, ptr %buffers, i64 32
  %public.32 = load ptr, ptr %public.addr.32, align 8
  %public.addr.33 = getelementptr ptr, ptr %buffers, i64 33
  %public.33 = load ptr, ptr %public.addr.33, align 8
  %public.addr.34 = getelementptr ptr, ptr %buffers, i64 34
  %public.34 = load ptr, ptr %public.addr.34, align 8
  %public.addr.35 = getelementptr ptr, ptr %buffers, i64 35
  %public.35 = load ptr, ptr %public.addr.35, align 8
  %public.addr.36 = getelementptr ptr, ptr %buffers, i64 36
  %public.36 = load ptr, ptr %public.addr.36, align 8
  %public.addr.37 = getelementptr ptr, ptr %buffers, i64 37
  %public.37 = load ptr, ptr %public.addr.37, align 8
  %public.addr.38 = getelementptr ptr, ptr %buffers, i64 38
  %public.38 = load ptr, ptr %public.addr.38, align 8
  %public.addr.39 = getelementptr ptr, ptr %buffers, i64 39
  %public.39 = load ptr, ptr %public.addr.39, align 8
  %public.addr.40 = getelementptr ptr, ptr %buffers, i64 40
  %public.40 = load ptr, ptr %public.addr.40, align 8
  %public.addr.41 = getelementptr ptr, ptr %buffers, i64 41
  %public.41 = load ptr, ptr %public.addr.41, align 8
  %public.addr.42 = getelementptr ptr, ptr %buffers, i64 42
  %public.42 = load ptr, ptr %public.addr.42, align 8
  %public.addr.43 = getelementptr ptr, ptr %buffers, i64 43
  %public.43 = load ptr, ptr %public.addr.43, align 8
  %public.addr.44 = getelementptr ptr, ptr %buffers, i64 44
  %public.44 = load ptr, ptr %public.addr.44, align 8
  %public.addr.45 = getelementptr ptr, ptr %buffers, i64 45
  %public.45 = load ptr, ptr %public.addr.45, align 8
  %public.addr.46 = getelementptr ptr, ptr %buffers, i64 46
  %public.46 = load ptr, ptr %public.addr.46, align 8
  %public.addr.47 = getelementptr ptr, ptr %buffers, i64 47
  %public.47 = load ptr, ptr %public.addr.47, align 8
  call void @__ssa_sech_core_pack__sech_core_pack(ptr %public.0, ptr %public.1, ptr %public.2, ptr %public.3, ptr %public.4, ptr %public.5, ptr %public.6, ptr %public.7, ptr %public.8, ptr %public.9, ptr %public.10, ptr %public.11, ptr %public.12, ptr %public.13, ptr %public.14, ptr %public.15, ptr %public.16, ptr %public.17, ptr %public.18, ptr %public.19, ptr %public.20, ptr %public.21, ptr %public.22, ptr %public.23, ptr %public.24, ptr %public.25, ptr %public.26, ptr %public.27, ptr %public.28, ptr %public.29, ptr %public.30, ptr %public.31, ptr %public.32, ptr %public.33, ptr %public.34, ptr %public.35, ptr %public.36, ptr %public.37, ptr %public.38, ptr %public.39, ptr %public.40, ptr %public.41, ptr %public.42, ptr %public.43, ptr %public.44, ptr %public.45, ptr %public.46, ptr %public.47, ptr %public.2)
  ret void
}
