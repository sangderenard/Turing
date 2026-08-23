source_filename = "turing.ssa-llvm.sin_core_pack__sin_core_pack"

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-windows-gnu"

declare double @llvm.fma.f64(double, double, double)

define void @__ssa_sin_core_pack__sin_core_pack__planned_region_0(ptr noalias %arg.0, ptr %arg.1, ptr noalias %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr %arg.11, ptr noalias %arg.12, ptr noalias %arg.13, ptr noalias %arg.14, ptr noalias %arg.15, ptr noalias %arg.16, ptr noalias %arg.17, ptr noalias %arg.18, ptr noalias %arg.19, ptr noalias %arg.20) {
entry:
  %value.60 = alloca i32, i64 1, align 8
  %value.61 = alloca i32, i64 1, align 8
  %value.63 = alloca double, i64 1, align 8
  %value.64 = alloca i32, i64 1, align 8
  %value.65 = alloca i32, i64 1, align 8
  %value.66 = alloca i32, i64 1, align 8
  %value.67 = alloca i32, i64 1, align 8
  %value.69 = alloca double, i64 1, align 8
  %value.14 = alloca double, i64 1, align 8
  %value.15 = alloca double, i64 1, align 8
  %value.70 = alloca double, i64 1, align 8
  %value.71 = alloca double, i64 1, align 8
  %value.72 = alloca double, i64 1, align 8
  %value.73 = alloca double, i64 1, align 8
  %value.74 = alloca double, i64 1, align 8
  %value.75 = alloca double, i64 1, align 8
  %value.76 = alloca double, i64 1, align 8
  %value.77 = alloca double, i64 1, align 8
  %value.16 = alloca double, i64 1, align 8
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
  %value.17 = alloca double, i64 1, align 8
  %value.89 = alloca double, i64 1, align 8
  %value.90 = alloca double, i64 1, align 8
  %value.91 = alloca double, i64 1, align 8
  %value.92 = alloca double, i64 1, align 8
  %value.93 = alloca double, i64 1, align 8
  %value.94 = alloca double, i64 1, align 8
  %value.95 = alloca double, i64 1, align 8
  %value.96 = alloca double, i64 1, align 8
  %value.18 = alloca double, i64 1, align 8
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
  %value.19 = alloca double, i64 1, align 8
  %value.108 = alloca double, i64 1, align 8
  %value.109 = alloca double, i64 1, align 8
  %value.110 = alloca double, i64 1, align 8
  %value.111 = alloca double, i64 1, align 8
  %value.112 = alloca double, i64 1, align 8
  %value.113 = alloca double, i64 1, align 8
  %value.114 = alloca double, i64 1, align 8
  %value.115 = alloca double, i64 1, align 8
  %value.20 = alloca double, i64 1, align 8
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
  %value.21 = alloca double, i64 1, align 8
  %value.127 = alloca double, i64 1, align 8
  %value.128 = alloca double, i64 1, align 8
  %value.129 = alloca double, i64 1, align 8
  %value.130 = alloca double, i64 1, align 8
  %value.131 = alloca double, i64 1, align 8
  %value.132 = alloca double, i64 1, align 8
  %value.133 = alloca double, i64 1, align 8
  %value.134 = alloca double, i64 1, align 8
  %value.22 = alloca double, i64 1, align 8
  %value.135 = alloca double, i64 1, align 8
  %value.136 = alloca double, i64 1, align 8
  %value.137 = alloca double, i64 1, align 8
  %value.138 = alloca double, i64 1, align 8
  %value.139 = alloca double, i64 1, align 8
  %value.140 = alloca double, i64 1, align 8
  %value.141 = alloca double, i64 1, align 8
  %value.142 = alloca double, i64 1, align 8
  %value.143 = alloca double, i64 1, align 8
  %value.144 = alloca double, i64 1, align 8
  %value.145 = alloca double, i64 1, align 8
  %value.23 = alloca double, i64 1, align 8
  %value.146 = alloca double, i64 1, align 8
  %value.147 = alloca double, i64 1, align 8
  %value.148 = alloca double, i64 1, align 8
  %value.149 = alloca double, i64 1, align 8
  %value.150 = alloca double, i64 1, align 8
  %value.151 = alloca double, i64 1, align 8
  %value.152 = alloca double, i64 1, align 8
  %value.153 = alloca double, i64 1, align 8
  %value.24 = alloca double, i64 1, align 8
  %value.154 = alloca double, i64 1, align 8
  %value.155 = alloca double, i64 1, align 8
  %value.156 = alloca double, i64 1, align 8
  %value.157 = alloca double, i64 1, align 8
  %value.158 = alloca double, i64 1, align 8
  %value.159 = alloca double, i64 1, align 8
  %value.160 = alloca double, i64 1, align 8
  %value.161 = alloca double, i64 1, align 8
  %value.162 = alloca double, i64 1, align 8
  %value.163 = alloca double, i64 1, align 8
  %value.164 = alloca double, i64 1, align 8
  %value.25 = alloca double, i64 1, align 8
  %value.165 = alloca double, i64 1, align 8
  %value.166 = alloca double, i64 1, align 8
  %value.167 = alloca double, i64 1, align 8
  %value.168 = alloca double, i64 1, align 8
  %value.169 = alloca double, i64 1, align 8
  %value.170 = alloca double, i64 1, align 8
  %value.171 = alloca double, i64 1, align 8
  %value.172 = alloca double, i64 1, align 8
  %value.26 = alloca double, i64 1, align 8
  %value.173 = alloca double, i64 1, align 8
  %value.174 = alloca double, i64 1, align 8
  %value.175 = alloca double, i64 1, align 8
  %value.176 = alloca double, i64 1, align 8
  %value.177 = alloca double, i64 1, align 8
  %value.178 = alloca double, i64 1, align 8
  %value.179 = alloca double, i64 1, align 8
  %value.180 = alloca double, i64 1, align 8
  %value.181 = alloca double, i64 1, align 8
  %value.182 = alloca double, i64 1, align 8
  %value.183 = alloca double, i64 1, align 8
  %value.27 = alloca double, i64 1, align 8
  %value.184 = alloca double, i64 1, align 8
  %value.185 = alloca double, i64 1, align 8
  %value.186 = alloca double, i64 1, align 8
  %value.187 = alloca double, i64 1, align 8
  %value.188 = alloca double, i64 1, align 8
  %value.189 = alloca double, i64 1, align 8
  %value.190 = alloca double, i64 1, align 8
  %value.191 = alloca double, i64 1, align 8
  %value.28 = alloca double, i64 1, align 8
  %value.192 = alloca double, i64 1, align 8
  %value.193 = alloca double, i64 1, align 8
  %value.194 = alloca double, i64 1, align 8
  %value.195 = alloca double, i64 1, align 8
  %value.196 = alloca double, i64 1, align 8
  %value.197 = alloca double, i64 1, align 8
  %value.198 = alloca double, i64 1, align 8
  %value.199 = alloca double, i64 1, align 8
  %value.200 = alloca double, i64 1, align 8
  %value.201 = alloca double, i64 1, align 8
  %value.202 = alloca double, i64 1, align 8
  %value.29 = alloca double, i64 1, align 8
  %value.203 = alloca double, i64 1, align 8
  %value.204 = alloca double, i64 1, align 8
  %value.205 = alloca double, i64 1, align 8
  %value.206 = alloca double, i64 1, align 8
  %value.207 = alloca double, i64 1, align 8
  %value.208 = alloca double, i64 1, align 8
  %value.209 = alloca double, i64 1, align 8
  %value.210 = alloca double, i64 1, align 8
  %value.30 = alloca double, i64 1, align 8
  %value.211 = alloca double, i64 1, align 8
  %value.212 = alloca double, i64 1, align 8
  %value.213 = alloca double, i64 1, align 8
  %value.214 = alloca double, i64 1, align 8
  %value.215 = alloca double, i64 1, align 8
  %value.216 = alloca double, i64 1, align 8
  %value.217 = alloca double, i64 1, align 8
  %value.218 = alloca double, i64 1, align 8
  %value.219 = alloca double, i64 1, align 8
  %value.220 = alloca double, i64 1, align 8
  %value.221 = alloca double, i64 1, align 8
  %value.31 = alloca double, i64 1, align 8
  %value.222 = alloca double, i64 1, align 8
  %value.223 = alloca double, i64 1, align 8
  %value.224 = alloca double, i64 1, align 8
  %value.225 = alloca double, i64 1, align 8
  %value.226 = alloca double, i64 1, align 8
  %value.227 = alloca double, i64 1, align 8
  %value.228 = alloca double, i64 1, align 8
  %value.229 = alloca double, i64 1, align 8
  %value.230 = alloca double, i64 1, align 8
  %value.231 = alloca double, i64 1, align 8
  %value.32 = alloca double, i64 1, align 8
  %value.232 = alloca i32, i64 1, align 8
  %value.233 = alloca i32, i64 1, align 8
  %value.235 = alloca i32, i64 1, align 8
  %value.236 = alloca i32, i64 1, align 8
  %value.237 = alloca i32, i64 1, align 8
  %value.238 = alloca i32, i64 1, align 8
  %load.0.40.0 = load i32, ptr %arg.1, align 4
  %address.0.40 = getelementptr double, ptr %arg.0, i32 %load.0.40.0
  store i32 2, ptr %value.60, align 4
  %scalar.2.61 = mul i32 %load.0.40.0, 2
  store i32 %scalar.2.61, ptr %value.61, align 4
  %address.3.62 = getelementptr double, ptr %arg.0, i32 %scalar.2.61
  %pinned.load.4.63 = load double, ptr %address.3.62, align 8
  store double %pinned.load.4.63, ptr %value.63, align 8
  store i32 2, ptr %value.64, align 4
  %scalar.6.65 = mul i32 %load.0.40.0, 2
  store i32 %scalar.6.65, ptr %value.65, align 4
  store i32 1, ptr %value.66, align 4
  %scalar.8.67 = add i32 %scalar.6.65, 1
  store i32 %scalar.8.67, ptr %value.67, align 4
  %address.9.68 = getelementptr double, ptr %arg.0, i32 %scalar.8.67
  %pinned.load.10.69 = load double, ptr %address.9.68, align 8
  store double %pinned.load.10.69, ptr %value.69, align 8
  %load.11.15.0 = load double, ptr %value.14, align 8
  %scalar.11.15 = fmul double %load.11.15.0, %load.11.15.0
  store double %scalar.11.15, ptr %value.15, align 8
  %load.12.70.0 = load double, ptr %arg.2, align 8
  %scalar.12.70 = fmul double %load.12.70.0, %scalar.11.15
  store double %scalar.12.70, ptr %value.70, align 8
  %scalar.13.71 = fneg double %scalar.12.70
  store double %scalar.13.71, ptr %value.71, align 8
  %scalar.14.72 = call double @llvm.fma.f64(double %load.12.70.0, double %scalar.11.15, double %scalar.13.71)
  store double %scalar.14.72, ptr %value.72, align 8
  %load.15.73.0 = load double, ptr %arg.12, align 8
  %scalar.15.73 = fmul double %load.15.73.0, %scalar.11.15
  store double %scalar.15.73, ptr %value.73, align 8
  %scalar.16.74 = fadd double %scalar.14.72, %scalar.15.73
  store double %scalar.16.74, ptr %value.74, align 8
  %scalar.17.75 = fadd double %scalar.12.70, %scalar.16.74
  store double %scalar.17.75, ptr %value.75, align 8
  %scalar.18.76 = fsub double %scalar.17.75, %scalar.12.70
  store double %scalar.18.76, ptr %value.76, align 8
  %scalar.19.77 = fsub double %scalar.16.74, %scalar.18.76
  store double %scalar.19.77, ptr %value.77, align 8
  %scalar.20.16 = fadd double %scalar.17.75, %scalar.19.77
  store double %scalar.20.16, ptr %value.16, align 8
  %load.21.78.0 = load double, ptr %arg.3, align 8
  %scalar.21.78 = fadd double %load.21.78.0, %scalar.17.75
  store double %scalar.21.78, ptr %value.78, align 8
  %scalar.22.79 = fsub double %scalar.21.78, %load.21.78.0
  store double %scalar.22.79, ptr %value.79, align 8
  %scalar.23.80 = fsub double %scalar.21.78, %scalar.22.79
  store double %scalar.23.80, ptr %value.80, align 8
  %scalar.24.81 = fsub double %load.21.78.0, %scalar.23.80
  store double %scalar.24.81, ptr %value.81, align 8
  %scalar.25.82 = fsub double %scalar.17.75, %scalar.22.79
  store double %scalar.25.82, ptr %value.82, align 8
  %scalar.26.83 = fadd double %scalar.24.81, %scalar.25.82
  store double %scalar.26.83, ptr %value.83, align 8
  %load.27.84.1 = load double, ptr %arg.13, align 8
  %scalar.27.84 = fadd double %scalar.26.83, %load.27.84.1
  store double %scalar.27.84, ptr %value.84, align 8
  %scalar.28.85 = fadd double %scalar.27.84, %scalar.19.77
  store double %scalar.28.85, ptr %value.85, align 8
  %scalar.29.86 = fadd double %scalar.21.78, %scalar.28.85
  store double %scalar.29.86, ptr %value.86, align 8
  %scalar.30.87 = fsub double %scalar.29.86, %scalar.21.78
  store double %scalar.30.87, ptr %value.87, align 8
  %scalar.31.88 = fsub double %scalar.28.85, %scalar.30.87
  store double %scalar.31.88, ptr %value.88, align 8
  %scalar.32.17 = fadd double %scalar.29.86, %scalar.31.88
  store double %scalar.32.17, ptr %value.17, align 8
  %scalar.33.89 = fmul double %scalar.11.15, %scalar.29.86
  store double %scalar.33.89, ptr %value.89, align 8
  %scalar.34.90 = fneg double %scalar.33.89
  store double %scalar.34.90, ptr %value.90, align 8
  %scalar.35.91 = call double @llvm.fma.f64(double %scalar.11.15, double %scalar.29.86, double %scalar.34.90)
  store double %scalar.35.91, ptr %value.91, align 8
  %scalar.36.92 = fmul double %scalar.11.15, %scalar.31.88
  store double %scalar.36.92, ptr %value.92, align 8
  %scalar.37.93 = fadd double %scalar.35.91, %scalar.36.92
  store double %scalar.37.93, ptr %value.93, align 8
  %scalar.38.94 = fadd double %scalar.33.89, %scalar.37.93
  store double %scalar.38.94, ptr %value.94, align 8
  %scalar.39.95 = fsub double %scalar.38.94, %scalar.33.89
  store double %scalar.39.95, ptr %value.95, align 8
  %scalar.40.96 = fsub double %scalar.37.93, %scalar.39.95
  store double %scalar.40.96, ptr %value.96, align 8
  %scalar.41.18 = fadd double %scalar.38.94, %scalar.40.96
  store double %scalar.41.18, ptr %value.18, align 8
  %load.42.97.0 = load double, ptr %arg.4, align 8
  %scalar.42.97 = fadd double %load.42.97.0, %scalar.38.94
  store double %scalar.42.97, ptr %value.97, align 8
  %scalar.43.98 = fsub double %scalar.42.97, %load.42.97.0
  store double %scalar.43.98, ptr %value.98, align 8
  %scalar.44.99 = fsub double %scalar.42.97, %scalar.43.98
  store double %scalar.44.99, ptr %value.99, align 8
  %scalar.45.100 = fsub double %load.42.97.0, %scalar.44.99
  store double %scalar.45.100, ptr %value.100, align 8
  %scalar.46.101 = fsub double %scalar.38.94, %scalar.43.98
  store double %scalar.46.101, ptr %value.101, align 8
  %scalar.47.102 = fadd double %scalar.45.100, %scalar.46.101
  store double %scalar.47.102, ptr %value.102, align 8
  %load.48.103.1 = load double, ptr %arg.14, align 8
  %scalar.48.103 = fadd double %scalar.47.102, %load.48.103.1
  store double %scalar.48.103, ptr %value.103, align 8
  %scalar.49.104 = fadd double %scalar.48.103, %scalar.40.96
  store double %scalar.49.104, ptr %value.104, align 8
  %scalar.50.105 = fadd double %scalar.42.97, %scalar.49.104
  store double %scalar.50.105, ptr %value.105, align 8
  %scalar.51.106 = fsub double %scalar.50.105, %scalar.42.97
  store double %scalar.51.106, ptr %value.106, align 8
  %scalar.52.107 = fsub double %scalar.49.104, %scalar.51.106
  store double %scalar.52.107, ptr %value.107, align 8
  %scalar.53.19 = fadd double %scalar.50.105, %scalar.52.107
  store double %scalar.53.19, ptr %value.19, align 8
  %scalar.54.108 = fmul double %scalar.11.15, %scalar.50.105
  store double %scalar.54.108, ptr %value.108, align 8
  %scalar.55.109 = fneg double %scalar.54.108
  store double %scalar.55.109, ptr %value.109, align 8
  %scalar.56.110 = call double @llvm.fma.f64(double %scalar.11.15, double %scalar.50.105, double %scalar.55.109)
  store double %scalar.56.110, ptr %value.110, align 8
  %scalar.57.111 = fmul double %scalar.11.15, %scalar.52.107
  store double %scalar.57.111, ptr %value.111, align 8
  %scalar.58.112 = fadd double %scalar.56.110, %scalar.57.111
  store double %scalar.58.112, ptr %value.112, align 8
  %scalar.59.113 = fadd double %scalar.54.108, %scalar.58.112
  store double %scalar.59.113, ptr %value.113, align 8
  %scalar.60.114 = fsub double %scalar.59.113, %scalar.54.108
  store double %scalar.60.114, ptr %value.114, align 8
  %scalar.61.115 = fsub double %scalar.58.112, %scalar.60.114
  store double %scalar.61.115, ptr %value.115, align 8
  %scalar.62.20 = fadd double %scalar.59.113, %scalar.61.115
  store double %scalar.62.20, ptr %value.20, align 8
  %load.63.116.0 = load double, ptr %arg.5, align 8
  %scalar.63.116 = fadd double %load.63.116.0, %scalar.59.113
  store double %scalar.63.116, ptr %value.116, align 8
  %scalar.64.117 = fsub double %scalar.63.116, %load.63.116.0
  store double %scalar.64.117, ptr %value.117, align 8
  %scalar.65.118 = fsub double %scalar.63.116, %scalar.64.117
  store double %scalar.65.118, ptr %value.118, align 8
  %scalar.66.119 = fsub double %load.63.116.0, %scalar.65.118
  store double %scalar.66.119, ptr %value.119, align 8
  %scalar.67.120 = fsub double %scalar.59.113, %scalar.64.117
  store double %scalar.67.120, ptr %value.120, align 8
  %scalar.68.121 = fadd double %scalar.66.119, %scalar.67.120
  store double %scalar.68.121, ptr %value.121, align 8
  %load.69.122.1 = load double, ptr %arg.15, align 8
  %scalar.69.122 = fadd double %scalar.68.121, %load.69.122.1
  store double %scalar.69.122, ptr %value.122, align 8
  %scalar.70.123 = fadd double %scalar.69.122, %scalar.61.115
  store double %scalar.70.123, ptr %value.123, align 8
  %scalar.71.124 = fadd double %scalar.63.116, %scalar.70.123
  store double %scalar.71.124, ptr %value.124, align 8
  %scalar.72.125 = fsub double %scalar.71.124, %scalar.63.116
  store double %scalar.72.125, ptr %value.125, align 8
  %scalar.73.126 = fsub double %scalar.70.123, %scalar.72.125
  store double %scalar.73.126, ptr %value.126, align 8
  %scalar.74.21 = fadd double %scalar.71.124, %scalar.73.126
  store double %scalar.74.21, ptr %value.21, align 8
  %scalar.75.127 = fmul double %scalar.11.15, %scalar.71.124
  store double %scalar.75.127, ptr %value.127, align 8
  %scalar.76.128 = fneg double %scalar.75.127
  store double %scalar.76.128, ptr %value.128, align 8
  %scalar.77.129 = call double @llvm.fma.f64(double %scalar.11.15, double %scalar.71.124, double %scalar.76.128)
  store double %scalar.77.129, ptr %value.129, align 8
  %scalar.78.130 = fmul double %scalar.11.15, %scalar.73.126
  store double %scalar.78.130, ptr %value.130, align 8
  %scalar.79.131 = fadd double %scalar.77.129, %scalar.78.130
  store double %scalar.79.131, ptr %value.131, align 8
  %scalar.80.132 = fadd double %scalar.75.127, %scalar.79.131
  store double %scalar.80.132, ptr %value.132, align 8
  %scalar.81.133 = fsub double %scalar.80.132, %scalar.75.127
  store double %scalar.81.133, ptr %value.133, align 8
  %scalar.82.134 = fsub double %scalar.79.131, %scalar.81.133
  store double %scalar.82.134, ptr %value.134, align 8
  %scalar.83.22 = fadd double %scalar.80.132, %scalar.82.134
  store double %scalar.83.22, ptr %value.22, align 8
  %load.84.135.0 = load double, ptr %arg.6, align 8
  %scalar.84.135 = fadd double %load.84.135.0, %scalar.80.132
  store double %scalar.84.135, ptr %value.135, align 8
  %scalar.85.136 = fsub double %scalar.84.135, %load.84.135.0
  store double %scalar.85.136, ptr %value.136, align 8
  %scalar.86.137 = fsub double %scalar.84.135, %scalar.85.136
  store double %scalar.86.137, ptr %value.137, align 8
  %scalar.87.138 = fsub double %load.84.135.0, %scalar.86.137
  store double %scalar.87.138, ptr %value.138, align 8
  %scalar.88.139 = fsub double %scalar.80.132, %scalar.85.136
  store double %scalar.88.139, ptr %value.139, align 8
  %scalar.89.140 = fadd double %scalar.87.138, %scalar.88.139
  store double %scalar.89.140, ptr %value.140, align 8
  %load.90.141.1 = load double, ptr %arg.16, align 8
  %scalar.90.141 = fadd double %scalar.89.140, %load.90.141.1
  store double %scalar.90.141, ptr %value.141, align 8
  %scalar.91.142 = fadd double %scalar.90.141, %scalar.82.134
  store double %scalar.91.142, ptr %value.142, align 8
  %scalar.92.143 = fadd double %scalar.84.135, %scalar.91.142
  store double %scalar.92.143, ptr %value.143, align 8
  %scalar.93.144 = fsub double %scalar.92.143, %scalar.84.135
  store double %scalar.93.144, ptr %value.144, align 8
  %scalar.94.145 = fsub double %scalar.91.142, %scalar.93.144
  store double %scalar.94.145, ptr %value.145, align 8
  %scalar.95.23 = fadd double %scalar.92.143, %scalar.94.145
  store double %scalar.95.23, ptr %value.23, align 8
  %scalar.96.146 = fmul double %scalar.11.15, %scalar.92.143
  store double %scalar.96.146, ptr %value.146, align 8
  %scalar.97.147 = fneg double %scalar.96.146
  store double %scalar.97.147, ptr %value.147, align 8
  %scalar.98.148 = call double @llvm.fma.f64(double %scalar.11.15, double %scalar.92.143, double %scalar.97.147)
  store double %scalar.98.148, ptr %value.148, align 8
  %scalar.99.149 = fmul double %scalar.11.15, %scalar.94.145
  store double %scalar.99.149, ptr %value.149, align 8
  %scalar.100.150 = fadd double %scalar.98.148, %scalar.99.149
  store double %scalar.100.150, ptr %value.150, align 8
  %scalar.101.151 = fadd double %scalar.96.146, %scalar.100.150
  store double %scalar.101.151, ptr %value.151, align 8
  %scalar.102.152 = fsub double %scalar.101.151, %scalar.96.146
  store double %scalar.102.152, ptr %value.152, align 8
  %scalar.103.153 = fsub double %scalar.100.150, %scalar.102.152
  store double %scalar.103.153, ptr %value.153, align 8
  %scalar.104.24 = fadd double %scalar.101.151, %scalar.103.153
  store double %scalar.104.24, ptr %value.24, align 8
  %load.105.154.0 = load double, ptr %arg.7, align 8
  %scalar.105.154 = fadd double %load.105.154.0, %scalar.101.151
  store double %scalar.105.154, ptr %value.154, align 8
  %scalar.106.155 = fsub double %scalar.105.154, %load.105.154.0
  store double %scalar.106.155, ptr %value.155, align 8
  %scalar.107.156 = fsub double %scalar.105.154, %scalar.106.155
  store double %scalar.107.156, ptr %value.156, align 8
  %scalar.108.157 = fsub double %load.105.154.0, %scalar.107.156
  store double %scalar.108.157, ptr %value.157, align 8
  %scalar.109.158 = fsub double %scalar.101.151, %scalar.106.155
  store double %scalar.109.158, ptr %value.158, align 8
  %scalar.110.159 = fadd double %scalar.108.157, %scalar.109.158
  store double %scalar.110.159, ptr %value.159, align 8
  %load.111.160.1 = load double, ptr %arg.17, align 8
  %scalar.111.160 = fadd double %scalar.110.159, %load.111.160.1
  store double %scalar.111.160, ptr %value.160, align 8
  %scalar.112.161 = fadd double %scalar.111.160, %scalar.103.153
  store double %scalar.112.161, ptr %value.161, align 8
  %scalar.113.162 = fadd double %scalar.105.154, %scalar.112.161
  store double %scalar.113.162, ptr %value.162, align 8
  %scalar.114.163 = fsub double %scalar.113.162, %scalar.105.154
  store double %scalar.114.163, ptr %value.163, align 8
  %scalar.115.164 = fsub double %scalar.112.161, %scalar.114.163
  store double %scalar.115.164, ptr %value.164, align 8
  %scalar.116.25 = fadd double %scalar.113.162, %scalar.115.164
  store double %scalar.116.25, ptr %value.25, align 8
  %scalar.117.165 = fmul double %scalar.11.15, %scalar.113.162
  store double %scalar.117.165, ptr %value.165, align 8
  %scalar.118.166 = fneg double %scalar.117.165
  store double %scalar.118.166, ptr %value.166, align 8
  %scalar.119.167 = call double @llvm.fma.f64(double %scalar.11.15, double %scalar.113.162, double %scalar.118.166)
  store double %scalar.119.167, ptr %value.167, align 8
  %scalar.120.168 = fmul double %scalar.11.15, %scalar.115.164
  store double %scalar.120.168, ptr %value.168, align 8
  %scalar.121.169 = fadd double %scalar.119.167, %scalar.120.168
  store double %scalar.121.169, ptr %value.169, align 8
  %scalar.122.170 = fadd double %scalar.117.165, %scalar.121.169
  store double %scalar.122.170, ptr %value.170, align 8
  %scalar.123.171 = fsub double %scalar.122.170, %scalar.117.165
  store double %scalar.123.171, ptr %value.171, align 8
  %scalar.124.172 = fsub double %scalar.121.169, %scalar.123.171
  store double %scalar.124.172, ptr %value.172, align 8
  %scalar.125.26 = fadd double %scalar.122.170, %scalar.124.172
  store double %scalar.125.26, ptr %value.26, align 8
  %load.126.173.0 = load double, ptr %arg.8, align 8
  %scalar.126.173 = fadd double %load.126.173.0, %scalar.122.170
  store double %scalar.126.173, ptr %value.173, align 8
  %scalar.127.174 = fsub double %scalar.126.173, %load.126.173.0
  store double %scalar.127.174, ptr %value.174, align 8
  %scalar.128.175 = fsub double %scalar.126.173, %scalar.127.174
  store double %scalar.128.175, ptr %value.175, align 8
  %scalar.129.176 = fsub double %load.126.173.0, %scalar.128.175
  store double %scalar.129.176, ptr %value.176, align 8
  %scalar.130.177 = fsub double %scalar.122.170, %scalar.127.174
  store double %scalar.130.177, ptr %value.177, align 8
  %scalar.131.178 = fadd double %scalar.129.176, %scalar.130.177
  store double %scalar.131.178, ptr %value.178, align 8
  %load.132.179.1 = load double, ptr %arg.18, align 8
  %scalar.132.179 = fadd double %scalar.131.178, %load.132.179.1
  store double %scalar.132.179, ptr %value.179, align 8
  %scalar.133.180 = fadd double %scalar.132.179, %scalar.124.172
  store double %scalar.133.180, ptr %value.180, align 8
  %scalar.134.181 = fadd double %scalar.126.173, %scalar.133.180
  store double %scalar.134.181, ptr %value.181, align 8
  %scalar.135.182 = fsub double %scalar.134.181, %scalar.126.173
  store double %scalar.135.182, ptr %value.182, align 8
  %scalar.136.183 = fsub double %scalar.133.180, %scalar.135.182
  store double %scalar.136.183, ptr %value.183, align 8
  %scalar.137.27 = fadd double %scalar.134.181, %scalar.136.183
  store double %scalar.137.27, ptr %value.27, align 8
  %scalar.138.184 = fmul double %scalar.11.15, %scalar.134.181
  store double %scalar.138.184, ptr %value.184, align 8
  %scalar.139.185 = fneg double %scalar.138.184
  store double %scalar.139.185, ptr %value.185, align 8
  %scalar.140.186 = call double @llvm.fma.f64(double %scalar.11.15, double %scalar.134.181, double %scalar.139.185)
  store double %scalar.140.186, ptr %value.186, align 8
  %scalar.141.187 = fmul double %scalar.11.15, %scalar.136.183
  store double %scalar.141.187, ptr %value.187, align 8
  %scalar.142.188 = fadd double %scalar.140.186, %scalar.141.187
  store double %scalar.142.188, ptr %value.188, align 8
  %scalar.143.189 = fadd double %scalar.138.184, %scalar.142.188
  store double %scalar.143.189, ptr %value.189, align 8
  %scalar.144.190 = fsub double %scalar.143.189, %scalar.138.184
  store double %scalar.144.190, ptr %value.190, align 8
  %scalar.145.191 = fsub double %scalar.142.188, %scalar.144.190
  store double %scalar.145.191, ptr %value.191, align 8
  %scalar.146.28 = fadd double %scalar.143.189, %scalar.145.191
  store double %scalar.146.28, ptr %value.28, align 8
  %load.147.192.0 = load double, ptr %arg.9, align 8
  %scalar.147.192 = fadd double %load.147.192.0, %scalar.143.189
  store double %scalar.147.192, ptr %value.192, align 8
  %scalar.148.193 = fsub double %scalar.147.192, %load.147.192.0
  store double %scalar.148.193, ptr %value.193, align 8
  %scalar.149.194 = fsub double %scalar.147.192, %scalar.148.193
  store double %scalar.149.194, ptr %value.194, align 8
  %scalar.150.195 = fsub double %load.147.192.0, %scalar.149.194
  store double %scalar.150.195, ptr %value.195, align 8
  %scalar.151.196 = fsub double %scalar.143.189, %scalar.148.193
  store double %scalar.151.196, ptr %value.196, align 8
  %scalar.152.197 = fadd double %scalar.150.195, %scalar.151.196
  store double %scalar.152.197, ptr %value.197, align 8
  %load.153.198.1 = load double, ptr %arg.19, align 8
  %scalar.153.198 = fadd double %scalar.152.197, %load.153.198.1
  store double %scalar.153.198, ptr %value.198, align 8
  %scalar.154.199 = fadd double %scalar.153.198, %scalar.145.191
  store double %scalar.154.199, ptr %value.199, align 8
  %scalar.155.200 = fadd double %scalar.147.192, %scalar.154.199
  store double %scalar.155.200, ptr %value.200, align 8
  %scalar.156.201 = fsub double %scalar.155.200, %scalar.147.192
  store double %scalar.156.201, ptr %value.201, align 8
  %scalar.157.202 = fsub double %scalar.154.199, %scalar.156.201
  store double %scalar.157.202, ptr %value.202, align 8
  %scalar.158.29 = fadd double %scalar.155.200, %scalar.157.202
  store double %scalar.158.29, ptr %value.29, align 8
  %scalar.159.203 = fmul double %scalar.11.15, %scalar.155.200
  store double %scalar.159.203, ptr %value.203, align 8
  %scalar.160.204 = fneg double %scalar.159.203
  store double %scalar.160.204, ptr %value.204, align 8
  %scalar.161.205 = call double @llvm.fma.f64(double %scalar.11.15, double %scalar.155.200, double %scalar.160.204)
  store double %scalar.161.205, ptr %value.205, align 8
  %scalar.162.206 = fmul double %scalar.11.15, %scalar.157.202
  store double %scalar.162.206, ptr %value.206, align 8
  %scalar.163.207 = fadd double %scalar.161.205, %scalar.162.206
  store double %scalar.163.207, ptr %value.207, align 8
  %scalar.164.208 = fadd double %scalar.159.203, %scalar.163.207
  store double %scalar.164.208, ptr %value.208, align 8
  %scalar.165.209 = fsub double %scalar.164.208, %scalar.159.203
  store double %scalar.165.209, ptr %value.209, align 8
  %scalar.166.210 = fsub double %scalar.163.207, %scalar.165.209
  store double %scalar.166.210, ptr %value.210, align 8
  %scalar.167.30 = fadd double %scalar.164.208, %scalar.166.210
  store double %scalar.167.30, ptr %value.30, align 8
  %load.168.211.0 = load double, ptr %arg.10, align 8
  %scalar.168.211 = fadd double %load.168.211.0, %scalar.164.208
  store double %scalar.168.211, ptr %value.211, align 8
  %scalar.169.212 = fsub double %scalar.168.211, %load.168.211.0
  store double %scalar.169.212, ptr %value.212, align 8
  %scalar.170.213 = fsub double %scalar.168.211, %scalar.169.212
  store double %scalar.170.213, ptr %value.213, align 8
  %scalar.171.214 = fsub double %load.168.211.0, %scalar.170.213
  store double %scalar.171.214, ptr %value.214, align 8
  %scalar.172.215 = fsub double %scalar.164.208, %scalar.169.212
  store double %scalar.172.215, ptr %value.215, align 8
  %scalar.173.216 = fadd double %scalar.171.214, %scalar.172.215
  store double %scalar.173.216, ptr %value.216, align 8
  %load.174.217.1 = load double, ptr %arg.20, align 8
  %scalar.174.217 = fadd double %scalar.173.216, %load.174.217.1
  store double %scalar.174.217, ptr %value.217, align 8
  %scalar.175.218 = fadd double %scalar.174.217, %scalar.166.210
  store double %scalar.175.218, ptr %value.218, align 8
  %scalar.176.219 = fadd double %scalar.168.211, %scalar.175.218
  store double %scalar.176.219, ptr %value.219, align 8
  %scalar.177.220 = fsub double %scalar.176.219, %scalar.168.211
  store double %scalar.177.220, ptr %value.220, align 8
  %scalar.178.221 = fsub double %scalar.175.218, %scalar.177.220
  store double %scalar.178.221, ptr %value.221, align 8
  %scalar.179.31 = fadd double %scalar.176.219, %scalar.178.221
  store double %scalar.179.31, ptr %value.31, align 8
  %load.180.222.0 = load double, ptr %value.63, align 8
  %scalar.180.222 = fmul double %load.180.222.0, %scalar.176.219
  store double %scalar.180.222, ptr %value.222, align 8
  %scalar.181.223 = fneg double %scalar.180.222
  store double %scalar.181.223, ptr %value.223, align 8
  %scalar.182.224 = call double @llvm.fma.f64(double %load.180.222.0, double %scalar.176.219, double %scalar.181.223)
  store double %scalar.182.224, ptr %value.224, align 8
  %scalar.183.225 = fmul double %load.180.222.0, %scalar.178.221
  store double %scalar.183.225, ptr %value.225, align 8
  %scalar.184.226 = fadd double %scalar.182.224, %scalar.183.225
  store double %scalar.184.226, ptr %value.226, align 8
  %load.185.227.0 = load double, ptr %value.69, align 8
  %scalar.185.227 = fmul double %load.185.227.0, %scalar.176.219
  store double %scalar.185.227, ptr %value.227, align 8
  %scalar.186.228 = fadd double %scalar.184.226, %scalar.185.227
  store double %scalar.186.228, ptr %value.228, align 8
  %scalar.187.229 = fadd double %scalar.180.222, %scalar.186.228
  store double %scalar.187.229, ptr %value.229, align 8
  %scalar.188.230 = fsub double %scalar.187.229, %scalar.180.222
  store double %scalar.188.230, ptr %value.230, align 8
  %scalar.189.231 = fsub double %scalar.186.228, %scalar.188.230
  store double %scalar.189.231, ptr %value.231, align 8
  %scalar.190.32 = fadd double %scalar.187.229, %scalar.189.231
  store double %scalar.190.32, ptr %value.32, align 8
  %address.191.41 = getelementptr double, ptr %arg.11, i32 %load.0.40.0
  store i32 2, ptr %value.232, align 4
  %scalar.193.233 = mul i32 %load.0.40.0, 2
  store i32 %scalar.193.233, ptr %value.233, align 4
  %address.194.234 = getelementptr double, ptr %arg.11, i32 %scalar.193.233
  store double %scalar.187.229, ptr %address.194.234, align 8
  store i32 2, ptr %value.235, align 4
  %load.197.236.0 = load i32, ptr %arg.1, align 4
  %scalar.197.236 = mul i32 %load.197.236.0, 2
  store i32 %scalar.197.236, ptr %value.236, align 4
  store i32 1, ptr %value.237, align 4
  %scalar.199.238 = add i32 %scalar.197.236, 1
  store i32 %scalar.199.238, ptr %value.238, align 4
  %address.200.239 = getelementptr double, ptr %arg.11, i32 %scalar.199.238
  %load.store.201.v = load double, ptr %value.231, align 8
  store double %load.store.201.v, ptr %address.200.239, align 8
  ret void
}

define void @__ssa_sin_core_pack__sin_core_pack(ptr noalias %arg.0, ptr noalias %arg.1, ptr %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr noalias %arg.11, ptr noalias %arg.12, ptr noalias %arg.13, ptr noalias %arg.14, ptr noalias %arg.15, ptr noalias %arg.16, ptr noalias %arg.17, ptr noalias %arg.18, ptr noalias %arg.19, ptr noalias %arg.20, ptr %out.0) {
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
  call void @__ssa_sin_core_pack__sin_core_pack__planned_region_0(ptr %arg.1, ptr %phi.37, ptr %arg.11, ptr %arg.10, ptr %arg.9, ptr %arg.8, ptr %arg.7, ptr %arg.6, ptr %arg.5, ptr %arg.4, ptr %arg.3, ptr %arg.2, ptr %arg.20, ptr %arg.19, ptr %arg.18, ptr %arg.17, ptr %arg.16, ptr %arg.15, ptr %arg.14, ptr %arg.13, ptr %arg.12)
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
  call void @__ssa_sin_core_pack__sin_core_pack(ptr %public.0, ptr %public.1, ptr %public.2, ptr %public.3, ptr %public.4, ptr %public.5, ptr %public.6, ptr %public.7, ptr %public.8, ptr %public.9, ptr %public.10, ptr %public.11, ptr %public.12, ptr %public.13, ptr %public.14, ptr %public.15, ptr %public.16, ptr %public.17, ptr %public.18, ptr %public.19, ptr %public.20, ptr %public.2)
  ret void
}
