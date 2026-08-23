source_filename = "turing.ssa-llvm.sinc_core_pack__sinc_core_pack"

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-windows-gnu"

declare double @llvm.fma.f64(double, double, double)

define void @__ssa_sinc_core_pack__sinc_core_pack__planned_region_0(ptr noalias %arg.0, ptr %arg.1, ptr %out.0, ptr %out.1) {
entry:
  %load.0.27.0 = load i32, ptr %arg.1, align 4
  %address.0.27 = getelementptr double, ptr %arg.0, i32 %load.0.27.0
  %pinned.load.1.14 = load double, ptr %address.0.27, align 8
  store double %pinned.load.1.14, ptr %out.1, align 8
  %load.2.15.0 = load double, ptr %out.1, align 8
  %scalar.2.15 = fmul double %load.2.15.0, %load.2.15.0
  store double %scalar.2.15, ptr %out.0, align 8
  ret void
}

define void @__ssa_sinc_core_pack__sinc_core_pack__planned_region_1(ptr %arg.0, ptr %arg.1, ptr %arg.2) {
entry:
  %load.0.28.0 = load i32, ptr %arg.1, align 4
  %address.0.28 = getelementptr double, ptr %arg.0, i32 %load.0.28.0
  %load.store.1.v = load double, ptr %arg.2, align 8
  store double %load.store.1.v, ptr %address.0.28, align 8
  ret void
}

define void @__ssa_sinc_core_pack__sinc_core_pack(ptr noalias %arg.0, ptr noalias %arg.1, ptr %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr noalias %arg.11, ptr noalias %arg.12, ptr noalias %arg.13, ptr noalias %arg.14, ptr noalias %arg.15, ptr noalias %arg.16, ptr noalias %arg.17, ptr noalias %arg.18, ptr noalias %arg.19, ptr noalias %arg.20, ptr %out.0) {
entry:
  %value.19 = alloca i64, i64 1, align 8
  %value.20 = alloca i64, i64 1, align 8
  %value.27 = alloca i32, i64 1, align 8
  %value.25 = alloca i64, i64 1, align 8
  %value.22 = alloca i64, i64 1, align 8
  %value.23 = alloca i1, i64 1, align 8
  %value.15 = alloca double, i64 1, align 8
  %value.14 = alloca double, i64 1, align 8
  %value.16 = alloca double, i64 1, align 8
  store i64 0, ptr %value.19, align 8
  store i64 1, ptr %value.20, align 8
  store i32 1, ptr %value.27, align 4
  store i64 0, ptr %value.25, align 8
  br label %loop_header
loop_header:
  %phi.21 = phi ptr [ %value.19, %entry ], [ %value.22, %loop_latch ]
  %load.6.23.0 = load i32, ptr %phi.21, align 4
  %load.6.23.1 = load i32, ptr %arg.0, align 4
  %scalar.6.23 = icmp slt i32 %load.6.23.0, %load.6.23.1
  store i1 %scalar.6.23, ptr %value.23, align 1
  br i1 %scalar.6.23, label %loop_body, label %loop_exit
loop_body:
  call void @__ssa_sinc_core_pack__sinc_core_pack__planned_region_0(ptr %arg.1, ptr %phi.21, ptr %value.15, ptr %value.14)
  call void @__ssa_sinc_core_pack__sinc_core(ptr %arg.3, ptr %arg.4, ptr %arg.5, ptr %arg.6, ptr %arg.7, ptr %arg.8, ptr %arg.9, ptr %arg.10, ptr %arg.11, ptr %value.15, ptr %arg.12, ptr %arg.13, ptr %arg.14, ptr %arg.15, ptr %arg.16, ptr %arg.17, ptr %arg.18, ptr %arg.19, ptr %arg.20, ptr %value.15, ptr %value.16)
  call void @__ssa_sinc_core_pack__sinc_core_pack__planned_region_1(ptr %arg.2, ptr %phi.21, ptr %value.16)
  br label %loop_latch
loop_latch:
  %load.16.22.0 = load i32, ptr %phi.21, align 4
  %load.16.22.1 = load i64, ptr %value.20, align 8
  %convert.16.22.1 = trunc i64 %load.16.22.1 to i32
  %scalar.16.22 = add i32 %load.16.22.0, %convert.16.22.1
  %declared.16.22 = sext i32 %scalar.16.22 to i64
  store i64 %declared.16.22, ptr %value.22, align 8
  br label %loop_header
loop_exit:
  %return.load.0.27 = load double, ptr %arg.2, align 8
  store double %return.load.0.27, ptr %out.0, align 8
  ret void
}

define void @__ssa_sinc_core_pack__sinc_core__planned_region_0(ptr noalias %arg.0, ptr %arg.1, ptr noalias %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr %arg.11, ptr noalias %arg.12, ptr noalias %arg.13, ptr noalias %arg.14, ptr noalias %arg.15, ptr noalias %arg.16, ptr noalias %arg.17, ptr noalias %arg.18, ptr noalias %arg.19, ptr %out.0, ptr %out.1, ptr %out.2, ptr %out.3, ptr %out.4, ptr %out.5, ptr %out.6, ptr %out.7, ptr %out.8, ptr %out.9, ptr %out.10, ptr %out.11, ptr %out.12, ptr %out.13, ptr %out.14, ptr %out.15) {
entry:
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
  %value.133 = alloca double, i64 1, align 8
  %value.134 = alloca double, i64 1, align 8
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
  %value.146 = alloca double, i64 1, align 8
  %value.147 = alloca double, i64 1, align 8
  %value.148 = alloca double, i64 1, align 8
  %value.149 = alloca double, i64 1, align 8
  %value.150 = alloca double, i64 1, align 8
  %value.151 = alloca double, i64 1, align 8
  %value.152 = alloca double, i64 1, align 8
  %value.153 = alloca double, i64 1, align 8
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
  %value.165 = alloca double, i64 1, align 8
  %value.166 = alloca double, i64 1, align 8
  %value.167 = alloca double, i64 1, align 8
  %value.168 = alloca double, i64 1, align 8
  %value.169 = alloca double, i64 1, align 8
  %value.170 = alloca double, i64 1, align 8
  %value.171 = alloca double, i64 1, align 8
  %value.172 = alloca double, i64 1, align 8
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
  %value.184 = alloca double, i64 1, align 8
  %value.185 = alloca double, i64 1, align 8
  %value.186 = alloca double, i64 1, align 8
  %value.187 = alloca double, i64 1, align 8
  %value.188 = alloca double, i64 1, align 8
  %value.189 = alloca double, i64 1, align 8
  %value.190 = alloca double, i64 1, align 8
  %value.191 = alloca double, i64 1, align 8
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
  %value.203 = alloca double, i64 1, align 8
  %value.204 = alloca double, i64 1, align 8
  %value.205 = alloca double, i64 1, align 8
  %value.206 = alloca double, i64 1, align 8
  %value.207 = alloca double, i64 1, align 8
  %value.208 = alloca double, i64 1, align 8
  %value.209 = alloca double, i64 1, align 8
  %value.210 = alloca double, i64 1, align 8
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
  %value.232 = alloca double, i64 1, align 8
  %value.233 = alloca double, i64 1, align 8
  %value.234 = alloca double, i64 1, align 8
  %value.235 = alloca double, i64 1, align 8
  %value.236 = alloca double, i64 1, align 8
  %value.237 = alloca double, i64 1, align 8
  %value.238 = alloca double, i64 1, align 8
  %value.239 = alloca double, i64 1, align 8
  %value.240 = alloca double, i64 1, align 8
  %value.241 = alloca double, i64 1, align 8
  %value.242 = alloca double, i64 1, align 8
  %value.243 = alloca double, i64 1, align 8
  %value.244 = alloca double, i64 1, align 8
  %value.245 = alloca double, i64 1, align 8
  %value.246 = alloca double, i64 1, align 8
  %value.247 = alloca double, i64 1, align 8
  %value.248 = alloca double, i64 1, align 8
  %value.249 = alloca double, i64 1, align 8
  %value.250 = alloca double, i64 1, align 8
  %value.251 = alloca double, i64 1, align 8
  %value.252 = alloca double, i64 1, align 8
  %value.253 = alloca double, i64 1, align 8
  %value.254 = alloca double, i64 1, align 8
  %value.255 = alloca double, i64 1, align 8
  %load.0.88.0 = load double, ptr %arg.0, align 8
  %load.0.88.1 = load double, ptr %arg.1, align 8
  %scalar.0.88 = fmul double %load.0.88.0, %load.0.88.1
  store double %scalar.0.88, ptr %value.88, align 8
  %scalar.1.89 = fneg double %scalar.0.88
  store double %scalar.1.89, ptr %value.89, align 8
  %scalar.2.90 = call double @llvm.fma.f64(double %load.0.88.0, double %load.0.88.1, double %scalar.1.89)
  store double %scalar.2.90, ptr %value.90, align 8
  %load.3.91.1 = load double, ptr %arg.11, align 8
  %scalar.3.91 = fmul double %load.0.88.0, %load.3.91.1
  store double %scalar.3.91, ptr %value.91, align 8
  %scalar.4.92 = fadd double %scalar.2.90, %scalar.3.91
  store double %scalar.4.92, ptr %value.92, align 8
  %load.5.93.0 = load double, ptr %arg.10, align 8
  %scalar.5.93 = fmul double %load.5.93.0, %load.0.88.1
  store double %scalar.5.93, ptr %value.93, align 8
  %scalar.6.94 = fadd double %scalar.4.92, %scalar.5.93
  store double %scalar.6.94, ptr %value.94, align 8
  %scalar.7.95 = fadd double %scalar.0.88, %scalar.6.94
  store double %scalar.7.95, ptr %value.95, align 8
  %scalar.8.96 = fsub double %scalar.7.95, %scalar.0.88
  store double %scalar.8.96, ptr %value.96, align 8
  %scalar.9.97 = fsub double %scalar.6.94, %scalar.8.96
  store double %scalar.9.97, ptr %value.97, align 8
  %scalar.10.10 = fadd double %scalar.7.95, %scalar.9.97
  store double %scalar.10.10, ptr %out.1, align 8
  %load.11.98.0 = load double, ptr %arg.2, align 8
  %scalar.11.98 = fadd double %load.11.98.0, %scalar.7.95
  store double %scalar.11.98, ptr %value.98, align 8
  %scalar.12.99 = fsub double %scalar.11.98, %load.11.98.0
  store double %scalar.12.99, ptr %value.99, align 8
  %scalar.13.100 = fsub double %scalar.11.98, %scalar.12.99
  store double %scalar.13.100, ptr %value.100, align 8
  %scalar.14.101 = fsub double %load.11.98.0, %scalar.13.100
  store double %scalar.14.101, ptr %value.101, align 8
  %scalar.15.102 = fsub double %scalar.7.95, %scalar.12.99
  store double %scalar.15.102, ptr %value.102, align 8
  %scalar.16.103 = fadd double %scalar.14.101, %scalar.15.102
  store double %scalar.16.103, ptr %value.103, align 8
  %load.17.104.1 = load double, ptr %arg.12, align 8
  %scalar.17.104 = fadd double %scalar.16.103, %load.17.104.1
  store double %scalar.17.104, ptr %value.104, align 8
  %scalar.18.105 = fadd double %scalar.17.104, %scalar.9.97
  store double %scalar.18.105, ptr %value.105, align 8
  %scalar.19.106 = fadd double %scalar.11.98, %scalar.18.105
  store double %scalar.19.106, ptr %value.106, align 8
  %scalar.20.107 = fsub double %scalar.19.106, %scalar.11.98
  store double %scalar.20.107, ptr %value.107, align 8
  %scalar.21.108 = fsub double %scalar.18.105, %scalar.20.107
  store double %scalar.21.108, ptr %value.108, align 8
  %scalar.22.11 = fadd double %scalar.19.106, %scalar.21.108
  store double %scalar.22.11, ptr %out.2, align 8
  %scalar.23.109 = fmul double %load.0.88.1, %scalar.19.106
  store double %scalar.23.109, ptr %value.109, align 8
  %scalar.24.110 = fneg double %scalar.23.109
  store double %scalar.24.110, ptr %value.110, align 8
  %scalar.25.111 = call double @llvm.fma.f64(double %load.0.88.1, double %scalar.19.106, double %scalar.24.110)
  store double %scalar.25.111, ptr %value.111, align 8
  %scalar.26.112 = fmul double %load.0.88.1, %scalar.21.108
  store double %scalar.26.112, ptr %value.112, align 8
  %scalar.27.113 = fadd double %scalar.25.111, %scalar.26.112
  store double %scalar.27.113, ptr %value.113, align 8
  %scalar.28.114 = fmul double %load.3.91.1, %scalar.19.106
  store double %scalar.28.114, ptr %value.114, align 8
  %scalar.29.115 = fadd double %scalar.27.113, %scalar.28.114
  store double %scalar.29.115, ptr %value.115, align 8
  %scalar.30.116 = fadd double %scalar.23.109, %scalar.29.115
  store double %scalar.30.116, ptr %value.116, align 8
  %scalar.31.117 = fsub double %scalar.30.116, %scalar.23.109
  store double %scalar.31.117, ptr %value.117, align 8
  %scalar.32.118 = fsub double %scalar.29.115, %scalar.31.117
  store double %scalar.32.118, ptr %value.118, align 8
  %scalar.33.12 = fadd double %scalar.30.116, %scalar.32.118
  store double %scalar.33.12, ptr %out.3, align 8
  %load.34.119.0 = load double, ptr %arg.3, align 8
  %scalar.34.119 = fadd double %load.34.119.0, %scalar.30.116
  store double %scalar.34.119, ptr %value.119, align 8
  %scalar.35.120 = fsub double %scalar.34.119, %load.34.119.0
  store double %scalar.35.120, ptr %value.120, align 8
  %scalar.36.121 = fsub double %scalar.34.119, %scalar.35.120
  store double %scalar.36.121, ptr %value.121, align 8
  %scalar.37.122 = fsub double %load.34.119.0, %scalar.36.121
  store double %scalar.37.122, ptr %value.122, align 8
  %scalar.38.123 = fsub double %scalar.30.116, %scalar.35.120
  store double %scalar.38.123, ptr %value.123, align 8
  %scalar.39.124 = fadd double %scalar.37.122, %scalar.38.123
  store double %scalar.39.124, ptr %value.124, align 8
  %load.40.125.1 = load double, ptr %arg.13, align 8
  %scalar.40.125 = fadd double %scalar.39.124, %load.40.125.1
  store double %scalar.40.125, ptr %value.125, align 8
  %scalar.41.126 = fadd double %scalar.40.125, %scalar.32.118
  store double %scalar.41.126, ptr %value.126, align 8
  %scalar.42.127 = fadd double %scalar.34.119, %scalar.41.126
  store double %scalar.42.127, ptr %value.127, align 8
  %scalar.43.128 = fsub double %scalar.42.127, %scalar.34.119
  store double %scalar.43.128, ptr %value.128, align 8
  %scalar.44.129 = fsub double %scalar.41.126, %scalar.43.128
  store double %scalar.44.129, ptr %value.129, align 8
  %scalar.45.13 = fadd double %scalar.42.127, %scalar.44.129
  store double %scalar.45.13, ptr %out.4, align 8
  %scalar.46.130 = fmul double %load.0.88.1, %scalar.42.127
  store double %scalar.46.130, ptr %value.130, align 8
  %scalar.47.131 = fneg double %scalar.46.130
  store double %scalar.47.131, ptr %value.131, align 8
  %scalar.48.132 = call double @llvm.fma.f64(double %load.0.88.1, double %scalar.42.127, double %scalar.47.131)
  store double %scalar.48.132, ptr %value.132, align 8
  %scalar.49.133 = fmul double %load.0.88.1, %scalar.44.129
  store double %scalar.49.133, ptr %value.133, align 8
  %scalar.50.134 = fadd double %scalar.48.132, %scalar.49.133
  store double %scalar.50.134, ptr %value.134, align 8
  %scalar.51.135 = fmul double %load.3.91.1, %scalar.42.127
  store double %scalar.51.135, ptr %value.135, align 8
  %scalar.52.136 = fadd double %scalar.50.134, %scalar.51.135
  store double %scalar.52.136, ptr %value.136, align 8
  %scalar.53.137 = fadd double %scalar.46.130, %scalar.52.136
  store double %scalar.53.137, ptr %value.137, align 8
  %scalar.54.138 = fsub double %scalar.53.137, %scalar.46.130
  store double %scalar.54.138, ptr %value.138, align 8
  %scalar.55.139 = fsub double %scalar.52.136, %scalar.54.138
  store double %scalar.55.139, ptr %value.139, align 8
  %scalar.56.14 = fadd double %scalar.53.137, %scalar.55.139
  store double %scalar.56.14, ptr %out.5, align 8
  %load.57.140.0 = load double, ptr %arg.4, align 8
  %scalar.57.140 = fadd double %load.57.140.0, %scalar.53.137
  store double %scalar.57.140, ptr %value.140, align 8
  %scalar.58.141 = fsub double %scalar.57.140, %load.57.140.0
  store double %scalar.58.141, ptr %value.141, align 8
  %scalar.59.142 = fsub double %scalar.57.140, %scalar.58.141
  store double %scalar.59.142, ptr %value.142, align 8
  %scalar.60.143 = fsub double %load.57.140.0, %scalar.59.142
  store double %scalar.60.143, ptr %value.143, align 8
  %scalar.61.144 = fsub double %scalar.53.137, %scalar.58.141
  store double %scalar.61.144, ptr %value.144, align 8
  %scalar.62.145 = fadd double %scalar.60.143, %scalar.61.144
  store double %scalar.62.145, ptr %value.145, align 8
  %load.63.146.1 = load double, ptr %arg.14, align 8
  %scalar.63.146 = fadd double %scalar.62.145, %load.63.146.1
  store double %scalar.63.146, ptr %value.146, align 8
  %scalar.64.147 = fadd double %scalar.63.146, %scalar.55.139
  store double %scalar.64.147, ptr %value.147, align 8
  %scalar.65.148 = fadd double %scalar.57.140, %scalar.64.147
  store double %scalar.65.148, ptr %value.148, align 8
  %scalar.66.149 = fsub double %scalar.65.148, %scalar.57.140
  store double %scalar.66.149, ptr %value.149, align 8
  %scalar.67.150 = fsub double %scalar.64.147, %scalar.66.149
  store double %scalar.67.150, ptr %value.150, align 8
  %scalar.68.15 = fadd double %scalar.65.148, %scalar.67.150
  store double %scalar.68.15, ptr %out.6, align 8
  %scalar.69.151 = fmul double %load.0.88.1, %scalar.65.148
  store double %scalar.69.151, ptr %value.151, align 8
  %scalar.70.152 = fneg double %scalar.69.151
  store double %scalar.70.152, ptr %value.152, align 8
  %scalar.71.153 = call double @llvm.fma.f64(double %load.0.88.1, double %scalar.65.148, double %scalar.70.152)
  store double %scalar.71.153, ptr %value.153, align 8
  %scalar.72.154 = fmul double %load.0.88.1, %scalar.67.150
  store double %scalar.72.154, ptr %value.154, align 8
  %scalar.73.155 = fadd double %scalar.71.153, %scalar.72.154
  store double %scalar.73.155, ptr %value.155, align 8
  %scalar.74.156 = fmul double %load.3.91.1, %scalar.65.148
  store double %scalar.74.156, ptr %value.156, align 8
  %scalar.75.157 = fadd double %scalar.73.155, %scalar.74.156
  store double %scalar.75.157, ptr %value.157, align 8
  %scalar.76.158 = fadd double %scalar.69.151, %scalar.75.157
  store double %scalar.76.158, ptr %value.158, align 8
  %scalar.77.159 = fsub double %scalar.76.158, %scalar.69.151
  store double %scalar.77.159, ptr %value.159, align 8
  %scalar.78.160 = fsub double %scalar.75.157, %scalar.77.159
  store double %scalar.78.160, ptr %value.160, align 8
  %scalar.79.16 = fadd double %scalar.76.158, %scalar.78.160
  store double %scalar.79.16, ptr %out.7, align 8
  %load.80.161.0 = load double, ptr %arg.5, align 8
  %scalar.80.161 = fadd double %load.80.161.0, %scalar.76.158
  store double %scalar.80.161, ptr %value.161, align 8
  %scalar.81.162 = fsub double %scalar.80.161, %load.80.161.0
  store double %scalar.81.162, ptr %value.162, align 8
  %scalar.82.163 = fsub double %scalar.80.161, %scalar.81.162
  store double %scalar.82.163, ptr %value.163, align 8
  %scalar.83.164 = fsub double %load.80.161.0, %scalar.82.163
  store double %scalar.83.164, ptr %value.164, align 8
  %scalar.84.165 = fsub double %scalar.76.158, %scalar.81.162
  store double %scalar.84.165, ptr %value.165, align 8
  %scalar.85.166 = fadd double %scalar.83.164, %scalar.84.165
  store double %scalar.85.166, ptr %value.166, align 8
  %load.86.167.1 = load double, ptr %arg.15, align 8
  %scalar.86.167 = fadd double %scalar.85.166, %load.86.167.1
  store double %scalar.86.167, ptr %value.167, align 8
  %scalar.87.168 = fadd double %scalar.86.167, %scalar.78.160
  store double %scalar.87.168, ptr %value.168, align 8
  %scalar.88.169 = fadd double %scalar.80.161, %scalar.87.168
  store double %scalar.88.169, ptr %value.169, align 8
  %scalar.89.170 = fsub double %scalar.88.169, %scalar.80.161
  store double %scalar.89.170, ptr %value.170, align 8
  %scalar.90.171 = fsub double %scalar.87.168, %scalar.89.170
  store double %scalar.90.171, ptr %value.171, align 8
  %scalar.91.17 = fadd double %scalar.88.169, %scalar.90.171
  store double %scalar.91.17, ptr %out.8, align 8
  %scalar.92.172 = fmul double %load.0.88.1, %scalar.88.169
  store double %scalar.92.172, ptr %value.172, align 8
  %scalar.93.173 = fneg double %scalar.92.172
  store double %scalar.93.173, ptr %value.173, align 8
  %scalar.94.174 = call double @llvm.fma.f64(double %load.0.88.1, double %scalar.88.169, double %scalar.93.173)
  store double %scalar.94.174, ptr %value.174, align 8
  %scalar.95.175 = fmul double %load.0.88.1, %scalar.90.171
  store double %scalar.95.175, ptr %value.175, align 8
  %scalar.96.176 = fadd double %scalar.94.174, %scalar.95.175
  store double %scalar.96.176, ptr %value.176, align 8
  %scalar.97.177 = fmul double %load.3.91.1, %scalar.88.169
  store double %scalar.97.177, ptr %value.177, align 8
  %scalar.98.178 = fadd double %scalar.96.176, %scalar.97.177
  store double %scalar.98.178, ptr %value.178, align 8
  %scalar.99.179 = fadd double %scalar.92.172, %scalar.98.178
  store double %scalar.99.179, ptr %value.179, align 8
  %scalar.100.180 = fsub double %scalar.99.179, %scalar.92.172
  store double %scalar.100.180, ptr %value.180, align 8
  %scalar.101.181 = fsub double %scalar.98.178, %scalar.100.180
  store double %scalar.101.181, ptr %value.181, align 8
  %scalar.102.18 = fadd double %scalar.99.179, %scalar.101.181
  store double %scalar.102.18, ptr %out.9, align 8
  %load.103.182.0 = load double, ptr %arg.6, align 8
  %scalar.103.182 = fadd double %load.103.182.0, %scalar.99.179
  store double %scalar.103.182, ptr %value.182, align 8
  %scalar.104.183 = fsub double %scalar.103.182, %load.103.182.0
  store double %scalar.104.183, ptr %value.183, align 8
  %scalar.105.184 = fsub double %scalar.103.182, %scalar.104.183
  store double %scalar.105.184, ptr %value.184, align 8
  %scalar.106.185 = fsub double %load.103.182.0, %scalar.105.184
  store double %scalar.106.185, ptr %value.185, align 8
  %scalar.107.186 = fsub double %scalar.99.179, %scalar.104.183
  store double %scalar.107.186, ptr %value.186, align 8
  %scalar.108.187 = fadd double %scalar.106.185, %scalar.107.186
  store double %scalar.108.187, ptr %value.187, align 8
  %load.109.188.1 = load double, ptr %arg.16, align 8
  %scalar.109.188 = fadd double %scalar.108.187, %load.109.188.1
  store double %scalar.109.188, ptr %value.188, align 8
  %scalar.110.189 = fadd double %scalar.109.188, %scalar.101.181
  store double %scalar.110.189, ptr %value.189, align 8
  %scalar.111.190 = fadd double %scalar.103.182, %scalar.110.189
  store double %scalar.111.190, ptr %value.190, align 8
  %scalar.112.191 = fsub double %scalar.111.190, %scalar.103.182
  store double %scalar.112.191, ptr %value.191, align 8
  %scalar.113.192 = fsub double %scalar.110.189, %scalar.112.191
  store double %scalar.113.192, ptr %value.192, align 8
  %scalar.114.19 = fadd double %scalar.111.190, %scalar.113.192
  store double %scalar.114.19, ptr %out.10, align 8
  %scalar.115.193 = fmul double %load.0.88.1, %scalar.111.190
  store double %scalar.115.193, ptr %value.193, align 8
  %scalar.116.194 = fneg double %scalar.115.193
  store double %scalar.116.194, ptr %value.194, align 8
  %scalar.117.195 = call double @llvm.fma.f64(double %load.0.88.1, double %scalar.111.190, double %scalar.116.194)
  store double %scalar.117.195, ptr %value.195, align 8
  %scalar.118.196 = fmul double %load.0.88.1, %scalar.113.192
  store double %scalar.118.196, ptr %value.196, align 8
  %scalar.119.197 = fadd double %scalar.117.195, %scalar.118.196
  store double %scalar.119.197, ptr %value.197, align 8
  %scalar.120.198 = fmul double %load.3.91.1, %scalar.111.190
  store double %scalar.120.198, ptr %value.198, align 8
  %scalar.121.199 = fadd double %scalar.119.197, %scalar.120.198
  store double %scalar.121.199, ptr %value.199, align 8
  %scalar.122.200 = fadd double %scalar.115.193, %scalar.121.199
  store double %scalar.122.200, ptr %value.200, align 8
  %scalar.123.201 = fsub double %scalar.122.200, %scalar.115.193
  store double %scalar.123.201, ptr %value.201, align 8
  %scalar.124.202 = fsub double %scalar.121.199, %scalar.123.201
  store double %scalar.124.202, ptr %value.202, align 8
  %scalar.125.20 = fadd double %scalar.122.200, %scalar.124.202
  store double %scalar.125.20, ptr %out.11, align 8
  %load.126.203.0 = load double, ptr %arg.7, align 8
  %scalar.126.203 = fadd double %load.126.203.0, %scalar.122.200
  store double %scalar.126.203, ptr %value.203, align 8
  %scalar.127.204 = fsub double %scalar.126.203, %load.126.203.0
  store double %scalar.127.204, ptr %value.204, align 8
  %scalar.128.205 = fsub double %scalar.126.203, %scalar.127.204
  store double %scalar.128.205, ptr %value.205, align 8
  %scalar.129.206 = fsub double %load.126.203.0, %scalar.128.205
  store double %scalar.129.206, ptr %value.206, align 8
  %scalar.130.207 = fsub double %scalar.122.200, %scalar.127.204
  store double %scalar.130.207, ptr %value.207, align 8
  %scalar.131.208 = fadd double %scalar.129.206, %scalar.130.207
  store double %scalar.131.208, ptr %value.208, align 8
  %load.132.209.1 = load double, ptr %arg.17, align 8
  %scalar.132.209 = fadd double %scalar.131.208, %load.132.209.1
  store double %scalar.132.209, ptr %value.209, align 8
  %scalar.133.210 = fadd double %scalar.132.209, %scalar.124.202
  store double %scalar.133.210, ptr %value.210, align 8
  %scalar.134.211 = fadd double %scalar.126.203, %scalar.133.210
  store double %scalar.134.211, ptr %value.211, align 8
  %scalar.135.212 = fsub double %scalar.134.211, %scalar.126.203
  store double %scalar.135.212, ptr %value.212, align 8
  %scalar.136.213 = fsub double %scalar.133.210, %scalar.135.212
  store double %scalar.136.213, ptr %value.213, align 8
  %scalar.137.21 = fadd double %scalar.134.211, %scalar.136.213
  store double %scalar.137.21, ptr %out.12, align 8
  %scalar.138.214 = fmul double %load.0.88.1, %scalar.134.211
  store double %scalar.138.214, ptr %value.214, align 8
  %scalar.139.215 = fneg double %scalar.138.214
  store double %scalar.139.215, ptr %value.215, align 8
  %scalar.140.216 = call double @llvm.fma.f64(double %load.0.88.1, double %scalar.134.211, double %scalar.139.215)
  store double %scalar.140.216, ptr %value.216, align 8
  %scalar.141.217 = fmul double %load.0.88.1, %scalar.136.213
  store double %scalar.141.217, ptr %value.217, align 8
  %scalar.142.218 = fadd double %scalar.140.216, %scalar.141.217
  store double %scalar.142.218, ptr %value.218, align 8
  %scalar.143.219 = fmul double %load.3.91.1, %scalar.134.211
  store double %scalar.143.219, ptr %value.219, align 8
  %scalar.144.220 = fadd double %scalar.142.218, %scalar.143.219
  store double %scalar.144.220, ptr %value.220, align 8
  %scalar.145.221 = fadd double %scalar.138.214, %scalar.144.220
  store double %scalar.145.221, ptr %value.221, align 8
  %scalar.146.222 = fsub double %scalar.145.221, %scalar.138.214
  store double %scalar.146.222, ptr %value.222, align 8
  %scalar.147.223 = fsub double %scalar.144.220, %scalar.146.222
  store double %scalar.147.223, ptr %value.223, align 8
  %scalar.148.22 = fadd double %scalar.145.221, %scalar.147.223
  store double %scalar.148.22, ptr %out.13, align 8
  %load.149.224.0 = load double, ptr %arg.8, align 8
  %scalar.149.224 = fadd double %load.149.224.0, %scalar.145.221
  store double %scalar.149.224, ptr %value.224, align 8
  %scalar.150.225 = fsub double %scalar.149.224, %load.149.224.0
  store double %scalar.150.225, ptr %value.225, align 8
  %scalar.151.226 = fsub double %scalar.149.224, %scalar.150.225
  store double %scalar.151.226, ptr %value.226, align 8
  %scalar.152.227 = fsub double %load.149.224.0, %scalar.151.226
  store double %scalar.152.227, ptr %value.227, align 8
  %scalar.153.228 = fsub double %scalar.145.221, %scalar.150.225
  store double %scalar.153.228, ptr %value.228, align 8
  %scalar.154.229 = fadd double %scalar.152.227, %scalar.153.228
  store double %scalar.154.229, ptr %value.229, align 8
  %load.155.230.1 = load double, ptr %arg.18, align 8
  %scalar.155.230 = fadd double %scalar.154.229, %load.155.230.1
  store double %scalar.155.230, ptr %value.230, align 8
  %scalar.156.231 = fadd double %scalar.155.230, %scalar.147.223
  store double %scalar.156.231, ptr %value.231, align 8
  %scalar.157.232 = fadd double %scalar.149.224, %scalar.156.231
  store double %scalar.157.232, ptr %value.232, align 8
  %scalar.158.233 = fsub double %scalar.157.232, %scalar.149.224
  store double %scalar.158.233, ptr %value.233, align 8
  %scalar.159.234 = fsub double %scalar.156.231, %scalar.158.233
  store double %scalar.159.234, ptr %value.234, align 8
  %scalar.160.23 = fadd double %scalar.157.232, %scalar.159.234
  store double %scalar.160.23, ptr %out.14, align 8
  %scalar.161.235 = fmul double %load.0.88.1, %scalar.157.232
  store double %scalar.161.235, ptr %value.235, align 8
  %scalar.162.236 = fneg double %scalar.161.235
  store double %scalar.162.236, ptr %value.236, align 8
  %scalar.163.237 = call double @llvm.fma.f64(double %load.0.88.1, double %scalar.157.232, double %scalar.162.236)
  store double %scalar.163.237, ptr %value.237, align 8
  %scalar.164.238 = fmul double %load.0.88.1, %scalar.159.234
  store double %scalar.164.238, ptr %value.238, align 8
  %scalar.165.239 = fadd double %scalar.163.237, %scalar.164.238
  store double %scalar.165.239, ptr %value.239, align 8
  %scalar.166.240 = fmul double %load.3.91.1, %scalar.157.232
  store double %scalar.166.240, ptr %value.240, align 8
  %scalar.167.241 = fadd double %scalar.165.239, %scalar.166.240
  store double %scalar.167.241, ptr %value.241, align 8
  %scalar.168.242 = fadd double %scalar.161.235, %scalar.167.241
  store double %scalar.168.242, ptr %value.242, align 8
  %scalar.169.243 = fsub double %scalar.168.242, %scalar.161.235
  store double %scalar.169.243, ptr %value.243, align 8
  %scalar.170.244 = fsub double %scalar.167.241, %scalar.169.243
  store double %scalar.170.244, ptr %value.244, align 8
  %scalar.171.24 = fadd double %scalar.168.242, %scalar.170.244
  store double %scalar.171.24, ptr %out.15, align 8
  %load.172.245.0 = load double, ptr %arg.9, align 8
  %scalar.172.245 = fadd double %load.172.245.0, %scalar.168.242
  store double %scalar.172.245, ptr %value.245, align 8
  %scalar.173.246 = fsub double %scalar.172.245, %load.172.245.0
  store double %scalar.173.246, ptr %value.246, align 8
  %scalar.174.247 = fsub double %scalar.172.245, %scalar.173.246
  store double %scalar.174.247, ptr %value.247, align 8
  %scalar.175.248 = fsub double %load.172.245.0, %scalar.174.247
  store double %scalar.175.248, ptr %value.248, align 8
  %scalar.176.249 = fsub double %scalar.168.242, %scalar.173.246
  store double %scalar.176.249, ptr %value.249, align 8
  %scalar.177.250 = fadd double %scalar.175.248, %scalar.176.249
  store double %scalar.177.250, ptr %value.250, align 8
  %load.178.251.1 = load double, ptr %arg.19, align 8
  %scalar.178.251 = fadd double %scalar.177.250, %load.178.251.1
  store double %scalar.178.251, ptr %value.251, align 8
  %scalar.179.252 = fadd double %scalar.178.251, %scalar.170.244
  store double %scalar.179.252, ptr %value.252, align 8
  %scalar.180.253 = fadd double %scalar.172.245, %scalar.179.252
  store double %scalar.180.253, ptr %value.253, align 8
  %scalar.181.254 = fsub double %scalar.180.253, %scalar.172.245
  store double %scalar.181.254, ptr %value.254, align 8
  %scalar.182.255 = fsub double %scalar.179.252, %scalar.181.254
  store double %scalar.182.255, ptr %value.255, align 8
  %scalar.183.25 = fadd double %scalar.180.253, %scalar.182.255
  store double %scalar.183.25, ptr %out.0, align 8
  ret void
}

define void @__ssa_sinc_core_pack__sinc_core(ptr noalias %arg.0, ptr noalias %arg.1, ptr noalias %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr %arg.9, ptr noalias %arg.10, ptr noalias %arg.11, ptr noalias %arg.12, ptr noalias %arg.13, ptr noalias %arg.14, ptr noalias %arg.15, ptr noalias %arg.16, ptr noalias %arg.17, ptr noalias %arg.18, ptr %arg.19, ptr %out.0) {
entry:
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
  %value.33 = alloca i32, i64 1, align 8
  %value.31 = alloca i32, i64 1, align 8
  %value.29 = alloca i32, i64 1, align 8
  %value.27 = alloca i64, i64 1, align 8
  %value.10 = alloca double, i64 1, align 8
  %value.11 = alloca double, i64 1, align 8
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
  store i32 15, ptr %value.57, align 4
  store i32 14, ptr %value.55, align 4
  store i32 13, ptr %value.53, align 4
  store i32 12, ptr %value.51, align 4
  store i32 11, ptr %value.49, align 4
  store i32 10, ptr %value.47, align 4
  store i32 9, ptr %value.45, align 4
  store i32 8, ptr %value.43, align 4
  store i32 7, ptr %value.41, align 4
  store i32 6, ptr %value.39, align 4
  store i32 5, ptr %value.37, align 4
  store i32 4, ptr %value.35, align 4
  store i32 3, ptr %value.33, align 4
  store i32 2, ptr %value.31, align 4
  store i32 1, ptr %value.29, align 4
  store i64 0, ptr %value.27, align 8
  call void @__ssa_sinc_core_pack__sinc_core__planned_region_0(ptr %arg.8, ptr %arg.9, ptr %arg.7, ptr %arg.6, ptr %arg.5, ptr %arg.4, ptr %arg.3, ptr %arg.2, ptr %arg.1, ptr %arg.0, ptr %arg.18, ptr %arg.19, ptr %arg.17, ptr %arg.16, ptr %arg.15, ptr %arg.14, ptr %arg.13, ptr %arg.12, ptr %arg.11, ptr %arg.10, ptr %out.0, ptr %value.10, ptr %value.11, ptr %value.12, ptr %value.13, ptr %value.14, ptr %value.15, ptr %value.16, ptr %value.17, ptr %value.18, ptr %value.19, ptr %value.20, ptr %value.21, ptr %value.22, ptr %value.23, ptr %value.24)
  ret void
}

define void @sinc_core_pack__sinc_core_pack(ptr %buffers, ptr %extents) {
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
  call void @__ssa_sinc_core_pack__sinc_core_pack(ptr %public.0, ptr %public.1, ptr %public.2, ptr %public.3, ptr %public.4, ptr %public.5, ptr %public.6, ptr %public.7, ptr %public.8, ptr %public.9, ptr %public.10, ptr %public.11, ptr %public.12, ptr %public.13, ptr %public.14, ptr %public.15, ptr %public.16, ptr %public.17, ptr %public.18, ptr %public.19, ptr %public.20, ptr %public.2)
  ret void
}
