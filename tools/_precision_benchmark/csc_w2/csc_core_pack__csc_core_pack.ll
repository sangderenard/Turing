source_filename = "turing.ssa-llvm.csc_core_pack__csc_core_pack"

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-windows-gnu"

declare double @llvm.fma.f64(double, double, double)

define void @__ssa_csc_core_pack__csc_core_pack__planned_region_0(ptr noalias %arg.0, ptr %arg.1, ptr %out.0, ptr %out.1) {
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

define void @__ssa_csc_core_pack__csc_core_pack__planned_region_1(ptr %arg.0, ptr %arg.1, ptr %arg.2) {
entry:
  %load.0.30.0 = load i32, ptr %arg.1, align 4
  %address.0.30 = getelementptr double, ptr %arg.0, i32 %load.0.30.0
  %load.store.1.v = load double, ptr %arg.2, align 8
  store double %load.store.1.v, ptr %address.0.30, align 8
  ret void
}

define void @__ssa_csc_core_pack__csc_core_pack(ptr noalias %arg.0, ptr noalias %arg.1, ptr %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr noalias %arg.11, ptr noalias %arg.12, ptr noalias %arg.13, ptr noalias %arg.14, ptr noalias %arg.15, ptr noalias %arg.16, ptr noalias %arg.17, ptr noalias %arg.18, ptr noalias %arg.19, ptr noalias %arg.20, ptr noalias %arg.21, ptr noalias %arg.22, ptr noalias %arg.23, ptr noalias %arg.24, ptr %out.0) {
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
  call void @__ssa_csc_core_pack__csc_core_pack__planned_region_0(ptr %arg.1, ptr %phi.23, ptr %value.17, ptr %value.16)
  call void @__ssa_csc_core_pack__csc_core(ptr %arg.3, ptr %arg.4, ptr %arg.5, ptr %arg.6, ptr %arg.7, ptr %arg.8, ptr %arg.9, ptr %arg.10, ptr %arg.11, ptr %arg.12, ptr %arg.13, ptr %value.17, ptr %arg.14, ptr %arg.15, ptr %arg.16, ptr %arg.17, ptr %arg.18, ptr %arg.19, ptr %arg.20, ptr %arg.21, ptr %arg.22, ptr %arg.23, ptr %arg.24, ptr %value.17, ptr %value.18)
  call void @__ssa_csc_core_pack__csc_core_pack__planned_region_1(ptr %arg.2, ptr %phi.23, ptr %value.18)
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

define void @__ssa_csc_core_pack__csc_core__planned_region_0(ptr noalias %arg.0, ptr %arg.1, ptr noalias %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr noalias %arg.11, ptr noalias %arg.12, ptr %arg.13, ptr noalias %arg.14, ptr noalias %arg.15, ptr noalias %arg.16, ptr noalias %arg.17, ptr noalias %arg.18, ptr noalias %arg.19, ptr noalias %arg.20, ptr noalias %arg.21, ptr noalias %arg.22, ptr noalias %arg.23, ptr %out.0, ptr %out.1, ptr %out.2, ptr %out.3, ptr %out.4, ptr %out.5, ptr %out.6, ptr %out.7, ptr %out.8, ptr %out.9, ptr %out.10, ptr %out.11, ptr %out.12, ptr %out.13, ptr %out.14, ptr %out.15, ptr %out.16, ptr %out.17, ptr %out.18, ptr %out.19) {
entry:
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
  %value.256 = alloca double, i64 1, align 8
  %value.257 = alloca double, i64 1, align 8
  %value.258 = alloca double, i64 1, align 8
  %value.259 = alloca double, i64 1, align 8
  %value.260 = alloca double, i64 1, align 8
  %value.261 = alloca double, i64 1, align 8
  %value.262 = alloca double, i64 1, align 8
  %value.263 = alloca double, i64 1, align 8
  %value.264 = alloca double, i64 1, align 8
  %value.265 = alloca double, i64 1, align 8
  %value.266 = alloca double, i64 1, align 8
  %value.267 = alloca double, i64 1, align 8
  %value.268 = alloca double, i64 1, align 8
  %value.269 = alloca double, i64 1, align 8
  %value.270 = alloca double, i64 1, align 8
  %value.271 = alloca double, i64 1, align 8
  %value.272 = alloca double, i64 1, align 8
  %value.273 = alloca double, i64 1, align 8
  %value.274 = alloca double, i64 1, align 8
  %value.275 = alloca double, i64 1, align 8
  %value.276 = alloca double, i64 1, align 8
  %value.277 = alloca double, i64 1, align 8
  %value.278 = alloca double, i64 1, align 8
  %value.279 = alloca double, i64 1, align 8
  %value.280 = alloca double, i64 1, align 8
  %value.281 = alloca double, i64 1, align 8
  %value.282 = alloca double, i64 1, align 8
  %value.283 = alloca double, i64 1, align 8
  %value.284 = alloca double, i64 1, align 8
  %value.285 = alloca double, i64 1, align 8
  %value.286 = alloca double, i64 1, align 8
  %value.287 = alloca double, i64 1, align 8
  %value.288 = alloca double, i64 1, align 8
  %value.289 = alloca double, i64 1, align 8
  %value.290 = alloca double, i64 1, align 8
  %value.291 = alloca double, i64 1, align 8
  %value.292 = alloca double, i64 1, align 8
  %value.293 = alloca double, i64 1, align 8
  %value.294 = alloca double, i64 1, align 8
  %value.295 = alloca double, i64 1, align 8
  %value.296 = alloca double, i64 1, align 8
  %value.297 = alloca double, i64 1, align 8
  %value.298 = alloca double, i64 1, align 8
  %value.299 = alloca double, i64 1, align 8
  %value.300 = alloca double, i64 1, align 8
  %value.301 = alloca double, i64 1, align 8
  %value.302 = alloca double, i64 1, align 8
  %value.303 = alloca double, i64 1, align 8
  %value.304 = alloca double, i64 1, align 8
  %value.305 = alloca double, i64 1, align 8
  %value.306 = alloca double, i64 1, align 8
  %value.307 = alloca double, i64 1, align 8
  %value.308 = alloca double, i64 1, align 8
  %value.309 = alloca double, i64 1, align 8
  %value.310 = alloca double, i64 1, align 8
  %value.311 = alloca double, i64 1, align 8
  %value.312 = alloca double, i64 1, align 8
  %value.313 = alloca double, i64 1, align 8
  %value.314 = alloca double, i64 1, align 8
  %value.315 = alloca double, i64 1, align 8
  %value.316 = alloca double, i64 1, align 8
  %value.317 = alloca double, i64 1, align 8
  %load.0.108.0 = load double, ptr %arg.0, align 8
  %load.0.108.1 = load double, ptr %arg.1, align 8
  %scalar.0.108 = fmul double %load.0.108.0, %load.0.108.1
  store double %scalar.0.108, ptr %value.108, align 8
  %scalar.1.109 = fneg double %scalar.0.108
  store double %scalar.1.109, ptr %value.109, align 8
  %scalar.2.110 = call double @llvm.fma.f64(double %load.0.108.0, double %load.0.108.1, double %scalar.1.109)
  store double %scalar.2.110, ptr %value.110, align 8
  %load.3.111.1 = load double, ptr %arg.13, align 8
  %scalar.3.111 = fmul double %load.0.108.0, %load.3.111.1
  store double %scalar.3.111, ptr %value.111, align 8
  %scalar.4.112 = fadd double %scalar.2.110, %scalar.3.111
  store double %scalar.4.112, ptr %value.112, align 8
  %load.5.113.0 = load double, ptr %arg.12, align 8
  %scalar.5.113 = fmul double %load.5.113.0, %load.0.108.1
  store double %scalar.5.113, ptr %value.113, align 8
  %scalar.6.114 = fadd double %scalar.4.112, %scalar.5.113
  store double %scalar.6.114, ptr %value.114, align 8
  %scalar.7.115 = fadd double %scalar.0.108, %scalar.6.114
  store double %scalar.7.115, ptr %value.115, align 8
  %scalar.8.116 = fsub double %scalar.7.115, %scalar.0.108
  store double %scalar.8.116, ptr %value.116, align 8
  %scalar.9.117 = fsub double %scalar.6.114, %scalar.8.116
  store double %scalar.9.117, ptr %value.117, align 8
  %scalar.10.12 = fadd double %scalar.7.115, %scalar.9.117
  store double %scalar.10.12, ptr %out.1, align 8
  %load.11.118.0 = load double, ptr %arg.2, align 8
  %scalar.11.118 = fadd double %load.11.118.0, %scalar.7.115
  store double %scalar.11.118, ptr %value.118, align 8
  %scalar.12.119 = fsub double %scalar.11.118, %load.11.118.0
  store double %scalar.12.119, ptr %value.119, align 8
  %scalar.13.120 = fsub double %scalar.11.118, %scalar.12.119
  store double %scalar.13.120, ptr %value.120, align 8
  %scalar.14.121 = fsub double %load.11.118.0, %scalar.13.120
  store double %scalar.14.121, ptr %value.121, align 8
  %scalar.15.122 = fsub double %scalar.7.115, %scalar.12.119
  store double %scalar.15.122, ptr %value.122, align 8
  %scalar.16.123 = fadd double %scalar.14.121, %scalar.15.122
  store double %scalar.16.123, ptr %value.123, align 8
  %load.17.124.1 = load double, ptr %arg.14, align 8
  %scalar.17.124 = fadd double %scalar.16.123, %load.17.124.1
  store double %scalar.17.124, ptr %value.124, align 8
  %scalar.18.125 = fadd double %scalar.17.124, %scalar.9.117
  store double %scalar.18.125, ptr %value.125, align 8
  %scalar.19.126 = fadd double %scalar.11.118, %scalar.18.125
  store double %scalar.19.126, ptr %value.126, align 8
  %scalar.20.127 = fsub double %scalar.19.126, %scalar.11.118
  store double %scalar.20.127, ptr %value.127, align 8
  %scalar.21.128 = fsub double %scalar.18.125, %scalar.20.127
  store double %scalar.21.128, ptr %value.128, align 8
  %scalar.22.13 = fadd double %scalar.19.126, %scalar.21.128
  store double %scalar.22.13, ptr %out.2, align 8
  %scalar.23.129 = fmul double %load.0.108.1, %scalar.19.126
  store double %scalar.23.129, ptr %value.129, align 8
  %scalar.24.130 = fneg double %scalar.23.129
  store double %scalar.24.130, ptr %value.130, align 8
  %scalar.25.131 = call double @llvm.fma.f64(double %load.0.108.1, double %scalar.19.126, double %scalar.24.130)
  store double %scalar.25.131, ptr %value.131, align 8
  %scalar.26.132 = fmul double %load.0.108.1, %scalar.21.128
  store double %scalar.26.132, ptr %value.132, align 8
  %scalar.27.133 = fadd double %scalar.25.131, %scalar.26.132
  store double %scalar.27.133, ptr %value.133, align 8
  %scalar.28.134 = fmul double %load.3.111.1, %scalar.19.126
  store double %scalar.28.134, ptr %value.134, align 8
  %scalar.29.135 = fadd double %scalar.27.133, %scalar.28.134
  store double %scalar.29.135, ptr %value.135, align 8
  %scalar.30.136 = fadd double %scalar.23.129, %scalar.29.135
  store double %scalar.30.136, ptr %value.136, align 8
  %scalar.31.137 = fsub double %scalar.30.136, %scalar.23.129
  store double %scalar.31.137, ptr %value.137, align 8
  %scalar.32.138 = fsub double %scalar.29.135, %scalar.31.137
  store double %scalar.32.138, ptr %value.138, align 8
  %scalar.33.14 = fadd double %scalar.30.136, %scalar.32.138
  store double %scalar.33.14, ptr %out.3, align 8
  %load.34.139.0 = load double, ptr %arg.3, align 8
  %scalar.34.139 = fadd double %load.34.139.0, %scalar.30.136
  store double %scalar.34.139, ptr %value.139, align 8
  %scalar.35.140 = fsub double %scalar.34.139, %load.34.139.0
  store double %scalar.35.140, ptr %value.140, align 8
  %scalar.36.141 = fsub double %scalar.34.139, %scalar.35.140
  store double %scalar.36.141, ptr %value.141, align 8
  %scalar.37.142 = fsub double %load.34.139.0, %scalar.36.141
  store double %scalar.37.142, ptr %value.142, align 8
  %scalar.38.143 = fsub double %scalar.30.136, %scalar.35.140
  store double %scalar.38.143, ptr %value.143, align 8
  %scalar.39.144 = fadd double %scalar.37.142, %scalar.38.143
  store double %scalar.39.144, ptr %value.144, align 8
  %load.40.145.1 = load double, ptr %arg.15, align 8
  %scalar.40.145 = fadd double %scalar.39.144, %load.40.145.1
  store double %scalar.40.145, ptr %value.145, align 8
  %scalar.41.146 = fadd double %scalar.40.145, %scalar.32.138
  store double %scalar.41.146, ptr %value.146, align 8
  %scalar.42.147 = fadd double %scalar.34.139, %scalar.41.146
  store double %scalar.42.147, ptr %value.147, align 8
  %scalar.43.148 = fsub double %scalar.42.147, %scalar.34.139
  store double %scalar.43.148, ptr %value.148, align 8
  %scalar.44.149 = fsub double %scalar.41.146, %scalar.43.148
  store double %scalar.44.149, ptr %value.149, align 8
  %scalar.45.15 = fadd double %scalar.42.147, %scalar.44.149
  store double %scalar.45.15, ptr %out.4, align 8
  %scalar.46.150 = fmul double %load.0.108.1, %scalar.42.147
  store double %scalar.46.150, ptr %value.150, align 8
  %scalar.47.151 = fneg double %scalar.46.150
  store double %scalar.47.151, ptr %value.151, align 8
  %scalar.48.152 = call double @llvm.fma.f64(double %load.0.108.1, double %scalar.42.147, double %scalar.47.151)
  store double %scalar.48.152, ptr %value.152, align 8
  %scalar.49.153 = fmul double %load.0.108.1, %scalar.44.149
  store double %scalar.49.153, ptr %value.153, align 8
  %scalar.50.154 = fadd double %scalar.48.152, %scalar.49.153
  store double %scalar.50.154, ptr %value.154, align 8
  %scalar.51.155 = fmul double %load.3.111.1, %scalar.42.147
  store double %scalar.51.155, ptr %value.155, align 8
  %scalar.52.156 = fadd double %scalar.50.154, %scalar.51.155
  store double %scalar.52.156, ptr %value.156, align 8
  %scalar.53.157 = fadd double %scalar.46.150, %scalar.52.156
  store double %scalar.53.157, ptr %value.157, align 8
  %scalar.54.158 = fsub double %scalar.53.157, %scalar.46.150
  store double %scalar.54.158, ptr %value.158, align 8
  %scalar.55.159 = fsub double %scalar.52.156, %scalar.54.158
  store double %scalar.55.159, ptr %value.159, align 8
  %scalar.56.16 = fadd double %scalar.53.157, %scalar.55.159
  store double %scalar.56.16, ptr %out.5, align 8
  %load.57.160.0 = load double, ptr %arg.4, align 8
  %scalar.57.160 = fadd double %load.57.160.0, %scalar.53.157
  store double %scalar.57.160, ptr %value.160, align 8
  %scalar.58.161 = fsub double %scalar.57.160, %load.57.160.0
  store double %scalar.58.161, ptr %value.161, align 8
  %scalar.59.162 = fsub double %scalar.57.160, %scalar.58.161
  store double %scalar.59.162, ptr %value.162, align 8
  %scalar.60.163 = fsub double %load.57.160.0, %scalar.59.162
  store double %scalar.60.163, ptr %value.163, align 8
  %scalar.61.164 = fsub double %scalar.53.157, %scalar.58.161
  store double %scalar.61.164, ptr %value.164, align 8
  %scalar.62.165 = fadd double %scalar.60.163, %scalar.61.164
  store double %scalar.62.165, ptr %value.165, align 8
  %load.63.166.1 = load double, ptr %arg.16, align 8
  %scalar.63.166 = fadd double %scalar.62.165, %load.63.166.1
  store double %scalar.63.166, ptr %value.166, align 8
  %scalar.64.167 = fadd double %scalar.63.166, %scalar.55.159
  store double %scalar.64.167, ptr %value.167, align 8
  %scalar.65.168 = fadd double %scalar.57.160, %scalar.64.167
  store double %scalar.65.168, ptr %value.168, align 8
  %scalar.66.169 = fsub double %scalar.65.168, %scalar.57.160
  store double %scalar.66.169, ptr %value.169, align 8
  %scalar.67.170 = fsub double %scalar.64.167, %scalar.66.169
  store double %scalar.67.170, ptr %value.170, align 8
  %scalar.68.17 = fadd double %scalar.65.168, %scalar.67.170
  store double %scalar.68.17, ptr %out.6, align 8
  %scalar.69.171 = fmul double %load.0.108.1, %scalar.65.168
  store double %scalar.69.171, ptr %value.171, align 8
  %scalar.70.172 = fneg double %scalar.69.171
  store double %scalar.70.172, ptr %value.172, align 8
  %scalar.71.173 = call double @llvm.fma.f64(double %load.0.108.1, double %scalar.65.168, double %scalar.70.172)
  store double %scalar.71.173, ptr %value.173, align 8
  %scalar.72.174 = fmul double %load.0.108.1, %scalar.67.170
  store double %scalar.72.174, ptr %value.174, align 8
  %scalar.73.175 = fadd double %scalar.71.173, %scalar.72.174
  store double %scalar.73.175, ptr %value.175, align 8
  %scalar.74.176 = fmul double %load.3.111.1, %scalar.65.168
  store double %scalar.74.176, ptr %value.176, align 8
  %scalar.75.177 = fadd double %scalar.73.175, %scalar.74.176
  store double %scalar.75.177, ptr %value.177, align 8
  %scalar.76.178 = fadd double %scalar.69.171, %scalar.75.177
  store double %scalar.76.178, ptr %value.178, align 8
  %scalar.77.179 = fsub double %scalar.76.178, %scalar.69.171
  store double %scalar.77.179, ptr %value.179, align 8
  %scalar.78.180 = fsub double %scalar.75.177, %scalar.77.179
  store double %scalar.78.180, ptr %value.180, align 8
  %scalar.79.18 = fadd double %scalar.76.178, %scalar.78.180
  store double %scalar.79.18, ptr %out.7, align 8
  %load.80.181.0 = load double, ptr %arg.5, align 8
  %scalar.80.181 = fadd double %load.80.181.0, %scalar.76.178
  store double %scalar.80.181, ptr %value.181, align 8
  %scalar.81.182 = fsub double %scalar.80.181, %load.80.181.0
  store double %scalar.81.182, ptr %value.182, align 8
  %scalar.82.183 = fsub double %scalar.80.181, %scalar.81.182
  store double %scalar.82.183, ptr %value.183, align 8
  %scalar.83.184 = fsub double %load.80.181.0, %scalar.82.183
  store double %scalar.83.184, ptr %value.184, align 8
  %scalar.84.185 = fsub double %scalar.76.178, %scalar.81.182
  store double %scalar.84.185, ptr %value.185, align 8
  %scalar.85.186 = fadd double %scalar.83.184, %scalar.84.185
  store double %scalar.85.186, ptr %value.186, align 8
  %load.86.187.1 = load double, ptr %arg.17, align 8
  %scalar.86.187 = fadd double %scalar.85.186, %load.86.187.1
  store double %scalar.86.187, ptr %value.187, align 8
  %scalar.87.188 = fadd double %scalar.86.187, %scalar.78.180
  store double %scalar.87.188, ptr %value.188, align 8
  %scalar.88.189 = fadd double %scalar.80.181, %scalar.87.188
  store double %scalar.88.189, ptr %value.189, align 8
  %scalar.89.190 = fsub double %scalar.88.189, %scalar.80.181
  store double %scalar.89.190, ptr %value.190, align 8
  %scalar.90.191 = fsub double %scalar.87.188, %scalar.89.190
  store double %scalar.90.191, ptr %value.191, align 8
  %scalar.91.19 = fadd double %scalar.88.189, %scalar.90.191
  store double %scalar.91.19, ptr %out.8, align 8
  %scalar.92.192 = fmul double %load.0.108.1, %scalar.88.189
  store double %scalar.92.192, ptr %value.192, align 8
  %scalar.93.193 = fneg double %scalar.92.192
  store double %scalar.93.193, ptr %value.193, align 8
  %scalar.94.194 = call double @llvm.fma.f64(double %load.0.108.1, double %scalar.88.189, double %scalar.93.193)
  store double %scalar.94.194, ptr %value.194, align 8
  %scalar.95.195 = fmul double %load.0.108.1, %scalar.90.191
  store double %scalar.95.195, ptr %value.195, align 8
  %scalar.96.196 = fadd double %scalar.94.194, %scalar.95.195
  store double %scalar.96.196, ptr %value.196, align 8
  %scalar.97.197 = fmul double %load.3.111.1, %scalar.88.189
  store double %scalar.97.197, ptr %value.197, align 8
  %scalar.98.198 = fadd double %scalar.96.196, %scalar.97.197
  store double %scalar.98.198, ptr %value.198, align 8
  %scalar.99.199 = fadd double %scalar.92.192, %scalar.98.198
  store double %scalar.99.199, ptr %value.199, align 8
  %scalar.100.200 = fsub double %scalar.99.199, %scalar.92.192
  store double %scalar.100.200, ptr %value.200, align 8
  %scalar.101.201 = fsub double %scalar.98.198, %scalar.100.200
  store double %scalar.101.201, ptr %value.201, align 8
  %scalar.102.20 = fadd double %scalar.99.199, %scalar.101.201
  store double %scalar.102.20, ptr %out.9, align 8
  %load.103.202.0 = load double, ptr %arg.6, align 8
  %scalar.103.202 = fadd double %load.103.202.0, %scalar.99.199
  store double %scalar.103.202, ptr %value.202, align 8
  %scalar.104.203 = fsub double %scalar.103.202, %load.103.202.0
  store double %scalar.104.203, ptr %value.203, align 8
  %scalar.105.204 = fsub double %scalar.103.202, %scalar.104.203
  store double %scalar.105.204, ptr %value.204, align 8
  %scalar.106.205 = fsub double %load.103.202.0, %scalar.105.204
  store double %scalar.106.205, ptr %value.205, align 8
  %scalar.107.206 = fsub double %scalar.99.199, %scalar.104.203
  store double %scalar.107.206, ptr %value.206, align 8
  %scalar.108.207 = fadd double %scalar.106.205, %scalar.107.206
  store double %scalar.108.207, ptr %value.207, align 8
  %load.109.208.1 = load double, ptr %arg.18, align 8
  %scalar.109.208 = fadd double %scalar.108.207, %load.109.208.1
  store double %scalar.109.208, ptr %value.208, align 8
  %scalar.110.209 = fadd double %scalar.109.208, %scalar.101.201
  store double %scalar.110.209, ptr %value.209, align 8
  %scalar.111.210 = fadd double %scalar.103.202, %scalar.110.209
  store double %scalar.111.210, ptr %value.210, align 8
  %scalar.112.211 = fsub double %scalar.111.210, %scalar.103.202
  store double %scalar.112.211, ptr %value.211, align 8
  %scalar.113.212 = fsub double %scalar.110.209, %scalar.112.211
  store double %scalar.113.212, ptr %value.212, align 8
  %scalar.114.21 = fadd double %scalar.111.210, %scalar.113.212
  store double %scalar.114.21, ptr %out.10, align 8
  %scalar.115.213 = fmul double %load.0.108.1, %scalar.111.210
  store double %scalar.115.213, ptr %value.213, align 8
  %scalar.116.214 = fneg double %scalar.115.213
  store double %scalar.116.214, ptr %value.214, align 8
  %scalar.117.215 = call double @llvm.fma.f64(double %load.0.108.1, double %scalar.111.210, double %scalar.116.214)
  store double %scalar.117.215, ptr %value.215, align 8
  %scalar.118.216 = fmul double %load.0.108.1, %scalar.113.212
  store double %scalar.118.216, ptr %value.216, align 8
  %scalar.119.217 = fadd double %scalar.117.215, %scalar.118.216
  store double %scalar.119.217, ptr %value.217, align 8
  %scalar.120.218 = fmul double %load.3.111.1, %scalar.111.210
  store double %scalar.120.218, ptr %value.218, align 8
  %scalar.121.219 = fadd double %scalar.119.217, %scalar.120.218
  store double %scalar.121.219, ptr %value.219, align 8
  %scalar.122.220 = fadd double %scalar.115.213, %scalar.121.219
  store double %scalar.122.220, ptr %value.220, align 8
  %scalar.123.221 = fsub double %scalar.122.220, %scalar.115.213
  store double %scalar.123.221, ptr %value.221, align 8
  %scalar.124.222 = fsub double %scalar.121.219, %scalar.123.221
  store double %scalar.124.222, ptr %value.222, align 8
  %scalar.125.22 = fadd double %scalar.122.220, %scalar.124.222
  store double %scalar.125.22, ptr %out.11, align 8
  %load.126.223.0 = load double, ptr %arg.7, align 8
  %scalar.126.223 = fadd double %load.126.223.0, %scalar.122.220
  store double %scalar.126.223, ptr %value.223, align 8
  %scalar.127.224 = fsub double %scalar.126.223, %load.126.223.0
  store double %scalar.127.224, ptr %value.224, align 8
  %scalar.128.225 = fsub double %scalar.126.223, %scalar.127.224
  store double %scalar.128.225, ptr %value.225, align 8
  %scalar.129.226 = fsub double %load.126.223.0, %scalar.128.225
  store double %scalar.129.226, ptr %value.226, align 8
  %scalar.130.227 = fsub double %scalar.122.220, %scalar.127.224
  store double %scalar.130.227, ptr %value.227, align 8
  %scalar.131.228 = fadd double %scalar.129.226, %scalar.130.227
  store double %scalar.131.228, ptr %value.228, align 8
  %load.132.229.1 = load double, ptr %arg.19, align 8
  %scalar.132.229 = fadd double %scalar.131.228, %load.132.229.1
  store double %scalar.132.229, ptr %value.229, align 8
  %scalar.133.230 = fadd double %scalar.132.229, %scalar.124.222
  store double %scalar.133.230, ptr %value.230, align 8
  %scalar.134.231 = fadd double %scalar.126.223, %scalar.133.230
  store double %scalar.134.231, ptr %value.231, align 8
  %scalar.135.232 = fsub double %scalar.134.231, %scalar.126.223
  store double %scalar.135.232, ptr %value.232, align 8
  %scalar.136.233 = fsub double %scalar.133.230, %scalar.135.232
  store double %scalar.136.233, ptr %value.233, align 8
  %scalar.137.23 = fadd double %scalar.134.231, %scalar.136.233
  store double %scalar.137.23, ptr %out.12, align 8
  %scalar.138.234 = fmul double %load.0.108.1, %scalar.134.231
  store double %scalar.138.234, ptr %value.234, align 8
  %scalar.139.235 = fneg double %scalar.138.234
  store double %scalar.139.235, ptr %value.235, align 8
  %scalar.140.236 = call double @llvm.fma.f64(double %load.0.108.1, double %scalar.134.231, double %scalar.139.235)
  store double %scalar.140.236, ptr %value.236, align 8
  %scalar.141.237 = fmul double %load.0.108.1, %scalar.136.233
  store double %scalar.141.237, ptr %value.237, align 8
  %scalar.142.238 = fadd double %scalar.140.236, %scalar.141.237
  store double %scalar.142.238, ptr %value.238, align 8
  %scalar.143.239 = fmul double %load.3.111.1, %scalar.134.231
  store double %scalar.143.239, ptr %value.239, align 8
  %scalar.144.240 = fadd double %scalar.142.238, %scalar.143.239
  store double %scalar.144.240, ptr %value.240, align 8
  %scalar.145.241 = fadd double %scalar.138.234, %scalar.144.240
  store double %scalar.145.241, ptr %value.241, align 8
  %scalar.146.242 = fsub double %scalar.145.241, %scalar.138.234
  store double %scalar.146.242, ptr %value.242, align 8
  %scalar.147.243 = fsub double %scalar.144.240, %scalar.146.242
  store double %scalar.147.243, ptr %value.243, align 8
  %scalar.148.24 = fadd double %scalar.145.241, %scalar.147.243
  store double %scalar.148.24, ptr %out.13, align 8
  %load.149.244.0 = load double, ptr %arg.8, align 8
  %scalar.149.244 = fadd double %load.149.244.0, %scalar.145.241
  store double %scalar.149.244, ptr %value.244, align 8
  %scalar.150.245 = fsub double %scalar.149.244, %load.149.244.0
  store double %scalar.150.245, ptr %value.245, align 8
  %scalar.151.246 = fsub double %scalar.149.244, %scalar.150.245
  store double %scalar.151.246, ptr %value.246, align 8
  %scalar.152.247 = fsub double %load.149.244.0, %scalar.151.246
  store double %scalar.152.247, ptr %value.247, align 8
  %scalar.153.248 = fsub double %scalar.145.241, %scalar.150.245
  store double %scalar.153.248, ptr %value.248, align 8
  %scalar.154.249 = fadd double %scalar.152.247, %scalar.153.248
  store double %scalar.154.249, ptr %value.249, align 8
  %load.155.250.1 = load double, ptr %arg.20, align 8
  %scalar.155.250 = fadd double %scalar.154.249, %load.155.250.1
  store double %scalar.155.250, ptr %value.250, align 8
  %scalar.156.251 = fadd double %scalar.155.250, %scalar.147.243
  store double %scalar.156.251, ptr %value.251, align 8
  %scalar.157.252 = fadd double %scalar.149.244, %scalar.156.251
  store double %scalar.157.252, ptr %value.252, align 8
  %scalar.158.253 = fsub double %scalar.157.252, %scalar.149.244
  store double %scalar.158.253, ptr %value.253, align 8
  %scalar.159.254 = fsub double %scalar.156.251, %scalar.158.253
  store double %scalar.159.254, ptr %value.254, align 8
  %scalar.160.25 = fadd double %scalar.157.252, %scalar.159.254
  store double %scalar.160.25, ptr %out.14, align 8
  %scalar.161.255 = fmul double %load.0.108.1, %scalar.157.252
  store double %scalar.161.255, ptr %value.255, align 8
  %scalar.162.256 = fneg double %scalar.161.255
  store double %scalar.162.256, ptr %value.256, align 8
  %scalar.163.257 = call double @llvm.fma.f64(double %load.0.108.1, double %scalar.157.252, double %scalar.162.256)
  store double %scalar.163.257, ptr %value.257, align 8
  %scalar.164.258 = fmul double %load.0.108.1, %scalar.159.254
  store double %scalar.164.258, ptr %value.258, align 8
  %scalar.165.259 = fadd double %scalar.163.257, %scalar.164.258
  store double %scalar.165.259, ptr %value.259, align 8
  %scalar.166.260 = fmul double %load.3.111.1, %scalar.157.252
  store double %scalar.166.260, ptr %value.260, align 8
  %scalar.167.261 = fadd double %scalar.165.259, %scalar.166.260
  store double %scalar.167.261, ptr %value.261, align 8
  %scalar.168.262 = fadd double %scalar.161.255, %scalar.167.261
  store double %scalar.168.262, ptr %value.262, align 8
  %scalar.169.263 = fsub double %scalar.168.262, %scalar.161.255
  store double %scalar.169.263, ptr %value.263, align 8
  %scalar.170.264 = fsub double %scalar.167.261, %scalar.169.263
  store double %scalar.170.264, ptr %value.264, align 8
  %scalar.171.26 = fadd double %scalar.168.262, %scalar.170.264
  store double %scalar.171.26, ptr %out.15, align 8
  %load.172.265.0 = load double, ptr %arg.9, align 8
  %scalar.172.265 = fadd double %load.172.265.0, %scalar.168.262
  store double %scalar.172.265, ptr %value.265, align 8
  %scalar.173.266 = fsub double %scalar.172.265, %load.172.265.0
  store double %scalar.173.266, ptr %value.266, align 8
  %scalar.174.267 = fsub double %scalar.172.265, %scalar.173.266
  store double %scalar.174.267, ptr %value.267, align 8
  %scalar.175.268 = fsub double %load.172.265.0, %scalar.174.267
  store double %scalar.175.268, ptr %value.268, align 8
  %scalar.176.269 = fsub double %scalar.168.262, %scalar.173.266
  store double %scalar.176.269, ptr %value.269, align 8
  %scalar.177.270 = fadd double %scalar.175.268, %scalar.176.269
  store double %scalar.177.270, ptr %value.270, align 8
  %load.178.271.1 = load double, ptr %arg.21, align 8
  %scalar.178.271 = fadd double %scalar.177.270, %load.178.271.1
  store double %scalar.178.271, ptr %value.271, align 8
  %scalar.179.272 = fadd double %scalar.178.271, %scalar.170.264
  store double %scalar.179.272, ptr %value.272, align 8
  %scalar.180.273 = fadd double %scalar.172.265, %scalar.179.272
  store double %scalar.180.273, ptr %value.273, align 8
  %scalar.181.274 = fsub double %scalar.180.273, %scalar.172.265
  store double %scalar.181.274, ptr %value.274, align 8
  %scalar.182.275 = fsub double %scalar.179.272, %scalar.181.274
  store double %scalar.182.275, ptr %value.275, align 8
  %scalar.183.27 = fadd double %scalar.180.273, %scalar.182.275
  store double %scalar.183.27, ptr %out.16, align 8
  %scalar.184.276 = fmul double %load.0.108.1, %scalar.180.273
  store double %scalar.184.276, ptr %value.276, align 8
  %scalar.185.277 = fneg double %scalar.184.276
  store double %scalar.185.277, ptr %value.277, align 8
  %scalar.186.278 = call double @llvm.fma.f64(double %load.0.108.1, double %scalar.180.273, double %scalar.185.277)
  store double %scalar.186.278, ptr %value.278, align 8
  %scalar.187.279 = fmul double %load.0.108.1, %scalar.182.275
  store double %scalar.187.279, ptr %value.279, align 8
  %scalar.188.280 = fadd double %scalar.186.278, %scalar.187.279
  store double %scalar.188.280, ptr %value.280, align 8
  %scalar.189.281 = fmul double %load.3.111.1, %scalar.180.273
  store double %scalar.189.281, ptr %value.281, align 8
  %scalar.190.282 = fadd double %scalar.188.280, %scalar.189.281
  store double %scalar.190.282, ptr %value.282, align 8
  %scalar.191.283 = fadd double %scalar.184.276, %scalar.190.282
  store double %scalar.191.283, ptr %value.283, align 8
  %scalar.192.284 = fsub double %scalar.191.283, %scalar.184.276
  store double %scalar.192.284, ptr %value.284, align 8
  %scalar.193.285 = fsub double %scalar.190.282, %scalar.192.284
  store double %scalar.193.285, ptr %value.285, align 8
  %scalar.194.28 = fadd double %scalar.191.283, %scalar.193.285
  store double %scalar.194.28, ptr %out.17, align 8
  %load.195.286.0 = load double, ptr %arg.10, align 8
  %scalar.195.286 = fadd double %load.195.286.0, %scalar.191.283
  store double %scalar.195.286, ptr %value.286, align 8
  %scalar.196.287 = fsub double %scalar.195.286, %load.195.286.0
  store double %scalar.196.287, ptr %value.287, align 8
  %scalar.197.288 = fsub double %scalar.195.286, %scalar.196.287
  store double %scalar.197.288, ptr %value.288, align 8
  %scalar.198.289 = fsub double %load.195.286.0, %scalar.197.288
  store double %scalar.198.289, ptr %value.289, align 8
  %scalar.199.290 = fsub double %scalar.191.283, %scalar.196.287
  store double %scalar.199.290, ptr %value.290, align 8
  %scalar.200.291 = fadd double %scalar.198.289, %scalar.199.290
  store double %scalar.200.291, ptr %value.291, align 8
  %load.201.292.1 = load double, ptr %arg.22, align 8
  %scalar.201.292 = fadd double %scalar.200.291, %load.201.292.1
  store double %scalar.201.292, ptr %value.292, align 8
  %scalar.202.293 = fadd double %scalar.201.292, %scalar.193.285
  store double %scalar.202.293, ptr %value.293, align 8
  %scalar.203.294 = fadd double %scalar.195.286, %scalar.202.293
  store double %scalar.203.294, ptr %value.294, align 8
  %scalar.204.295 = fsub double %scalar.203.294, %scalar.195.286
  store double %scalar.204.295, ptr %value.295, align 8
  %scalar.205.296 = fsub double %scalar.202.293, %scalar.204.295
  store double %scalar.205.296, ptr %value.296, align 8
  %scalar.206.29 = fadd double %scalar.203.294, %scalar.205.296
  store double %scalar.206.29, ptr %out.18, align 8
  %scalar.207.297 = fmul double %load.0.108.1, %scalar.203.294
  store double %scalar.207.297, ptr %value.297, align 8
  %scalar.208.298 = fneg double %scalar.207.297
  store double %scalar.208.298, ptr %value.298, align 8
  %scalar.209.299 = call double @llvm.fma.f64(double %load.0.108.1, double %scalar.203.294, double %scalar.208.298)
  store double %scalar.209.299, ptr %value.299, align 8
  %scalar.210.300 = fmul double %load.0.108.1, %scalar.205.296
  store double %scalar.210.300, ptr %value.300, align 8
  %scalar.211.301 = fadd double %scalar.209.299, %scalar.210.300
  store double %scalar.211.301, ptr %value.301, align 8
  %scalar.212.302 = fmul double %load.3.111.1, %scalar.203.294
  store double %scalar.212.302, ptr %value.302, align 8
  %scalar.213.303 = fadd double %scalar.211.301, %scalar.212.302
  store double %scalar.213.303, ptr %value.303, align 8
  %scalar.214.304 = fadd double %scalar.207.297, %scalar.213.303
  store double %scalar.214.304, ptr %value.304, align 8
  %scalar.215.305 = fsub double %scalar.214.304, %scalar.207.297
  store double %scalar.215.305, ptr %value.305, align 8
  %scalar.216.306 = fsub double %scalar.213.303, %scalar.215.305
  store double %scalar.216.306, ptr %value.306, align 8
  %scalar.217.30 = fadd double %scalar.214.304, %scalar.216.306
  store double %scalar.217.30, ptr %out.19, align 8
  %load.218.307.0 = load double, ptr %arg.11, align 8
  %scalar.218.307 = fadd double %load.218.307.0, %scalar.214.304
  store double %scalar.218.307, ptr %value.307, align 8
  %scalar.219.308 = fsub double %scalar.218.307, %load.218.307.0
  store double %scalar.219.308, ptr %value.308, align 8
  %scalar.220.309 = fsub double %scalar.218.307, %scalar.219.308
  store double %scalar.220.309, ptr %value.309, align 8
  %scalar.221.310 = fsub double %load.218.307.0, %scalar.220.309
  store double %scalar.221.310, ptr %value.310, align 8
  %scalar.222.311 = fsub double %scalar.214.304, %scalar.219.308
  store double %scalar.222.311, ptr %value.311, align 8
  %scalar.223.312 = fadd double %scalar.221.310, %scalar.222.311
  store double %scalar.223.312, ptr %value.312, align 8
  %load.224.313.1 = load double, ptr %arg.23, align 8
  %scalar.224.313 = fadd double %scalar.223.312, %load.224.313.1
  store double %scalar.224.313, ptr %value.313, align 8
  %scalar.225.314 = fadd double %scalar.224.313, %scalar.216.306
  store double %scalar.225.314, ptr %value.314, align 8
  %scalar.226.315 = fadd double %scalar.218.307, %scalar.225.314
  store double %scalar.226.315, ptr %value.315, align 8
  %scalar.227.316 = fsub double %scalar.226.315, %scalar.218.307
  store double %scalar.227.316, ptr %value.316, align 8
  %scalar.228.317 = fsub double %scalar.225.314, %scalar.227.316
  store double %scalar.228.317, ptr %value.317, align 8
  %scalar.229.31 = fadd double %scalar.226.315, %scalar.228.317
  store double %scalar.229.31, ptr %out.0, align 8
  ret void
}

define void @__ssa_csc_core_pack__csc_core(ptr noalias %arg.0, ptr noalias %arg.1, ptr noalias %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr %arg.11, ptr noalias %arg.12, ptr noalias %arg.13, ptr noalias %arg.14, ptr noalias %arg.15, ptr noalias %arg.16, ptr noalias %arg.17, ptr noalias %arg.18, ptr noalias %arg.19, ptr noalias %arg.20, ptr noalias %arg.21, ptr noalias %arg.22, ptr %arg.23, ptr %out.0) {
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
  call void @__ssa_csc_core_pack__csc_core__planned_region_0(ptr %arg.2, ptr %arg.11, ptr %arg.10, ptr %arg.9, ptr %arg.8, ptr %arg.7, ptr %arg.6, ptr %arg.5, ptr %arg.4, ptr %arg.3, ptr %arg.1, ptr %arg.0, ptr %arg.14, ptr %arg.23, ptr %arg.22, ptr %arg.21, ptr %arg.20, ptr %arg.19, ptr %arg.18, ptr %arg.17, ptr %arg.16, ptr %arg.15, ptr %arg.13, ptr %arg.12, ptr %out.0, ptr %value.12, ptr %value.13, ptr %value.14, ptr %value.15, ptr %value.16, ptr %value.17, ptr %value.18, ptr %value.19, ptr %value.20, ptr %value.21, ptr %value.22, ptr %value.23, ptr %value.24, ptr %value.25, ptr %value.26, ptr %value.27, ptr %value.28, ptr %value.29, ptr %value.30)
  ret void
}

define void @csc_core_pack__csc_core_pack(ptr %buffers, ptr %extents) {
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
  call void @__ssa_csc_core_pack__csc_core_pack(ptr %public.0, ptr %public.1, ptr %public.2, ptr %public.3, ptr %public.4, ptr %public.5, ptr %public.6, ptr %public.7, ptr %public.8, ptr %public.9, ptr %public.10, ptr %public.11, ptr %public.12, ptr %public.13, ptr %public.14, ptr %public.15, ptr %public.16, ptr %public.17, ptr %public.18, ptr %public.19, ptr %public.20, ptr %public.21, ptr %public.22, ptr %public.23, ptr %public.24, ptr %public.2)
  ret void
}
