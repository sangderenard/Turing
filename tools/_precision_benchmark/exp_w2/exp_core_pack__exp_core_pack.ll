source_filename = "turing.ssa-llvm.exp_core_pack__exp_core_pack"

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-windows-gnu"

declare double @llvm.fma.f64(double, double, double)

define void @__ssa_exp_core_pack__exp_core_pack__planned_region_0(ptr noalias %arg.0, ptr %arg.1, ptr %out.0) {
entry:
  %load.0.31.0 = load i32, ptr %arg.1, align 4
  %address.0.31 = getelementptr double, ptr %arg.0, i32 %load.0.31.0
  %pinned.load.1.19 = load double, ptr %address.0.31, align 8
  store double %pinned.load.1.19, ptr %out.0, align 8
  ret void
}

define void @__ssa_exp_core_pack__exp_core_pack__planned_region_1(ptr %arg.0, ptr %arg.1, ptr %arg.2) {
entry:
  %load.0.32.0 = load i32, ptr %arg.1, align 4
  %address.0.32 = getelementptr double, ptr %arg.0, i32 %load.0.32.0
  %load.store.1.v = load double, ptr %arg.2, align 8
  store double %load.store.1.v, ptr %address.0.32, align 8
  ret void
}

define void @__ssa_exp_core_pack__exp_core_pack(ptr noalias %arg.0, ptr noalias %arg.1, ptr %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr noalias %arg.11, ptr noalias %arg.12, ptr noalias %arg.13, ptr noalias %arg.14, ptr noalias %arg.15, ptr noalias %arg.16, ptr noalias %arg.17, ptr noalias %arg.18, ptr noalias %arg.19, ptr noalias %arg.20, ptr noalias %arg.21, ptr noalias %arg.22, ptr noalias %arg.23, ptr noalias %arg.24, ptr noalias %arg.25, ptr noalias %arg.26, ptr noalias %arg.27, ptr noalias %arg.28, ptr noalias %arg.29, ptr noalias %arg.30, ptr %out.0) {
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
  call void @__ssa_exp_core_pack__exp_core_pack__planned_region_0(ptr %arg.1, ptr %phi.25, ptr %value.19)
  call void @__ssa_exp_core_pack__exp_core(ptr %arg.3, ptr %arg.4, ptr %arg.5, ptr %arg.6, ptr %arg.7, ptr %arg.8, ptr %arg.9, ptr %arg.10, ptr %arg.11, ptr %arg.12, ptr %arg.13, ptr %arg.14, ptr %arg.15, ptr %arg.16, ptr %value.19, ptr %arg.17, ptr %arg.18, ptr %arg.19, ptr %arg.20, ptr %arg.21, ptr %arg.22, ptr %arg.23, ptr %arg.24, ptr %arg.25, ptr %arg.26, ptr %arg.27, ptr %arg.28, ptr %arg.29, ptr %arg.30, ptr %value.19, ptr %value.20)
  call void @__ssa_exp_core_pack__exp_core_pack__planned_region_1(ptr %arg.2, ptr %phi.25, ptr %value.20)
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

define void @__ssa_exp_core_pack__exp_core__planned_region_0(ptr noalias %arg.0, ptr %arg.1, ptr noalias %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr noalias %arg.11, ptr noalias %arg.12, ptr noalias %arg.13, ptr noalias %arg.14, ptr noalias %arg.15, ptr %arg.16, ptr noalias %arg.17, ptr noalias %arg.18, ptr noalias %arg.19, ptr noalias %arg.20, ptr noalias %arg.21, ptr noalias %arg.22, ptr noalias %arg.23, ptr noalias %arg.24, ptr noalias %arg.25, ptr noalias %arg.26, ptr noalias %arg.27, ptr noalias %arg.28, ptr noalias %arg.29, ptr %out.0, ptr %out.1, ptr %out.2, ptr %out.3, ptr %out.4, ptr %out.5, ptr %out.6, ptr %out.7, ptr %out.8, ptr %out.9, ptr %out.10, ptr %out.11, ptr %out.12, ptr %out.13, ptr %out.14, ptr %out.15, ptr %out.16, ptr %out.17, ptr %out.18, ptr %out.19, ptr %out.20, ptr %out.21, ptr %out.22, ptr %out.23, ptr %out.24, ptr %out.25) {
entry:
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
  %value.318 = alloca double, i64 1, align 8
  %value.319 = alloca double, i64 1, align 8
  %value.320 = alloca double, i64 1, align 8
  %value.321 = alloca double, i64 1, align 8
  %value.322 = alloca double, i64 1, align 8
  %value.323 = alloca double, i64 1, align 8
  %value.324 = alloca double, i64 1, align 8
  %value.325 = alloca double, i64 1, align 8
  %value.326 = alloca double, i64 1, align 8
  %value.327 = alloca double, i64 1, align 8
  %value.328 = alloca double, i64 1, align 8
  %value.329 = alloca double, i64 1, align 8
  %value.330 = alloca double, i64 1, align 8
  %value.331 = alloca double, i64 1, align 8
  %value.332 = alloca double, i64 1, align 8
  %value.333 = alloca double, i64 1, align 8
  %value.334 = alloca double, i64 1, align 8
  %value.335 = alloca double, i64 1, align 8
  %value.336 = alloca double, i64 1, align 8
  %value.337 = alloca double, i64 1, align 8
  %value.338 = alloca double, i64 1, align 8
  %value.339 = alloca double, i64 1, align 8
  %value.340 = alloca double, i64 1, align 8
  %value.341 = alloca double, i64 1, align 8
  %value.342 = alloca double, i64 1, align 8
  %value.343 = alloca double, i64 1, align 8
  %value.344 = alloca double, i64 1, align 8
  %value.345 = alloca double, i64 1, align 8
  %value.346 = alloca double, i64 1, align 8
  %value.347 = alloca double, i64 1, align 8
  %value.348 = alloca double, i64 1, align 8
  %value.349 = alloca double, i64 1, align 8
  %value.350 = alloca double, i64 1, align 8
  %value.351 = alloca double, i64 1, align 8
  %value.352 = alloca double, i64 1, align 8
  %value.353 = alloca double, i64 1, align 8
  %value.354 = alloca double, i64 1, align 8
  %value.355 = alloca double, i64 1, align 8
  %value.356 = alloca double, i64 1, align 8
  %value.357 = alloca double, i64 1, align 8
  %value.358 = alloca double, i64 1, align 8
  %value.359 = alloca double, i64 1, align 8
  %value.360 = alloca double, i64 1, align 8
  %value.361 = alloca double, i64 1, align 8
  %value.362 = alloca double, i64 1, align 8
  %value.363 = alloca double, i64 1, align 8
  %value.364 = alloca double, i64 1, align 8
  %value.365 = alloca double, i64 1, align 8
  %value.366 = alloca double, i64 1, align 8
  %value.367 = alloca double, i64 1, align 8
  %value.368 = alloca double, i64 1, align 8
  %value.369 = alloca double, i64 1, align 8
  %value.370 = alloca double, i64 1, align 8
  %value.371 = alloca double, i64 1, align 8
  %value.372 = alloca double, i64 1, align 8
  %value.373 = alloca double, i64 1, align 8
  %value.374 = alloca double, i64 1, align 8
  %value.375 = alloca double, i64 1, align 8
  %value.376 = alloca double, i64 1, align 8
  %value.377 = alloca double, i64 1, align 8
  %value.378 = alloca double, i64 1, align 8
  %value.379 = alloca double, i64 1, align 8
  %value.380 = alloca double, i64 1, align 8
  %value.381 = alloca double, i64 1, align 8
  %value.382 = alloca double, i64 1, align 8
  %value.383 = alloca double, i64 1, align 8
  %value.384 = alloca double, i64 1, align 8
  %value.385 = alloca double, i64 1, align 8
  %value.386 = alloca double, i64 1, align 8
  %value.387 = alloca double, i64 1, align 8
  %value.388 = alloca double, i64 1, align 8
  %value.389 = alloca double, i64 1, align 8
  %value.390 = alloca double, i64 1, align 8
  %value.391 = alloca double, i64 1, align 8
  %value.392 = alloca double, i64 1, align 8
  %value.393 = alloca double, i64 1, align 8
  %value.394 = alloca double, i64 1, align 8
  %value.395 = alloca double, i64 1, align 8
  %value.396 = alloca double, i64 1, align 8
  %value.397 = alloca double, i64 1, align 8
  %value.398 = alloca double, i64 1, align 8
  %value.399 = alloca double, i64 1, align 8
  %value.400 = alloca double, i64 1, align 8
  %value.401 = alloca double, i64 1, align 8
  %value.402 = alloca double, i64 1, align 8
  %value.403 = alloca double, i64 1, align 8
  %value.404 = alloca double, i64 1, align 8
  %value.405 = alloca double, i64 1, align 8
  %value.406 = alloca double, i64 1, align 8
  %value.407 = alloca double, i64 1, align 8
  %value.408 = alloca double, i64 1, align 8
  %value.409 = alloca double, i64 1, align 8
  %value.410 = alloca double, i64 1, align 8
  %load.0.138.0 = load double, ptr %arg.0, align 8
  %load.0.138.1 = load double, ptr %arg.1, align 8
  %scalar.0.138 = fmul double %load.0.138.0, %load.0.138.1
  store double %scalar.0.138, ptr %value.138, align 8
  %scalar.1.139 = fneg double %scalar.0.138
  store double %scalar.1.139, ptr %value.139, align 8
  %scalar.2.140 = call double @llvm.fma.f64(double %load.0.138.0, double %load.0.138.1, double %scalar.1.139)
  store double %scalar.2.140, ptr %value.140, align 8
  %load.3.141.1 = load double, ptr %arg.16, align 8
  %scalar.3.141 = fmul double %load.0.138.0, %load.3.141.1
  store double %scalar.3.141, ptr %value.141, align 8
  %scalar.4.142 = fadd double %scalar.2.140, %scalar.3.141
  store double %scalar.4.142, ptr %value.142, align 8
  %load.5.143.0 = load double, ptr %arg.15, align 8
  %scalar.5.143 = fmul double %load.5.143.0, %load.0.138.1
  store double %scalar.5.143, ptr %value.143, align 8
  %scalar.6.144 = fadd double %scalar.4.142, %scalar.5.143
  store double %scalar.6.144, ptr %value.144, align 8
  %scalar.7.145 = fadd double %scalar.0.138, %scalar.6.144
  store double %scalar.7.145, ptr %value.145, align 8
  %scalar.8.146 = fsub double %scalar.7.145, %scalar.0.138
  store double %scalar.8.146, ptr %value.146, align 8
  %scalar.9.147 = fsub double %scalar.6.144, %scalar.8.146
  store double %scalar.9.147, ptr %value.147, align 8
  %scalar.10.15 = fadd double %scalar.7.145, %scalar.9.147
  store double %scalar.10.15, ptr %out.1, align 8
  %load.11.148.0 = load double, ptr %arg.2, align 8
  %scalar.11.148 = fadd double %load.11.148.0, %scalar.7.145
  store double %scalar.11.148, ptr %value.148, align 8
  %scalar.12.149 = fsub double %scalar.11.148, %load.11.148.0
  store double %scalar.12.149, ptr %value.149, align 8
  %scalar.13.150 = fsub double %scalar.11.148, %scalar.12.149
  store double %scalar.13.150, ptr %value.150, align 8
  %scalar.14.151 = fsub double %load.11.148.0, %scalar.13.150
  store double %scalar.14.151, ptr %value.151, align 8
  %scalar.15.152 = fsub double %scalar.7.145, %scalar.12.149
  store double %scalar.15.152, ptr %value.152, align 8
  %scalar.16.153 = fadd double %scalar.14.151, %scalar.15.152
  store double %scalar.16.153, ptr %value.153, align 8
  %load.17.154.1 = load double, ptr %arg.17, align 8
  %scalar.17.154 = fadd double %scalar.16.153, %load.17.154.1
  store double %scalar.17.154, ptr %value.154, align 8
  %scalar.18.155 = fadd double %scalar.17.154, %scalar.9.147
  store double %scalar.18.155, ptr %value.155, align 8
  %scalar.19.156 = fadd double %scalar.11.148, %scalar.18.155
  store double %scalar.19.156, ptr %value.156, align 8
  %scalar.20.157 = fsub double %scalar.19.156, %scalar.11.148
  store double %scalar.20.157, ptr %value.157, align 8
  %scalar.21.158 = fsub double %scalar.18.155, %scalar.20.157
  store double %scalar.21.158, ptr %value.158, align 8
  %scalar.22.16 = fadd double %scalar.19.156, %scalar.21.158
  store double %scalar.22.16, ptr %out.2, align 8
  %scalar.23.159 = fmul double %load.0.138.1, %scalar.19.156
  store double %scalar.23.159, ptr %value.159, align 8
  %scalar.24.160 = fneg double %scalar.23.159
  store double %scalar.24.160, ptr %value.160, align 8
  %scalar.25.161 = call double @llvm.fma.f64(double %load.0.138.1, double %scalar.19.156, double %scalar.24.160)
  store double %scalar.25.161, ptr %value.161, align 8
  %scalar.26.162 = fmul double %load.0.138.1, %scalar.21.158
  store double %scalar.26.162, ptr %value.162, align 8
  %scalar.27.163 = fadd double %scalar.25.161, %scalar.26.162
  store double %scalar.27.163, ptr %value.163, align 8
  %scalar.28.164 = fmul double %load.3.141.1, %scalar.19.156
  store double %scalar.28.164, ptr %value.164, align 8
  %scalar.29.165 = fadd double %scalar.27.163, %scalar.28.164
  store double %scalar.29.165, ptr %value.165, align 8
  %scalar.30.166 = fadd double %scalar.23.159, %scalar.29.165
  store double %scalar.30.166, ptr %value.166, align 8
  %scalar.31.167 = fsub double %scalar.30.166, %scalar.23.159
  store double %scalar.31.167, ptr %value.167, align 8
  %scalar.32.168 = fsub double %scalar.29.165, %scalar.31.167
  store double %scalar.32.168, ptr %value.168, align 8
  %scalar.33.17 = fadd double %scalar.30.166, %scalar.32.168
  store double %scalar.33.17, ptr %out.3, align 8
  %load.34.169.0 = load double, ptr %arg.3, align 8
  %scalar.34.169 = fadd double %load.34.169.0, %scalar.30.166
  store double %scalar.34.169, ptr %value.169, align 8
  %scalar.35.170 = fsub double %scalar.34.169, %load.34.169.0
  store double %scalar.35.170, ptr %value.170, align 8
  %scalar.36.171 = fsub double %scalar.34.169, %scalar.35.170
  store double %scalar.36.171, ptr %value.171, align 8
  %scalar.37.172 = fsub double %load.34.169.0, %scalar.36.171
  store double %scalar.37.172, ptr %value.172, align 8
  %scalar.38.173 = fsub double %scalar.30.166, %scalar.35.170
  store double %scalar.38.173, ptr %value.173, align 8
  %scalar.39.174 = fadd double %scalar.37.172, %scalar.38.173
  store double %scalar.39.174, ptr %value.174, align 8
  %load.40.175.1 = load double, ptr %arg.18, align 8
  %scalar.40.175 = fadd double %scalar.39.174, %load.40.175.1
  store double %scalar.40.175, ptr %value.175, align 8
  %scalar.41.176 = fadd double %scalar.40.175, %scalar.32.168
  store double %scalar.41.176, ptr %value.176, align 8
  %scalar.42.177 = fadd double %scalar.34.169, %scalar.41.176
  store double %scalar.42.177, ptr %value.177, align 8
  %scalar.43.178 = fsub double %scalar.42.177, %scalar.34.169
  store double %scalar.43.178, ptr %value.178, align 8
  %scalar.44.179 = fsub double %scalar.41.176, %scalar.43.178
  store double %scalar.44.179, ptr %value.179, align 8
  %scalar.45.18 = fadd double %scalar.42.177, %scalar.44.179
  store double %scalar.45.18, ptr %out.4, align 8
  %scalar.46.180 = fmul double %load.0.138.1, %scalar.42.177
  store double %scalar.46.180, ptr %value.180, align 8
  %scalar.47.181 = fneg double %scalar.46.180
  store double %scalar.47.181, ptr %value.181, align 8
  %scalar.48.182 = call double @llvm.fma.f64(double %load.0.138.1, double %scalar.42.177, double %scalar.47.181)
  store double %scalar.48.182, ptr %value.182, align 8
  %scalar.49.183 = fmul double %load.0.138.1, %scalar.44.179
  store double %scalar.49.183, ptr %value.183, align 8
  %scalar.50.184 = fadd double %scalar.48.182, %scalar.49.183
  store double %scalar.50.184, ptr %value.184, align 8
  %scalar.51.185 = fmul double %load.3.141.1, %scalar.42.177
  store double %scalar.51.185, ptr %value.185, align 8
  %scalar.52.186 = fadd double %scalar.50.184, %scalar.51.185
  store double %scalar.52.186, ptr %value.186, align 8
  %scalar.53.187 = fadd double %scalar.46.180, %scalar.52.186
  store double %scalar.53.187, ptr %value.187, align 8
  %scalar.54.188 = fsub double %scalar.53.187, %scalar.46.180
  store double %scalar.54.188, ptr %value.188, align 8
  %scalar.55.189 = fsub double %scalar.52.186, %scalar.54.188
  store double %scalar.55.189, ptr %value.189, align 8
  %scalar.56.19 = fadd double %scalar.53.187, %scalar.55.189
  store double %scalar.56.19, ptr %out.5, align 8
  %load.57.190.0 = load double, ptr %arg.4, align 8
  %scalar.57.190 = fadd double %load.57.190.0, %scalar.53.187
  store double %scalar.57.190, ptr %value.190, align 8
  %scalar.58.191 = fsub double %scalar.57.190, %load.57.190.0
  store double %scalar.58.191, ptr %value.191, align 8
  %scalar.59.192 = fsub double %scalar.57.190, %scalar.58.191
  store double %scalar.59.192, ptr %value.192, align 8
  %scalar.60.193 = fsub double %load.57.190.0, %scalar.59.192
  store double %scalar.60.193, ptr %value.193, align 8
  %scalar.61.194 = fsub double %scalar.53.187, %scalar.58.191
  store double %scalar.61.194, ptr %value.194, align 8
  %scalar.62.195 = fadd double %scalar.60.193, %scalar.61.194
  store double %scalar.62.195, ptr %value.195, align 8
  %load.63.196.1 = load double, ptr %arg.19, align 8
  %scalar.63.196 = fadd double %scalar.62.195, %load.63.196.1
  store double %scalar.63.196, ptr %value.196, align 8
  %scalar.64.197 = fadd double %scalar.63.196, %scalar.55.189
  store double %scalar.64.197, ptr %value.197, align 8
  %scalar.65.198 = fadd double %scalar.57.190, %scalar.64.197
  store double %scalar.65.198, ptr %value.198, align 8
  %scalar.66.199 = fsub double %scalar.65.198, %scalar.57.190
  store double %scalar.66.199, ptr %value.199, align 8
  %scalar.67.200 = fsub double %scalar.64.197, %scalar.66.199
  store double %scalar.67.200, ptr %value.200, align 8
  %scalar.68.20 = fadd double %scalar.65.198, %scalar.67.200
  store double %scalar.68.20, ptr %out.6, align 8
  %scalar.69.201 = fmul double %load.0.138.1, %scalar.65.198
  store double %scalar.69.201, ptr %value.201, align 8
  %scalar.70.202 = fneg double %scalar.69.201
  store double %scalar.70.202, ptr %value.202, align 8
  %scalar.71.203 = call double @llvm.fma.f64(double %load.0.138.1, double %scalar.65.198, double %scalar.70.202)
  store double %scalar.71.203, ptr %value.203, align 8
  %scalar.72.204 = fmul double %load.0.138.1, %scalar.67.200
  store double %scalar.72.204, ptr %value.204, align 8
  %scalar.73.205 = fadd double %scalar.71.203, %scalar.72.204
  store double %scalar.73.205, ptr %value.205, align 8
  %scalar.74.206 = fmul double %load.3.141.1, %scalar.65.198
  store double %scalar.74.206, ptr %value.206, align 8
  %scalar.75.207 = fadd double %scalar.73.205, %scalar.74.206
  store double %scalar.75.207, ptr %value.207, align 8
  %scalar.76.208 = fadd double %scalar.69.201, %scalar.75.207
  store double %scalar.76.208, ptr %value.208, align 8
  %scalar.77.209 = fsub double %scalar.76.208, %scalar.69.201
  store double %scalar.77.209, ptr %value.209, align 8
  %scalar.78.210 = fsub double %scalar.75.207, %scalar.77.209
  store double %scalar.78.210, ptr %value.210, align 8
  %scalar.79.21 = fadd double %scalar.76.208, %scalar.78.210
  store double %scalar.79.21, ptr %out.7, align 8
  %load.80.211.0 = load double, ptr %arg.5, align 8
  %scalar.80.211 = fadd double %load.80.211.0, %scalar.76.208
  store double %scalar.80.211, ptr %value.211, align 8
  %scalar.81.212 = fsub double %scalar.80.211, %load.80.211.0
  store double %scalar.81.212, ptr %value.212, align 8
  %scalar.82.213 = fsub double %scalar.80.211, %scalar.81.212
  store double %scalar.82.213, ptr %value.213, align 8
  %scalar.83.214 = fsub double %load.80.211.0, %scalar.82.213
  store double %scalar.83.214, ptr %value.214, align 8
  %scalar.84.215 = fsub double %scalar.76.208, %scalar.81.212
  store double %scalar.84.215, ptr %value.215, align 8
  %scalar.85.216 = fadd double %scalar.83.214, %scalar.84.215
  store double %scalar.85.216, ptr %value.216, align 8
  %load.86.217.1 = load double, ptr %arg.20, align 8
  %scalar.86.217 = fadd double %scalar.85.216, %load.86.217.1
  store double %scalar.86.217, ptr %value.217, align 8
  %scalar.87.218 = fadd double %scalar.86.217, %scalar.78.210
  store double %scalar.87.218, ptr %value.218, align 8
  %scalar.88.219 = fadd double %scalar.80.211, %scalar.87.218
  store double %scalar.88.219, ptr %value.219, align 8
  %scalar.89.220 = fsub double %scalar.88.219, %scalar.80.211
  store double %scalar.89.220, ptr %value.220, align 8
  %scalar.90.221 = fsub double %scalar.87.218, %scalar.89.220
  store double %scalar.90.221, ptr %value.221, align 8
  %scalar.91.22 = fadd double %scalar.88.219, %scalar.90.221
  store double %scalar.91.22, ptr %out.8, align 8
  %scalar.92.222 = fmul double %load.0.138.1, %scalar.88.219
  store double %scalar.92.222, ptr %value.222, align 8
  %scalar.93.223 = fneg double %scalar.92.222
  store double %scalar.93.223, ptr %value.223, align 8
  %scalar.94.224 = call double @llvm.fma.f64(double %load.0.138.1, double %scalar.88.219, double %scalar.93.223)
  store double %scalar.94.224, ptr %value.224, align 8
  %scalar.95.225 = fmul double %load.0.138.1, %scalar.90.221
  store double %scalar.95.225, ptr %value.225, align 8
  %scalar.96.226 = fadd double %scalar.94.224, %scalar.95.225
  store double %scalar.96.226, ptr %value.226, align 8
  %scalar.97.227 = fmul double %load.3.141.1, %scalar.88.219
  store double %scalar.97.227, ptr %value.227, align 8
  %scalar.98.228 = fadd double %scalar.96.226, %scalar.97.227
  store double %scalar.98.228, ptr %value.228, align 8
  %scalar.99.229 = fadd double %scalar.92.222, %scalar.98.228
  store double %scalar.99.229, ptr %value.229, align 8
  %scalar.100.230 = fsub double %scalar.99.229, %scalar.92.222
  store double %scalar.100.230, ptr %value.230, align 8
  %scalar.101.231 = fsub double %scalar.98.228, %scalar.100.230
  store double %scalar.101.231, ptr %value.231, align 8
  %scalar.102.23 = fadd double %scalar.99.229, %scalar.101.231
  store double %scalar.102.23, ptr %out.9, align 8
  %load.103.232.0 = load double, ptr %arg.6, align 8
  %scalar.103.232 = fadd double %load.103.232.0, %scalar.99.229
  store double %scalar.103.232, ptr %value.232, align 8
  %scalar.104.233 = fsub double %scalar.103.232, %load.103.232.0
  store double %scalar.104.233, ptr %value.233, align 8
  %scalar.105.234 = fsub double %scalar.103.232, %scalar.104.233
  store double %scalar.105.234, ptr %value.234, align 8
  %scalar.106.235 = fsub double %load.103.232.0, %scalar.105.234
  store double %scalar.106.235, ptr %value.235, align 8
  %scalar.107.236 = fsub double %scalar.99.229, %scalar.104.233
  store double %scalar.107.236, ptr %value.236, align 8
  %scalar.108.237 = fadd double %scalar.106.235, %scalar.107.236
  store double %scalar.108.237, ptr %value.237, align 8
  %load.109.238.1 = load double, ptr %arg.21, align 8
  %scalar.109.238 = fadd double %scalar.108.237, %load.109.238.1
  store double %scalar.109.238, ptr %value.238, align 8
  %scalar.110.239 = fadd double %scalar.109.238, %scalar.101.231
  store double %scalar.110.239, ptr %value.239, align 8
  %scalar.111.240 = fadd double %scalar.103.232, %scalar.110.239
  store double %scalar.111.240, ptr %value.240, align 8
  %scalar.112.241 = fsub double %scalar.111.240, %scalar.103.232
  store double %scalar.112.241, ptr %value.241, align 8
  %scalar.113.242 = fsub double %scalar.110.239, %scalar.112.241
  store double %scalar.113.242, ptr %value.242, align 8
  %scalar.114.24 = fadd double %scalar.111.240, %scalar.113.242
  store double %scalar.114.24, ptr %out.10, align 8
  %scalar.115.243 = fmul double %load.0.138.1, %scalar.111.240
  store double %scalar.115.243, ptr %value.243, align 8
  %scalar.116.244 = fneg double %scalar.115.243
  store double %scalar.116.244, ptr %value.244, align 8
  %scalar.117.245 = call double @llvm.fma.f64(double %load.0.138.1, double %scalar.111.240, double %scalar.116.244)
  store double %scalar.117.245, ptr %value.245, align 8
  %scalar.118.246 = fmul double %load.0.138.1, %scalar.113.242
  store double %scalar.118.246, ptr %value.246, align 8
  %scalar.119.247 = fadd double %scalar.117.245, %scalar.118.246
  store double %scalar.119.247, ptr %value.247, align 8
  %scalar.120.248 = fmul double %load.3.141.1, %scalar.111.240
  store double %scalar.120.248, ptr %value.248, align 8
  %scalar.121.249 = fadd double %scalar.119.247, %scalar.120.248
  store double %scalar.121.249, ptr %value.249, align 8
  %scalar.122.250 = fadd double %scalar.115.243, %scalar.121.249
  store double %scalar.122.250, ptr %value.250, align 8
  %scalar.123.251 = fsub double %scalar.122.250, %scalar.115.243
  store double %scalar.123.251, ptr %value.251, align 8
  %scalar.124.252 = fsub double %scalar.121.249, %scalar.123.251
  store double %scalar.124.252, ptr %value.252, align 8
  %scalar.125.25 = fadd double %scalar.122.250, %scalar.124.252
  store double %scalar.125.25, ptr %out.11, align 8
  %load.126.253.0 = load double, ptr %arg.7, align 8
  %scalar.126.253 = fadd double %load.126.253.0, %scalar.122.250
  store double %scalar.126.253, ptr %value.253, align 8
  %scalar.127.254 = fsub double %scalar.126.253, %load.126.253.0
  store double %scalar.127.254, ptr %value.254, align 8
  %scalar.128.255 = fsub double %scalar.126.253, %scalar.127.254
  store double %scalar.128.255, ptr %value.255, align 8
  %scalar.129.256 = fsub double %load.126.253.0, %scalar.128.255
  store double %scalar.129.256, ptr %value.256, align 8
  %scalar.130.257 = fsub double %scalar.122.250, %scalar.127.254
  store double %scalar.130.257, ptr %value.257, align 8
  %scalar.131.258 = fadd double %scalar.129.256, %scalar.130.257
  store double %scalar.131.258, ptr %value.258, align 8
  %load.132.259.1 = load double, ptr %arg.22, align 8
  %scalar.132.259 = fadd double %scalar.131.258, %load.132.259.1
  store double %scalar.132.259, ptr %value.259, align 8
  %scalar.133.260 = fadd double %scalar.132.259, %scalar.124.252
  store double %scalar.133.260, ptr %value.260, align 8
  %scalar.134.261 = fadd double %scalar.126.253, %scalar.133.260
  store double %scalar.134.261, ptr %value.261, align 8
  %scalar.135.262 = fsub double %scalar.134.261, %scalar.126.253
  store double %scalar.135.262, ptr %value.262, align 8
  %scalar.136.263 = fsub double %scalar.133.260, %scalar.135.262
  store double %scalar.136.263, ptr %value.263, align 8
  %scalar.137.26 = fadd double %scalar.134.261, %scalar.136.263
  store double %scalar.137.26, ptr %out.12, align 8
  %scalar.138.264 = fmul double %load.0.138.1, %scalar.134.261
  store double %scalar.138.264, ptr %value.264, align 8
  %scalar.139.265 = fneg double %scalar.138.264
  store double %scalar.139.265, ptr %value.265, align 8
  %scalar.140.266 = call double @llvm.fma.f64(double %load.0.138.1, double %scalar.134.261, double %scalar.139.265)
  store double %scalar.140.266, ptr %value.266, align 8
  %scalar.141.267 = fmul double %load.0.138.1, %scalar.136.263
  store double %scalar.141.267, ptr %value.267, align 8
  %scalar.142.268 = fadd double %scalar.140.266, %scalar.141.267
  store double %scalar.142.268, ptr %value.268, align 8
  %scalar.143.269 = fmul double %load.3.141.1, %scalar.134.261
  store double %scalar.143.269, ptr %value.269, align 8
  %scalar.144.270 = fadd double %scalar.142.268, %scalar.143.269
  store double %scalar.144.270, ptr %value.270, align 8
  %scalar.145.271 = fadd double %scalar.138.264, %scalar.144.270
  store double %scalar.145.271, ptr %value.271, align 8
  %scalar.146.272 = fsub double %scalar.145.271, %scalar.138.264
  store double %scalar.146.272, ptr %value.272, align 8
  %scalar.147.273 = fsub double %scalar.144.270, %scalar.146.272
  store double %scalar.147.273, ptr %value.273, align 8
  %scalar.148.27 = fadd double %scalar.145.271, %scalar.147.273
  store double %scalar.148.27, ptr %out.13, align 8
  %load.149.274.0 = load double, ptr %arg.8, align 8
  %scalar.149.274 = fadd double %load.149.274.0, %scalar.145.271
  store double %scalar.149.274, ptr %value.274, align 8
  %scalar.150.275 = fsub double %scalar.149.274, %load.149.274.0
  store double %scalar.150.275, ptr %value.275, align 8
  %scalar.151.276 = fsub double %scalar.149.274, %scalar.150.275
  store double %scalar.151.276, ptr %value.276, align 8
  %scalar.152.277 = fsub double %load.149.274.0, %scalar.151.276
  store double %scalar.152.277, ptr %value.277, align 8
  %scalar.153.278 = fsub double %scalar.145.271, %scalar.150.275
  store double %scalar.153.278, ptr %value.278, align 8
  %scalar.154.279 = fadd double %scalar.152.277, %scalar.153.278
  store double %scalar.154.279, ptr %value.279, align 8
  %load.155.280.1 = load double, ptr %arg.23, align 8
  %scalar.155.280 = fadd double %scalar.154.279, %load.155.280.1
  store double %scalar.155.280, ptr %value.280, align 8
  %scalar.156.281 = fadd double %scalar.155.280, %scalar.147.273
  store double %scalar.156.281, ptr %value.281, align 8
  %scalar.157.282 = fadd double %scalar.149.274, %scalar.156.281
  store double %scalar.157.282, ptr %value.282, align 8
  %scalar.158.283 = fsub double %scalar.157.282, %scalar.149.274
  store double %scalar.158.283, ptr %value.283, align 8
  %scalar.159.284 = fsub double %scalar.156.281, %scalar.158.283
  store double %scalar.159.284, ptr %value.284, align 8
  %scalar.160.28 = fadd double %scalar.157.282, %scalar.159.284
  store double %scalar.160.28, ptr %out.14, align 8
  %scalar.161.285 = fmul double %load.0.138.1, %scalar.157.282
  store double %scalar.161.285, ptr %value.285, align 8
  %scalar.162.286 = fneg double %scalar.161.285
  store double %scalar.162.286, ptr %value.286, align 8
  %scalar.163.287 = call double @llvm.fma.f64(double %load.0.138.1, double %scalar.157.282, double %scalar.162.286)
  store double %scalar.163.287, ptr %value.287, align 8
  %scalar.164.288 = fmul double %load.0.138.1, %scalar.159.284
  store double %scalar.164.288, ptr %value.288, align 8
  %scalar.165.289 = fadd double %scalar.163.287, %scalar.164.288
  store double %scalar.165.289, ptr %value.289, align 8
  %scalar.166.290 = fmul double %load.3.141.1, %scalar.157.282
  store double %scalar.166.290, ptr %value.290, align 8
  %scalar.167.291 = fadd double %scalar.165.289, %scalar.166.290
  store double %scalar.167.291, ptr %value.291, align 8
  %scalar.168.292 = fadd double %scalar.161.285, %scalar.167.291
  store double %scalar.168.292, ptr %value.292, align 8
  %scalar.169.293 = fsub double %scalar.168.292, %scalar.161.285
  store double %scalar.169.293, ptr %value.293, align 8
  %scalar.170.294 = fsub double %scalar.167.291, %scalar.169.293
  store double %scalar.170.294, ptr %value.294, align 8
  %scalar.171.29 = fadd double %scalar.168.292, %scalar.170.294
  store double %scalar.171.29, ptr %out.15, align 8
  %load.172.295.0 = load double, ptr %arg.9, align 8
  %scalar.172.295 = fadd double %load.172.295.0, %scalar.168.292
  store double %scalar.172.295, ptr %value.295, align 8
  %scalar.173.296 = fsub double %scalar.172.295, %load.172.295.0
  store double %scalar.173.296, ptr %value.296, align 8
  %scalar.174.297 = fsub double %scalar.172.295, %scalar.173.296
  store double %scalar.174.297, ptr %value.297, align 8
  %scalar.175.298 = fsub double %load.172.295.0, %scalar.174.297
  store double %scalar.175.298, ptr %value.298, align 8
  %scalar.176.299 = fsub double %scalar.168.292, %scalar.173.296
  store double %scalar.176.299, ptr %value.299, align 8
  %scalar.177.300 = fadd double %scalar.175.298, %scalar.176.299
  store double %scalar.177.300, ptr %value.300, align 8
  %load.178.301.1 = load double, ptr %arg.24, align 8
  %scalar.178.301 = fadd double %scalar.177.300, %load.178.301.1
  store double %scalar.178.301, ptr %value.301, align 8
  %scalar.179.302 = fadd double %scalar.178.301, %scalar.170.294
  store double %scalar.179.302, ptr %value.302, align 8
  %scalar.180.303 = fadd double %scalar.172.295, %scalar.179.302
  store double %scalar.180.303, ptr %value.303, align 8
  %scalar.181.304 = fsub double %scalar.180.303, %scalar.172.295
  store double %scalar.181.304, ptr %value.304, align 8
  %scalar.182.305 = fsub double %scalar.179.302, %scalar.181.304
  store double %scalar.182.305, ptr %value.305, align 8
  %scalar.183.30 = fadd double %scalar.180.303, %scalar.182.305
  store double %scalar.183.30, ptr %out.16, align 8
  %scalar.184.306 = fmul double %load.0.138.1, %scalar.180.303
  store double %scalar.184.306, ptr %value.306, align 8
  %scalar.185.307 = fneg double %scalar.184.306
  store double %scalar.185.307, ptr %value.307, align 8
  %scalar.186.308 = call double @llvm.fma.f64(double %load.0.138.1, double %scalar.180.303, double %scalar.185.307)
  store double %scalar.186.308, ptr %value.308, align 8
  %scalar.187.309 = fmul double %load.0.138.1, %scalar.182.305
  store double %scalar.187.309, ptr %value.309, align 8
  %scalar.188.310 = fadd double %scalar.186.308, %scalar.187.309
  store double %scalar.188.310, ptr %value.310, align 8
  %scalar.189.311 = fmul double %load.3.141.1, %scalar.180.303
  store double %scalar.189.311, ptr %value.311, align 8
  %scalar.190.312 = fadd double %scalar.188.310, %scalar.189.311
  store double %scalar.190.312, ptr %value.312, align 8
  %scalar.191.313 = fadd double %scalar.184.306, %scalar.190.312
  store double %scalar.191.313, ptr %value.313, align 8
  %scalar.192.314 = fsub double %scalar.191.313, %scalar.184.306
  store double %scalar.192.314, ptr %value.314, align 8
  %scalar.193.315 = fsub double %scalar.190.312, %scalar.192.314
  store double %scalar.193.315, ptr %value.315, align 8
  %scalar.194.31 = fadd double %scalar.191.313, %scalar.193.315
  store double %scalar.194.31, ptr %out.17, align 8
  %load.195.316.0 = load double, ptr %arg.10, align 8
  %scalar.195.316 = fadd double %load.195.316.0, %scalar.191.313
  store double %scalar.195.316, ptr %value.316, align 8
  %scalar.196.317 = fsub double %scalar.195.316, %load.195.316.0
  store double %scalar.196.317, ptr %value.317, align 8
  %scalar.197.318 = fsub double %scalar.195.316, %scalar.196.317
  store double %scalar.197.318, ptr %value.318, align 8
  %scalar.198.319 = fsub double %load.195.316.0, %scalar.197.318
  store double %scalar.198.319, ptr %value.319, align 8
  %scalar.199.320 = fsub double %scalar.191.313, %scalar.196.317
  store double %scalar.199.320, ptr %value.320, align 8
  %scalar.200.321 = fadd double %scalar.198.319, %scalar.199.320
  store double %scalar.200.321, ptr %value.321, align 8
  %load.201.322.1 = load double, ptr %arg.25, align 8
  %scalar.201.322 = fadd double %scalar.200.321, %load.201.322.1
  store double %scalar.201.322, ptr %value.322, align 8
  %scalar.202.323 = fadd double %scalar.201.322, %scalar.193.315
  store double %scalar.202.323, ptr %value.323, align 8
  %scalar.203.324 = fadd double %scalar.195.316, %scalar.202.323
  store double %scalar.203.324, ptr %value.324, align 8
  %scalar.204.325 = fsub double %scalar.203.324, %scalar.195.316
  store double %scalar.204.325, ptr %value.325, align 8
  %scalar.205.326 = fsub double %scalar.202.323, %scalar.204.325
  store double %scalar.205.326, ptr %value.326, align 8
  %scalar.206.32 = fadd double %scalar.203.324, %scalar.205.326
  store double %scalar.206.32, ptr %out.18, align 8
  %scalar.207.327 = fmul double %load.0.138.1, %scalar.203.324
  store double %scalar.207.327, ptr %value.327, align 8
  %scalar.208.328 = fneg double %scalar.207.327
  store double %scalar.208.328, ptr %value.328, align 8
  %scalar.209.329 = call double @llvm.fma.f64(double %load.0.138.1, double %scalar.203.324, double %scalar.208.328)
  store double %scalar.209.329, ptr %value.329, align 8
  %scalar.210.330 = fmul double %load.0.138.1, %scalar.205.326
  store double %scalar.210.330, ptr %value.330, align 8
  %scalar.211.331 = fadd double %scalar.209.329, %scalar.210.330
  store double %scalar.211.331, ptr %value.331, align 8
  %scalar.212.332 = fmul double %load.3.141.1, %scalar.203.324
  store double %scalar.212.332, ptr %value.332, align 8
  %scalar.213.333 = fadd double %scalar.211.331, %scalar.212.332
  store double %scalar.213.333, ptr %value.333, align 8
  %scalar.214.334 = fadd double %scalar.207.327, %scalar.213.333
  store double %scalar.214.334, ptr %value.334, align 8
  %scalar.215.335 = fsub double %scalar.214.334, %scalar.207.327
  store double %scalar.215.335, ptr %value.335, align 8
  %scalar.216.336 = fsub double %scalar.213.333, %scalar.215.335
  store double %scalar.216.336, ptr %value.336, align 8
  %scalar.217.33 = fadd double %scalar.214.334, %scalar.216.336
  store double %scalar.217.33, ptr %out.19, align 8
  %load.218.337.0 = load double, ptr %arg.11, align 8
  %scalar.218.337 = fadd double %load.218.337.0, %scalar.214.334
  store double %scalar.218.337, ptr %value.337, align 8
  %scalar.219.338 = fsub double %scalar.218.337, %load.218.337.0
  store double %scalar.219.338, ptr %value.338, align 8
  %scalar.220.339 = fsub double %scalar.218.337, %scalar.219.338
  store double %scalar.220.339, ptr %value.339, align 8
  %scalar.221.340 = fsub double %load.218.337.0, %scalar.220.339
  store double %scalar.221.340, ptr %value.340, align 8
  %scalar.222.341 = fsub double %scalar.214.334, %scalar.219.338
  store double %scalar.222.341, ptr %value.341, align 8
  %scalar.223.342 = fadd double %scalar.221.340, %scalar.222.341
  store double %scalar.223.342, ptr %value.342, align 8
  %load.224.343.1 = load double, ptr %arg.26, align 8
  %scalar.224.343 = fadd double %scalar.223.342, %load.224.343.1
  store double %scalar.224.343, ptr %value.343, align 8
  %scalar.225.344 = fadd double %scalar.224.343, %scalar.216.336
  store double %scalar.225.344, ptr %value.344, align 8
  %scalar.226.345 = fadd double %scalar.218.337, %scalar.225.344
  store double %scalar.226.345, ptr %value.345, align 8
  %scalar.227.346 = fsub double %scalar.226.345, %scalar.218.337
  store double %scalar.227.346, ptr %value.346, align 8
  %scalar.228.347 = fsub double %scalar.225.344, %scalar.227.346
  store double %scalar.228.347, ptr %value.347, align 8
  %scalar.229.34 = fadd double %scalar.226.345, %scalar.228.347
  store double %scalar.229.34, ptr %out.20, align 8
  %scalar.230.348 = fmul double %load.0.138.1, %scalar.226.345
  store double %scalar.230.348, ptr %value.348, align 8
  %scalar.231.349 = fneg double %scalar.230.348
  store double %scalar.231.349, ptr %value.349, align 8
  %scalar.232.350 = call double @llvm.fma.f64(double %load.0.138.1, double %scalar.226.345, double %scalar.231.349)
  store double %scalar.232.350, ptr %value.350, align 8
  %scalar.233.351 = fmul double %load.0.138.1, %scalar.228.347
  store double %scalar.233.351, ptr %value.351, align 8
  %scalar.234.352 = fadd double %scalar.232.350, %scalar.233.351
  store double %scalar.234.352, ptr %value.352, align 8
  %scalar.235.353 = fmul double %load.3.141.1, %scalar.226.345
  store double %scalar.235.353, ptr %value.353, align 8
  %scalar.236.354 = fadd double %scalar.234.352, %scalar.235.353
  store double %scalar.236.354, ptr %value.354, align 8
  %scalar.237.355 = fadd double %scalar.230.348, %scalar.236.354
  store double %scalar.237.355, ptr %value.355, align 8
  %scalar.238.356 = fsub double %scalar.237.355, %scalar.230.348
  store double %scalar.238.356, ptr %value.356, align 8
  %scalar.239.357 = fsub double %scalar.236.354, %scalar.238.356
  store double %scalar.239.357, ptr %value.357, align 8
  %scalar.240.35 = fadd double %scalar.237.355, %scalar.239.357
  store double %scalar.240.35, ptr %out.21, align 8
  %load.241.358.0 = load double, ptr %arg.12, align 8
  %scalar.241.358 = fadd double %load.241.358.0, %scalar.237.355
  store double %scalar.241.358, ptr %value.358, align 8
  %scalar.242.359 = fsub double %scalar.241.358, %load.241.358.0
  store double %scalar.242.359, ptr %value.359, align 8
  %scalar.243.360 = fsub double %scalar.241.358, %scalar.242.359
  store double %scalar.243.360, ptr %value.360, align 8
  %scalar.244.361 = fsub double %load.241.358.0, %scalar.243.360
  store double %scalar.244.361, ptr %value.361, align 8
  %scalar.245.362 = fsub double %scalar.237.355, %scalar.242.359
  store double %scalar.245.362, ptr %value.362, align 8
  %scalar.246.363 = fadd double %scalar.244.361, %scalar.245.362
  store double %scalar.246.363, ptr %value.363, align 8
  %load.247.364.1 = load double, ptr %arg.27, align 8
  %scalar.247.364 = fadd double %scalar.246.363, %load.247.364.1
  store double %scalar.247.364, ptr %value.364, align 8
  %scalar.248.365 = fadd double %scalar.247.364, %scalar.239.357
  store double %scalar.248.365, ptr %value.365, align 8
  %scalar.249.366 = fadd double %scalar.241.358, %scalar.248.365
  store double %scalar.249.366, ptr %value.366, align 8
  %scalar.250.367 = fsub double %scalar.249.366, %scalar.241.358
  store double %scalar.250.367, ptr %value.367, align 8
  %scalar.251.368 = fsub double %scalar.248.365, %scalar.250.367
  store double %scalar.251.368, ptr %value.368, align 8
  %scalar.252.36 = fadd double %scalar.249.366, %scalar.251.368
  store double %scalar.252.36, ptr %out.22, align 8
  %scalar.253.369 = fmul double %load.0.138.1, %scalar.249.366
  store double %scalar.253.369, ptr %value.369, align 8
  %scalar.254.370 = fneg double %scalar.253.369
  store double %scalar.254.370, ptr %value.370, align 8
  %scalar.255.371 = call double @llvm.fma.f64(double %load.0.138.1, double %scalar.249.366, double %scalar.254.370)
  store double %scalar.255.371, ptr %value.371, align 8
  %scalar.256.372 = fmul double %load.0.138.1, %scalar.251.368
  store double %scalar.256.372, ptr %value.372, align 8
  %scalar.257.373 = fadd double %scalar.255.371, %scalar.256.372
  store double %scalar.257.373, ptr %value.373, align 8
  %scalar.258.374 = fmul double %load.3.141.1, %scalar.249.366
  store double %scalar.258.374, ptr %value.374, align 8
  %scalar.259.375 = fadd double %scalar.257.373, %scalar.258.374
  store double %scalar.259.375, ptr %value.375, align 8
  %scalar.260.376 = fadd double %scalar.253.369, %scalar.259.375
  store double %scalar.260.376, ptr %value.376, align 8
  %scalar.261.377 = fsub double %scalar.260.376, %scalar.253.369
  store double %scalar.261.377, ptr %value.377, align 8
  %scalar.262.378 = fsub double %scalar.259.375, %scalar.261.377
  store double %scalar.262.378, ptr %value.378, align 8
  %scalar.263.37 = fadd double %scalar.260.376, %scalar.262.378
  store double %scalar.263.37, ptr %out.23, align 8
  %load.264.379.0 = load double, ptr %arg.13, align 8
  %scalar.264.379 = fadd double %load.264.379.0, %scalar.260.376
  store double %scalar.264.379, ptr %value.379, align 8
  %scalar.265.380 = fsub double %scalar.264.379, %load.264.379.0
  store double %scalar.265.380, ptr %value.380, align 8
  %scalar.266.381 = fsub double %scalar.264.379, %scalar.265.380
  store double %scalar.266.381, ptr %value.381, align 8
  %scalar.267.382 = fsub double %load.264.379.0, %scalar.266.381
  store double %scalar.267.382, ptr %value.382, align 8
  %scalar.268.383 = fsub double %scalar.260.376, %scalar.265.380
  store double %scalar.268.383, ptr %value.383, align 8
  %scalar.269.384 = fadd double %scalar.267.382, %scalar.268.383
  store double %scalar.269.384, ptr %value.384, align 8
  %load.270.385.1 = load double, ptr %arg.28, align 8
  %scalar.270.385 = fadd double %scalar.269.384, %load.270.385.1
  store double %scalar.270.385, ptr %value.385, align 8
  %scalar.271.386 = fadd double %scalar.270.385, %scalar.262.378
  store double %scalar.271.386, ptr %value.386, align 8
  %scalar.272.387 = fadd double %scalar.264.379, %scalar.271.386
  store double %scalar.272.387, ptr %value.387, align 8
  %scalar.273.388 = fsub double %scalar.272.387, %scalar.264.379
  store double %scalar.273.388, ptr %value.388, align 8
  %scalar.274.389 = fsub double %scalar.271.386, %scalar.273.388
  store double %scalar.274.389, ptr %value.389, align 8
  %scalar.275.38 = fadd double %scalar.272.387, %scalar.274.389
  store double %scalar.275.38, ptr %out.24, align 8
  %scalar.276.390 = fmul double %load.0.138.1, %scalar.272.387
  store double %scalar.276.390, ptr %value.390, align 8
  %scalar.277.391 = fneg double %scalar.276.390
  store double %scalar.277.391, ptr %value.391, align 8
  %scalar.278.392 = call double @llvm.fma.f64(double %load.0.138.1, double %scalar.272.387, double %scalar.277.391)
  store double %scalar.278.392, ptr %value.392, align 8
  %scalar.279.393 = fmul double %load.0.138.1, %scalar.274.389
  store double %scalar.279.393, ptr %value.393, align 8
  %scalar.280.394 = fadd double %scalar.278.392, %scalar.279.393
  store double %scalar.280.394, ptr %value.394, align 8
  %scalar.281.395 = fmul double %load.3.141.1, %scalar.272.387
  store double %scalar.281.395, ptr %value.395, align 8
  %scalar.282.396 = fadd double %scalar.280.394, %scalar.281.395
  store double %scalar.282.396, ptr %value.396, align 8
  %scalar.283.397 = fadd double %scalar.276.390, %scalar.282.396
  store double %scalar.283.397, ptr %value.397, align 8
  %scalar.284.398 = fsub double %scalar.283.397, %scalar.276.390
  store double %scalar.284.398, ptr %value.398, align 8
  %scalar.285.399 = fsub double %scalar.282.396, %scalar.284.398
  store double %scalar.285.399, ptr %value.399, align 8
  %scalar.286.39 = fadd double %scalar.283.397, %scalar.285.399
  store double %scalar.286.39, ptr %out.25, align 8
  %load.287.400.0 = load double, ptr %arg.14, align 8
  %scalar.287.400 = fadd double %load.287.400.0, %scalar.283.397
  store double %scalar.287.400, ptr %value.400, align 8
  %scalar.288.401 = fsub double %scalar.287.400, %load.287.400.0
  store double %scalar.288.401, ptr %value.401, align 8
  %scalar.289.402 = fsub double %scalar.287.400, %scalar.288.401
  store double %scalar.289.402, ptr %value.402, align 8
  %scalar.290.403 = fsub double %load.287.400.0, %scalar.289.402
  store double %scalar.290.403, ptr %value.403, align 8
  %scalar.291.404 = fsub double %scalar.283.397, %scalar.288.401
  store double %scalar.291.404, ptr %value.404, align 8
  %scalar.292.405 = fadd double %scalar.290.403, %scalar.291.404
  store double %scalar.292.405, ptr %value.405, align 8
  %load.293.406.1 = load double, ptr %arg.29, align 8
  %scalar.293.406 = fadd double %scalar.292.405, %load.293.406.1
  store double %scalar.293.406, ptr %value.406, align 8
  %scalar.294.407 = fadd double %scalar.293.406, %scalar.285.399
  store double %scalar.294.407, ptr %value.407, align 8
  %scalar.295.408 = fadd double %scalar.287.400, %scalar.294.407
  store double %scalar.295.408, ptr %value.408, align 8
  %scalar.296.409 = fsub double %scalar.295.408, %scalar.287.400
  store double %scalar.296.409, ptr %value.409, align 8
  %scalar.297.410 = fsub double %scalar.294.407, %scalar.296.409
  store double %scalar.297.410, ptr %value.410, align 8
  %scalar.298.40 = fadd double %scalar.295.408, %scalar.297.410
  store double %scalar.298.40, ptr %out.0, align 8
  ret void
}

define void @__ssa_exp_core_pack__exp_core(ptr noalias %arg.0, ptr noalias %arg.1, ptr noalias %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr noalias %arg.11, ptr noalias %arg.12, ptr noalias %arg.13, ptr %arg.14, ptr noalias %arg.15, ptr noalias %arg.16, ptr noalias %arg.17, ptr noalias %arg.18, ptr noalias %arg.19, ptr noalias %arg.20, ptr noalias %arg.21, ptr noalias %arg.22, ptr noalias %arg.23, ptr noalias %arg.24, ptr noalias %arg.25, ptr noalias %arg.26, ptr noalias %arg.27, ptr noalias %arg.28, ptr %arg.29, ptr %out.0) {
entry:
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
  %value.44 = alloca i32, i64 1, align 8
  %value.42 = alloca i64, i64 1, align 8
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
  %value.33 = alloca double, i64 1, align 8
  %value.34 = alloca double, i64 1, align 8
  %value.35 = alloca double, i64 1, align 8
  %value.36 = alloca double, i64 1, align 8
  %value.37 = alloca double, i64 1, align 8
  %value.38 = alloca double, i64 1, align 8
  %value.39 = alloca double, i64 1, align 8
  store i32 25, ptr %value.92, align 4
  store i32 24, ptr %value.90, align 4
  store i32 23, ptr %value.88, align 4
  store i32 22, ptr %value.86, align 4
  store i32 21, ptr %value.84, align 4
  store i32 20, ptr %value.82, align 4
  store i32 19, ptr %value.80, align 4
  store i32 18, ptr %value.78, align 4
  store i32 17, ptr %value.76, align 4
  store i32 16, ptr %value.74, align 4
  store i32 15, ptr %value.72, align 4
  store i32 14, ptr %value.70, align 4
  store i32 13, ptr %value.68, align 4
  store i32 12, ptr %value.66, align 4
  store i32 11, ptr %value.64, align 4
  store i32 10, ptr %value.62, align 4
  store i32 9, ptr %value.60, align 4
  store i32 8, ptr %value.58, align 4
  store i32 7, ptr %value.56, align 4
  store i32 6, ptr %value.54, align 4
  store i32 5, ptr %value.52, align 4
  store i32 4, ptr %value.50, align 4
  store i32 3, ptr %value.48, align 4
  store i32 2, ptr %value.46, align 4
  store i32 1, ptr %value.44, align 4
  store i64 0, ptr %value.42, align 8
  call void @__ssa_exp_core_pack__exp_core__planned_region_0(ptr %arg.5, ptr %arg.14, ptr %arg.4, ptr %arg.3, ptr %arg.2, ptr %arg.13, ptr %arg.12, ptr %arg.11, ptr %arg.10, ptr %arg.9, ptr %arg.8, ptr %arg.7, ptr %arg.6, ptr %arg.1, ptr %arg.0, ptr %arg.20, ptr %arg.29, ptr %arg.19, ptr %arg.18, ptr %arg.17, ptr %arg.28, ptr %arg.27, ptr %arg.26, ptr %arg.25, ptr %arg.24, ptr %arg.23, ptr %arg.22, ptr %arg.21, ptr %arg.16, ptr %arg.15, ptr %out.0, ptr %value.15, ptr %value.16, ptr %value.17, ptr %value.18, ptr %value.19, ptr %value.20, ptr %value.21, ptr %value.22, ptr %value.23, ptr %value.24, ptr %value.25, ptr %value.26, ptr %value.27, ptr %value.28, ptr %value.29, ptr %value.30, ptr %value.31, ptr %value.32, ptr %value.33, ptr %value.34, ptr %value.35, ptr %value.36, ptr %value.37, ptr %value.38, ptr %value.39)
  ret void
}

define void @exp_core_pack__exp_core_pack(ptr %buffers, ptr %extents) {
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
  call void @__ssa_exp_core_pack__exp_core_pack(ptr %public.0, ptr %public.1, ptr %public.2, ptr %public.3, ptr %public.4, ptr %public.5, ptr %public.6, ptr %public.7, ptr %public.8, ptr %public.9, ptr %public.10, ptr %public.11, ptr %public.12, ptr %public.13, ptr %public.14, ptr %public.15, ptr %public.16, ptr %public.17, ptr %public.18, ptr %public.19, ptr %public.20, ptr %public.21, ptr %public.22, ptr %public.23, ptr %public.24, ptr %public.25, ptr %public.26, ptr %public.27, ptr %public.28, ptr %public.29, ptr %public.30, ptr %public.2)
  ret void
}
