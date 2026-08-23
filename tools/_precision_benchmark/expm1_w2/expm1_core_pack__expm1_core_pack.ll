source_filename = "turing.ssa-llvm.expm1_core_pack__expm1_core_pack"

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-windows-gnu"

declare double @llvm.fma.f64(double, double, double)

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

define void @__ssa_expm1_core_pack__expm1_core_pack(ptr noalias %arg.0, ptr noalias %arg.1, ptr %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr noalias %arg.11, ptr noalias %arg.12, ptr noalias %arg.13, ptr noalias %arg.14, ptr noalias %arg.15, ptr noalias %arg.16, ptr noalias %arg.17, ptr noalias %arg.18, ptr noalias %arg.19, ptr noalias %arg.20, ptr noalias %arg.21, ptr noalias %arg.22, ptr noalias %arg.23, ptr noalias %arg.24, ptr noalias %arg.25, ptr noalias %arg.26, ptr noalias %arg.27, ptr noalias %arg.28, ptr noalias %arg.29, ptr noalias %arg.30, ptr %out.0) {
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
  call void @__ssa_expm1_core_pack__expm1_core(ptr %arg.3, ptr %arg.4, ptr %arg.5, ptr %arg.6, ptr %arg.7, ptr %arg.8, ptr %arg.9, ptr %arg.10, ptr %arg.11, ptr %arg.12, ptr %arg.13, ptr %arg.14, ptr %arg.15, ptr %arg.16, ptr %value.19, ptr %value.19, ptr %arg.17, ptr %arg.18, ptr %arg.19, ptr %arg.20, ptr %arg.21, ptr %arg.22, ptr %arg.23, ptr %arg.24, ptr %arg.25, ptr %arg.26, ptr %arg.27, ptr %arg.28, ptr %arg.29, ptr %arg.30, ptr %value.19, ptr %value.19, ptr %value.20)
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

define void @__ssa_expm1_core_pack__expm1_core__planned_region_0(ptr noalias %arg.0, ptr %arg.1, ptr noalias %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr noalias %arg.11, ptr noalias %arg.12, ptr noalias %arg.13, ptr noalias %arg.14, ptr %arg.15, ptr noalias %arg.16, ptr %arg.17, ptr noalias %arg.18, ptr noalias %arg.19, ptr noalias %arg.20, ptr noalias %arg.21, ptr noalias %arg.22, ptr noalias %arg.23, ptr noalias %arg.24, ptr noalias %arg.25, ptr noalias %arg.26, ptr noalias %arg.27, ptr noalias %arg.28, ptr noalias %arg.29, ptr noalias %arg.30, ptr %arg.31, ptr %out.0, ptr %out.1, ptr %out.2, ptr %out.3, ptr %out.4, ptr %out.5, ptr %out.6, ptr %out.7, ptr %out.8, ptr %out.9, ptr %out.10, ptr %out.11, ptr %out.12, ptr %out.13, ptr %out.14, ptr %out.15, ptr %out.16, ptr %out.17, ptr %out.18, ptr %out.19, ptr %out.20, ptr %out.21, ptr %out.22, ptr %out.23, ptr %out.24, ptr %out.25, ptr %out.26) {
entry:
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
  %value.411 = alloca double, i64 1, align 8
  %value.412 = alloca double, i64 1, align 8
  %value.413 = alloca double, i64 1, align 8
  %value.414 = alloca double, i64 1, align 8
  %value.415 = alloca double, i64 1, align 8
  %value.416 = alloca double, i64 1, align 8
  %value.417 = alloca double, i64 1, align 8
  %value.418 = alloca double, i64 1, align 8
  %value.419 = alloca double, i64 1, align 8
  %value.420 = alloca double, i64 1, align 8
  %value.421 = alloca double, i64 1, align 8
  %value.422 = alloca double, i64 1, align 8
  %value.423 = alloca double, i64 1, align 8
  %value.424 = alloca double, i64 1, align 8
  %value.425 = alloca double, i64 1, align 8
  %value.426 = alloca double, i64 1, align 8
  %load.0.144.0 = load double, ptr %arg.0, align 8
  %load.0.144.1 = load double, ptr %arg.1, align 8
  %scalar.0.144 = fmul double %load.0.144.0, %load.0.144.1
  store double %scalar.0.144, ptr %value.144, align 8
  %scalar.1.145 = fneg double %scalar.0.144
  store double %scalar.1.145, ptr %value.145, align 8
  %scalar.2.146 = call double @llvm.fma.f64(double %load.0.144.0, double %load.0.144.1, double %scalar.1.145)
  store double %scalar.2.146, ptr %value.146, align 8
  %load.3.147.1 = load double, ptr %arg.17, align 8
  %scalar.3.147 = fmul double %load.0.144.0, %load.3.147.1
  store double %scalar.3.147, ptr %value.147, align 8
  %scalar.4.148 = fadd double %scalar.2.146, %scalar.3.147
  store double %scalar.4.148, ptr %value.148, align 8
  %load.5.149.0 = load double, ptr %arg.16, align 8
  %scalar.5.149 = fmul double %load.5.149.0, %load.0.144.1
  store double %scalar.5.149, ptr %value.149, align 8
  %scalar.6.150 = fadd double %scalar.4.148, %scalar.5.149
  store double %scalar.6.150, ptr %value.150, align 8
  %scalar.7.151 = fadd double %scalar.0.144, %scalar.6.150
  store double %scalar.7.151, ptr %value.151, align 8
  %scalar.8.152 = fsub double %scalar.7.151, %scalar.0.144
  store double %scalar.8.152, ptr %value.152, align 8
  %scalar.9.153 = fsub double %scalar.6.150, %scalar.8.152
  store double %scalar.9.153, ptr %value.153, align 8
  %scalar.10.16 = fadd double %scalar.7.151, %scalar.9.153
  store double %scalar.10.16, ptr %out.1, align 8
  %load.11.154.0 = load double, ptr %arg.2, align 8
  %scalar.11.154 = fadd double %load.11.154.0, %scalar.7.151
  store double %scalar.11.154, ptr %value.154, align 8
  %scalar.12.155 = fsub double %scalar.11.154, %load.11.154.0
  store double %scalar.12.155, ptr %value.155, align 8
  %scalar.13.156 = fsub double %scalar.11.154, %scalar.12.155
  store double %scalar.13.156, ptr %value.156, align 8
  %scalar.14.157 = fsub double %load.11.154.0, %scalar.13.156
  store double %scalar.14.157, ptr %value.157, align 8
  %scalar.15.158 = fsub double %scalar.7.151, %scalar.12.155
  store double %scalar.15.158, ptr %value.158, align 8
  %scalar.16.159 = fadd double %scalar.14.157, %scalar.15.158
  store double %scalar.16.159, ptr %value.159, align 8
  %load.17.160.1 = load double, ptr %arg.18, align 8
  %scalar.17.160 = fadd double %scalar.16.159, %load.17.160.1
  store double %scalar.17.160, ptr %value.160, align 8
  %scalar.18.161 = fadd double %scalar.17.160, %scalar.9.153
  store double %scalar.18.161, ptr %value.161, align 8
  %scalar.19.162 = fadd double %scalar.11.154, %scalar.18.161
  store double %scalar.19.162, ptr %value.162, align 8
  %scalar.20.163 = fsub double %scalar.19.162, %scalar.11.154
  store double %scalar.20.163, ptr %value.163, align 8
  %scalar.21.164 = fsub double %scalar.18.161, %scalar.20.163
  store double %scalar.21.164, ptr %value.164, align 8
  %scalar.22.17 = fadd double %scalar.19.162, %scalar.21.164
  store double %scalar.22.17, ptr %out.2, align 8
  %scalar.23.165 = fmul double %load.0.144.1, %scalar.19.162
  store double %scalar.23.165, ptr %value.165, align 8
  %scalar.24.166 = fneg double %scalar.23.165
  store double %scalar.24.166, ptr %value.166, align 8
  %scalar.25.167 = call double @llvm.fma.f64(double %load.0.144.1, double %scalar.19.162, double %scalar.24.166)
  store double %scalar.25.167, ptr %value.167, align 8
  %scalar.26.168 = fmul double %load.0.144.1, %scalar.21.164
  store double %scalar.26.168, ptr %value.168, align 8
  %scalar.27.169 = fadd double %scalar.25.167, %scalar.26.168
  store double %scalar.27.169, ptr %value.169, align 8
  %scalar.28.170 = fmul double %load.3.147.1, %scalar.19.162
  store double %scalar.28.170, ptr %value.170, align 8
  %scalar.29.171 = fadd double %scalar.27.169, %scalar.28.170
  store double %scalar.29.171, ptr %value.171, align 8
  %scalar.30.172 = fadd double %scalar.23.165, %scalar.29.171
  store double %scalar.30.172, ptr %value.172, align 8
  %scalar.31.173 = fsub double %scalar.30.172, %scalar.23.165
  store double %scalar.31.173, ptr %value.173, align 8
  %scalar.32.174 = fsub double %scalar.29.171, %scalar.31.173
  store double %scalar.32.174, ptr %value.174, align 8
  %scalar.33.18 = fadd double %scalar.30.172, %scalar.32.174
  store double %scalar.33.18, ptr %out.3, align 8
  %load.34.175.0 = load double, ptr %arg.3, align 8
  %scalar.34.175 = fadd double %load.34.175.0, %scalar.30.172
  store double %scalar.34.175, ptr %value.175, align 8
  %scalar.35.176 = fsub double %scalar.34.175, %load.34.175.0
  store double %scalar.35.176, ptr %value.176, align 8
  %scalar.36.177 = fsub double %scalar.34.175, %scalar.35.176
  store double %scalar.36.177, ptr %value.177, align 8
  %scalar.37.178 = fsub double %load.34.175.0, %scalar.36.177
  store double %scalar.37.178, ptr %value.178, align 8
  %scalar.38.179 = fsub double %scalar.30.172, %scalar.35.176
  store double %scalar.38.179, ptr %value.179, align 8
  %scalar.39.180 = fadd double %scalar.37.178, %scalar.38.179
  store double %scalar.39.180, ptr %value.180, align 8
  %load.40.181.1 = load double, ptr %arg.19, align 8
  %scalar.40.181 = fadd double %scalar.39.180, %load.40.181.1
  store double %scalar.40.181, ptr %value.181, align 8
  %scalar.41.182 = fadd double %scalar.40.181, %scalar.32.174
  store double %scalar.41.182, ptr %value.182, align 8
  %scalar.42.183 = fadd double %scalar.34.175, %scalar.41.182
  store double %scalar.42.183, ptr %value.183, align 8
  %scalar.43.184 = fsub double %scalar.42.183, %scalar.34.175
  store double %scalar.43.184, ptr %value.184, align 8
  %scalar.44.185 = fsub double %scalar.41.182, %scalar.43.184
  store double %scalar.44.185, ptr %value.185, align 8
  %scalar.45.19 = fadd double %scalar.42.183, %scalar.44.185
  store double %scalar.45.19, ptr %out.4, align 8
  %scalar.46.186 = fmul double %load.0.144.1, %scalar.42.183
  store double %scalar.46.186, ptr %value.186, align 8
  %scalar.47.187 = fneg double %scalar.46.186
  store double %scalar.47.187, ptr %value.187, align 8
  %scalar.48.188 = call double @llvm.fma.f64(double %load.0.144.1, double %scalar.42.183, double %scalar.47.187)
  store double %scalar.48.188, ptr %value.188, align 8
  %scalar.49.189 = fmul double %load.0.144.1, %scalar.44.185
  store double %scalar.49.189, ptr %value.189, align 8
  %scalar.50.190 = fadd double %scalar.48.188, %scalar.49.189
  store double %scalar.50.190, ptr %value.190, align 8
  %scalar.51.191 = fmul double %load.3.147.1, %scalar.42.183
  store double %scalar.51.191, ptr %value.191, align 8
  %scalar.52.192 = fadd double %scalar.50.190, %scalar.51.191
  store double %scalar.52.192, ptr %value.192, align 8
  %scalar.53.193 = fadd double %scalar.46.186, %scalar.52.192
  store double %scalar.53.193, ptr %value.193, align 8
  %scalar.54.194 = fsub double %scalar.53.193, %scalar.46.186
  store double %scalar.54.194, ptr %value.194, align 8
  %scalar.55.195 = fsub double %scalar.52.192, %scalar.54.194
  store double %scalar.55.195, ptr %value.195, align 8
  %scalar.56.20 = fadd double %scalar.53.193, %scalar.55.195
  store double %scalar.56.20, ptr %out.5, align 8
  %load.57.196.0 = load double, ptr %arg.4, align 8
  %scalar.57.196 = fadd double %load.57.196.0, %scalar.53.193
  store double %scalar.57.196, ptr %value.196, align 8
  %scalar.58.197 = fsub double %scalar.57.196, %load.57.196.0
  store double %scalar.58.197, ptr %value.197, align 8
  %scalar.59.198 = fsub double %scalar.57.196, %scalar.58.197
  store double %scalar.59.198, ptr %value.198, align 8
  %scalar.60.199 = fsub double %load.57.196.0, %scalar.59.198
  store double %scalar.60.199, ptr %value.199, align 8
  %scalar.61.200 = fsub double %scalar.53.193, %scalar.58.197
  store double %scalar.61.200, ptr %value.200, align 8
  %scalar.62.201 = fadd double %scalar.60.199, %scalar.61.200
  store double %scalar.62.201, ptr %value.201, align 8
  %load.63.202.1 = load double, ptr %arg.20, align 8
  %scalar.63.202 = fadd double %scalar.62.201, %load.63.202.1
  store double %scalar.63.202, ptr %value.202, align 8
  %scalar.64.203 = fadd double %scalar.63.202, %scalar.55.195
  store double %scalar.64.203, ptr %value.203, align 8
  %scalar.65.204 = fadd double %scalar.57.196, %scalar.64.203
  store double %scalar.65.204, ptr %value.204, align 8
  %scalar.66.205 = fsub double %scalar.65.204, %scalar.57.196
  store double %scalar.66.205, ptr %value.205, align 8
  %scalar.67.206 = fsub double %scalar.64.203, %scalar.66.205
  store double %scalar.67.206, ptr %value.206, align 8
  %scalar.68.21 = fadd double %scalar.65.204, %scalar.67.206
  store double %scalar.68.21, ptr %out.6, align 8
  %scalar.69.207 = fmul double %load.0.144.1, %scalar.65.204
  store double %scalar.69.207, ptr %value.207, align 8
  %scalar.70.208 = fneg double %scalar.69.207
  store double %scalar.70.208, ptr %value.208, align 8
  %scalar.71.209 = call double @llvm.fma.f64(double %load.0.144.1, double %scalar.65.204, double %scalar.70.208)
  store double %scalar.71.209, ptr %value.209, align 8
  %scalar.72.210 = fmul double %load.0.144.1, %scalar.67.206
  store double %scalar.72.210, ptr %value.210, align 8
  %scalar.73.211 = fadd double %scalar.71.209, %scalar.72.210
  store double %scalar.73.211, ptr %value.211, align 8
  %scalar.74.212 = fmul double %load.3.147.1, %scalar.65.204
  store double %scalar.74.212, ptr %value.212, align 8
  %scalar.75.213 = fadd double %scalar.73.211, %scalar.74.212
  store double %scalar.75.213, ptr %value.213, align 8
  %scalar.76.214 = fadd double %scalar.69.207, %scalar.75.213
  store double %scalar.76.214, ptr %value.214, align 8
  %scalar.77.215 = fsub double %scalar.76.214, %scalar.69.207
  store double %scalar.77.215, ptr %value.215, align 8
  %scalar.78.216 = fsub double %scalar.75.213, %scalar.77.215
  store double %scalar.78.216, ptr %value.216, align 8
  %scalar.79.22 = fadd double %scalar.76.214, %scalar.78.216
  store double %scalar.79.22, ptr %out.7, align 8
  %load.80.217.0 = load double, ptr %arg.5, align 8
  %scalar.80.217 = fadd double %load.80.217.0, %scalar.76.214
  store double %scalar.80.217, ptr %value.217, align 8
  %scalar.81.218 = fsub double %scalar.80.217, %load.80.217.0
  store double %scalar.81.218, ptr %value.218, align 8
  %scalar.82.219 = fsub double %scalar.80.217, %scalar.81.218
  store double %scalar.82.219, ptr %value.219, align 8
  %scalar.83.220 = fsub double %load.80.217.0, %scalar.82.219
  store double %scalar.83.220, ptr %value.220, align 8
  %scalar.84.221 = fsub double %scalar.76.214, %scalar.81.218
  store double %scalar.84.221, ptr %value.221, align 8
  %scalar.85.222 = fadd double %scalar.83.220, %scalar.84.221
  store double %scalar.85.222, ptr %value.222, align 8
  %load.86.223.1 = load double, ptr %arg.21, align 8
  %scalar.86.223 = fadd double %scalar.85.222, %load.86.223.1
  store double %scalar.86.223, ptr %value.223, align 8
  %scalar.87.224 = fadd double %scalar.86.223, %scalar.78.216
  store double %scalar.87.224, ptr %value.224, align 8
  %scalar.88.225 = fadd double %scalar.80.217, %scalar.87.224
  store double %scalar.88.225, ptr %value.225, align 8
  %scalar.89.226 = fsub double %scalar.88.225, %scalar.80.217
  store double %scalar.89.226, ptr %value.226, align 8
  %scalar.90.227 = fsub double %scalar.87.224, %scalar.89.226
  store double %scalar.90.227, ptr %value.227, align 8
  %scalar.91.23 = fadd double %scalar.88.225, %scalar.90.227
  store double %scalar.91.23, ptr %out.8, align 8
  %scalar.92.228 = fmul double %load.0.144.1, %scalar.88.225
  store double %scalar.92.228, ptr %value.228, align 8
  %scalar.93.229 = fneg double %scalar.92.228
  store double %scalar.93.229, ptr %value.229, align 8
  %scalar.94.230 = call double @llvm.fma.f64(double %load.0.144.1, double %scalar.88.225, double %scalar.93.229)
  store double %scalar.94.230, ptr %value.230, align 8
  %scalar.95.231 = fmul double %load.0.144.1, %scalar.90.227
  store double %scalar.95.231, ptr %value.231, align 8
  %scalar.96.232 = fadd double %scalar.94.230, %scalar.95.231
  store double %scalar.96.232, ptr %value.232, align 8
  %scalar.97.233 = fmul double %load.3.147.1, %scalar.88.225
  store double %scalar.97.233, ptr %value.233, align 8
  %scalar.98.234 = fadd double %scalar.96.232, %scalar.97.233
  store double %scalar.98.234, ptr %value.234, align 8
  %scalar.99.235 = fadd double %scalar.92.228, %scalar.98.234
  store double %scalar.99.235, ptr %value.235, align 8
  %scalar.100.236 = fsub double %scalar.99.235, %scalar.92.228
  store double %scalar.100.236, ptr %value.236, align 8
  %scalar.101.237 = fsub double %scalar.98.234, %scalar.100.236
  store double %scalar.101.237, ptr %value.237, align 8
  %scalar.102.24 = fadd double %scalar.99.235, %scalar.101.237
  store double %scalar.102.24, ptr %out.9, align 8
  %load.103.238.0 = load double, ptr %arg.6, align 8
  %scalar.103.238 = fadd double %load.103.238.0, %scalar.99.235
  store double %scalar.103.238, ptr %value.238, align 8
  %scalar.104.239 = fsub double %scalar.103.238, %load.103.238.0
  store double %scalar.104.239, ptr %value.239, align 8
  %scalar.105.240 = fsub double %scalar.103.238, %scalar.104.239
  store double %scalar.105.240, ptr %value.240, align 8
  %scalar.106.241 = fsub double %load.103.238.0, %scalar.105.240
  store double %scalar.106.241, ptr %value.241, align 8
  %scalar.107.242 = fsub double %scalar.99.235, %scalar.104.239
  store double %scalar.107.242, ptr %value.242, align 8
  %scalar.108.243 = fadd double %scalar.106.241, %scalar.107.242
  store double %scalar.108.243, ptr %value.243, align 8
  %load.109.244.1 = load double, ptr %arg.22, align 8
  %scalar.109.244 = fadd double %scalar.108.243, %load.109.244.1
  store double %scalar.109.244, ptr %value.244, align 8
  %scalar.110.245 = fadd double %scalar.109.244, %scalar.101.237
  store double %scalar.110.245, ptr %value.245, align 8
  %scalar.111.246 = fadd double %scalar.103.238, %scalar.110.245
  store double %scalar.111.246, ptr %value.246, align 8
  %scalar.112.247 = fsub double %scalar.111.246, %scalar.103.238
  store double %scalar.112.247, ptr %value.247, align 8
  %scalar.113.248 = fsub double %scalar.110.245, %scalar.112.247
  store double %scalar.113.248, ptr %value.248, align 8
  %scalar.114.25 = fadd double %scalar.111.246, %scalar.113.248
  store double %scalar.114.25, ptr %out.10, align 8
  %scalar.115.249 = fmul double %load.0.144.1, %scalar.111.246
  store double %scalar.115.249, ptr %value.249, align 8
  %scalar.116.250 = fneg double %scalar.115.249
  store double %scalar.116.250, ptr %value.250, align 8
  %scalar.117.251 = call double @llvm.fma.f64(double %load.0.144.1, double %scalar.111.246, double %scalar.116.250)
  store double %scalar.117.251, ptr %value.251, align 8
  %scalar.118.252 = fmul double %load.0.144.1, %scalar.113.248
  store double %scalar.118.252, ptr %value.252, align 8
  %scalar.119.253 = fadd double %scalar.117.251, %scalar.118.252
  store double %scalar.119.253, ptr %value.253, align 8
  %scalar.120.254 = fmul double %load.3.147.1, %scalar.111.246
  store double %scalar.120.254, ptr %value.254, align 8
  %scalar.121.255 = fadd double %scalar.119.253, %scalar.120.254
  store double %scalar.121.255, ptr %value.255, align 8
  %scalar.122.256 = fadd double %scalar.115.249, %scalar.121.255
  store double %scalar.122.256, ptr %value.256, align 8
  %scalar.123.257 = fsub double %scalar.122.256, %scalar.115.249
  store double %scalar.123.257, ptr %value.257, align 8
  %scalar.124.258 = fsub double %scalar.121.255, %scalar.123.257
  store double %scalar.124.258, ptr %value.258, align 8
  %scalar.125.26 = fadd double %scalar.122.256, %scalar.124.258
  store double %scalar.125.26, ptr %out.11, align 8
  %load.126.259.0 = load double, ptr %arg.7, align 8
  %scalar.126.259 = fadd double %load.126.259.0, %scalar.122.256
  store double %scalar.126.259, ptr %value.259, align 8
  %scalar.127.260 = fsub double %scalar.126.259, %load.126.259.0
  store double %scalar.127.260, ptr %value.260, align 8
  %scalar.128.261 = fsub double %scalar.126.259, %scalar.127.260
  store double %scalar.128.261, ptr %value.261, align 8
  %scalar.129.262 = fsub double %load.126.259.0, %scalar.128.261
  store double %scalar.129.262, ptr %value.262, align 8
  %scalar.130.263 = fsub double %scalar.122.256, %scalar.127.260
  store double %scalar.130.263, ptr %value.263, align 8
  %scalar.131.264 = fadd double %scalar.129.262, %scalar.130.263
  store double %scalar.131.264, ptr %value.264, align 8
  %load.132.265.1 = load double, ptr %arg.23, align 8
  %scalar.132.265 = fadd double %scalar.131.264, %load.132.265.1
  store double %scalar.132.265, ptr %value.265, align 8
  %scalar.133.266 = fadd double %scalar.132.265, %scalar.124.258
  store double %scalar.133.266, ptr %value.266, align 8
  %scalar.134.267 = fadd double %scalar.126.259, %scalar.133.266
  store double %scalar.134.267, ptr %value.267, align 8
  %scalar.135.268 = fsub double %scalar.134.267, %scalar.126.259
  store double %scalar.135.268, ptr %value.268, align 8
  %scalar.136.269 = fsub double %scalar.133.266, %scalar.135.268
  store double %scalar.136.269, ptr %value.269, align 8
  %scalar.137.27 = fadd double %scalar.134.267, %scalar.136.269
  store double %scalar.137.27, ptr %out.12, align 8
  %scalar.138.270 = fmul double %load.0.144.1, %scalar.134.267
  store double %scalar.138.270, ptr %value.270, align 8
  %scalar.139.271 = fneg double %scalar.138.270
  store double %scalar.139.271, ptr %value.271, align 8
  %scalar.140.272 = call double @llvm.fma.f64(double %load.0.144.1, double %scalar.134.267, double %scalar.139.271)
  store double %scalar.140.272, ptr %value.272, align 8
  %scalar.141.273 = fmul double %load.0.144.1, %scalar.136.269
  store double %scalar.141.273, ptr %value.273, align 8
  %scalar.142.274 = fadd double %scalar.140.272, %scalar.141.273
  store double %scalar.142.274, ptr %value.274, align 8
  %scalar.143.275 = fmul double %load.3.147.1, %scalar.134.267
  store double %scalar.143.275, ptr %value.275, align 8
  %scalar.144.276 = fadd double %scalar.142.274, %scalar.143.275
  store double %scalar.144.276, ptr %value.276, align 8
  %scalar.145.277 = fadd double %scalar.138.270, %scalar.144.276
  store double %scalar.145.277, ptr %value.277, align 8
  %scalar.146.278 = fsub double %scalar.145.277, %scalar.138.270
  store double %scalar.146.278, ptr %value.278, align 8
  %scalar.147.279 = fsub double %scalar.144.276, %scalar.146.278
  store double %scalar.147.279, ptr %value.279, align 8
  %scalar.148.28 = fadd double %scalar.145.277, %scalar.147.279
  store double %scalar.148.28, ptr %out.13, align 8
  %load.149.280.0 = load double, ptr %arg.8, align 8
  %scalar.149.280 = fadd double %load.149.280.0, %scalar.145.277
  store double %scalar.149.280, ptr %value.280, align 8
  %scalar.150.281 = fsub double %scalar.149.280, %load.149.280.0
  store double %scalar.150.281, ptr %value.281, align 8
  %scalar.151.282 = fsub double %scalar.149.280, %scalar.150.281
  store double %scalar.151.282, ptr %value.282, align 8
  %scalar.152.283 = fsub double %load.149.280.0, %scalar.151.282
  store double %scalar.152.283, ptr %value.283, align 8
  %scalar.153.284 = fsub double %scalar.145.277, %scalar.150.281
  store double %scalar.153.284, ptr %value.284, align 8
  %scalar.154.285 = fadd double %scalar.152.283, %scalar.153.284
  store double %scalar.154.285, ptr %value.285, align 8
  %load.155.286.1 = load double, ptr %arg.24, align 8
  %scalar.155.286 = fadd double %scalar.154.285, %load.155.286.1
  store double %scalar.155.286, ptr %value.286, align 8
  %scalar.156.287 = fadd double %scalar.155.286, %scalar.147.279
  store double %scalar.156.287, ptr %value.287, align 8
  %scalar.157.288 = fadd double %scalar.149.280, %scalar.156.287
  store double %scalar.157.288, ptr %value.288, align 8
  %scalar.158.289 = fsub double %scalar.157.288, %scalar.149.280
  store double %scalar.158.289, ptr %value.289, align 8
  %scalar.159.290 = fsub double %scalar.156.287, %scalar.158.289
  store double %scalar.159.290, ptr %value.290, align 8
  %scalar.160.29 = fadd double %scalar.157.288, %scalar.159.290
  store double %scalar.160.29, ptr %out.14, align 8
  %scalar.161.291 = fmul double %load.0.144.1, %scalar.157.288
  store double %scalar.161.291, ptr %value.291, align 8
  %scalar.162.292 = fneg double %scalar.161.291
  store double %scalar.162.292, ptr %value.292, align 8
  %scalar.163.293 = call double @llvm.fma.f64(double %load.0.144.1, double %scalar.157.288, double %scalar.162.292)
  store double %scalar.163.293, ptr %value.293, align 8
  %scalar.164.294 = fmul double %load.0.144.1, %scalar.159.290
  store double %scalar.164.294, ptr %value.294, align 8
  %scalar.165.295 = fadd double %scalar.163.293, %scalar.164.294
  store double %scalar.165.295, ptr %value.295, align 8
  %scalar.166.296 = fmul double %load.3.147.1, %scalar.157.288
  store double %scalar.166.296, ptr %value.296, align 8
  %scalar.167.297 = fadd double %scalar.165.295, %scalar.166.296
  store double %scalar.167.297, ptr %value.297, align 8
  %scalar.168.298 = fadd double %scalar.161.291, %scalar.167.297
  store double %scalar.168.298, ptr %value.298, align 8
  %scalar.169.299 = fsub double %scalar.168.298, %scalar.161.291
  store double %scalar.169.299, ptr %value.299, align 8
  %scalar.170.300 = fsub double %scalar.167.297, %scalar.169.299
  store double %scalar.170.300, ptr %value.300, align 8
  %scalar.171.30 = fadd double %scalar.168.298, %scalar.170.300
  store double %scalar.171.30, ptr %out.15, align 8
  %load.172.301.0 = load double, ptr %arg.9, align 8
  %scalar.172.301 = fadd double %load.172.301.0, %scalar.168.298
  store double %scalar.172.301, ptr %value.301, align 8
  %scalar.173.302 = fsub double %scalar.172.301, %load.172.301.0
  store double %scalar.173.302, ptr %value.302, align 8
  %scalar.174.303 = fsub double %scalar.172.301, %scalar.173.302
  store double %scalar.174.303, ptr %value.303, align 8
  %scalar.175.304 = fsub double %load.172.301.0, %scalar.174.303
  store double %scalar.175.304, ptr %value.304, align 8
  %scalar.176.305 = fsub double %scalar.168.298, %scalar.173.302
  store double %scalar.176.305, ptr %value.305, align 8
  %scalar.177.306 = fadd double %scalar.175.304, %scalar.176.305
  store double %scalar.177.306, ptr %value.306, align 8
  %load.178.307.1 = load double, ptr %arg.25, align 8
  %scalar.178.307 = fadd double %scalar.177.306, %load.178.307.1
  store double %scalar.178.307, ptr %value.307, align 8
  %scalar.179.308 = fadd double %scalar.178.307, %scalar.170.300
  store double %scalar.179.308, ptr %value.308, align 8
  %scalar.180.309 = fadd double %scalar.172.301, %scalar.179.308
  store double %scalar.180.309, ptr %value.309, align 8
  %scalar.181.310 = fsub double %scalar.180.309, %scalar.172.301
  store double %scalar.181.310, ptr %value.310, align 8
  %scalar.182.311 = fsub double %scalar.179.308, %scalar.181.310
  store double %scalar.182.311, ptr %value.311, align 8
  %scalar.183.31 = fadd double %scalar.180.309, %scalar.182.311
  store double %scalar.183.31, ptr %out.16, align 8
  %scalar.184.312 = fmul double %load.0.144.1, %scalar.180.309
  store double %scalar.184.312, ptr %value.312, align 8
  %scalar.185.313 = fneg double %scalar.184.312
  store double %scalar.185.313, ptr %value.313, align 8
  %scalar.186.314 = call double @llvm.fma.f64(double %load.0.144.1, double %scalar.180.309, double %scalar.185.313)
  store double %scalar.186.314, ptr %value.314, align 8
  %scalar.187.315 = fmul double %load.0.144.1, %scalar.182.311
  store double %scalar.187.315, ptr %value.315, align 8
  %scalar.188.316 = fadd double %scalar.186.314, %scalar.187.315
  store double %scalar.188.316, ptr %value.316, align 8
  %scalar.189.317 = fmul double %load.3.147.1, %scalar.180.309
  store double %scalar.189.317, ptr %value.317, align 8
  %scalar.190.318 = fadd double %scalar.188.316, %scalar.189.317
  store double %scalar.190.318, ptr %value.318, align 8
  %scalar.191.319 = fadd double %scalar.184.312, %scalar.190.318
  store double %scalar.191.319, ptr %value.319, align 8
  %scalar.192.320 = fsub double %scalar.191.319, %scalar.184.312
  store double %scalar.192.320, ptr %value.320, align 8
  %scalar.193.321 = fsub double %scalar.190.318, %scalar.192.320
  store double %scalar.193.321, ptr %value.321, align 8
  %scalar.194.32 = fadd double %scalar.191.319, %scalar.193.321
  store double %scalar.194.32, ptr %out.17, align 8
  %load.195.322.0 = load double, ptr %arg.10, align 8
  %scalar.195.322 = fadd double %load.195.322.0, %scalar.191.319
  store double %scalar.195.322, ptr %value.322, align 8
  %scalar.196.323 = fsub double %scalar.195.322, %load.195.322.0
  store double %scalar.196.323, ptr %value.323, align 8
  %scalar.197.324 = fsub double %scalar.195.322, %scalar.196.323
  store double %scalar.197.324, ptr %value.324, align 8
  %scalar.198.325 = fsub double %load.195.322.0, %scalar.197.324
  store double %scalar.198.325, ptr %value.325, align 8
  %scalar.199.326 = fsub double %scalar.191.319, %scalar.196.323
  store double %scalar.199.326, ptr %value.326, align 8
  %scalar.200.327 = fadd double %scalar.198.325, %scalar.199.326
  store double %scalar.200.327, ptr %value.327, align 8
  %load.201.328.1 = load double, ptr %arg.26, align 8
  %scalar.201.328 = fadd double %scalar.200.327, %load.201.328.1
  store double %scalar.201.328, ptr %value.328, align 8
  %scalar.202.329 = fadd double %scalar.201.328, %scalar.193.321
  store double %scalar.202.329, ptr %value.329, align 8
  %scalar.203.330 = fadd double %scalar.195.322, %scalar.202.329
  store double %scalar.203.330, ptr %value.330, align 8
  %scalar.204.331 = fsub double %scalar.203.330, %scalar.195.322
  store double %scalar.204.331, ptr %value.331, align 8
  %scalar.205.332 = fsub double %scalar.202.329, %scalar.204.331
  store double %scalar.205.332, ptr %value.332, align 8
  %scalar.206.33 = fadd double %scalar.203.330, %scalar.205.332
  store double %scalar.206.33, ptr %out.18, align 8
  %scalar.207.333 = fmul double %load.0.144.1, %scalar.203.330
  store double %scalar.207.333, ptr %value.333, align 8
  %scalar.208.334 = fneg double %scalar.207.333
  store double %scalar.208.334, ptr %value.334, align 8
  %scalar.209.335 = call double @llvm.fma.f64(double %load.0.144.1, double %scalar.203.330, double %scalar.208.334)
  store double %scalar.209.335, ptr %value.335, align 8
  %scalar.210.336 = fmul double %load.0.144.1, %scalar.205.332
  store double %scalar.210.336, ptr %value.336, align 8
  %scalar.211.337 = fadd double %scalar.209.335, %scalar.210.336
  store double %scalar.211.337, ptr %value.337, align 8
  %scalar.212.338 = fmul double %load.3.147.1, %scalar.203.330
  store double %scalar.212.338, ptr %value.338, align 8
  %scalar.213.339 = fadd double %scalar.211.337, %scalar.212.338
  store double %scalar.213.339, ptr %value.339, align 8
  %scalar.214.340 = fadd double %scalar.207.333, %scalar.213.339
  store double %scalar.214.340, ptr %value.340, align 8
  %scalar.215.341 = fsub double %scalar.214.340, %scalar.207.333
  store double %scalar.215.341, ptr %value.341, align 8
  %scalar.216.342 = fsub double %scalar.213.339, %scalar.215.341
  store double %scalar.216.342, ptr %value.342, align 8
  %scalar.217.34 = fadd double %scalar.214.340, %scalar.216.342
  store double %scalar.217.34, ptr %out.19, align 8
  %load.218.343.0 = load double, ptr %arg.11, align 8
  %scalar.218.343 = fadd double %load.218.343.0, %scalar.214.340
  store double %scalar.218.343, ptr %value.343, align 8
  %scalar.219.344 = fsub double %scalar.218.343, %load.218.343.0
  store double %scalar.219.344, ptr %value.344, align 8
  %scalar.220.345 = fsub double %scalar.218.343, %scalar.219.344
  store double %scalar.220.345, ptr %value.345, align 8
  %scalar.221.346 = fsub double %load.218.343.0, %scalar.220.345
  store double %scalar.221.346, ptr %value.346, align 8
  %scalar.222.347 = fsub double %scalar.214.340, %scalar.219.344
  store double %scalar.222.347, ptr %value.347, align 8
  %scalar.223.348 = fadd double %scalar.221.346, %scalar.222.347
  store double %scalar.223.348, ptr %value.348, align 8
  %load.224.349.1 = load double, ptr %arg.27, align 8
  %scalar.224.349 = fadd double %scalar.223.348, %load.224.349.1
  store double %scalar.224.349, ptr %value.349, align 8
  %scalar.225.350 = fadd double %scalar.224.349, %scalar.216.342
  store double %scalar.225.350, ptr %value.350, align 8
  %scalar.226.351 = fadd double %scalar.218.343, %scalar.225.350
  store double %scalar.226.351, ptr %value.351, align 8
  %scalar.227.352 = fsub double %scalar.226.351, %scalar.218.343
  store double %scalar.227.352, ptr %value.352, align 8
  %scalar.228.353 = fsub double %scalar.225.350, %scalar.227.352
  store double %scalar.228.353, ptr %value.353, align 8
  %scalar.229.35 = fadd double %scalar.226.351, %scalar.228.353
  store double %scalar.229.35, ptr %out.20, align 8
  %scalar.230.354 = fmul double %load.0.144.1, %scalar.226.351
  store double %scalar.230.354, ptr %value.354, align 8
  %scalar.231.355 = fneg double %scalar.230.354
  store double %scalar.231.355, ptr %value.355, align 8
  %scalar.232.356 = call double @llvm.fma.f64(double %load.0.144.1, double %scalar.226.351, double %scalar.231.355)
  store double %scalar.232.356, ptr %value.356, align 8
  %scalar.233.357 = fmul double %load.0.144.1, %scalar.228.353
  store double %scalar.233.357, ptr %value.357, align 8
  %scalar.234.358 = fadd double %scalar.232.356, %scalar.233.357
  store double %scalar.234.358, ptr %value.358, align 8
  %scalar.235.359 = fmul double %load.3.147.1, %scalar.226.351
  store double %scalar.235.359, ptr %value.359, align 8
  %scalar.236.360 = fadd double %scalar.234.358, %scalar.235.359
  store double %scalar.236.360, ptr %value.360, align 8
  %scalar.237.361 = fadd double %scalar.230.354, %scalar.236.360
  store double %scalar.237.361, ptr %value.361, align 8
  %scalar.238.362 = fsub double %scalar.237.361, %scalar.230.354
  store double %scalar.238.362, ptr %value.362, align 8
  %scalar.239.363 = fsub double %scalar.236.360, %scalar.238.362
  store double %scalar.239.363, ptr %value.363, align 8
  %scalar.240.36 = fadd double %scalar.237.361, %scalar.239.363
  store double %scalar.240.36, ptr %out.21, align 8
  %load.241.364.0 = load double, ptr %arg.12, align 8
  %scalar.241.364 = fadd double %load.241.364.0, %scalar.237.361
  store double %scalar.241.364, ptr %value.364, align 8
  %scalar.242.365 = fsub double %scalar.241.364, %load.241.364.0
  store double %scalar.242.365, ptr %value.365, align 8
  %scalar.243.366 = fsub double %scalar.241.364, %scalar.242.365
  store double %scalar.243.366, ptr %value.366, align 8
  %scalar.244.367 = fsub double %load.241.364.0, %scalar.243.366
  store double %scalar.244.367, ptr %value.367, align 8
  %scalar.245.368 = fsub double %scalar.237.361, %scalar.242.365
  store double %scalar.245.368, ptr %value.368, align 8
  %scalar.246.369 = fadd double %scalar.244.367, %scalar.245.368
  store double %scalar.246.369, ptr %value.369, align 8
  %load.247.370.1 = load double, ptr %arg.28, align 8
  %scalar.247.370 = fadd double %scalar.246.369, %load.247.370.1
  store double %scalar.247.370, ptr %value.370, align 8
  %scalar.248.371 = fadd double %scalar.247.370, %scalar.239.363
  store double %scalar.248.371, ptr %value.371, align 8
  %scalar.249.372 = fadd double %scalar.241.364, %scalar.248.371
  store double %scalar.249.372, ptr %value.372, align 8
  %scalar.250.373 = fsub double %scalar.249.372, %scalar.241.364
  store double %scalar.250.373, ptr %value.373, align 8
  %scalar.251.374 = fsub double %scalar.248.371, %scalar.250.373
  store double %scalar.251.374, ptr %value.374, align 8
  %scalar.252.37 = fadd double %scalar.249.372, %scalar.251.374
  store double %scalar.252.37, ptr %out.22, align 8
  %scalar.253.375 = fmul double %load.0.144.1, %scalar.249.372
  store double %scalar.253.375, ptr %value.375, align 8
  %scalar.254.376 = fneg double %scalar.253.375
  store double %scalar.254.376, ptr %value.376, align 8
  %scalar.255.377 = call double @llvm.fma.f64(double %load.0.144.1, double %scalar.249.372, double %scalar.254.376)
  store double %scalar.255.377, ptr %value.377, align 8
  %scalar.256.378 = fmul double %load.0.144.1, %scalar.251.374
  store double %scalar.256.378, ptr %value.378, align 8
  %scalar.257.379 = fadd double %scalar.255.377, %scalar.256.378
  store double %scalar.257.379, ptr %value.379, align 8
  %scalar.258.380 = fmul double %load.3.147.1, %scalar.249.372
  store double %scalar.258.380, ptr %value.380, align 8
  %scalar.259.381 = fadd double %scalar.257.379, %scalar.258.380
  store double %scalar.259.381, ptr %value.381, align 8
  %scalar.260.382 = fadd double %scalar.253.375, %scalar.259.381
  store double %scalar.260.382, ptr %value.382, align 8
  %scalar.261.383 = fsub double %scalar.260.382, %scalar.253.375
  store double %scalar.261.383, ptr %value.383, align 8
  %scalar.262.384 = fsub double %scalar.259.381, %scalar.261.383
  store double %scalar.262.384, ptr %value.384, align 8
  %scalar.263.38 = fadd double %scalar.260.382, %scalar.262.384
  store double %scalar.263.38, ptr %out.23, align 8
  %load.264.385.0 = load double, ptr %arg.13, align 8
  %scalar.264.385 = fadd double %load.264.385.0, %scalar.260.382
  store double %scalar.264.385, ptr %value.385, align 8
  %scalar.265.386 = fsub double %scalar.264.385, %load.264.385.0
  store double %scalar.265.386, ptr %value.386, align 8
  %scalar.266.387 = fsub double %scalar.264.385, %scalar.265.386
  store double %scalar.266.387, ptr %value.387, align 8
  %scalar.267.388 = fsub double %load.264.385.0, %scalar.266.387
  store double %scalar.267.388, ptr %value.388, align 8
  %scalar.268.389 = fsub double %scalar.260.382, %scalar.265.386
  store double %scalar.268.389, ptr %value.389, align 8
  %scalar.269.390 = fadd double %scalar.267.388, %scalar.268.389
  store double %scalar.269.390, ptr %value.390, align 8
  %load.270.391.1 = load double, ptr %arg.29, align 8
  %scalar.270.391 = fadd double %scalar.269.390, %load.270.391.1
  store double %scalar.270.391, ptr %value.391, align 8
  %scalar.271.392 = fadd double %scalar.270.391, %scalar.262.384
  store double %scalar.271.392, ptr %value.392, align 8
  %scalar.272.393 = fadd double %scalar.264.385, %scalar.271.392
  store double %scalar.272.393, ptr %value.393, align 8
  %scalar.273.394 = fsub double %scalar.272.393, %scalar.264.385
  store double %scalar.273.394, ptr %value.394, align 8
  %scalar.274.395 = fsub double %scalar.271.392, %scalar.273.394
  store double %scalar.274.395, ptr %value.395, align 8
  %scalar.275.39 = fadd double %scalar.272.393, %scalar.274.395
  store double %scalar.275.39, ptr %out.24, align 8
  %scalar.276.396 = fmul double %load.0.144.1, %scalar.272.393
  store double %scalar.276.396, ptr %value.396, align 8
  %scalar.277.397 = fneg double %scalar.276.396
  store double %scalar.277.397, ptr %value.397, align 8
  %scalar.278.398 = call double @llvm.fma.f64(double %load.0.144.1, double %scalar.272.393, double %scalar.277.397)
  store double %scalar.278.398, ptr %value.398, align 8
  %scalar.279.399 = fmul double %load.0.144.1, %scalar.274.395
  store double %scalar.279.399, ptr %value.399, align 8
  %scalar.280.400 = fadd double %scalar.278.398, %scalar.279.399
  store double %scalar.280.400, ptr %value.400, align 8
  %scalar.281.401 = fmul double %load.3.147.1, %scalar.272.393
  store double %scalar.281.401, ptr %value.401, align 8
  %scalar.282.402 = fadd double %scalar.280.400, %scalar.281.401
  store double %scalar.282.402, ptr %value.402, align 8
  %scalar.283.403 = fadd double %scalar.276.396, %scalar.282.402
  store double %scalar.283.403, ptr %value.403, align 8
  %scalar.284.404 = fsub double %scalar.283.403, %scalar.276.396
  store double %scalar.284.404, ptr %value.404, align 8
  %scalar.285.405 = fsub double %scalar.282.402, %scalar.284.404
  store double %scalar.285.405, ptr %value.405, align 8
  %scalar.286.40 = fadd double %scalar.283.403, %scalar.285.405
  store double %scalar.286.40, ptr %out.25, align 8
  %load.287.406.0 = load double, ptr %arg.14, align 8
  %scalar.287.406 = fadd double %load.287.406.0, %scalar.283.403
  store double %scalar.287.406, ptr %value.406, align 8
  %scalar.288.407 = fsub double %scalar.287.406, %load.287.406.0
  store double %scalar.288.407, ptr %value.407, align 8
  %scalar.289.408 = fsub double %scalar.287.406, %scalar.288.407
  store double %scalar.289.408, ptr %value.408, align 8
  %scalar.290.409 = fsub double %load.287.406.0, %scalar.289.408
  store double %scalar.290.409, ptr %value.409, align 8
  %scalar.291.410 = fsub double %scalar.283.403, %scalar.288.407
  store double %scalar.291.410, ptr %value.410, align 8
  %scalar.292.411 = fadd double %scalar.290.409, %scalar.291.410
  store double %scalar.292.411, ptr %value.411, align 8
  %load.293.412.1 = load double, ptr %arg.30, align 8
  %scalar.293.412 = fadd double %scalar.292.411, %load.293.412.1
  store double %scalar.293.412, ptr %value.412, align 8
  %scalar.294.413 = fadd double %scalar.293.412, %scalar.285.405
  store double %scalar.294.413, ptr %value.413, align 8
  %scalar.295.414 = fadd double %scalar.287.406, %scalar.294.413
  store double %scalar.295.414, ptr %value.414, align 8
  %scalar.296.415 = fsub double %scalar.295.414, %scalar.287.406
  store double %scalar.296.415, ptr %value.415, align 8
  %scalar.297.416 = fsub double %scalar.294.413, %scalar.296.415
  store double %scalar.297.416, ptr %value.416, align 8
  %scalar.298.41 = fadd double %scalar.295.414, %scalar.297.416
  store double %scalar.298.41, ptr %out.26, align 8
  %load.299.417.0 = load double, ptr %arg.15, align 8
  %scalar.299.417 = fmul double %load.299.417.0, %scalar.295.414
  store double %scalar.299.417, ptr %value.417, align 8
  %scalar.300.418 = fneg double %scalar.299.417
  store double %scalar.300.418, ptr %value.418, align 8
  %scalar.301.419 = call double @llvm.fma.f64(double %load.299.417.0, double %scalar.295.414, double %scalar.300.418)
  store double %scalar.301.419, ptr %value.419, align 8
  %scalar.302.420 = fmul double %load.299.417.0, %scalar.297.416
  store double %scalar.302.420, ptr %value.420, align 8
  %scalar.303.421 = fadd double %scalar.301.419, %scalar.302.420
  store double %scalar.303.421, ptr %value.421, align 8
  %load.304.422.0 = load double, ptr %arg.31, align 8
  %scalar.304.422 = fmul double %load.304.422.0, %scalar.295.414
  store double %scalar.304.422, ptr %value.422, align 8
  %scalar.305.423 = fadd double %scalar.303.421, %scalar.304.422
  store double %scalar.305.423, ptr %value.423, align 8
  %scalar.306.424 = fadd double %scalar.299.417, %scalar.305.423
  store double %scalar.306.424, ptr %value.424, align 8
  %scalar.307.425 = fsub double %scalar.306.424, %scalar.299.417
  store double %scalar.307.425, ptr %value.425, align 8
  %scalar.308.426 = fsub double %scalar.305.423, %scalar.307.425
  store double %scalar.308.426, ptr %value.426, align 8
  %scalar.309.42 = fadd double %scalar.306.424, %scalar.308.426
  store double %scalar.309.42, ptr %out.0, align 8
  ret void
}

define void @__ssa_expm1_core_pack__expm1_core(ptr noalias %arg.0, ptr noalias %arg.1, ptr noalias %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr noalias %arg.11, ptr noalias %arg.12, ptr noalias %arg.13, ptr %arg.14, ptr %arg.15, ptr noalias %arg.16, ptr noalias %arg.17, ptr noalias %arg.18, ptr noalias %arg.19, ptr noalias %arg.20, ptr noalias %arg.21, ptr noalias %arg.22, ptr noalias %arg.23, ptr noalias %arg.24, ptr noalias %arg.25, ptr noalias %arg.26, ptr noalias %arg.27, ptr noalias %arg.28, ptr noalias %arg.29, ptr %arg.30, ptr %arg.31, ptr %out.0) {
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
  call void @__ssa_expm1_core_pack__expm1_core__planned_region_0(ptr %arg.5, ptr %arg.14, ptr %arg.4, ptr %arg.3, ptr %arg.2, ptr %arg.13, ptr %arg.12, ptr %arg.11, ptr %arg.10, ptr %arg.9, ptr %arg.8, ptr %arg.7, ptr %arg.6, ptr %arg.1, ptr %arg.0, ptr %arg.15, ptr %arg.21, ptr %arg.30, ptr %arg.20, ptr %arg.19, ptr %arg.18, ptr %arg.29, ptr %arg.28, ptr %arg.27, ptr %arg.26, ptr %arg.25, ptr %arg.24, ptr %arg.23, ptr %arg.22, ptr %arg.17, ptr %arg.16, ptr %arg.31, ptr %out.0, ptr %value.16, ptr %value.17, ptr %value.18, ptr %value.19, ptr %value.20, ptr %value.21, ptr %value.22, ptr %value.23, ptr %value.24, ptr %value.25, ptr %value.26, ptr %value.27, ptr %value.28, ptr %value.29, ptr %value.30, ptr %value.31, ptr %value.32, ptr %value.33, ptr %value.34, ptr %value.35, ptr %value.36, ptr %value.37, ptr %value.38, ptr %value.39, ptr %value.40, ptr %value.41)
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
  call void @__ssa_expm1_core_pack__expm1_core_pack(ptr %public.0, ptr %public.1, ptr %public.2, ptr %public.3, ptr %public.4, ptr %public.5, ptr %public.6, ptr %public.7, ptr %public.8, ptr %public.9, ptr %public.10, ptr %public.11, ptr %public.12, ptr %public.13, ptr %public.14, ptr %public.15, ptr %public.16, ptr %public.17, ptr %public.18, ptr %public.19, ptr %public.20, ptr %public.21, ptr %public.22, ptr %public.23, ptr %public.24, ptr %public.25, ptr %public.26, ptr %public.27, ptr %public.28, ptr %public.29, ptr %public.30, ptr %public.2)
  ret void
}
