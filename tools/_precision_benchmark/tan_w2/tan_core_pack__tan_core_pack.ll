source_filename = "turing.ssa-llvm.tan_core_pack__tan_core_pack"

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-windows-gnu"

declare double @llvm.fma.f64(double, double, double)

define void @__ssa_tan_core_pack__tan_core_pack__planned_region_0(ptr noalias %arg.0, ptr %arg.1, ptr %out.0, ptr %out.1) {
entry:
  %value.184 = alloca i32, i64 1, align 8
  %value.185 = alloca i32, i64 1, align 8
  %value.187 = alloca double, i64 1, align 8
  %value.188 = alloca i32, i64 1, align 8
  %value.189 = alloca i32, i64 1, align 8
  %value.190 = alloca i32, i64 1, align 8
  %value.191 = alloca i32, i64 1, align 8
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
  %load.0.38.0 = load i32, ptr %arg.1, align 4
  %address.0.38 = getelementptr double, ptr %arg.0, i32 %load.0.38.0
  store i32 2, ptr %value.184, align 4
  %scalar.2.185 = mul i32 %load.0.38.0, 2
  store i32 %scalar.2.185, ptr %value.185, align 4
  %address.3.186 = getelementptr double, ptr %arg.0, i32 %scalar.2.185
  %pinned.load.4.187 = load double, ptr %address.3.186, align 8
  store double %pinned.load.4.187, ptr %value.187, align 8
  store i32 2, ptr %value.188, align 4
  %scalar.6.189 = mul i32 %load.0.38.0, 2
  store i32 %scalar.6.189, ptr %value.189, align 4
  store i32 1, ptr %value.190, align 4
  %scalar.8.191 = add i32 %scalar.6.189, 1
  store i32 %scalar.8.191, ptr %value.191, align 4
  %address.9.192 = getelementptr double, ptr %arg.0, i32 %scalar.8.191
  %pinned.load.10.193 = load double, ptr %address.9.192, align 8
  store double %pinned.load.10.193, ptr %value.193, align 8
  %load.11.194.0 = load double, ptr %value.187, align 8
  %scalar.11.194 = fmul double %load.11.194.0, %load.11.194.0
  store double %scalar.11.194, ptr %value.194, align 8
  %scalar.12.195 = fneg double %scalar.11.194
  store double %scalar.12.195, ptr %value.195, align 8
  %scalar.13.196 = call double @llvm.fma.f64(double %load.11.194.0, double %load.11.194.0, double %scalar.12.195)
  store double %scalar.13.196, ptr %value.196, align 8
  %load.14.197.1 = load double, ptr %value.193, align 8
  %scalar.14.197 = fmul double %load.11.194.0, %load.14.197.1
  store double %scalar.14.197, ptr %value.197, align 8
  %scalar.15.198 = fadd double %scalar.13.196, %scalar.14.197
  store double %scalar.15.198, ptr %value.198, align 8
  %scalar.16.199 = fmul double %load.14.197.1, %load.11.194.0
  store double %scalar.16.199, ptr %value.199, align 8
  %scalar.17.200 = fadd double %scalar.15.198, %scalar.16.199
  store double %scalar.17.200, ptr %value.200, align 8
  %scalar.18.201 = fadd double %scalar.11.194, %scalar.17.200
  store double %scalar.18.201, ptr %value.201, align 8
  %scalar.19.202 = fsub double %scalar.18.201, %scalar.11.194
  store double %scalar.19.202, ptr %value.202, align 8
  %scalar.20.203 = fsub double %scalar.17.200, %scalar.19.202
  store double %scalar.20.203, ptr %value.203, align 8
  %scalar.21.24 = fadd double %scalar.18.201, %scalar.20.203
  store double %scalar.21.24, ptr %out.1, align 8
  ret void
}

define void @__ssa_tan_core_pack__tan_core_pack__planned_region_1(ptr %arg.0, ptr %arg.1, ptr %arg.2) {
entry:
  %load.0.39.0 = load i32, ptr %arg.1, align 4
  %address.0.39 = getelementptr double, ptr %arg.0, i32 %load.0.39.0
  %load.store.1.v = load double, ptr %arg.2, align 8
  store double %load.store.1.v, ptr %address.0.39, align 8
  ret void
}

define void @__ssa_tan_core_pack__tan_core_pack(ptr noalias %arg.0, ptr noalias %arg.1, ptr %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr noalias %arg.11, ptr noalias %arg.12, ptr noalias %arg.13, ptr noalias %arg.14, ptr noalias %arg.15, ptr noalias %arg.16, ptr noalias %arg.17, ptr noalias %arg.18, ptr noalias %arg.19, ptr noalias %arg.20, ptr noalias %arg.21, ptr noalias %arg.22, ptr noalias %arg.23, ptr noalias %arg.24, ptr noalias %arg.25, ptr noalias %arg.26, ptr noalias %arg.27, ptr noalias %arg.28, ptr noalias %arg.29, ptr noalias %arg.30, ptr noalias %arg.31, ptr noalias %arg.32, ptr noalias %arg.33, ptr noalias %arg.34, ptr noalias %arg.35, ptr noalias %arg.36, ptr noalias %arg.37, ptr noalias %arg.38, ptr %out.0) {
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
  call void @__ssa_tan_core_pack__tan_core_pack__planned_region_0(ptr %arg.1, ptr %phi.30, ptr %value.23, ptr %value.24)
  call void @__ssa_tan_core_pack__tan_core(ptr %arg.3, ptr %arg.4, ptr %arg.5, ptr %arg.6, ptr %arg.7, ptr %arg.8, ptr %arg.9, ptr %arg.10, ptr %arg.11, ptr %arg.12, ptr %arg.13, ptr %arg.14, ptr %arg.15, ptr %arg.16, ptr %arg.17, ptr %arg.18, ptr %arg.19, ptr %arg.20, ptr %value.24, ptr %value.23, ptr %arg.21, ptr %arg.22, ptr %arg.23, ptr %arg.24, ptr %arg.25, ptr %arg.26, ptr %arg.27, ptr %arg.28, ptr %arg.29, ptr %arg.30, ptr %arg.31, ptr %arg.32, ptr %arg.33, ptr %arg.34, ptr %arg.35, ptr %arg.36, ptr %arg.37, ptr %arg.38, ptr %value.24, ptr %value.23, ptr %value.25)
  call void @__ssa_tan_core_pack__tan_core_pack__planned_region_1(ptr %arg.2, ptr %phi.30, ptr %value.25)
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

define void @__ssa_tan_core_pack__tan_core__planned_region_0(ptr noalias %arg.0, ptr %arg.1, ptr noalias %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr noalias %arg.11, ptr noalias %arg.12, ptr noalias %arg.13, ptr noalias %arg.14, ptr noalias %arg.15, ptr noalias %arg.16, ptr noalias %arg.17, ptr noalias %arg.18, ptr %arg.19, ptr noalias %arg.20, ptr %arg.21, ptr noalias %arg.22, ptr noalias %arg.23, ptr noalias %arg.24, ptr noalias %arg.25, ptr noalias %arg.26, ptr noalias %arg.27, ptr noalias %arg.28, ptr noalias %arg.29, ptr noalias %arg.30, ptr noalias %arg.31, ptr noalias %arg.32, ptr noalias %arg.33, ptr noalias %arg.34, ptr noalias %arg.35, ptr noalias %arg.36, ptr noalias %arg.37, ptr noalias %arg.38, ptr %arg.39, ptr %out.0, ptr %out.1, ptr %out.2, ptr %out.3, ptr %out.4, ptr %out.5, ptr %out.6, ptr %out.7, ptr %out.8, ptr %out.9, ptr %out.10, ptr %out.11, ptr %out.12, ptr %out.13, ptr %out.14, ptr %out.15, ptr %out.16, ptr %out.17, ptr %out.18, ptr %out.19, ptr %out.20, ptr %out.21, ptr %out.22, ptr %out.23, ptr %out.24, ptr %out.25, ptr %out.26, ptr %out.27, ptr %out.28, ptr %out.29, ptr %out.30, ptr %out.31, ptr %out.32, ptr %out.33, ptr %out.34) {
entry:
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
  %value.427 = alloca double, i64 1, align 8
  %value.428 = alloca double, i64 1, align 8
  %value.429 = alloca double, i64 1, align 8
  %value.430 = alloca double, i64 1, align 8
  %value.431 = alloca double, i64 1, align 8
  %value.432 = alloca double, i64 1, align 8
  %value.433 = alloca double, i64 1, align 8
  %value.434 = alloca double, i64 1, align 8
  %value.435 = alloca double, i64 1, align 8
  %value.436 = alloca double, i64 1, align 8
  %value.437 = alloca double, i64 1, align 8
  %value.438 = alloca double, i64 1, align 8
  %value.439 = alloca double, i64 1, align 8
  %value.440 = alloca double, i64 1, align 8
  %value.441 = alloca double, i64 1, align 8
  %value.442 = alloca double, i64 1, align 8
  %value.443 = alloca double, i64 1, align 8
  %value.444 = alloca double, i64 1, align 8
  %value.445 = alloca double, i64 1, align 8
  %value.446 = alloca double, i64 1, align 8
  %value.447 = alloca double, i64 1, align 8
  %value.448 = alloca double, i64 1, align 8
  %value.449 = alloca double, i64 1, align 8
  %value.450 = alloca double, i64 1, align 8
  %value.451 = alloca double, i64 1, align 8
  %value.452 = alloca double, i64 1, align 8
  %value.453 = alloca double, i64 1, align 8
  %value.454 = alloca double, i64 1, align 8
  %value.455 = alloca double, i64 1, align 8
  %value.456 = alloca double, i64 1, align 8
  %value.457 = alloca double, i64 1, align 8
  %value.458 = alloca double, i64 1, align 8
  %value.459 = alloca double, i64 1, align 8
  %value.460 = alloca double, i64 1, align 8
  %value.461 = alloca double, i64 1, align 8
  %value.462 = alloca double, i64 1, align 8
  %value.463 = alloca double, i64 1, align 8
  %value.464 = alloca double, i64 1, align 8
  %value.465 = alloca double, i64 1, align 8
  %value.466 = alloca double, i64 1, align 8
  %value.467 = alloca double, i64 1, align 8
  %value.468 = alloca double, i64 1, align 8
  %value.469 = alloca double, i64 1, align 8
  %value.470 = alloca double, i64 1, align 8
  %value.471 = alloca double, i64 1, align 8
  %value.472 = alloca double, i64 1, align 8
  %value.473 = alloca double, i64 1, align 8
  %value.474 = alloca double, i64 1, align 8
  %value.475 = alloca double, i64 1, align 8
  %value.476 = alloca double, i64 1, align 8
  %value.477 = alloca double, i64 1, align 8
  %value.478 = alloca double, i64 1, align 8
  %value.479 = alloca double, i64 1, align 8
  %value.480 = alloca double, i64 1, align 8
  %value.481 = alloca double, i64 1, align 8
  %value.482 = alloca double, i64 1, align 8
  %value.483 = alloca double, i64 1, align 8
  %value.484 = alloca double, i64 1, align 8
  %value.485 = alloca double, i64 1, align 8
  %value.486 = alloca double, i64 1, align 8
  %value.487 = alloca double, i64 1, align 8
  %value.488 = alloca double, i64 1, align 8
  %value.489 = alloca double, i64 1, align 8
  %value.490 = alloca double, i64 1, align 8
  %value.491 = alloca double, i64 1, align 8
  %value.492 = alloca double, i64 1, align 8
  %value.493 = alloca double, i64 1, align 8
  %value.494 = alloca double, i64 1, align 8
  %value.495 = alloca double, i64 1, align 8
  %value.496 = alloca double, i64 1, align 8
  %value.497 = alloca double, i64 1, align 8
  %value.498 = alloca double, i64 1, align 8
  %value.499 = alloca double, i64 1, align 8
  %value.500 = alloca double, i64 1, align 8
  %value.501 = alloca double, i64 1, align 8
  %value.502 = alloca double, i64 1, align 8
  %value.503 = alloca double, i64 1, align 8
  %value.504 = alloca double, i64 1, align 8
  %value.505 = alloca double, i64 1, align 8
  %value.506 = alloca double, i64 1, align 8
  %value.507 = alloca double, i64 1, align 8
  %value.508 = alloca double, i64 1, align 8
  %value.509 = alloca double, i64 1, align 8
  %value.510 = alloca double, i64 1, align 8
  %value.511 = alloca double, i64 1, align 8
  %value.512 = alloca double, i64 1, align 8
  %value.513 = alloca double, i64 1, align 8
  %value.514 = alloca double, i64 1, align 8
  %value.515 = alloca double, i64 1, align 8
  %value.516 = alloca double, i64 1, align 8
  %value.517 = alloca double, i64 1, align 8
  %value.518 = alloca double, i64 1, align 8
  %value.519 = alloca double, i64 1, align 8
  %value.520 = alloca double, i64 1, align 8
  %value.521 = alloca double, i64 1, align 8
  %value.522 = alloca double, i64 1, align 8
  %value.523 = alloca double, i64 1, align 8
  %value.524 = alloca double, i64 1, align 8
  %value.525 = alloca double, i64 1, align 8
  %value.526 = alloca double, i64 1, align 8
  %value.527 = alloca double, i64 1, align 8
  %value.528 = alloca double, i64 1, align 8
  %value.529 = alloca double, i64 1, align 8
  %value.530 = alloca double, i64 1, align 8
  %value.531 = alloca double, i64 1, align 8
  %value.532 = alloca double, i64 1, align 8
  %value.533 = alloca double, i64 1, align 8
  %value.534 = alloca double, i64 1, align 8
  %value.535 = alloca double, i64 1, align 8
  %value.536 = alloca double, i64 1, align 8
  %value.537 = alloca double, i64 1, align 8
  %value.538 = alloca double, i64 1, align 8
  %value.539 = alloca double, i64 1, align 8
  %value.540 = alloca double, i64 1, align 8
  %value.541 = alloca double, i64 1, align 8
  %value.542 = alloca double, i64 1, align 8
  %value.543 = alloca double, i64 1, align 8
  %value.544 = alloca double, i64 1, align 8
  %value.545 = alloca double, i64 1, align 8
  %value.546 = alloca double, i64 1, align 8
  %value.547 = alloca double, i64 1, align 8
  %value.548 = alloca double, i64 1, align 8
  %value.549 = alloca double, i64 1, align 8
  %value.550 = alloca double, i64 1, align 8
  %value.551 = alloca double, i64 1, align 8
  %value.552 = alloca double, i64 1, align 8
  %value.553 = alloca double, i64 1, align 8
  %value.554 = alloca double, i64 1, align 8
  %value.555 = alloca double, i64 1, align 8
  %value.556 = alloca double, i64 1, align 8
  %value.557 = alloca double, i64 1, align 8
  %value.558 = alloca double, i64 1, align 8
  %value.559 = alloca double, i64 1, align 8
  %value.560 = alloca double, i64 1, align 8
  %value.561 = alloca double, i64 1, align 8
  %value.562 = alloca double, i64 1, align 8
  %value.563 = alloca double, i64 1, align 8
  %value.564 = alloca double, i64 1, align 8
  %value.565 = alloca double, i64 1, align 8
  %value.566 = alloca double, i64 1, align 8
  %value.567 = alloca double, i64 1, align 8
  %value.568 = alloca double, i64 1, align 8
  %value.569 = alloca double, i64 1, align 8
  %value.570 = alloca double, i64 1, align 8
  %load.0.204.0 = load double, ptr %arg.0, align 8
  %load.0.204.1 = load double, ptr %arg.1, align 8
  %scalar.0.204 = fmul double %load.0.204.0, %load.0.204.1
  store double %scalar.0.204, ptr %value.204, align 8
  %scalar.1.205 = fneg double %scalar.0.204
  store double %scalar.1.205, ptr %value.205, align 8
  %scalar.2.206 = call double @llvm.fma.f64(double %load.0.204.0, double %load.0.204.1, double %scalar.1.205)
  store double %scalar.2.206, ptr %value.206, align 8
  %load.3.207.1 = load double, ptr %arg.21, align 8
  %scalar.3.207 = fmul double %load.0.204.0, %load.3.207.1
  store double %scalar.3.207, ptr %value.207, align 8
  %scalar.4.208 = fadd double %scalar.2.206, %scalar.3.207
  store double %scalar.4.208, ptr %value.208, align 8
  %load.5.209.0 = load double, ptr %arg.20, align 8
  %scalar.5.209 = fmul double %load.5.209.0, %load.0.204.1
  store double %scalar.5.209, ptr %value.209, align 8
  %scalar.6.210 = fadd double %scalar.4.208, %scalar.5.209
  store double %scalar.6.210, ptr %value.210, align 8
  %scalar.7.211 = fadd double %scalar.0.204, %scalar.6.210
  store double %scalar.7.211, ptr %value.211, align 8
  %scalar.8.212 = fsub double %scalar.7.211, %scalar.0.204
  store double %scalar.8.212, ptr %value.212, align 8
  %scalar.9.213 = fsub double %scalar.6.210, %scalar.8.212
  store double %scalar.9.213, ptr %value.213, align 8
  %scalar.10.20 = fadd double %scalar.7.211, %scalar.9.213
  store double %scalar.10.20, ptr %out.1, align 8
  %load.11.214.0 = load double, ptr %arg.2, align 8
  %scalar.11.214 = fadd double %load.11.214.0, %scalar.7.211
  store double %scalar.11.214, ptr %value.214, align 8
  %scalar.12.215 = fsub double %scalar.11.214, %load.11.214.0
  store double %scalar.12.215, ptr %value.215, align 8
  %scalar.13.216 = fsub double %scalar.11.214, %scalar.12.215
  store double %scalar.13.216, ptr %value.216, align 8
  %scalar.14.217 = fsub double %load.11.214.0, %scalar.13.216
  store double %scalar.14.217, ptr %value.217, align 8
  %scalar.15.218 = fsub double %scalar.7.211, %scalar.12.215
  store double %scalar.15.218, ptr %value.218, align 8
  %scalar.16.219 = fadd double %scalar.14.217, %scalar.15.218
  store double %scalar.16.219, ptr %value.219, align 8
  %load.17.220.1 = load double, ptr %arg.22, align 8
  %scalar.17.220 = fadd double %scalar.16.219, %load.17.220.1
  store double %scalar.17.220, ptr %value.220, align 8
  %scalar.18.221 = fadd double %scalar.17.220, %scalar.9.213
  store double %scalar.18.221, ptr %value.221, align 8
  %scalar.19.222 = fadd double %scalar.11.214, %scalar.18.221
  store double %scalar.19.222, ptr %value.222, align 8
  %scalar.20.223 = fsub double %scalar.19.222, %scalar.11.214
  store double %scalar.20.223, ptr %value.223, align 8
  %scalar.21.224 = fsub double %scalar.18.221, %scalar.20.223
  store double %scalar.21.224, ptr %value.224, align 8
  %scalar.22.21 = fadd double %scalar.19.222, %scalar.21.224
  store double %scalar.22.21, ptr %out.2, align 8
  %scalar.23.225 = fmul double %load.0.204.1, %scalar.19.222
  store double %scalar.23.225, ptr %value.225, align 8
  %scalar.24.226 = fneg double %scalar.23.225
  store double %scalar.24.226, ptr %value.226, align 8
  %scalar.25.227 = call double @llvm.fma.f64(double %load.0.204.1, double %scalar.19.222, double %scalar.24.226)
  store double %scalar.25.227, ptr %value.227, align 8
  %scalar.26.228 = fmul double %load.0.204.1, %scalar.21.224
  store double %scalar.26.228, ptr %value.228, align 8
  %scalar.27.229 = fadd double %scalar.25.227, %scalar.26.228
  store double %scalar.27.229, ptr %value.229, align 8
  %scalar.28.230 = fmul double %load.3.207.1, %scalar.19.222
  store double %scalar.28.230, ptr %value.230, align 8
  %scalar.29.231 = fadd double %scalar.27.229, %scalar.28.230
  store double %scalar.29.231, ptr %value.231, align 8
  %scalar.30.232 = fadd double %scalar.23.225, %scalar.29.231
  store double %scalar.30.232, ptr %value.232, align 8
  %scalar.31.233 = fsub double %scalar.30.232, %scalar.23.225
  store double %scalar.31.233, ptr %value.233, align 8
  %scalar.32.234 = fsub double %scalar.29.231, %scalar.31.233
  store double %scalar.32.234, ptr %value.234, align 8
  %scalar.33.22 = fadd double %scalar.30.232, %scalar.32.234
  store double %scalar.33.22, ptr %out.3, align 8
  %load.34.235.0 = load double, ptr %arg.3, align 8
  %scalar.34.235 = fadd double %load.34.235.0, %scalar.30.232
  store double %scalar.34.235, ptr %value.235, align 8
  %scalar.35.236 = fsub double %scalar.34.235, %load.34.235.0
  store double %scalar.35.236, ptr %value.236, align 8
  %scalar.36.237 = fsub double %scalar.34.235, %scalar.35.236
  store double %scalar.36.237, ptr %value.237, align 8
  %scalar.37.238 = fsub double %load.34.235.0, %scalar.36.237
  store double %scalar.37.238, ptr %value.238, align 8
  %scalar.38.239 = fsub double %scalar.30.232, %scalar.35.236
  store double %scalar.38.239, ptr %value.239, align 8
  %scalar.39.240 = fadd double %scalar.37.238, %scalar.38.239
  store double %scalar.39.240, ptr %value.240, align 8
  %load.40.241.1 = load double, ptr %arg.23, align 8
  %scalar.40.241 = fadd double %scalar.39.240, %load.40.241.1
  store double %scalar.40.241, ptr %value.241, align 8
  %scalar.41.242 = fadd double %scalar.40.241, %scalar.32.234
  store double %scalar.41.242, ptr %value.242, align 8
  %scalar.42.243 = fadd double %scalar.34.235, %scalar.41.242
  store double %scalar.42.243, ptr %value.243, align 8
  %scalar.43.244 = fsub double %scalar.42.243, %scalar.34.235
  store double %scalar.43.244, ptr %value.244, align 8
  %scalar.44.245 = fsub double %scalar.41.242, %scalar.43.244
  store double %scalar.44.245, ptr %value.245, align 8
  %scalar.45.23 = fadd double %scalar.42.243, %scalar.44.245
  store double %scalar.45.23, ptr %out.4, align 8
  %scalar.46.246 = fmul double %load.0.204.1, %scalar.42.243
  store double %scalar.46.246, ptr %value.246, align 8
  %scalar.47.247 = fneg double %scalar.46.246
  store double %scalar.47.247, ptr %value.247, align 8
  %scalar.48.248 = call double @llvm.fma.f64(double %load.0.204.1, double %scalar.42.243, double %scalar.47.247)
  store double %scalar.48.248, ptr %value.248, align 8
  %scalar.49.249 = fmul double %load.0.204.1, %scalar.44.245
  store double %scalar.49.249, ptr %value.249, align 8
  %scalar.50.250 = fadd double %scalar.48.248, %scalar.49.249
  store double %scalar.50.250, ptr %value.250, align 8
  %scalar.51.251 = fmul double %load.3.207.1, %scalar.42.243
  store double %scalar.51.251, ptr %value.251, align 8
  %scalar.52.252 = fadd double %scalar.50.250, %scalar.51.251
  store double %scalar.52.252, ptr %value.252, align 8
  %scalar.53.253 = fadd double %scalar.46.246, %scalar.52.252
  store double %scalar.53.253, ptr %value.253, align 8
  %scalar.54.254 = fsub double %scalar.53.253, %scalar.46.246
  store double %scalar.54.254, ptr %value.254, align 8
  %scalar.55.255 = fsub double %scalar.52.252, %scalar.54.254
  store double %scalar.55.255, ptr %value.255, align 8
  %scalar.56.24 = fadd double %scalar.53.253, %scalar.55.255
  store double %scalar.56.24, ptr %out.5, align 8
  %load.57.256.0 = load double, ptr %arg.4, align 8
  %scalar.57.256 = fadd double %load.57.256.0, %scalar.53.253
  store double %scalar.57.256, ptr %value.256, align 8
  %scalar.58.257 = fsub double %scalar.57.256, %load.57.256.0
  store double %scalar.58.257, ptr %value.257, align 8
  %scalar.59.258 = fsub double %scalar.57.256, %scalar.58.257
  store double %scalar.59.258, ptr %value.258, align 8
  %scalar.60.259 = fsub double %load.57.256.0, %scalar.59.258
  store double %scalar.60.259, ptr %value.259, align 8
  %scalar.61.260 = fsub double %scalar.53.253, %scalar.58.257
  store double %scalar.61.260, ptr %value.260, align 8
  %scalar.62.261 = fadd double %scalar.60.259, %scalar.61.260
  store double %scalar.62.261, ptr %value.261, align 8
  %load.63.262.1 = load double, ptr %arg.24, align 8
  %scalar.63.262 = fadd double %scalar.62.261, %load.63.262.1
  store double %scalar.63.262, ptr %value.262, align 8
  %scalar.64.263 = fadd double %scalar.63.262, %scalar.55.255
  store double %scalar.64.263, ptr %value.263, align 8
  %scalar.65.264 = fadd double %scalar.57.256, %scalar.64.263
  store double %scalar.65.264, ptr %value.264, align 8
  %scalar.66.265 = fsub double %scalar.65.264, %scalar.57.256
  store double %scalar.66.265, ptr %value.265, align 8
  %scalar.67.266 = fsub double %scalar.64.263, %scalar.66.265
  store double %scalar.67.266, ptr %value.266, align 8
  %scalar.68.25 = fadd double %scalar.65.264, %scalar.67.266
  store double %scalar.68.25, ptr %out.6, align 8
  %scalar.69.267 = fmul double %load.0.204.1, %scalar.65.264
  store double %scalar.69.267, ptr %value.267, align 8
  %scalar.70.268 = fneg double %scalar.69.267
  store double %scalar.70.268, ptr %value.268, align 8
  %scalar.71.269 = call double @llvm.fma.f64(double %load.0.204.1, double %scalar.65.264, double %scalar.70.268)
  store double %scalar.71.269, ptr %value.269, align 8
  %scalar.72.270 = fmul double %load.0.204.1, %scalar.67.266
  store double %scalar.72.270, ptr %value.270, align 8
  %scalar.73.271 = fadd double %scalar.71.269, %scalar.72.270
  store double %scalar.73.271, ptr %value.271, align 8
  %scalar.74.272 = fmul double %load.3.207.1, %scalar.65.264
  store double %scalar.74.272, ptr %value.272, align 8
  %scalar.75.273 = fadd double %scalar.73.271, %scalar.74.272
  store double %scalar.75.273, ptr %value.273, align 8
  %scalar.76.274 = fadd double %scalar.69.267, %scalar.75.273
  store double %scalar.76.274, ptr %value.274, align 8
  %scalar.77.275 = fsub double %scalar.76.274, %scalar.69.267
  store double %scalar.77.275, ptr %value.275, align 8
  %scalar.78.276 = fsub double %scalar.75.273, %scalar.77.275
  store double %scalar.78.276, ptr %value.276, align 8
  %scalar.79.26 = fadd double %scalar.76.274, %scalar.78.276
  store double %scalar.79.26, ptr %out.7, align 8
  %load.80.277.0 = load double, ptr %arg.5, align 8
  %scalar.80.277 = fadd double %load.80.277.0, %scalar.76.274
  store double %scalar.80.277, ptr %value.277, align 8
  %scalar.81.278 = fsub double %scalar.80.277, %load.80.277.0
  store double %scalar.81.278, ptr %value.278, align 8
  %scalar.82.279 = fsub double %scalar.80.277, %scalar.81.278
  store double %scalar.82.279, ptr %value.279, align 8
  %scalar.83.280 = fsub double %load.80.277.0, %scalar.82.279
  store double %scalar.83.280, ptr %value.280, align 8
  %scalar.84.281 = fsub double %scalar.76.274, %scalar.81.278
  store double %scalar.84.281, ptr %value.281, align 8
  %scalar.85.282 = fadd double %scalar.83.280, %scalar.84.281
  store double %scalar.85.282, ptr %value.282, align 8
  %load.86.283.1 = load double, ptr %arg.25, align 8
  %scalar.86.283 = fadd double %scalar.85.282, %load.86.283.1
  store double %scalar.86.283, ptr %value.283, align 8
  %scalar.87.284 = fadd double %scalar.86.283, %scalar.78.276
  store double %scalar.87.284, ptr %value.284, align 8
  %scalar.88.285 = fadd double %scalar.80.277, %scalar.87.284
  store double %scalar.88.285, ptr %value.285, align 8
  %scalar.89.286 = fsub double %scalar.88.285, %scalar.80.277
  store double %scalar.89.286, ptr %value.286, align 8
  %scalar.90.287 = fsub double %scalar.87.284, %scalar.89.286
  store double %scalar.90.287, ptr %value.287, align 8
  %scalar.91.27 = fadd double %scalar.88.285, %scalar.90.287
  store double %scalar.91.27, ptr %out.8, align 8
  %scalar.92.288 = fmul double %load.0.204.1, %scalar.88.285
  store double %scalar.92.288, ptr %value.288, align 8
  %scalar.93.289 = fneg double %scalar.92.288
  store double %scalar.93.289, ptr %value.289, align 8
  %scalar.94.290 = call double @llvm.fma.f64(double %load.0.204.1, double %scalar.88.285, double %scalar.93.289)
  store double %scalar.94.290, ptr %value.290, align 8
  %scalar.95.291 = fmul double %load.0.204.1, %scalar.90.287
  store double %scalar.95.291, ptr %value.291, align 8
  %scalar.96.292 = fadd double %scalar.94.290, %scalar.95.291
  store double %scalar.96.292, ptr %value.292, align 8
  %scalar.97.293 = fmul double %load.3.207.1, %scalar.88.285
  store double %scalar.97.293, ptr %value.293, align 8
  %scalar.98.294 = fadd double %scalar.96.292, %scalar.97.293
  store double %scalar.98.294, ptr %value.294, align 8
  %scalar.99.295 = fadd double %scalar.92.288, %scalar.98.294
  store double %scalar.99.295, ptr %value.295, align 8
  %scalar.100.296 = fsub double %scalar.99.295, %scalar.92.288
  store double %scalar.100.296, ptr %value.296, align 8
  %scalar.101.297 = fsub double %scalar.98.294, %scalar.100.296
  store double %scalar.101.297, ptr %value.297, align 8
  %scalar.102.28 = fadd double %scalar.99.295, %scalar.101.297
  store double %scalar.102.28, ptr %out.9, align 8
  %load.103.298.0 = load double, ptr %arg.6, align 8
  %scalar.103.298 = fadd double %load.103.298.0, %scalar.99.295
  store double %scalar.103.298, ptr %value.298, align 8
  %scalar.104.299 = fsub double %scalar.103.298, %load.103.298.0
  store double %scalar.104.299, ptr %value.299, align 8
  %scalar.105.300 = fsub double %scalar.103.298, %scalar.104.299
  store double %scalar.105.300, ptr %value.300, align 8
  %scalar.106.301 = fsub double %load.103.298.0, %scalar.105.300
  store double %scalar.106.301, ptr %value.301, align 8
  %scalar.107.302 = fsub double %scalar.99.295, %scalar.104.299
  store double %scalar.107.302, ptr %value.302, align 8
  %scalar.108.303 = fadd double %scalar.106.301, %scalar.107.302
  store double %scalar.108.303, ptr %value.303, align 8
  %load.109.304.1 = load double, ptr %arg.26, align 8
  %scalar.109.304 = fadd double %scalar.108.303, %load.109.304.1
  store double %scalar.109.304, ptr %value.304, align 8
  %scalar.110.305 = fadd double %scalar.109.304, %scalar.101.297
  store double %scalar.110.305, ptr %value.305, align 8
  %scalar.111.306 = fadd double %scalar.103.298, %scalar.110.305
  store double %scalar.111.306, ptr %value.306, align 8
  %scalar.112.307 = fsub double %scalar.111.306, %scalar.103.298
  store double %scalar.112.307, ptr %value.307, align 8
  %scalar.113.308 = fsub double %scalar.110.305, %scalar.112.307
  store double %scalar.113.308, ptr %value.308, align 8
  %scalar.114.29 = fadd double %scalar.111.306, %scalar.113.308
  store double %scalar.114.29, ptr %out.10, align 8
  %scalar.115.309 = fmul double %load.0.204.1, %scalar.111.306
  store double %scalar.115.309, ptr %value.309, align 8
  %scalar.116.310 = fneg double %scalar.115.309
  store double %scalar.116.310, ptr %value.310, align 8
  %scalar.117.311 = call double @llvm.fma.f64(double %load.0.204.1, double %scalar.111.306, double %scalar.116.310)
  store double %scalar.117.311, ptr %value.311, align 8
  %scalar.118.312 = fmul double %load.0.204.1, %scalar.113.308
  store double %scalar.118.312, ptr %value.312, align 8
  %scalar.119.313 = fadd double %scalar.117.311, %scalar.118.312
  store double %scalar.119.313, ptr %value.313, align 8
  %scalar.120.314 = fmul double %load.3.207.1, %scalar.111.306
  store double %scalar.120.314, ptr %value.314, align 8
  %scalar.121.315 = fadd double %scalar.119.313, %scalar.120.314
  store double %scalar.121.315, ptr %value.315, align 8
  %scalar.122.316 = fadd double %scalar.115.309, %scalar.121.315
  store double %scalar.122.316, ptr %value.316, align 8
  %scalar.123.317 = fsub double %scalar.122.316, %scalar.115.309
  store double %scalar.123.317, ptr %value.317, align 8
  %scalar.124.318 = fsub double %scalar.121.315, %scalar.123.317
  store double %scalar.124.318, ptr %value.318, align 8
  %scalar.125.30 = fadd double %scalar.122.316, %scalar.124.318
  store double %scalar.125.30, ptr %out.11, align 8
  %load.126.319.0 = load double, ptr %arg.7, align 8
  %scalar.126.319 = fadd double %load.126.319.0, %scalar.122.316
  store double %scalar.126.319, ptr %value.319, align 8
  %scalar.127.320 = fsub double %scalar.126.319, %load.126.319.0
  store double %scalar.127.320, ptr %value.320, align 8
  %scalar.128.321 = fsub double %scalar.126.319, %scalar.127.320
  store double %scalar.128.321, ptr %value.321, align 8
  %scalar.129.322 = fsub double %load.126.319.0, %scalar.128.321
  store double %scalar.129.322, ptr %value.322, align 8
  %scalar.130.323 = fsub double %scalar.122.316, %scalar.127.320
  store double %scalar.130.323, ptr %value.323, align 8
  %scalar.131.324 = fadd double %scalar.129.322, %scalar.130.323
  store double %scalar.131.324, ptr %value.324, align 8
  %load.132.325.1 = load double, ptr %arg.27, align 8
  %scalar.132.325 = fadd double %scalar.131.324, %load.132.325.1
  store double %scalar.132.325, ptr %value.325, align 8
  %scalar.133.326 = fadd double %scalar.132.325, %scalar.124.318
  store double %scalar.133.326, ptr %value.326, align 8
  %scalar.134.327 = fadd double %scalar.126.319, %scalar.133.326
  store double %scalar.134.327, ptr %value.327, align 8
  %scalar.135.328 = fsub double %scalar.134.327, %scalar.126.319
  store double %scalar.135.328, ptr %value.328, align 8
  %scalar.136.329 = fsub double %scalar.133.326, %scalar.135.328
  store double %scalar.136.329, ptr %value.329, align 8
  %scalar.137.31 = fadd double %scalar.134.327, %scalar.136.329
  store double %scalar.137.31, ptr %out.12, align 8
  %scalar.138.330 = fmul double %load.0.204.1, %scalar.134.327
  store double %scalar.138.330, ptr %value.330, align 8
  %scalar.139.331 = fneg double %scalar.138.330
  store double %scalar.139.331, ptr %value.331, align 8
  %scalar.140.332 = call double @llvm.fma.f64(double %load.0.204.1, double %scalar.134.327, double %scalar.139.331)
  store double %scalar.140.332, ptr %value.332, align 8
  %scalar.141.333 = fmul double %load.0.204.1, %scalar.136.329
  store double %scalar.141.333, ptr %value.333, align 8
  %scalar.142.334 = fadd double %scalar.140.332, %scalar.141.333
  store double %scalar.142.334, ptr %value.334, align 8
  %scalar.143.335 = fmul double %load.3.207.1, %scalar.134.327
  store double %scalar.143.335, ptr %value.335, align 8
  %scalar.144.336 = fadd double %scalar.142.334, %scalar.143.335
  store double %scalar.144.336, ptr %value.336, align 8
  %scalar.145.337 = fadd double %scalar.138.330, %scalar.144.336
  store double %scalar.145.337, ptr %value.337, align 8
  %scalar.146.338 = fsub double %scalar.145.337, %scalar.138.330
  store double %scalar.146.338, ptr %value.338, align 8
  %scalar.147.339 = fsub double %scalar.144.336, %scalar.146.338
  store double %scalar.147.339, ptr %value.339, align 8
  %scalar.148.32 = fadd double %scalar.145.337, %scalar.147.339
  store double %scalar.148.32, ptr %out.13, align 8
  %load.149.340.0 = load double, ptr %arg.8, align 8
  %scalar.149.340 = fadd double %load.149.340.0, %scalar.145.337
  store double %scalar.149.340, ptr %value.340, align 8
  %scalar.150.341 = fsub double %scalar.149.340, %load.149.340.0
  store double %scalar.150.341, ptr %value.341, align 8
  %scalar.151.342 = fsub double %scalar.149.340, %scalar.150.341
  store double %scalar.151.342, ptr %value.342, align 8
  %scalar.152.343 = fsub double %load.149.340.0, %scalar.151.342
  store double %scalar.152.343, ptr %value.343, align 8
  %scalar.153.344 = fsub double %scalar.145.337, %scalar.150.341
  store double %scalar.153.344, ptr %value.344, align 8
  %scalar.154.345 = fadd double %scalar.152.343, %scalar.153.344
  store double %scalar.154.345, ptr %value.345, align 8
  %load.155.346.1 = load double, ptr %arg.28, align 8
  %scalar.155.346 = fadd double %scalar.154.345, %load.155.346.1
  store double %scalar.155.346, ptr %value.346, align 8
  %scalar.156.347 = fadd double %scalar.155.346, %scalar.147.339
  store double %scalar.156.347, ptr %value.347, align 8
  %scalar.157.348 = fadd double %scalar.149.340, %scalar.156.347
  store double %scalar.157.348, ptr %value.348, align 8
  %scalar.158.349 = fsub double %scalar.157.348, %scalar.149.340
  store double %scalar.158.349, ptr %value.349, align 8
  %scalar.159.350 = fsub double %scalar.156.347, %scalar.158.349
  store double %scalar.159.350, ptr %value.350, align 8
  %scalar.160.33 = fadd double %scalar.157.348, %scalar.159.350
  store double %scalar.160.33, ptr %out.14, align 8
  %scalar.161.351 = fmul double %load.0.204.1, %scalar.157.348
  store double %scalar.161.351, ptr %value.351, align 8
  %scalar.162.352 = fneg double %scalar.161.351
  store double %scalar.162.352, ptr %value.352, align 8
  %scalar.163.353 = call double @llvm.fma.f64(double %load.0.204.1, double %scalar.157.348, double %scalar.162.352)
  store double %scalar.163.353, ptr %value.353, align 8
  %scalar.164.354 = fmul double %load.0.204.1, %scalar.159.350
  store double %scalar.164.354, ptr %value.354, align 8
  %scalar.165.355 = fadd double %scalar.163.353, %scalar.164.354
  store double %scalar.165.355, ptr %value.355, align 8
  %scalar.166.356 = fmul double %load.3.207.1, %scalar.157.348
  store double %scalar.166.356, ptr %value.356, align 8
  %scalar.167.357 = fadd double %scalar.165.355, %scalar.166.356
  store double %scalar.167.357, ptr %value.357, align 8
  %scalar.168.358 = fadd double %scalar.161.351, %scalar.167.357
  store double %scalar.168.358, ptr %value.358, align 8
  %scalar.169.359 = fsub double %scalar.168.358, %scalar.161.351
  store double %scalar.169.359, ptr %value.359, align 8
  %scalar.170.360 = fsub double %scalar.167.357, %scalar.169.359
  store double %scalar.170.360, ptr %value.360, align 8
  %scalar.171.34 = fadd double %scalar.168.358, %scalar.170.360
  store double %scalar.171.34, ptr %out.15, align 8
  %load.172.361.0 = load double, ptr %arg.9, align 8
  %scalar.172.361 = fadd double %load.172.361.0, %scalar.168.358
  store double %scalar.172.361, ptr %value.361, align 8
  %scalar.173.362 = fsub double %scalar.172.361, %load.172.361.0
  store double %scalar.173.362, ptr %value.362, align 8
  %scalar.174.363 = fsub double %scalar.172.361, %scalar.173.362
  store double %scalar.174.363, ptr %value.363, align 8
  %scalar.175.364 = fsub double %load.172.361.0, %scalar.174.363
  store double %scalar.175.364, ptr %value.364, align 8
  %scalar.176.365 = fsub double %scalar.168.358, %scalar.173.362
  store double %scalar.176.365, ptr %value.365, align 8
  %scalar.177.366 = fadd double %scalar.175.364, %scalar.176.365
  store double %scalar.177.366, ptr %value.366, align 8
  %load.178.367.1 = load double, ptr %arg.29, align 8
  %scalar.178.367 = fadd double %scalar.177.366, %load.178.367.1
  store double %scalar.178.367, ptr %value.367, align 8
  %scalar.179.368 = fadd double %scalar.178.367, %scalar.170.360
  store double %scalar.179.368, ptr %value.368, align 8
  %scalar.180.369 = fadd double %scalar.172.361, %scalar.179.368
  store double %scalar.180.369, ptr %value.369, align 8
  %scalar.181.370 = fsub double %scalar.180.369, %scalar.172.361
  store double %scalar.181.370, ptr %value.370, align 8
  %scalar.182.371 = fsub double %scalar.179.368, %scalar.181.370
  store double %scalar.182.371, ptr %value.371, align 8
  %scalar.183.35 = fadd double %scalar.180.369, %scalar.182.371
  store double %scalar.183.35, ptr %out.16, align 8
  %scalar.184.372 = fmul double %load.0.204.1, %scalar.180.369
  store double %scalar.184.372, ptr %value.372, align 8
  %scalar.185.373 = fneg double %scalar.184.372
  store double %scalar.185.373, ptr %value.373, align 8
  %scalar.186.374 = call double @llvm.fma.f64(double %load.0.204.1, double %scalar.180.369, double %scalar.185.373)
  store double %scalar.186.374, ptr %value.374, align 8
  %scalar.187.375 = fmul double %load.0.204.1, %scalar.182.371
  store double %scalar.187.375, ptr %value.375, align 8
  %scalar.188.376 = fadd double %scalar.186.374, %scalar.187.375
  store double %scalar.188.376, ptr %value.376, align 8
  %scalar.189.377 = fmul double %load.3.207.1, %scalar.180.369
  store double %scalar.189.377, ptr %value.377, align 8
  %scalar.190.378 = fadd double %scalar.188.376, %scalar.189.377
  store double %scalar.190.378, ptr %value.378, align 8
  %scalar.191.379 = fadd double %scalar.184.372, %scalar.190.378
  store double %scalar.191.379, ptr %value.379, align 8
  %scalar.192.380 = fsub double %scalar.191.379, %scalar.184.372
  store double %scalar.192.380, ptr %value.380, align 8
  %scalar.193.381 = fsub double %scalar.190.378, %scalar.192.380
  store double %scalar.193.381, ptr %value.381, align 8
  %scalar.194.36 = fadd double %scalar.191.379, %scalar.193.381
  store double %scalar.194.36, ptr %out.17, align 8
  %load.195.382.0 = load double, ptr %arg.10, align 8
  %scalar.195.382 = fadd double %load.195.382.0, %scalar.191.379
  store double %scalar.195.382, ptr %value.382, align 8
  %scalar.196.383 = fsub double %scalar.195.382, %load.195.382.0
  store double %scalar.196.383, ptr %value.383, align 8
  %scalar.197.384 = fsub double %scalar.195.382, %scalar.196.383
  store double %scalar.197.384, ptr %value.384, align 8
  %scalar.198.385 = fsub double %load.195.382.0, %scalar.197.384
  store double %scalar.198.385, ptr %value.385, align 8
  %scalar.199.386 = fsub double %scalar.191.379, %scalar.196.383
  store double %scalar.199.386, ptr %value.386, align 8
  %scalar.200.387 = fadd double %scalar.198.385, %scalar.199.386
  store double %scalar.200.387, ptr %value.387, align 8
  %load.201.388.1 = load double, ptr %arg.30, align 8
  %scalar.201.388 = fadd double %scalar.200.387, %load.201.388.1
  store double %scalar.201.388, ptr %value.388, align 8
  %scalar.202.389 = fadd double %scalar.201.388, %scalar.193.381
  store double %scalar.202.389, ptr %value.389, align 8
  %scalar.203.390 = fadd double %scalar.195.382, %scalar.202.389
  store double %scalar.203.390, ptr %value.390, align 8
  %scalar.204.391 = fsub double %scalar.203.390, %scalar.195.382
  store double %scalar.204.391, ptr %value.391, align 8
  %scalar.205.392 = fsub double %scalar.202.389, %scalar.204.391
  store double %scalar.205.392, ptr %value.392, align 8
  %scalar.206.37 = fadd double %scalar.203.390, %scalar.205.392
  store double %scalar.206.37, ptr %out.18, align 8
  %scalar.207.393 = fmul double %load.0.204.1, %scalar.203.390
  store double %scalar.207.393, ptr %value.393, align 8
  %scalar.208.394 = fneg double %scalar.207.393
  store double %scalar.208.394, ptr %value.394, align 8
  %scalar.209.395 = call double @llvm.fma.f64(double %load.0.204.1, double %scalar.203.390, double %scalar.208.394)
  store double %scalar.209.395, ptr %value.395, align 8
  %scalar.210.396 = fmul double %load.0.204.1, %scalar.205.392
  store double %scalar.210.396, ptr %value.396, align 8
  %scalar.211.397 = fadd double %scalar.209.395, %scalar.210.396
  store double %scalar.211.397, ptr %value.397, align 8
  %scalar.212.398 = fmul double %load.3.207.1, %scalar.203.390
  store double %scalar.212.398, ptr %value.398, align 8
  %scalar.213.399 = fadd double %scalar.211.397, %scalar.212.398
  store double %scalar.213.399, ptr %value.399, align 8
  %scalar.214.400 = fadd double %scalar.207.393, %scalar.213.399
  store double %scalar.214.400, ptr %value.400, align 8
  %scalar.215.401 = fsub double %scalar.214.400, %scalar.207.393
  store double %scalar.215.401, ptr %value.401, align 8
  %scalar.216.402 = fsub double %scalar.213.399, %scalar.215.401
  store double %scalar.216.402, ptr %value.402, align 8
  %scalar.217.38 = fadd double %scalar.214.400, %scalar.216.402
  store double %scalar.217.38, ptr %out.19, align 8
  %load.218.403.0 = load double, ptr %arg.11, align 8
  %scalar.218.403 = fadd double %load.218.403.0, %scalar.214.400
  store double %scalar.218.403, ptr %value.403, align 8
  %scalar.219.404 = fsub double %scalar.218.403, %load.218.403.0
  store double %scalar.219.404, ptr %value.404, align 8
  %scalar.220.405 = fsub double %scalar.218.403, %scalar.219.404
  store double %scalar.220.405, ptr %value.405, align 8
  %scalar.221.406 = fsub double %load.218.403.0, %scalar.220.405
  store double %scalar.221.406, ptr %value.406, align 8
  %scalar.222.407 = fsub double %scalar.214.400, %scalar.219.404
  store double %scalar.222.407, ptr %value.407, align 8
  %scalar.223.408 = fadd double %scalar.221.406, %scalar.222.407
  store double %scalar.223.408, ptr %value.408, align 8
  %load.224.409.1 = load double, ptr %arg.31, align 8
  %scalar.224.409 = fadd double %scalar.223.408, %load.224.409.1
  store double %scalar.224.409, ptr %value.409, align 8
  %scalar.225.410 = fadd double %scalar.224.409, %scalar.216.402
  store double %scalar.225.410, ptr %value.410, align 8
  %scalar.226.411 = fadd double %scalar.218.403, %scalar.225.410
  store double %scalar.226.411, ptr %value.411, align 8
  %scalar.227.412 = fsub double %scalar.226.411, %scalar.218.403
  store double %scalar.227.412, ptr %value.412, align 8
  %scalar.228.413 = fsub double %scalar.225.410, %scalar.227.412
  store double %scalar.228.413, ptr %value.413, align 8
  %scalar.229.39 = fadd double %scalar.226.411, %scalar.228.413
  store double %scalar.229.39, ptr %out.20, align 8
  %scalar.230.414 = fmul double %load.0.204.1, %scalar.226.411
  store double %scalar.230.414, ptr %value.414, align 8
  %scalar.231.415 = fneg double %scalar.230.414
  store double %scalar.231.415, ptr %value.415, align 8
  %scalar.232.416 = call double @llvm.fma.f64(double %load.0.204.1, double %scalar.226.411, double %scalar.231.415)
  store double %scalar.232.416, ptr %value.416, align 8
  %scalar.233.417 = fmul double %load.0.204.1, %scalar.228.413
  store double %scalar.233.417, ptr %value.417, align 8
  %scalar.234.418 = fadd double %scalar.232.416, %scalar.233.417
  store double %scalar.234.418, ptr %value.418, align 8
  %scalar.235.419 = fmul double %load.3.207.1, %scalar.226.411
  store double %scalar.235.419, ptr %value.419, align 8
  %scalar.236.420 = fadd double %scalar.234.418, %scalar.235.419
  store double %scalar.236.420, ptr %value.420, align 8
  %scalar.237.421 = fadd double %scalar.230.414, %scalar.236.420
  store double %scalar.237.421, ptr %value.421, align 8
  %scalar.238.422 = fsub double %scalar.237.421, %scalar.230.414
  store double %scalar.238.422, ptr %value.422, align 8
  %scalar.239.423 = fsub double %scalar.236.420, %scalar.238.422
  store double %scalar.239.423, ptr %value.423, align 8
  %scalar.240.40 = fadd double %scalar.237.421, %scalar.239.423
  store double %scalar.240.40, ptr %out.21, align 8
  %load.241.424.0 = load double, ptr %arg.12, align 8
  %scalar.241.424 = fadd double %load.241.424.0, %scalar.237.421
  store double %scalar.241.424, ptr %value.424, align 8
  %scalar.242.425 = fsub double %scalar.241.424, %load.241.424.0
  store double %scalar.242.425, ptr %value.425, align 8
  %scalar.243.426 = fsub double %scalar.241.424, %scalar.242.425
  store double %scalar.243.426, ptr %value.426, align 8
  %scalar.244.427 = fsub double %load.241.424.0, %scalar.243.426
  store double %scalar.244.427, ptr %value.427, align 8
  %scalar.245.428 = fsub double %scalar.237.421, %scalar.242.425
  store double %scalar.245.428, ptr %value.428, align 8
  %scalar.246.429 = fadd double %scalar.244.427, %scalar.245.428
  store double %scalar.246.429, ptr %value.429, align 8
  %load.247.430.1 = load double, ptr %arg.32, align 8
  %scalar.247.430 = fadd double %scalar.246.429, %load.247.430.1
  store double %scalar.247.430, ptr %value.430, align 8
  %scalar.248.431 = fadd double %scalar.247.430, %scalar.239.423
  store double %scalar.248.431, ptr %value.431, align 8
  %scalar.249.432 = fadd double %scalar.241.424, %scalar.248.431
  store double %scalar.249.432, ptr %value.432, align 8
  %scalar.250.433 = fsub double %scalar.249.432, %scalar.241.424
  store double %scalar.250.433, ptr %value.433, align 8
  %scalar.251.434 = fsub double %scalar.248.431, %scalar.250.433
  store double %scalar.251.434, ptr %value.434, align 8
  %scalar.252.41 = fadd double %scalar.249.432, %scalar.251.434
  store double %scalar.252.41, ptr %out.22, align 8
  %scalar.253.435 = fmul double %load.0.204.1, %scalar.249.432
  store double %scalar.253.435, ptr %value.435, align 8
  %scalar.254.436 = fneg double %scalar.253.435
  store double %scalar.254.436, ptr %value.436, align 8
  %scalar.255.437 = call double @llvm.fma.f64(double %load.0.204.1, double %scalar.249.432, double %scalar.254.436)
  store double %scalar.255.437, ptr %value.437, align 8
  %scalar.256.438 = fmul double %load.0.204.1, %scalar.251.434
  store double %scalar.256.438, ptr %value.438, align 8
  %scalar.257.439 = fadd double %scalar.255.437, %scalar.256.438
  store double %scalar.257.439, ptr %value.439, align 8
  %scalar.258.440 = fmul double %load.3.207.1, %scalar.249.432
  store double %scalar.258.440, ptr %value.440, align 8
  %scalar.259.441 = fadd double %scalar.257.439, %scalar.258.440
  store double %scalar.259.441, ptr %value.441, align 8
  %scalar.260.442 = fadd double %scalar.253.435, %scalar.259.441
  store double %scalar.260.442, ptr %value.442, align 8
  %scalar.261.443 = fsub double %scalar.260.442, %scalar.253.435
  store double %scalar.261.443, ptr %value.443, align 8
  %scalar.262.444 = fsub double %scalar.259.441, %scalar.261.443
  store double %scalar.262.444, ptr %value.444, align 8
  %scalar.263.42 = fadd double %scalar.260.442, %scalar.262.444
  store double %scalar.263.42, ptr %out.23, align 8
  %load.264.445.0 = load double, ptr %arg.13, align 8
  %scalar.264.445 = fadd double %load.264.445.0, %scalar.260.442
  store double %scalar.264.445, ptr %value.445, align 8
  %scalar.265.446 = fsub double %scalar.264.445, %load.264.445.0
  store double %scalar.265.446, ptr %value.446, align 8
  %scalar.266.447 = fsub double %scalar.264.445, %scalar.265.446
  store double %scalar.266.447, ptr %value.447, align 8
  %scalar.267.448 = fsub double %load.264.445.0, %scalar.266.447
  store double %scalar.267.448, ptr %value.448, align 8
  %scalar.268.449 = fsub double %scalar.260.442, %scalar.265.446
  store double %scalar.268.449, ptr %value.449, align 8
  %scalar.269.450 = fadd double %scalar.267.448, %scalar.268.449
  store double %scalar.269.450, ptr %value.450, align 8
  %load.270.451.1 = load double, ptr %arg.33, align 8
  %scalar.270.451 = fadd double %scalar.269.450, %load.270.451.1
  store double %scalar.270.451, ptr %value.451, align 8
  %scalar.271.452 = fadd double %scalar.270.451, %scalar.262.444
  store double %scalar.271.452, ptr %value.452, align 8
  %scalar.272.453 = fadd double %scalar.264.445, %scalar.271.452
  store double %scalar.272.453, ptr %value.453, align 8
  %scalar.273.454 = fsub double %scalar.272.453, %scalar.264.445
  store double %scalar.273.454, ptr %value.454, align 8
  %scalar.274.455 = fsub double %scalar.271.452, %scalar.273.454
  store double %scalar.274.455, ptr %value.455, align 8
  %scalar.275.43 = fadd double %scalar.272.453, %scalar.274.455
  store double %scalar.275.43, ptr %out.24, align 8
  %scalar.276.456 = fmul double %load.0.204.1, %scalar.272.453
  store double %scalar.276.456, ptr %value.456, align 8
  %scalar.277.457 = fneg double %scalar.276.456
  store double %scalar.277.457, ptr %value.457, align 8
  %scalar.278.458 = call double @llvm.fma.f64(double %load.0.204.1, double %scalar.272.453, double %scalar.277.457)
  store double %scalar.278.458, ptr %value.458, align 8
  %scalar.279.459 = fmul double %load.0.204.1, %scalar.274.455
  store double %scalar.279.459, ptr %value.459, align 8
  %scalar.280.460 = fadd double %scalar.278.458, %scalar.279.459
  store double %scalar.280.460, ptr %value.460, align 8
  %scalar.281.461 = fmul double %load.3.207.1, %scalar.272.453
  store double %scalar.281.461, ptr %value.461, align 8
  %scalar.282.462 = fadd double %scalar.280.460, %scalar.281.461
  store double %scalar.282.462, ptr %value.462, align 8
  %scalar.283.463 = fadd double %scalar.276.456, %scalar.282.462
  store double %scalar.283.463, ptr %value.463, align 8
  %scalar.284.464 = fsub double %scalar.283.463, %scalar.276.456
  store double %scalar.284.464, ptr %value.464, align 8
  %scalar.285.465 = fsub double %scalar.282.462, %scalar.284.464
  store double %scalar.285.465, ptr %value.465, align 8
  %scalar.286.44 = fadd double %scalar.283.463, %scalar.285.465
  store double %scalar.286.44, ptr %out.25, align 8
  %load.287.466.0 = load double, ptr %arg.14, align 8
  %scalar.287.466 = fadd double %load.287.466.0, %scalar.283.463
  store double %scalar.287.466, ptr %value.466, align 8
  %scalar.288.467 = fsub double %scalar.287.466, %load.287.466.0
  store double %scalar.288.467, ptr %value.467, align 8
  %scalar.289.468 = fsub double %scalar.287.466, %scalar.288.467
  store double %scalar.289.468, ptr %value.468, align 8
  %scalar.290.469 = fsub double %load.287.466.0, %scalar.289.468
  store double %scalar.290.469, ptr %value.469, align 8
  %scalar.291.470 = fsub double %scalar.283.463, %scalar.288.467
  store double %scalar.291.470, ptr %value.470, align 8
  %scalar.292.471 = fadd double %scalar.290.469, %scalar.291.470
  store double %scalar.292.471, ptr %value.471, align 8
  %load.293.472.1 = load double, ptr %arg.34, align 8
  %scalar.293.472 = fadd double %scalar.292.471, %load.293.472.1
  store double %scalar.293.472, ptr %value.472, align 8
  %scalar.294.473 = fadd double %scalar.293.472, %scalar.285.465
  store double %scalar.294.473, ptr %value.473, align 8
  %scalar.295.474 = fadd double %scalar.287.466, %scalar.294.473
  store double %scalar.295.474, ptr %value.474, align 8
  %scalar.296.475 = fsub double %scalar.295.474, %scalar.287.466
  store double %scalar.296.475, ptr %value.475, align 8
  %scalar.297.476 = fsub double %scalar.294.473, %scalar.296.475
  store double %scalar.297.476, ptr %value.476, align 8
  %scalar.298.45 = fadd double %scalar.295.474, %scalar.297.476
  store double %scalar.298.45, ptr %out.26, align 8
  %scalar.299.477 = fmul double %load.0.204.1, %scalar.295.474
  store double %scalar.299.477, ptr %value.477, align 8
  %scalar.300.478 = fneg double %scalar.299.477
  store double %scalar.300.478, ptr %value.478, align 8
  %scalar.301.479 = call double @llvm.fma.f64(double %load.0.204.1, double %scalar.295.474, double %scalar.300.478)
  store double %scalar.301.479, ptr %value.479, align 8
  %scalar.302.480 = fmul double %load.0.204.1, %scalar.297.476
  store double %scalar.302.480, ptr %value.480, align 8
  %scalar.303.481 = fadd double %scalar.301.479, %scalar.302.480
  store double %scalar.303.481, ptr %value.481, align 8
  %scalar.304.482 = fmul double %load.3.207.1, %scalar.295.474
  store double %scalar.304.482, ptr %value.482, align 8
  %scalar.305.483 = fadd double %scalar.303.481, %scalar.304.482
  store double %scalar.305.483, ptr %value.483, align 8
  %scalar.306.484 = fadd double %scalar.299.477, %scalar.305.483
  store double %scalar.306.484, ptr %value.484, align 8
  %scalar.307.485 = fsub double %scalar.306.484, %scalar.299.477
  store double %scalar.307.485, ptr %value.485, align 8
  %scalar.308.486 = fsub double %scalar.305.483, %scalar.307.485
  store double %scalar.308.486, ptr %value.486, align 8
  %scalar.309.46 = fadd double %scalar.306.484, %scalar.308.486
  store double %scalar.309.46, ptr %out.27, align 8
  %load.310.487.0 = load double, ptr %arg.15, align 8
  %scalar.310.487 = fadd double %load.310.487.0, %scalar.306.484
  store double %scalar.310.487, ptr %value.487, align 8
  %scalar.311.488 = fsub double %scalar.310.487, %load.310.487.0
  store double %scalar.311.488, ptr %value.488, align 8
  %scalar.312.489 = fsub double %scalar.310.487, %scalar.311.488
  store double %scalar.312.489, ptr %value.489, align 8
  %scalar.313.490 = fsub double %load.310.487.0, %scalar.312.489
  store double %scalar.313.490, ptr %value.490, align 8
  %scalar.314.491 = fsub double %scalar.306.484, %scalar.311.488
  store double %scalar.314.491, ptr %value.491, align 8
  %scalar.315.492 = fadd double %scalar.313.490, %scalar.314.491
  store double %scalar.315.492, ptr %value.492, align 8
  %load.316.493.1 = load double, ptr %arg.35, align 8
  %scalar.316.493 = fadd double %scalar.315.492, %load.316.493.1
  store double %scalar.316.493, ptr %value.493, align 8
  %scalar.317.494 = fadd double %scalar.316.493, %scalar.308.486
  store double %scalar.317.494, ptr %value.494, align 8
  %scalar.318.495 = fadd double %scalar.310.487, %scalar.317.494
  store double %scalar.318.495, ptr %value.495, align 8
  %scalar.319.496 = fsub double %scalar.318.495, %scalar.310.487
  store double %scalar.319.496, ptr %value.496, align 8
  %scalar.320.497 = fsub double %scalar.317.494, %scalar.319.496
  store double %scalar.320.497, ptr %value.497, align 8
  %scalar.321.47 = fadd double %scalar.318.495, %scalar.320.497
  store double %scalar.321.47, ptr %out.28, align 8
  %scalar.322.498 = fmul double %load.0.204.1, %scalar.318.495
  store double %scalar.322.498, ptr %value.498, align 8
  %scalar.323.499 = fneg double %scalar.322.498
  store double %scalar.323.499, ptr %value.499, align 8
  %scalar.324.500 = call double @llvm.fma.f64(double %load.0.204.1, double %scalar.318.495, double %scalar.323.499)
  store double %scalar.324.500, ptr %value.500, align 8
  %scalar.325.501 = fmul double %load.0.204.1, %scalar.320.497
  store double %scalar.325.501, ptr %value.501, align 8
  %scalar.326.502 = fadd double %scalar.324.500, %scalar.325.501
  store double %scalar.326.502, ptr %value.502, align 8
  %scalar.327.503 = fmul double %load.3.207.1, %scalar.318.495
  store double %scalar.327.503, ptr %value.503, align 8
  %scalar.328.504 = fadd double %scalar.326.502, %scalar.327.503
  store double %scalar.328.504, ptr %value.504, align 8
  %scalar.329.505 = fadd double %scalar.322.498, %scalar.328.504
  store double %scalar.329.505, ptr %value.505, align 8
  %scalar.330.506 = fsub double %scalar.329.505, %scalar.322.498
  store double %scalar.330.506, ptr %value.506, align 8
  %scalar.331.507 = fsub double %scalar.328.504, %scalar.330.506
  store double %scalar.331.507, ptr %value.507, align 8
  %scalar.332.48 = fadd double %scalar.329.505, %scalar.331.507
  store double %scalar.332.48, ptr %out.29, align 8
  %load.333.508.0 = load double, ptr %arg.16, align 8
  %scalar.333.508 = fadd double %load.333.508.0, %scalar.329.505
  store double %scalar.333.508, ptr %value.508, align 8
  %scalar.334.509 = fsub double %scalar.333.508, %load.333.508.0
  store double %scalar.334.509, ptr %value.509, align 8
  %scalar.335.510 = fsub double %scalar.333.508, %scalar.334.509
  store double %scalar.335.510, ptr %value.510, align 8
  %scalar.336.511 = fsub double %load.333.508.0, %scalar.335.510
  store double %scalar.336.511, ptr %value.511, align 8
  %scalar.337.512 = fsub double %scalar.329.505, %scalar.334.509
  store double %scalar.337.512, ptr %value.512, align 8
  %scalar.338.513 = fadd double %scalar.336.511, %scalar.337.512
  store double %scalar.338.513, ptr %value.513, align 8
  %load.339.514.1 = load double, ptr %arg.36, align 8
  %scalar.339.514 = fadd double %scalar.338.513, %load.339.514.1
  store double %scalar.339.514, ptr %value.514, align 8
  %scalar.340.515 = fadd double %scalar.339.514, %scalar.331.507
  store double %scalar.340.515, ptr %value.515, align 8
  %scalar.341.516 = fadd double %scalar.333.508, %scalar.340.515
  store double %scalar.341.516, ptr %value.516, align 8
  %scalar.342.517 = fsub double %scalar.341.516, %scalar.333.508
  store double %scalar.342.517, ptr %value.517, align 8
  %scalar.343.518 = fsub double %scalar.340.515, %scalar.342.517
  store double %scalar.343.518, ptr %value.518, align 8
  %scalar.344.49 = fadd double %scalar.341.516, %scalar.343.518
  store double %scalar.344.49, ptr %out.30, align 8
  %scalar.345.519 = fmul double %load.0.204.1, %scalar.341.516
  store double %scalar.345.519, ptr %value.519, align 8
  %scalar.346.520 = fneg double %scalar.345.519
  store double %scalar.346.520, ptr %value.520, align 8
  %scalar.347.521 = call double @llvm.fma.f64(double %load.0.204.1, double %scalar.341.516, double %scalar.346.520)
  store double %scalar.347.521, ptr %value.521, align 8
  %scalar.348.522 = fmul double %load.0.204.1, %scalar.343.518
  store double %scalar.348.522, ptr %value.522, align 8
  %scalar.349.523 = fadd double %scalar.347.521, %scalar.348.522
  store double %scalar.349.523, ptr %value.523, align 8
  %scalar.350.524 = fmul double %load.3.207.1, %scalar.341.516
  store double %scalar.350.524, ptr %value.524, align 8
  %scalar.351.525 = fadd double %scalar.349.523, %scalar.350.524
  store double %scalar.351.525, ptr %value.525, align 8
  %scalar.352.526 = fadd double %scalar.345.519, %scalar.351.525
  store double %scalar.352.526, ptr %value.526, align 8
  %scalar.353.527 = fsub double %scalar.352.526, %scalar.345.519
  store double %scalar.353.527, ptr %value.527, align 8
  %scalar.354.528 = fsub double %scalar.351.525, %scalar.353.527
  store double %scalar.354.528, ptr %value.528, align 8
  %scalar.355.50 = fadd double %scalar.352.526, %scalar.354.528
  store double %scalar.355.50, ptr %out.31, align 8
  %load.356.529.0 = load double, ptr %arg.17, align 8
  %scalar.356.529 = fadd double %load.356.529.0, %scalar.352.526
  store double %scalar.356.529, ptr %value.529, align 8
  %scalar.357.530 = fsub double %scalar.356.529, %load.356.529.0
  store double %scalar.357.530, ptr %value.530, align 8
  %scalar.358.531 = fsub double %scalar.356.529, %scalar.357.530
  store double %scalar.358.531, ptr %value.531, align 8
  %scalar.359.532 = fsub double %load.356.529.0, %scalar.358.531
  store double %scalar.359.532, ptr %value.532, align 8
  %scalar.360.533 = fsub double %scalar.352.526, %scalar.357.530
  store double %scalar.360.533, ptr %value.533, align 8
  %scalar.361.534 = fadd double %scalar.359.532, %scalar.360.533
  store double %scalar.361.534, ptr %value.534, align 8
  %load.362.535.1 = load double, ptr %arg.37, align 8
  %scalar.362.535 = fadd double %scalar.361.534, %load.362.535.1
  store double %scalar.362.535, ptr %value.535, align 8
  %scalar.363.536 = fadd double %scalar.362.535, %scalar.354.528
  store double %scalar.363.536, ptr %value.536, align 8
  %scalar.364.537 = fadd double %scalar.356.529, %scalar.363.536
  store double %scalar.364.537, ptr %value.537, align 8
  %scalar.365.538 = fsub double %scalar.364.537, %scalar.356.529
  store double %scalar.365.538, ptr %value.538, align 8
  %scalar.366.539 = fsub double %scalar.363.536, %scalar.365.538
  store double %scalar.366.539, ptr %value.539, align 8
  %scalar.367.51 = fadd double %scalar.364.537, %scalar.366.539
  store double %scalar.367.51, ptr %out.32, align 8
  %scalar.368.540 = fmul double %load.0.204.1, %scalar.364.537
  store double %scalar.368.540, ptr %value.540, align 8
  %scalar.369.541 = fneg double %scalar.368.540
  store double %scalar.369.541, ptr %value.541, align 8
  %scalar.370.542 = call double @llvm.fma.f64(double %load.0.204.1, double %scalar.364.537, double %scalar.369.541)
  store double %scalar.370.542, ptr %value.542, align 8
  %scalar.371.543 = fmul double %load.0.204.1, %scalar.366.539
  store double %scalar.371.543, ptr %value.543, align 8
  %scalar.372.544 = fadd double %scalar.370.542, %scalar.371.543
  store double %scalar.372.544, ptr %value.544, align 8
  %scalar.373.545 = fmul double %load.3.207.1, %scalar.364.537
  store double %scalar.373.545, ptr %value.545, align 8
  %scalar.374.546 = fadd double %scalar.372.544, %scalar.373.545
  store double %scalar.374.546, ptr %value.546, align 8
  %scalar.375.547 = fadd double %scalar.368.540, %scalar.374.546
  store double %scalar.375.547, ptr %value.547, align 8
  %scalar.376.548 = fsub double %scalar.375.547, %scalar.368.540
  store double %scalar.376.548, ptr %value.548, align 8
  %scalar.377.549 = fsub double %scalar.374.546, %scalar.376.548
  store double %scalar.377.549, ptr %value.549, align 8
  %scalar.378.52 = fadd double %scalar.375.547, %scalar.377.549
  store double %scalar.378.52, ptr %out.33, align 8
  %load.379.550.0 = load double, ptr %arg.18, align 8
  %scalar.379.550 = fadd double %load.379.550.0, %scalar.375.547
  store double %scalar.379.550, ptr %value.550, align 8
  %scalar.380.551 = fsub double %scalar.379.550, %load.379.550.0
  store double %scalar.380.551, ptr %value.551, align 8
  %scalar.381.552 = fsub double %scalar.379.550, %scalar.380.551
  store double %scalar.381.552, ptr %value.552, align 8
  %scalar.382.553 = fsub double %load.379.550.0, %scalar.381.552
  store double %scalar.382.553, ptr %value.553, align 8
  %scalar.383.554 = fsub double %scalar.375.547, %scalar.380.551
  store double %scalar.383.554, ptr %value.554, align 8
  %scalar.384.555 = fadd double %scalar.382.553, %scalar.383.554
  store double %scalar.384.555, ptr %value.555, align 8
  %load.385.556.1 = load double, ptr %arg.38, align 8
  %scalar.385.556 = fadd double %scalar.384.555, %load.385.556.1
  store double %scalar.385.556, ptr %value.556, align 8
  %scalar.386.557 = fadd double %scalar.385.556, %scalar.377.549
  store double %scalar.386.557, ptr %value.557, align 8
  %scalar.387.558 = fadd double %scalar.379.550, %scalar.386.557
  store double %scalar.387.558, ptr %value.558, align 8
  %scalar.388.559 = fsub double %scalar.387.558, %scalar.379.550
  store double %scalar.388.559, ptr %value.559, align 8
  %scalar.389.560 = fsub double %scalar.386.557, %scalar.388.559
  store double %scalar.389.560, ptr %value.560, align 8
  %scalar.390.53 = fadd double %scalar.387.558, %scalar.389.560
  store double %scalar.390.53, ptr %out.34, align 8
  %load.391.561.0 = load double, ptr %arg.19, align 8
  %scalar.391.561 = fmul double %load.391.561.0, %scalar.387.558
  store double %scalar.391.561, ptr %value.561, align 8
  %scalar.392.562 = fneg double %scalar.391.561
  store double %scalar.392.562, ptr %value.562, align 8
  %scalar.393.563 = call double @llvm.fma.f64(double %load.391.561.0, double %scalar.387.558, double %scalar.392.562)
  store double %scalar.393.563, ptr %value.563, align 8
  %scalar.394.564 = fmul double %load.391.561.0, %scalar.389.560
  store double %scalar.394.564, ptr %value.564, align 8
  %scalar.395.565 = fadd double %scalar.393.563, %scalar.394.564
  store double %scalar.395.565, ptr %value.565, align 8
  %load.396.566.0 = load double, ptr %arg.39, align 8
  %scalar.396.566 = fmul double %load.396.566.0, %scalar.387.558
  store double %scalar.396.566, ptr %value.566, align 8
  %scalar.397.567 = fadd double %scalar.395.565, %scalar.396.566
  store double %scalar.397.567, ptr %value.567, align 8
  %scalar.398.568 = fadd double %scalar.391.561, %scalar.397.567
  store double %scalar.398.568, ptr %value.568, align 8
  %scalar.399.569 = fsub double %scalar.398.568, %scalar.391.561
  store double %scalar.399.569, ptr %value.569, align 8
  %scalar.400.570 = fsub double %scalar.397.567, %scalar.399.569
  store double %scalar.400.570, ptr %value.570, align 8
  %scalar.401.54 = fadd double %scalar.398.568, %scalar.400.570
  store double %scalar.401.54, ptr %out.0, align 8
  ret void
}

define void @__ssa_tan_core_pack__tan_core(ptr noalias %arg.0, ptr noalias %arg.1, ptr noalias %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr noalias %arg.11, ptr noalias %arg.12, ptr noalias %arg.13, ptr noalias %arg.14, ptr noalias %arg.15, ptr noalias %arg.16, ptr noalias %arg.17, ptr %arg.18, ptr %arg.19, ptr noalias %arg.20, ptr noalias %arg.21, ptr noalias %arg.22, ptr noalias %arg.23, ptr noalias %arg.24, ptr noalias %arg.25, ptr noalias %arg.26, ptr noalias %arg.27, ptr noalias %arg.28, ptr noalias %arg.29, ptr noalias %arg.30, ptr noalias %arg.31, ptr noalias %arg.32, ptr noalias %arg.33, ptr noalias %arg.34, ptr noalias %arg.35, ptr noalias %arg.36, ptr noalias %arg.37, ptr %arg.38, ptr %arg.39, ptr %out.0) {
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
  call void @__ssa_tan_core_pack__tan_core__planned_region_0(ptr %arg.9, ptr %arg.18, ptr %arg.8, ptr %arg.7, ptr %arg.6, ptr %arg.5, ptr %arg.4, ptr %arg.3, ptr %arg.2, ptr %arg.17, ptr %arg.16, ptr %arg.15, ptr %arg.14, ptr %arg.13, ptr %arg.12, ptr %arg.11, ptr %arg.10, ptr %arg.1, ptr %arg.0, ptr %arg.19, ptr %arg.29, ptr %arg.38, ptr %arg.28, ptr %arg.27, ptr %arg.26, ptr %arg.25, ptr %arg.24, ptr %arg.23, ptr %arg.22, ptr %arg.37, ptr %arg.36, ptr %arg.35, ptr %arg.34, ptr %arg.33, ptr %arg.32, ptr %arg.31, ptr %arg.30, ptr %arg.21, ptr %arg.20, ptr %arg.39, ptr %out.0, ptr %value.20, ptr %value.21, ptr %value.22, ptr %value.23, ptr %value.24, ptr %value.25, ptr %value.26, ptr %value.27, ptr %value.28, ptr %value.29, ptr %value.30, ptr %value.31, ptr %value.32, ptr %value.33, ptr %value.34, ptr %value.35, ptr %value.36, ptr %value.37, ptr %value.38, ptr %value.39, ptr %value.40, ptr %value.41, ptr %value.42, ptr %value.43, ptr %value.44, ptr %value.45, ptr %value.46, ptr %value.47, ptr %value.48, ptr %value.49, ptr %value.50, ptr %value.51, ptr %value.52, ptr %value.53)
  ret void
}

define void @tan_core_pack__tan_core_pack(ptr %buffers, ptr %extents) {
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
  call void @__ssa_tan_core_pack__tan_core_pack(ptr %public.0, ptr %public.1, ptr %public.2, ptr %public.3, ptr %public.4, ptr %public.5, ptr %public.6, ptr %public.7, ptr %public.8, ptr %public.9, ptr %public.10, ptr %public.11, ptr %public.12, ptr %public.13, ptr %public.14, ptr %public.15, ptr %public.16, ptr %public.17, ptr %public.18, ptr %public.19, ptr %public.20, ptr %public.21, ptr %public.22, ptr %public.23, ptr %public.24, ptr %public.25, ptr %public.26, ptr %public.27, ptr %public.28, ptr %public.29, ptr %public.30, ptr %public.31, ptr %public.32, ptr %public.33, ptr %public.34, ptr %public.35, ptr %public.36, ptr %public.37, ptr %public.38, ptr %public.2)
  ret void
}
