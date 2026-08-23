source_filename = "turing.ssa-llvm.atan_core_pack__atan_core_pack"

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-windows-gnu"

declare double @llvm.fma.f64(double, double, double)

define void @__ssa_atan_core_pack__atan_core_pack__planned_region_0(ptr noalias %arg.0, ptr %arg.1, ptr %out.0, ptr %out.1) {
entry:
  %value.214 = alloca i32, i64 1, align 8
  %value.215 = alloca i32, i64 1, align 8
  %value.217 = alloca double, i64 1, align 8
  %value.218 = alloca i32, i64 1, align 8
  %value.219 = alloca i32, i64 1, align 8
  %value.220 = alloca i32, i64 1, align 8
  %value.221 = alloca i32, i64 1, align 8
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
  %load.0.41.0 = load i32, ptr %arg.1, align 4
  %address.0.41 = getelementptr double, ptr %arg.0, i32 %load.0.41.0
  store i32 2, ptr %value.214, align 4
  %scalar.2.215 = mul i32 %load.0.41.0, 2
  store i32 %scalar.2.215, ptr %value.215, align 4
  %address.3.216 = getelementptr double, ptr %arg.0, i32 %scalar.2.215
  %pinned.load.4.217 = load double, ptr %address.3.216, align 8
  store double %pinned.load.4.217, ptr %value.217, align 8
  store i32 2, ptr %value.218, align 4
  %scalar.6.219 = mul i32 %load.0.41.0, 2
  store i32 %scalar.6.219, ptr %value.219, align 4
  store i32 1, ptr %value.220, align 4
  %scalar.8.221 = add i32 %scalar.6.219, 1
  store i32 %scalar.8.221, ptr %value.221, align 4
  %address.9.222 = getelementptr double, ptr %arg.0, i32 %scalar.8.221
  %pinned.load.10.223 = load double, ptr %address.9.222, align 8
  store double %pinned.load.10.223, ptr %value.223, align 8
  %load.11.224.0 = load double, ptr %value.217, align 8
  %scalar.11.224 = fmul double %load.11.224.0, %load.11.224.0
  store double %scalar.11.224, ptr %value.224, align 8
  %scalar.12.225 = fneg double %scalar.11.224
  store double %scalar.12.225, ptr %value.225, align 8
  %scalar.13.226 = call double @llvm.fma.f64(double %load.11.224.0, double %load.11.224.0, double %scalar.12.225)
  store double %scalar.13.226, ptr %value.226, align 8
  %load.14.227.1 = load double, ptr %value.223, align 8
  %scalar.14.227 = fmul double %load.11.224.0, %load.14.227.1
  store double %scalar.14.227, ptr %value.227, align 8
  %scalar.15.228 = fadd double %scalar.13.226, %scalar.14.227
  store double %scalar.15.228, ptr %value.228, align 8
  %scalar.16.229 = fmul double %load.14.227.1, %load.11.224.0
  store double %scalar.16.229, ptr %value.229, align 8
  %scalar.17.230 = fadd double %scalar.15.228, %scalar.16.229
  store double %scalar.17.230, ptr %value.230, align 8
  %scalar.18.231 = fadd double %scalar.11.224, %scalar.17.230
  store double %scalar.18.231, ptr %value.231, align 8
  %scalar.19.232 = fsub double %scalar.18.231, %scalar.11.224
  store double %scalar.19.232, ptr %value.232, align 8
  %scalar.20.233 = fsub double %scalar.17.230, %scalar.19.232
  store double %scalar.20.233, ptr %value.233, align 8
  %scalar.21.27 = fadd double %scalar.18.231, %scalar.20.233
  store double %scalar.21.27, ptr %out.1, align 8
  ret void
}

define void @__ssa_atan_core_pack__atan_core_pack__planned_region_1(ptr %arg.0, ptr %arg.1, ptr %arg.2) {
entry:
  %load.0.42.0 = load i32, ptr %arg.1, align 4
  %address.0.42 = getelementptr double, ptr %arg.0, i32 %load.0.42.0
  %load.store.1.v = load double, ptr %arg.2, align 8
  store double %load.store.1.v, ptr %address.0.42, align 8
  ret void
}

define void @__ssa_atan_core_pack__atan_core_pack(ptr noalias %arg.0, ptr noalias %arg.1, ptr %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr noalias %arg.11, ptr noalias %arg.12, ptr noalias %arg.13, ptr noalias %arg.14, ptr noalias %arg.15, ptr noalias %arg.16, ptr noalias %arg.17, ptr noalias %arg.18, ptr noalias %arg.19, ptr noalias %arg.20, ptr noalias %arg.21, ptr noalias %arg.22, ptr noalias %arg.23, ptr noalias %arg.24, ptr noalias %arg.25, ptr noalias %arg.26, ptr noalias %arg.27, ptr noalias %arg.28, ptr noalias %arg.29, ptr noalias %arg.30, ptr noalias %arg.31, ptr noalias %arg.32, ptr noalias %arg.33, ptr noalias %arg.34, ptr noalias %arg.35, ptr noalias %arg.36, ptr noalias %arg.37, ptr noalias %arg.38, ptr noalias %arg.39, ptr noalias %arg.40, ptr noalias %arg.41, ptr noalias %arg.42, ptr noalias %arg.43, ptr noalias %arg.44, ptr %out.0) {
entry:
  %value.31 = alloca i64, i64 1, align 8
  %value.32 = alloca i64, i64 1, align 8
  %value.37 = alloca i64, i64 1, align 8
  %value.39 = alloca i64, i64 1, align 8
  %value.34 = alloca i64, i64 1, align 8
  %value.35 = alloca i1, i64 1, align 8
  %value.26 = alloca double, i64 1, align 8
  %value.27 = alloca double, i64 1, align 8
  %value.28 = alloca double, i64 1, align 8
  store i64 0, ptr %value.31, align 8
  store i64 1, ptr %value.32, align 8
  store i64 0, ptr %value.37, align 8
  store i64 1, ptr %value.39, align 8
  br label %loop_header
loop_header:
  %phi.33 = phi ptr [ %value.31, %entry ], [ %value.34, %loop_latch ]
  %load.6.35.0 = load i32, ptr %phi.33, align 4
  %load.6.35.1 = load i32, ptr %arg.0, align 4
  %scalar.6.35 = icmp slt i32 %load.6.35.0, %load.6.35.1
  store i1 %scalar.6.35, ptr %value.35, align 1
  br i1 %scalar.6.35, label %loop_body, label %loop_exit
loop_body:
  call void @__ssa_atan_core_pack__atan_core_pack__planned_region_0(ptr %arg.1, ptr %phi.33, ptr %value.26, ptr %value.27)
  call void @__ssa_atan_core_pack__atan_core(ptr %arg.3, ptr %arg.4, ptr %arg.5, ptr %arg.6, ptr %arg.7, ptr %arg.8, ptr %arg.9, ptr %arg.10, ptr %arg.11, ptr %arg.12, ptr %arg.13, ptr %arg.14, ptr %arg.15, ptr %arg.16, ptr %arg.17, ptr %arg.18, ptr %arg.19, ptr %arg.20, ptr %arg.21, ptr %arg.22, ptr %arg.23, ptr %value.27, ptr %value.26, ptr %arg.24, ptr %arg.25, ptr %arg.26, ptr %arg.27, ptr %arg.28, ptr %arg.29, ptr %arg.30, ptr %arg.31, ptr %arg.32, ptr %arg.33, ptr %arg.34, ptr %arg.35, ptr %arg.36, ptr %arg.37, ptr %arg.38, ptr %arg.39, ptr %arg.40, ptr %arg.41, ptr %arg.42, ptr %arg.43, ptr %arg.44, ptr %value.27, ptr %value.26, ptr %value.28)
  call void @__ssa_atan_core_pack__atan_core_pack__planned_region_1(ptr %arg.2, ptr %phi.33, ptr %value.28)
  br label %loop_latch
loop_latch:
  %load.16.34.0 = load i32, ptr %phi.33, align 4
  %load.16.34.1 = load i64, ptr %value.32, align 8
  %convert.16.34.1 = trunc i64 %load.16.34.1 to i32
  %scalar.16.34 = add i32 %load.16.34.0, %convert.16.34.1
  %declared.16.34 = sext i32 %scalar.16.34 to i64
  store i64 %declared.16.34, ptr %value.34, align 8
  br label %loop_header
loop_exit:
  %return.load.0.27 = load double, ptr %arg.2, align 8
  store double %return.load.0.27, ptr %out.0, align 8
  ret void
}

define void @__ssa_atan_core_pack__atan_core__planned_region_0(ptr noalias %arg.0, ptr %arg.1, ptr noalias %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr noalias %arg.11, ptr noalias %arg.12, ptr noalias %arg.13, ptr noalias %arg.14, ptr noalias %arg.15, ptr noalias %arg.16, ptr noalias %arg.17, ptr noalias %arg.18, ptr noalias %arg.19, ptr noalias %arg.20, ptr noalias %arg.21, ptr %arg.22, ptr noalias %arg.23, ptr %arg.24, ptr noalias %arg.25, ptr noalias %arg.26, ptr noalias %arg.27, ptr noalias %arg.28, ptr noalias %arg.29, ptr noalias %arg.30, ptr noalias %arg.31, ptr noalias %arg.32, ptr noalias %arg.33, ptr noalias %arg.34, ptr noalias %arg.35, ptr noalias %arg.36, ptr noalias %arg.37, ptr noalias %arg.38, ptr noalias %arg.39, ptr noalias %arg.40, ptr noalias %arg.41, ptr noalias %arg.42, ptr noalias %arg.43, ptr noalias %arg.44, ptr %arg.45, ptr %out.0, ptr %out.1, ptr %out.2, ptr %out.3, ptr %out.4, ptr %out.5, ptr %out.6, ptr %out.7, ptr %out.8, ptr %out.9, ptr %out.10, ptr %out.11, ptr %out.12, ptr %out.13, ptr %out.14, ptr %out.15, ptr %out.16, ptr %out.17, ptr %out.18, ptr %out.19, ptr %out.20, ptr %out.21, ptr %out.22, ptr %out.23, ptr %out.24, ptr %out.25, ptr %out.26, ptr %out.27, ptr %out.28, ptr %out.29, ptr %out.30, ptr %out.31, ptr %out.32, ptr %out.33, ptr %out.34, ptr %out.35, ptr %out.36, ptr %out.37, ptr %out.38, ptr %out.39, ptr %out.40) {
entry:
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
  %value.571 = alloca double, i64 1, align 8
  %value.572 = alloca double, i64 1, align 8
  %value.573 = alloca double, i64 1, align 8
  %value.574 = alloca double, i64 1, align 8
  %value.575 = alloca double, i64 1, align 8
  %value.576 = alloca double, i64 1, align 8
  %value.577 = alloca double, i64 1, align 8
  %value.578 = alloca double, i64 1, align 8
  %value.579 = alloca double, i64 1, align 8
  %value.580 = alloca double, i64 1, align 8
  %value.581 = alloca double, i64 1, align 8
  %value.582 = alloca double, i64 1, align 8
  %value.583 = alloca double, i64 1, align 8
  %value.584 = alloca double, i64 1, align 8
  %value.585 = alloca double, i64 1, align 8
  %value.586 = alloca double, i64 1, align 8
  %value.587 = alloca double, i64 1, align 8
  %value.588 = alloca double, i64 1, align 8
  %value.589 = alloca double, i64 1, align 8
  %value.590 = alloca double, i64 1, align 8
  %value.591 = alloca double, i64 1, align 8
  %value.592 = alloca double, i64 1, align 8
  %value.593 = alloca double, i64 1, align 8
  %value.594 = alloca double, i64 1, align 8
  %value.595 = alloca double, i64 1, align 8
  %value.596 = alloca double, i64 1, align 8
  %value.597 = alloca double, i64 1, align 8
  %value.598 = alloca double, i64 1, align 8
  %value.599 = alloca double, i64 1, align 8
  %value.600 = alloca double, i64 1, align 8
  %value.601 = alloca double, i64 1, align 8
  %value.602 = alloca double, i64 1, align 8
  %value.603 = alloca double, i64 1, align 8
  %value.604 = alloca double, i64 1, align 8
  %value.605 = alloca double, i64 1, align 8
  %value.606 = alloca double, i64 1, align 8
  %value.607 = alloca double, i64 1, align 8
  %value.608 = alloca double, i64 1, align 8
  %value.609 = alloca double, i64 1, align 8
  %value.610 = alloca double, i64 1, align 8
  %value.611 = alloca double, i64 1, align 8
  %value.612 = alloca double, i64 1, align 8
  %value.613 = alloca double, i64 1, align 8
  %value.614 = alloca double, i64 1, align 8
  %value.615 = alloca double, i64 1, align 8
  %value.616 = alloca double, i64 1, align 8
  %value.617 = alloca double, i64 1, align 8
  %value.618 = alloca double, i64 1, align 8
  %value.619 = alloca double, i64 1, align 8
  %value.620 = alloca double, i64 1, align 8
  %value.621 = alloca double, i64 1, align 8
  %value.622 = alloca double, i64 1, align 8
  %value.623 = alloca double, i64 1, align 8
  %value.624 = alloca double, i64 1, align 8
  %value.625 = alloca double, i64 1, align 8
  %value.626 = alloca double, i64 1, align 8
  %value.627 = alloca double, i64 1, align 8
  %value.628 = alloca double, i64 1, align 8
  %value.629 = alloca double, i64 1, align 8
  %value.630 = alloca double, i64 1, align 8
  %value.631 = alloca double, i64 1, align 8
  %value.632 = alloca double, i64 1, align 8
  %value.633 = alloca double, i64 1, align 8
  %value.634 = alloca double, i64 1, align 8
  %value.635 = alloca double, i64 1, align 8
  %value.636 = alloca double, i64 1, align 8
  %value.637 = alloca double, i64 1, align 8
  %value.638 = alloca double, i64 1, align 8
  %value.639 = alloca double, i64 1, align 8
  %value.640 = alloca double, i64 1, align 8
  %value.641 = alloca double, i64 1, align 8
  %value.642 = alloca double, i64 1, align 8
  %value.643 = alloca double, i64 1, align 8
  %value.644 = alloca double, i64 1, align 8
  %value.645 = alloca double, i64 1, align 8
  %value.646 = alloca double, i64 1, align 8
  %value.647 = alloca double, i64 1, align 8
  %value.648 = alloca double, i64 1, align 8
  %value.649 = alloca double, i64 1, align 8
  %value.650 = alloca double, i64 1, align 8
  %value.651 = alloca double, i64 1, align 8
  %value.652 = alloca double, i64 1, align 8
  %value.653 = alloca double, i64 1, align 8
  %value.654 = alloca double, i64 1, align 8
  %value.655 = alloca double, i64 1, align 8
  %value.656 = alloca double, i64 1, align 8
  %value.657 = alloca double, i64 1, align 8
  %value.658 = alloca double, i64 1, align 8
  %value.659 = alloca double, i64 1, align 8
  %value.660 = alloca double, i64 1, align 8
  %value.661 = alloca double, i64 1, align 8
  %value.662 = alloca double, i64 1, align 8
  %value.663 = alloca double, i64 1, align 8
  %load.0.234.0 = load double, ptr %arg.0, align 8
  %load.0.234.1 = load double, ptr %arg.1, align 8
  %scalar.0.234 = fmul double %load.0.234.0, %load.0.234.1
  store double %scalar.0.234, ptr %value.234, align 8
  %scalar.1.235 = fneg double %scalar.0.234
  store double %scalar.1.235, ptr %value.235, align 8
  %scalar.2.236 = call double @llvm.fma.f64(double %load.0.234.0, double %load.0.234.1, double %scalar.1.235)
  store double %scalar.2.236, ptr %value.236, align 8
  %load.3.237.1 = load double, ptr %arg.24, align 8
  %scalar.3.237 = fmul double %load.0.234.0, %load.3.237.1
  store double %scalar.3.237, ptr %value.237, align 8
  %scalar.4.238 = fadd double %scalar.2.236, %scalar.3.237
  store double %scalar.4.238, ptr %value.238, align 8
  %load.5.239.0 = load double, ptr %arg.23, align 8
  %scalar.5.239 = fmul double %load.5.239.0, %load.0.234.1
  store double %scalar.5.239, ptr %value.239, align 8
  %scalar.6.240 = fadd double %scalar.4.238, %scalar.5.239
  store double %scalar.6.240, ptr %value.240, align 8
  %scalar.7.241 = fadd double %scalar.0.234, %scalar.6.240
  store double %scalar.7.241, ptr %value.241, align 8
  %scalar.8.242 = fsub double %scalar.7.241, %scalar.0.234
  store double %scalar.8.242, ptr %value.242, align 8
  %scalar.9.243 = fsub double %scalar.6.240, %scalar.8.242
  store double %scalar.9.243, ptr %value.243, align 8
  %scalar.10.23 = fadd double %scalar.7.241, %scalar.9.243
  store double %scalar.10.23, ptr %out.1, align 8
  %load.11.244.0 = load double, ptr %arg.2, align 8
  %scalar.11.244 = fadd double %load.11.244.0, %scalar.7.241
  store double %scalar.11.244, ptr %value.244, align 8
  %scalar.12.245 = fsub double %scalar.11.244, %load.11.244.0
  store double %scalar.12.245, ptr %value.245, align 8
  %scalar.13.246 = fsub double %scalar.11.244, %scalar.12.245
  store double %scalar.13.246, ptr %value.246, align 8
  %scalar.14.247 = fsub double %load.11.244.0, %scalar.13.246
  store double %scalar.14.247, ptr %value.247, align 8
  %scalar.15.248 = fsub double %scalar.7.241, %scalar.12.245
  store double %scalar.15.248, ptr %value.248, align 8
  %scalar.16.249 = fadd double %scalar.14.247, %scalar.15.248
  store double %scalar.16.249, ptr %value.249, align 8
  %load.17.250.1 = load double, ptr %arg.25, align 8
  %scalar.17.250 = fadd double %scalar.16.249, %load.17.250.1
  store double %scalar.17.250, ptr %value.250, align 8
  %scalar.18.251 = fadd double %scalar.17.250, %scalar.9.243
  store double %scalar.18.251, ptr %value.251, align 8
  %scalar.19.252 = fadd double %scalar.11.244, %scalar.18.251
  store double %scalar.19.252, ptr %value.252, align 8
  %scalar.20.253 = fsub double %scalar.19.252, %scalar.11.244
  store double %scalar.20.253, ptr %value.253, align 8
  %scalar.21.254 = fsub double %scalar.18.251, %scalar.20.253
  store double %scalar.21.254, ptr %value.254, align 8
  %scalar.22.24 = fadd double %scalar.19.252, %scalar.21.254
  store double %scalar.22.24, ptr %out.2, align 8
  %scalar.23.255 = fmul double %load.0.234.1, %scalar.19.252
  store double %scalar.23.255, ptr %value.255, align 8
  %scalar.24.256 = fneg double %scalar.23.255
  store double %scalar.24.256, ptr %value.256, align 8
  %scalar.25.257 = call double @llvm.fma.f64(double %load.0.234.1, double %scalar.19.252, double %scalar.24.256)
  store double %scalar.25.257, ptr %value.257, align 8
  %scalar.26.258 = fmul double %load.0.234.1, %scalar.21.254
  store double %scalar.26.258, ptr %value.258, align 8
  %scalar.27.259 = fadd double %scalar.25.257, %scalar.26.258
  store double %scalar.27.259, ptr %value.259, align 8
  %scalar.28.260 = fmul double %load.3.237.1, %scalar.19.252
  store double %scalar.28.260, ptr %value.260, align 8
  %scalar.29.261 = fadd double %scalar.27.259, %scalar.28.260
  store double %scalar.29.261, ptr %value.261, align 8
  %scalar.30.262 = fadd double %scalar.23.255, %scalar.29.261
  store double %scalar.30.262, ptr %value.262, align 8
  %scalar.31.263 = fsub double %scalar.30.262, %scalar.23.255
  store double %scalar.31.263, ptr %value.263, align 8
  %scalar.32.264 = fsub double %scalar.29.261, %scalar.31.263
  store double %scalar.32.264, ptr %value.264, align 8
  %scalar.33.25 = fadd double %scalar.30.262, %scalar.32.264
  store double %scalar.33.25, ptr %out.3, align 8
  %load.34.265.0 = load double, ptr %arg.3, align 8
  %scalar.34.265 = fadd double %load.34.265.0, %scalar.30.262
  store double %scalar.34.265, ptr %value.265, align 8
  %scalar.35.266 = fsub double %scalar.34.265, %load.34.265.0
  store double %scalar.35.266, ptr %value.266, align 8
  %scalar.36.267 = fsub double %scalar.34.265, %scalar.35.266
  store double %scalar.36.267, ptr %value.267, align 8
  %scalar.37.268 = fsub double %load.34.265.0, %scalar.36.267
  store double %scalar.37.268, ptr %value.268, align 8
  %scalar.38.269 = fsub double %scalar.30.262, %scalar.35.266
  store double %scalar.38.269, ptr %value.269, align 8
  %scalar.39.270 = fadd double %scalar.37.268, %scalar.38.269
  store double %scalar.39.270, ptr %value.270, align 8
  %load.40.271.1 = load double, ptr %arg.26, align 8
  %scalar.40.271 = fadd double %scalar.39.270, %load.40.271.1
  store double %scalar.40.271, ptr %value.271, align 8
  %scalar.41.272 = fadd double %scalar.40.271, %scalar.32.264
  store double %scalar.41.272, ptr %value.272, align 8
  %scalar.42.273 = fadd double %scalar.34.265, %scalar.41.272
  store double %scalar.42.273, ptr %value.273, align 8
  %scalar.43.274 = fsub double %scalar.42.273, %scalar.34.265
  store double %scalar.43.274, ptr %value.274, align 8
  %scalar.44.275 = fsub double %scalar.41.272, %scalar.43.274
  store double %scalar.44.275, ptr %value.275, align 8
  %scalar.45.26 = fadd double %scalar.42.273, %scalar.44.275
  store double %scalar.45.26, ptr %out.4, align 8
  %scalar.46.276 = fmul double %load.0.234.1, %scalar.42.273
  store double %scalar.46.276, ptr %value.276, align 8
  %scalar.47.277 = fneg double %scalar.46.276
  store double %scalar.47.277, ptr %value.277, align 8
  %scalar.48.278 = call double @llvm.fma.f64(double %load.0.234.1, double %scalar.42.273, double %scalar.47.277)
  store double %scalar.48.278, ptr %value.278, align 8
  %scalar.49.279 = fmul double %load.0.234.1, %scalar.44.275
  store double %scalar.49.279, ptr %value.279, align 8
  %scalar.50.280 = fadd double %scalar.48.278, %scalar.49.279
  store double %scalar.50.280, ptr %value.280, align 8
  %scalar.51.281 = fmul double %load.3.237.1, %scalar.42.273
  store double %scalar.51.281, ptr %value.281, align 8
  %scalar.52.282 = fadd double %scalar.50.280, %scalar.51.281
  store double %scalar.52.282, ptr %value.282, align 8
  %scalar.53.283 = fadd double %scalar.46.276, %scalar.52.282
  store double %scalar.53.283, ptr %value.283, align 8
  %scalar.54.284 = fsub double %scalar.53.283, %scalar.46.276
  store double %scalar.54.284, ptr %value.284, align 8
  %scalar.55.285 = fsub double %scalar.52.282, %scalar.54.284
  store double %scalar.55.285, ptr %value.285, align 8
  %scalar.56.27 = fadd double %scalar.53.283, %scalar.55.285
  store double %scalar.56.27, ptr %out.5, align 8
  %load.57.286.0 = load double, ptr %arg.4, align 8
  %scalar.57.286 = fadd double %load.57.286.0, %scalar.53.283
  store double %scalar.57.286, ptr %value.286, align 8
  %scalar.58.287 = fsub double %scalar.57.286, %load.57.286.0
  store double %scalar.58.287, ptr %value.287, align 8
  %scalar.59.288 = fsub double %scalar.57.286, %scalar.58.287
  store double %scalar.59.288, ptr %value.288, align 8
  %scalar.60.289 = fsub double %load.57.286.0, %scalar.59.288
  store double %scalar.60.289, ptr %value.289, align 8
  %scalar.61.290 = fsub double %scalar.53.283, %scalar.58.287
  store double %scalar.61.290, ptr %value.290, align 8
  %scalar.62.291 = fadd double %scalar.60.289, %scalar.61.290
  store double %scalar.62.291, ptr %value.291, align 8
  %load.63.292.1 = load double, ptr %arg.27, align 8
  %scalar.63.292 = fadd double %scalar.62.291, %load.63.292.1
  store double %scalar.63.292, ptr %value.292, align 8
  %scalar.64.293 = fadd double %scalar.63.292, %scalar.55.285
  store double %scalar.64.293, ptr %value.293, align 8
  %scalar.65.294 = fadd double %scalar.57.286, %scalar.64.293
  store double %scalar.65.294, ptr %value.294, align 8
  %scalar.66.295 = fsub double %scalar.65.294, %scalar.57.286
  store double %scalar.66.295, ptr %value.295, align 8
  %scalar.67.296 = fsub double %scalar.64.293, %scalar.66.295
  store double %scalar.67.296, ptr %value.296, align 8
  %scalar.68.28 = fadd double %scalar.65.294, %scalar.67.296
  store double %scalar.68.28, ptr %out.6, align 8
  %scalar.69.297 = fmul double %load.0.234.1, %scalar.65.294
  store double %scalar.69.297, ptr %value.297, align 8
  %scalar.70.298 = fneg double %scalar.69.297
  store double %scalar.70.298, ptr %value.298, align 8
  %scalar.71.299 = call double @llvm.fma.f64(double %load.0.234.1, double %scalar.65.294, double %scalar.70.298)
  store double %scalar.71.299, ptr %value.299, align 8
  %scalar.72.300 = fmul double %load.0.234.1, %scalar.67.296
  store double %scalar.72.300, ptr %value.300, align 8
  %scalar.73.301 = fadd double %scalar.71.299, %scalar.72.300
  store double %scalar.73.301, ptr %value.301, align 8
  %scalar.74.302 = fmul double %load.3.237.1, %scalar.65.294
  store double %scalar.74.302, ptr %value.302, align 8
  %scalar.75.303 = fadd double %scalar.73.301, %scalar.74.302
  store double %scalar.75.303, ptr %value.303, align 8
  %scalar.76.304 = fadd double %scalar.69.297, %scalar.75.303
  store double %scalar.76.304, ptr %value.304, align 8
  %scalar.77.305 = fsub double %scalar.76.304, %scalar.69.297
  store double %scalar.77.305, ptr %value.305, align 8
  %scalar.78.306 = fsub double %scalar.75.303, %scalar.77.305
  store double %scalar.78.306, ptr %value.306, align 8
  %scalar.79.29 = fadd double %scalar.76.304, %scalar.78.306
  store double %scalar.79.29, ptr %out.7, align 8
  %load.80.307.0 = load double, ptr %arg.5, align 8
  %scalar.80.307 = fadd double %load.80.307.0, %scalar.76.304
  store double %scalar.80.307, ptr %value.307, align 8
  %scalar.81.308 = fsub double %scalar.80.307, %load.80.307.0
  store double %scalar.81.308, ptr %value.308, align 8
  %scalar.82.309 = fsub double %scalar.80.307, %scalar.81.308
  store double %scalar.82.309, ptr %value.309, align 8
  %scalar.83.310 = fsub double %load.80.307.0, %scalar.82.309
  store double %scalar.83.310, ptr %value.310, align 8
  %scalar.84.311 = fsub double %scalar.76.304, %scalar.81.308
  store double %scalar.84.311, ptr %value.311, align 8
  %scalar.85.312 = fadd double %scalar.83.310, %scalar.84.311
  store double %scalar.85.312, ptr %value.312, align 8
  %load.86.313.1 = load double, ptr %arg.28, align 8
  %scalar.86.313 = fadd double %scalar.85.312, %load.86.313.1
  store double %scalar.86.313, ptr %value.313, align 8
  %scalar.87.314 = fadd double %scalar.86.313, %scalar.78.306
  store double %scalar.87.314, ptr %value.314, align 8
  %scalar.88.315 = fadd double %scalar.80.307, %scalar.87.314
  store double %scalar.88.315, ptr %value.315, align 8
  %scalar.89.316 = fsub double %scalar.88.315, %scalar.80.307
  store double %scalar.89.316, ptr %value.316, align 8
  %scalar.90.317 = fsub double %scalar.87.314, %scalar.89.316
  store double %scalar.90.317, ptr %value.317, align 8
  %scalar.91.30 = fadd double %scalar.88.315, %scalar.90.317
  store double %scalar.91.30, ptr %out.8, align 8
  %scalar.92.318 = fmul double %load.0.234.1, %scalar.88.315
  store double %scalar.92.318, ptr %value.318, align 8
  %scalar.93.319 = fneg double %scalar.92.318
  store double %scalar.93.319, ptr %value.319, align 8
  %scalar.94.320 = call double @llvm.fma.f64(double %load.0.234.1, double %scalar.88.315, double %scalar.93.319)
  store double %scalar.94.320, ptr %value.320, align 8
  %scalar.95.321 = fmul double %load.0.234.1, %scalar.90.317
  store double %scalar.95.321, ptr %value.321, align 8
  %scalar.96.322 = fadd double %scalar.94.320, %scalar.95.321
  store double %scalar.96.322, ptr %value.322, align 8
  %scalar.97.323 = fmul double %load.3.237.1, %scalar.88.315
  store double %scalar.97.323, ptr %value.323, align 8
  %scalar.98.324 = fadd double %scalar.96.322, %scalar.97.323
  store double %scalar.98.324, ptr %value.324, align 8
  %scalar.99.325 = fadd double %scalar.92.318, %scalar.98.324
  store double %scalar.99.325, ptr %value.325, align 8
  %scalar.100.326 = fsub double %scalar.99.325, %scalar.92.318
  store double %scalar.100.326, ptr %value.326, align 8
  %scalar.101.327 = fsub double %scalar.98.324, %scalar.100.326
  store double %scalar.101.327, ptr %value.327, align 8
  %scalar.102.31 = fadd double %scalar.99.325, %scalar.101.327
  store double %scalar.102.31, ptr %out.9, align 8
  %load.103.328.0 = load double, ptr %arg.6, align 8
  %scalar.103.328 = fadd double %load.103.328.0, %scalar.99.325
  store double %scalar.103.328, ptr %value.328, align 8
  %scalar.104.329 = fsub double %scalar.103.328, %load.103.328.0
  store double %scalar.104.329, ptr %value.329, align 8
  %scalar.105.330 = fsub double %scalar.103.328, %scalar.104.329
  store double %scalar.105.330, ptr %value.330, align 8
  %scalar.106.331 = fsub double %load.103.328.0, %scalar.105.330
  store double %scalar.106.331, ptr %value.331, align 8
  %scalar.107.332 = fsub double %scalar.99.325, %scalar.104.329
  store double %scalar.107.332, ptr %value.332, align 8
  %scalar.108.333 = fadd double %scalar.106.331, %scalar.107.332
  store double %scalar.108.333, ptr %value.333, align 8
  %load.109.334.1 = load double, ptr %arg.29, align 8
  %scalar.109.334 = fadd double %scalar.108.333, %load.109.334.1
  store double %scalar.109.334, ptr %value.334, align 8
  %scalar.110.335 = fadd double %scalar.109.334, %scalar.101.327
  store double %scalar.110.335, ptr %value.335, align 8
  %scalar.111.336 = fadd double %scalar.103.328, %scalar.110.335
  store double %scalar.111.336, ptr %value.336, align 8
  %scalar.112.337 = fsub double %scalar.111.336, %scalar.103.328
  store double %scalar.112.337, ptr %value.337, align 8
  %scalar.113.338 = fsub double %scalar.110.335, %scalar.112.337
  store double %scalar.113.338, ptr %value.338, align 8
  %scalar.114.32 = fadd double %scalar.111.336, %scalar.113.338
  store double %scalar.114.32, ptr %out.10, align 8
  %scalar.115.339 = fmul double %load.0.234.1, %scalar.111.336
  store double %scalar.115.339, ptr %value.339, align 8
  %scalar.116.340 = fneg double %scalar.115.339
  store double %scalar.116.340, ptr %value.340, align 8
  %scalar.117.341 = call double @llvm.fma.f64(double %load.0.234.1, double %scalar.111.336, double %scalar.116.340)
  store double %scalar.117.341, ptr %value.341, align 8
  %scalar.118.342 = fmul double %load.0.234.1, %scalar.113.338
  store double %scalar.118.342, ptr %value.342, align 8
  %scalar.119.343 = fadd double %scalar.117.341, %scalar.118.342
  store double %scalar.119.343, ptr %value.343, align 8
  %scalar.120.344 = fmul double %load.3.237.1, %scalar.111.336
  store double %scalar.120.344, ptr %value.344, align 8
  %scalar.121.345 = fadd double %scalar.119.343, %scalar.120.344
  store double %scalar.121.345, ptr %value.345, align 8
  %scalar.122.346 = fadd double %scalar.115.339, %scalar.121.345
  store double %scalar.122.346, ptr %value.346, align 8
  %scalar.123.347 = fsub double %scalar.122.346, %scalar.115.339
  store double %scalar.123.347, ptr %value.347, align 8
  %scalar.124.348 = fsub double %scalar.121.345, %scalar.123.347
  store double %scalar.124.348, ptr %value.348, align 8
  %scalar.125.33 = fadd double %scalar.122.346, %scalar.124.348
  store double %scalar.125.33, ptr %out.11, align 8
  %load.126.349.0 = load double, ptr %arg.7, align 8
  %scalar.126.349 = fadd double %load.126.349.0, %scalar.122.346
  store double %scalar.126.349, ptr %value.349, align 8
  %scalar.127.350 = fsub double %scalar.126.349, %load.126.349.0
  store double %scalar.127.350, ptr %value.350, align 8
  %scalar.128.351 = fsub double %scalar.126.349, %scalar.127.350
  store double %scalar.128.351, ptr %value.351, align 8
  %scalar.129.352 = fsub double %load.126.349.0, %scalar.128.351
  store double %scalar.129.352, ptr %value.352, align 8
  %scalar.130.353 = fsub double %scalar.122.346, %scalar.127.350
  store double %scalar.130.353, ptr %value.353, align 8
  %scalar.131.354 = fadd double %scalar.129.352, %scalar.130.353
  store double %scalar.131.354, ptr %value.354, align 8
  %load.132.355.1 = load double, ptr %arg.30, align 8
  %scalar.132.355 = fadd double %scalar.131.354, %load.132.355.1
  store double %scalar.132.355, ptr %value.355, align 8
  %scalar.133.356 = fadd double %scalar.132.355, %scalar.124.348
  store double %scalar.133.356, ptr %value.356, align 8
  %scalar.134.357 = fadd double %scalar.126.349, %scalar.133.356
  store double %scalar.134.357, ptr %value.357, align 8
  %scalar.135.358 = fsub double %scalar.134.357, %scalar.126.349
  store double %scalar.135.358, ptr %value.358, align 8
  %scalar.136.359 = fsub double %scalar.133.356, %scalar.135.358
  store double %scalar.136.359, ptr %value.359, align 8
  %scalar.137.34 = fadd double %scalar.134.357, %scalar.136.359
  store double %scalar.137.34, ptr %out.12, align 8
  %scalar.138.360 = fmul double %load.0.234.1, %scalar.134.357
  store double %scalar.138.360, ptr %value.360, align 8
  %scalar.139.361 = fneg double %scalar.138.360
  store double %scalar.139.361, ptr %value.361, align 8
  %scalar.140.362 = call double @llvm.fma.f64(double %load.0.234.1, double %scalar.134.357, double %scalar.139.361)
  store double %scalar.140.362, ptr %value.362, align 8
  %scalar.141.363 = fmul double %load.0.234.1, %scalar.136.359
  store double %scalar.141.363, ptr %value.363, align 8
  %scalar.142.364 = fadd double %scalar.140.362, %scalar.141.363
  store double %scalar.142.364, ptr %value.364, align 8
  %scalar.143.365 = fmul double %load.3.237.1, %scalar.134.357
  store double %scalar.143.365, ptr %value.365, align 8
  %scalar.144.366 = fadd double %scalar.142.364, %scalar.143.365
  store double %scalar.144.366, ptr %value.366, align 8
  %scalar.145.367 = fadd double %scalar.138.360, %scalar.144.366
  store double %scalar.145.367, ptr %value.367, align 8
  %scalar.146.368 = fsub double %scalar.145.367, %scalar.138.360
  store double %scalar.146.368, ptr %value.368, align 8
  %scalar.147.369 = fsub double %scalar.144.366, %scalar.146.368
  store double %scalar.147.369, ptr %value.369, align 8
  %scalar.148.35 = fadd double %scalar.145.367, %scalar.147.369
  store double %scalar.148.35, ptr %out.13, align 8
  %load.149.370.0 = load double, ptr %arg.8, align 8
  %scalar.149.370 = fadd double %load.149.370.0, %scalar.145.367
  store double %scalar.149.370, ptr %value.370, align 8
  %scalar.150.371 = fsub double %scalar.149.370, %load.149.370.0
  store double %scalar.150.371, ptr %value.371, align 8
  %scalar.151.372 = fsub double %scalar.149.370, %scalar.150.371
  store double %scalar.151.372, ptr %value.372, align 8
  %scalar.152.373 = fsub double %load.149.370.0, %scalar.151.372
  store double %scalar.152.373, ptr %value.373, align 8
  %scalar.153.374 = fsub double %scalar.145.367, %scalar.150.371
  store double %scalar.153.374, ptr %value.374, align 8
  %scalar.154.375 = fadd double %scalar.152.373, %scalar.153.374
  store double %scalar.154.375, ptr %value.375, align 8
  %load.155.376.1 = load double, ptr %arg.31, align 8
  %scalar.155.376 = fadd double %scalar.154.375, %load.155.376.1
  store double %scalar.155.376, ptr %value.376, align 8
  %scalar.156.377 = fadd double %scalar.155.376, %scalar.147.369
  store double %scalar.156.377, ptr %value.377, align 8
  %scalar.157.378 = fadd double %scalar.149.370, %scalar.156.377
  store double %scalar.157.378, ptr %value.378, align 8
  %scalar.158.379 = fsub double %scalar.157.378, %scalar.149.370
  store double %scalar.158.379, ptr %value.379, align 8
  %scalar.159.380 = fsub double %scalar.156.377, %scalar.158.379
  store double %scalar.159.380, ptr %value.380, align 8
  %scalar.160.36 = fadd double %scalar.157.378, %scalar.159.380
  store double %scalar.160.36, ptr %out.14, align 8
  %scalar.161.381 = fmul double %load.0.234.1, %scalar.157.378
  store double %scalar.161.381, ptr %value.381, align 8
  %scalar.162.382 = fneg double %scalar.161.381
  store double %scalar.162.382, ptr %value.382, align 8
  %scalar.163.383 = call double @llvm.fma.f64(double %load.0.234.1, double %scalar.157.378, double %scalar.162.382)
  store double %scalar.163.383, ptr %value.383, align 8
  %scalar.164.384 = fmul double %load.0.234.1, %scalar.159.380
  store double %scalar.164.384, ptr %value.384, align 8
  %scalar.165.385 = fadd double %scalar.163.383, %scalar.164.384
  store double %scalar.165.385, ptr %value.385, align 8
  %scalar.166.386 = fmul double %load.3.237.1, %scalar.157.378
  store double %scalar.166.386, ptr %value.386, align 8
  %scalar.167.387 = fadd double %scalar.165.385, %scalar.166.386
  store double %scalar.167.387, ptr %value.387, align 8
  %scalar.168.388 = fadd double %scalar.161.381, %scalar.167.387
  store double %scalar.168.388, ptr %value.388, align 8
  %scalar.169.389 = fsub double %scalar.168.388, %scalar.161.381
  store double %scalar.169.389, ptr %value.389, align 8
  %scalar.170.390 = fsub double %scalar.167.387, %scalar.169.389
  store double %scalar.170.390, ptr %value.390, align 8
  %scalar.171.37 = fadd double %scalar.168.388, %scalar.170.390
  store double %scalar.171.37, ptr %out.15, align 8
  %load.172.391.0 = load double, ptr %arg.9, align 8
  %scalar.172.391 = fadd double %load.172.391.0, %scalar.168.388
  store double %scalar.172.391, ptr %value.391, align 8
  %scalar.173.392 = fsub double %scalar.172.391, %load.172.391.0
  store double %scalar.173.392, ptr %value.392, align 8
  %scalar.174.393 = fsub double %scalar.172.391, %scalar.173.392
  store double %scalar.174.393, ptr %value.393, align 8
  %scalar.175.394 = fsub double %load.172.391.0, %scalar.174.393
  store double %scalar.175.394, ptr %value.394, align 8
  %scalar.176.395 = fsub double %scalar.168.388, %scalar.173.392
  store double %scalar.176.395, ptr %value.395, align 8
  %scalar.177.396 = fadd double %scalar.175.394, %scalar.176.395
  store double %scalar.177.396, ptr %value.396, align 8
  %load.178.397.1 = load double, ptr %arg.32, align 8
  %scalar.178.397 = fadd double %scalar.177.396, %load.178.397.1
  store double %scalar.178.397, ptr %value.397, align 8
  %scalar.179.398 = fadd double %scalar.178.397, %scalar.170.390
  store double %scalar.179.398, ptr %value.398, align 8
  %scalar.180.399 = fadd double %scalar.172.391, %scalar.179.398
  store double %scalar.180.399, ptr %value.399, align 8
  %scalar.181.400 = fsub double %scalar.180.399, %scalar.172.391
  store double %scalar.181.400, ptr %value.400, align 8
  %scalar.182.401 = fsub double %scalar.179.398, %scalar.181.400
  store double %scalar.182.401, ptr %value.401, align 8
  %scalar.183.38 = fadd double %scalar.180.399, %scalar.182.401
  store double %scalar.183.38, ptr %out.16, align 8
  %scalar.184.402 = fmul double %load.0.234.1, %scalar.180.399
  store double %scalar.184.402, ptr %value.402, align 8
  %scalar.185.403 = fneg double %scalar.184.402
  store double %scalar.185.403, ptr %value.403, align 8
  %scalar.186.404 = call double @llvm.fma.f64(double %load.0.234.1, double %scalar.180.399, double %scalar.185.403)
  store double %scalar.186.404, ptr %value.404, align 8
  %scalar.187.405 = fmul double %load.0.234.1, %scalar.182.401
  store double %scalar.187.405, ptr %value.405, align 8
  %scalar.188.406 = fadd double %scalar.186.404, %scalar.187.405
  store double %scalar.188.406, ptr %value.406, align 8
  %scalar.189.407 = fmul double %load.3.237.1, %scalar.180.399
  store double %scalar.189.407, ptr %value.407, align 8
  %scalar.190.408 = fadd double %scalar.188.406, %scalar.189.407
  store double %scalar.190.408, ptr %value.408, align 8
  %scalar.191.409 = fadd double %scalar.184.402, %scalar.190.408
  store double %scalar.191.409, ptr %value.409, align 8
  %scalar.192.410 = fsub double %scalar.191.409, %scalar.184.402
  store double %scalar.192.410, ptr %value.410, align 8
  %scalar.193.411 = fsub double %scalar.190.408, %scalar.192.410
  store double %scalar.193.411, ptr %value.411, align 8
  %scalar.194.39 = fadd double %scalar.191.409, %scalar.193.411
  store double %scalar.194.39, ptr %out.17, align 8
  %load.195.412.0 = load double, ptr %arg.10, align 8
  %scalar.195.412 = fadd double %load.195.412.0, %scalar.191.409
  store double %scalar.195.412, ptr %value.412, align 8
  %scalar.196.413 = fsub double %scalar.195.412, %load.195.412.0
  store double %scalar.196.413, ptr %value.413, align 8
  %scalar.197.414 = fsub double %scalar.195.412, %scalar.196.413
  store double %scalar.197.414, ptr %value.414, align 8
  %scalar.198.415 = fsub double %load.195.412.0, %scalar.197.414
  store double %scalar.198.415, ptr %value.415, align 8
  %scalar.199.416 = fsub double %scalar.191.409, %scalar.196.413
  store double %scalar.199.416, ptr %value.416, align 8
  %scalar.200.417 = fadd double %scalar.198.415, %scalar.199.416
  store double %scalar.200.417, ptr %value.417, align 8
  %load.201.418.1 = load double, ptr %arg.33, align 8
  %scalar.201.418 = fadd double %scalar.200.417, %load.201.418.1
  store double %scalar.201.418, ptr %value.418, align 8
  %scalar.202.419 = fadd double %scalar.201.418, %scalar.193.411
  store double %scalar.202.419, ptr %value.419, align 8
  %scalar.203.420 = fadd double %scalar.195.412, %scalar.202.419
  store double %scalar.203.420, ptr %value.420, align 8
  %scalar.204.421 = fsub double %scalar.203.420, %scalar.195.412
  store double %scalar.204.421, ptr %value.421, align 8
  %scalar.205.422 = fsub double %scalar.202.419, %scalar.204.421
  store double %scalar.205.422, ptr %value.422, align 8
  %scalar.206.40 = fadd double %scalar.203.420, %scalar.205.422
  store double %scalar.206.40, ptr %out.18, align 8
  %scalar.207.423 = fmul double %load.0.234.1, %scalar.203.420
  store double %scalar.207.423, ptr %value.423, align 8
  %scalar.208.424 = fneg double %scalar.207.423
  store double %scalar.208.424, ptr %value.424, align 8
  %scalar.209.425 = call double @llvm.fma.f64(double %load.0.234.1, double %scalar.203.420, double %scalar.208.424)
  store double %scalar.209.425, ptr %value.425, align 8
  %scalar.210.426 = fmul double %load.0.234.1, %scalar.205.422
  store double %scalar.210.426, ptr %value.426, align 8
  %scalar.211.427 = fadd double %scalar.209.425, %scalar.210.426
  store double %scalar.211.427, ptr %value.427, align 8
  %scalar.212.428 = fmul double %load.3.237.1, %scalar.203.420
  store double %scalar.212.428, ptr %value.428, align 8
  %scalar.213.429 = fadd double %scalar.211.427, %scalar.212.428
  store double %scalar.213.429, ptr %value.429, align 8
  %scalar.214.430 = fadd double %scalar.207.423, %scalar.213.429
  store double %scalar.214.430, ptr %value.430, align 8
  %scalar.215.431 = fsub double %scalar.214.430, %scalar.207.423
  store double %scalar.215.431, ptr %value.431, align 8
  %scalar.216.432 = fsub double %scalar.213.429, %scalar.215.431
  store double %scalar.216.432, ptr %value.432, align 8
  %scalar.217.41 = fadd double %scalar.214.430, %scalar.216.432
  store double %scalar.217.41, ptr %out.19, align 8
  %load.218.433.0 = load double, ptr %arg.11, align 8
  %scalar.218.433 = fadd double %load.218.433.0, %scalar.214.430
  store double %scalar.218.433, ptr %value.433, align 8
  %scalar.219.434 = fsub double %scalar.218.433, %load.218.433.0
  store double %scalar.219.434, ptr %value.434, align 8
  %scalar.220.435 = fsub double %scalar.218.433, %scalar.219.434
  store double %scalar.220.435, ptr %value.435, align 8
  %scalar.221.436 = fsub double %load.218.433.0, %scalar.220.435
  store double %scalar.221.436, ptr %value.436, align 8
  %scalar.222.437 = fsub double %scalar.214.430, %scalar.219.434
  store double %scalar.222.437, ptr %value.437, align 8
  %scalar.223.438 = fadd double %scalar.221.436, %scalar.222.437
  store double %scalar.223.438, ptr %value.438, align 8
  %load.224.439.1 = load double, ptr %arg.34, align 8
  %scalar.224.439 = fadd double %scalar.223.438, %load.224.439.1
  store double %scalar.224.439, ptr %value.439, align 8
  %scalar.225.440 = fadd double %scalar.224.439, %scalar.216.432
  store double %scalar.225.440, ptr %value.440, align 8
  %scalar.226.441 = fadd double %scalar.218.433, %scalar.225.440
  store double %scalar.226.441, ptr %value.441, align 8
  %scalar.227.442 = fsub double %scalar.226.441, %scalar.218.433
  store double %scalar.227.442, ptr %value.442, align 8
  %scalar.228.443 = fsub double %scalar.225.440, %scalar.227.442
  store double %scalar.228.443, ptr %value.443, align 8
  %scalar.229.42 = fadd double %scalar.226.441, %scalar.228.443
  store double %scalar.229.42, ptr %out.20, align 8
  %scalar.230.444 = fmul double %load.0.234.1, %scalar.226.441
  store double %scalar.230.444, ptr %value.444, align 8
  %scalar.231.445 = fneg double %scalar.230.444
  store double %scalar.231.445, ptr %value.445, align 8
  %scalar.232.446 = call double @llvm.fma.f64(double %load.0.234.1, double %scalar.226.441, double %scalar.231.445)
  store double %scalar.232.446, ptr %value.446, align 8
  %scalar.233.447 = fmul double %load.0.234.1, %scalar.228.443
  store double %scalar.233.447, ptr %value.447, align 8
  %scalar.234.448 = fadd double %scalar.232.446, %scalar.233.447
  store double %scalar.234.448, ptr %value.448, align 8
  %scalar.235.449 = fmul double %load.3.237.1, %scalar.226.441
  store double %scalar.235.449, ptr %value.449, align 8
  %scalar.236.450 = fadd double %scalar.234.448, %scalar.235.449
  store double %scalar.236.450, ptr %value.450, align 8
  %scalar.237.451 = fadd double %scalar.230.444, %scalar.236.450
  store double %scalar.237.451, ptr %value.451, align 8
  %scalar.238.452 = fsub double %scalar.237.451, %scalar.230.444
  store double %scalar.238.452, ptr %value.452, align 8
  %scalar.239.453 = fsub double %scalar.236.450, %scalar.238.452
  store double %scalar.239.453, ptr %value.453, align 8
  %scalar.240.43 = fadd double %scalar.237.451, %scalar.239.453
  store double %scalar.240.43, ptr %out.21, align 8
  %load.241.454.0 = load double, ptr %arg.12, align 8
  %scalar.241.454 = fadd double %load.241.454.0, %scalar.237.451
  store double %scalar.241.454, ptr %value.454, align 8
  %scalar.242.455 = fsub double %scalar.241.454, %load.241.454.0
  store double %scalar.242.455, ptr %value.455, align 8
  %scalar.243.456 = fsub double %scalar.241.454, %scalar.242.455
  store double %scalar.243.456, ptr %value.456, align 8
  %scalar.244.457 = fsub double %load.241.454.0, %scalar.243.456
  store double %scalar.244.457, ptr %value.457, align 8
  %scalar.245.458 = fsub double %scalar.237.451, %scalar.242.455
  store double %scalar.245.458, ptr %value.458, align 8
  %scalar.246.459 = fadd double %scalar.244.457, %scalar.245.458
  store double %scalar.246.459, ptr %value.459, align 8
  %load.247.460.1 = load double, ptr %arg.35, align 8
  %scalar.247.460 = fadd double %scalar.246.459, %load.247.460.1
  store double %scalar.247.460, ptr %value.460, align 8
  %scalar.248.461 = fadd double %scalar.247.460, %scalar.239.453
  store double %scalar.248.461, ptr %value.461, align 8
  %scalar.249.462 = fadd double %scalar.241.454, %scalar.248.461
  store double %scalar.249.462, ptr %value.462, align 8
  %scalar.250.463 = fsub double %scalar.249.462, %scalar.241.454
  store double %scalar.250.463, ptr %value.463, align 8
  %scalar.251.464 = fsub double %scalar.248.461, %scalar.250.463
  store double %scalar.251.464, ptr %value.464, align 8
  %scalar.252.44 = fadd double %scalar.249.462, %scalar.251.464
  store double %scalar.252.44, ptr %out.22, align 8
  %scalar.253.465 = fmul double %load.0.234.1, %scalar.249.462
  store double %scalar.253.465, ptr %value.465, align 8
  %scalar.254.466 = fneg double %scalar.253.465
  store double %scalar.254.466, ptr %value.466, align 8
  %scalar.255.467 = call double @llvm.fma.f64(double %load.0.234.1, double %scalar.249.462, double %scalar.254.466)
  store double %scalar.255.467, ptr %value.467, align 8
  %scalar.256.468 = fmul double %load.0.234.1, %scalar.251.464
  store double %scalar.256.468, ptr %value.468, align 8
  %scalar.257.469 = fadd double %scalar.255.467, %scalar.256.468
  store double %scalar.257.469, ptr %value.469, align 8
  %scalar.258.470 = fmul double %load.3.237.1, %scalar.249.462
  store double %scalar.258.470, ptr %value.470, align 8
  %scalar.259.471 = fadd double %scalar.257.469, %scalar.258.470
  store double %scalar.259.471, ptr %value.471, align 8
  %scalar.260.472 = fadd double %scalar.253.465, %scalar.259.471
  store double %scalar.260.472, ptr %value.472, align 8
  %scalar.261.473 = fsub double %scalar.260.472, %scalar.253.465
  store double %scalar.261.473, ptr %value.473, align 8
  %scalar.262.474 = fsub double %scalar.259.471, %scalar.261.473
  store double %scalar.262.474, ptr %value.474, align 8
  %scalar.263.45 = fadd double %scalar.260.472, %scalar.262.474
  store double %scalar.263.45, ptr %out.23, align 8
  %load.264.475.0 = load double, ptr %arg.13, align 8
  %scalar.264.475 = fadd double %load.264.475.0, %scalar.260.472
  store double %scalar.264.475, ptr %value.475, align 8
  %scalar.265.476 = fsub double %scalar.264.475, %load.264.475.0
  store double %scalar.265.476, ptr %value.476, align 8
  %scalar.266.477 = fsub double %scalar.264.475, %scalar.265.476
  store double %scalar.266.477, ptr %value.477, align 8
  %scalar.267.478 = fsub double %load.264.475.0, %scalar.266.477
  store double %scalar.267.478, ptr %value.478, align 8
  %scalar.268.479 = fsub double %scalar.260.472, %scalar.265.476
  store double %scalar.268.479, ptr %value.479, align 8
  %scalar.269.480 = fadd double %scalar.267.478, %scalar.268.479
  store double %scalar.269.480, ptr %value.480, align 8
  %load.270.481.1 = load double, ptr %arg.36, align 8
  %scalar.270.481 = fadd double %scalar.269.480, %load.270.481.1
  store double %scalar.270.481, ptr %value.481, align 8
  %scalar.271.482 = fadd double %scalar.270.481, %scalar.262.474
  store double %scalar.271.482, ptr %value.482, align 8
  %scalar.272.483 = fadd double %scalar.264.475, %scalar.271.482
  store double %scalar.272.483, ptr %value.483, align 8
  %scalar.273.484 = fsub double %scalar.272.483, %scalar.264.475
  store double %scalar.273.484, ptr %value.484, align 8
  %scalar.274.485 = fsub double %scalar.271.482, %scalar.273.484
  store double %scalar.274.485, ptr %value.485, align 8
  %scalar.275.46 = fadd double %scalar.272.483, %scalar.274.485
  store double %scalar.275.46, ptr %out.24, align 8
  %scalar.276.486 = fmul double %load.0.234.1, %scalar.272.483
  store double %scalar.276.486, ptr %value.486, align 8
  %scalar.277.487 = fneg double %scalar.276.486
  store double %scalar.277.487, ptr %value.487, align 8
  %scalar.278.488 = call double @llvm.fma.f64(double %load.0.234.1, double %scalar.272.483, double %scalar.277.487)
  store double %scalar.278.488, ptr %value.488, align 8
  %scalar.279.489 = fmul double %load.0.234.1, %scalar.274.485
  store double %scalar.279.489, ptr %value.489, align 8
  %scalar.280.490 = fadd double %scalar.278.488, %scalar.279.489
  store double %scalar.280.490, ptr %value.490, align 8
  %scalar.281.491 = fmul double %load.3.237.1, %scalar.272.483
  store double %scalar.281.491, ptr %value.491, align 8
  %scalar.282.492 = fadd double %scalar.280.490, %scalar.281.491
  store double %scalar.282.492, ptr %value.492, align 8
  %scalar.283.493 = fadd double %scalar.276.486, %scalar.282.492
  store double %scalar.283.493, ptr %value.493, align 8
  %scalar.284.494 = fsub double %scalar.283.493, %scalar.276.486
  store double %scalar.284.494, ptr %value.494, align 8
  %scalar.285.495 = fsub double %scalar.282.492, %scalar.284.494
  store double %scalar.285.495, ptr %value.495, align 8
  %scalar.286.47 = fadd double %scalar.283.493, %scalar.285.495
  store double %scalar.286.47, ptr %out.25, align 8
  %load.287.496.0 = load double, ptr %arg.14, align 8
  %scalar.287.496 = fadd double %load.287.496.0, %scalar.283.493
  store double %scalar.287.496, ptr %value.496, align 8
  %scalar.288.497 = fsub double %scalar.287.496, %load.287.496.0
  store double %scalar.288.497, ptr %value.497, align 8
  %scalar.289.498 = fsub double %scalar.287.496, %scalar.288.497
  store double %scalar.289.498, ptr %value.498, align 8
  %scalar.290.499 = fsub double %load.287.496.0, %scalar.289.498
  store double %scalar.290.499, ptr %value.499, align 8
  %scalar.291.500 = fsub double %scalar.283.493, %scalar.288.497
  store double %scalar.291.500, ptr %value.500, align 8
  %scalar.292.501 = fadd double %scalar.290.499, %scalar.291.500
  store double %scalar.292.501, ptr %value.501, align 8
  %load.293.502.1 = load double, ptr %arg.37, align 8
  %scalar.293.502 = fadd double %scalar.292.501, %load.293.502.1
  store double %scalar.293.502, ptr %value.502, align 8
  %scalar.294.503 = fadd double %scalar.293.502, %scalar.285.495
  store double %scalar.294.503, ptr %value.503, align 8
  %scalar.295.504 = fadd double %scalar.287.496, %scalar.294.503
  store double %scalar.295.504, ptr %value.504, align 8
  %scalar.296.505 = fsub double %scalar.295.504, %scalar.287.496
  store double %scalar.296.505, ptr %value.505, align 8
  %scalar.297.506 = fsub double %scalar.294.503, %scalar.296.505
  store double %scalar.297.506, ptr %value.506, align 8
  %scalar.298.48 = fadd double %scalar.295.504, %scalar.297.506
  store double %scalar.298.48, ptr %out.26, align 8
  %scalar.299.507 = fmul double %load.0.234.1, %scalar.295.504
  store double %scalar.299.507, ptr %value.507, align 8
  %scalar.300.508 = fneg double %scalar.299.507
  store double %scalar.300.508, ptr %value.508, align 8
  %scalar.301.509 = call double @llvm.fma.f64(double %load.0.234.1, double %scalar.295.504, double %scalar.300.508)
  store double %scalar.301.509, ptr %value.509, align 8
  %scalar.302.510 = fmul double %load.0.234.1, %scalar.297.506
  store double %scalar.302.510, ptr %value.510, align 8
  %scalar.303.511 = fadd double %scalar.301.509, %scalar.302.510
  store double %scalar.303.511, ptr %value.511, align 8
  %scalar.304.512 = fmul double %load.3.237.1, %scalar.295.504
  store double %scalar.304.512, ptr %value.512, align 8
  %scalar.305.513 = fadd double %scalar.303.511, %scalar.304.512
  store double %scalar.305.513, ptr %value.513, align 8
  %scalar.306.514 = fadd double %scalar.299.507, %scalar.305.513
  store double %scalar.306.514, ptr %value.514, align 8
  %scalar.307.515 = fsub double %scalar.306.514, %scalar.299.507
  store double %scalar.307.515, ptr %value.515, align 8
  %scalar.308.516 = fsub double %scalar.305.513, %scalar.307.515
  store double %scalar.308.516, ptr %value.516, align 8
  %scalar.309.49 = fadd double %scalar.306.514, %scalar.308.516
  store double %scalar.309.49, ptr %out.27, align 8
  %load.310.517.0 = load double, ptr %arg.15, align 8
  %scalar.310.517 = fadd double %load.310.517.0, %scalar.306.514
  store double %scalar.310.517, ptr %value.517, align 8
  %scalar.311.518 = fsub double %scalar.310.517, %load.310.517.0
  store double %scalar.311.518, ptr %value.518, align 8
  %scalar.312.519 = fsub double %scalar.310.517, %scalar.311.518
  store double %scalar.312.519, ptr %value.519, align 8
  %scalar.313.520 = fsub double %load.310.517.0, %scalar.312.519
  store double %scalar.313.520, ptr %value.520, align 8
  %scalar.314.521 = fsub double %scalar.306.514, %scalar.311.518
  store double %scalar.314.521, ptr %value.521, align 8
  %scalar.315.522 = fadd double %scalar.313.520, %scalar.314.521
  store double %scalar.315.522, ptr %value.522, align 8
  %load.316.523.1 = load double, ptr %arg.38, align 8
  %scalar.316.523 = fadd double %scalar.315.522, %load.316.523.1
  store double %scalar.316.523, ptr %value.523, align 8
  %scalar.317.524 = fadd double %scalar.316.523, %scalar.308.516
  store double %scalar.317.524, ptr %value.524, align 8
  %scalar.318.525 = fadd double %scalar.310.517, %scalar.317.524
  store double %scalar.318.525, ptr %value.525, align 8
  %scalar.319.526 = fsub double %scalar.318.525, %scalar.310.517
  store double %scalar.319.526, ptr %value.526, align 8
  %scalar.320.527 = fsub double %scalar.317.524, %scalar.319.526
  store double %scalar.320.527, ptr %value.527, align 8
  %scalar.321.50 = fadd double %scalar.318.525, %scalar.320.527
  store double %scalar.321.50, ptr %out.28, align 8
  %scalar.322.528 = fmul double %load.0.234.1, %scalar.318.525
  store double %scalar.322.528, ptr %value.528, align 8
  %scalar.323.529 = fneg double %scalar.322.528
  store double %scalar.323.529, ptr %value.529, align 8
  %scalar.324.530 = call double @llvm.fma.f64(double %load.0.234.1, double %scalar.318.525, double %scalar.323.529)
  store double %scalar.324.530, ptr %value.530, align 8
  %scalar.325.531 = fmul double %load.0.234.1, %scalar.320.527
  store double %scalar.325.531, ptr %value.531, align 8
  %scalar.326.532 = fadd double %scalar.324.530, %scalar.325.531
  store double %scalar.326.532, ptr %value.532, align 8
  %scalar.327.533 = fmul double %load.3.237.1, %scalar.318.525
  store double %scalar.327.533, ptr %value.533, align 8
  %scalar.328.534 = fadd double %scalar.326.532, %scalar.327.533
  store double %scalar.328.534, ptr %value.534, align 8
  %scalar.329.535 = fadd double %scalar.322.528, %scalar.328.534
  store double %scalar.329.535, ptr %value.535, align 8
  %scalar.330.536 = fsub double %scalar.329.535, %scalar.322.528
  store double %scalar.330.536, ptr %value.536, align 8
  %scalar.331.537 = fsub double %scalar.328.534, %scalar.330.536
  store double %scalar.331.537, ptr %value.537, align 8
  %scalar.332.51 = fadd double %scalar.329.535, %scalar.331.537
  store double %scalar.332.51, ptr %out.29, align 8
  %load.333.538.0 = load double, ptr %arg.16, align 8
  %scalar.333.538 = fadd double %load.333.538.0, %scalar.329.535
  store double %scalar.333.538, ptr %value.538, align 8
  %scalar.334.539 = fsub double %scalar.333.538, %load.333.538.0
  store double %scalar.334.539, ptr %value.539, align 8
  %scalar.335.540 = fsub double %scalar.333.538, %scalar.334.539
  store double %scalar.335.540, ptr %value.540, align 8
  %scalar.336.541 = fsub double %load.333.538.0, %scalar.335.540
  store double %scalar.336.541, ptr %value.541, align 8
  %scalar.337.542 = fsub double %scalar.329.535, %scalar.334.539
  store double %scalar.337.542, ptr %value.542, align 8
  %scalar.338.543 = fadd double %scalar.336.541, %scalar.337.542
  store double %scalar.338.543, ptr %value.543, align 8
  %load.339.544.1 = load double, ptr %arg.39, align 8
  %scalar.339.544 = fadd double %scalar.338.543, %load.339.544.1
  store double %scalar.339.544, ptr %value.544, align 8
  %scalar.340.545 = fadd double %scalar.339.544, %scalar.331.537
  store double %scalar.340.545, ptr %value.545, align 8
  %scalar.341.546 = fadd double %scalar.333.538, %scalar.340.545
  store double %scalar.341.546, ptr %value.546, align 8
  %scalar.342.547 = fsub double %scalar.341.546, %scalar.333.538
  store double %scalar.342.547, ptr %value.547, align 8
  %scalar.343.548 = fsub double %scalar.340.545, %scalar.342.547
  store double %scalar.343.548, ptr %value.548, align 8
  %scalar.344.52 = fadd double %scalar.341.546, %scalar.343.548
  store double %scalar.344.52, ptr %out.30, align 8
  %scalar.345.549 = fmul double %load.0.234.1, %scalar.341.546
  store double %scalar.345.549, ptr %value.549, align 8
  %scalar.346.550 = fneg double %scalar.345.549
  store double %scalar.346.550, ptr %value.550, align 8
  %scalar.347.551 = call double @llvm.fma.f64(double %load.0.234.1, double %scalar.341.546, double %scalar.346.550)
  store double %scalar.347.551, ptr %value.551, align 8
  %scalar.348.552 = fmul double %load.0.234.1, %scalar.343.548
  store double %scalar.348.552, ptr %value.552, align 8
  %scalar.349.553 = fadd double %scalar.347.551, %scalar.348.552
  store double %scalar.349.553, ptr %value.553, align 8
  %scalar.350.554 = fmul double %load.3.237.1, %scalar.341.546
  store double %scalar.350.554, ptr %value.554, align 8
  %scalar.351.555 = fadd double %scalar.349.553, %scalar.350.554
  store double %scalar.351.555, ptr %value.555, align 8
  %scalar.352.556 = fadd double %scalar.345.549, %scalar.351.555
  store double %scalar.352.556, ptr %value.556, align 8
  %scalar.353.557 = fsub double %scalar.352.556, %scalar.345.549
  store double %scalar.353.557, ptr %value.557, align 8
  %scalar.354.558 = fsub double %scalar.351.555, %scalar.353.557
  store double %scalar.354.558, ptr %value.558, align 8
  %scalar.355.53 = fadd double %scalar.352.556, %scalar.354.558
  store double %scalar.355.53, ptr %out.31, align 8
  %load.356.559.0 = load double, ptr %arg.17, align 8
  %scalar.356.559 = fadd double %load.356.559.0, %scalar.352.556
  store double %scalar.356.559, ptr %value.559, align 8
  %scalar.357.560 = fsub double %scalar.356.559, %load.356.559.0
  store double %scalar.357.560, ptr %value.560, align 8
  %scalar.358.561 = fsub double %scalar.356.559, %scalar.357.560
  store double %scalar.358.561, ptr %value.561, align 8
  %scalar.359.562 = fsub double %load.356.559.0, %scalar.358.561
  store double %scalar.359.562, ptr %value.562, align 8
  %scalar.360.563 = fsub double %scalar.352.556, %scalar.357.560
  store double %scalar.360.563, ptr %value.563, align 8
  %scalar.361.564 = fadd double %scalar.359.562, %scalar.360.563
  store double %scalar.361.564, ptr %value.564, align 8
  %load.362.565.1 = load double, ptr %arg.40, align 8
  %scalar.362.565 = fadd double %scalar.361.564, %load.362.565.1
  store double %scalar.362.565, ptr %value.565, align 8
  %scalar.363.566 = fadd double %scalar.362.565, %scalar.354.558
  store double %scalar.363.566, ptr %value.566, align 8
  %scalar.364.567 = fadd double %scalar.356.559, %scalar.363.566
  store double %scalar.364.567, ptr %value.567, align 8
  %scalar.365.568 = fsub double %scalar.364.567, %scalar.356.559
  store double %scalar.365.568, ptr %value.568, align 8
  %scalar.366.569 = fsub double %scalar.363.566, %scalar.365.568
  store double %scalar.366.569, ptr %value.569, align 8
  %scalar.367.54 = fadd double %scalar.364.567, %scalar.366.569
  store double %scalar.367.54, ptr %out.32, align 8
  %scalar.368.570 = fmul double %load.0.234.1, %scalar.364.567
  store double %scalar.368.570, ptr %value.570, align 8
  %scalar.369.571 = fneg double %scalar.368.570
  store double %scalar.369.571, ptr %value.571, align 8
  %scalar.370.572 = call double @llvm.fma.f64(double %load.0.234.1, double %scalar.364.567, double %scalar.369.571)
  store double %scalar.370.572, ptr %value.572, align 8
  %scalar.371.573 = fmul double %load.0.234.1, %scalar.366.569
  store double %scalar.371.573, ptr %value.573, align 8
  %scalar.372.574 = fadd double %scalar.370.572, %scalar.371.573
  store double %scalar.372.574, ptr %value.574, align 8
  %scalar.373.575 = fmul double %load.3.237.1, %scalar.364.567
  store double %scalar.373.575, ptr %value.575, align 8
  %scalar.374.576 = fadd double %scalar.372.574, %scalar.373.575
  store double %scalar.374.576, ptr %value.576, align 8
  %scalar.375.577 = fadd double %scalar.368.570, %scalar.374.576
  store double %scalar.375.577, ptr %value.577, align 8
  %scalar.376.578 = fsub double %scalar.375.577, %scalar.368.570
  store double %scalar.376.578, ptr %value.578, align 8
  %scalar.377.579 = fsub double %scalar.374.576, %scalar.376.578
  store double %scalar.377.579, ptr %value.579, align 8
  %scalar.378.55 = fadd double %scalar.375.577, %scalar.377.579
  store double %scalar.378.55, ptr %out.33, align 8
  %load.379.580.0 = load double, ptr %arg.18, align 8
  %scalar.379.580 = fadd double %load.379.580.0, %scalar.375.577
  store double %scalar.379.580, ptr %value.580, align 8
  %scalar.380.581 = fsub double %scalar.379.580, %load.379.580.0
  store double %scalar.380.581, ptr %value.581, align 8
  %scalar.381.582 = fsub double %scalar.379.580, %scalar.380.581
  store double %scalar.381.582, ptr %value.582, align 8
  %scalar.382.583 = fsub double %load.379.580.0, %scalar.381.582
  store double %scalar.382.583, ptr %value.583, align 8
  %scalar.383.584 = fsub double %scalar.375.577, %scalar.380.581
  store double %scalar.383.584, ptr %value.584, align 8
  %scalar.384.585 = fadd double %scalar.382.583, %scalar.383.584
  store double %scalar.384.585, ptr %value.585, align 8
  %load.385.586.1 = load double, ptr %arg.41, align 8
  %scalar.385.586 = fadd double %scalar.384.585, %load.385.586.1
  store double %scalar.385.586, ptr %value.586, align 8
  %scalar.386.587 = fadd double %scalar.385.586, %scalar.377.579
  store double %scalar.386.587, ptr %value.587, align 8
  %scalar.387.588 = fadd double %scalar.379.580, %scalar.386.587
  store double %scalar.387.588, ptr %value.588, align 8
  %scalar.388.589 = fsub double %scalar.387.588, %scalar.379.580
  store double %scalar.388.589, ptr %value.589, align 8
  %scalar.389.590 = fsub double %scalar.386.587, %scalar.388.589
  store double %scalar.389.590, ptr %value.590, align 8
  %scalar.390.56 = fadd double %scalar.387.588, %scalar.389.590
  store double %scalar.390.56, ptr %out.34, align 8
  %scalar.391.591 = fmul double %load.0.234.1, %scalar.387.588
  store double %scalar.391.591, ptr %value.591, align 8
  %scalar.392.592 = fneg double %scalar.391.591
  store double %scalar.392.592, ptr %value.592, align 8
  %scalar.393.593 = call double @llvm.fma.f64(double %load.0.234.1, double %scalar.387.588, double %scalar.392.592)
  store double %scalar.393.593, ptr %value.593, align 8
  %scalar.394.594 = fmul double %load.0.234.1, %scalar.389.590
  store double %scalar.394.594, ptr %value.594, align 8
  %scalar.395.595 = fadd double %scalar.393.593, %scalar.394.594
  store double %scalar.395.595, ptr %value.595, align 8
  %scalar.396.596 = fmul double %load.3.237.1, %scalar.387.588
  store double %scalar.396.596, ptr %value.596, align 8
  %scalar.397.597 = fadd double %scalar.395.595, %scalar.396.596
  store double %scalar.397.597, ptr %value.597, align 8
  %scalar.398.598 = fadd double %scalar.391.591, %scalar.397.597
  store double %scalar.398.598, ptr %value.598, align 8
  %scalar.399.599 = fsub double %scalar.398.598, %scalar.391.591
  store double %scalar.399.599, ptr %value.599, align 8
  %scalar.400.600 = fsub double %scalar.397.597, %scalar.399.599
  store double %scalar.400.600, ptr %value.600, align 8
  %scalar.401.57 = fadd double %scalar.398.598, %scalar.400.600
  store double %scalar.401.57, ptr %out.35, align 8
  %load.402.601.0 = load double, ptr %arg.19, align 8
  %scalar.402.601 = fadd double %load.402.601.0, %scalar.398.598
  store double %scalar.402.601, ptr %value.601, align 8
  %scalar.403.602 = fsub double %scalar.402.601, %load.402.601.0
  store double %scalar.403.602, ptr %value.602, align 8
  %scalar.404.603 = fsub double %scalar.402.601, %scalar.403.602
  store double %scalar.404.603, ptr %value.603, align 8
  %scalar.405.604 = fsub double %load.402.601.0, %scalar.404.603
  store double %scalar.405.604, ptr %value.604, align 8
  %scalar.406.605 = fsub double %scalar.398.598, %scalar.403.602
  store double %scalar.406.605, ptr %value.605, align 8
  %scalar.407.606 = fadd double %scalar.405.604, %scalar.406.605
  store double %scalar.407.606, ptr %value.606, align 8
  %load.408.607.1 = load double, ptr %arg.42, align 8
  %scalar.408.607 = fadd double %scalar.407.606, %load.408.607.1
  store double %scalar.408.607, ptr %value.607, align 8
  %scalar.409.608 = fadd double %scalar.408.607, %scalar.400.600
  store double %scalar.409.608, ptr %value.608, align 8
  %scalar.410.609 = fadd double %scalar.402.601, %scalar.409.608
  store double %scalar.410.609, ptr %value.609, align 8
  %scalar.411.610 = fsub double %scalar.410.609, %scalar.402.601
  store double %scalar.411.610, ptr %value.610, align 8
  %scalar.412.611 = fsub double %scalar.409.608, %scalar.411.610
  store double %scalar.412.611, ptr %value.611, align 8
  %scalar.413.58 = fadd double %scalar.410.609, %scalar.412.611
  store double %scalar.413.58, ptr %out.36, align 8
  %scalar.414.612 = fmul double %load.0.234.1, %scalar.410.609
  store double %scalar.414.612, ptr %value.612, align 8
  %scalar.415.613 = fneg double %scalar.414.612
  store double %scalar.415.613, ptr %value.613, align 8
  %scalar.416.614 = call double @llvm.fma.f64(double %load.0.234.1, double %scalar.410.609, double %scalar.415.613)
  store double %scalar.416.614, ptr %value.614, align 8
  %scalar.417.615 = fmul double %load.0.234.1, %scalar.412.611
  store double %scalar.417.615, ptr %value.615, align 8
  %scalar.418.616 = fadd double %scalar.416.614, %scalar.417.615
  store double %scalar.418.616, ptr %value.616, align 8
  %scalar.419.617 = fmul double %load.3.237.1, %scalar.410.609
  store double %scalar.419.617, ptr %value.617, align 8
  %scalar.420.618 = fadd double %scalar.418.616, %scalar.419.617
  store double %scalar.420.618, ptr %value.618, align 8
  %scalar.421.619 = fadd double %scalar.414.612, %scalar.420.618
  store double %scalar.421.619, ptr %value.619, align 8
  %scalar.422.620 = fsub double %scalar.421.619, %scalar.414.612
  store double %scalar.422.620, ptr %value.620, align 8
  %scalar.423.621 = fsub double %scalar.420.618, %scalar.422.620
  store double %scalar.423.621, ptr %value.621, align 8
  %scalar.424.59 = fadd double %scalar.421.619, %scalar.423.621
  store double %scalar.424.59, ptr %out.37, align 8
  %load.425.622.0 = load double, ptr %arg.20, align 8
  %scalar.425.622 = fadd double %load.425.622.0, %scalar.421.619
  store double %scalar.425.622, ptr %value.622, align 8
  %scalar.426.623 = fsub double %scalar.425.622, %load.425.622.0
  store double %scalar.426.623, ptr %value.623, align 8
  %scalar.427.624 = fsub double %scalar.425.622, %scalar.426.623
  store double %scalar.427.624, ptr %value.624, align 8
  %scalar.428.625 = fsub double %load.425.622.0, %scalar.427.624
  store double %scalar.428.625, ptr %value.625, align 8
  %scalar.429.626 = fsub double %scalar.421.619, %scalar.426.623
  store double %scalar.429.626, ptr %value.626, align 8
  %scalar.430.627 = fadd double %scalar.428.625, %scalar.429.626
  store double %scalar.430.627, ptr %value.627, align 8
  %load.431.628.1 = load double, ptr %arg.43, align 8
  %scalar.431.628 = fadd double %scalar.430.627, %load.431.628.1
  store double %scalar.431.628, ptr %value.628, align 8
  %scalar.432.629 = fadd double %scalar.431.628, %scalar.423.621
  store double %scalar.432.629, ptr %value.629, align 8
  %scalar.433.630 = fadd double %scalar.425.622, %scalar.432.629
  store double %scalar.433.630, ptr %value.630, align 8
  %scalar.434.631 = fsub double %scalar.433.630, %scalar.425.622
  store double %scalar.434.631, ptr %value.631, align 8
  %scalar.435.632 = fsub double %scalar.432.629, %scalar.434.631
  store double %scalar.435.632, ptr %value.632, align 8
  %scalar.436.60 = fadd double %scalar.433.630, %scalar.435.632
  store double %scalar.436.60, ptr %out.38, align 8
  %scalar.437.633 = fmul double %load.0.234.1, %scalar.433.630
  store double %scalar.437.633, ptr %value.633, align 8
  %scalar.438.634 = fneg double %scalar.437.633
  store double %scalar.438.634, ptr %value.634, align 8
  %scalar.439.635 = call double @llvm.fma.f64(double %load.0.234.1, double %scalar.433.630, double %scalar.438.634)
  store double %scalar.439.635, ptr %value.635, align 8
  %scalar.440.636 = fmul double %load.0.234.1, %scalar.435.632
  store double %scalar.440.636, ptr %value.636, align 8
  %scalar.441.637 = fadd double %scalar.439.635, %scalar.440.636
  store double %scalar.441.637, ptr %value.637, align 8
  %scalar.442.638 = fmul double %load.3.237.1, %scalar.433.630
  store double %scalar.442.638, ptr %value.638, align 8
  %scalar.443.639 = fadd double %scalar.441.637, %scalar.442.638
  store double %scalar.443.639, ptr %value.639, align 8
  %scalar.444.640 = fadd double %scalar.437.633, %scalar.443.639
  store double %scalar.444.640, ptr %value.640, align 8
  %scalar.445.641 = fsub double %scalar.444.640, %scalar.437.633
  store double %scalar.445.641, ptr %value.641, align 8
  %scalar.446.642 = fsub double %scalar.443.639, %scalar.445.641
  store double %scalar.446.642, ptr %value.642, align 8
  %scalar.447.61 = fadd double %scalar.444.640, %scalar.446.642
  store double %scalar.447.61, ptr %out.39, align 8
  %load.448.643.0 = load double, ptr %arg.21, align 8
  %scalar.448.643 = fadd double %load.448.643.0, %scalar.444.640
  store double %scalar.448.643, ptr %value.643, align 8
  %scalar.449.644 = fsub double %scalar.448.643, %load.448.643.0
  store double %scalar.449.644, ptr %value.644, align 8
  %scalar.450.645 = fsub double %scalar.448.643, %scalar.449.644
  store double %scalar.450.645, ptr %value.645, align 8
  %scalar.451.646 = fsub double %load.448.643.0, %scalar.450.645
  store double %scalar.451.646, ptr %value.646, align 8
  %scalar.452.647 = fsub double %scalar.444.640, %scalar.449.644
  store double %scalar.452.647, ptr %value.647, align 8
  %scalar.453.648 = fadd double %scalar.451.646, %scalar.452.647
  store double %scalar.453.648, ptr %value.648, align 8
  %load.454.649.1 = load double, ptr %arg.44, align 8
  %scalar.454.649 = fadd double %scalar.453.648, %load.454.649.1
  store double %scalar.454.649, ptr %value.649, align 8
  %scalar.455.650 = fadd double %scalar.454.649, %scalar.446.642
  store double %scalar.455.650, ptr %value.650, align 8
  %scalar.456.651 = fadd double %scalar.448.643, %scalar.455.650
  store double %scalar.456.651, ptr %value.651, align 8
  %scalar.457.652 = fsub double %scalar.456.651, %scalar.448.643
  store double %scalar.457.652, ptr %value.652, align 8
  %scalar.458.653 = fsub double %scalar.455.650, %scalar.457.652
  store double %scalar.458.653, ptr %value.653, align 8
  %scalar.459.62 = fadd double %scalar.456.651, %scalar.458.653
  store double %scalar.459.62, ptr %out.40, align 8
  %load.460.654.0 = load double, ptr %arg.22, align 8
  %scalar.460.654 = fmul double %load.460.654.0, %scalar.456.651
  store double %scalar.460.654, ptr %value.654, align 8
  %scalar.461.655 = fneg double %scalar.460.654
  store double %scalar.461.655, ptr %value.655, align 8
  %scalar.462.656 = call double @llvm.fma.f64(double %load.460.654.0, double %scalar.456.651, double %scalar.461.655)
  store double %scalar.462.656, ptr %value.656, align 8
  %scalar.463.657 = fmul double %load.460.654.0, %scalar.458.653
  store double %scalar.463.657, ptr %value.657, align 8
  %scalar.464.658 = fadd double %scalar.462.656, %scalar.463.657
  store double %scalar.464.658, ptr %value.658, align 8
  %load.465.659.0 = load double, ptr %arg.45, align 8
  %scalar.465.659 = fmul double %load.465.659.0, %scalar.456.651
  store double %scalar.465.659, ptr %value.659, align 8
  %scalar.466.660 = fadd double %scalar.464.658, %scalar.465.659
  store double %scalar.466.660, ptr %value.660, align 8
  %scalar.467.661 = fadd double %scalar.460.654, %scalar.466.660
  store double %scalar.467.661, ptr %value.661, align 8
  %scalar.468.662 = fsub double %scalar.467.661, %scalar.460.654
  store double %scalar.468.662, ptr %value.662, align 8
  %scalar.469.663 = fsub double %scalar.466.660, %scalar.468.662
  store double %scalar.469.663, ptr %value.663, align 8
  %scalar.470.63 = fadd double %scalar.467.661, %scalar.469.663
  store double %scalar.470.63, ptr %out.0, align 8
  ret void
}

define void @__ssa_atan_core_pack__atan_core(ptr noalias %arg.0, ptr noalias %arg.1, ptr noalias %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr noalias %arg.11, ptr noalias %arg.12, ptr noalias %arg.13, ptr noalias %arg.14, ptr noalias %arg.15, ptr noalias %arg.16, ptr noalias %arg.17, ptr noalias %arg.18, ptr noalias %arg.19, ptr noalias %arg.20, ptr %arg.21, ptr %arg.22, ptr noalias %arg.23, ptr noalias %arg.24, ptr noalias %arg.25, ptr noalias %arg.26, ptr noalias %arg.27, ptr noalias %arg.28, ptr noalias %arg.29, ptr noalias %arg.30, ptr noalias %arg.31, ptr noalias %arg.32, ptr noalias %arg.33, ptr noalias %arg.34, ptr noalias %arg.35, ptr noalias %arg.36, ptr noalias %arg.37, ptr noalias %arg.38, ptr noalias %arg.39, ptr noalias %arg.40, ptr noalias %arg.41, ptr noalias %arg.42, ptr noalias %arg.43, ptr %arg.44, ptr %arg.45, ptr %out.0) {
entry:
  %value.145 = alloca i32, i64 1, align 8
  %value.143 = alloca i32, i64 1, align 8
  %value.141 = alloca i32, i64 1, align 8
  %value.139 = alloca i32, i64 1, align 8
  %value.137 = alloca i32, i64 1, align 8
  %value.135 = alloca i32, i64 1, align 8
  %value.133 = alloca i32, i64 1, align 8
  %value.131 = alloca i32, i64 1, align 8
  %value.129 = alloca i32, i64 1, align 8
  %value.127 = alloca i32, i64 1, align 8
  %value.125 = alloca i32, i64 1, align 8
  %value.123 = alloca i32, i64 1, align 8
  %value.121 = alloca i32, i64 1, align 8
  %value.119 = alloca i32, i64 1, align 8
  %value.117 = alloca i32, i64 1, align 8
  %value.115 = alloca i32, i64 1, align 8
  %value.113 = alloca i32, i64 1, align 8
  %value.111 = alloca i32, i64 1, align 8
  %value.109 = alloca i32, i64 1, align 8
  %value.107 = alloca i32, i64 1, align 8
  %value.105 = alloca i32, i64 1, align 8
  %value.103 = alloca i32, i64 1, align 8
  %value.101 = alloca i32, i64 1, align 8
  %value.99 = alloca i32, i64 1, align 8
  %value.97 = alloca i32, i64 1, align 8
  %value.95 = alloca i32, i64 1, align 8
  %value.93 = alloca i32, i64 1, align 8
  %value.91 = alloca i32, i64 1, align 8
  %value.89 = alloca i32, i64 1, align 8
  %value.87 = alloca i32, i64 1, align 8
  %value.85 = alloca i32, i64 1, align 8
  %value.83 = alloca i32, i64 1, align 8
  %value.81 = alloca i32, i64 1, align 8
  %value.79 = alloca i32, i64 1, align 8
  %value.77 = alloca i32, i64 1, align 8
  %value.75 = alloca i32, i64 1, align 8
  %value.73 = alloca i32, i64 1, align 8
  %value.71 = alloca i32, i64 1, align 8
  %value.69 = alloca i32, i64 1, align 8
  %value.67 = alloca i32, i64 1, align 8
  %value.65 = alloca i64, i64 1, align 8
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
  %value.54 = alloca double, i64 1, align 8
  %value.55 = alloca double, i64 1, align 8
  %value.56 = alloca double, i64 1, align 8
  %value.57 = alloca double, i64 1, align 8
  %value.58 = alloca double, i64 1, align 8
  %value.59 = alloca double, i64 1, align 8
  %value.60 = alloca double, i64 1, align 8
  %value.61 = alloca double, i64 1, align 8
  %value.62 = alloca double, i64 1, align 8
  store i32 40, ptr %value.145, align 4
  store i32 39, ptr %value.143, align 4
  store i32 38, ptr %value.141, align 4
  store i32 37, ptr %value.139, align 4
  store i32 36, ptr %value.137, align 4
  store i32 35, ptr %value.135, align 4
  store i32 34, ptr %value.133, align 4
  store i32 33, ptr %value.131, align 4
  store i32 32, ptr %value.129, align 4
  store i32 31, ptr %value.127, align 4
  store i32 30, ptr %value.125, align 4
  store i32 29, ptr %value.123, align 4
  store i32 28, ptr %value.121, align 4
  store i32 27, ptr %value.119, align 4
  store i32 26, ptr %value.117, align 4
  store i32 25, ptr %value.115, align 4
  store i32 24, ptr %value.113, align 4
  store i32 23, ptr %value.111, align 4
  store i32 22, ptr %value.109, align 4
  store i32 21, ptr %value.107, align 4
  store i32 20, ptr %value.105, align 4
  store i32 19, ptr %value.103, align 4
  store i32 18, ptr %value.101, align 4
  store i32 17, ptr %value.99, align 4
  store i32 16, ptr %value.97, align 4
  store i32 15, ptr %value.95, align 4
  store i32 14, ptr %value.93, align 4
  store i32 13, ptr %value.91, align 4
  store i32 12, ptr %value.89, align 4
  store i32 11, ptr %value.87, align 4
  store i32 10, ptr %value.85, align 4
  store i32 9, ptr %value.83, align 4
  store i32 8, ptr %value.81, align 4
  store i32 7, ptr %value.79, align 4
  store i32 6, ptr %value.77, align 4
  store i32 5, ptr %value.75, align 4
  store i32 4, ptr %value.73, align 4
  store i32 3, ptr %value.71, align 4
  store i32 2, ptr %value.69, align 4
  store i32 1, ptr %value.67, align 4
  store i64 0, ptr %value.65, align 8
  call void @__ssa_atan_core_pack__atan_core__planned_region_0(ptr %arg.13, ptr %arg.21, ptr %arg.11, ptr %arg.10, ptr %arg.9, ptr %arg.8, ptr %arg.7, ptr %arg.6, ptr %arg.5, ptr %arg.4, ptr %arg.3, ptr %arg.2, ptr %arg.20, ptr %arg.19, ptr %arg.18, ptr %arg.17, ptr %arg.16, ptr %arg.15, ptr %arg.14, ptr %arg.12, ptr %arg.1, ptr %arg.0, ptr %arg.22, ptr %arg.36, ptr %arg.44, ptr %arg.34, ptr %arg.33, ptr %arg.32, ptr %arg.31, ptr %arg.30, ptr %arg.29, ptr %arg.28, ptr %arg.27, ptr %arg.26, ptr %arg.25, ptr %arg.43, ptr %arg.42, ptr %arg.41, ptr %arg.40, ptr %arg.39, ptr %arg.38, ptr %arg.37, ptr %arg.35, ptr %arg.24, ptr %arg.23, ptr %arg.45, ptr %out.0, ptr %value.23, ptr %value.24, ptr %value.25, ptr %value.26, ptr %value.27, ptr %value.28, ptr %value.29, ptr %value.30, ptr %value.31, ptr %value.32, ptr %value.33, ptr %value.34, ptr %value.35, ptr %value.36, ptr %value.37, ptr %value.38, ptr %value.39, ptr %value.40, ptr %value.41, ptr %value.42, ptr %value.43, ptr %value.44, ptr %value.45, ptr %value.46, ptr %value.47, ptr %value.48, ptr %value.49, ptr %value.50, ptr %value.51, ptr %value.52, ptr %value.53, ptr %value.54, ptr %value.55, ptr %value.56, ptr %value.57, ptr %value.58, ptr %value.59, ptr %value.60, ptr %value.61, ptr %value.62)
  ret void
}

define void @atan_core_pack__atan_core_pack(ptr %buffers, ptr %extents) {
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
  call void @__ssa_atan_core_pack__atan_core_pack(ptr %public.0, ptr %public.1, ptr %public.2, ptr %public.3, ptr %public.4, ptr %public.5, ptr %public.6, ptr %public.7, ptr %public.8, ptr %public.9, ptr %public.10, ptr %public.11, ptr %public.12, ptr %public.13, ptr %public.14, ptr %public.15, ptr %public.16, ptr %public.17, ptr %public.18, ptr %public.19, ptr %public.20, ptr %public.21, ptr %public.22, ptr %public.23, ptr %public.24, ptr %public.25, ptr %public.26, ptr %public.27, ptr %public.28, ptr %public.29, ptr %public.30, ptr %public.31, ptr %public.32, ptr %public.33, ptr %public.34, ptr %public.35, ptr %public.36, ptr %public.37, ptr %public.38, ptr %public.39, ptr %public.40, ptr %public.41, ptr %public.42, ptr %public.43, ptr %public.44, ptr %public.2)
  ret void
}
