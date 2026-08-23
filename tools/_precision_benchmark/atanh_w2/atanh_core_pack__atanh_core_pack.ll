source_filename = "turing.ssa-llvm.atanh_core_pack__atanh_core_pack"

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-windows-gnu"

declare double @llvm.fma.f64(double, double, double)

define void @__ssa_atanh_core_pack__atanh_core_pack__planned_region_0(ptr noalias %arg.0, ptr %arg.1, ptr %out.0, ptr %out.1) {
entry:
  %value.264 = alloca i32, i64 1, align 8
  %value.265 = alloca i32, i64 1, align 8
  %value.267 = alloca double, i64 1, align 8
  %value.268 = alloca i32, i64 1, align 8
  %value.269 = alloca i32, i64 1, align 8
  %value.270 = alloca i32, i64 1, align 8
  %value.271 = alloca i32, i64 1, align 8
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
  %load.0.46.0 = load i32, ptr %arg.1, align 4
  %address.0.46 = getelementptr double, ptr %arg.0, i32 %load.0.46.0
  store i32 2, ptr %value.264, align 4
  %scalar.2.265 = mul i32 %load.0.46.0, 2
  store i32 %scalar.2.265, ptr %value.265, align 4
  %address.3.266 = getelementptr double, ptr %arg.0, i32 %scalar.2.265
  %pinned.load.4.267 = load double, ptr %address.3.266, align 8
  store double %pinned.load.4.267, ptr %value.267, align 8
  store i32 2, ptr %value.268, align 4
  %scalar.6.269 = mul i32 %load.0.46.0, 2
  store i32 %scalar.6.269, ptr %value.269, align 4
  store i32 1, ptr %value.270, align 4
  %scalar.8.271 = add i32 %scalar.6.269, 1
  store i32 %scalar.8.271, ptr %value.271, align 4
  %address.9.272 = getelementptr double, ptr %arg.0, i32 %scalar.8.271
  %pinned.load.10.273 = load double, ptr %address.9.272, align 8
  store double %pinned.load.10.273, ptr %value.273, align 8
  %load.11.274.0 = load double, ptr %value.267, align 8
  %scalar.11.274 = fmul double %load.11.274.0, %load.11.274.0
  store double %scalar.11.274, ptr %value.274, align 8
  %scalar.12.275 = fneg double %scalar.11.274
  store double %scalar.12.275, ptr %value.275, align 8
  %scalar.13.276 = call double @llvm.fma.f64(double %load.11.274.0, double %load.11.274.0, double %scalar.12.275)
  store double %scalar.13.276, ptr %value.276, align 8
  %load.14.277.1 = load double, ptr %value.273, align 8
  %scalar.14.277 = fmul double %load.11.274.0, %load.14.277.1
  store double %scalar.14.277, ptr %value.277, align 8
  %scalar.15.278 = fadd double %scalar.13.276, %scalar.14.277
  store double %scalar.15.278, ptr %value.278, align 8
  %scalar.16.279 = fmul double %load.14.277.1, %load.11.274.0
  store double %scalar.16.279, ptr %value.279, align 8
  %scalar.17.280 = fadd double %scalar.15.278, %scalar.16.279
  store double %scalar.17.280, ptr %value.280, align 8
  %scalar.18.281 = fadd double %scalar.11.274, %scalar.17.280
  store double %scalar.18.281, ptr %value.281, align 8
  %scalar.19.282 = fsub double %scalar.18.281, %scalar.11.274
  store double %scalar.19.282, ptr %value.282, align 8
  %scalar.20.283 = fsub double %scalar.17.280, %scalar.19.282
  store double %scalar.20.283, ptr %value.283, align 8
  %scalar.21.32 = fadd double %scalar.18.281, %scalar.20.283
  store double %scalar.21.32, ptr %out.1, align 8
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

define void @__ssa_atanh_core_pack__atanh_core_pack(ptr noalias %arg.0, ptr noalias %arg.1, ptr %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr noalias %arg.11, ptr noalias %arg.12, ptr noalias %arg.13, ptr noalias %arg.14, ptr noalias %arg.15, ptr noalias %arg.16, ptr noalias %arg.17, ptr noalias %arg.18, ptr noalias %arg.19, ptr noalias %arg.20, ptr noalias %arg.21, ptr noalias %arg.22, ptr noalias %arg.23, ptr noalias %arg.24, ptr noalias %arg.25, ptr noalias %arg.26, ptr noalias %arg.27, ptr noalias %arg.28, ptr noalias %arg.29, ptr noalias %arg.30, ptr noalias %arg.31, ptr noalias %arg.32, ptr noalias %arg.33, ptr noalias %arg.34, ptr noalias %arg.35, ptr noalias %arg.36, ptr noalias %arg.37, ptr noalias %arg.38, ptr noalias %arg.39, ptr noalias %arg.40, ptr noalias %arg.41, ptr noalias %arg.42, ptr noalias %arg.43, ptr noalias %arg.44, ptr noalias %arg.45, ptr noalias %arg.46, ptr noalias %arg.47, ptr noalias %arg.48, ptr noalias %arg.49, ptr noalias %arg.50, ptr noalias %arg.51, ptr noalias %arg.52, ptr noalias %arg.53, ptr noalias %arg.54, ptr %out.0) {
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
  call void @__ssa_atanh_core_pack__atanh_core(ptr %arg.3, ptr %arg.4, ptr %arg.5, ptr %arg.6, ptr %arg.7, ptr %arg.8, ptr %arg.9, ptr %arg.10, ptr %arg.11, ptr %arg.12, ptr %arg.13, ptr %arg.14, ptr %arg.15, ptr %arg.16, ptr %arg.17, ptr %arg.18, ptr %arg.19, ptr %arg.20, ptr %arg.21, ptr %arg.22, ptr %arg.23, ptr %arg.24, ptr %arg.25, ptr %arg.26, ptr %arg.27, ptr %arg.28, ptr %value.32, ptr %value.31, ptr %arg.29, ptr %arg.30, ptr %arg.31, ptr %arg.32, ptr %arg.33, ptr %arg.34, ptr %arg.35, ptr %arg.36, ptr %arg.37, ptr %arg.38, ptr %arg.39, ptr %arg.40, ptr %arg.41, ptr %arg.42, ptr %arg.43, ptr %arg.44, ptr %arg.45, ptr %arg.46, ptr %arg.47, ptr %arg.48, ptr %arg.49, ptr %arg.50, ptr %arg.51, ptr %arg.52, ptr %arg.53, ptr %arg.54, ptr %value.32, ptr %value.31, ptr %value.33)
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

define void @__ssa_atanh_core_pack__atanh_core__planned_region_0(ptr noalias %arg.0, ptr %arg.1, ptr noalias %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr noalias %arg.11, ptr noalias %arg.12, ptr noalias %arg.13, ptr noalias %arg.14, ptr noalias %arg.15, ptr noalias %arg.16, ptr noalias %arg.17, ptr noalias %arg.18, ptr noalias %arg.19, ptr noalias %arg.20, ptr noalias %arg.21, ptr noalias %arg.22, ptr noalias %arg.23, ptr noalias %arg.24, ptr noalias %arg.25, ptr noalias %arg.26, ptr %arg.27, ptr noalias %arg.28, ptr %arg.29, ptr noalias %arg.30, ptr noalias %arg.31, ptr noalias %arg.32, ptr noalias %arg.33, ptr noalias %arg.34, ptr noalias %arg.35, ptr noalias %arg.36, ptr noalias %arg.37, ptr noalias %arg.38, ptr noalias %arg.39, ptr noalias %arg.40, ptr noalias %arg.41, ptr noalias %arg.42, ptr noalias %arg.43, ptr noalias %arg.44, ptr noalias %arg.45, ptr noalias %arg.46, ptr noalias %arg.47, ptr noalias %arg.48, ptr noalias %arg.49, ptr noalias %arg.50, ptr noalias %arg.51, ptr noalias %arg.52, ptr noalias %arg.53, ptr noalias %arg.54, ptr %arg.55, ptr %out.0, ptr %out.1, ptr %out.2, ptr %out.3, ptr %out.4, ptr %out.5, ptr %out.6, ptr %out.7, ptr %out.8, ptr %out.9, ptr %out.10, ptr %out.11, ptr %out.12, ptr %out.13, ptr %out.14, ptr %out.15, ptr %out.16, ptr %out.17, ptr %out.18, ptr %out.19, ptr %out.20, ptr %out.21, ptr %out.22, ptr %out.23, ptr %out.24, ptr %out.25, ptr %out.26, ptr %out.27, ptr %out.28, ptr %out.29, ptr %out.30, ptr %out.31, ptr %out.32, ptr %out.33, ptr %out.34, ptr %out.35, ptr %out.36, ptr %out.37, ptr %out.38, ptr %out.39, ptr %out.40, ptr %out.41, ptr %out.42, ptr %out.43, ptr %out.44, ptr %out.45, ptr %out.46, ptr %out.47, ptr %out.48, ptr %out.49, ptr %out.50) {
entry:
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
  %value.664 = alloca double, i64 1, align 8
  %value.665 = alloca double, i64 1, align 8
  %value.666 = alloca double, i64 1, align 8
  %value.667 = alloca double, i64 1, align 8
  %value.668 = alloca double, i64 1, align 8
  %value.669 = alloca double, i64 1, align 8
  %value.670 = alloca double, i64 1, align 8
  %value.671 = alloca double, i64 1, align 8
  %value.672 = alloca double, i64 1, align 8
  %value.673 = alloca double, i64 1, align 8
  %value.674 = alloca double, i64 1, align 8
  %value.675 = alloca double, i64 1, align 8
  %value.676 = alloca double, i64 1, align 8
  %value.677 = alloca double, i64 1, align 8
  %value.678 = alloca double, i64 1, align 8
  %value.679 = alloca double, i64 1, align 8
  %value.680 = alloca double, i64 1, align 8
  %value.681 = alloca double, i64 1, align 8
  %value.682 = alloca double, i64 1, align 8
  %value.683 = alloca double, i64 1, align 8
  %value.684 = alloca double, i64 1, align 8
  %value.685 = alloca double, i64 1, align 8
  %value.686 = alloca double, i64 1, align 8
  %value.687 = alloca double, i64 1, align 8
  %value.688 = alloca double, i64 1, align 8
  %value.689 = alloca double, i64 1, align 8
  %value.690 = alloca double, i64 1, align 8
  %value.691 = alloca double, i64 1, align 8
  %value.692 = alloca double, i64 1, align 8
  %value.693 = alloca double, i64 1, align 8
  %value.694 = alloca double, i64 1, align 8
  %value.695 = alloca double, i64 1, align 8
  %value.696 = alloca double, i64 1, align 8
  %value.697 = alloca double, i64 1, align 8
  %value.698 = alloca double, i64 1, align 8
  %value.699 = alloca double, i64 1, align 8
  %value.700 = alloca double, i64 1, align 8
  %value.701 = alloca double, i64 1, align 8
  %value.702 = alloca double, i64 1, align 8
  %value.703 = alloca double, i64 1, align 8
  %value.704 = alloca double, i64 1, align 8
  %value.705 = alloca double, i64 1, align 8
  %value.706 = alloca double, i64 1, align 8
  %value.707 = alloca double, i64 1, align 8
  %value.708 = alloca double, i64 1, align 8
  %value.709 = alloca double, i64 1, align 8
  %value.710 = alloca double, i64 1, align 8
  %value.711 = alloca double, i64 1, align 8
  %value.712 = alloca double, i64 1, align 8
  %value.713 = alloca double, i64 1, align 8
  %value.714 = alloca double, i64 1, align 8
  %value.715 = alloca double, i64 1, align 8
  %value.716 = alloca double, i64 1, align 8
  %value.717 = alloca double, i64 1, align 8
  %value.718 = alloca double, i64 1, align 8
  %value.719 = alloca double, i64 1, align 8
  %value.720 = alloca double, i64 1, align 8
  %value.721 = alloca double, i64 1, align 8
  %value.722 = alloca double, i64 1, align 8
  %value.723 = alloca double, i64 1, align 8
  %value.724 = alloca double, i64 1, align 8
  %value.725 = alloca double, i64 1, align 8
  %value.726 = alloca double, i64 1, align 8
  %value.727 = alloca double, i64 1, align 8
  %value.728 = alloca double, i64 1, align 8
  %value.729 = alloca double, i64 1, align 8
  %value.730 = alloca double, i64 1, align 8
  %value.731 = alloca double, i64 1, align 8
  %value.732 = alloca double, i64 1, align 8
  %value.733 = alloca double, i64 1, align 8
  %value.734 = alloca double, i64 1, align 8
  %value.735 = alloca double, i64 1, align 8
  %value.736 = alloca double, i64 1, align 8
  %value.737 = alloca double, i64 1, align 8
  %value.738 = alloca double, i64 1, align 8
  %value.739 = alloca double, i64 1, align 8
  %value.740 = alloca double, i64 1, align 8
  %value.741 = alloca double, i64 1, align 8
  %value.742 = alloca double, i64 1, align 8
  %value.743 = alloca double, i64 1, align 8
  %value.744 = alloca double, i64 1, align 8
  %value.745 = alloca double, i64 1, align 8
  %value.746 = alloca double, i64 1, align 8
  %value.747 = alloca double, i64 1, align 8
  %value.748 = alloca double, i64 1, align 8
  %value.749 = alloca double, i64 1, align 8
  %value.750 = alloca double, i64 1, align 8
  %value.751 = alloca double, i64 1, align 8
  %value.752 = alloca double, i64 1, align 8
  %value.753 = alloca double, i64 1, align 8
  %value.754 = alloca double, i64 1, align 8
  %value.755 = alloca double, i64 1, align 8
  %value.756 = alloca double, i64 1, align 8
  %value.757 = alloca double, i64 1, align 8
  %value.758 = alloca double, i64 1, align 8
  %value.759 = alloca double, i64 1, align 8
  %value.760 = alloca double, i64 1, align 8
  %value.761 = alloca double, i64 1, align 8
  %value.762 = alloca double, i64 1, align 8
  %value.763 = alloca double, i64 1, align 8
  %value.764 = alloca double, i64 1, align 8
  %value.765 = alloca double, i64 1, align 8
  %value.766 = alloca double, i64 1, align 8
  %value.767 = alloca double, i64 1, align 8
  %value.768 = alloca double, i64 1, align 8
  %value.769 = alloca double, i64 1, align 8
  %value.770 = alloca double, i64 1, align 8
  %value.771 = alloca double, i64 1, align 8
  %value.772 = alloca double, i64 1, align 8
  %value.773 = alloca double, i64 1, align 8
  %value.774 = alloca double, i64 1, align 8
  %value.775 = alloca double, i64 1, align 8
  %value.776 = alloca double, i64 1, align 8
  %value.777 = alloca double, i64 1, align 8
  %value.778 = alloca double, i64 1, align 8
  %value.779 = alloca double, i64 1, align 8
  %value.780 = alloca double, i64 1, align 8
  %value.781 = alloca double, i64 1, align 8
  %value.782 = alloca double, i64 1, align 8
  %value.783 = alloca double, i64 1, align 8
  %value.784 = alloca double, i64 1, align 8
  %value.785 = alloca double, i64 1, align 8
  %value.786 = alloca double, i64 1, align 8
  %value.787 = alloca double, i64 1, align 8
  %value.788 = alloca double, i64 1, align 8
  %value.789 = alloca double, i64 1, align 8
  %value.790 = alloca double, i64 1, align 8
  %value.791 = alloca double, i64 1, align 8
  %value.792 = alloca double, i64 1, align 8
  %value.793 = alloca double, i64 1, align 8
  %value.794 = alloca double, i64 1, align 8
  %value.795 = alloca double, i64 1, align 8
  %value.796 = alloca double, i64 1, align 8
  %value.797 = alloca double, i64 1, align 8
  %value.798 = alloca double, i64 1, align 8
  %value.799 = alloca double, i64 1, align 8
  %value.800 = alloca double, i64 1, align 8
  %value.801 = alloca double, i64 1, align 8
  %value.802 = alloca double, i64 1, align 8
  %value.803 = alloca double, i64 1, align 8
  %value.804 = alloca double, i64 1, align 8
  %value.805 = alloca double, i64 1, align 8
  %value.806 = alloca double, i64 1, align 8
  %value.807 = alloca double, i64 1, align 8
  %value.808 = alloca double, i64 1, align 8
  %value.809 = alloca double, i64 1, align 8
  %value.810 = alloca double, i64 1, align 8
  %value.811 = alloca double, i64 1, align 8
  %value.812 = alloca double, i64 1, align 8
  %value.813 = alloca double, i64 1, align 8
  %value.814 = alloca double, i64 1, align 8
  %value.815 = alloca double, i64 1, align 8
  %value.816 = alloca double, i64 1, align 8
  %value.817 = alloca double, i64 1, align 8
  %value.818 = alloca double, i64 1, align 8
  %load.0.284.0 = load double, ptr %arg.0, align 8
  %load.0.284.1 = load double, ptr %arg.1, align 8
  %scalar.0.284 = fmul double %load.0.284.0, %load.0.284.1
  store double %scalar.0.284, ptr %value.284, align 8
  %scalar.1.285 = fneg double %scalar.0.284
  store double %scalar.1.285, ptr %value.285, align 8
  %scalar.2.286 = call double @llvm.fma.f64(double %load.0.284.0, double %load.0.284.1, double %scalar.1.285)
  store double %scalar.2.286, ptr %value.286, align 8
  %load.3.287.1 = load double, ptr %arg.29, align 8
  %scalar.3.287 = fmul double %load.0.284.0, %load.3.287.1
  store double %scalar.3.287, ptr %value.287, align 8
  %scalar.4.288 = fadd double %scalar.2.286, %scalar.3.287
  store double %scalar.4.288, ptr %value.288, align 8
  %load.5.289.0 = load double, ptr %arg.28, align 8
  %scalar.5.289 = fmul double %load.5.289.0, %load.0.284.1
  store double %scalar.5.289, ptr %value.289, align 8
  %scalar.6.290 = fadd double %scalar.4.288, %scalar.5.289
  store double %scalar.6.290, ptr %value.290, align 8
  %scalar.7.291 = fadd double %scalar.0.284, %scalar.6.290
  store double %scalar.7.291, ptr %value.291, align 8
  %scalar.8.292 = fsub double %scalar.7.291, %scalar.0.284
  store double %scalar.8.292, ptr %value.292, align 8
  %scalar.9.293 = fsub double %scalar.6.290, %scalar.8.292
  store double %scalar.9.293, ptr %value.293, align 8
  %scalar.10.28 = fadd double %scalar.7.291, %scalar.9.293
  store double %scalar.10.28, ptr %out.1, align 8
  %load.11.294.0 = load double, ptr %arg.2, align 8
  %scalar.11.294 = fadd double %load.11.294.0, %scalar.7.291
  store double %scalar.11.294, ptr %value.294, align 8
  %scalar.12.295 = fsub double %scalar.11.294, %load.11.294.0
  store double %scalar.12.295, ptr %value.295, align 8
  %scalar.13.296 = fsub double %scalar.11.294, %scalar.12.295
  store double %scalar.13.296, ptr %value.296, align 8
  %scalar.14.297 = fsub double %load.11.294.0, %scalar.13.296
  store double %scalar.14.297, ptr %value.297, align 8
  %scalar.15.298 = fsub double %scalar.7.291, %scalar.12.295
  store double %scalar.15.298, ptr %value.298, align 8
  %scalar.16.299 = fadd double %scalar.14.297, %scalar.15.298
  store double %scalar.16.299, ptr %value.299, align 8
  %load.17.300.1 = load double, ptr %arg.30, align 8
  %scalar.17.300 = fadd double %scalar.16.299, %load.17.300.1
  store double %scalar.17.300, ptr %value.300, align 8
  %scalar.18.301 = fadd double %scalar.17.300, %scalar.9.293
  store double %scalar.18.301, ptr %value.301, align 8
  %scalar.19.302 = fadd double %scalar.11.294, %scalar.18.301
  store double %scalar.19.302, ptr %value.302, align 8
  %scalar.20.303 = fsub double %scalar.19.302, %scalar.11.294
  store double %scalar.20.303, ptr %value.303, align 8
  %scalar.21.304 = fsub double %scalar.18.301, %scalar.20.303
  store double %scalar.21.304, ptr %value.304, align 8
  %scalar.22.29 = fadd double %scalar.19.302, %scalar.21.304
  store double %scalar.22.29, ptr %out.2, align 8
  %scalar.23.305 = fmul double %load.0.284.1, %scalar.19.302
  store double %scalar.23.305, ptr %value.305, align 8
  %scalar.24.306 = fneg double %scalar.23.305
  store double %scalar.24.306, ptr %value.306, align 8
  %scalar.25.307 = call double @llvm.fma.f64(double %load.0.284.1, double %scalar.19.302, double %scalar.24.306)
  store double %scalar.25.307, ptr %value.307, align 8
  %scalar.26.308 = fmul double %load.0.284.1, %scalar.21.304
  store double %scalar.26.308, ptr %value.308, align 8
  %scalar.27.309 = fadd double %scalar.25.307, %scalar.26.308
  store double %scalar.27.309, ptr %value.309, align 8
  %scalar.28.310 = fmul double %load.3.287.1, %scalar.19.302
  store double %scalar.28.310, ptr %value.310, align 8
  %scalar.29.311 = fadd double %scalar.27.309, %scalar.28.310
  store double %scalar.29.311, ptr %value.311, align 8
  %scalar.30.312 = fadd double %scalar.23.305, %scalar.29.311
  store double %scalar.30.312, ptr %value.312, align 8
  %scalar.31.313 = fsub double %scalar.30.312, %scalar.23.305
  store double %scalar.31.313, ptr %value.313, align 8
  %scalar.32.314 = fsub double %scalar.29.311, %scalar.31.313
  store double %scalar.32.314, ptr %value.314, align 8
  %scalar.33.30 = fadd double %scalar.30.312, %scalar.32.314
  store double %scalar.33.30, ptr %out.3, align 8
  %load.34.315.0 = load double, ptr %arg.3, align 8
  %scalar.34.315 = fadd double %load.34.315.0, %scalar.30.312
  store double %scalar.34.315, ptr %value.315, align 8
  %scalar.35.316 = fsub double %scalar.34.315, %load.34.315.0
  store double %scalar.35.316, ptr %value.316, align 8
  %scalar.36.317 = fsub double %scalar.34.315, %scalar.35.316
  store double %scalar.36.317, ptr %value.317, align 8
  %scalar.37.318 = fsub double %load.34.315.0, %scalar.36.317
  store double %scalar.37.318, ptr %value.318, align 8
  %scalar.38.319 = fsub double %scalar.30.312, %scalar.35.316
  store double %scalar.38.319, ptr %value.319, align 8
  %scalar.39.320 = fadd double %scalar.37.318, %scalar.38.319
  store double %scalar.39.320, ptr %value.320, align 8
  %load.40.321.1 = load double, ptr %arg.31, align 8
  %scalar.40.321 = fadd double %scalar.39.320, %load.40.321.1
  store double %scalar.40.321, ptr %value.321, align 8
  %scalar.41.322 = fadd double %scalar.40.321, %scalar.32.314
  store double %scalar.41.322, ptr %value.322, align 8
  %scalar.42.323 = fadd double %scalar.34.315, %scalar.41.322
  store double %scalar.42.323, ptr %value.323, align 8
  %scalar.43.324 = fsub double %scalar.42.323, %scalar.34.315
  store double %scalar.43.324, ptr %value.324, align 8
  %scalar.44.325 = fsub double %scalar.41.322, %scalar.43.324
  store double %scalar.44.325, ptr %value.325, align 8
  %scalar.45.31 = fadd double %scalar.42.323, %scalar.44.325
  store double %scalar.45.31, ptr %out.4, align 8
  %scalar.46.326 = fmul double %load.0.284.1, %scalar.42.323
  store double %scalar.46.326, ptr %value.326, align 8
  %scalar.47.327 = fneg double %scalar.46.326
  store double %scalar.47.327, ptr %value.327, align 8
  %scalar.48.328 = call double @llvm.fma.f64(double %load.0.284.1, double %scalar.42.323, double %scalar.47.327)
  store double %scalar.48.328, ptr %value.328, align 8
  %scalar.49.329 = fmul double %load.0.284.1, %scalar.44.325
  store double %scalar.49.329, ptr %value.329, align 8
  %scalar.50.330 = fadd double %scalar.48.328, %scalar.49.329
  store double %scalar.50.330, ptr %value.330, align 8
  %scalar.51.331 = fmul double %load.3.287.1, %scalar.42.323
  store double %scalar.51.331, ptr %value.331, align 8
  %scalar.52.332 = fadd double %scalar.50.330, %scalar.51.331
  store double %scalar.52.332, ptr %value.332, align 8
  %scalar.53.333 = fadd double %scalar.46.326, %scalar.52.332
  store double %scalar.53.333, ptr %value.333, align 8
  %scalar.54.334 = fsub double %scalar.53.333, %scalar.46.326
  store double %scalar.54.334, ptr %value.334, align 8
  %scalar.55.335 = fsub double %scalar.52.332, %scalar.54.334
  store double %scalar.55.335, ptr %value.335, align 8
  %scalar.56.32 = fadd double %scalar.53.333, %scalar.55.335
  store double %scalar.56.32, ptr %out.5, align 8
  %load.57.336.0 = load double, ptr %arg.4, align 8
  %scalar.57.336 = fadd double %load.57.336.0, %scalar.53.333
  store double %scalar.57.336, ptr %value.336, align 8
  %scalar.58.337 = fsub double %scalar.57.336, %load.57.336.0
  store double %scalar.58.337, ptr %value.337, align 8
  %scalar.59.338 = fsub double %scalar.57.336, %scalar.58.337
  store double %scalar.59.338, ptr %value.338, align 8
  %scalar.60.339 = fsub double %load.57.336.0, %scalar.59.338
  store double %scalar.60.339, ptr %value.339, align 8
  %scalar.61.340 = fsub double %scalar.53.333, %scalar.58.337
  store double %scalar.61.340, ptr %value.340, align 8
  %scalar.62.341 = fadd double %scalar.60.339, %scalar.61.340
  store double %scalar.62.341, ptr %value.341, align 8
  %load.63.342.1 = load double, ptr %arg.32, align 8
  %scalar.63.342 = fadd double %scalar.62.341, %load.63.342.1
  store double %scalar.63.342, ptr %value.342, align 8
  %scalar.64.343 = fadd double %scalar.63.342, %scalar.55.335
  store double %scalar.64.343, ptr %value.343, align 8
  %scalar.65.344 = fadd double %scalar.57.336, %scalar.64.343
  store double %scalar.65.344, ptr %value.344, align 8
  %scalar.66.345 = fsub double %scalar.65.344, %scalar.57.336
  store double %scalar.66.345, ptr %value.345, align 8
  %scalar.67.346 = fsub double %scalar.64.343, %scalar.66.345
  store double %scalar.67.346, ptr %value.346, align 8
  %scalar.68.33 = fadd double %scalar.65.344, %scalar.67.346
  store double %scalar.68.33, ptr %out.6, align 8
  %scalar.69.347 = fmul double %load.0.284.1, %scalar.65.344
  store double %scalar.69.347, ptr %value.347, align 8
  %scalar.70.348 = fneg double %scalar.69.347
  store double %scalar.70.348, ptr %value.348, align 8
  %scalar.71.349 = call double @llvm.fma.f64(double %load.0.284.1, double %scalar.65.344, double %scalar.70.348)
  store double %scalar.71.349, ptr %value.349, align 8
  %scalar.72.350 = fmul double %load.0.284.1, %scalar.67.346
  store double %scalar.72.350, ptr %value.350, align 8
  %scalar.73.351 = fadd double %scalar.71.349, %scalar.72.350
  store double %scalar.73.351, ptr %value.351, align 8
  %scalar.74.352 = fmul double %load.3.287.1, %scalar.65.344
  store double %scalar.74.352, ptr %value.352, align 8
  %scalar.75.353 = fadd double %scalar.73.351, %scalar.74.352
  store double %scalar.75.353, ptr %value.353, align 8
  %scalar.76.354 = fadd double %scalar.69.347, %scalar.75.353
  store double %scalar.76.354, ptr %value.354, align 8
  %scalar.77.355 = fsub double %scalar.76.354, %scalar.69.347
  store double %scalar.77.355, ptr %value.355, align 8
  %scalar.78.356 = fsub double %scalar.75.353, %scalar.77.355
  store double %scalar.78.356, ptr %value.356, align 8
  %scalar.79.34 = fadd double %scalar.76.354, %scalar.78.356
  store double %scalar.79.34, ptr %out.7, align 8
  %load.80.357.0 = load double, ptr %arg.5, align 8
  %scalar.80.357 = fadd double %load.80.357.0, %scalar.76.354
  store double %scalar.80.357, ptr %value.357, align 8
  %scalar.81.358 = fsub double %scalar.80.357, %load.80.357.0
  store double %scalar.81.358, ptr %value.358, align 8
  %scalar.82.359 = fsub double %scalar.80.357, %scalar.81.358
  store double %scalar.82.359, ptr %value.359, align 8
  %scalar.83.360 = fsub double %load.80.357.0, %scalar.82.359
  store double %scalar.83.360, ptr %value.360, align 8
  %scalar.84.361 = fsub double %scalar.76.354, %scalar.81.358
  store double %scalar.84.361, ptr %value.361, align 8
  %scalar.85.362 = fadd double %scalar.83.360, %scalar.84.361
  store double %scalar.85.362, ptr %value.362, align 8
  %load.86.363.1 = load double, ptr %arg.33, align 8
  %scalar.86.363 = fadd double %scalar.85.362, %load.86.363.1
  store double %scalar.86.363, ptr %value.363, align 8
  %scalar.87.364 = fadd double %scalar.86.363, %scalar.78.356
  store double %scalar.87.364, ptr %value.364, align 8
  %scalar.88.365 = fadd double %scalar.80.357, %scalar.87.364
  store double %scalar.88.365, ptr %value.365, align 8
  %scalar.89.366 = fsub double %scalar.88.365, %scalar.80.357
  store double %scalar.89.366, ptr %value.366, align 8
  %scalar.90.367 = fsub double %scalar.87.364, %scalar.89.366
  store double %scalar.90.367, ptr %value.367, align 8
  %scalar.91.35 = fadd double %scalar.88.365, %scalar.90.367
  store double %scalar.91.35, ptr %out.8, align 8
  %scalar.92.368 = fmul double %load.0.284.1, %scalar.88.365
  store double %scalar.92.368, ptr %value.368, align 8
  %scalar.93.369 = fneg double %scalar.92.368
  store double %scalar.93.369, ptr %value.369, align 8
  %scalar.94.370 = call double @llvm.fma.f64(double %load.0.284.1, double %scalar.88.365, double %scalar.93.369)
  store double %scalar.94.370, ptr %value.370, align 8
  %scalar.95.371 = fmul double %load.0.284.1, %scalar.90.367
  store double %scalar.95.371, ptr %value.371, align 8
  %scalar.96.372 = fadd double %scalar.94.370, %scalar.95.371
  store double %scalar.96.372, ptr %value.372, align 8
  %scalar.97.373 = fmul double %load.3.287.1, %scalar.88.365
  store double %scalar.97.373, ptr %value.373, align 8
  %scalar.98.374 = fadd double %scalar.96.372, %scalar.97.373
  store double %scalar.98.374, ptr %value.374, align 8
  %scalar.99.375 = fadd double %scalar.92.368, %scalar.98.374
  store double %scalar.99.375, ptr %value.375, align 8
  %scalar.100.376 = fsub double %scalar.99.375, %scalar.92.368
  store double %scalar.100.376, ptr %value.376, align 8
  %scalar.101.377 = fsub double %scalar.98.374, %scalar.100.376
  store double %scalar.101.377, ptr %value.377, align 8
  %scalar.102.36 = fadd double %scalar.99.375, %scalar.101.377
  store double %scalar.102.36, ptr %out.9, align 8
  %load.103.378.0 = load double, ptr %arg.6, align 8
  %scalar.103.378 = fadd double %load.103.378.0, %scalar.99.375
  store double %scalar.103.378, ptr %value.378, align 8
  %scalar.104.379 = fsub double %scalar.103.378, %load.103.378.0
  store double %scalar.104.379, ptr %value.379, align 8
  %scalar.105.380 = fsub double %scalar.103.378, %scalar.104.379
  store double %scalar.105.380, ptr %value.380, align 8
  %scalar.106.381 = fsub double %load.103.378.0, %scalar.105.380
  store double %scalar.106.381, ptr %value.381, align 8
  %scalar.107.382 = fsub double %scalar.99.375, %scalar.104.379
  store double %scalar.107.382, ptr %value.382, align 8
  %scalar.108.383 = fadd double %scalar.106.381, %scalar.107.382
  store double %scalar.108.383, ptr %value.383, align 8
  %load.109.384.1 = load double, ptr %arg.34, align 8
  %scalar.109.384 = fadd double %scalar.108.383, %load.109.384.1
  store double %scalar.109.384, ptr %value.384, align 8
  %scalar.110.385 = fadd double %scalar.109.384, %scalar.101.377
  store double %scalar.110.385, ptr %value.385, align 8
  %scalar.111.386 = fadd double %scalar.103.378, %scalar.110.385
  store double %scalar.111.386, ptr %value.386, align 8
  %scalar.112.387 = fsub double %scalar.111.386, %scalar.103.378
  store double %scalar.112.387, ptr %value.387, align 8
  %scalar.113.388 = fsub double %scalar.110.385, %scalar.112.387
  store double %scalar.113.388, ptr %value.388, align 8
  %scalar.114.37 = fadd double %scalar.111.386, %scalar.113.388
  store double %scalar.114.37, ptr %out.10, align 8
  %scalar.115.389 = fmul double %load.0.284.1, %scalar.111.386
  store double %scalar.115.389, ptr %value.389, align 8
  %scalar.116.390 = fneg double %scalar.115.389
  store double %scalar.116.390, ptr %value.390, align 8
  %scalar.117.391 = call double @llvm.fma.f64(double %load.0.284.1, double %scalar.111.386, double %scalar.116.390)
  store double %scalar.117.391, ptr %value.391, align 8
  %scalar.118.392 = fmul double %load.0.284.1, %scalar.113.388
  store double %scalar.118.392, ptr %value.392, align 8
  %scalar.119.393 = fadd double %scalar.117.391, %scalar.118.392
  store double %scalar.119.393, ptr %value.393, align 8
  %scalar.120.394 = fmul double %load.3.287.1, %scalar.111.386
  store double %scalar.120.394, ptr %value.394, align 8
  %scalar.121.395 = fadd double %scalar.119.393, %scalar.120.394
  store double %scalar.121.395, ptr %value.395, align 8
  %scalar.122.396 = fadd double %scalar.115.389, %scalar.121.395
  store double %scalar.122.396, ptr %value.396, align 8
  %scalar.123.397 = fsub double %scalar.122.396, %scalar.115.389
  store double %scalar.123.397, ptr %value.397, align 8
  %scalar.124.398 = fsub double %scalar.121.395, %scalar.123.397
  store double %scalar.124.398, ptr %value.398, align 8
  %scalar.125.38 = fadd double %scalar.122.396, %scalar.124.398
  store double %scalar.125.38, ptr %out.11, align 8
  %load.126.399.0 = load double, ptr %arg.7, align 8
  %scalar.126.399 = fadd double %load.126.399.0, %scalar.122.396
  store double %scalar.126.399, ptr %value.399, align 8
  %scalar.127.400 = fsub double %scalar.126.399, %load.126.399.0
  store double %scalar.127.400, ptr %value.400, align 8
  %scalar.128.401 = fsub double %scalar.126.399, %scalar.127.400
  store double %scalar.128.401, ptr %value.401, align 8
  %scalar.129.402 = fsub double %load.126.399.0, %scalar.128.401
  store double %scalar.129.402, ptr %value.402, align 8
  %scalar.130.403 = fsub double %scalar.122.396, %scalar.127.400
  store double %scalar.130.403, ptr %value.403, align 8
  %scalar.131.404 = fadd double %scalar.129.402, %scalar.130.403
  store double %scalar.131.404, ptr %value.404, align 8
  %load.132.405.1 = load double, ptr %arg.35, align 8
  %scalar.132.405 = fadd double %scalar.131.404, %load.132.405.1
  store double %scalar.132.405, ptr %value.405, align 8
  %scalar.133.406 = fadd double %scalar.132.405, %scalar.124.398
  store double %scalar.133.406, ptr %value.406, align 8
  %scalar.134.407 = fadd double %scalar.126.399, %scalar.133.406
  store double %scalar.134.407, ptr %value.407, align 8
  %scalar.135.408 = fsub double %scalar.134.407, %scalar.126.399
  store double %scalar.135.408, ptr %value.408, align 8
  %scalar.136.409 = fsub double %scalar.133.406, %scalar.135.408
  store double %scalar.136.409, ptr %value.409, align 8
  %scalar.137.39 = fadd double %scalar.134.407, %scalar.136.409
  store double %scalar.137.39, ptr %out.12, align 8
  %scalar.138.410 = fmul double %load.0.284.1, %scalar.134.407
  store double %scalar.138.410, ptr %value.410, align 8
  %scalar.139.411 = fneg double %scalar.138.410
  store double %scalar.139.411, ptr %value.411, align 8
  %scalar.140.412 = call double @llvm.fma.f64(double %load.0.284.1, double %scalar.134.407, double %scalar.139.411)
  store double %scalar.140.412, ptr %value.412, align 8
  %scalar.141.413 = fmul double %load.0.284.1, %scalar.136.409
  store double %scalar.141.413, ptr %value.413, align 8
  %scalar.142.414 = fadd double %scalar.140.412, %scalar.141.413
  store double %scalar.142.414, ptr %value.414, align 8
  %scalar.143.415 = fmul double %load.3.287.1, %scalar.134.407
  store double %scalar.143.415, ptr %value.415, align 8
  %scalar.144.416 = fadd double %scalar.142.414, %scalar.143.415
  store double %scalar.144.416, ptr %value.416, align 8
  %scalar.145.417 = fadd double %scalar.138.410, %scalar.144.416
  store double %scalar.145.417, ptr %value.417, align 8
  %scalar.146.418 = fsub double %scalar.145.417, %scalar.138.410
  store double %scalar.146.418, ptr %value.418, align 8
  %scalar.147.419 = fsub double %scalar.144.416, %scalar.146.418
  store double %scalar.147.419, ptr %value.419, align 8
  %scalar.148.40 = fadd double %scalar.145.417, %scalar.147.419
  store double %scalar.148.40, ptr %out.13, align 8
  %load.149.420.0 = load double, ptr %arg.8, align 8
  %scalar.149.420 = fadd double %load.149.420.0, %scalar.145.417
  store double %scalar.149.420, ptr %value.420, align 8
  %scalar.150.421 = fsub double %scalar.149.420, %load.149.420.0
  store double %scalar.150.421, ptr %value.421, align 8
  %scalar.151.422 = fsub double %scalar.149.420, %scalar.150.421
  store double %scalar.151.422, ptr %value.422, align 8
  %scalar.152.423 = fsub double %load.149.420.0, %scalar.151.422
  store double %scalar.152.423, ptr %value.423, align 8
  %scalar.153.424 = fsub double %scalar.145.417, %scalar.150.421
  store double %scalar.153.424, ptr %value.424, align 8
  %scalar.154.425 = fadd double %scalar.152.423, %scalar.153.424
  store double %scalar.154.425, ptr %value.425, align 8
  %load.155.426.1 = load double, ptr %arg.36, align 8
  %scalar.155.426 = fadd double %scalar.154.425, %load.155.426.1
  store double %scalar.155.426, ptr %value.426, align 8
  %scalar.156.427 = fadd double %scalar.155.426, %scalar.147.419
  store double %scalar.156.427, ptr %value.427, align 8
  %scalar.157.428 = fadd double %scalar.149.420, %scalar.156.427
  store double %scalar.157.428, ptr %value.428, align 8
  %scalar.158.429 = fsub double %scalar.157.428, %scalar.149.420
  store double %scalar.158.429, ptr %value.429, align 8
  %scalar.159.430 = fsub double %scalar.156.427, %scalar.158.429
  store double %scalar.159.430, ptr %value.430, align 8
  %scalar.160.41 = fadd double %scalar.157.428, %scalar.159.430
  store double %scalar.160.41, ptr %out.14, align 8
  %scalar.161.431 = fmul double %load.0.284.1, %scalar.157.428
  store double %scalar.161.431, ptr %value.431, align 8
  %scalar.162.432 = fneg double %scalar.161.431
  store double %scalar.162.432, ptr %value.432, align 8
  %scalar.163.433 = call double @llvm.fma.f64(double %load.0.284.1, double %scalar.157.428, double %scalar.162.432)
  store double %scalar.163.433, ptr %value.433, align 8
  %scalar.164.434 = fmul double %load.0.284.1, %scalar.159.430
  store double %scalar.164.434, ptr %value.434, align 8
  %scalar.165.435 = fadd double %scalar.163.433, %scalar.164.434
  store double %scalar.165.435, ptr %value.435, align 8
  %scalar.166.436 = fmul double %load.3.287.1, %scalar.157.428
  store double %scalar.166.436, ptr %value.436, align 8
  %scalar.167.437 = fadd double %scalar.165.435, %scalar.166.436
  store double %scalar.167.437, ptr %value.437, align 8
  %scalar.168.438 = fadd double %scalar.161.431, %scalar.167.437
  store double %scalar.168.438, ptr %value.438, align 8
  %scalar.169.439 = fsub double %scalar.168.438, %scalar.161.431
  store double %scalar.169.439, ptr %value.439, align 8
  %scalar.170.440 = fsub double %scalar.167.437, %scalar.169.439
  store double %scalar.170.440, ptr %value.440, align 8
  %scalar.171.42 = fadd double %scalar.168.438, %scalar.170.440
  store double %scalar.171.42, ptr %out.15, align 8
  %load.172.441.0 = load double, ptr %arg.9, align 8
  %scalar.172.441 = fadd double %load.172.441.0, %scalar.168.438
  store double %scalar.172.441, ptr %value.441, align 8
  %scalar.173.442 = fsub double %scalar.172.441, %load.172.441.0
  store double %scalar.173.442, ptr %value.442, align 8
  %scalar.174.443 = fsub double %scalar.172.441, %scalar.173.442
  store double %scalar.174.443, ptr %value.443, align 8
  %scalar.175.444 = fsub double %load.172.441.0, %scalar.174.443
  store double %scalar.175.444, ptr %value.444, align 8
  %scalar.176.445 = fsub double %scalar.168.438, %scalar.173.442
  store double %scalar.176.445, ptr %value.445, align 8
  %scalar.177.446 = fadd double %scalar.175.444, %scalar.176.445
  store double %scalar.177.446, ptr %value.446, align 8
  %load.178.447.1 = load double, ptr %arg.37, align 8
  %scalar.178.447 = fadd double %scalar.177.446, %load.178.447.1
  store double %scalar.178.447, ptr %value.447, align 8
  %scalar.179.448 = fadd double %scalar.178.447, %scalar.170.440
  store double %scalar.179.448, ptr %value.448, align 8
  %scalar.180.449 = fadd double %scalar.172.441, %scalar.179.448
  store double %scalar.180.449, ptr %value.449, align 8
  %scalar.181.450 = fsub double %scalar.180.449, %scalar.172.441
  store double %scalar.181.450, ptr %value.450, align 8
  %scalar.182.451 = fsub double %scalar.179.448, %scalar.181.450
  store double %scalar.182.451, ptr %value.451, align 8
  %scalar.183.43 = fadd double %scalar.180.449, %scalar.182.451
  store double %scalar.183.43, ptr %out.16, align 8
  %scalar.184.452 = fmul double %load.0.284.1, %scalar.180.449
  store double %scalar.184.452, ptr %value.452, align 8
  %scalar.185.453 = fneg double %scalar.184.452
  store double %scalar.185.453, ptr %value.453, align 8
  %scalar.186.454 = call double @llvm.fma.f64(double %load.0.284.1, double %scalar.180.449, double %scalar.185.453)
  store double %scalar.186.454, ptr %value.454, align 8
  %scalar.187.455 = fmul double %load.0.284.1, %scalar.182.451
  store double %scalar.187.455, ptr %value.455, align 8
  %scalar.188.456 = fadd double %scalar.186.454, %scalar.187.455
  store double %scalar.188.456, ptr %value.456, align 8
  %scalar.189.457 = fmul double %load.3.287.1, %scalar.180.449
  store double %scalar.189.457, ptr %value.457, align 8
  %scalar.190.458 = fadd double %scalar.188.456, %scalar.189.457
  store double %scalar.190.458, ptr %value.458, align 8
  %scalar.191.459 = fadd double %scalar.184.452, %scalar.190.458
  store double %scalar.191.459, ptr %value.459, align 8
  %scalar.192.460 = fsub double %scalar.191.459, %scalar.184.452
  store double %scalar.192.460, ptr %value.460, align 8
  %scalar.193.461 = fsub double %scalar.190.458, %scalar.192.460
  store double %scalar.193.461, ptr %value.461, align 8
  %scalar.194.44 = fadd double %scalar.191.459, %scalar.193.461
  store double %scalar.194.44, ptr %out.17, align 8
  %load.195.462.0 = load double, ptr %arg.10, align 8
  %scalar.195.462 = fadd double %load.195.462.0, %scalar.191.459
  store double %scalar.195.462, ptr %value.462, align 8
  %scalar.196.463 = fsub double %scalar.195.462, %load.195.462.0
  store double %scalar.196.463, ptr %value.463, align 8
  %scalar.197.464 = fsub double %scalar.195.462, %scalar.196.463
  store double %scalar.197.464, ptr %value.464, align 8
  %scalar.198.465 = fsub double %load.195.462.0, %scalar.197.464
  store double %scalar.198.465, ptr %value.465, align 8
  %scalar.199.466 = fsub double %scalar.191.459, %scalar.196.463
  store double %scalar.199.466, ptr %value.466, align 8
  %scalar.200.467 = fadd double %scalar.198.465, %scalar.199.466
  store double %scalar.200.467, ptr %value.467, align 8
  %load.201.468.1 = load double, ptr %arg.38, align 8
  %scalar.201.468 = fadd double %scalar.200.467, %load.201.468.1
  store double %scalar.201.468, ptr %value.468, align 8
  %scalar.202.469 = fadd double %scalar.201.468, %scalar.193.461
  store double %scalar.202.469, ptr %value.469, align 8
  %scalar.203.470 = fadd double %scalar.195.462, %scalar.202.469
  store double %scalar.203.470, ptr %value.470, align 8
  %scalar.204.471 = fsub double %scalar.203.470, %scalar.195.462
  store double %scalar.204.471, ptr %value.471, align 8
  %scalar.205.472 = fsub double %scalar.202.469, %scalar.204.471
  store double %scalar.205.472, ptr %value.472, align 8
  %scalar.206.45 = fadd double %scalar.203.470, %scalar.205.472
  store double %scalar.206.45, ptr %out.18, align 8
  %scalar.207.473 = fmul double %load.0.284.1, %scalar.203.470
  store double %scalar.207.473, ptr %value.473, align 8
  %scalar.208.474 = fneg double %scalar.207.473
  store double %scalar.208.474, ptr %value.474, align 8
  %scalar.209.475 = call double @llvm.fma.f64(double %load.0.284.1, double %scalar.203.470, double %scalar.208.474)
  store double %scalar.209.475, ptr %value.475, align 8
  %scalar.210.476 = fmul double %load.0.284.1, %scalar.205.472
  store double %scalar.210.476, ptr %value.476, align 8
  %scalar.211.477 = fadd double %scalar.209.475, %scalar.210.476
  store double %scalar.211.477, ptr %value.477, align 8
  %scalar.212.478 = fmul double %load.3.287.1, %scalar.203.470
  store double %scalar.212.478, ptr %value.478, align 8
  %scalar.213.479 = fadd double %scalar.211.477, %scalar.212.478
  store double %scalar.213.479, ptr %value.479, align 8
  %scalar.214.480 = fadd double %scalar.207.473, %scalar.213.479
  store double %scalar.214.480, ptr %value.480, align 8
  %scalar.215.481 = fsub double %scalar.214.480, %scalar.207.473
  store double %scalar.215.481, ptr %value.481, align 8
  %scalar.216.482 = fsub double %scalar.213.479, %scalar.215.481
  store double %scalar.216.482, ptr %value.482, align 8
  %scalar.217.46 = fadd double %scalar.214.480, %scalar.216.482
  store double %scalar.217.46, ptr %out.19, align 8
  %load.218.483.0 = load double, ptr %arg.11, align 8
  %scalar.218.483 = fadd double %load.218.483.0, %scalar.214.480
  store double %scalar.218.483, ptr %value.483, align 8
  %scalar.219.484 = fsub double %scalar.218.483, %load.218.483.0
  store double %scalar.219.484, ptr %value.484, align 8
  %scalar.220.485 = fsub double %scalar.218.483, %scalar.219.484
  store double %scalar.220.485, ptr %value.485, align 8
  %scalar.221.486 = fsub double %load.218.483.0, %scalar.220.485
  store double %scalar.221.486, ptr %value.486, align 8
  %scalar.222.487 = fsub double %scalar.214.480, %scalar.219.484
  store double %scalar.222.487, ptr %value.487, align 8
  %scalar.223.488 = fadd double %scalar.221.486, %scalar.222.487
  store double %scalar.223.488, ptr %value.488, align 8
  %load.224.489.1 = load double, ptr %arg.39, align 8
  %scalar.224.489 = fadd double %scalar.223.488, %load.224.489.1
  store double %scalar.224.489, ptr %value.489, align 8
  %scalar.225.490 = fadd double %scalar.224.489, %scalar.216.482
  store double %scalar.225.490, ptr %value.490, align 8
  %scalar.226.491 = fadd double %scalar.218.483, %scalar.225.490
  store double %scalar.226.491, ptr %value.491, align 8
  %scalar.227.492 = fsub double %scalar.226.491, %scalar.218.483
  store double %scalar.227.492, ptr %value.492, align 8
  %scalar.228.493 = fsub double %scalar.225.490, %scalar.227.492
  store double %scalar.228.493, ptr %value.493, align 8
  %scalar.229.47 = fadd double %scalar.226.491, %scalar.228.493
  store double %scalar.229.47, ptr %out.20, align 8
  %scalar.230.494 = fmul double %load.0.284.1, %scalar.226.491
  store double %scalar.230.494, ptr %value.494, align 8
  %scalar.231.495 = fneg double %scalar.230.494
  store double %scalar.231.495, ptr %value.495, align 8
  %scalar.232.496 = call double @llvm.fma.f64(double %load.0.284.1, double %scalar.226.491, double %scalar.231.495)
  store double %scalar.232.496, ptr %value.496, align 8
  %scalar.233.497 = fmul double %load.0.284.1, %scalar.228.493
  store double %scalar.233.497, ptr %value.497, align 8
  %scalar.234.498 = fadd double %scalar.232.496, %scalar.233.497
  store double %scalar.234.498, ptr %value.498, align 8
  %scalar.235.499 = fmul double %load.3.287.1, %scalar.226.491
  store double %scalar.235.499, ptr %value.499, align 8
  %scalar.236.500 = fadd double %scalar.234.498, %scalar.235.499
  store double %scalar.236.500, ptr %value.500, align 8
  %scalar.237.501 = fadd double %scalar.230.494, %scalar.236.500
  store double %scalar.237.501, ptr %value.501, align 8
  %scalar.238.502 = fsub double %scalar.237.501, %scalar.230.494
  store double %scalar.238.502, ptr %value.502, align 8
  %scalar.239.503 = fsub double %scalar.236.500, %scalar.238.502
  store double %scalar.239.503, ptr %value.503, align 8
  %scalar.240.48 = fadd double %scalar.237.501, %scalar.239.503
  store double %scalar.240.48, ptr %out.21, align 8
  %load.241.504.0 = load double, ptr %arg.12, align 8
  %scalar.241.504 = fadd double %load.241.504.0, %scalar.237.501
  store double %scalar.241.504, ptr %value.504, align 8
  %scalar.242.505 = fsub double %scalar.241.504, %load.241.504.0
  store double %scalar.242.505, ptr %value.505, align 8
  %scalar.243.506 = fsub double %scalar.241.504, %scalar.242.505
  store double %scalar.243.506, ptr %value.506, align 8
  %scalar.244.507 = fsub double %load.241.504.0, %scalar.243.506
  store double %scalar.244.507, ptr %value.507, align 8
  %scalar.245.508 = fsub double %scalar.237.501, %scalar.242.505
  store double %scalar.245.508, ptr %value.508, align 8
  %scalar.246.509 = fadd double %scalar.244.507, %scalar.245.508
  store double %scalar.246.509, ptr %value.509, align 8
  %load.247.510.1 = load double, ptr %arg.40, align 8
  %scalar.247.510 = fadd double %scalar.246.509, %load.247.510.1
  store double %scalar.247.510, ptr %value.510, align 8
  %scalar.248.511 = fadd double %scalar.247.510, %scalar.239.503
  store double %scalar.248.511, ptr %value.511, align 8
  %scalar.249.512 = fadd double %scalar.241.504, %scalar.248.511
  store double %scalar.249.512, ptr %value.512, align 8
  %scalar.250.513 = fsub double %scalar.249.512, %scalar.241.504
  store double %scalar.250.513, ptr %value.513, align 8
  %scalar.251.514 = fsub double %scalar.248.511, %scalar.250.513
  store double %scalar.251.514, ptr %value.514, align 8
  %scalar.252.49 = fadd double %scalar.249.512, %scalar.251.514
  store double %scalar.252.49, ptr %out.22, align 8
  %scalar.253.515 = fmul double %load.0.284.1, %scalar.249.512
  store double %scalar.253.515, ptr %value.515, align 8
  %scalar.254.516 = fneg double %scalar.253.515
  store double %scalar.254.516, ptr %value.516, align 8
  %scalar.255.517 = call double @llvm.fma.f64(double %load.0.284.1, double %scalar.249.512, double %scalar.254.516)
  store double %scalar.255.517, ptr %value.517, align 8
  %scalar.256.518 = fmul double %load.0.284.1, %scalar.251.514
  store double %scalar.256.518, ptr %value.518, align 8
  %scalar.257.519 = fadd double %scalar.255.517, %scalar.256.518
  store double %scalar.257.519, ptr %value.519, align 8
  %scalar.258.520 = fmul double %load.3.287.1, %scalar.249.512
  store double %scalar.258.520, ptr %value.520, align 8
  %scalar.259.521 = fadd double %scalar.257.519, %scalar.258.520
  store double %scalar.259.521, ptr %value.521, align 8
  %scalar.260.522 = fadd double %scalar.253.515, %scalar.259.521
  store double %scalar.260.522, ptr %value.522, align 8
  %scalar.261.523 = fsub double %scalar.260.522, %scalar.253.515
  store double %scalar.261.523, ptr %value.523, align 8
  %scalar.262.524 = fsub double %scalar.259.521, %scalar.261.523
  store double %scalar.262.524, ptr %value.524, align 8
  %scalar.263.50 = fadd double %scalar.260.522, %scalar.262.524
  store double %scalar.263.50, ptr %out.23, align 8
  %load.264.525.0 = load double, ptr %arg.13, align 8
  %scalar.264.525 = fadd double %load.264.525.0, %scalar.260.522
  store double %scalar.264.525, ptr %value.525, align 8
  %scalar.265.526 = fsub double %scalar.264.525, %load.264.525.0
  store double %scalar.265.526, ptr %value.526, align 8
  %scalar.266.527 = fsub double %scalar.264.525, %scalar.265.526
  store double %scalar.266.527, ptr %value.527, align 8
  %scalar.267.528 = fsub double %load.264.525.0, %scalar.266.527
  store double %scalar.267.528, ptr %value.528, align 8
  %scalar.268.529 = fsub double %scalar.260.522, %scalar.265.526
  store double %scalar.268.529, ptr %value.529, align 8
  %scalar.269.530 = fadd double %scalar.267.528, %scalar.268.529
  store double %scalar.269.530, ptr %value.530, align 8
  %load.270.531.1 = load double, ptr %arg.41, align 8
  %scalar.270.531 = fadd double %scalar.269.530, %load.270.531.1
  store double %scalar.270.531, ptr %value.531, align 8
  %scalar.271.532 = fadd double %scalar.270.531, %scalar.262.524
  store double %scalar.271.532, ptr %value.532, align 8
  %scalar.272.533 = fadd double %scalar.264.525, %scalar.271.532
  store double %scalar.272.533, ptr %value.533, align 8
  %scalar.273.534 = fsub double %scalar.272.533, %scalar.264.525
  store double %scalar.273.534, ptr %value.534, align 8
  %scalar.274.535 = fsub double %scalar.271.532, %scalar.273.534
  store double %scalar.274.535, ptr %value.535, align 8
  %scalar.275.51 = fadd double %scalar.272.533, %scalar.274.535
  store double %scalar.275.51, ptr %out.24, align 8
  %scalar.276.536 = fmul double %load.0.284.1, %scalar.272.533
  store double %scalar.276.536, ptr %value.536, align 8
  %scalar.277.537 = fneg double %scalar.276.536
  store double %scalar.277.537, ptr %value.537, align 8
  %scalar.278.538 = call double @llvm.fma.f64(double %load.0.284.1, double %scalar.272.533, double %scalar.277.537)
  store double %scalar.278.538, ptr %value.538, align 8
  %scalar.279.539 = fmul double %load.0.284.1, %scalar.274.535
  store double %scalar.279.539, ptr %value.539, align 8
  %scalar.280.540 = fadd double %scalar.278.538, %scalar.279.539
  store double %scalar.280.540, ptr %value.540, align 8
  %scalar.281.541 = fmul double %load.3.287.1, %scalar.272.533
  store double %scalar.281.541, ptr %value.541, align 8
  %scalar.282.542 = fadd double %scalar.280.540, %scalar.281.541
  store double %scalar.282.542, ptr %value.542, align 8
  %scalar.283.543 = fadd double %scalar.276.536, %scalar.282.542
  store double %scalar.283.543, ptr %value.543, align 8
  %scalar.284.544 = fsub double %scalar.283.543, %scalar.276.536
  store double %scalar.284.544, ptr %value.544, align 8
  %scalar.285.545 = fsub double %scalar.282.542, %scalar.284.544
  store double %scalar.285.545, ptr %value.545, align 8
  %scalar.286.52 = fadd double %scalar.283.543, %scalar.285.545
  store double %scalar.286.52, ptr %out.25, align 8
  %load.287.546.0 = load double, ptr %arg.14, align 8
  %scalar.287.546 = fadd double %load.287.546.0, %scalar.283.543
  store double %scalar.287.546, ptr %value.546, align 8
  %scalar.288.547 = fsub double %scalar.287.546, %load.287.546.0
  store double %scalar.288.547, ptr %value.547, align 8
  %scalar.289.548 = fsub double %scalar.287.546, %scalar.288.547
  store double %scalar.289.548, ptr %value.548, align 8
  %scalar.290.549 = fsub double %load.287.546.0, %scalar.289.548
  store double %scalar.290.549, ptr %value.549, align 8
  %scalar.291.550 = fsub double %scalar.283.543, %scalar.288.547
  store double %scalar.291.550, ptr %value.550, align 8
  %scalar.292.551 = fadd double %scalar.290.549, %scalar.291.550
  store double %scalar.292.551, ptr %value.551, align 8
  %load.293.552.1 = load double, ptr %arg.42, align 8
  %scalar.293.552 = fadd double %scalar.292.551, %load.293.552.1
  store double %scalar.293.552, ptr %value.552, align 8
  %scalar.294.553 = fadd double %scalar.293.552, %scalar.285.545
  store double %scalar.294.553, ptr %value.553, align 8
  %scalar.295.554 = fadd double %scalar.287.546, %scalar.294.553
  store double %scalar.295.554, ptr %value.554, align 8
  %scalar.296.555 = fsub double %scalar.295.554, %scalar.287.546
  store double %scalar.296.555, ptr %value.555, align 8
  %scalar.297.556 = fsub double %scalar.294.553, %scalar.296.555
  store double %scalar.297.556, ptr %value.556, align 8
  %scalar.298.53 = fadd double %scalar.295.554, %scalar.297.556
  store double %scalar.298.53, ptr %out.26, align 8
  %scalar.299.557 = fmul double %load.0.284.1, %scalar.295.554
  store double %scalar.299.557, ptr %value.557, align 8
  %scalar.300.558 = fneg double %scalar.299.557
  store double %scalar.300.558, ptr %value.558, align 8
  %scalar.301.559 = call double @llvm.fma.f64(double %load.0.284.1, double %scalar.295.554, double %scalar.300.558)
  store double %scalar.301.559, ptr %value.559, align 8
  %scalar.302.560 = fmul double %load.0.284.1, %scalar.297.556
  store double %scalar.302.560, ptr %value.560, align 8
  %scalar.303.561 = fadd double %scalar.301.559, %scalar.302.560
  store double %scalar.303.561, ptr %value.561, align 8
  %scalar.304.562 = fmul double %load.3.287.1, %scalar.295.554
  store double %scalar.304.562, ptr %value.562, align 8
  %scalar.305.563 = fadd double %scalar.303.561, %scalar.304.562
  store double %scalar.305.563, ptr %value.563, align 8
  %scalar.306.564 = fadd double %scalar.299.557, %scalar.305.563
  store double %scalar.306.564, ptr %value.564, align 8
  %scalar.307.565 = fsub double %scalar.306.564, %scalar.299.557
  store double %scalar.307.565, ptr %value.565, align 8
  %scalar.308.566 = fsub double %scalar.305.563, %scalar.307.565
  store double %scalar.308.566, ptr %value.566, align 8
  %scalar.309.54 = fadd double %scalar.306.564, %scalar.308.566
  store double %scalar.309.54, ptr %out.27, align 8
  %load.310.567.0 = load double, ptr %arg.15, align 8
  %scalar.310.567 = fadd double %load.310.567.0, %scalar.306.564
  store double %scalar.310.567, ptr %value.567, align 8
  %scalar.311.568 = fsub double %scalar.310.567, %load.310.567.0
  store double %scalar.311.568, ptr %value.568, align 8
  %scalar.312.569 = fsub double %scalar.310.567, %scalar.311.568
  store double %scalar.312.569, ptr %value.569, align 8
  %scalar.313.570 = fsub double %load.310.567.0, %scalar.312.569
  store double %scalar.313.570, ptr %value.570, align 8
  %scalar.314.571 = fsub double %scalar.306.564, %scalar.311.568
  store double %scalar.314.571, ptr %value.571, align 8
  %scalar.315.572 = fadd double %scalar.313.570, %scalar.314.571
  store double %scalar.315.572, ptr %value.572, align 8
  %load.316.573.1 = load double, ptr %arg.43, align 8
  %scalar.316.573 = fadd double %scalar.315.572, %load.316.573.1
  store double %scalar.316.573, ptr %value.573, align 8
  %scalar.317.574 = fadd double %scalar.316.573, %scalar.308.566
  store double %scalar.317.574, ptr %value.574, align 8
  %scalar.318.575 = fadd double %scalar.310.567, %scalar.317.574
  store double %scalar.318.575, ptr %value.575, align 8
  %scalar.319.576 = fsub double %scalar.318.575, %scalar.310.567
  store double %scalar.319.576, ptr %value.576, align 8
  %scalar.320.577 = fsub double %scalar.317.574, %scalar.319.576
  store double %scalar.320.577, ptr %value.577, align 8
  %scalar.321.55 = fadd double %scalar.318.575, %scalar.320.577
  store double %scalar.321.55, ptr %out.28, align 8
  %scalar.322.578 = fmul double %load.0.284.1, %scalar.318.575
  store double %scalar.322.578, ptr %value.578, align 8
  %scalar.323.579 = fneg double %scalar.322.578
  store double %scalar.323.579, ptr %value.579, align 8
  %scalar.324.580 = call double @llvm.fma.f64(double %load.0.284.1, double %scalar.318.575, double %scalar.323.579)
  store double %scalar.324.580, ptr %value.580, align 8
  %scalar.325.581 = fmul double %load.0.284.1, %scalar.320.577
  store double %scalar.325.581, ptr %value.581, align 8
  %scalar.326.582 = fadd double %scalar.324.580, %scalar.325.581
  store double %scalar.326.582, ptr %value.582, align 8
  %scalar.327.583 = fmul double %load.3.287.1, %scalar.318.575
  store double %scalar.327.583, ptr %value.583, align 8
  %scalar.328.584 = fadd double %scalar.326.582, %scalar.327.583
  store double %scalar.328.584, ptr %value.584, align 8
  %scalar.329.585 = fadd double %scalar.322.578, %scalar.328.584
  store double %scalar.329.585, ptr %value.585, align 8
  %scalar.330.586 = fsub double %scalar.329.585, %scalar.322.578
  store double %scalar.330.586, ptr %value.586, align 8
  %scalar.331.587 = fsub double %scalar.328.584, %scalar.330.586
  store double %scalar.331.587, ptr %value.587, align 8
  %scalar.332.56 = fadd double %scalar.329.585, %scalar.331.587
  store double %scalar.332.56, ptr %out.29, align 8
  %load.333.588.0 = load double, ptr %arg.16, align 8
  %scalar.333.588 = fadd double %load.333.588.0, %scalar.329.585
  store double %scalar.333.588, ptr %value.588, align 8
  %scalar.334.589 = fsub double %scalar.333.588, %load.333.588.0
  store double %scalar.334.589, ptr %value.589, align 8
  %scalar.335.590 = fsub double %scalar.333.588, %scalar.334.589
  store double %scalar.335.590, ptr %value.590, align 8
  %scalar.336.591 = fsub double %load.333.588.0, %scalar.335.590
  store double %scalar.336.591, ptr %value.591, align 8
  %scalar.337.592 = fsub double %scalar.329.585, %scalar.334.589
  store double %scalar.337.592, ptr %value.592, align 8
  %scalar.338.593 = fadd double %scalar.336.591, %scalar.337.592
  store double %scalar.338.593, ptr %value.593, align 8
  %load.339.594.1 = load double, ptr %arg.44, align 8
  %scalar.339.594 = fadd double %scalar.338.593, %load.339.594.1
  store double %scalar.339.594, ptr %value.594, align 8
  %scalar.340.595 = fadd double %scalar.339.594, %scalar.331.587
  store double %scalar.340.595, ptr %value.595, align 8
  %scalar.341.596 = fadd double %scalar.333.588, %scalar.340.595
  store double %scalar.341.596, ptr %value.596, align 8
  %scalar.342.597 = fsub double %scalar.341.596, %scalar.333.588
  store double %scalar.342.597, ptr %value.597, align 8
  %scalar.343.598 = fsub double %scalar.340.595, %scalar.342.597
  store double %scalar.343.598, ptr %value.598, align 8
  %scalar.344.57 = fadd double %scalar.341.596, %scalar.343.598
  store double %scalar.344.57, ptr %out.30, align 8
  %scalar.345.599 = fmul double %load.0.284.1, %scalar.341.596
  store double %scalar.345.599, ptr %value.599, align 8
  %scalar.346.600 = fneg double %scalar.345.599
  store double %scalar.346.600, ptr %value.600, align 8
  %scalar.347.601 = call double @llvm.fma.f64(double %load.0.284.1, double %scalar.341.596, double %scalar.346.600)
  store double %scalar.347.601, ptr %value.601, align 8
  %scalar.348.602 = fmul double %load.0.284.1, %scalar.343.598
  store double %scalar.348.602, ptr %value.602, align 8
  %scalar.349.603 = fadd double %scalar.347.601, %scalar.348.602
  store double %scalar.349.603, ptr %value.603, align 8
  %scalar.350.604 = fmul double %load.3.287.1, %scalar.341.596
  store double %scalar.350.604, ptr %value.604, align 8
  %scalar.351.605 = fadd double %scalar.349.603, %scalar.350.604
  store double %scalar.351.605, ptr %value.605, align 8
  %scalar.352.606 = fadd double %scalar.345.599, %scalar.351.605
  store double %scalar.352.606, ptr %value.606, align 8
  %scalar.353.607 = fsub double %scalar.352.606, %scalar.345.599
  store double %scalar.353.607, ptr %value.607, align 8
  %scalar.354.608 = fsub double %scalar.351.605, %scalar.353.607
  store double %scalar.354.608, ptr %value.608, align 8
  %scalar.355.58 = fadd double %scalar.352.606, %scalar.354.608
  store double %scalar.355.58, ptr %out.31, align 8
  %load.356.609.0 = load double, ptr %arg.17, align 8
  %scalar.356.609 = fadd double %load.356.609.0, %scalar.352.606
  store double %scalar.356.609, ptr %value.609, align 8
  %scalar.357.610 = fsub double %scalar.356.609, %load.356.609.0
  store double %scalar.357.610, ptr %value.610, align 8
  %scalar.358.611 = fsub double %scalar.356.609, %scalar.357.610
  store double %scalar.358.611, ptr %value.611, align 8
  %scalar.359.612 = fsub double %load.356.609.0, %scalar.358.611
  store double %scalar.359.612, ptr %value.612, align 8
  %scalar.360.613 = fsub double %scalar.352.606, %scalar.357.610
  store double %scalar.360.613, ptr %value.613, align 8
  %scalar.361.614 = fadd double %scalar.359.612, %scalar.360.613
  store double %scalar.361.614, ptr %value.614, align 8
  %load.362.615.1 = load double, ptr %arg.45, align 8
  %scalar.362.615 = fadd double %scalar.361.614, %load.362.615.1
  store double %scalar.362.615, ptr %value.615, align 8
  %scalar.363.616 = fadd double %scalar.362.615, %scalar.354.608
  store double %scalar.363.616, ptr %value.616, align 8
  %scalar.364.617 = fadd double %scalar.356.609, %scalar.363.616
  store double %scalar.364.617, ptr %value.617, align 8
  %scalar.365.618 = fsub double %scalar.364.617, %scalar.356.609
  store double %scalar.365.618, ptr %value.618, align 8
  %scalar.366.619 = fsub double %scalar.363.616, %scalar.365.618
  store double %scalar.366.619, ptr %value.619, align 8
  %scalar.367.59 = fadd double %scalar.364.617, %scalar.366.619
  store double %scalar.367.59, ptr %out.32, align 8
  %scalar.368.620 = fmul double %load.0.284.1, %scalar.364.617
  store double %scalar.368.620, ptr %value.620, align 8
  %scalar.369.621 = fneg double %scalar.368.620
  store double %scalar.369.621, ptr %value.621, align 8
  %scalar.370.622 = call double @llvm.fma.f64(double %load.0.284.1, double %scalar.364.617, double %scalar.369.621)
  store double %scalar.370.622, ptr %value.622, align 8
  %scalar.371.623 = fmul double %load.0.284.1, %scalar.366.619
  store double %scalar.371.623, ptr %value.623, align 8
  %scalar.372.624 = fadd double %scalar.370.622, %scalar.371.623
  store double %scalar.372.624, ptr %value.624, align 8
  %scalar.373.625 = fmul double %load.3.287.1, %scalar.364.617
  store double %scalar.373.625, ptr %value.625, align 8
  %scalar.374.626 = fadd double %scalar.372.624, %scalar.373.625
  store double %scalar.374.626, ptr %value.626, align 8
  %scalar.375.627 = fadd double %scalar.368.620, %scalar.374.626
  store double %scalar.375.627, ptr %value.627, align 8
  %scalar.376.628 = fsub double %scalar.375.627, %scalar.368.620
  store double %scalar.376.628, ptr %value.628, align 8
  %scalar.377.629 = fsub double %scalar.374.626, %scalar.376.628
  store double %scalar.377.629, ptr %value.629, align 8
  %scalar.378.60 = fadd double %scalar.375.627, %scalar.377.629
  store double %scalar.378.60, ptr %out.33, align 8
  %load.379.630.0 = load double, ptr %arg.18, align 8
  %scalar.379.630 = fadd double %load.379.630.0, %scalar.375.627
  store double %scalar.379.630, ptr %value.630, align 8
  %scalar.380.631 = fsub double %scalar.379.630, %load.379.630.0
  store double %scalar.380.631, ptr %value.631, align 8
  %scalar.381.632 = fsub double %scalar.379.630, %scalar.380.631
  store double %scalar.381.632, ptr %value.632, align 8
  %scalar.382.633 = fsub double %load.379.630.0, %scalar.381.632
  store double %scalar.382.633, ptr %value.633, align 8
  %scalar.383.634 = fsub double %scalar.375.627, %scalar.380.631
  store double %scalar.383.634, ptr %value.634, align 8
  %scalar.384.635 = fadd double %scalar.382.633, %scalar.383.634
  store double %scalar.384.635, ptr %value.635, align 8
  %load.385.636.1 = load double, ptr %arg.46, align 8
  %scalar.385.636 = fadd double %scalar.384.635, %load.385.636.1
  store double %scalar.385.636, ptr %value.636, align 8
  %scalar.386.637 = fadd double %scalar.385.636, %scalar.377.629
  store double %scalar.386.637, ptr %value.637, align 8
  %scalar.387.638 = fadd double %scalar.379.630, %scalar.386.637
  store double %scalar.387.638, ptr %value.638, align 8
  %scalar.388.639 = fsub double %scalar.387.638, %scalar.379.630
  store double %scalar.388.639, ptr %value.639, align 8
  %scalar.389.640 = fsub double %scalar.386.637, %scalar.388.639
  store double %scalar.389.640, ptr %value.640, align 8
  %scalar.390.61 = fadd double %scalar.387.638, %scalar.389.640
  store double %scalar.390.61, ptr %out.34, align 8
  %scalar.391.641 = fmul double %load.0.284.1, %scalar.387.638
  store double %scalar.391.641, ptr %value.641, align 8
  %scalar.392.642 = fneg double %scalar.391.641
  store double %scalar.392.642, ptr %value.642, align 8
  %scalar.393.643 = call double @llvm.fma.f64(double %load.0.284.1, double %scalar.387.638, double %scalar.392.642)
  store double %scalar.393.643, ptr %value.643, align 8
  %scalar.394.644 = fmul double %load.0.284.1, %scalar.389.640
  store double %scalar.394.644, ptr %value.644, align 8
  %scalar.395.645 = fadd double %scalar.393.643, %scalar.394.644
  store double %scalar.395.645, ptr %value.645, align 8
  %scalar.396.646 = fmul double %load.3.287.1, %scalar.387.638
  store double %scalar.396.646, ptr %value.646, align 8
  %scalar.397.647 = fadd double %scalar.395.645, %scalar.396.646
  store double %scalar.397.647, ptr %value.647, align 8
  %scalar.398.648 = fadd double %scalar.391.641, %scalar.397.647
  store double %scalar.398.648, ptr %value.648, align 8
  %scalar.399.649 = fsub double %scalar.398.648, %scalar.391.641
  store double %scalar.399.649, ptr %value.649, align 8
  %scalar.400.650 = fsub double %scalar.397.647, %scalar.399.649
  store double %scalar.400.650, ptr %value.650, align 8
  %scalar.401.62 = fadd double %scalar.398.648, %scalar.400.650
  store double %scalar.401.62, ptr %out.35, align 8
  %load.402.651.0 = load double, ptr %arg.19, align 8
  %scalar.402.651 = fadd double %load.402.651.0, %scalar.398.648
  store double %scalar.402.651, ptr %value.651, align 8
  %scalar.403.652 = fsub double %scalar.402.651, %load.402.651.0
  store double %scalar.403.652, ptr %value.652, align 8
  %scalar.404.653 = fsub double %scalar.402.651, %scalar.403.652
  store double %scalar.404.653, ptr %value.653, align 8
  %scalar.405.654 = fsub double %load.402.651.0, %scalar.404.653
  store double %scalar.405.654, ptr %value.654, align 8
  %scalar.406.655 = fsub double %scalar.398.648, %scalar.403.652
  store double %scalar.406.655, ptr %value.655, align 8
  %scalar.407.656 = fadd double %scalar.405.654, %scalar.406.655
  store double %scalar.407.656, ptr %value.656, align 8
  %load.408.657.1 = load double, ptr %arg.47, align 8
  %scalar.408.657 = fadd double %scalar.407.656, %load.408.657.1
  store double %scalar.408.657, ptr %value.657, align 8
  %scalar.409.658 = fadd double %scalar.408.657, %scalar.400.650
  store double %scalar.409.658, ptr %value.658, align 8
  %scalar.410.659 = fadd double %scalar.402.651, %scalar.409.658
  store double %scalar.410.659, ptr %value.659, align 8
  %scalar.411.660 = fsub double %scalar.410.659, %scalar.402.651
  store double %scalar.411.660, ptr %value.660, align 8
  %scalar.412.661 = fsub double %scalar.409.658, %scalar.411.660
  store double %scalar.412.661, ptr %value.661, align 8
  %scalar.413.63 = fadd double %scalar.410.659, %scalar.412.661
  store double %scalar.413.63, ptr %out.36, align 8
  %scalar.414.662 = fmul double %load.0.284.1, %scalar.410.659
  store double %scalar.414.662, ptr %value.662, align 8
  %scalar.415.663 = fneg double %scalar.414.662
  store double %scalar.415.663, ptr %value.663, align 8
  %scalar.416.664 = call double @llvm.fma.f64(double %load.0.284.1, double %scalar.410.659, double %scalar.415.663)
  store double %scalar.416.664, ptr %value.664, align 8
  %scalar.417.665 = fmul double %load.0.284.1, %scalar.412.661
  store double %scalar.417.665, ptr %value.665, align 8
  %scalar.418.666 = fadd double %scalar.416.664, %scalar.417.665
  store double %scalar.418.666, ptr %value.666, align 8
  %scalar.419.667 = fmul double %load.3.287.1, %scalar.410.659
  store double %scalar.419.667, ptr %value.667, align 8
  %scalar.420.668 = fadd double %scalar.418.666, %scalar.419.667
  store double %scalar.420.668, ptr %value.668, align 8
  %scalar.421.669 = fadd double %scalar.414.662, %scalar.420.668
  store double %scalar.421.669, ptr %value.669, align 8
  %scalar.422.670 = fsub double %scalar.421.669, %scalar.414.662
  store double %scalar.422.670, ptr %value.670, align 8
  %scalar.423.671 = fsub double %scalar.420.668, %scalar.422.670
  store double %scalar.423.671, ptr %value.671, align 8
  %scalar.424.64 = fadd double %scalar.421.669, %scalar.423.671
  store double %scalar.424.64, ptr %out.37, align 8
  %load.425.672.0 = load double, ptr %arg.20, align 8
  %scalar.425.672 = fadd double %load.425.672.0, %scalar.421.669
  store double %scalar.425.672, ptr %value.672, align 8
  %scalar.426.673 = fsub double %scalar.425.672, %load.425.672.0
  store double %scalar.426.673, ptr %value.673, align 8
  %scalar.427.674 = fsub double %scalar.425.672, %scalar.426.673
  store double %scalar.427.674, ptr %value.674, align 8
  %scalar.428.675 = fsub double %load.425.672.0, %scalar.427.674
  store double %scalar.428.675, ptr %value.675, align 8
  %scalar.429.676 = fsub double %scalar.421.669, %scalar.426.673
  store double %scalar.429.676, ptr %value.676, align 8
  %scalar.430.677 = fadd double %scalar.428.675, %scalar.429.676
  store double %scalar.430.677, ptr %value.677, align 8
  %load.431.678.1 = load double, ptr %arg.48, align 8
  %scalar.431.678 = fadd double %scalar.430.677, %load.431.678.1
  store double %scalar.431.678, ptr %value.678, align 8
  %scalar.432.679 = fadd double %scalar.431.678, %scalar.423.671
  store double %scalar.432.679, ptr %value.679, align 8
  %scalar.433.680 = fadd double %scalar.425.672, %scalar.432.679
  store double %scalar.433.680, ptr %value.680, align 8
  %scalar.434.681 = fsub double %scalar.433.680, %scalar.425.672
  store double %scalar.434.681, ptr %value.681, align 8
  %scalar.435.682 = fsub double %scalar.432.679, %scalar.434.681
  store double %scalar.435.682, ptr %value.682, align 8
  %scalar.436.65 = fadd double %scalar.433.680, %scalar.435.682
  store double %scalar.436.65, ptr %out.38, align 8
  %scalar.437.683 = fmul double %load.0.284.1, %scalar.433.680
  store double %scalar.437.683, ptr %value.683, align 8
  %scalar.438.684 = fneg double %scalar.437.683
  store double %scalar.438.684, ptr %value.684, align 8
  %scalar.439.685 = call double @llvm.fma.f64(double %load.0.284.1, double %scalar.433.680, double %scalar.438.684)
  store double %scalar.439.685, ptr %value.685, align 8
  %scalar.440.686 = fmul double %load.0.284.1, %scalar.435.682
  store double %scalar.440.686, ptr %value.686, align 8
  %scalar.441.687 = fadd double %scalar.439.685, %scalar.440.686
  store double %scalar.441.687, ptr %value.687, align 8
  %scalar.442.688 = fmul double %load.3.287.1, %scalar.433.680
  store double %scalar.442.688, ptr %value.688, align 8
  %scalar.443.689 = fadd double %scalar.441.687, %scalar.442.688
  store double %scalar.443.689, ptr %value.689, align 8
  %scalar.444.690 = fadd double %scalar.437.683, %scalar.443.689
  store double %scalar.444.690, ptr %value.690, align 8
  %scalar.445.691 = fsub double %scalar.444.690, %scalar.437.683
  store double %scalar.445.691, ptr %value.691, align 8
  %scalar.446.692 = fsub double %scalar.443.689, %scalar.445.691
  store double %scalar.446.692, ptr %value.692, align 8
  %scalar.447.66 = fadd double %scalar.444.690, %scalar.446.692
  store double %scalar.447.66, ptr %out.39, align 8
  %load.448.693.0 = load double, ptr %arg.21, align 8
  %scalar.448.693 = fadd double %load.448.693.0, %scalar.444.690
  store double %scalar.448.693, ptr %value.693, align 8
  %scalar.449.694 = fsub double %scalar.448.693, %load.448.693.0
  store double %scalar.449.694, ptr %value.694, align 8
  %scalar.450.695 = fsub double %scalar.448.693, %scalar.449.694
  store double %scalar.450.695, ptr %value.695, align 8
  %scalar.451.696 = fsub double %load.448.693.0, %scalar.450.695
  store double %scalar.451.696, ptr %value.696, align 8
  %scalar.452.697 = fsub double %scalar.444.690, %scalar.449.694
  store double %scalar.452.697, ptr %value.697, align 8
  %scalar.453.698 = fadd double %scalar.451.696, %scalar.452.697
  store double %scalar.453.698, ptr %value.698, align 8
  %load.454.699.1 = load double, ptr %arg.49, align 8
  %scalar.454.699 = fadd double %scalar.453.698, %load.454.699.1
  store double %scalar.454.699, ptr %value.699, align 8
  %scalar.455.700 = fadd double %scalar.454.699, %scalar.446.692
  store double %scalar.455.700, ptr %value.700, align 8
  %scalar.456.701 = fadd double %scalar.448.693, %scalar.455.700
  store double %scalar.456.701, ptr %value.701, align 8
  %scalar.457.702 = fsub double %scalar.456.701, %scalar.448.693
  store double %scalar.457.702, ptr %value.702, align 8
  %scalar.458.703 = fsub double %scalar.455.700, %scalar.457.702
  store double %scalar.458.703, ptr %value.703, align 8
  %scalar.459.67 = fadd double %scalar.456.701, %scalar.458.703
  store double %scalar.459.67, ptr %out.40, align 8
  %scalar.460.704 = fmul double %load.0.284.1, %scalar.456.701
  store double %scalar.460.704, ptr %value.704, align 8
  %scalar.461.705 = fneg double %scalar.460.704
  store double %scalar.461.705, ptr %value.705, align 8
  %scalar.462.706 = call double @llvm.fma.f64(double %load.0.284.1, double %scalar.456.701, double %scalar.461.705)
  store double %scalar.462.706, ptr %value.706, align 8
  %scalar.463.707 = fmul double %load.0.284.1, %scalar.458.703
  store double %scalar.463.707, ptr %value.707, align 8
  %scalar.464.708 = fadd double %scalar.462.706, %scalar.463.707
  store double %scalar.464.708, ptr %value.708, align 8
  %scalar.465.709 = fmul double %load.3.287.1, %scalar.456.701
  store double %scalar.465.709, ptr %value.709, align 8
  %scalar.466.710 = fadd double %scalar.464.708, %scalar.465.709
  store double %scalar.466.710, ptr %value.710, align 8
  %scalar.467.711 = fadd double %scalar.460.704, %scalar.466.710
  store double %scalar.467.711, ptr %value.711, align 8
  %scalar.468.712 = fsub double %scalar.467.711, %scalar.460.704
  store double %scalar.468.712, ptr %value.712, align 8
  %scalar.469.713 = fsub double %scalar.466.710, %scalar.468.712
  store double %scalar.469.713, ptr %value.713, align 8
  %scalar.470.68 = fadd double %scalar.467.711, %scalar.469.713
  store double %scalar.470.68, ptr %out.41, align 8
  %load.471.714.0 = load double, ptr %arg.22, align 8
  %scalar.471.714 = fadd double %load.471.714.0, %scalar.467.711
  store double %scalar.471.714, ptr %value.714, align 8
  %scalar.472.715 = fsub double %scalar.471.714, %load.471.714.0
  store double %scalar.472.715, ptr %value.715, align 8
  %scalar.473.716 = fsub double %scalar.471.714, %scalar.472.715
  store double %scalar.473.716, ptr %value.716, align 8
  %scalar.474.717 = fsub double %load.471.714.0, %scalar.473.716
  store double %scalar.474.717, ptr %value.717, align 8
  %scalar.475.718 = fsub double %scalar.467.711, %scalar.472.715
  store double %scalar.475.718, ptr %value.718, align 8
  %scalar.476.719 = fadd double %scalar.474.717, %scalar.475.718
  store double %scalar.476.719, ptr %value.719, align 8
  %load.477.720.1 = load double, ptr %arg.50, align 8
  %scalar.477.720 = fadd double %scalar.476.719, %load.477.720.1
  store double %scalar.477.720, ptr %value.720, align 8
  %scalar.478.721 = fadd double %scalar.477.720, %scalar.469.713
  store double %scalar.478.721, ptr %value.721, align 8
  %scalar.479.722 = fadd double %scalar.471.714, %scalar.478.721
  store double %scalar.479.722, ptr %value.722, align 8
  %scalar.480.723 = fsub double %scalar.479.722, %scalar.471.714
  store double %scalar.480.723, ptr %value.723, align 8
  %scalar.481.724 = fsub double %scalar.478.721, %scalar.480.723
  store double %scalar.481.724, ptr %value.724, align 8
  %scalar.482.69 = fadd double %scalar.479.722, %scalar.481.724
  store double %scalar.482.69, ptr %out.42, align 8
  %scalar.483.725 = fmul double %load.0.284.1, %scalar.479.722
  store double %scalar.483.725, ptr %value.725, align 8
  %scalar.484.726 = fneg double %scalar.483.725
  store double %scalar.484.726, ptr %value.726, align 8
  %scalar.485.727 = call double @llvm.fma.f64(double %load.0.284.1, double %scalar.479.722, double %scalar.484.726)
  store double %scalar.485.727, ptr %value.727, align 8
  %scalar.486.728 = fmul double %load.0.284.1, %scalar.481.724
  store double %scalar.486.728, ptr %value.728, align 8
  %scalar.487.729 = fadd double %scalar.485.727, %scalar.486.728
  store double %scalar.487.729, ptr %value.729, align 8
  %scalar.488.730 = fmul double %load.3.287.1, %scalar.479.722
  store double %scalar.488.730, ptr %value.730, align 8
  %scalar.489.731 = fadd double %scalar.487.729, %scalar.488.730
  store double %scalar.489.731, ptr %value.731, align 8
  %scalar.490.732 = fadd double %scalar.483.725, %scalar.489.731
  store double %scalar.490.732, ptr %value.732, align 8
  %scalar.491.733 = fsub double %scalar.490.732, %scalar.483.725
  store double %scalar.491.733, ptr %value.733, align 8
  %scalar.492.734 = fsub double %scalar.489.731, %scalar.491.733
  store double %scalar.492.734, ptr %value.734, align 8
  %scalar.493.70 = fadd double %scalar.490.732, %scalar.492.734
  store double %scalar.493.70, ptr %out.43, align 8
  %load.494.735.0 = load double, ptr %arg.23, align 8
  %scalar.494.735 = fadd double %load.494.735.0, %scalar.490.732
  store double %scalar.494.735, ptr %value.735, align 8
  %scalar.495.736 = fsub double %scalar.494.735, %load.494.735.0
  store double %scalar.495.736, ptr %value.736, align 8
  %scalar.496.737 = fsub double %scalar.494.735, %scalar.495.736
  store double %scalar.496.737, ptr %value.737, align 8
  %scalar.497.738 = fsub double %load.494.735.0, %scalar.496.737
  store double %scalar.497.738, ptr %value.738, align 8
  %scalar.498.739 = fsub double %scalar.490.732, %scalar.495.736
  store double %scalar.498.739, ptr %value.739, align 8
  %scalar.499.740 = fadd double %scalar.497.738, %scalar.498.739
  store double %scalar.499.740, ptr %value.740, align 8
  %load.500.741.1 = load double, ptr %arg.51, align 8
  %scalar.500.741 = fadd double %scalar.499.740, %load.500.741.1
  store double %scalar.500.741, ptr %value.741, align 8
  %scalar.501.742 = fadd double %scalar.500.741, %scalar.492.734
  store double %scalar.501.742, ptr %value.742, align 8
  %scalar.502.743 = fadd double %scalar.494.735, %scalar.501.742
  store double %scalar.502.743, ptr %value.743, align 8
  %scalar.503.744 = fsub double %scalar.502.743, %scalar.494.735
  store double %scalar.503.744, ptr %value.744, align 8
  %scalar.504.745 = fsub double %scalar.501.742, %scalar.503.744
  store double %scalar.504.745, ptr %value.745, align 8
  %scalar.505.71 = fadd double %scalar.502.743, %scalar.504.745
  store double %scalar.505.71, ptr %out.44, align 8
  %scalar.506.746 = fmul double %load.0.284.1, %scalar.502.743
  store double %scalar.506.746, ptr %value.746, align 8
  %scalar.507.747 = fneg double %scalar.506.746
  store double %scalar.507.747, ptr %value.747, align 8
  %scalar.508.748 = call double @llvm.fma.f64(double %load.0.284.1, double %scalar.502.743, double %scalar.507.747)
  store double %scalar.508.748, ptr %value.748, align 8
  %scalar.509.749 = fmul double %load.0.284.1, %scalar.504.745
  store double %scalar.509.749, ptr %value.749, align 8
  %scalar.510.750 = fadd double %scalar.508.748, %scalar.509.749
  store double %scalar.510.750, ptr %value.750, align 8
  %scalar.511.751 = fmul double %load.3.287.1, %scalar.502.743
  store double %scalar.511.751, ptr %value.751, align 8
  %scalar.512.752 = fadd double %scalar.510.750, %scalar.511.751
  store double %scalar.512.752, ptr %value.752, align 8
  %scalar.513.753 = fadd double %scalar.506.746, %scalar.512.752
  store double %scalar.513.753, ptr %value.753, align 8
  %scalar.514.754 = fsub double %scalar.513.753, %scalar.506.746
  store double %scalar.514.754, ptr %value.754, align 8
  %scalar.515.755 = fsub double %scalar.512.752, %scalar.514.754
  store double %scalar.515.755, ptr %value.755, align 8
  %scalar.516.72 = fadd double %scalar.513.753, %scalar.515.755
  store double %scalar.516.72, ptr %out.45, align 8
  %load.517.756.0 = load double, ptr %arg.24, align 8
  %scalar.517.756 = fadd double %load.517.756.0, %scalar.513.753
  store double %scalar.517.756, ptr %value.756, align 8
  %scalar.518.757 = fsub double %scalar.517.756, %load.517.756.0
  store double %scalar.518.757, ptr %value.757, align 8
  %scalar.519.758 = fsub double %scalar.517.756, %scalar.518.757
  store double %scalar.519.758, ptr %value.758, align 8
  %scalar.520.759 = fsub double %load.517.756.0, %scalar.519.758
  store double %scalar.520.759, ptr %value.759, align 8
  %scalar.521.760 = fsub double %scalar.513.753, %scalar.518.757
  store double %scalar.521.760, ptr %value.760, align 8
  %scalar.522.761 = fadd double %scalar.520.759, %scalar.521.760
  store double %scalar.522.761, ptr %value.761, align 8
  %load.523.762.1 = load double, ptr %arg.52, align 8
  %scalar.523.762 = fadd double %scalar.522.761, %load.523.762.1
  store double %scalar.523.762, ptr %value.762, align 8
  %scalar.524.763 = fadd double %scalar.523.762, %scalar.515.755
  store double %scalar.524.763, ptr %value.763, align 8
  %scalar.525.764 = fadd double %scalar.517.756, %scalar.524.763
  store double %scalar.525.764, ptr %value.764, align 8
  %scalar.526.765 = fsub double %scalar.525.764, %scalar.517.756
  store double %scalar.526.765, ptr %value.765, align 8
  %scalar.527.766 = fsub double %scalar.524.763, %scalar.526.765
  store double %scalar.527.766, ptr %value.766, align 8
  %scalar.528.73 = fadd double %scalar.525.764, %scalar.527.766
  store double %scalar.528.73, ptr %out.46, align 8
  %scalar.529.767 = fmul double %load.0.284.1, %scalar.525.764
  store double %scalar.529.767, ptr %value.767, align 8
  %scalar.530.768 = fneg double %scalar.529.767
  store double %scalar.530.768, ptr %value.768, align 8
  %scalar.531.769 = call double @llvm.fma.f64(double %load.0.284.1, double %scalar.525.764, double %scalar.530.768)
  store double %scalar.531.769, ptr %value.769, align 8
  %scalar.532.770 = fmul double %load.0.284.1, %scalar.527.766
  store double %scalar.532.770, ptr %value.770, align 8
  %scalar.533.771 = fadd double %scalar.531.769, %scalar.532.770
  store double %scalar.533.771, ptr %value.771, align 8
  %scalar.534.772 = fmul double %load.3.287.1, %scalar.525.764
  store double %scalar.534.772, ptr %value.772, align 8
  %scalar.535.773 = fadd double %scalar.533.771, %scalar.534.772
  store double %scalar.535.773, ptr %value.773, align 8
  %scalar.536.774 = fadd double %scalar.529.767, %scalar.535.773
  store double %scalar.536.774, ptr %value.774, align 8
  %scalar.537.775 = fsub double %scalar.536.774, %scalar.529.767
  store double %scalar.537.775, ptr %value.775, align 8
  %scalar.538.776 = fsub double %scalar.535.773, %scalar.537.775
  store double %scalar.538.776, ptr %value.776, align 8
  %scalar.539.74 = fadd double %scalar.536.774, %scalar.538.776
  store double %scalar.539.74, ptr %out.47, align 8
  %load.540.777.0 = load double, ptr %arg.25, align 8
  %scalar.540.777 = fadd double %load.540.777.0, %scalar.536.774
  store double %scalar.540.777, ptr %value.777, align 8
  %scalar.541.778 = fsub double %scalar.540.777, %load.540.777.0
  store double %scalar.541.778, ptr %value.778, align 8
  %scalar.542.779 = fsub double %scalar.540.777, %scalar.541.778
  store double %scalar.542.779, ptr %value.779, align 8
  %scalar.543.780 = fsub double %load.540.777.0, %scalar.542.779
  store double %scalar.543.780, ptr %value.780, align 8
  %scalar.544.781 = fsub double %scalar.536.774, %scalar.541.778
  store double %scalar.544.781, ptr %value.781, align 8
  %scalar.545.782 = fadd double %scalar.543.780, %scalar.544.781
  store double %scalar.545.782, ptr %value.782, align 8
  %load.546.783.1 = load double, ptr %arg.53, align 8
  %scalar.546.783 = fadd double %scalar.545.782, %load.546.783.1
  store double %scalar.546.783, ptr %value.783, align 8
  %scalar.547.784 = fadd double %scalar.546.783, %scalar.538.776
  store double %scalar.547.784, ptr %value.784, align 8
  %scalar.548.785 = fadd double %scalar.540.777, %scalar.547.784
  store double %scalar.548.785, ptr %value.785, align 8
  %scalar.549.786 = fsub double %scalar.548.785, %scalar.540.777
  store double %scalar.549.786, ptr %value.786, align 8
  %scalar.550.787 = fsub double %scalar.547.784, %scalar.549.786
  store double %scalar.550.787, ptr %value.787, align 8
  %scalar.551.75 = fadd double %scalar.548.785, %scalar.550.787
  store double %scalar.551.75, ptr %out.48, align 8
  %scalar.552.788 = fmul double %load.0.284.1, %scalar.548.785
  store double %scalar.552.788, ptr %value.788, align 8
  %scalar.553.789 = fneg double %scalar.552.788
  store double %scalar.553.789, ptr %value.789, align 8
  %scalar.554.790 = call double @llvm.fma.f64(double %load.0.284.1, double %scalar.548.785, double %scalar.553.789)
  store double %scalar.554.790, ptr %value.790, align 8
  %scalar.555.791 = fmul double %load.0.284.1, %scalar.550.787
  store double %scalar.555.791, ptr %value.791, align 8
  %scalar.556.792 = fadd double %scalar.554.790, %scalar.555.791
  store double %scalar.556.792, ptr %value.792, align 8
  %scalar.557.793 = fmul double %load.3.287.1, %scalar.548.785
  store double %scalar.557.793, ptr %value.793, align 8
  %scalar.558.794 = fadd double %scalar.556.792, %scalar.557.793
  store double %scalar.558.794, ptr %value.794, align 8
  %scalar.559.795 = fadd double %scalar.552.788, %scalar.558.794
  store double %scalar.559.795, ptr %value.795, align 8
  %scalar.560.796 = fsub double %scalar.559.795, %scalar.552.788
  store double %scalar.560.796, ptr %value.796, align 8
  %scalar.561.797 = fsub double %scalar.558.794, %scalar.560.796
  store double %scalar.561.797, ptr %value.797, align 8
  %scalar.562.76 = fadd double %scalar.559.795, %scalar.561.797
  store double %scalar.562.76, ptr %out.49, align 8
  %load.563.798.0 = load double, ptr %arg.26, align 8
  %scalar.563.798 = fadd double %load.563.798.0, %scalar.559.795
  store double %scalar.563.798, ptr %value.798, align 8
  %scalar.564.799 = fsub double %scalar.563.798, %load.563.798.0
  store double %scalar.564.799, ptr %value.799, align 8
  %scalar.565.800 = fsub double %scalar.563.798, %scalar.564.799
  store double %scalar.565.800, ptr %value.800, align 8
  %scalar.566.801 = fsub double %load.563.798.0, %scalar.565.800
  store double %scalar.566.801, ptr %value.801, align 8
  %scalar.567.802 = fsub double %scalar.559.795, %scalar.564.799
  store double %scalar.567.802, ptr %value.802, align 8
  %scalar.568.803 = fadd double %scalar.566.801, %scalar.567.802
  store double %scalar.568.803, ptr %value.803, align 8
  %load.569.804.1 = load double, ptr %arg.54, align 8
  %scalar.569.804 = fadd double %scalar.568.803, %load.569.804.1
  store double %scalar.569.804, ptr %value.804, align 8
  %scalar.570.805 = fadd double %scalar.569.804, %scalar.561.797
  store double %scalar.570.805, ptr %value.805, align 8
  %scalar.571.806 = fadd double %scalar.563.798, %scalar.570.805
  store double %scalar.571.806, ptr %value.806, align 8
  %scalar.572.807 = fsub double %scalar.571.806, %scalar.563.798
  store double %scalar.572.807, ptr %value.807, align 8
  %scalar.573.808 = fsub double %scalar.570.805, %scalar.572.807
  store double %scalar.573.808, ptr %value.808, align 8
  %scalar.574.77 = fadd double %scalar.571.806, %scalar.573.808
  store double %scalar.574.77, ptr %out.50, align 8
  %load.575.809.0 = load double, ptr %arg.27, align 8
  %scalar.575.809 = fmul double %load.575.809.0, %scalar.571.806
  store double %scalar.575.809, ptr %value.809, align 8
  %scalar.576.810 = fneg double %scalar.575.809
  store double %scalar.576.810, ptr %value.810, align 8
  %scalar.577.811 = call double @llvm.fma.f64(double %load.575.809.0, double %scalar.571.806, double %scalar.576.810)
  store double %scalar.577.811, ptr %value.811, align 8
  %scalar.578.812 = fmul double %load.575.809.0, %scalar.573.808
  store double %scalar.578.812, ptr %value.812, align 8
  %scalar.579.813 = fadd double %scalar.577.811, %scalar.578.812
  store double %scalar.579.813, ptr %value.813, align 8
  %load.580.814.0 = load double, ptr %arg.55, align 8
  %scalar.580.814 = fmul double %load.580.814.0, %scalar.571.806
  store double %scalar.580.814, ptr %value.814, align 8
  %scalar.581.815 = fadd double %scalar.579.813, %scalar.580.814
  store double %scalar.581.815, ptr %value.815, align 8
  %scalar.582.816 = fadd double %scalar.575.809, %scalar.581.815
  store double %scalar.582.816, ptr %value.816, align 8
  %scalar.583.817 = fsub double %scalar.582.816, %scalar.575.809
  store double %scalar.583.817, ptr %value.817, align 8
  %scalar.584.818 = fsub double %scalar.581.815, %scalar.583.817
  store double %scalar.584.818, ptr %value.818, align 8
  %scalar.585.78 = fadd double %scalar.582.816, %scalar.584.818
  store double %scalar.585.78, ptr %out.0, align 8
  ret void
}

define void @__ssa_atanh_core_pack__atanh_core(ptr noalias %arg.0, ptr noalias %arg.1, ptr noalias %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr noalias %arg.11, ptr noalias %arg.12, ptr noalias %arg.13, ptr noalias %arg.14, ptr noalias %arg.15, ptr noalias %arg.16, ptr noalias %arg.17, ptr noalias %arg.18, ptr noalias %arg.19, ptr noalias %arg.20, ptr noalias %arg.21, ptr noalias %arg.22, ptr noalias %arg.23, ptr noalias %arg.24, ptr noalias %arg.25, ptr %arg.26, ptr %arg.27, ptr noalias %arg.28, ptr noalias %arg.29, ptr noalias %arg.30, ptr noalias %arg.31, ptr noalias %arg.32, ptr noalias %arg.33, ptr noalias %arg.34, ptr noalias %arg.35, ptr noalias %arg.36, ptr noalias %arg.37, ptr noalias %arg.38, ptr noalias %arg.39, ptr noalias %arg.40, ptr noalias %arg.41, ptr noalias %arg.42, ptr noalias %arg.43, ptr noalias %arg.44, ptr noalias %arg.45, ptr noalias %arg.46, ptr noalias %arg.47, ptr noalias %arg.48, ptr noalias %arg.49, ptr noalias %arg.50, ptr noalias %arg.51, ptr noalias %arg.52, ptr noalias %arg.53, ptr %arg.54, ptr %arg.55, ptr %out.0) {
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
  call void @__ssa_atanh_core_pack__atanh_core__planned_region_0(ptr %arg.18, ptr %arg.26, ptr %arg.17, ptr %arg.16, ptr %arg.15, ptr %arg.14, ptr %arg.13, ptr %arg.11, ptr %arg.10, ptr %arg.9, ptr %arg.8, ptr %arg.7, ptr %arg.6, ptr %arg.5, ptr %arg.4, ptr %arg.3, ptr %arg.2, ptr %arg.25, ptr %arg.24, ptr %arg.23, ptr %arg.22, ptr %arg.21, ptr %arg.20, ptr %arg.19, ptr %arg.12, ptr %arg.1, ptr %arg.0, ptr %arg.27, ptr %arg.46, ptr %arg.54, ptr %arg.45, ptr %arg.44, ptr %arg.43, ptr %arg.42, ptr %arg.41, ptr %arg.39, ptr %arg.38, ptr %arg.37, ptr %arg.36, ptr %arg.35, ptr %arg.34, ptr %arg.33, ptr %arg.32, ptr %arg.31, ptr %arg.30, ptr %arg.53, ptr %arg.52, ptr %arg.51, ptr %arg.50, ptr %arg.49, ptr %arg.48, ptr %arg.47, ptr %arg.40, ptr %arg.29, ptr %arg.28, ptr %arg.55, ptr %out.0, ptr %value.28, ptr %value.29, ptr %value.30, ptr %value.31, ptr %value.32, ptr %value.33, ptr %value.34, ptr %value.35, ptr %value.36, ptr %value.37, ptr %value.38, ptr %value.39, ptr %value.40, ptr %value.41, ptr %value.42, ptr %value.43, ptr %value.44, ptr %value.45, ptr %value.46, ptr %value.47, ptr %value.48, ptr %value.49, ptr %value.50, ptr %value.51, ptr %value.52, ptr %value.53, ptr %value.54, ptr %value.55, ptr %value.56, ptr %value.57, ptr %value.58, ptr %value.59, ptr %value.60, ptr %value.61, ptr %value.62, ptr %value.63, ptr %value.64, ptr %value.65, ptr %value.66, ptr %value.67, ptr %value.68, ptr %value.69, ptr %value.70, ptr %value.71, ptr %value.72, ptr %value.73, ptr %value.74, ptr %value.75, ptr %value.76, ptr %value.77)
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
  %public.addr.48 = getelementptr ptr, ptr %buffers, i64 48
  %public.48 = load ptr, ptr %public.addr.48, align 8
  %public.addr.49 = getelementptr ptr, ptr %buffers, i64 49
  %public.49 = load ptr, ptr %public.addr.49, align 8
  %public.addr.50 = getelementptr ptr, ptr %buffers, i64 50
  %public.50 = load ptr, ptr %public.addr.50, align 8
  %public.addr.51 = getelementptr ptr, ptr %buffers, i64 51
  %public.51 = load ptr, ptr %public.addr.51, align 8
  %public.addr.52 = getelementptr ptr, ptr %buffers, i64 52
  %public.52 = load ptr, ptr %public.addr.52, align 8
  %public.addr.53 = getelementptr ptr, ptr %buffers, i64 53
  %public.53 = load ptr, ptr %public.addr.53, align 8
  %public.addr.54 = getelementptr ptr, ptr %buffers, i64 54
  %public.54 = load ptr, ptr %public.addr.54, align 8
  call void @__ssa_atanh_core_pack__atanh_core_pack(ptr %public.0, ptr %public.1, ptr %public.2, ptr %public.3, ptr %public.4, ptr %public.5, ptr %public.6, ptr %public.7, ptr %public.8, ptr %public.9, ptr %public.10, ptr %public.11, ptr %public.12, ptr %public.13, ptr %public.14, ptr %public.15, ptr %public.16, ptr %public.17, ptr %public.18, ptr %public.19, ptr %public.20, ptr %public.21, ptr %public.22, ptr %public.23, ptr %public.24, ptr %public.25, ptr %public.26, ptr %public.27, ptr %public.28, ptr %public.29, ptr %public.30, ptr %public.31, ptr %public.32, ptr %public.33, ptr %public.34, ptr %public.35, ptr %public.36, ptr %public.37, ptr %public.38, ptr %public.39, ptr %public.40, ptr %public.41, ptr %public.42, ptr %public.43, ptr %public.44, ptr %public.45, ptr %public.46, ptr %public.47, ptr %public.48, ptr %public.49, ptr %public.50, ptr %public.51, ptr %public.52, ptr %public.53, ptr %public.54, ptr %public.2)
  ret void
}
