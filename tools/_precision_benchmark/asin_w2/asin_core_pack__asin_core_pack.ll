source_filename = "turing.ssa-llvm.asin_core_pack__asin_core_pack"

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-windows-gnu"

declare double @llvm.fma.f64(double, double, double)

define void @__ssa_asin_core_pack__asin_core_pack__planned_region_0(ptr noalias %arg.0, ptr %arg.1, ptr %out.0, ptr %out.1) {
entry:
  %value.254 = alloca i32, i64 1, align 8
  %value.255 = alloca i32, i64 1, align 8
  %value.257 = alloca double, i64 1, align 8
  %value.258 = alloca i32, i64 1, align 8
  %value.259 = alloca i32, i64 1, align 8
  %value.260 = alloca i32, i64 1, align 8
  %value.261 = alloca i32, i64 1, align 8
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
  %load.0.45.0 = load i32, ptr %arg.1, align 4
  %address.0.45 = getelementptr double, ptr %arg.0, i32 %load.0.45.0
  store i32 2, ptr %value.254, align 4
  %scalar.2.255 = mul i32 %load.0.45.0, 2
  store i32 %scalar.2.255, ptr %value.255, align 4
  %address.3.256 = getelementptr double, ptr %arg.0, i32 %scalar.2.255
  %pinned.load.4.257 = load double, ptr %address.3.256, align 8
  store double %pinned.load.4.257, ptr %value.257, align 8
  store i32 2, ptr %value.258, align 4
  %scalar.6.259 = mul i32 %load.0.45.0, 2
  store i32 %scalar.6.259, ptr %value.259, align 4
  store i32 1, ptr %value.260, align 4
  %scalar.8.261 = add i32 %scalar.6.259, 1
  store i32 %scalar.8.261, ptr %value.261, align 4
  %address.9.262 = getelementptr double, ptr %arg.0, i32 %scalar.8.261
  %pinned.load.10.263 = load double, ptr %address.9.262, align 8
  store double %pinned.load.10.263, ptr %value.263, align 8
  %load.11.264.0 = load double, ptr %value.257, align 8
  %scalar.11.264 = fmul double %load.11.264.0, %load.11.264.0
  store double %scalar.11.264, ptr %value.264, align 8
  %scalar.12.265 = fneg double %scalar.11.264
  store double %scalar.12.265, ptr %value.265, align 8
  %scalar.13.266 = call double @llvm.fma.f64(double %load.11.264.0, double %load.11.264.0, double %scalar.12.265)
  store double %scalar.13.266, ptr %value.266, align 8
  %load.14.267.1 = load double, ptr %value.263, align 8
  %scalar.14.267 = fmul double %load.11.264.0, %load.14.267.1
  store double %scalar.14.267, ptr %value.267, align 8
  %scalar.15.268 = fadd double %scalar.13.266, %scalar.14.267
  store double %scalar.15.268, ptr %value.268, align 8
  %scalar.16.269 = fmul double %load.14.267.1, %load.11.264.0
  store double %scalar.16.269, ptr %value.269, align 8
  %scalar.17.270 = fadd double %scalar.15.268, %scalar.16.269
  store double %scalar.17.270, ptr %value.270, align 8
  %scalar.18.271 = fadd double %scalar.11.264, %scalar.17.270
  store double %scalar.18.271, ptr %value.271, align 8
  %scalar.19.272 = fsub double %scalar.18.271, %scalar.11.264
  store double %scalar.19.272, ptr %value.272, align 8
  %scalar.20.273 = fsub double %scalar.17.270, %scalar.19.272
  store double %scalar.20.273, ptr %value.273, align 8
  %scalar.21.31 = fadd double %scalar.18.271, %scalar.20.273
  store double %scalar.21.31, ptr %out.1, align 8
  ret void
}

define void @__ssa_asin_core_pack__asin_core_pack__planned_region_1(ptr %arg.0, ptr %arg.1, ptr %arg.2) {
entry:
  %load.0.46.0 = load i32, ptr %arg.1, align 4
  %address.0.46 = getelementptr double, ptr %arg.0, i32 %load.0.46.0
  %load.store.1.v = load double, ptr %arg.2, align 8
  store double %load.store.1.v, ptr %address.0.46, align 8
  ret void
}

define void @__ssa_asin_core_pack__asin_core_pack(ptr noalias %arg.0, ptr noalias %arg.1, ptr %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr noalias %arg.11, ptr noalias %arg.12, ptr noalias %arg.13, ptr noalias %arg.14, ptr noalias %arg.15, ptr noalias %arg.16, ptr noalias %arg.17, ptr noalias %arg.18, ptr noalias %arg.19, ptr noalias %arg.20, ptr noalias %arg.21, ptr noalias %arg.22, ptr noalias %arg.23, ptr noalias %arg.24, ptr noalias %arg.25, ptr noalias %arg.26, ptr noalias %arg.27, ptr noalias %arg.28, ptr noalias %arg.29, ptr noalias %arg.30, ptr noalias %arg.31, ptr noalias %arg.32, ptr noalias %arg.33, ptr noalias %arg.34, ptr noalias %arg.35, ptr noalias %arg.36, ptr noalias %arg.37, ptr noalias %arg.38, ptr noalias %arg.39, ptr noalias %arg.40, ptr noalias %arg.41, ptr noalias %arg.42, ptr noalias %arg.43, ptr noalias %arg.44, ptr noalias %arg.45, ptr noalias %arg.46, ptr noalias %arg.47, ptr noalias %arg.48, ptr noalias %arg.49, ptr noalias %arg.50, ptr noalias %arg.51, ptr noalias %arg.52, ptr %out.0) {
entry:
  %value.35 = alloca i64, i64 1, align 8
  %value.36 = alloca i64, i64 1, align 8
  %value.41 = alloca i64, i64 1, align 8
  %value.43 = alloca i64, i64 1, align 8
  %value.38 = alloca i64, i64 1, align 8
  %value.39 = alloca i1, i64 1, align 8
  %value.30 = alloca double, i64 1, align 8
  %value.31 = alloca double, i64 1, align 8
  %value.32 = alloca double, i64 1, align 8
  store i64 0, ptr %value.35, align 8
  store i64 1, ptr %value.36, align 8
  store i64 0, ptr %value.41, align 8
  store i64 1, ptr %value.43, align 8
  br label %loop_header
loop_header:
  %phi.37 = phi ptr [ %value.35, %entry ], [ %value.38, %loop_latch ]
  %load.6.39.0 = load i32, ptr %phi.37, align 4
  %load.6.39.1 = load i32, ptr %arg.0, align 4
  %scalar.6.39 = icmp slt i32 %load.6.39.0, %load.6.39.1
  store i1 %scalar.6.39, ptr %value.39, align 1
  br i1 %scalar.6.39, label %loop_body, label %loop_exit
loop_body:
  call void @__ssa_asin_core_pack__asin_core_pack__planned_region_0(ptr %arg.1, ptr %phi.37, ptr %value.30, ptr %value.31)
  call void @__ssa_asin_core_pack__asin_core(ptr %arg.3, ptr %arg.4, ptr %arg.5, ptr %arg.6, ptr %arg.7, ptr %arg.8, ptr %arg.9, ptr %arg.10, ptr %arg.11, ptr %arg.12, ptr %arg.13, ptr %arg.14, ptr %arg.15, ptr %arg.16, ptr %arg.17, ptr %arg.18, ptr %arg.19, ptr %arg.20, ptr %arg.21, ptr %arg.22, ptr %arg.23, ptr %arg.24, ptr %arg.25, ptr %arg.26, ptr %arg.27, ptr %value.31, ptr %value.30, ptr %arg.28, ptr %arg.29, ptr %arg.30, ptr %arg.31, ptr %arg.32, ptr %arg.33, ptr %arg.34, ptr %arg.35, ptr %arg.36, ptr %arg.37, ptr %arg.38, ptr %arg.39, ptr %arg.40, ptr %arg.41, ptr %arg.42, ptr %arg.43, ptr %arg.44, ptr %arg.45, ptr %arg.46, ptr %arg.47, ptr %arg.48, ptr %arg.49, ptr %arg.50, ptr %arg.51, ptr %arg.52, ptr %value.31, ptr %value.30, ptr %value.32)
  call void @__ssa_asin_core_pack__asin_core_pack__planned_region_1(ptr %arg.2, ptr %phi.37, ptr %value.32)
  br label %loop_latch
loop_latch:
  %load.16.38.0 = load i32, ptr %phi.37, align 4
  %load.16.38.1 = load i64, ptr %value.36, align 8
  %convert.16.38.1 = trunc i64 %load.16.38.1 to i32
  %scalar.16.38 = add i32 %load.16.38.0, %convert.16.38.1
  %declared.16.38 = sext i32 %scalar.16.38 to i64
  store i64 %declared.16.38, ptr %value.38, align 8
  br label %loop_header
loop_exit:
  %return.load.0.27 = load double, ptr %arg.2, align 8
  store double %return.load.0.27, ptr %out.0, align 8
  ret void
}

define void @__ssa_asin_core_pack__asin_core__planned_region_0(ptr noalias %arg.0, ptr %arg.1, ptr noalias %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr noalias %arg.11, ptr noalias %arg.12, ptr noalias %arg.13, ptr noalias %arg.14, ptr noalias %arg.15, ptr noalias %arg.16, ptr noalias %arg.17, ptr noalias %arg.18, ptr noalias %arg.19, ptr noalias %arg.20, ptr noalias %arg.21, ptr noalias %arg.22, ptr noalias %arg.23, ptr noalias %arg.24, ptr noalias %arg.25, ptr %arg.26, ptr noalias %arg.27, ptr %arg.28, ptr noalias %arg.29, ptr noalias %arg.30, ptr noalias %arg.31, ptr noalias %arg.32, ptr noalias %arg.33, ptr noalias %arg.34, ptr noalias %arg.35, ptr noalias %arg.36, ptr noalias %arg.37, ptr noalias %arg.38, ptr noalias %arg.39, ptr noalias %arg.40, ptr noalias %arg.41, ptr noalias %arg.42, ptr noalias %arg.43, ptr noalias %arg.44, ptr noalias %arg.45, ptr noalias %arg.46, ptr noalias %arg.47, ptr noalias %arg.48, ptr noalias %arg.49, ptr noalias %arg.50, ptr noalias %arg.51, ptr noalias %arg.52, ptr %arg.53, ptr %out.0, ptr %out.1, ptr %out.2, ptr %out.3, ptr %out.4, ptr %out.5, ptr %out.6, ptr %out.7, ptr %out.8, ptr %out.9, ptr %out.10, ptr %out.11, ptr %out.12, ptr %out.13, ptr %out.14, ptr %out.15, ptr %out.16, ptr %out.17, ptr %out.18, ptr %out.19, ptr %out.20, ptr %out.21, ptr %out.22, ptr %out.23, ptr %out.24, ptr %out.25, ptr %out.26, ptr %out.27, ptr %out.28, ptr %out.29, ptr %out.30, ptr %out.31, ptr %out.32, ptr %out.33, ptr %out.34, ptr %out.35, ptr %out.36, ptr %out.37, ptr %out.38, ptr %out.39, ptr %out.40, ptr %out.41, ptr %out.42, ptr %out.43, ptr %out.44, ptr %out.45, ptr %out.46, ptr %out.47, ptr %out.48) {
entry:
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
  %load.0.274.0 = load double, ptr %arg.0, align 8
  %load.0.274.1 = load double, ptr %arg.1, align 8
  %scalar.0.274 = fmul double %load.0.274.0, %load.0.274.1
  store double %scalar.0.274, ptr %value.274, align 8
  %scalar.1.275 = fneg double %scalar.0.274
  store double %scalar.1.275, ptr %value.275, align 8
  %scalar.2.276 = call double @llvm.fma.f64(double %load.0.274.0, double %load.0.274.1, double %scalar.1.275)
  store double %scalar.2.276, ptr %value.276, align 8
  %load.3.277.1 = load double, ptr %arg.28, align 8
  %scalar.3.277 = fmul double %load.0.274.0, %load.3.277.1
  store double %scalar.3.277, ptr %value.277, align 8
  %scalar.4.278 = fadd double %scalar.2.276, %scalar.3.277
  store double %scalar.4.278, ptr %value.278, align 8
  %load.5.279.0 = load double, ptr %arg.27, align 8
  %scalar.5.279 = fmul double %load.5.279.0, %load.0.274.1
  store double %scalar.5.279, ptr %value.279, align 8
  %scalar.6.280 = fadd double %scalar.4.278, %scalar.5.279
  store double %scalar.6.280, ptr %value.280, align 8
  %scalar.7.281 = fadd double %scalar.0.274, %scalar.6.280
  store double %scalar.7.281, ptr %value.281, align 8
  %scalar.8.282 = fsub double %scalar.7.281, %scalar.0.274
  store double %scalar.8.282, ptr %value.282, align 8
  %scalar.9.283 = fsub double %scalar.6.280, %scalar.8.282
  store double %scalar.9.283, ptr %value.283, align 8
  %scalar.10.27 = fadd double %scalar.7.281, %scalar.9.283
  store double %scalar.10.27, ptr %out.1, align 8
  %load.11.284.0 = load double, ptr %arg.2, align 8
  %scalar.11.284 = fadd double %load.11.284.0, %scalar.7.281
  store double %scalar.11.284, ptr %value.284, align 8
  %scalar.12.285 = fsub double %scalar.11.284, %load.11.284.0
  store double %scalar.12.285, ptr %value.285, align 8
  %scalar.13.286 = fsub double %scalar.11.284, %scalar.12.285
  store double %scalar.13.286, ptr %value.286, align 8
  %scalar.14.287 = fsub double %load.11.284.0, %scalar.13.286
  store double %scalar.14.287, ptr %value.287, align 8
  %scalar.15.288 = fsub double %scalar.7.281, %scalar.12.285
  store double %scalar.15.288, ptr %value.288, align 8
  %scalar.16.289 = fadd double %scalar.14.287, %scalar.15.288
  store double %scalar.16.289, ptr %value.289, align 8
  %load.17.290.1 = load double, ptr %arg.29, align 8
  %scalar.17.290 = fadd double %scalar.16.289, %load.17.290.1
  store double %scalar.17.290, ptr %value.290, align 8
  %scalar.18.291 = fadd double %scalar.17.290, %scalar.9.283
  store double %scalar.18.291, ptr %value.291, align 8
  %scalar.19.292 = fadd double %scalar.11.284, %scalar.18.291
  store double %scalar.19.292, ptr %value.292, align 8
  %scalar.20.293 = fsub double %scalar.19.292, %scalar.11.284
  store double %scalar.20.293, ptr %value.293, align 8
  %scalar.21.294 = fsub double %scalar.18.291, %scalar.20.293
  store double %scalar.21.294, ptr %value.294, align 8
  %scalar.22.28 = fadd double %scalar.19.292, %scalar.21.294
  store double %scalar.22.28, ptr %out.2, align 8
  %scalar.23.295 = fmul double %load.0.274.1, %scalar.19.292
  store double %scalar.23.295, ptr %value.295, align 8
  %scalar.24.296 = fneg double %scalar.23.295
  store double %scalar.24.296, ptr %value.296, align 8
  %scalar.25.297 = call double @llvm.fma.f64(double %load.0.274.1, double %scalar.19.292, double %scalar.24.296)
  store double %scalar.25.297, ptr %value.297, align 8
  %scalar.26.298 = fmul double %load.0.274.1, %scalar.21.294
  store double %scalar.26.298, ptr %value.298, align 8
  %scalar.27.299 = fadd double %scalar.25.297, %scalar.26.298
  store double %scalar.27.299, ptr %value.299, align 8
  %scalar.28.300 = fmul double %load.3.277.1, %scalar.19.292
  store double %scalar.28.300, ptr %value.300, align 8
  %scalar.29.301 = fadd double %scalar.27.299, %scalar.28.300
  store double %scalar.29.301, ptr %value.301, align 8
  %scalar.30.302 = fadd double %scalar.23.295, %scalar.29.301
  store double %scalar.30.302, ptr %value.302, align 8
  %scalar.31.303 = fsub double %scalar.30.302, %scalar.23.295
  store double %scalar.31.303, ptr %value.303, align 8
  %scalar.32.304 = fsub double %scalar.29.301, %scalar.31.303
  store double %scalar.32.304, ptr %value.304, align 8
  %scalar.33.29 = fadd double %scalar.30.302, %scalar.32.304
  store double %scalar.33.29, ptr %out.3, align 8
  %load.34.305.0 = load double, ptr %arg.3, align 8
  %scalar.34.305 = fadd double %load.34.305.0, %scalar.30.302
  store double %scalar.34.305, ptr %value.305, align 8
  %scalar.35.306 = fsub double %scalar.34.305, %load.34.305.0
  store double %scalar.35.306, ptr %value.306, align 8
  %scalar.36.307 = fsub double %scalar.34.305, %scalar.35.306
  store double %scalar.36.307, ptr %value.307, align 8
  %scalar.37.308 = fsub double %load.34.305.0, %scalar.36.307
  store double %scalar.37.308, ptr %value.308, align 8
  %scalar.38.309 = fsub double %scalar.30.302, %scalar.35.306
  store double %scalar.38.309, ptr %value.309, align 8
  %scalar.39.310 = fadd double %scalar.37.308, %scalar.38.309
  store double %scalar.39.310, ptr %value.310, align 8
  %load.40.311.1 = load double, ptr %arg.30, align 8
  %scalar.40.311 = fadd double %scalar.39.310, %load.40.311.1
  store double %scalar.40.311, ptr %value.311, align 8
  %scalar.41.312 = fadd double %scalar.40.311, %scalar.32.304
  store double %scalar.41.312, ptr %value.312, align 8
  %scalar.42.313 = fadd double %scalar.34.305, %scalar.41.312
  store double %scalar.42.313, ptr %value.313, align 8
  %scalar.43.314 = fsub double %scalar.42.313, %scalar.34.305
  store double %scalar.43.314, ptr %value.314, align 8
  %scalar.44.315 = fsub double %scalar.41.312, %scalar.43.314
  store double %scalar.44.315, ptr %value.315, align 8
  %scalar.45.30 = fadd double %scalar.42.313, %scalar.44.315
  store double %scalar.45.30, ptr %out.4, align 8
  %scalar.46.316 = fmul double %load.0.274.1, %scalar.42.313
  store double %scalar.46.316, ptr %value.316, align 8
  %scalar.47.317 = fneg double %scalar.46.316
  store double %scalar.47.317, ptr %value.317, align 8
  %scalar.48.318 = call double @llvm.fma.f64(double %load.0.274.1, double %scalar.42.313, double %scalar.47.317)
  store double %scalar.48.318, ptr %value.318, align 8
  %scalar.49.319 = fmul double %load.0.274.1, %scalar.44.315
  store double %scalar.49.319, ptr %value.319, align 8
  %scalar.50.320 = fadd double %scalar.48.318, %scalar.49.319
  store double %scalar.50.320, ptr %value.320, align 8
  %scalar.51.321 = fmul double %load.3.277.1, %scalar.42.313
  store double %scalar.51.321, ptr %value.321, align 8
  %scalar.52.322 = fadd double %scalar.50.320, %scalar.51.321
  store double %scalar.52.322, ptr %value.322, align 8
  %scalar.53.323 = fadd double %scalar.46.316, %scalar.52.322
  store double %scalar.53.323, ptr %value.323, align 8
  %scalar.54.324 = fsub double %scalar.53.323, %scalar.46.316
  store double %scalar.54.324, ptr %value.324, align 8
  %scalar.55.325 = fsub double %scalar.52.322, %scalar.54.324
  store double %scalar.55.325, ptr %value.325, align 8
  %scalar.56.31 = fadd double %scalar.53.323, %scalar.55.325
  store double %scalar.56.31, ptr %out.5, align 8
  %load.57.326.0 = load double, ptr %arg.4, align 8
  %scalar.57.326 = fadd double %load.57.326.0, %scalar.53.323
  store double %scalar.57.326, ptr %value.326, align 8
  %scalar.58.327 = fsub double %scalar.57.326, %load.57.326.0
  store double %scalar.58.327, ptr %value.327, align 8
  %scalar.59.328 = fsub double %scalar.57.326, %scalar.58.327
  store double %scalar.59.328, ptr %value.328, align 8
  %scalar.60.329 = fsub double %load.57.326.0, %scalar.59.328
  store double %scalar.60.329, ptr %value.329, align 8
  %scalar.61.330 = fsub double %scalar.53.323, %scalar.58.327
  store double %scalar.61.330, ptr %value.330, align 8
  %scalar.62.331 = fadd double %scalar.60.329, %scalar.61.330
  store double %scalar.62.331, ptr %value.331, align 8
  %load.63.332.1 = load double, ptr %arg.31, align 8
  %scalar.63.332 = fadd double %scalar.62.331, %load.63.332.1
  store double %scalar.63.332, ptr %value.332, align 8
  %scalar.64.333 = fadd double %scalar.63.332, %scalar.55.325
  store double %scalar.64.333, ptr %value.333, align 8
  %scalar.65.334 = fadd double %scalar.57.326, %scalar.64.333
  store double %scalar.65.334, ptr %value.334, align 8
  %scalar.66.335 = fsub double %scalar.65.334, %scalar.57.326
  store double %scalar.66.335, ptr %value.335, align 8
  %scalar.67.336 = fsub double %scalar.64.333, %scalar.66.335
  store double %scalar.67.336, ptr %value.336, align 8
  %scalar.68.32 = fadd double %scalar.65.334, %scalar.67.336
  store double %scalar.68.32, ptr %out.6, align 8
  %scalar.69.337 = fmul double %load.0.274.1, %scalar.65.334
  store double %scalar.69.337, ptr %value.337, align 8
  %scalar.70.338 = fneg double %scalar.69.337
  store double %scalar.70.338, ptr %value.338, align 8
  %scalar.71.339 = call double @llvm.fma.f64(double %load.0.274.1, double %scalar.65.334, double %scalar.70.338)
  store double %scalar.71.339, ptr %value.339, align 8
  %scalar.72.340 = fmul double %load.0.274.1, %scalar.67.336
  store double %scalar.72.340, ptr %value.340, align 8
  %scalar.73.341 = fadd double %scalar.71.339, %scalar.72.340
  store double %scalar.73.341, ptr %value.341, align 8
  %scalar.74.342 = fmul double %load.3.277.1, %scalar.65.334
  store double %scalar.74.342, ptr %value.342, align 8
  %scalar.75.343 = fadd double %scalar.73.341, %scalar.74.342
  store double %scalar.75.343, ptr %value.343, align 8
  %scalar.76.344 = fadd double %scalar.69.337, %scalar.75.343
  store double %scalar.76.344, ptr %value.344, align 8
  %scalar.77.345 = fsub double %scalar.76.344, %scalar.69.337
  store double %scalar.77.345, ptr %value.345, align 8
  %scalar.78.346 = fsub double %scalar.75.343, %scalar.77.345
  store double %scalar.78.346, ptr %value.346, align 8
  %scalar.79.33 = fadd double %scalar.76.344, %scalar.78.346
  store double %scalar.79.33, ptr %out.7, align 8
  %load.80.347.0 = load double, ptr %arg.5, align 8
  %scalar.80.347 = fadd double %load.80.347.0, %scalar.76.344
  store double %scalar.80.347, ptr %value.347, align 8
  %scalar.81.348 = fsub double %scalar.80.347, %load.80.347.0
  store double %scalar.81.348, ptr %value.348, align 8
  %scalar.82.349 = fsub double %scalar.80.347, %scalar.81.348
  store double %scalar.82.349, ptr %value.349, align 8
  %scalar.83.350 = fsub double %load.80.347.0, %scalar.82.349
  store double %scalar.83.350, ptr %value.350, align 8
  %scalar.84.351 = fsub double %scalar.76.344, %scalar.81.348
  store double %scalar.84.351, ptr %value.351, align 8
  %scalar.85.352 = fadd double %scalar.83.350, %scalar.84.351
  store double %scalar.85.352, ptr %value.352, align 8
  %load.86.353.1 = load double, ptr %arg.32, align 8
  %scalar.86.353 = fadd double %scalar.85.352, %load.86.353.1
  store double %scalar.86.353, ptr %value.353, align 8
  %scalar.87.354 = fadd double %scalar.86.353, %scalar.78.346
  store double %scalar.87.354, ptr %value.354, align 8
  %scalar.88.355 = fadd double %scalar.80.347, %scalar.87.354
  store double %scalar.88.355, ptr %value.355, align 8
  %scalar.89.356 = fsub double %scalar.88.355, %scalar.80.347
  store double %scalar.89.356, ptr %value.356, align 8
  %scalar.90.357 = fsub double %scalar.87.354, %scalar.89.356
  store double %scalar.90.357, ptr %value.357, align 8
  %scalar.91.34 = fadd double %scalar.88.355, %scalar.90.357
  store double %scalar.91.34, ptr %out.8, align 8
  %scalar.92.358 = fmul double %load.0.274.1, %scalar.88.355
  store double %scalar.92.358, ptr %value.358, align 8
  %scalar.93.359 = fneg double %scalar.92.358
  store double %scalar.93.359, ptr %value.359, align 8
  %scalar.94.360 = call double @llvm.fma.f64(double %load.0.274.1, double %scalar.88.355, double %scalar.93.359)
  store double %scalar.94.360, ptr %value.360, align 8
  %scalar.95.361 = fmul double %load.0.274.1, %scalar.90.357
  store double %scalar.95.361, ptr %value.361, align 8
  %scalar.96.362 = fadd double %scalar.94.360, %scalar.95.361
  store double %scalar.96.362, ptr %value.362, align 8
  %scalar.97.363 = fmul double %load.3.277.1, %scalar.88.355
  store double %scalar.97.363, ptr %value.363, align 8
  %scalar.98.364 = fadd double %scalar.96.362, %scalar.97.363
  store double %scalar.98.364, ptr %value.364, align 8
  %scalar.99.365 = fadd double %scalar.92.358, %scalar.98.364
  store double %scalar.99.365, ptr %value.365, align 8
  %scalar.100.366 = fsub double %scalar.99.365, %scalar.92.358
  store double %scalar.100.366, ptr %value.366, align 8
  %scalar.101.367 = fsub double %scalar.98.364, %scalar.100.366
  store double %scalar.101.367, ptr %value.367, align 8
  %scalar.102.35 = fadd double %scalar.99.365, %scalar.101.367
  store double %scalar.102.35, ptr %out.9, align 8
  %load.103.368.0 = load double, ptr %arg.6, align 8
  %scalar.103.368 = fadd double %load.103.368.0, %scalar.99.365
  store double %scalar.103.368, ptr %value.368, align 8
  %scalar.104.369 = fsub double %scalar.103.368, %load.103.368.0
  store double %scalar.104.369, ptr %value.369, align 8
  %scalar.105.370 = fsub double %scalar.103.368, %scalar.104.369
  store double %scalar.105.370, ptr %value.370, align 8
  %scalar.106.371 = fsub double %load.103.368.0, %scalar.105.370
  store double %scalar.106.371, ptr %value.371, align 8
  %scalar.107.372 = fsub double %scalar.99.365, %scalar.104.369
  store double %scalar.107.372, ptr %value.372, align 8
  %scalar.108.373 = fadd double %scalar.106.371, %scalar.107.372
  store double %scalar.108.373, ptr %value.373, align 8
  %load.109.374.1 = load double, ptr %arg.33, align 8
  %scalar.109.374 = fadd double %scalar.108.373, %load.109.374.1
  store double %scalar.109.374, ptr %value.374, align 8
  %scalar.110.375 = fadd double %scalar.109.374, %scalar.101.367
  store double %scalar.110.375, ptr %value.375, align 8
  %scalar.111.376 = fadd double %scalar.103.368, %scalar.110.375
  store double %scalar.111.376, ptr %value.376, align 8
  %scalar.112.377 = fsub double %scalar.111.376, %scalar.103.368
  store double %scalar.112.377, ptr %value.377, align 8
  %scalar.113.378 = fsub double %scalar.110.375, %scalar.112.377
  store double %scalar.113.378, ptr %value.378, align 8
  %scalar.114.36 = fadd double %scalar.111.376, %scalar.113.378
  store double %scalar.114.36, ptr %out.10, align 8
  %scalar.115.379 = fmul double %load.0.274.1, %scalar.111.376
  store double %scalar.115.379, ptr %value.379, align 8
  %scalar.116.380 = fneg double %scalar.115.379
  store double %scalar.116.380, ptr %value.380, align 8
  %scalar.117.381 = call double @llvm.fma.f64(double %load.0.274.1, double %scalar.111.376, double %scalar.116.380)
  store double %scalar.117.381, ptr %value.381, align 8
  %scalar.118.382 = fmul double %load.0.274.1, %scalar.113.378
  store double %scalar.118.382, ptr %value.382, align 8
  %scalar.119.383 = fadd double %scalar.117.381, %scalar.118.382
  store double %scalar.119.383, ptr %value.383, align 8
  %scalar.120.384 = fmul double %load.3.277.1, %scalar.111.376
  store double %scalar.120.384, ptr %value.384, align 8
  %scalar.121.385 = fadd double %scalar.119.383, %scalar.120.384
  store double %scalar.121.385, ptr %value.385, align 8
  %scalar.122.386 = fadd double %scalar.115.379, %scalar.121.385
  store double %scalar.122.386, ptr %value.386, align 8
  %scalar.123.387 = fsub double %scalar.122.386, %scalar.115.379
  store double %scalar.123.387, ptr %value.387, align 8
  %scalar.124.388 = fsub double %scalar.121.385, %scalar.123.387
  store double %scalar.124.388, ptr %value.388, align 8
  %scalar.125.37 = fadd double %scalar.122.386, %scalar.124.388
  store double %scalar.125.37, ptr %out.11, align 8
  %load.126.389.0 = load double, ptr %arg.7, align 8
  %scalar.126.389 = fadd double %load.126.389.0, %scalar.122.386
  store double %scalar.126.389, ptr %value.389, align 8
  %scalar.127.390 = fsub double %scalar.126.389, %load.126.389.0
  store double %scalar.127.390, ptr %value.390, align 8
  %scalar.128.391 = fsub double %scalar.126.389, %scalar.127.390
  store double %scalar.128.391, ptr %value.391, align 8
  %scalar.129.392 = fsub double %load.126.389.0, %scalar.128.391
  store double %scalar.129.392, ptr %value.392, align 8
  %scalar.130.393 = fsub double %scalar.122.386, %scalar.127.390
  store double %scalar.130.393, ptr %value.393, align 8
  %scalar.131.394 = fadd double %scalar.129.392, %scalar.130.393
  store double %scalar.131.394, ptr %value.394, align 8
  %load.132.395.1 = load double, ptr %arg.34, align 8
  %scalar.132.395 = fadd double %scalar.131.394, %load.132.395.1
  store double %scalar.132.395, ptr %value.395, align 8
  %scalar.133.396 = fadd double %scalar.132.395, %scalar.124.388
  store double %scalar.133.396, ptr %value.396, align 8
  %scalar.134.397 = fadd double %scalar.126.389, %scalar.133.396
  store double %scalar.134.397, ptr %value.397, align 8
  %scalar.135.398 = fsub double %scalar.134.397, %scalar.126.389
  store double %scalar.135.398, ptr %value.398, align 8
  %scalar.136.399 = fsub double %scalar.133.396, %scalar.135.398
  store double %scalar.136.399, ptr %value.399, align 8
  %scalar.137.38 = fadd double %scalar.134.397, %scalar.136.399
  store double %scalar.137.38, ptr %out.12, align 8
  %scalar.138.400 = fmul double %load.0.274.1, %scalar.134.397
  store double %scalar.138.400, ptr %value.400, align 8
  %scalar.139.401 = fneg double %scalar.138.400
  store double %scalar.139.401, ptr %value.401, align 8
  %scalar.140.402 = call double @llvm.fma.f64(double %load.0.274.1, double %scalar.134.397, double %scalar.139.401)
  store double %scalar.140.402, ptr %value.402, align 8
  %scalar.141.403 = fmul double %load.0.274.1, %scalar.136.399
  store double %scalar.141.403, ptr %value.403, align 8
  %scalar.142.404 = fadd double %scalar.140.402, %scalar.141.403
  store double %scalar.142.404, ptr %value.404, align 8
  %scalar.143.405 = fmul double %load.3.277.1, %scalar.134.397
  store double %scalar.143.405, ptr %value.405, align 8
  %scalar.144.406 = fadd double %scalar.142.404, %scalar.143.405
  store double %scalar.144.406, ptr %value.406, align 8
  %scalar.145.407 = fadd double %scalar.138.400, %scalar.144.406
  store double %scalar.145.407, ptr %value.407, align 8
  %scalar.146.408 = fsub double %scalar.145.407, %scalar.138.400
  store double %scalar.146.408, ptr %value.408, align 8
  %scalar.147.409 = fsub double %scalar.144.406, %scalar.146.408
  store double %scalar.147.409, ptr %value.409, align 8
  %scalar.148.39 = fadd double %scalar.145.407, %scalar.147.409
  store double %scalar.148.39, ptr %out.13, align 8
  %load.149.410.0 = load double, ptr %arg.8, align 8
  %scalar.149.410 = fadd double %load.149.410.0, %scalar.145.407
  store double %scalar.149.410, ptr %value.410, align 8
  %scalar.150.411 = fsub double %scalar.149.410, %load.149.410.0
  store double %scalar.150.411, ptr %value.411, align 8
  %scalar.151.412 = fsub double %scalar.149.410, %scalar.150.411
  store double %scalar.151.412, ptr %value.412, align 8
  %scalar.152.413 = fsub double %load.149.410.0, %scalar.151.412
  store double %scalar.152.413, ptr %value.413, align 8
  %scalar.153.414 = fsub double %scalar.145.407, %scalar.150.411
  store double %scalar.153.414, ptr %value.414, align 8
  %scalar.154.415 = fadd double %scalar.152.413, %scalar.153.414
  store double %scalar.154.415, ptr %value.415, align 8
  %load.155.416.1 = load double, ptr %arg.35, align 8
  %scalar.155.416 = fadd double %scalar.154.415, %load.155.416.1
  store double %scalar.155.416, ptr %value.416, align 8
  %scalar.156.417 = fadd double %scalar.155.416, %scalar.147.409
  store double %scalar.156.417, ptr %value.417, align 8
  %scalar.157.418 = fadd double %scalar.149.410, %scalar.156.417
  store double %scalar.157.418, ptr %value.418, align 8
  %scalar.158.419 = fsub double %scalar.157.418, %scalar.149.410
  store double %scalar.158.419, ptr %value.419, align 8
  %scalar.159.420 = fsub double %scalar.156.417, %scalar.158.419
  store double %scalar.159.420, ptr %value.420, align 8
  %scalar.160.40 = fadd double %scalar.157.418, %scalar.159.420
  store double %scalar.160.40, ptr %out.14, align 8
  %scalar.161.421 = fmul double %load.0.274.1, %scalar.157.418
  store double %scalar.161.421, ptr %value.421, align 8
  %scalar.162.422 = fneg double %scalar.161.421
  store double %scalar.162.422, ptr %value.422, align 8
  %scalar.163.423 = call double @llvm.fma.f64(double %load.0.274.1, double %scalar.157.418, double %scalar.162.422)
  store double %scalar.163.423, ptr %value.423, align 8
  %scalar.164.424 = fmul double %load.0.274.1, %scalar.159.420
  store double %scalar.164.424, ptr %value.424, align 8
  %scalar.165.425 = fadd double %scalar.163.423, %scalar.164.424
  store double %scalar.165.425, ptr %value.425, align 8
  %scalar.166.426 = fmul double %load.3.277.1, %scalar.157.418
  store double %scalar.166.426, ptr %value.426, align 8
  %scalar.167.427 = fadd double %scalar.165.425, %scalar.166.426
  store double %scalar.167.427, ptr %value.427, align 8
  %scalar.168.428 = fadd double %scalar.161.421, %scalar.167.427
  store double %scalar.168.428, ptr %value.428, align 8
  %scalar.169.429 = fsub double %scalar.168.428, %scalar.161.421
  store double %scalar.169.429, ptr %value.429, align 8
  %scalar.170.430 = fsub double %scalar.167.427, %scalar.169.429
  store double %scalar.170.430, ptr %value.430, align 8
  %scalar.171.41 = fadd double %scalar.168.428, %scalar.170.430
  store double %scalar.171.41, ptr %out.15, align 8
  %load.172.431.0 = load double, ptr %arg.9, align 8
  %scalar.172.431 = fadd double %load.172.431.0, %scalar.168.428
  store double %scalar.172.431, ptr %value.431, align 8
  %scalar.173.432 = fsub double %scalar.172.431, %load.172.431.0
  store double %scalar.173.432, ptr %value.432, align 8
  %scalar.174.433 = fsub double %scalar.172.431, %scalar.173.432
  store double %scalar.174.433, ptr %value.433, align 8
  %scalar.175.434 = fsub double %load.172.431.0, %scalar.174.433
  store double %scalar.175.434, ptr %value.434, align 8
  %scalar.176.435 = fsub double %scalar.168.428, %scalar.173.432
  store double %scalar.176.435, ptr %value.435, align 8
  %scalar.177.436 = fadd double %scalar.175.434, %scalar.176.435
  store double %scalar.177.436, ptr %value.436, align 8
  %load.178.437.1 = load double, ptr %arg.36, align 8
  %scalar.178.437 = fadd double %scalar.177.436, %load.178.437.1
  store double %scalar.178.437, ptr %value.437, align 8
  %scalar.179.438 = fadd double %scalar.178.437, %scalar.170.430
  store double %scalar.179.438, ptr %value.438, align 8
  %scalar.180.439 = fadd double %scalar.172.431, %scalar.179.438
  store double %scalar.180.439, ptr %value.439, align 8
  %scalar.181.440 = fsub double %scalar.180.439, %scalar.172.431
  store double %scalar.181.440, ptr %value.440, align 8
  %scalar.182.441 = fsub double %scalar.179.438, %scalar.181.440
  store double %scalar.182.441, ptr %value.441, align 8
  %scalar.183.42 = fadd double %scalar.180.439, %scalar.182.441
  store double %scalar.183.42, ptr %out.16, align 8
  %scalar.184.442 = fmul double %load.0.274.1, %scalar.180.439
  store double %scalar.184.442, ptr %value.442, align 8
  %scalar.185.443 = fneg double %scalar.184.442
  store double %scalar.185.443, ptr %value.443, align 8
  %scalar.186.444 = call double @llvm.fma.f64(double %load.0.274.1, double %scalar.180.439, double %scalar.185.443)
  store double %scalar.186.444, ptr %value.444, align 8
  %scalar.187.445 = fmul double %load.0.274.1, %scalar.182.441
  store double %scalar.187.445, ptr %value.445, align 8
  %scalar.188.446 = fadd double %scalar.186.444, %scalar.187.445
  store double %scalar.188.446, ptr %value.446, align 8
  %scalar.189.447 = fmul double %load.3.277.1, %scalar.180.439
  store double %scalar.189.447, ptr %value.447, align 8
  %scalar.190.448 = fadd double %scalar.188.446, %scalar.189.447
  store double %scalar.190.448, ptr %value.448, align 8
  %scalar.191.449 = fadd double %scalar.184.442, %scalar.190.448
  store double %scalar.191.449, ptr %value.449, align 8
  %scalar.192.450 = fsub double %scalar.191.449, %scalar.184.442
  store double %scalar.192.450, ptr %value.450, align 8
  %scalar.193.451 = fsub double %scalar.190.448, %scalar.192.450
  store double %scalar.193.451, ptr %value.451, align 8
  %scalar.194.43 = fadd double %scalar.191.449, %scalar.193.451
  store double %scalar.194.43, ptr %out.17, align 8
  %load.195.452.0 = load double, ptr %arg.10, align 8
  %scalar.195.452 = fadd double %load.195.452.0, %scalar.191.449
  store double %scalar.195.452, ptr %value.452, align 8
  %scalar.196.453 = fsub double %scalar.195.452, %load.195.452.0
  store double %scalar.196.453, ptr %value.453, align 8
  %scalar.197.454 = fsub double %scalar.195.452, %scalar.196.453
  store double %scalar.197.454, ptr %value.454, align 8
  %scalar.198.455 = fsub double %load.195.452.0, %scalar.197.454
  store double %scalar.198.455, ptr %value.455, align 8
  %scalar.199.456 = fsub double %scalar.191.449, %scalar.196.453
  store double %scalar.199.456, ptr %value.456, align 8
  %scalar.200.457 = fadd double %scalar.198.455, %scalar.199.456
  store double %scalar.200.457, ptr %value.457, align 8
  %load.201.458.1 = load double, ptr %arg.37, align 8
  %scalar.201.458 = fadd double %scalar.200.457, %load.201.458.1
  store double %scalar.201.458, ptr %value.458, align 8
  %scalar.202.459 = fadd double %scalar.201.458, %scalar.193.451
  store double %scalar.202.459, ptr %value.459, align 8
  %scalar.203.460 = fadd double %scalar.195.452, %scalar.202.459
  store double %scalar.203.460, ptr %value.460, align 8
  %scalar.204.461 = fsub double %scalar.203.460, %scalar.195.452
  store double %scalar.204.461, ptr %value.461, align 8
  %scalar.205.462 = fsub double %scalar.202.459, %scalar.204.461
  store double %scalar.205.462, ptr %value.462, align 8
  %scalar.206.44 = fadd double %scalar.203.460, %scalar.205.462
  store double %scalar.206.44, ptr %out.18, align 8
  %scalar.207.463 = fmul double %load.0.274.1, %scalar.203.460
  store double %scalar.207.463, ptr %value.463, align 8
  %scalar.208.464 = fneg double %scalar.207.463
  store double %scalar.208.464, ptr %value.464, align 8
  %scalar.209.465 = call double @llvm.fma.f64(double %load.0.274.1, double %scalar.203.460, double %scalar.208.464)
  store double %scalar.209.465, ptr %value.465, align 8
  %scalar.210.466 = fmul double %load.0.274.1, %scalar.205.462
  store double %scalar.210.466, ptr %value.466, align 8
  %scalar.211.467 = fadd double %scalar.209.465, %scalar.210.466
  store double %scalar.211.467, ptr %value.467, align 8
  %scalar.212.468 = fmul double %load.3.277.1, %scalar.203.460
  store double %scalar.212.468, ptr %value.468, align 8
  %scalar.213.469 = fadd double %scalar.211.467, %scalar.212.468
  store double %scalar.213.469, ptr %value.469, align 8
  %scalar.214.470 = fadd double %scalar.207.463, %scalar.213.469
  store double %scalar.214.470, ptr %value.470, align 8
  %scalar.215.471 = fsub double %scalar.214.470, %scalar.207.463
  store double %scalar.215.471, ptr %value.471, align 8
  %scalar.216.472 = fsub double %scalar.213.469, %scalar.215.471
  store double %scalar.216.472, ptr %value.472, align 8
  %scalar.217.45 = fadd double %scalar.214.470, %scalar.216.472
  store double %scalar.217.45, ptr %out.19, align 8
  %load.218.473.0 = load double, ptr %arg.11, align 8
  %scalar.218.473 = fadd double %load.218.473.0, %scalar.214.470
  store double %scalar.218.473, ptr %value.473, align 8
  %scalar.219.474 = fsub double %scalar.218.473, %load.218.473.0
  store double %scalar.219.474, ptr %value.474, align 8
  %scalar.220.475 = fsub double %scalar.218.473, %scalar.219.474
  store double %scalar.220.475, ptr %value.475, align 8
  %scalar.221.476 = fsub double %load.218.473.0, %scalar.220.475
  store double %scalar.221.476, ptr %value.476, align 8
  %scalar.222.477 = fsub double %scalar.214.470, %scalar.219.474
  store double %scalar.222.477, ptr %value.477, align 8
  %scalar.223.478 = fadd double %scalar.221.476, %scalar.222.477
  store double %scalar.223.478, ptr %value.478, align 8
  %load.224.479.1 = load double, ptr %arg.38, align 8
  %scalar.224.479 = fadd double %scalar.223.478, %load.224.479.1
  store double %scalar.224.479, ptr %value.479, align 8
  %scalar.225.480 = fadd double %scalar.224.479, %scalar.216.472
  store double %scalar.225.480, ptr %value.480, align 8
  %scalar.226.481 = fadd double %scalar.218.473, %scalar.225.480
  store double %scalar.226.481, ptr %value.481, align 8
  %scalar.227.482 = fsub double %scalar.226.481, %scalar.218.473
  store double %scalar.227.482, ptr %value.482, align 8
  %scalar.228.483 = fsub double %scalar.225.480, %scalar.227.482
  store double %scalar.228.483, ptr %value.483, align 8
  %scalar.229.46 = fadd double %scalar.226.481, %scalar.228.483
  store double %scalar.229.46, ptr %out.20, align 8
  %scalar.230.484 = fmul double %load.0.274.1, %scalar.226.481
  store double %scalar.230.484, ptr %value.484, align 8
  %scalar.231.485 = fneg double %scalar.230.484
  store double %scalar.231.485, ptr %value.485, align 8
  %scalar.232.486 = call double @llvm.fma.f64(double %load.0.274.1, double %scalar.226.481, double %scalar.231.485)
  store double %scalar.232.486, ptr %value.486, align 8
  %scalar.233.487 = fmul double %load.0.274.1, %scalar.228.483
  store double %scalar.233.487, ptr %value.487, align 8
  %scalar.234.488 = fadd double %scalar.232.486, %scalar.233.487
  store double %scalar.234.488, ptr %value.488, align 8
  %scalar.235.489 = fmul double %load.3.277.1, %scalar.226.481
  store double %scalar.235.489, ptr %value.489, align 8
  %scalar.236.490 = fadd double %scalar.234.488, %scalar.235.489
  store double %scalar.236.490, ptr %value.490, align 8
  %scalar.237.491 = fadd double %scalar.230.484, %scalar.236.490
  store double %scalar.237.491, ptr %value.491, align 8
  %scalar.238.492 = fsub double %scalar.237.491, %scalar.230.484
  store double %scalar.238.492, ptr %value.492, align 8
  %scalar.239.493 = fsub double %scalar.236.490, %scalar.238.492
  store double %scalar.239.493, ptr %value.493, align 8
  %scalar.240.47 = fadd double %scalar.237.491, %scalar.239.493
  store double %scalar.240.47, ptr %out.21, align 8
  %load.241.494.0 = load double, ptr %arg.12, align 8
  %scalar.241.494 = fadd double %load.241.494.0, %scalar.237.491
  store double %scalar.241.494, ptr %value.494, align 8
  %scalar.242.495 = fsub double %scalar.241.494, %load.241.494.0
  store double %scalar.242.495, ptr %value.495, align 8
  %scalar.243.496 = fsub double %scalar.241.494, %scalar.242.495
  store double %scalar.243.496, ptr %value.496, align 8
  %scalar.244.497 = fsub double %load.241.494.0, %scalar.243.496
  store double %scalar.244.497, ptr %value.497, align 8
  %scalar.245.498 = fsub double %scalar.237.491, %scalar.242.495
  store double %scalar.245.498, ptr %value.498, align 8
  %scalar.246.499 = fadd double %scalar.244.497, %scalar.245.498
  store double %scalar.246.499, ptr %value.499, align 8
  %load.247.500.1 = load double, ptr %arg.39, align 8
  %scalar.247.500 = fadd double %scalar.246.499, %load.247.500.1
  store double %scalar.247.500, ptr %value.500, align 8
  %scalar.248.501 = fadd double %scalar.247.500, %scalar.239.493
  store double %scalar.248.501, ptr %value.501, align 8
  %scalar.249.502 = fadd double %scalar.241.494, %scalar.248.501
  store double %scalar.249.502, ptr %value.502, align 8
  %scalar.250.503 = fsub double %scalar.249.502, %scalar.241.494
  store double %scalar.250.503, ptr %value.503, align 8
  %scalar.251.504 = fsub double %scalar.248.501, %scalar.250.503
  store double %scalar.251.504, ptr %value.504, align 8
  %scalar.252.48 = fadd double %scalar.249.502, %scalar.251.504
  store double %scalar.252.48, ptr %out.22, align 8
  %scalar.253.505 = fmul double %load.0.274.1, %scalar.249.502
  store double %scalar.253.505, ptr %value.505, align 8
  %scalar.254.506 = fneg double %scalar.253.505
  store double %scalar.254.506, ptr %value.506, align 8
  %scalar.255.507 = call double @llvm.fma.f64(double %load.0.274.1, double %scalar.249.502, double %scalar.254.506)
  store double %scalar.255.507, ptr %value.507, align 8
  %scalar.256.508 = fmul double %load.0.274.1, %scalar.251.504
  store double %scalar.256.508, ptr %value.508, align 8
  %scalar.257.509 = fadd double %scalar.255.507, %scalar.256.508
  store double %scalar.257.509, ptr %value.509, align 8
  %scalar.258.510 = fmul double %load.3.277.1, %scalar.249.502
  store double %scalar.258.510, ptr %value.510, align 8
  %scalar.259.511 = fadd double %scalar.257.509, %scalar.258.510
  store double %scalar.259.511, ptr %value.511, align 8
  %scalar.260.512 = fadd double %scalar.253.505, %scalar.259.511
  store double %scalar.260.512, ptr %value.512, align 8
  %scalar.261.513 = fsub double %scalar.260.512, %scalar.253.505
  store double %scalar.261.513, ptr %value.513, align 8
  %scalar.262.514 = fsub double %scalar.259.511, %scalar.261.513
  store double %scalar.262.514, ptr %value.514, align 8
  %scalar.263.49 = fadd double %scalar.260.512, %scalar.262.514
  store double %scalar.263.49, ptr %out.23, align 8
  %load.264.515.0 = load double, ptr %arg.13, align 8
  %scalar.264.515 = fadd double %load.264.515.0, %scalar.260.512
  store double %scalar.264.515, ptr %value.515, align 8
  %scalar.265.516 = fsub double %scalar.264.515, %load.264.515.0
  store double %scalar.265.516, ptr %value.516, align 8
  %scalar.266.517 = fsub double %scalar.264.515, %scalar.265.516
  store double %scalar.266.517, ptr %value.517, align 8
  %scalar.267.518 = fsub double %load.264.515.0, %scalar.266.517
  store double %scalar.267.518, ptr %value.518, align 8
  %scalar.268.519 = fsub double %scalar.260.512, %scalar.265.516
  store double %scalar.268.519, ptr %value.519, align 8
  %scalar.269.520 = fadd double %scalar.267.518, %scalar.268.519
  store double %scalar.269.520, ptr %value.520, align 8
  %load.270.521.1 = load double, ptr %arg.40, align 8
  %scalar.270.521 = fadd double %scalar.269.520, %load.270.521.1
  store double %scalar.270.521, ptr %value.521, align 8
  %scalar.271.522 = fadd double %scalar.270.521, %scalar.262.514
  store double %scalar.271.522, ptr %value.522, align 8
  %scalar.272.523 = fadd double %scalar.264.515, %scalar.271.522
  store double %scalar.272.523, ptr %value.523, align 8
  %scalar.273.524 = fsub double %scalar.272.523, %scalar.264.515
  store double %scalar.273.524, ptr %value.524, align 8
  %scalar.274.525 = fsub double %scalar.271.522, %scalar.273.524
  store double %scalar.274.525, ptr %value.525, align 8
  %scalar.275.50 = fadd double %scalar.272.523, %scalar.274.525
  store double %scalar.275.50, ptr %out.24, align 8
  %scalar.276.526 = fmul double %load.0.274.1, %scalar.272.523
  store double %scalar.276.526, ptr %value.526, align 8
  %scalar.277.527 = fneg double %scalar.276.526
  store double %scalar.277.527, ptr %value.527, align 8
  %scalar.278.528 = call double @llvm.fma.f64(double %load.0.274.1, double %scalar.272.523, double %scalar.277.527)
  store double %scalar.278.528, ptr %value.528, align 8
  %scalar.279.529 = fmul double %load.0.274.1, %scalar.274.525
  store double %scalar.279.529, ptr %value.529, align 8
  %scalar.280.530 = fadd double %scalar.278.528, %scalar.279.529
  store double %scalar.280.530, ptr %value.530, align 8
  %scalar.281.531 = fmul double %load.3.277.1, %scalar.272.523
  store double %scalar.281.531, ptr %value.531, align 8
  %scalar.282.532 = fadd double %scalar.280.530, %scalar.281.531
  store double %scalar.282.532, ptr %value.532, align 8
  %scalar.283.533 = fadd double %scalar.276.526, %scalar.282.532
  store double %scalar.283.533, ptr %value.533, align 8
  %scalar.284.534 = fsub double %scalar.283.533, %scalar.276.526
  store double %scalar.284.534, ptr %value.534, align 8
  %scalar.285.535 = fsub double %scalar.282.532, %scalar.284.534
  store double %scalar.285.535, ptr %value.535, align 8
  %scalar.286.51 = fadd double %scalar.283.533, %scalar.285.535
  store double %scalar.286.51, ptr %out.25, align 8
  %load.287.536.0 = load double, ptr %arg.14, align 8
  %scalar.287.536 = fadd double %load.287.536.0, %scalar.283.533
  store double %scalar.287.536, ptr %value.536, align 8
  %scalar.288.537 = fsub double %scalar.287.536, %load.287.536.0
  store double %scalar.288.537, ptr %value.537, align 8
  %scalar.289.538 = fsub double %scalar.287.536, %scalar.288.537
  store double %scalar.289.538, ptr %value.538, align 8
  %scalar.290.539 = fsub double %load.287.536.0, %scalar.289.538
  store double %scalar.290.539, ptr %value.539, align 8
  %scalar.291.540 = fsub double %scalar.283.533, %scalar.288.537
  store double %scalar.291.540, ptr %value.540, align 8
  %scalar.292.541 = fadd double %scalar.290.539, %scalar.291.540
  store double %scalar.292.541, ptr %value.541, align 8
  %load.293.542.1 = load double, ptr %arg.41, align 8
  %scalar.293.542 = fadd double %scalar.292.541, %load.293.542.1
  store double %scalar.293.542, ptr %value.542, align 8
  %scalar.294.543 = fadd double %scalar.293.542, %scalar.285.535
  store double %scalar.294.543, ptr %value.543, align 8
  %scalar.295.544 = fadd double %scalar.287.536, %scalar.294.543
  store double %scalar.295.544, ptr %value.544, align 8
  %scalar.296.545 = fsub double %scalar.295.544, %scalar.287.536
  store double %scalar.296.545, ptr %value.545, align 8
  %scalar.297.546 = fsub double %scalar.294.543, %scalar.296.545
  store double %scalar.297.546, ptr %value.546, align 8
  %scalar.298.52 = fadd double %scalar.295.544, %scalar.297.546
  store double %scalar.298.52, ptr %out.26, align 8
  %scalar.299.547 = fmul double %load.0.274.1, %scalar.295.544
  store double %scalar.299.547, ptr %value.547, align 8
  %scalar.300.548 = fneg double %scalar.299.547
  store double %scalar.300.548, ptr %value.548, align 8
  %scalar.301.549 = call double @llvm.fma.f64(double %load.0.274.1, double %scalar.295.544, double %scalar.300.548)
  store double %scalar.301.549, ptr %value.549, align 8
  %scalar.302.550 = fmul double %load.0.274.1, %scalar.297.546
  store double %scalar.302.550, ptr %value.550, align 8
  %scalar.303.551 = fadd double %scalar.301.549, %scalar.302.550
  store double %scalar.303.551, ptr %value.551, align 8
  %scalar.304.552 = fmul double %load.3.277.1, %scalar.295.544
  store double %scalar.304.552, ptr %value.552, align 8
  %scalar.305.553 = fadd double %scalar.303.551, %scalar.304.552
  store double %scalar.305.553, ptr %value.553, align 8
  %scalar.306.554 = fadd double %scalar.299.547, %scalar.305.553
  store double %scalar.306.554, ptr %value.554, align 8
  %scalar.307.555 = fsub double %scalar.306.554, %scalar.299.547
  store double %scalar.307.555, ptr %value.555, align 8
  %scalar.308.556 = fsub double %scalar.305.553, %scalar.307.555
  store double %scalar.308.556, ptr %value.556, align 8
  %scalar.309.53 = fadd double %scalar.306.554, %scalar.308.556
  store double %scalar.309.53, ptr %out.27, align 8
  %load.310.557.0 = load double, ptr %arg.15, align 8
  %scalar.310.557 = fadd double %load.310.557.0, %scalar.306.554
  store double %scalar.310.557, ptr %value.557, align 8
  %scalar.311.558 = fsub double %scalar.310.557, %load.310.557.0
  store double %scalar.311.558, ptr %value.558, align 8
  %scalar.312.559 = fsub double %scalar.310.557, %scalar.311.558
  store double %scalar.312.559, ptr %value.559, align 8
  %scalar.313.560 = fsub double %load.310.557.0, %scalar.312.559
  store double %scalar.313.560, ptr %value.560, align 8
  %scalar.314.561 = fsub double %scalar.306.554, %scalar.311.558
  store double %scalar.314.561, ptr %value.561, align 8
  %scalar.315.562 = fadd double %scalar.313.560, %scalar.314.561
  store double %scalar.315.562, ptr %value.562, align 8
  %load.316.563.1 = load double, ptr %arg.42, align 8
  %scalar.316.563 = fadd double %scalar.315.562, %load.316.563.1
  store double %scalar.316.563, ptr %value.563, align 8
  %scalar.317.564 = fadd double %scalar.316.563, %scalar.308.556
  store double %scalar.317.564, ptr %value.564, align 8
  %scalar.318.565 = fadd double %scalar.310.557, %scalar.317.564
  store double %scalar.318.565, ptr %value.565, align 8
  %scalar.319.566 = fsub double %scalar.318.565, %scalar.310.557
  store double %scalar.319.566, ptr %value.566, align 8
  %scalar.320.567 = fsub double %scalar.317.564, %scalar.319.566
  store double %scalar.320.567, ptr %value.567, align 8
  %scalar.321.54 = fadd double %scalar.318.565, %scalar.320.567
  store double %scalar.321.54, ptr %out.28, align 8
  %scalar.322.568 = fmul double %load.0.274.1, %scalar.318.565
  store double %scalar.322.568, ptr %value.568, align 8
  %scalar.323.569 = fneg double %scalar.322.568
  store double %scalar.323.569, ptr %value.569, align 8
  %scalar.324.570 = call double @llvm.fma.f64(double %load.0.274.1, double %scalar.318.565, double %scalar.323.569)
  store double %scalar.324.570, ptr %value.570, align 8
  %scalar.325.571 = fmul double %load.0.274.1, %scalar.320.567
  store double %scalar.325.571, ptr %value.571, align 8
  %scalar.326.572 = fadd double %scalar.324.570, %scalar.325.571
  store double %scalar.326.572, ptr %value.572, align 8
  %scalar.327.573 = fmul double %load.3.277.1, %scalar.318.565
  store double %scalar.327.573, ptr %value.573, align 8
  %scalar.328.574 = fadd double %scalar.326.572, %scalar.327.573
  store double %scalar.328.574, ptr %value.574, align 8
  %scalar.329.575 = fadd double %scalar.322.568, %scalar.328.574
  store double %scalar.329.575, ptr %value.575, align 8
  %scalar.330.576 = fsub double %scalar.329.575, %scalar.322.568
  store double %scalar.330.576, ptr %value.576, align 8
  %scalar.331.577 = fsub double %scalar.328.574, %scalar.330.576
  store double %scalar.331.577, ptr %value.577, align 8
  %scalar.332.55 = fadd double %scalar.329.575, %scalar.331.577
  store double %scalar.332.55, ptr %out.29, align 8
  %load.333.578.0 = load double, ptr %arg.16, align 8
  %scalar.333.578 = fadd double %load.333.578.0, %scalar.329.575
  store double %scalar.333.578, ptr %value.578, align 8
  %scalar.334.579 = fsub double %scalar.333.578, %load.333.578.0
  store double %scalar.334.579, ptr %value.579, align 8
  %scalar.335.580 = fsub double %scalar.333.578, %scalar.334.579
  store double %scalar.335.580, ptr %value.580, align 8
  %scalar.336.581 = fsub double %load.333.578.0, %scalar.335.580
  store double %scalar.336.581, ptr %value.581, align 8
  %scalar.337.582 = fsub double %scalar.329.575, %scalar.334.579
  store double %scalar.337.582, ptr %value.582, align 8
  %scalar.338.583 = fadd double %scalar.336.581, %scalar.337.582
  store double %scalar.338.583, ptr %value.583, align 8
  %load.339.584.1 = load double, ptr %arg.43, align 8
  %scalar.339.584 = fadd double %scalar.338.583, %load.339.584.1
  store double %scalar.339.584, ptr %value.584, align 8
  %scalar.340.585 = fadd double %scalar.339.584, %scalar.331.577
  store double %scalar.340.585, ptr %value.585, align 8
  %scalar.341.586 = fadd double %scalar.333.578, %scalar.340.585
  store double %scalar.341.586, ptr %value.586, align 8
  %scalar.342.587 = fsub double %scalar.341.586, %scalar.333.578
  store double %scalar.342.587, ptr %value.587, align 8
  %scalar.343.588 = fsub double %scalar.340.585, %scalar.342.587
  store double %scalar.343.588, ptr %value.588, align 8
  %scalar.344.56 = fadd double %scalar.341.586, %scalar.343.588
  store double %scalar.344.56, ptr %out.30, align 8
  %scalar.345.589 = fmul double %load.0.274.1, %scalar.341.586
  store double %scalar.345.589, ptr %value.589, align 8
  %scalar.346.590 = fneg double %scalar.345.589
  store double %scalar.346.590, ptr %value.590, align 8
  %scalar.347.591 = call double @llvm.fma.f64(double %load.0.274.1, double %scalar.341.586, double %scalar.346.590)
  store double %scalar.347.591, ptr %value.591, align 8
  %scalar.348.592 = fmul double %load.0.274.1, %scalar.343.588
  store double %scalar.348.592, ptr %value.592, align 8
  %scalar.349.593 = fadd double %scalar.347.591, %scalar.348.592
  store double %scalar.349.593, ptr %value.593, align 8
  %scalar.350.594 = fmul double %load.3.277.1, %scalar.341.586
  store double %scalar.350.594, ptr %value.594, align 8
  %scalar.351.595 = fadd double %scalar.349.593, %scalar.350.594
  store double %scalar.351.595, ptr %value.595, align 8
  %scalar.352.596 = fadd double %scalar.345.589, %scalar.351.595
  store double %scalar.352.596, ptr %value.596, align 8
  %scalar.353.597 = fsub double %scalar.352.596, %scalar.345.589
  store double %scalar.353.597, ptr %value.597, align 8
  %scalar.354.598 = fsub double %scalar.351.595, %scalar.353.597
  store double %scalar.354.598, ptr %value.598, align 8
  %scalar.355.57 = fadd double %scalar.352.596, %scalar.354.598
  store double %scalar.355.57, ptr %out.31, align 8
  %load.356.599.0 = load double, ptr %arg.17, align 8
  %scalar.356.599 = fadd double %load.356.599.0, %scalar.352.596
  store double %scalar.356.599, ptr %value.599, align 8
  %scalar.357.600 = fsub double %scalar.356.599, %load.356.599.0
  store double %scalar.357.600, ptr %value.600, align 8
  %scalar.358.601 = fsub double %scalar.356.599, %scalar.357.600
  store double %scalar.358.601, ptr %value.601, align 8
  %scalar.359.602 = fsub double %load.356.599.0, %scalar.358.601
  store double %scalar.359.602, ptr %value.602, align 8
  %scalar.360.603 = fsub double %scalar.352.596, %scalar.357.600
  store double %scalar.360.603, ptr %value.603, align 8
  %scalar.361.604 = fadd double %scalar.359.602, %scalar.360.603
  store double %scalar.361.604, ptr %value.604, align 8
  %load.362.605.1 = load double, ptr %arg.44, align 8
  %scalar.362.605 = fadd double %scalar.361.604, %load.362.605.1
  store double %scalar.362.605, ptr %value.605, align 8
  %scalar.363.606 = fadd double %scalar.362.605, %scalar.354.598
  store double %scalar.363.606, ptr %value.606, align 8
  %scalar.364.607 = fadd double %scalar.356.599, %scalar.363.606
  store double %scalar.364.607, ptr %value.607, align 8
  %scalar.365.608 = fsub double %scalar.364.607, %scalar.356.599
  store double %scalar.365.608, ptr %value.608, align 8
  %scalar.366.609 = fsub double %scalar.363.606, %scalar.365.608
  store double %scalar.366.609, ptr %value.609, align 8
  %scalar.367.58 = fadd double %scalar.364.607, %scalar.366.609
  store double %scalar.367.58, ptr %out.32, align 8
  %scalar.368.610 = fmul double %load.0.274.1, %scalar.364.607
  store double %scalar.368.610, ptr %value.610, align 8
  %scalar.369.611 = fneg double %scalar.368.610
  store double %scalar.369.611, ptr %value.611, align 8
  %scalar.370.612 = call double @llvm.fma.f64(double %load.0.274.1, double %scalar.364.607, double %scalar.369.611)
  store double %scalar.370.612, ptr %value.612, align 8
  %scalar.371.613 = fmul double %load.0.274.1, %scalar.366.609
  store double %scalar.371.613, ptr %value.613, align 8
  %scalar.372.614 = fadd double %scalar.370.612, %scalar.371.613
  store double %scalar.372.614, ptr %value.614, align 8
  %scalar.373.615 = fmul double %load.3.277.1, %scalar.364.607
  store double %scalar.373.615, ptr %value.615, align 8
  %scalar.374.616 = fadd double %scalar.372.614, %scalar.373.615
  store double %scalar.374.616, ptr %value.616, align 8
  %scalar.375.617 = fadd double %scalar.368.610, %scalar.374.616
  store double %scalar.375.617, ptr %value.617, align 8
  %scalar.376.618 = fsub double %scalar.375.617, %scalar.368.610
  store double %scalar.376.618, ptr %value.618, align 8
  %scalar.377.619 = fsub double %scalar.374.616, %scalar.376.618
  store double %scalar.377.619, ptr %value.619, align 8
  %scalar.378.59 = fadd double %scalar.375.617, %scalar.377.619
  store double %scalar.378.59, ptr %out.33, align 8
  %load.379.620.0 = load double, ptr %arg.18, align 8
  %scalar.379.620 = fadd double %load.379.620.0, %scalar.375.617
  store double %scalar.379.620, ptr %value.620, align 8
  %scalar.380.621 = fsub double %scalar.379.620, %load.379.620.0
  store double %scalar.380.621, ptr %value.621, align 8
  %scalar.381.622 = fsub double %scalar.379.620, %scalar.380.621
  store double %scalar.381.622, ptr %value.622, align 8
  %scalar.382.623 = fsub double %load.379.620.0, %scalar.381.622
  store double %scalar.382.623, ptr %value.623, align 8
  %scalar.383.624 = fsub double %scalar.375.617, %scalar.380.621
  store double %scalar.383.624, ptr %value.624, align 8
  %scalar.384.625 = fadd double %scalar.382.623, %scalar.383.624
  store double %scalar.384.625, ptr %value.625, align 8
  %load.385.626.1 = load double, ptr %arg.45, align 8
  %scalar.385.626 = fadd double %scalar.384.625, %load.385.626.1
  store double %scalar.385.626, ptr %value.626, align 8
  %scalar.386.627 = fadd double %scalar.385.626, %scalar.377.619
  store double %scalar.386.627, ptr %value.627, align 8
  %scalar.387.628 = fadd double %scalar.379.620, %scalar.386.627
  store double %scalar.387.628, ptr %value.628, align 8
  %scalar.388.629 = fsub double %scalar.387.628, %scalar.379.620
  store double %scalar.388.629, ptr %value.629, align 8
  %scalar.389.630 = fsub double %scalar.386.627, %scalar.388.629
  store double %scalar.389.630, ptr %value.630, align 8
  %scalar.390.60 = fadd double %scalar.387.628, %scalar.389.630
  store double %scalar.390.60, ptr %out.34, align 8
  %scalar.391.631 = fmul double %load.0.274.1, %scalar.387.628
  store double %scalar.391.631, ptr %value.631, align 8
  %scalar.392.632 = fneg double %scalar.391.631
  store double %scalar.392.632, ptr %value.632, align 8
  %scalar.393.633 = call double @llvm.fma.f64(double %load.0.274.1, double %scalar.387.628, double %scalar.392.632)
  store double %scalar.393.633, ptr %value.633, align 8
  %scalar.394.634 = fmul double %load.0.274.1, %scalar.389.630
  store double %scalar.394.634, ptr %value.634, align 8
  %scalar.395.635 = fadd double %scalar.393.633, %scalar.394.634
  store double %scalar.395.635, ptr %value.635, align 8
  %scalar.396.636 = fmul double %load.3.277.1, %scalar.387.628
  store double %scalar.396.636, ptr %value.636, align 8
  %scalar.397.637 = fadd double %scalar.395.635, %scalar.396.636
  store double %scalar.397.637, ptr %value.637, align 8
  %scalar.398.638 = fadd double %scalar.391.631, %scalar.397.637
  store double %scalar.398.638, ptr %value.638, align 8
  %scalar.399.639 = fsub double %scalar.398.638, %scalar.391.631
  store double %scalar.399.639, ptr %value.639, align 8
  %scalar.400.640 = fsub double %scalar.397.637, %scalar.399.639
  store double %scalar.400.640, ptr %value.640, align 8
  %scalar.401.61 = fadd double %scalar.398.638, %scalar.400.640
  store double %scalar.401.61, ptr %out.35, align 8
  %load.402.641.0 = load double, ptr %arg.19, align 8
  %scalar.402.641 = fadd double %load.402.641.0, %scalar.398.638
  store double %scalar.402.641, ptr %value.641, align 8
  %scalar.403.642 = fsub double %scalar.402.641, %load.402.641.0
  store double %scalar.403.642, ptr %value.642, align 8
  %scalar.404.643 = fsub double %scalar.402.641, %scalar.403.642
  store double %scalar.404.643, ptr %value.643, align 8
  %scalar.405.644 = fsub double %load.402.641.0, %scalar.404.643
  store double %scalar.405.644, ptr %value.644, align 8
  %scalar.406.645 = fsub double %scalar.398.638, %scalar.403.642
  store double %scalar.406.645, ptr %value.645, align 8
  %scalar.407.646 = fadd double %scalar.405.644, %scalar.406.645
  store double %scalar.407.646, ptr %value.646, align 8
  %load.408.647.1 = load double, ptr %arg.46, align 8
  %scalar.408.647 = fadd double %scalar.407.646, %load.408.647.1
  store double %scalar.408.647, ptr %value.647, align 8
  %scalar.409.648 = fadd double %scalar.408.647, %scalar.400.640
  store double %scalar.409.648, ptr %value.648, align 8
  %scalar.410.649 = fadd double %scalar.402.641, %scalar.409.648
  store double %scalar.410.649, ptr %value.649, align 8
  %scalar.411.650 = fsub double %scalar.410.649, %scalar.402.641
  store double %scalar.411.650, ptr %value.650, align 8
  %scalar.412.651 = fsub double %scalar.409.648, %scalar.411.650
  store double %scalar.412.651, ptr %value.651, align 8
  %scalar.413.62 = fadd double %scalar.410.649, %scalar.412.651
  store double %scalar.413.62, ptr %out.36, align 8
  %scalar.414.652 = fmul double %load.0.274.1, %scalar.410.649
  store double %scalar.414.652, ptr %value.652, align 8
  %scalar.415.653 = fneg double %scalar.414.652
  store double %scalar.415.653, ptr %value.653, align 8
  %scalar.416.654 = call double @llvm.fma.f64(double %load.0.274.1, double %scalar.410.649, double %scalar.415.653)
  store double %scalar.416.654, ptr %value.654, align 8
  %scalar.417.655 = fmul double %load.0.274.1, %scalar.412.651
  store double %scalar.417.655, ptr %value.655, align 8
  %scalar.418.656 = fadd double %scalar.416.654, %scalar.417.655
  store double %scalar.418.656, ptr %value.656, align 8
  %scalar.419.657 = fmul double %load.3.277.1, %scalar.410.649
  store double %scalar.419.657, ptr %value.657, align 8
  %scalar.420.658 = fadd double %scalar.418.656, %scalar.419.657
  store double %scalar.420.658, ptr %value.658, align 8
  %scalar.421.659 = fadd double %scalar.414.652, %scalar.420.658
  store double %scalar.421.659, ptr %value.659, align 8
  %scalar.422.660 = fsub double %scalar.421.659, %scalar.414.652
  store double %scalar.422.660, ptr %value.660, align 8
  %scalar.423.661 = fsub double %scalar.420.658, %scalar.422.660
  store double %scalar.423.661, ptr %value.661, align 8
  %scalar.424.63 = fadd double %scalar.421.659, %scalar.423.661
  store double %scalar.424.63, ptr %out.37, align 8
  %load.425.662.0 = load double, ptr %arg.20, align 8
  %scalar.425.662 = fadd double %load.425.662.0, %scalar.421.659
  store double %scalar.425.662, ptr %value.662, align 8
  %scalar.426.663 = fsub double %scalar.425.662, %load.425.662.0
  store double %scalar.426.663, ptr %value.663, align 8
  %scalar.427.664 = fsub double %scalar.425.662, %scalar.426.663
  store double %scalar.427.664, ptr %value.664, align 8
  %scalar.428.665 = fsub double %load.425.662.0, %scalar.427.664
  store double %scalar.428.665, ptr %value.665, align 8
  %scalar.429.666 = fsub double %scalar.421.659, %scalar.426.663
  store double %scalar.429.666, ptr %value.666, align 8
  %scalar.430.667 = fadd double %scalar.428.665, %scalar.429.666
  store double %scalar.430.667, ptr %value.667, align 8
  %load.431.668.1 = load double, ptr %arg.47, align 8
  %scalar.431.668 = fadd double %scalar.430.667, %load.431.668.1
  store double %scalar.431.668, ptr %value.668, align 8
  %scalar.432.669 = fadd double %scalar.431.668, %scalar.423.661
  store double %scalar.432.669, ptr %value.669, align 8
  %scalar.433.670 = fadd double %scalar.425.662, %scalar.432.669
  store double %scalar.433.670, ptr %value.670, align 8
  %scalar.434.671 = fsub double %scalar.433.670, %scalar.425.662
  store double %scalar.434.671, ptr %value.671, align 8
  %scalar.435.672 = fsub double %scalar.432.669, %scalar.434.671
  store double %scalar.435.672, ptr %value.672, align 8
  %scalar.436.64 = fadd double %scalar.433.670, %scalar.435.672
  store double %scalar.436.64, ptr %out.38, align 8
  %scalar.437.673 = fmul double %load.0.274.1, %scalar.433.670
  store double %scalar.437.673, ptr %value.673, align 8
  %scalar.438.674 = fneg double %scalar.437.673
  store double %scalar.438.674, ptr %value.674, align 8
  %scalar.439.675 = call double @llvm.fma.f64(double %load.0.274.1, double %scalar.433.670, double %scalar.438.674)
  store double %scalar.439.675, ptr %value.675, align 8
  %scalar.440.676 = fmul double %load.0.274.1, %scalar.435.672
  store double %scalar.440.676, ptr %value.676, align 8
  %scalar.441.677 = fadd double %scalar.439.675, %scalar.440.676
  store double %scalar.441.677, ptr %value.677, align 8
  %scalar.442.678 = fmul double %load.3.277.1, %scalar.433.670
  store double %scalar.442.678, ptr %value.678, align 8
  %scalar.443.679 = fadd double %scalar.441.677, %scalar.442.678
  store double %scalar.443.679, ptr %value.679, align 8
  %scalar.444.680 = fadd double %scalar.437.673, %scalar.443.679
  store double %scalar.444.680, ptr %value.680, align 8
  %scalar.445.681 = fsub double %scalar.444.680, %scalar.437.673
  store double %scalar.445.681, ptr %value.681, align 8
  %scalar.446.682 = fsub double %scalar.443.679, %scalar.445.681
  store double %scalar.446.682, ptr %value.682, align 8
  %scalar.447.65 = fadd double %scalar.444.680, %scalar.446.682
  store double %scalar.447.65, ptr %out.39, align 8
  %load.448.683.0 = load double, ptr %arg.21, align 8
  %scalar.448.683 = fadd double %load.448.683.0, %scalar.444.680
  store double %scalar.448.683, ptr %value.683, align 8
  %scalar.449.684 = fsub double %scalar.448.683, %load.448.683.0
  store double %scalar.449.684, ptr %value.684, align 8
  %scalar.450.685 = fsub double %scalar.448.683, %scalar.449.684
  store double %scalar.450.685, ptr %value.685, align 8
  %scalar.451.686 = fsub double %load.448.683.0, %scalar.450.685
  store double %scalar.451.686, ptr %value.686, align 8
  %scalar.452.687 = fsub double %scalar.444.680, %scalar.449.684
  store double %scalar.452.687, ptr %value.687, align 8
  %scalar.453.688 = fadd double %scalar.451.686, %scalar.452.687
  store double %scalar.453.688, ptr %value.688, align 8
  %load.454.689.1 = load double, ptr %arg.48, align 8
  %scalar.454.689 = fadd double %scalar.453.688, %load.454.689.1
  store double %scalar.454.689, ptr %value.689, align 8
  %scalar.455.690 = fadd double %scalar.454.689, %scalar.446.682
  store double %scalar.455.690, ptr %value.690, align 8
  %scalar.456.691 = fadd double %scalar.448.683, %scalar.455.690
  store double %scalar.456.691, ptr %value.691, align 8
  %scalar.457.692 = fsub double %scalar.456.691, %scalar.448.683
  store double %scalar.457.692, ptr %value.692, align 8
  %scalar.458.693 = fsub double %scalar.455.690, %scalar.457.692
  store double %scalar.458.693, ptr %value.693, align 8
  %scalar.459.66 = fadd double %scalar.456.691, %scalar.458.693
  store double %scalar.459.66, ptr %out.40, align 8
  %scalar.460.694 = fmul double %load.0.274.1, %scalar.456.691
  store double %scalar.460.694, ptr %value.694, align 8
  %scalar.461.695 = fneg double %scalar.460.694
  store double %scalar.461.695, ptr %value.695, align 8
  %scalar.462.696 = call double @llvm.fma.f64(double %load.0.274.1, double %scalar.456.691, double %scalar.461.695)
  store double %scalar.462.696, ptr %value.696, align 8
  %scalar.463.697 = fmul double %load.0.274.1, %scalar.458.693
  store double %scalar.463.697, ptr %value.697, align 8
  %scalar.464.698 = fadd double %scalar.462.696, %scalar.463.697
  store double %scalar.464.698, ptr %value.698, align 8
  %scalar.465.699 = fmul double %load.3.277.1, %scalar.456.691
  store double %scalar.465.699, ptr %value.699, align 8
  %scalar.466.700 = fadd double %scalar.464.698, %scalar.465.699
  store double %scalar.466.700, ptr %value.700, align 8
  %scalar.467.701 = fadd double %scalar.460.694, %scalar.466.700
  store double %scalar.467.701, ptr %value.701, align 8
  %scalar.468.702 = fsub double %scalar.467.701, %scalar.460.694
  store double %scalar.468.702, ptr %value.702, align 8
  %scalar.469.703 = fsub double %scalar.466.700, %scalar.468.702
  store double %scalar.469.703, ptr %value.703, align 8
  %scalar.470.67 = fadd double %scalar.467.701, %scalar.469.703
  store double %scalar.470.67, ptr %out.41, align 8
  %load.471.704.0 = load double, ptr %arg.22, align 8
  %scalar.471.704 = fadd double %load.471.704.0, %scalar.467.701
  store double %scalar.471.704, ptr %value.704, align 8
  %scalar.472.705 = fsub double %scalar.471.704, %load.471.704.0
  store double %scalar.472.705, ptr %value.705, align 8
  %scalar.473.706 = fsub double %scalar.471.704, %scalar.472.705
  store double %scalar.473.706, ptr %value.706, align 8
  %scalar.474.707 = fsub double %load.471.704.0, %scalar.473.706
  store double %scalar.474.707, ptr %value.707, align 8
  %scalar.475.708 = fsub double %scalar.467.701, %scalar.472.705
  store double %scalar.475.708, ptr %value.708, align 8
  %scalar.476.709 = fadd double %scalar.474.707, %scalar.475.708
  store double %scalar.476.709, ptr %value.709, align 8
  %load.477.710.1 = load double, ptr %arg.49, align 8
  %scalar.477.710 = fadd double %scalar.476.709, %load.477.710.1
  store double %scalar.477.710, ptr %value.710, align 8
  %scalar.478.711 = fadd double %scalar.477.710, %scalar.469.703
  store double %scalar.478.711, ptr %value.711, align 8
  %scalar.479.712 = fadd double %scalar.471.704, %scalar.478.711
  store double %scalar.479.712, ptr %value.712, align 8
  %scalar.480.713 = fsub double %scalar.479.712, %scalar.471.704
  store double %scalar.480.713, ptr %value.713, align 8
  %scalar.481.714 = fsub double %scalar.478.711, %scalar.480.713
  store double %scalar.481.714, ptr %value.714, align 8
  %scalar.482.68 = fadd double %scalar.479.712, %scalar.481.714
  store double %scalar.482.68, ptr %out.42, align 8
  %scalar.483.715 = fmul double %load.0.274.1, %scalar.479.712
  store double %scalar.483.715, ptr %value.715, align 8
  %scalar.484.716 = fneg double %scalar.483.715
  store double %scalar.484.716, ptr %value.716, align 8
  %scalar.485.717 = call double @llvm.fma.f64(double %load.0.274.1, double %scalar.479.712, double %scalar.484.716)
  store double %scalar.485.717, ptr %value.717, align 8
  %scalar.486.718 = fmul double %load.0.274.1, %scalar.481.714
  store double %scalar.486.718, ptr %value.718, align 8
  %scalar.487.719 = fadd double %scalar.485.717, %scalar.486.718
  store double %scalar.487.719, ptr %value.719, align 8
  %scalar.488.720 = fmul double %load.3.277.1, %scalar.479.712
  store double %scalar.488.720, ptr %value.720, align 8
  %scalar.489.721 = fadd double %scalar.487.719, %scalar.488.720
  store double %scalar.489.721, ptr %value.721, align 8
  %scalar.490.722 = fadd double %scalar.483.715, %scalar.489.721
  store double %scalar.490.722, ptr %value.722, align 8
  %scalar.491.723 = fsub double %scalar.490.722, %scalar.483.715
  store double %scalar.491.723, ptr %value.723, align 8
  %scalar.492.724 = fsub double %scalar.489.721, %scalar.491.723
  store double %scalar.492.724, ptr %value.724, align 8
  %scalar.493.69 = fadd double %scalar.490.722, %scalar.492.724
  store double %scalar.493.69, ptr %out.43, align 8
  %load.494.725.0 = load double, ptr %arg.23, align 8
  %scalar.494.725 = fadd double %load.494.725.0, %scalar.490.722
  store double %scalar.494.725, ptr %value.725, align 8
  %scalar.495.726 = fsub double %scalar.494.725, %load.494.725.0
  store double %scalar.495.726, ptr %value.726, align 8
  %scalar.496.727 = fsub double %scalar.494.725, %scalar.495.726
  store double %scalar.496.727, ptr %value.727, align 8
  %scalar.497.728 = fsub double %load.494.725.0, %scalar.496.727
  store double %scalar.497.728, ptr %value.728, align 8
  %scalar.498.729 = fsub double %scalar.490.722, %scalar.495.726
  store double %scalar.498.729, ptr %value.729, align 8
  %scalar.499.730 = fadd double %scalar.497.728, %scalar.498.729
  store double %scalar.499.730, ptr %value.730, align 8
  %load.500.731.1 = load double, ptr %arg.50, align 8
  %scalar.500.731 = fadd double %scalar.499.730, %load.500.731.1
  store double %scalar.500.731, ptr %value.731, align 8
  %scalar.501.732 = fadd double %scalar.500.731, %scalar.492.724
  store double %scalar.501.732, ptr %value.732, align 8
  %scalar.502.733 = fadd double %scalar.494.725, %scalar.501.732
  store double %scalar.502.733, ptr %value.733, align 8
  %scalar.503.734 = fsub double %scalar.502.733, %scalar.494.725
  store double %scalar.503.734, ptr %value.734, align 8
  %scalar.504.735 = fsub double %scalar.501.732, %scalar.503.734
  store double %scalar.504.735, ptr %value.735, align 8
  %scalar.505.70 = fadd double %scalar.502.733, %scalar.504.735
  store double %scalar.505.70, ptr %out.44, align 8
  %scalar.506.736 = fmul double %load.0.274.1, %scalar.502.733
  store double %scalar.506.736, ptr %value.736, align 8
  %scalar.507.737 = fneg double %scalar.506.736
  store double %scalar.507.737, ptr %value.737, align 8
  %scalar.508.738 = call double @llvm.fma.f64(double %load.0.274.1, double %scalar.502.733, double %scalar.507.737)
  store double %scalar.508.738, ptr %value.738, align 8
  %scalar.509.739 = fmul double %load.0.274.1, %scalar.504.735
  store double %scalar.509.739, ptr %value.739, align 8
  %scalar.510.740 = fadd double %scalar.508.738, %scalar.509.739
  store double %scalar.510.740, ptr %value.740, align 8
  %scalar.511.741 = fmul double %load.3.277.1, %scalar.502.733
  store double %scalar.511.741, ptr %value.741, align 8
  %scalar.512.742 = fadd double %scalar.510.740, %scalar.511.741
  store double %scalar.512.742, ptr %value.742, align 8
  %scalar.513.743 = fadd double %scalar.506.736, %scalar.512.742
  store double %scalar.513.743, ptr %value.743, align 8
  %scalar.514.744 = fsub double %scalar.513.743, %scalar.506.736
  store double %scalar.514.744, ptr %value.744, align 8
  %scalar.515.745 = fsub double %scalar.512.742, %scalar.514.744
  store double %scalar.515.745, ptr %value.745, align 8
  %scalar.516.71 = fadd double %scalar.513.743, %scalar.515.745
  store double %scalar.516.71, ptr %out.45, align 8
  %load.517.746.0 = load double, ptr %arg.24, align 8
  %scalar.517.746 = fadd double %load.517.746.0, %scalar.513.743
  store double %scalar.517.746, ptr %value.746, align 8
  %scalar.518.747 = fsub double %scalar.517.746, %load.517.746.0
  store double %scalar.518.747, ptr %value.747, align 8
  %scalar.519.748 = fsub double %scalar.517.746, %scalar.518.747
  store double %scalar.519.748, ptr %value.748, align 8
  %scalar.520.749 = fsub double %load.517.746.0, %scalar.519.748
  store double %scalar.520.749, ptr %value.749, align 8
  %scalar.521.750 = fsub double %scalar.513.743, %scalar.518.747
  store double %scalar.521.750, ptr %value.750, align 8
  %scalar.522.751 = fadd double %scalar.520.749, %scalar.521.750
  store double %scalar.522.751, ptr %value.751, align 8
  %load.523.752.1 = load double, ptr %arg.51, align 8
  %scalar.523.752 = fadd double %scalar.522.751, %load.523.752.1
  store double %scalar.523.752, ptr %value.752, align 8
  %scalar.524.753 = fadd double %scalar.523.752, %scalar.515.745
  store double %scalar.524.753, ptr %value.753, align 8
  %scalar.525.754 = fadd double %scalar.517.746, %scalar.524.753
  store double %scalar.525.754, ptr %value.754, align 8
  %scalar.526.755 = fsub double %scalar.525.754, %scalar.517.746
  store double %scalar.526.755, ptr %value.755, align 8
  %scalar.527.756 = fsub double %scalar.524.753, %scalar.526.755
  store double %scalar.527.756, ptr %value.756, align 8
  %scalar.528.72 = fadd double %scalar.525.754, %scalar.527.756
  store double %scalar.528.72, ptr %out.46, align 8
  %scalar.529.757 = fmul double %load.0.274.1, %scalar.525.754
  store double %scalar.529.757, ptr %value.757, align 8
  %scalar.530.758 = fneg double %scalar.529.757
  store double %scalar.530.758, ptr %value.758, align 8
  %scalar.531.759 = call double @llvm.fma.f64(double %load.0.274.1, double %scalar.525.754, double %scalar.530.758)
  store double %scalar.531.759, ptr %value.759, align 8
  %scalar.532.760 = fmul double %load.0.274.1, %scalar.527.756
  store double %scalar.532.760, ptr %value.760, align 8
  %scalar.533.761 = fadd double %scalar.531.759, %scalar.532.760
  store double %scalar.533.761, ptr %value.761, align 8
  %scalar.534.762 = fmul double %load.3.277.1, %scalar.525.754
  store double %scalar.534.762, ptr %value.762, align 8
  %scalar.535.763 = fadd double %scalar.533.761, %scalar.534.762
  store double %scalar.535.763, ptr %value.763, align 8
  %scalar.536.764 = fadd double %scalar.529.757, %scalar.535.763
  store double %scalar.536.764, ptr %value.764, align 8
  %scalar.537.765 = fsub double %scalar.536.764, %scalar.529.757
  store double %scalar.537.765, ptr %value.765, align 8
  %scalar.538.766 = fsub double %scalar.535.763, %scalar.537.765
  store double %scalar.538.766, ptr %value.766, align 8
  %scalar.539.73 = fadd double %scalar.536.764, %scalar.538.766
  store double %scalar.539.73, ptr %out.47, align 8
  %load.540.767.0 = load double, ptr %arg.25, align 8
  %scalar.540.767 = fadd double %load.540.767.0, %scalar.536.764
  store double %scalar.540.767, ptr %value.767, align 8
  %scalar.541.768 = fsub double %scalar.540.767, %load.540.767.0
  store double %scalar.541.768, ptr %value.768, align 8
  %scalar.542.769 = fsub double %scalar.540.767, %scalar.541.768
  store double %scalar.542.769, ptr %value.769, align 8
  %scalar.543.770 = fsub double %load.540.767.0, %scalar.542.769
  store double %scalar.543.770, ptr %value.770, align 8
  %scalar.544.771 = fsub double %scalar.536.764, %scalar.541.768
  store double %scalar.544.771, ptr %value.771, align 8
  %scalar.545.772 = fadd double %scalar.543.770, %scalar.544.771
  store double %scalar.545.772, ptr %value.772, align 8
  %load.546.773.1 = load double, ptr %arg.52, align 8
  %scalar.546.773 = fadd double %scalar.545.772, %load.546.773.1
  store double %scalar.546.773, ptr %value.773, align 8
  %scalar.547.774 = fadd double %scalar.546.773, %scalar.538.766
  store double %scalar.547.774, ptr %value.774, align 8
  %scalar.548.775 = fadd double %scalar.540.767, %scalar.547.774
  store double %scalar.548.775, ptr %value.775, align 8
  %scalar.549.776 = fsub double %scalar.548.775, %scalar.540.767
  store double %scalar.549.776, ptr %value.776, align 8
  %scalar.550.777 = fsub double %scalar.547.774, %scalar.549.776
  store double %scalar.550.777, ptr %value.777, align 8
  %scalar.551.74 = fadd double %scalar.548.775, %scalar.550.777
  store double %scalar.551.74, ptr %out.48, align 8
  %load.552.778.0 = load double, ptr %arg.26, align 8
  %scalar.552.778 = fmul double %load.552.778.0, %scalar.548.775
  store double %scalar.552.778, ptr %value.778, align 8
  %scalar.553.779 = fneg double %scalar.552.778
  store double %scalar.553.779, ptr %value.779, align 8
  %scalar.554.780 = call double @llvm.fma.f64(double %load.552.778.0, double %scalar.548.775, double %scalar.553.779)
  store double %scalar.554.780, ptr %value.780, align 8
  %scalar.555.781 = fmul double %load.552.778.0, %scalar.550.777
  store double %scalar.555.781, ptr %value.781, align 8
  %scalar.556.782 = fadd double %scalar.554.780, %scalar.555.781
  store double %scalar.556.782, ptr %value.782, align 8
  %load.557.783.0 = load double, ptr %arg.53, align 8
  %scalar.557.783 = fmul double %load.557.783.0, %scalar.548.775
  store double %scalar.557.783, ptr %value.783, align 8
  %scalar.558.784 = fadd double %scalar.556.782, %scalar.557.783
  store double %scalar.558.784, ptr %value.784, align 8
  %scalar.559.785 = fadd double %scalar.552.778, %scalar.558.784
  store double %scalar.559.785, ptr %value.785, align 8
  %scalar.560.786 = fsub double %scalar.559.785, %scalar.552.778
  store double %scalar.560.786, ptr %value.786, align 8
  %scalar.561.787 = fsub double %scalar.558.784, %scalar.560.786
  store double %scalar.561.787, ptr %value.787, align 8
  %scalar.562.75 = fadd double %scalar.559.785, %scalar.561.787
  store double %scalar.562.75, ptr %out.0, align 8
  ret void
}

define void @__ssa_asin_core_pack__asin_core(ptr noalias %arg.0, ptr noalias %arg.1, ptr noalias %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr noalias %arg.11, ptr noalias %arg.12, ptr noalias %arg.13, ptr noalias %arg.14, ptr noalias %arg.15, ptr noalias %arg.16, ptr noalias %arg.17, ptr noalias %arg.18, ptr noalias %arg.19, ptr noalias %arg.20, ptr noalias %arg.21, ptr noalias %arg.22, ptr noalias %arg.23, ptr noalias %arg.24, ptr %arg.25, ptr %arg.26, ptr noalias %arg.27, ptr noalias %arg.28, ptr noalias %arg.29, ptr noalias %arg.30, ptr noalias %arg.31, ptr noalias %arg.32, ptr noalias %arg.33, ptr noalias %arg.34, ptr noalias %arg.35, ptr noalias %arg.36, ptr noalias %arg.37, ptr noalias %arg.38, ptr noalias %arg.39, ptr noalias %arg.40, ptr noalias %arg.41, ptr noalias %arg.42, ptr noalias %arg.43, ptr noalias %arg.44, ptr noalias %arg.45, ptr noalias %arg.46, ptr noalias %arg.47, ptr noalias %arg.48, ptr noalias %arg.49, ptr noalias %arg.50, ptr noalias %arg.51, ptr %arg.52, ptr %arg.53, ptr %out.0) {
entry:
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
  %value.77 = alloca i64, i64 1, align 8
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
  store i32 48, ptr %value.173, align 4
  store i32 47, ptr %value.171, align 4
  store i32 46, ptr %value.169, align 4
  store i32 45, ptr %value.167, align 4
  store i32 44, ptr %value.165, align 4
  store i32 43, ptr %value.163, align 4
  store i32 42, ptr %value.161, align 4
  store i32 41, ptr %value.159, align 4
  store i32 40, ptr %value.157, align 4
  store i32 39, ptr %value.155, align 4
  store i32 38, ptr %value.153, align 4
  store i32 37, ptr %value.151, align 4
  store i32 36, ptr %value.149, align 4
  store i32 35, ptr %value.147, align 4
  store i32 34, ptr %value.145, align 4
  store i32 33, ptr %value.143, align 4
  store i32 32, ptr %value.141, align 4
  store i32 31, ptr %value.139, align 4
  store i32 30, ptr %value.137, align 4
  store i32 29, ptr %value.135, align 4
  store i32 28, ptr %value.133, align 4
  store i32 27, ptr %value.131, align 4
  store i32 26, ptr %value.129, align 4
  store i32 25, ptr %value.127, align 4
  store i32 24, ptr %value.125, align 4
  store i32 23, ptr %value.123, align 4
  store i32 22, ptr %value.121, align 4
  store i32 21, ptr %value.119, align 4
  store i32 20, ptr %value.117, align 4
  store i32 19, ptr %value.115, align 4
  store i32 18, ptr %value.113, align 4
  store i32 17, ptr %value.111, align 4
  store i32 16, ptr %value.109, align 4
  store i32 15, ptr %value.107, align 4
  store i32 14, ptr %value.105, align 4
  store i32 13, ptr %value.103, align 4
  store i32 12, ptr %value.101, align 4
  store i32 11, ptr %value.99, align 4
  store i32 10, ptr %value.97, align 4
  store i32 9, ptr %value.95, align 4
  store i32 8, ptr %value.93, align 4
  store i32 7, ptr %value.91, align 4
  store i32 6, ptr %value.89, align 4
  store i32 5, ptr %value.87, align 4
  store i32 4, ptr %value.85, align 4
  store i32 3, ptr %value.83, align 4
  store i32 2, ptr %value.81, align 4
  store i32 1, ptr %value.79, align 4
  store i64 0, ptr %value.77, align 8
  call void @__ssa_asin_core_pack__asin_core__planned_region_0(ptr %arg.17, ptr %arg.25, ptr %arg.16, ptr %arg.15, ptr %arg.14, ptr %arg.13, ptr %arg.11, ptr %arg.10, ptr %arg.9, ptr %arg.8, ptr %arg.7, ptr %arg.6, ptr %arg.5, ptr %arg.4, ptr %arg.3, ptr %arg.2, ptr %arg.24, ptr %arg.23, ptr %arg.22, ptr %arg.21, ptr %arg.20, ptr %arg.19, ptr %arg.18, ptr %arg.12, ptr %arg.1, ptr %arg.0, ptr %arg.26, ptr %arg.44, ptr %arg.52, ptr %arg.43, ptr %arg.42, ptr %arg.41, ptr %arg.40, ptr %arg.38, ptr %arg.37, ptr %arg.36, ptr %arg.35, ptr %arg.34, ptr %arg.33, ptr %arg.32, ptr %arg.31, ptr %arg.30, ptr %arg.29, ptr %arg.51, ptr %arg.50, ptr %arg.49, ptr %arg.48, ptr %arg.47, ptr %arg.46, ptr %arg.45, ptr %arg.39, ptr %arg.28, ptr %arg.27, ptr %arg.53, ptr %out.0, ptr %value.27, ptr %value.28, ptr %value.29, ptr %value.30, ptr %value.31, ptr %value.32, ptr %value.33, ptr %value.34, ptr %value.35, ptr %value.36, ptr %value.37, ptr %value.38, ptr %value.39, ptr %value.40, ptr %value.41, ptr %value.42, ptr %value.43, ptr %value.44, ptr %value.45, ptr %value.46, ptr %value.47, ptr %value.48, ptr %value.49, ptr %value.50, ptr %value.51, ptr %value.52, ptr %value.53, ptr %value.54, ptr %value.55, ptr %value.56, ptr %value.57, ptr %value.58, ptr %value.59, ptr %value.60, ptr %value.61, ptr %value.62, ptr %value.63, ptr %value.64, ptr %value.65, ptr %value.66, ptr %value.67, ptr %value.68, ptr %value.69, ptr %value.70, ptr %value.71, ptr %value.72, ptr %value.73, ptr %value.74)
  ret void
}

define void @asin_core_pack__asin_core_pack(ptr %buffers, ptr %extents) {
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
  call void @__ssa_asin_core_pack__asin_core_pack(ptr %public.0, ptr %public.1, ptr %public.2, ptr %public.3, ptr %public.4, ptr %public.5, ptr %public.6, ptr %public.7, ptr %public.8, ptr %public.9, ptr %public.10, ptr %public.11, ptr %public.12, ptr %public.13, ptr %public.14, ptr %public.15, ptr %public.16, ptr %public.17, ptr %public.18, ptr %public.19, ptr %public.20, ptr %public.21, ptr %public.22, ptr %public.23, ptr %public.24, ptr %public.25, ptr %public.26, ptr %public.27, ptr %public.28, ptr %public.29, ptr %public.30, ptr %public.31, ptr %public.32, ptr %public.33, ptr %public.34, ptr %public.35, ptr %public.36, ptr %public.37, ptr %public.38, ptr %public.39, ptr %public.40, ptr %public.41, ptr %public.42, ptr %public.43, ptr %public.44, ptr %public.45, ptr %public.46, ptr %public.47, ptr %public.48, ptr %public.49, ptr %public.50, ptr %public.51, ptr %public.52, ptr %public.2)
  ret void
}
