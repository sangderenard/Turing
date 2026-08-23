source_filename = "turing.ssa-llvm.sec_core_pack__sec_core_pack"

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-windows-gnu"

declare double @llvm.fma.f64(double, double, double)

define void @__ssa_sec_core_pack__sec_core_pack__planned_region_0(ptr noalias %arg.0, ptr %arg.1, ptr %out.0, ptr %out.1) {
entry:
  %load.0.36.0 = load i32, ptr %arg.1, align 4
  %address.0.36 = getelementptr double, ptr %arg.0, i32 %load.0.36.0
  %pinned.load.1.23 = load double, ptr %address.0.36, align 8
  store double %pinned.load.1.23, ptr %out.1, align 8
  %load.2.24.0 = load double, ptr %out.1, align 8
  %scalar.2.24 = fmul double %load.2.24.0, %load.2.24.0
  store double %scalar.2.24, ptr %out.0, align 8
  ret void
}

define void @__ssa_sec_core_pack__sec_core_pack__planned_region_1(ptr %arg.0, ptr %arg.1, ptr %arg.2) {
entry:
  %load.0.37.0 = load i32, ptr %arg.1, align 4
  %address.0.37 = getelementptr double, ptr %arg.0, i32 %load.0.37.0
  %load.store.1.v = load double, ptr %arg.2, align 8
  store double %load.store.1.v, ptr %address.0.37, align 8
  ret void
}

define void @__ssa_sec_core_pack__sec_core_pack(ptr noalias %arg.0, ptr noalias %arg.1, ptr %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr noalias %arg.11, ptr noalias %arg.12, ptr noalias %arg.13, ptr noalias %arg.14, ptr noalias %arg.15, ptr noalias %arg.16, ptr noalias %arg.17, ptr noalias %arg.18, ptr noalias %arg.19, ptr noalias %arg.20, ptr noalias %arg.21, ptr noalias %arg.22, ptr noalias %arg.23, ptr noalias %arg.24, ptr noalias %arg.25, ptr noalias %arg.26, ptr noalias %arg.27, ptr noalias %arg.28, ptr noalias %arg.29, ptr noalias %arg.30, ptr noalias %arg.31, ptr noalias %arg.32, ptr noalias %arg.33, ptr noalias %arg.34, ptr noalias %arg.35, ptr noalias %arg.36, ptr noalias %arg.37, ptr noalias %arg.38, ptr %out.0) {
entry:
  %value.28 = alloca i64, i64 1, align 8
  %value.29 = alloca i64, i64 1, align 8
  %value.36 = alloca i32, i64 1, align 8
  %value.34 = alloca i64, i64 1, align 8
  %value.31 = alloca i64, i64 1, align 8
  %value.32 = alloca i1, i64 1, align 8
  %value.24 = alloca double, i64 1, align 8
  %value.23 = alloca double, i64 1, align 8
  %value.25 = alloca double, i64 1, align 8
  store i64 0, ptr %value.28, align 8
  store i64 1, ptr %value.29, align 8
  store i32 1, ptr %value.36, align 4
  store i64 0, ptr %value.34, align 8
  br label %loop_header
loop_header:
  %phi.30 = phi ptr [ %value.28, %entry ], [ %value.31, %loop_latch ]
  %load.6.32.0 = load i32, ptr %phi.30, align 4
  %load.6.32.1 = load i32, ptr %arg.0, align 4
  %scalar.6.32 = icmp slt i32 %load.6.32.0, %load.6.32.1
  store i1 %scalar.6.32, ptr %value.32, align 1
  br i1 %scalar.6.32, label %loop_body, label %loop_exit
loop_body:
  call void @__ssa_sec_core_pack__sec_core_pack__planned_region_0(ptr %arg.1, ptr %phi.30, ptr %value.24, ptr %value.23)
  call void @__ssa_sec_core_pack__sec_core(ptr %arg.3, ptr %arg.4, ptr %arg.5, ptr %arg.6, ptr %arg.7, ptr %arg.8, ptr %arg.9, ptr %arg.10, ptr %arg.11, ptr %arg.12, ptr %arg.13, ptr %arg.14, ptr %arg.15, ptr %arg.16, ptr %arg.17, ptr %arg.18, ptr %arg.19, ptr %arg.20, ptr %value.24, ptr %arg.21, ptr %arg.22, ptr %arg.23, ptr %arg.24, ptr %arg.25, ptr %arg.26, ptr %arg.27, ptr %arg.28, ptr %arg.29, ptr %arg.30, ptr %arg.31, ptr %arg.32, ptr %arg.33, ptr %arg.34, ptr %arg.35, ptr %arg.36, ptr %arg.37, ptr %arg.38, ptr %value.24, ptr %value.25)
  call void @__ssa_sec_core_pack__sec_core_pack__planned_region_1(ptr %arg.2, ptr %phi.30, ptr %value.25)
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

define void @__ssa_sec_core_pack__sec_core__planned_region_0(ptr noalias %arg.0, ptr %arg.1, ptr noalias %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr noalias %arg.11, ptr noalias %arg.12, ptr noalias %arg.13, ptr noalias %arg.14, ptr noalias %arg.15, ptr noalias %arg.16, ptr noalias %arg.17, ptr noalias %arg.18, ptr noalias %arg.19, ptr %arg.20, ptr noalias %arg.21, ptr noalias %arg.22, ptr noalias %arg.23, ptr noalias %arg.24, ptr noalias %arg.25, ptr noalias %arg.26, ptr noalias %arg.27, ptr noalias %arg.28, ptr noalias %arg.29, ptr noalias %arg.30, ptr noalias %arg.31, ptr noalias %arg.32, ptr noalias %arg.33, ptr noalias %arg.34, ptr noalias %arg.35, ptr noalias %arg.36, ptr noalias %arg.37, ptr %out.0, ptr %out.1, ptr %out.2, ptr %out.3, ptr %out.4, ptr %out.5, ptr %out.6, ptr %out.7, ptr %out.8, ptr %out.9, ptr %out.10, ptr %out.11, ptr %out.12, ptr %out.13, ptr %out.14, ptr %out.15, ptr %out.16, ptr %out.17, ptr %out.18, ptr %out.19, ptr %out.20, ptr %out.21, ptr %out.22, ptr %out.23, ptr %out.24, ptr %out.25, ptr %out.26, ptr %out.27, ptr %out.28, ptr %out.29, ptr %out.30, ptr %out.31, ptr %out.32, ptr %out.33) {
entry:
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
  %load.0.178.0 = load double, ptr %arg.0, align 8
  %load.0.178.1 = load double, ptr %arg.1, align 8
  %scalar.0.178 = fmul double %load.0.178.0, %load.0.178.1
  store double %scalar.0.178, ptr %value.178, align 8
  %scalar.1.179 = fneg double %scalar.0.178
  store double %scalar.1.179, ptr %value.179, align 8
  %scalar.2.180 = call double @llvm.fma.f64(double %load.0.178.0, double %load.0.178.1, double %scalar.1.179)
  store double %scalar.2.180, ptr %value.180, align 8
  %load.3.181.1 = load double, ptr %arg.20, align 8
  %scalar.3.181 = fmul double %load.0.178.0, %load.3.181.1
  store double %scalar.3.181, ptr %value.181, align 8
  %scalar.4.182 = fadd double %scalar.2.180, %scalar.3.181
  store double %scalar.4.182, ptr %value.182, align 8
  %load.5.183.0 = load double, ptr %arg.19, align 8
  %scalar.5.183 = fmul double %load.5.183.0, %load.0.178.1
  store double %scalar.5.183, ptr %value.183, align 8
  %scalar.6.184 = fadd double %scalar.4.182, %scalar.5.183
  store double %scalar.6.184, ptr %value.184, align 8
  %scalar.7.185 = fadd double %scalar.0.178, %scalar.6.184
  store double %scalar.7.185, ptr %value.185, align 8
  %scalar.8.186 = fsub double %scalar.7.185, %scalar.0.178
  store double %scalar.8.186, ptr %value.186, align 8
  %scalar.9.187 = fsub double %scalar.6.184, %scalar.8.186
  store double %scalar.9.187, ptr %value.187, align 8
  %scalar.10.19 = fadd double %scalar.7.185, %scalar.9.187
  store double %scalar.10.19, ptr %out.1, align 8
  %load.11.188.0 = load double, ptr %arg.2, align 8
  %scalar.11.188 = fadd double %load.11.188.0, %scalar.7.185
  store double %scalar.11.188, ptr %value.188, align 8
  %scalar.12.189 = fsub double %scalar.11.188, %load.11.188.0
  store double %scalar.12.189, ptr %value.189, align 8
  %scalar.13.190 = fsub double %scalar.11.188, %scalar.12.189
  store double %scalar.13.190, ptr %value.190, align 8
  %scalar.14.191 = fsub double %load.11.188.0, %scalar.13.190
  store double %scalar.14.191, ptr %value.191, align 8
  %scalar.15.192 = fsub double %scalar.7.185, %scalar.12.189
  store double %scalar.15.192, ptr %value.192, align 8
  %scalar.16.193 = fadd double %scalar.14.191, %scalar.15.192
  store double %scalar.16.193, ptr %value.193, align 8
  %load.17.194.1 = load double, ptr %arg.21, align 8
  %scalar.17.194 = fadd double %scalar.16.193, %load.17.194.1
  store double %scalar.17.194, ptr %value.194, align 8
  %scalar.18.195 = fadd double %scalar.17.194, %scalar.9.187
  store double %scalar.18.195, ptr %value.195, align 8
  %scalar.19.196 = fadd double %scalar.11.188, %scalar.18.195
  store double %scalar.19.196, ptr %value.196, align 8
  %scalar.20.197 = fsub double %scalar.19.196, %scalar.11.188
  store double %scalar.20.197, ptr %value.197, align 8
  %scalar.21.198 = fsub double %scalar.18.195, %scalar.20.197
  store double %scalar.21.198, ptr %value.198, align 8
  %scalar.22.20 = fadd double %scalar.19.196, %scalar.21.198
  store double %scalar.22.20, ptr %out.2, align 8
  %scalar.23.199 = fmul double %load.0.178.1, %scalar.19.196
  store double %scalar.23.199, ptr %value.199, align 8
  %scalar.24.200 = fneg double %scalar.23.199
  store double %scalar.24.200, ptr %value.200, align 8
  %scalar.25.201 = call double @llvm.fma.f64(double %load.0.178.1, double %scalar.19.196, double %scalar.24.200)
  store double %scalar.25.201, ptr %value.201, align 8
  %scalar.26.202 = fmul double %load.0.178.1, %scalar.21.198
  store double %scalar.26.202, ptr %value.202, align 8
  %scalar.27.203 = fadd double %scalar.25.201, %scalar.26.202
  store double %scalar.27.203, ptr %value.203, align 8
  %scalar.28.204 = fmul double %load.3.181.1, %scalar.19.196
  store double %scalar.28.204, ptr %value.204, align 8
  %scalar.29.205 = fadd double %scalar.27.203, %scalar.28.204
  store double %scalar.29.205, ptr %value.205, align 8
  %scalar.30.206 = fadd double %scalar.23.199, %scalar.29.205
  store double %scalar.30.206, ptr %value.206, align 8
  %scalar.31.207 = fsub double %scalar.30.206, %scalar.23.199
  store double %scalar.31.207, ptr %value.207, align 8
  %scalar.32.208 = fsub double %scalar.29.205, %scalar.31.207
  store double %scalar.32.208, ptr %value.208, align 8
  %scalar.33.21 = fadd double %scalar.30.206, %scalar.32.208
  store double %scalar.33.21, ptr %out.3, align 8
  %load.34.209.0 = load double, ptr %arg.3, align 8
  %scalar.34.209 = fadd double %load.34.209.0, %scalar.30.206
  store double %scalar.34.209, ptr %value.209, align 8
  %scalar.35.210 = fsub double %scalar.34.209, %load.34.209.0
  store double %scalar.35.210, ptr %value.210, align 8
  %scalar.36.211 = fsub double %scalar.34.209, %scalar.35.210
  store double %scalar.36.211, ptr %value.211, align 8
  %scalar.37.212 = fsub double %load.34.209.0, %scalar.36.211
  store double %scalar.37.212, ptr %value.212, align 8
  %scalar.38.213 = fsub double %scalar.30.206, %scalar.35.210
  store double %scalar.38.213, ptr %value.213, align 8
  %scalar.39.214 = fadd double %scalar.37.212, %scalar.38.213
  store double %scalar.39.214, ptr %value.214, align 8
  %load.40.215.1 = load double, ptr %arg.22, align 8
  %scalar.40.215 = fadd double %scalar.39.214, %load.40.215.1
  store double %scalar.40.215, ptr %value.215, align 8
  %scalar.41.216 = fadd double %scalar.40.215, %scalar.32.208
  store double %scalar.41.216, ptr %value.216, align 8
  %scalar.42.217 = fadd double %scalar.34.209, %scalar.41.216
  store double %scalar.42.217, ptr %value.217, align 8
  %scalar.43.218 = fsub double %scalar.42.217, %scalar.34.209
  store double %scalar.43.218, ptr %value.218, align 8
  %scalar.44.219 = fsub double %scalar.41.216, %scalar.43.218
  store double %scalar.44.219, ptr %value.219, align 8
  %scalar.45.22 = fadd double %scalar.42.217, %scalar.44.219
  store double %scalar.45.22, ptr %out.4, align 8
  %scalar.46.220 = fmul double %load.0.178.1, %scalar.42.217
  store double %scalar.46.220, ptr %value.220, align 8
  %scalar.47.221 = fneg double %scalar.46.220
  store double %scalar.47.221, ptr %value.221, align 8
  %scalar.48.222 = call double @llvm.fma.f64(double %load.0.178.1, double %scalar.42.217, double %scalar.47.221)
  store double %scalar.48.222, ptr %value.222, align 8
  %scalar.49.223 = fmul double %load.0.178.1, %scalar.44.219
  store double %scalar.49.223, ptr %value.223, align 8
  %scalar.50.224 = fadd double %scalar.48.222, %scalar.49.223
  store double %scalar.50.224, ptr %value.224, align 8
  %scalar.51.225 = fmul double %load.3.181.1, %scalar.42.217
  store double %scalar.51.225, ptr %value.225, align 8
  %scalar.52.226 = fadd double %scalar.50.224, %scalar.51.225
  store double %scalar.52.226, ptr %value.226, align 8
  %scalar.53.227 = fadd double %scalar.46.220, %scalar.52.226
  store double %scalar.53.227, ptr %value.227, align 8
  %scalar.54.228 = fsub double %scalar.53.227, %scalar.46.220
  store double %scalar.54.228, ptr %value.228, align 8
  %scalar.55.229 = fsub double %scalar.52.226, %scalar.54.228
  store double %scalar.55.229, ptr %value.229, align 8
  %scalar.56.23 = fadd double %scalar.53.227, %scalar.55.229
  store double %scalar.56.23, ptr %out.5, align 8
  %load.57.230.0 = load double, ptr %arg.4, align 8
  %scalar.57.230 = fadd double %load.57.230.0, %scalar.53.227
  store double %scalar.57.230, ptr %value.230, align 8
  %scalar.58.231 = fsub double %scalar.57.230, %load.57.230.0
  store double %scalar.58.231, ptr %value.231, align 8
  %scalar.59.232 = fsub double %scalar.57.230, %scalar.58.231
  store double %scalar.59.232, ptr %value.232, align 8
  %scalar.60.233 = fsub double %load.57.230.0, %scalar.59.232
  store double %scalar.60.233, ptr %value.233, align 8
  %scalar.61.234 = fsub double %scalar.53.227, %scalar.58.231
  store double %scalar.61.234, ptr %value.234, align 8
  %scalar.62.235 = fadd double %scalar.60.233, %scalar.61.234
  store double %scalar.62.235, ptr %value.235, align 8
  %load.63.236.1 = load double, ptr %arg.23, align 8
  %scalar.63.236 = fadd double %scalar.62.235, %load.63.236.1
  store double %scalar.63.236, ptr %value.236, align 8
  %scalar.64.237 = fadd double %scalar.63.236, %scalar.55.229
  store double %scalar.64.237, ptr %value.237, align 8
  %scalar.65.238 = fadd double %scalar.57.230, %scalar.64.237
  store double %scalar.65.238, ptr %value.238, align 8
  %scalar.66.239 = fsub double %scalar.65.238, %scalar.57.230
  store double %scalar.66.239, ptr %value.239, align 8
  %scalar.67.240 = fsub double %scalar.64.237, %scalar.66.239
  store double %scalar.67.240, ptr %value.240, align 8
  %scalar.68.24 = fadd double %scalar.65.238, %scalar.67.240
  store double %scalar.68.24, ptr %out.6, align 8
  %scalar.69.241 = fmul double %load.0.178.1, %scalar.65.238
  store double %scalar.69.241, ptr %value.241, align 8
  %scalar.70.242 = fneg double %scalar.69.241
  store double %scalar.70.242, ptr %value.242, align 8
  %scalar.71.243 = call double @llvm.fma.f64(double %load.0.178.1, double %scalar.65.238, double %scalar.70.242)
  store double %scalar.71.243, ptr %value.243, align 8
  %scalar.72.244 = fmul double %load.0.178.1, %scalar.67.240
  store double %scalar.72.244, ptr %value.244, align 8
  %scalar.73.245 = fadd double %scalar.71.243, %scalar.72.244
  store double %scalar.73.245, ptr %value.245, align 8
  %scalar.74.246 = fmul double %load.3.181.1, %scalar.65.238
  store double %scalar.74.246, ptr %value.246, align 8
  %scalar.75.247 = fadd double %scalar.73.245, %scalar.74.246
  store double %scalar.75.247, ptr %value.247, align 8
  %scalar.76.248 = fadd double %scalar.69.241, %scalar.75.247
  store double %scalar.76.248, ptr %value.248, align 8
  %scalar.77.249 = fsub double %scalar.76.248, %scalar.69.241
  store double %scalar.77.249, ptr %value.249, align 8
  %scalar.78.250 = fsub double %scalar.75.247, %scalar.77.249
  store double %scalar.78.250, ptr %value.250, align 8
  %scalar.79.25 = fadd double %scalar.76.248, %scalar.78.250
  store double %scalar.79.25, ptr %out.7, align 8
  %load.80.251.0 = load double, ptr %arg.5, align 8
  %scalar.80.251 = fadd double %load.80.251.0, %scalar.76.248
  store double %scalar.80.251, ptr %value.251, align 8
  %scalar.81.252 = fsub double %scalar.80.251, %load.80.251.0
  store double %scalar.81.252, ptr %value.252, align 8
  %scalar.82.253 = fsub double %scalar.80.251, %scalar.81.252
  store double %scalar.82.253, ptr %value.253, align 8
  %scalar.83.254 = fsub double %load.80.251.0, %scalar.82.253
  store double %scalar.83.254, ptr %value.254, align 8
  %scalar.84.255 = fsub double %scalar.76.248, %scalar.81.252
  store double %scalar.84.255, ptr %value.255, align 8
  %scalar.85.256 = fadd double %scalar.83.254, %scalar.84.255
  store double %scalar.85.256, ptr %value.256, align 8
  %load.86.257.1 = load double, ptr %arg.24, align 8
  %scalar.86.257 = fadd double %scalar.85.256, %load.86.257.1
  store double %scalar.86.257, ptr %value.257, align 8
  %scalar.87.258 = fadd double %scalar.86.257, %scalar.78.250
  store double %scalar.87.258, ptr %value.258, align 8
  %scalar.88.259 = fadd double %scalar.80.251, %scalar.87.258
  store double %scalar.88.259, ptr %value.259, align 8
  %scalar.89.260 = fsub double %scalar.88.259, %scalar.80.251
  store double %scalar.89.260, ptr %value.260, align 8
  %scalar.90.261 = fsub double %scalar.87.258, %scalar.89.260
  store double %scalar.90.261, ptr %value.261, align 8
  %scalar.91.26 = fadd double %scalar.88.259, %scalar.90.261
  store double %scalar.91.26, ptr %out.8, align 8
  %scalar.92.262 = fmul double %load.0.178.1, %scalar.88.259
  store double %scalar.92.262, ptr %value.262, align 8
  %scalar.93.263 = fneg double %scalar.92.262
  store double %scalar.93.263, ptr %value.263, align 8
  %scalar.94.264 = call double @llvm.fma.f64(double %load.0.178.1, double %scalar.88.259, double %scalar.93.263)
  store double %scalar.94.264, ptr %value.264, align 8
  %scalar.95.265 = fmul double %load.0.178.1, %scalar.90.261
  store double %scalar.95.265, ptr %value.265, align 8
  %scalar.96.266 = fadd double %scalar.94.264, %scalar.95.265
  store double %scalar.96.266, ptr %value.266, align 8
  %scalar.97.267 = fmul double %load.3.181.1, %scalar.88.259
  store double %scalar.97.267, ptr %value.267, align 8
  %scalar.98.268 = fadd double %scalar.96.266, %scalar.97.267
  store double %scalar.98.268, ptr %value.268, align 8
  %scalar.99.269 = fadd double %scalar.92.262, %scalar.98.268
  store double %scalar.99.269, ptr %value.269, align 8
  %scalar.100.270 = fsub double %scalar.99.269, %scalar.92.262
  store double %scalar.100.270, ptr %value.270, align 8
  %scalar.101.271 = fsub double %scalar.98.268, %scalar.100.270
  store double %scalar.101.271, ptr %value.271, align 8
  %scalar.102.27 = fadd double %scalar.99.269, %scalar.101.271
  store double %scalar.102.27, ptr %out.9, align 8
  %load.103.272.0 = load double, ptr %arg.6, align 8
  %scalar.103.272 = fadd double %load.103.272.0, %scalar.99.269
  store double %scalar.103.272, ptr %value.272, align 8
  %scalar.104.273 = fsub double %scalar.103.272, %load.103.272.0
  store double %scalar.104.273, ptr %value.273, align 8
  %scalar.105.274 = fsub double %scalar.103.272, %scalar.104.273
  store double %scalar.105.274, ptr %value.274, align 8
  %scalar.106.275 = fsub double %load.103.272.0, %scalar.105.274
  store double %scalar.106.275, ptr %value.275, align 8
  %scalar.107.276 = fsub double %scalar.99.269, %scalar.104.273
  store double %scalar.107.276, ptr %value.276, align 8
  %scalar.108.277 = fadd double %scalar.106.275, %scalar.107.276
  store double %scalar.108.277, ptr %value.277, align 8
  %load.109.278.1 = load double, ptr %arg.25, align 8
  %scalar.109.278 = fadd double %scalar.108.277, %load.109.278.1
  store double %scalar.109.278, ptr %value.278, align 8
  %scalar.110.279 = fadd double %scalar.109.278, %scalar.101.271
  store double %scalar.110.279, ptr %value.279, align 8
  %scalar.111.280 = fadd double %scalar.103.272, %scalar.110.279
  store double %scalar.111.280, ptr %value.280, align 8
  %scalar.112.281 = fsub double %scalar.111.280, %scalar.103.272
  store double %scalar.112.281, ptr %value.281, align 8
  %scalar.113.282 = fsub double %scalar.110.279, %scalar.112.281
  store double %scalar.113.282, ptr %value.282, align 8
  %scalar.114.28 = fadd double %scalar.111.280, %scalar.113.282
  store double %scalar.114.28, ptr %out.10, align 8
  %scalar.115.283 = fmul double %load.0.178.1, %scalar.111.280
  store double %scalar.115.283, ptr %value.283, align 8
  %scalar.116.284 = fneg double %scalar.115.283
  store double %scalar.116.284, ptr %value.284, align 8
  %scalar.117.285 = call double @llvm.fma.f64(double %load.0.178.1, double %scalar.111.280, double %scalar.116.284)
  store double %scalar.117.285, ptr %value.285, align 8
  %scalar.118.286 = fmul double %load.0.178.1, %scalar.113.282
  store double %scalar.118.286, ptr %value.286, align 8
  %scalar.119.287 = fadd double %scalar.117.285, %scalar.118.286
  store double %scalar.119.287, ptr %value.287, align 8
  %scalar.120.288 = fmul double %load.3.181.1, %scalar.111.280
  store double %scalar.120.288, ptr %value.288, align 8
  %scalar.121.289 = fadd double %scalar.119.287, %scalar.120.288
  store double %scalar.121.289, ptr %value.289, align 8
  %scalar.122.290 = fadd double %scalar.115.283, %scalar.121.289
  store double %scalar.122.290, ptr %value.290, align 8
  %scalar.123.291 = fsub double %scalar.122.290, %scalar.115.283
  store double %scalar.123.291, ptr %value.291, align 8
  %scalar.124.292 = fsub double %scalar.121.289, %scalar.123.291
  store double %scalar.124.292, ptr %value.292, align 8
  %scalar.125.29 = fadd double %scalar.122.290, %scalar.124.292
  store double %scalar.125.29, ptr %out.11, align 8
  %load.126.293.0 = load double, ptr %arg.7, align 8
  %scalar.126.293 = fadd double %load.126.293.0, %scalar.122.290
  store double %scalar.126.293, ptr %value.293, align 8
  %scalar.127.294 = fsub double %scalar.126.293, %load.126.293.0
  store double %scalar.127.294, ptr %value.294, align 8
  %scalar.128.295 = fsub double %scalar.126.293, %scalar.127.294
  store double %scalar.128.295, ptr %value.295, align 8
  %scalar.129.296 = fsub double %load.126.293.0, %scalar.128.295
  store double %scalar.129.296, ptr %value.296, align 8
  %scalar.130.297 = fsub double %scalar.122.290, %scalar.127.294
  store double %scalar.130.297, ptr %value.297, align 8
  %scalar.131.298 = fadd double %scalar.129.296, %scalar.130.297
  store double %scalar.131.298, ptr %value.298, align 8
  %load.132.299.1 = load double, ptr %arg.26, align 8
  %scalar.132.299 = fadd double %scalar.131.298, %load.132.299.1
  store double %scalar.132.299, ptr %value.299, align 8
  %scalar.133.300 = fadd double %scalar.132.299, %scalar.124.292
  store double %scalar.133.300, ptr %value.300, align 8
  %scalar.134.301 = fadd double %scalar.126.293, %scalar.133.300
  store double %scalar.134.301, ptr %value.301, align 8
  %scalar.135.302 = fsub double %scalar.134.301, %scalar.126.293
  store double %scalar.135.302, ptr %value.302, align 8
  %scalar.136.303 = fsub double %scalar.133.300, %scalar.135.302
  store double %scalar.136.303, ptr %value.303, align 8
  %scalar.137.30 = fadd double %scalar.134.301, %scalar.136.303
  store double %scalar.137.30, ptr %out.12, align 8
  %scalar.138.304 = fmul double %load.0.178.1, %scalar.134.301
  store double %scalar.138.304, ptr %value.304, align 8
  %scalar.139.305 = fneg double %scalar.138.304
  store double %scalar.139.305, ptr %value.305, align 8
  %scalar.140.306 = call double @llvm.fma.f64(double %load.0.178.1, double %scalar.134.301, double %scalar.139.305)
  store double %scalar.140.306, ptr %value.306, align 8
  %scalar.141.307 = fmul double %load.0.178.1, %scalar.136.303
  store double %scalar.141.307, ptr %value.307, align 8
  %scalar.142.308 = fadd double %scalar.140.306, %scalar.141.307
  store double %scalar.142.308, ptr %value.308, align 8
  %scalar.143.309 = fmul double %load.3.181.1, %scalar.134.301
  store double %scalar.143.309, ptr %value.309, align 8
  %scalar.144.310 = fadd double %scalar.142.308, %scalar.143.309
  store double %scalar.144.310, ptr %value.310, align 8
  %scalar.145.311 = fadd double %scalar.138.304, %scalar.144.310
  store double %scalar.145.311, ptr %value.311, align 8
  %scalar.146.312 = fsub double %scalar.145.311, %scalar.138.304
  store double %scalar.146.312, ptr %value.312, align 8
  %scalar.147.313 = fsub double %scalar.144.310, %scalar.146.312
  store double %scalar.147.313, ptr %value.313, align 8
  %scalar.148.31 = fadd double %scalar.145.311, %scalar.147.313
  store double %scalar.148.31, ptr %out.13, align 8
  %load.149.314.0 = load double, ptr %arg.8, align 8
  %scalar.149.314 = fadd double %load.149.314.0, %scalar.145.311
  store double %scalar.149.314, ptr %value.314, align 8
  %scalar.150.315 = fsub double %scalar.149.314, %load.149.314.0
  store double %scalar.150.315, ptr %value.315, align 8
  %scalar.151.316 = fsub double %scalar.149.314, %scalar.150.315
  store double %scalar.151.316, ptr %value.316, align 8
  %scalar.152.317 = fsub double %load.149.314.0, %scalar.151.316
  store double %scalar.152.317, ptr %value.317, align 8
  %scalar.153.318 = fsub double %scalar.145.311, %scalar.150.315
  store double %scalar.153.318, ptr %value.318, align 8
  %scalar.154.319 = fadd double %scalar.152.317, %scalar.153.318
  store double %scalar.154.319, ptr %value.319, align 8
  %load.155.320.1 = load double, ptr %arg.27, align 8
  %scalar.155.320 = fadd double %scalar.154.319, %load.155.320.1
  store double %scalar.155.320, ptr %value.320, align 8
  %scalar.156.321 = fadd double %scalar.155.320, %scalar.147.313
  store double %scalar.156.321, ptr %value.321, align 8
  %scalar.157.322 = fadd double %scalar.149.314, %scalar.156.321
  store double %scalar.157.322, ptr %value.322, align 8
  %scalar.158.323 = fsub double %scalar.157.322, %scalar.149.314
  store double %scalar.158.323, ptr %value.323, align 8
  %scalar.159.324 = fsub double %scalar.156.321, %scalar.158.323
  store double %scalar.159.324, ptr %value.324, align 8
  %scalar.160.32 = fadd double %scalar.157.322, %scalar.159.324
  store double %scalar.160.32, ptr %out.14, align 8
  %scalar.161.325 = fmul double %load.0.178.1, %scalar.157.322
  store double %scalar.161.325, ptr %value.325, align 8
  %scalar.162.326 = fneg double %scalar.161.325
  store double %scalar.162.326, ptr %value.326, align 8
  %scalar.163.327 = call double @llvm.fma.f64(double %load.0.178.1, double %scalar.157.322, double %scalar.162.326)
  store double %scalar.163.327, ptr %value.327, align 8
  %scalar.164.328 = fmul double %load.0.178.1, %scalar.159.324
  store double %scalar.164.328, ptr %value.328, align 8
  %scalar.165.329 = fadd double %scalar.163.327, %scalar.164.328
  store double %scalar.165.329, ptr %value.329, align 8
  %scalar.166.330 = fmul double %load.3.181.1, %scalar.157.322
  store double %scalar.166.330, ptr %value.330, align 8
  %scalar.167.331 = fadd double %scalar.165.329, %scalar.166.330
  store double %scalar.167.331, ptr %value.331, align 8
  %scalar.168.332 = fadd double %scalar.161.325, %scalar.167.331
  store double %scalar.168.332, ptr %value.332, align 8
  %scalar.169.333 = fsub double %scalar.168.332, %scalar.161.325
  store double %scalar.169.333, ptr %value.333, align 8
  %scalar.170.334 = fsub double %scalar.167.331, %scalar.169.333
  store double %scalar.170.334, ptr %value.334, align 8
  %scalar.171.33 = fadd double %scalar.168.332, %scalar.170.334
  store double %scalar.171.33, ptr %out.15, align 8
  %load.172.335.0 = load double, ptr %arg.9, align 8
  %scalar.172.335 = fadd double %load.172.335.0, %scalar.168.332
  store double %scalar.172.335, ptr %value.335, align 8
  %scalar.173.336 = fsub double %scalar.172.335, %load.172.335.0
  store double %scalar.173.336, ptr %value.336, align 8
  %scalar.174.337 = fsub double %scalar.172.335, %scalar.173.336
  store double %scalar.174.337, ptr %value.337, align 8
  %scalar.175.338 = fsub double %load.172.335.0, %scalar.174.337
  store double %scalar.175.338, ptr %value.338, align 8
  %scalar.176.339 = fsub double %scalar.168.332, %scalar.173.336
  store double %scalar.176.339, ptr %value.339, align 8
  %scalar.177.340 = fadd double %scalar.175.338, %scalar.176.339
  store double %scalar.177.340, ptr %value.340, align 8
  %load.178.341.1 = load double, ptr %arg.28, align 8
  %scalar.178.341 = fadd double %scalar.177.340, %load.178.341.1
  store double %scalar.178.341, ptr %value.341, align 8
  %scalar.179.342 = fadd double %scalar.178.341, %scalar.170.334
  store double %scalar.179.342, ptr %value.342, align 8
  %scalar.180.343 = fadd double %scalar.172.335, %scalar.179.342
  store double %scalar.180.343, ptr %value.343, align 8
  %scalar.181.344 = fsub double %scalar.180.343, %scalar.172.335
  store double %scalar.181.344, ptr %value.344, align 8
  %scalar.182.345 = fsub double %scalar.179.342, %scalar.181.344
  store double %scalar.182.345, ptr %value.345, align 8
  %scalar.183.34 = fadd double %scalar.180.343, %scalar.182.345
  store double %scalar.183.34, ptr %out.16, align 8
  %scalar.184.346 = fmul double %load.0.178.1, %scalar.180.343
  store double %scalar.184.346, ptr %value.346, align 8
  %scalar.185.347 = fneg double %scalar.184.346
  store double %scalar.185.347, ptr %value.347, align 8
  %scalar.186.348 = call double @llvm.fma.f64(double %load.0.178.1, double %scalar.180.343, double %scalar.185.347)
  store double %scalar.186.348, ptr %value.348, align 8
  %scalar.187.349 = fmul double %load.0.178.1, %scalar.182.345
  store double %scalar.187.349, ptr %value.349, align 8
  %scalar.188.350 = fadd double %scalar.186.348, %scalar.187.349
  store double %scalar.188.350, ptr %value.350, align 8
  %scalar.189.351 = fmul double %load.3.181.1, %scalar.180.343
  store double %scalar.189.351, ptr %value.351, align 8
  %scalar.190.352 = fadd double %scalar.188.350, %scalar.189.351
  store double %scalar.190.352, ptr %value.352, align 8
  %scalar.191.353 = fadd double %scalar.184.346, %scalar.190.352
  store double %scalar.191.353, ptr %value.353, align 8
  %scalar.192.354 = fsub double %scalar.191.353, %scalar.184.346
  store double %scalar.192.354, ptr %value.354, align 8
  %scalar.193.355 = fsub double %scalar.190.352, %scalar.192.354
  store double %scalar.193.355, ptr %value.355, align 8
  %scalar.194.35 = fadd double %scalar.191.353, %scalar.193.355
  store double %scalar.194.35, ptr %out.17, align 8
  %load.195.356.0 = load double, ptr %arg.10, align 8
  %scalar.195.356 = fadd double %load.195.356.0, %scalar.191.353
  store double %scalar.195.356, ptr %value.356, align 8
  %scalar.196.357 = fsub double %scalar.195.356, %load.195.356.0
  store double %scalar.196.357, ptr %value.357, align 8
  %scalar.197.358 = fsub double %scalar.195.356, %scalar.196.357
  store double %scalar.197.358, ptr %value.358, align 8
  %scalar.198.359 = fsub double %load.195.356.0, %scalar.197.358
  store double %scalar.198.359, ptr %value.359, align 8
  %scalar.199.360 = fsub double %scalar.191.353, %scalar.196.357
  store double %scalar.199.360, ptr %value.360, align 8
  %scalar.200.361 = fadd double %scalar.198.359, %scalar.199.360
  store double %scalar.200.361, ptr %value.361, align 8
  %load.201.362.1 = load double, ptr %arg.29, align 8
  %scalar.201.362 = fadd double %scalar.200.361, %load.201.362.1
  store double %scalar.201.362, ptr %value.362, align 8
  %scalar.202.363 = fadd double %scalar.201.362, %scalar.193.355
  store double %scalar.202.363, ptr %value.363, align 8
  %scalar.203.364 = fadd double %scalar.195.356, %scalar.202.363
  store double %scalar.203.364, ptr %value.364, align 8
  %scalar.204.365 = fsub double %scalar.203.364, %scalar.195.356
  store double %scalar.204.365, ptr %value.365, align 8
  %scalar.205.366 = fsub double %scalar.202.363, %scalar.204.365
  store double %scalar.205.366, ptr %value.366, align 8
  %scalar.206.36 = fadd double %scalar.203.364, %scalar.205.366
  store double %scalar.206.36, ptr %out.18, align 8
  %scalar.207.367 = fmul double %load.0.178.1, %scalar.203.364
  store double %scalar.207.367, ptr %value.367, align 8
  %scalar.208.368 = fneg double %scalar.207.367
  store double %scalar.208.368, ptr %value.368, align 8
  %scalar.209.369 = call double @llvm.fma.f64(double %load.0.178.1, double %scalar.203.364, double %scalar.208.368)
  store double %scalar.209.369, ptr %value.369, align 8
  %scalar.210.370 = fmul double %load.0.178.1, %scalar.205.366
  store double %scalar.210.370, ptr %value.370, align 8
  %scalar.211.371 = fadd double %scalar.209.369, %scalar.210.370
  store double %scalar.211.371, ptr %value.371, align 8
  %scalar.212.372 = fmul double %load.3.181.1, %scalar.203.364
  store double %scalar.212.372, ptr %value.372, align 8
  %scalar.213.373 = fadd double %scalar.211.371, %scalar.212.372
  store double %scalar.213.373, ptr %value.373, align 8
  %scalar.214.374 = fadd double %scalar.207.367, %scalar.213.373
  store double %scalar.214.374, ptr %value.374, align 8
  %scalar.215.375 = fsub double %scalar.214.374, %scalar.207.367
  store double %scalar.215.375, ptr %value.375, align 8
  %scalar.216.376 = fsub double %scalar.213.373, %scalar.215.375
  store double %scalar.216.376, ptr %value.376, align 8
  %scalar.217.37 = fadd double %scalar.214.374, %scalar.216.376
  store double %scalar.217.37, ptr %out.19, align 8
  %load.218.377.0 = load double, ptr %arg.11, align 8
  %scalar.218.377 = fadd double %load.218.377.0, %scalar.214.374
  store double %scalar.218.377, ptr %value.377, align 8
  %scalar.219.378 = fsub double %scalar.218.377, %load.218.377.0
  store double %scalar.219.378, ptr %value.378, align 8
  %scalar.220.379 = fsub double %scalar.218.377, %scalar.219.378
  store double %scalar.220.379, ptr %value.379, align 8
  %scalar.221.380 = fsub double %load.218.377.0, %scalar.220.379
  store double %scalar.221.380, ptr %value.380, align 8
  %scalar.222.381 = fsub double %scalar.214.374, %scalar.219.378
  store double %scalar.222.381, ptr %value.381, align 8
  %scalar.223.382 = fadd double %scalar.221.380, %scalar.222.381
  store double %scalar.223.382, ptr %value.382, align 8
  %load.224.383.1 = load double, ptr %arg.30, align 8
  %scalar.224.383 = fadd double %scalar.223.382, %load.224.383.1
  store double %scalar.224.383, ptr %value.383, align 8
  %scalar.225.384 = fadd double %scalar.224.383, %scalar.216.376
  store double %scalar.225.384, ptr %value.384, align 8
  %scalar.226.385 = fadd double %scalar.218.377, %scalar.225.384
  store double %scalar.226.385, ptr %value.385, align 8
  %scalar.227.386 = fsub double %scalar.226.385, %scalar.218.377
  store double %scalar.227.386, ptr %value.386, align 8
  %scalar.228.387 = fsub double %scalar.225.384, %scalar.227.386
  store double %scalar.228.387, ptr %value.387, align 8
  %scalar.229.38 = fadd double %scalar.226.385, %scalar.228.387
  store double %scalar.229.38, ptr %out.20, align 8
  %scalar.230.388 = fmul double %load.0.178.1, %scalar.226.385
  store double %scalar.230.388, ptr %value.388, align 8
  %scalar.231.389 = fneg double %scalar.230.388
  store double %scalar.231.389, ptr %value.389, align 8
  %scalar.232.390 = call double @llvm.fma.f64(double %load.0.178.1, double %scalar.226.385, double %scalar.231.389)
  store double %scalar.232.390, ptr %value.390, align 8
  %scalar.233.391 = fmul double %load.0.178.1, %scalar.228.387
  store double %scalar.233.391, ptr %value.391, align 8
  %scalar.234.392 = fadd double %scalar.232.390, %scalar.233.391
  store double %scalar.234.392, ptr %value.392, align 8
  %scalar.235.393 = fmul double %load.3.181.1, %scalar.226.385
  store double %scalar.235.393, ptr %value.393, align 8
  %scalar.236.394 = fadd double %scalar.234.392, %scalar.235.393
  store double %scalar.236.394, ptr %value.394, align 8
  %scalar.237.395 = fadd double %scalar.230.388, %scalar.236.394
  store double %scalar.237.395, ptr %value.395, align 8
  %scalar.238.396 = fsub double %scalar.237.395, %scalar.230.388
  store double %scalar.238.396, ptr %value.396, align 8
  %scalar.239.397 = fsub double %scalar.236.394, %scalar.238.396
  store double %scalar.239.397, ptr %value.397, align 8
  %scalar.240.39 = fadd double %scalar.237.395, %scalar.239.397
  store double %scalar.240.39, ptr %out.21, align 8
  %load.241.398.0 = load double, ptr %arg.12, align 8
  %scalar.241.398 = fadd double %load.241.398.0, %scalar.237.395
  store double %scalar.241.398, ptr %value.398, align 8
  %scalar.242.399 = fsub double %scalar.241.398, %load.241.398.0
  store double %scalar.242.399, ptr %value.399, align 8
  %scalar.243.400 = fsub double %scalar.241.398, %scalar.242.399
  store double %scalar.243.400, ptr %value.400, align 8
  %scalar.244.401 = fsub double %load.241.398.0, %scalar.243.400
  store double %scalar.244.401, ptr %value.401, align 8
  %scalar.245.402 = fsub double %scalar.237.395, %scalar.242.399
  store double %scalar.245.402, ptr %value.402, align 8
  %scalar.246.403 = fadd double %scalar.244.401, %scalar.245.402
  store double %scalar.246.403, ptr %value.403, align 8
  %load.247.404.1 = load double, ptr %arg.31, align 8
  %scalar.247.404 = fadd double %scalar.246.403, %load.247.404.1
  store double %scalar.247.404, ptr %value.404, align 8
  %scalar.248.405 = fadd double %scalar.247.404, %scalar.239.397
  store double %scalar.248.405, ptr %value.405, align 8
  %scalar.249.406 = fadd double %scalar.241.398, %scalar.248.405
  store double %scalar.249.406, ptr %value.406, align 8
  %scalar.250.407 = fsub double %scalar.249.406, %scalar.241.398
  store double %scalar.250.407, ptr %value.407, align 8
  %scalar.251.408 = fsub double %scalar.248.405, %scalar.250.407
  store double %scalar.251.408, ptr %value.408, align 8
  %scalar.252.40 = fadd double %scalar.249.406, %scalar.251.408
  store double %scalar.252.40, ptr %out.22, align 8
  %scalar.253.409 = fmul double %load.0.178.1, %scalar.249.406
  store double %scalar.253.409, ptr %value.409, align 8
  %scalar.254.410 = fneg double %scalar.253.409
  store double %scalar.254.410, ptr %value.410, align 8
  %scalar.255.411 = call double @llvm.fma.f64(double %load.0.178.1, double %scalar.249.406, double %scalar.254.410)
  store double %scalar.255.411, ptr %value.411, align 8
  %scalar.256.412 = fmul double %load.0.178.1, %scalar.251.408
  store double %scalar.256.412, ptr %value.412, align 8
  %scalar.257.413 = fadd double %scalar.255.411, %scalar.256.412
  store double %scalar.257.413, ptr %value.413, align 8
  %scalar.258.414 = fmul double %load.3.181.1, %scalar.249.406
  store double %scalar.258.414, ptr %value.414, align 8
  %scalar.259.415 = fadd double %scalar.257.413, %scalar.258.414
  store double %scalar.259.415, ptr %value.415, align 8
  %scalar.260.416 = fadd double %scalar.253.409, %scalar.259.415
  store double %scalar.260.416, ptr %value.416, align 8
  %scalar.261.417 = fsub double %scalar.260.416, %scalar.253.409
  store double %scalar.261.417, ptr %value.417, align 8
  %scalar.262.418 = fsub double %scalar.259.415, %scalar.261.417
  store double %scalar.262.418, ptr %value.418, align 8
  %scalar.263.41 = fadd double %scalar.260.416, %scalar.262.418
  store double %scalar.263.41, ptr %out.23, align 8
  %load.264.419.0 = load double, ptr %arg.13, align 8
  %scalar.264.419 = fadd double %load.264.419.0, %scalar.260.416
  store double %scalar.264.419, ptr %value.419, align 8
  %scalar.265.420 = fsub double %scalar.264.419, %load.264.419.0
  store double %scalar.265.420, ptr %value.420, align 8
  %scalar.266.421 = fsub double %scalar.264.419, %scalar.265.420
  store double %scalar.266.421, ptr %value.421, align 8
  %scalar.267.422 = fsub double %load.264.419.0, %scalar.266.421
  store double %scalar.267.422, ptr %value.422, align 8
  %scalar.268.423 = fsub double %scalar.260.416, %scalar.265.420
  store double %scalar.268.423, ptr %value.423, align 8
  %scalar.269.424 = fadd double %scalar.267.422, %scalar.268.423
  store double %scalar.269.424, ptr %value.424, align 8
  %load.270.425.1 = load double, ptr %arg.32, align 8
  %scalar.270.425 = fadd double %scalar.269.424, %load.270.425.1
  store double %scalar.270.425, ptr %value.425, align 8
  %scalar.271.426 = fadd double %scalar.270.425, %scalar.262.418
  store double %scalar.271.426, ptr %value.426, align 8
  %scalar.272.427 = fadd double %scalar.264.419, %scalar.271.426
  store double %scalar.272.427, ptr %value.427, align 8
  %scalar.273.428 = fsub double %scalar.272.427, %scalar.264.419
  store double %scalar.273.428, ptr %value.428, align 8
  %scalar.274.429 = fsub double %scalar.271.426, %scalar.273.428
  store double %scalar.274.429, ptr %value.429, align 8
  %scalar.275.42 = fadd double %scalar.272.427, %scalar.274.429
  store double %scalar.275.42, ptr %out.24, align 8
  %scalar.276.430 = fmul double %load.0.178.1, %scalar.272.427
  store double %scalar.276.430, ptr %value.430, align 8
  %scalar.277.431 = fneg double %scalar.276.430
  store double %scalar.277.431, ptr %value.431, align 8
  %scalar.278.432 = call double @llvm.fma.f64(double %load.0.178.1, double %scalar.272.427, double %scalar.277.431)
  store double %scalar.278.432, ptr %value.432, align 8
  %scalar.279.433 = fmul double %load.0.178.1, %scalar.274.429
  store double %scalar.279.433, ptr %value.433, align 8
  %scalar.280.434 = fadd double %scalar.278.432, %scalar.279.433
  store double %scalar.280.434, ptr %value.434, align 8
  %scalar.281.435 = fmul double %load.3.181.1, %scalar.272.427
  store double %scalar.281.435, ptr %value.435, align 8
  %scalar.282.436 = fadd double %scalar.280.434, %scalar.281.435
  store double %scalar.282.436, ptr %value.436, align 8
  %scalar.283.437 = fadd double %scalar.276.430, %scalar.282.436
  store double %scalar.283.437, ptr %value.437, align 8
  %scalar.284.438 = fsub double %scalar.283.437, %scalar.276.430
  store double %scalar.284.438, ptr %value.438, align 8
  %scalar.285.439 = fsub double %scalar.282.436, %scalar.284.438
  store double %scalar.285.439, ptr %value.439, align 8
  %scalar.286.43 = fadd double %scalar.283.437, %scalar.285.439
  store double %scalar.286.43, ptr %out.25, align 8
  %load.287.440.0 = load double, ptr %arg.14, align 8
  %scalar.287.440 = fadd double %load.287.440.0, %scalar.283.437
  store double %scalar.287.440, ptr %value.440, align 8
  %scalar.288.441 = fsub double %scalar.287.440, %load.287.440.0
  store double %scalar.288.441, ptr %value.441, align 8
  %scalar.289.442 = fsub double %scalar.287.440, %scalar.288.441
  store double %scalar.289.442, ptr %value.442, align 8
  %scalar.290.443 = fsub double %load.287.440.0, %scalar.289.442
  store double %scalar.290.443, ptr %value.443, align 8
  %scalar.291.444 = fsub double %scalar.283.437, %scalar.288.441
  store double %scalar.291.444, ptr %value.444, align 8
  %scalar.292.445 = fadd double %scalar.290.443, %scalar.291.444
  store double %scalar.292.445, ptr %value.445, align 8
  %load.293.446.1 = load double, ptr %arg.33, align 8
  %scalar.293.446 = fadd double %scalar.292.445, %load.293.446.1
  store double %scalar.293.446, ptr %value.446, align 8
  %scalar.294.447 = fadd double %scalar.293.446, %scalar.285.439
  store double %scalar.294.447, ptr %value.447, align 8
  %scalar.295.448 = fadd double %scalar.287.440, %scalar.294.447
  store double %scalar.295.448, ptr %value.448, align 8
  %scalar.296.449 = fsub double %scalar.295.448, %scalar.287.440
  store double %scalar.296.449, ptr %value.449, align 8
  %scalar.297.450 = fsub double %scalar.294.447, %scalar.296.449
  store double %scalar.297.450, ptr %value.450, align 8
  %scalar.298.44 = fadd double %scalar.295.448, %scalar.297.450
  store double %scalar.298.44, ptr %out.26, align 8
  %scalar.299.451 = fmul double %load.0.178.1, %scalar.295.448
  store double %scalar.299.451, ptr %value.451, align 8
  %scalar.300.452 = fneg double %scalar.299.451
  store double %scalar.300.452, ptr %value.452, align 8
  %scalar.301.453 = call double @llvm.fma.f64(double %load.0.178.1, double %scalar.295.448, double %scalar.300.452)
  store double %scalar.301.453, ptr %value.453, align 8
  %scalar.302.454 = fmul double %load.0.178.1, %scalar.297.450
  store double %scalar.302.454, ptr %value.454, align 8
  %scalar.303.455 = fadd double %scalar.301.453, %scalar.302.454
  store double %scalar.303.455, ptr %value.455, align 8
  %scalar.304.456 = fmul double %load.3.181.1, %scalar.295.448
  store double %scalar.304.456, ptr %value.456, align 8
  %scalar.305.457 = fadd double %scalar.303.455, %scalar.304.456
  store double %scalar.305.457, ptr %value.457, align 8
  %scalar.306.458 = fadd double %scalar.299.451, %scalar.305.457
  store double %scalar.306.458, ptr %value.458, align 8
  %scalar.307.459 = fsub double %scalar.306.458, %scalar.299.451
  store double %scalar.307.459, ptr %value.459, align 8
  %scalar.308.460 = fsub double %scalar.305.457, %scalar.307.459
  store double %scalar.308.460, ptr %value.460, align 8
  %scalar.309.45 = fadd double %scalar.306.458, %scalar.308.460
  store double %scalar.309.45, ptr %out.27, align 8
  %load.310.461.0 = load double, ptr %arg.15, align 8
  %scalar.310.461 = fadd double %load.310.461.0, %scalar.306.458
  store double %scalar.310.461, ptr %value.461, align 8
  %scalar.311.462 = fsub double %scalar.310.461, %load.310.461.0
  store double %scalar.311.462, ptr %value.462, align 8
  %scalar.312.463 = fsub double %scalar.310.461, %scalar.311.462
  store double %scalar.312.463, ptr %value.463, align 8
  %scalar.313.464 = fsub double %load.310.461.0, %scalar.312.463
  store double %scalar.313.464, ptr %value.464, align 8
  %scalar.314.465 = fsub double %scalar.306.458, %scalar.311.462
  store double %scalar.314.465, ptr %value.465, align 8
  %scalar.315.466 = fadd double %scalar.313.464, %scalar.314.465
  store double %scalar.315.466, ptr %value.466, align 8
  %load.316.467.1 = load double, ptr %arg.34, align 8
  %scalar.316.467 = fadd double %scalar.315.466, %load.316.467.1
  store double %scalar.316.467, ptr %value.467, align 8
  %scalar.317.468 = fadd double %scalar.316.467, %scalar.308.460
  store double %scalar.317.468, ptr %value.468, align 8
  %scalar.318.469 = fadd double %scalar.310.461, %scalar.317.468
  store double %scalar.318.469, ptr %value.469, align 8
  %scalar.319.470 = fsub double %scalar.318.469, %scalar.310.461
  store double %scalar.319.470, ptr %value.470, align 8
  %scalar.320.471 = fsub double %scalar.317.468, %scalar.319.470
  store double %scalar.320.471, ptr %value.471, align 8
  %scalar.321.46 = fadd double %scalar.318.469, %scalar.320.471
  store double %scalar.321.46, ptr %out.28, align 8
  %scalar.322.472 = fmul double %load.0.178.1, %scalar.318.469
  store double %scalar.322.472, ptr %value.472, align 8
  %scalar.323.473 = fneg double %scalar.322.472
  store double %scalar.323.473, ptr %value.473, align 8
  %scalar.324.474 = call double @llvm.fma.f64(double %load.0.178.1, double %scalar.318.469, double %scalar.323.473)
  store double %scalar.324.474, ptr %value.474, align 8
  %scalar.325.475 = fmul double %load.0.178.1, %scalar.320.471
  store double %scalar.325.475, ptr %value.475, align 8
  %scalar.326.476 = fadd double %scalar.324.474, %scalar.325.475
  store double %scalar.326.476, ptr %value.476, align 8
  %scalar.327.477 = fmul double %load.3.181.1, %scalar.318.469
  store double %scalar.327.477, ptr %value.477, align 8
  %scalar.328.478 = fadd double %scalar.326.476, %scalar.327.477
  store double %scalar.328.478, ptr %value.478, align 8
  %scalar.329.479 = fadd double %scalar.322.472, %scalar.328.478
  store double %scalar.329.479, ptr %value.479, align 8
  %scalar.330.480 = fsub double %scalar.329.479, %scalar.322.472
  store double %scalar.330.480, ptr %value.480, align 8
  %scalar.331.481 = fsub double %scalar.328.478, %scalar.330.480
  store double %scalar.331.481, ptr %value.481, align 8
  %scalar.332.47 = fadd double %scalar.329.479, %scalar.331.481
  store double %scalar.332.47, ptr %out.29, align 8
  %load.333.482.0 = load double, ptr %arg.16, align 8
  %scalar.333.482 = fadd double %load.333.482.0, %scalar.329.479
  store double %scalar.333.482, ptr %value.482, align 8
  %scalar.334.483 = fsub double %scalar.333.482, %load.333.482.0
  store double %scalar.334.483, ptr %value.483, align 8
  %scalar.335.484 = fsub double %scalar.333.482, %scalar.334.483
  store double %scalar.335.484, ptr %value.484, align 8
  %scalar.336.485 = fsub double %load.333.482.0, %scalar.335.484
  store double %scalar.336.485, ptr %value.485, align 8
  %scalar.337.486 = fsub double %scalar.329.479, %scalar.334.483
  store double %scalar.337.486, ptr %value.486, align 8
  %scalar.338.487 = fadd double %scalar.336.485, %scalar.337.486
  store double %scalar.338.487, ptr %value.487, align 8
  %load.339.488.1 = load double, ptr %arg.35, align 8
  %scalar.339.488 = fadd double %scalar.338.487, %load.339.488.1
  store double %scalar.339.488, ptr %value.488, align 8
  %scalar.340.489 = fadd double %scalar.339.488, %scalar.331.481
  store double %scalar.340.489, ptr %value.489, align 8
  %scalar.341.490 = fadd double %scalar.333.482, %scalar.340.489
  store double %scalar.341.490, ptr %value.490, align 8
  %scalar.342.491 = fsub double %scalar.341.490, %scalar.333.482
  store double %scalar.342.491, ptr %value.491, align 8
  %scalar.343.492 = fsub double %scalar.340.489, %scalar.342.491
  store double %scalar.343.492, ptr %value.492, align 8
  %scalar.344.48 = fadd double %scalar.341.490, %scalar.343.492
  store double %scalar.344.48, ptr %out.30, align 8
  %scalar.345.493 = fmul double %load.0.178.1, %scalar.341.490
  store double %scalar.345.493, ptr %value.493, align 8
  %scalar.346.494 = fneg double %scalar.345.493
  store double %scalar.346.494, ptr %value.494, align 8
  %scalar.347.495 = call double @llvm.fma.f64(double %load.0.178.1, double %scalar.341.490, double %scalar.346.494)
  store double %scalar.347.495, ptr %value.495, align 8
  %scalar.348.496 = fmul double %load.0.178.1, %scalar.343.492
  store double %scalar.348.496, ptr %value.496, align 8
  %scalar.349.497 = fadd double %scalar.347.495, %scalar.348.496
  store double %scalar.349.497, ptr %value.497, align 8
  %scalar.350.498 = fmul double %load.3.181.1, %scalar.341.490
  store double %scalar.350.498, ptr %value.498, align 8
  %scalar.351.499 = fadd double %scalar.349.497, %scalar.350.498
  store double %scalar.351.499, ptr %value.499, align 8
  %scalar.352.500 = fadd double %scalar.345.493, %scalar.351.499
  store double %scalar.352.500, ptr %value.500, align 8
  %scalar.353.501 = fsub double %scalar.352.500, %scalar.345.493
  store double %scalar.353.501, ptr %value.501, align 8
  %scalar.354.502 = fsub double %scalar.351.499, %scalar.353.501
  store double %scalar.354.502, ptr %value.502, align 8
  %scalar.355.49 = fadd double %scalar.352.500, %scalar.354.502
  store double %scalar.355.49, ptr %out.31, align 8
  %load.356.503.0 = load double, ptr %arg.17, align 8
  %scalar.356.503 = fadd double %load.356.503.0, %scalar.352.500
  store double %scalar.356.503, ptr %value.503, align 8
  %scalar.357.504 = fsub double %scalar.356.503, %load.356.503.0
  store double %scalar.357.504, ptr %value.504, align 8
  %scalar.358.505 = fsub double %scalar.356.503, %scalar.357.504
  store double %scalar.358.505, ptr %value.505, align 8
  %scalar.359.506 = fsub double %load.356.503.0, %scalar.358.505
  store double %scalar.359.506, ptr %value.506, align 8
  %scalar.360.507 = fsub double %scalar.352.500, %scalar.357.504
  store double %scalar.360.507, ptr %value.507, align 8
  %scalar.361.508 = fadd double %scalar.359.506, %scalar.360.507
  store double %scalar.361.508, ptr %value.508, align 8
  %load.362.509.1 = load double, ptr %arg.36, align 8
  %scalar.362.509 = fadd double %scalar.361.508, %load.362.509.1
  store double %scalar.362.509, ptr %value.509, align 8
  %scalar.363.510 = fadd double %scalar.362.509, %scalar.354.502
  store double %scalar.363.510, ptr %value.510, align 8
  %scalar.364.511 = fadd double %scalar.356.503, %scalar.363.510
  store double %scalar.364.511, ptr %value.511, align 8
  %scalar.365.512 = fsub double %scalar.364.511, %scalar.356.503
  store double %scalar.365.512, ptr %value.512, align 8
  %scalar.366.513 = fsub double %scalar.363.510, %scalar.365.512
  store double %scalar.366.513, ptr %value.513, align 8
  %scalar.367.50 = fadd double %scalar.364.511, %scalar.366.513
  store double %scalar.367.50, ptr %out.32, align 8
  %scalar.368.514 = fmul double %load.0.178.1, %scalar.364.511
  store double %scalar.368.514, ptr %value.514, align 8
  %scalar.369.515 = fneg double %scalar.368.514
  store double %scalar.369.515, ptr %value.515, align 8
  %scalar.370.516 = call double @llvm.fma.f64(double %load.0.178.1, double %scalar.364.511, double %scalar.369.515)
  store double %scalar.370.516, ptr %value.516, align 8
  %scalar.371.517 = fmul double %load.0.178.1, %scalar.366.513
  store double %scalar.371.517, ptr %value.517, align 8
  %scalar.372.518 = fadd double %scalar.370.516, %scalar.371.517
  store double %scalar.372.518, ptr %value.518, align 8
  %scalar.373.519 = fmul double %load.3.181.1, %scalar.364.511
  store double %scalar.373.519, ptr %value.519, align 8
  %scalar.374.520 = fadd double %scalar.372.518, %scalar.373.519
  store double %scalar.374.520, ptr %value.520, align 8
  %scalar.375.521 = fadd double %scalar.368.514, %scalar.374.520
  store double %scalar.375.521, ptr %value.521, align 8
  %scalar.376.522 = fsub double %scalar.375.521, %scalar.368.514
  store double %scalar.376.522, ptr %value.522, align 8
  %scalar.377.523 = fsub double %scalar.374.520, %scalar.376.522
  store double %scalar.377.523, ptr %value.523, align 8
  %scalar.378.51 = fadd double %scalar.375.521, %scalar.377.523
  store double %scalar.378.51, ptr %out.33, align 8
  %load.379.524.0 = load double, ptr %arg.18, align 8
  %scalar.379.524 = fadd double %load.379.524.0, %scalar.375.521
  store double %scalar.379.524, ptr %value.524, align 8
  %scalar.380.525 = fsub double %scalar.379.524, %load.379.524.0
  store double %scalar.380.525, ptr %value.525, align 8
  %scalar.381.526 = fsub double %scalar.379.524, %scalar.380.525
  store double %scalar.381.526, ptr %value.526, align 8
  %scalar.382.527 = fsub double %load.379.524.0, %scalar.381.526
  store double %scalar.382.527, ptr %value.527, align 8
  %scalar.383.528 = fsub double %scalar.375.521, %scalar.380.525
  store double %scalar.383.528, ptr %value.528, align 8
  %scalar.384.529 = fadd double %scalar.382.527, %scalar.383.528
  store double %scalar.384.529, ptr %value.529, align 8
  %load.385.530.1 = load double, ptr %arg.37, align 8
  %scalar.385.530 = fadd double %scalar.384.529, %load.385.530.1
  store double %scalar.385.530, ptr %value.530, align 8
  %scalar.386.531 = fadd double %scalar.385.530, %scalar.377.523
  store double %scalar.386.531, ptr %value.531, align 8
  %scalar.387.532 = fadd double %scalar.379.524, %scalar.386.531
  store double %scalar.387.532, ptr %value.532, align 8
  %scalar.388.533 = fsub double %scalar.387.532, %scalar.379.524
  store double %scalar.388.533, ptr %value.533, align 8
  %scalar.389.534 = fsub double %scalar.386.531, %scalar.388.533
  store double %scalar.389.534, ptr %value.534, align 8
  %scalar.390.52 = fadd double %scalar.387.532, %scalar.389.534
  store double %scalar.390.52, ptr %out.0, align 8
  ret void
}

define void @__ssa_sec_core_pack__sec_core(ptr noalias %arg.0, ptr noalias %arg.1, ptr noalias %arg.2, ptr noalias %arg.3, ptr noalias %arg.4, ptr noalias %arg.5, ptr noalias %arg.6, ptr noalias %arg.7, ptr noalias %arg.8, ptr noalias %arg.9, ptr noalias %arg.10, ptr noalias %arg.11, ptr noalias %arg.12, ptr noalias %arg.13, ptr noalias %arg.14, ptr noalias %arg.15, ptr noalias %arg.16, ptr noalias %arg.17, ptr %arg.18, ptr noalias %arg.19, ptr noalias %arg.20, ptr noalias %arg.21, ptr noalias %arg.22, ptr noalias %arg.23, ptr noalias %arg.24, ptr noalias %arg.25, ptr noalias %arg.26, ptr noalias %arg.27, ptr noalias %arg.28, ptr noalias %arg.29, ptr noalias %arg.30, ptr noalias %arg.31, ptr noalias %arg.32, ptr noalias %arg.33, ptr noalias %arg.34, ptr noalias %arg.35, ptr noalias %arg.36, ptr %arg.37, ptr %out.0) {
entry:
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
  %value.56 = alloca i32, i64 1, align 8
  %value.54 = alloca i64, i64 1, align 8
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
  store i32 33, ptr %value.120, align 4
  store i32 32, ptr %value.118, align 4
  store i32 31, ptr %value.116, align 4
  store i32 30, ptr %value.114, align 4
  store i32 29, ptr %value.112, align 4
  store i32 28, ptr %value.110, align 4
  store i32 27, ptr %value.108, align 4
  store i32 26, ptr %value.106, align 4
  store i32 25, ptr %value.104, align 4
  store i32 24, ptr %value.102, align 4
  store i32 23, ptr %value.100, align 4
  store i32 22, ptr %value.98, align 4
  store i32 21, ptr %value.96, align 4
  store i32 20, ptr %value.94, align 4
  store i32 19, ptr %value.92, align 4
  store i32 18, ptr %value.90, align 4
  store i32 17, ptr %value.88, align 4
  store i32 16, ptr %value.86, align 4
  store i32 15, ptr %value.84, align 4
  store i32 14, ptr %value.82, align 4
  store i32 13, ptr %value.80, align 4
  store i32 12, ptr %value.78, align 4
  store i32 11, ptr %value.76, align 4
  store i32 10, ptr %value.74, align 4
  store i32 9, ptr %value.72, align 4
  store i32 8, ptr %value.70, align 4
  store i32 7, ptr %value.68, align 4
  store i32 6, ptr %value.66, align 4
  store i32 5, ptr %value.64, align 4
  store i32 4, ptr %value.62, align 4
  store i32 3, ptr %value.60, align 4
  store i32 2, ptr %value.58, align 4
  store i32 1, ptr %value.56, align 4
  store i64 0, ptr %value.54, align 8
  call void @__ssa_sec_core_pack__sec_core__planned_region_0(ptr %arg.9, ptr %arg.18, ptr %arg.8, ptr %arg.7, ptr %arg.6, ptr %arg.5, ptr %arg.4, ptr %arg.3, ptr %arg.2, ptr %arg.17, ptr %arg.16, ptr %arg.15, ptr %arg.14, ptr %arg.13, ptr %arg.12, ptr %arg.11, ptr %arg.10, ptr %arg.1, ptr %arg.0, ptr %arg.28, ptr %arg.37, ptr %arg.27, ptr %arg.26, ptr %arg.25, ptr %arg.24, ptr %arg.23, ptr %arg.22, ptr %arg.21, ptr %arg.36, ptr %arg.35, ptr %arg.34, ptr %arg.33, ptr %arg.32, ptr %arg.31, ptr %arg.30, ptr %arg.29, ptr %arg.20, ptr %arg.19, ptr %out.0, ptr %value.19, ptr %value.20, ptr %value.21, ptr %value.22, ptr %value.23, ptr %value.24, ptr %value.25, ptr %value.26, ptr %value.27, ptr %value.28, ptr %value.29, ptr %value.30, ptr %value.31, ptr %value.32, ptr %value.33, ptr %value.34, ptr %value.35, ptr %value.36, ptr %value.37, ptr %value.38, ptr %value.39, ptr %value.40, ptr %value.41, ptr %value.42, ptr %value.43, ptr %value.44, ptr %value.45, ptr %value.46, ptr %value.47, ptr %value.48, ptr %value.49, ptr %value.50, ptr %value.51)
  ret void
}

define void @sec_core_pack__sec_core_pack(ptr %buffers, ptr %extents) {
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
  call void @__ssa_sec_core_pack__sec_core_pack(ptr %public.0, ptr %public.1, ptr %public.2, ptr %public.3, ptr %public.4, ptr %public.5, ptr %public.6, ptr %public.7, ptr %public.8, ptr %public.9, ptr %public.10, ptr %public.11, ptr %public.12, ptr %public.13, ptr %public.14, ptr %public.15, ptr %public.16, ptr %public.17, ptr %public.18, ptr %public.19, ptr %public.20, ptr %public.21, ptr %public.22, ptr %public.23, ptr %public.24, ptr %public.25, ptr %public.26, ptr %public.27, ptr %public.28, ptr %public.29, ptr %public.30, ptr %public.31, ptr %public.32, ptr %public.33, ptr %public.34, ptr %public.35, ptr %public.36, ptr %public.37, ptr %public.38, ptr %public.2)
  ret void
}
