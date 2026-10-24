2026-10-24 MOBILE REFERENCE PRODUCTS PROVED BIT-EXACT (MEASURED, cloud Linux):
first item of the low-priority extra frontends/backends lane, on main c5af44c
(Albert's 31e0fdf7 and 8fb85e7). The built PWA (`build_mobile.py --web`) run in
node through its own `turing-wasm-shell.js` binding, and the android-arm64-v8a
and ios-arm64 LLVM modules run natively after swapping only their target
triple for the host's, return `program.py`'s exact bits on 300 and 500 rows.
Each ARM64 module differs from the windows-amd64 module only in the triple
line and SSA value numbering. The iOS workflow's unsigned Xcode link run
40093986493 on 8fb85e7 succeeded. `tests/test_mobile_reference_oracle.py`
(3 tests) holds this; the older Wasm test uses `pytest.approx` and is left as
it is. Not proved here: the Android APK and the shells' Vulkan/Metal drawing
(Windows NDK paths, no ARM64 device in the cloud).
NOT GREEN ON MAIN: gate of main c5af44c vs its parent a6fdb31 adds 10 names,
all "invalid ELF header": 31e0fdf7 defaults every lowering to the
windows-amd64 build target, so Linux LLVM products are PE files. Part aa: nine
test_loop_carried_producers native_llvm cases; part ab:
test_project_unit_recognizes_exact_lowered_closure_formals. Lane E owns the
fix. This branch adds a test file outside the gate parts and changes no code.
