2026-10-24 DIFFERENTIAL FUZZ RUNS ON WINDOWS (MEASURED, cloud): the fuzzer's
isolation child used the `fork` context and a nested closure target, so on
Windows every program would stop with "cannot find context for 'fork'"
before compiling. `isolated()` now spawns a module-level `_isolated_entry` on
Windows and keeps fork elsewhere (the typed check's Windows parity rule).
`differential_fuzz_rerun.py` is main-guarded so a spawned child does not
re-run it. Measured on Linux: seed 2, 3 programs, 3 PASS through fork; the
spawn branch (forced) returns PASS for the same program. Tool-only change,
no compiler pass touched. `javascript_differential_fuzz.py` still uses fork
and stays Unix-only.
