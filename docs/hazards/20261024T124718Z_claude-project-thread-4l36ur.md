2026-10-24 X86 BATCH 4 (MEASURED, cloud): BTS_RM64_IMM8 and BTR_RM64_IMM8
(REX.W 0f ba /5 /6), TEST_RM64_IMM32 (REX.W f7 /0, sign-extended imm32) and
CMP_RM16_IMM16 (66 81 /7), from the Py_CompileString closure of the Linux
libpython3.13.so. Each has an equivalence row, a family-lifting case and a
native CPU oracle case (tests/test_x86_native_oracle.py, which runs the bytes
on the host CPU). Silent wrong answer fixed: the CFG lifter split an
immediate bit index on a memory operand into a word offset plus a bit
(`bts dword [m], 35` touched bit 3 of [m+4]); the CPU takes an immediate
index modulo the operand width and never moves the word. A register index
on memory, which does address across words, stays refused by name.
Hazard: the gate parts do not run the machine/Wasm suites; this change was
also checked with tests/test_machine_*, test_amd64*, test_binary*, test_elf*,
test_host_code* and the oracle. Gate at 0cbd5884 vs main 8fb85e7b (llvmlite
installed): failure names identical in all four parts; suites: no new
failures (642 vs 630 passed, the difference being the new cases).
