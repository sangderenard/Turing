2026-10-24 LIFTED MACHINE CODE EXECUTES (MEASURED, cloud): the SSA reference
evaluator runs functions lifted from x86 (metadata machine_state_model) with
fixed-width integer meanings read from each value's intN dtype (And/Or/Xor
bitwise, Trunc to the result width, Shr logical, AShr arithmetic; the
Python frontend's logical And and float Trunc do not apply), loads and
stores over the machine semantics' PagedByteMemory, and lifted calls into
other lifted functions. Any other op in lifted code is refused by name.
map_elf_image (amd64_machine_semantics) maps an ELF64 image's program
sections and fills each dynamic relocation slot as ld.so would when the
value is a fact of the image; other slots get unmapped EXTERNAL_TARGET_BASE
addresses. Loader-only tables stay unmapped, so a null-page read faults.
Two lifter changes. (1) Formals are int64 (an int32 formal dropped the high
half of RDI). (2) Calls: a Call passed only the four Windows x64 argument
registers and projected only Windows' caller-saved set, even for ELF code,
so RSI/RDI and the XMM registers were assumed to survive a System V call.
A Call now carries the complete register state with a state_layout and the
image's calling convention (lift_elf_export_to_ssa passes sysv-amd64), and
Ret returns [RAX, memory]. Measured: _PyTime_AsMicroseconds,
_PyUnicode_ToLowercase/IsAlpha/ToDecimalDigit and PyCode_Addr2Line (with its
lifted callee) from libpython3.13.so match the native exports.
Hazard: the evaluator reads lifted Shr as logical; Lane E is moving logical
shifts to a distinct LShr, after which the evaluator's machine set must
switch to it. Gate at a40358bc vs main 1f0b21ac: failure names identical in
all four parts; machine/Wasm suites + evaluator tests: no new failures
(691 vs 671 passed, the difference being the new tests).
