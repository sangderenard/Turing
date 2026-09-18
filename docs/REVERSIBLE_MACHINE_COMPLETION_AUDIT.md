# Reversible binary-machine completion audit

This audit distinguishes verified end-to-end behavior from architecture that
is merely prepared or intentionally fail-closed. Passing one real program is
strong integration evidence, but it is not evidence that arbitrary Windows
AMD64 binaries are supported.

## Verified end-to-end

| Requirement | Current evidence |
|---|---|
| Real binary ingestion | The existing PE decompiler raises the system `cmd.exe` image into the token multigraph and constructs executable AMD64 state. |
| Relocatable PE loading | Bounded PE relocation records move mapped bytes and decoded address identities together. A real `cmd.exe` moved by 256 MiB applied 372 relocations and completed `/c "echo hello"` with exit code zero. |
| Capability-supplied DLL linking | Approved dependency bytes are decompiled, mapped, recursively resolved through exports/forwarders, and executed in one guest address space. Module bytes, bases, and per-IAT witnesses survive JSONL and segmented replay. |
| Actual execution | `cmd.exe /c "echo hello"` halts with exit code zero; the interactive image accepts repeated terminal commands and returns to `ReadConsoleW`. |
| Reversal and replay | Every instruction, external completion, shell input, shared-memory synchronization, and child deployment is an exact tape edge. JSONL and segmented cold reversal are tested. |
| Bounded storage | Main execution keeps 4,096 states resident. Possible-world exact states and trace SSA use separate content-addressed parent/suffix DAGs with bounded chunk caches; dormant exact heads can seal and evict their mutable cursor/state, then append again from a cold tip. The latest compiled real `cmd.exe` run wrote 16,510 records directly into 66 immutable segments (8,570,148 bytes); cold reopen decoded only its final segment. |
| Executable-code coherence | Guest or admitted capability writes to executable pages increment reversible page versions, invalidate translated blocks, and force decoding from current guest memory. Rewind restores both code and version state. |
| Threads versus forks | Virtual cores use an explicit sequentially-consistent shared-memory schedule. Possible-world heads retain isolated memory and exact/SSA suffixes. |
| External capture | Imports carry stable symbolic identities. The deterministic Windows port has no generic host-call fallback. Registered `CreateProcessW` targets create linked child tapes. Filesystem and registry operations use separate immutable capability-owned namespaces whose handles, contents, metadata, and effects survive exact replay. |
| Observation | Complete `TMSNAP01` flips contain all 54 contiguous register cells and terminal/framebuffer outputs. Native OpenGL and HTML/WebGL displays consume the same ABI. |
| Live HTML operation | A loopback-only host owns execution, streams immutable flips, and accepts bounded terminal input without exposing the machine object to request threads. |
| Fail-closed behavior | Missing code, semantics, external capabilities, corrupt segments, invalid dependencies, and unsafe web mounts stop with structured diagnostics. |

The current broad machine/tape/SSA/recompilation/VFS/registry/virtual-memory and
shell regression slice passes 247 tests; the focused pipe/nonlocal-control
slice adds exact lifecycle, segmented-replay, child-routing, and SSA-domain
coverage. A copied
69,949-record real interactive tape continued for 7,756 events, produced the
requested terminal output, retained exactly 4,096 hot states, and reopened at
77,705 dependency-validated records.

The fresh segmented `/c "echo hello"` proof restored absolute executor position
16,509 and output `hello\r\n` from disk. Reversing once hydrated only positions
16,384 through 16,509, returned to the non-halted 16,009-step predecessor, and
one ordinary forward step reproduced the persisted halted state exactly.
The registry-era proof deliberately stopped at a previously unhandled
`MAXIMUM_ALLOWED` open, retained that amber frontier annotation, resumed to the
same halt, and cold-reversed from 16,010 to 16,009 steps. One forward step
reproduced the complete halted state, six-root empty virtual registry, and
`hello\r\n` device output exactly.
The following virtual-memory proof again halted at guest step 16,010 with 499
capability completions and 15,459 authenticated Wasm-journalled instructions.
Cold segmented reopen restored exit code zero and `hello\r\n` exactly without
servicing another capability request.

## Not yet complete

1. **General Windows import coverage.** The current system `cmd.exe` catalog
   contains 286 symbolic imports; the deterministic port currently exposes 205
   handler identities, matching 153 of those imports and leaving 133
   catalogued but unsupported. The latest bounded
   families add all seven mutex/semaphore/single-object-wait imports and all
   fifteen file-L1 imports, followed by all seven remaining registry imports.
   The virtual-memory tier adds all four memory-L1 imports (`VirtualAlloc`,
   `VirtualFree`, `VirtualQuery`, and bounded current-process
   `ReadProcessMemory`).
   The pipe/control tier adds the six imported pipe/descriptor/duplication
   identities plus `longjmp`; the same handlers also expose `CreatePipe` for
   binaries that import it directly.
   Most remaining imports were not reached by the proven
   command path. Another program can legitimately stop on them.
2. **General AMD64 coverage.** The instruction families exercised by `cmd.exe`
   are implemented, and absent semantic handlers annotate the exact RIP, but
   this is not a complete AMD64 interpreter or proof over all decoded operand
   forms.
3. **Full PE dynamic linking.** Preferred and relocated image loading plus
   import identities work. Export address/name/ordinal tables and symbolic
   forwarders are parsed with strict bounds, and explicitly capability-supplied
   dependency DLLs are mapped and dispatched automatically. A bounded explicit
   provider can recursively supply missing modules and delay-import IATs are
   deterministically lowered, but native first-use delay-helper side effects,
   dynamic-load notification suppression/TLS reclamation, process detach, and
   full unwind/exception semantics remain partial. The x64 MSVCRT
   `_local_unwind(frame, target)` route now reads the guest PE exception
   directory and C scope table, dispatches one bounded `__finally` callback
   with the Windows ABI arguments, and refuses chained, malformed, non-C, or
   multi-callback shapes. Clean auxiliary
   thread return executes TLS and DllMain `DLL_THREAD_DETACH` notifications
   before signalling its handle.
4. **Native recompilation.** Cached basic blocks pre-bind Python semantic
   operations and measured a modest synthetic speedup. A first register-only
   tier emits and executes real WebAssembly for register MOV and core integer
   arithmetic/logical operations across all AMD64 scalar widths, returning
   authenticated complete state checkpoints per instruction to the reversible
   journal. Bounded static RIP-relative/displacement MOV memory accesses carry
   validated read/write witnesses. Exact-state-specialized direct calls and
   ordinary returns carry both stack-memory and shadow-call-stack witnesses;
   loader/termination returns remain in the interpreter. A bounded persistent
   Node host now caches instantiated modules, and the ordinary shell-ticked,
   free-spin, and live HTML-controller runner can automatically execute safe
   prefixes under the explicit `node-wasm` policy before falling back at the
   exact unsupported instruction. Exact block-entry specialization now covers
   dynamic base/index/scale/displacement MOV loads and stores, including FS/GS
   bases, and internal register-resolved indirect calls/jumps. Specialization
   inputs are guarded and cache-keyed; memory-resolved or external indirect
   targets stay in the interpreter. Direct and flag-conditional relative exits
   are proven. Runtime LEA consumes register values produced earlier in the
   same block; register/immediate CMP and TEST compute flags without modifying
   operands. Exact-entry register/immediate PUSH and register POP carry stack
   memory witnesses without changing the shadow call stack. Static CMP/TEST
   memory reads work anywhere in a safe prefix; dynamic forms specialize at
   entry. Both retain signed-immediate behavior and authenticated read effects.
   Guest-window planning is greedy: a distant later access seals the maximal
   bounded prefix instead of poisoning earlier instructions. Memory-destination
   scalar arithmetic journals one exact read-modify-write before/after effect,
   and MOVSXD handles register or witnessed memory sources. Immediate SHL,
   NEG/NOT, SETcc, INC/DEC, register exchange, and the implicitly atomic
   memory/register XCHG now use the same per-instruction checkpoint ABI; memory
   forms retain one authenticated before/after effect and INC/DEC preserve CF.
   Register or witnessed-memory CMOV, carry-consuming SBB, arithmetic-right
   shift, rotate-left, and full-width unsigned MUL are also admitted. MUL
   computes the complete 128-bit product into RDX:RAX using exact 32-bit limbs.
   CDQE/CQO and register or witnessed-memory BT/BTR/BTS/BTC are admitted too.
   Register-indexed memory bit tests specialize the signed adjacent-bit-string
   address adjustment as well as the modulo bit index; only CF changes, and
   modifying forms retain one exact read-modify-write witness. Operand-width
   parsing distinguishes memory width from a trailing immediate width such as
   `BTR_RM32_IMM8`.
   Exact-entry `REP STOSW` uses a bounded Wasm loop and an authenticated fill
   descriptor. The descriptor records destination, word count, value,
   direction, and final registers; immutable parent pages retain the prior
   bytes for reversal, so the journal does not duplicate a 16 KiB before/after
   payload or split one guest instruction into thousands of tape edges.
   Checkpoint ABI v2 also carries all sixteen 128-bit XMM registers as 32
   contiguous qwords. XMM XOR and aligned/unaligned vector loads/stores use
   exact 128-bit memory witnesses with independent low/high halves. The native
   and HTML snapshot ABI observes those compiled XMM changes through its
   existing 54-cell register bank. A page-aligned dynamic guest-buffer offset
   prevents widened journals from overlapping the Wasm memory mirror.
   Native code and GPU
   execution kernels are still absent. Self-modifying code is coherent at the interpreter/cache boundary,
   but only instruction forms understood by the reference decoder are admitted.
   A live `cmd.exe` automatic-dispatch probe reached its first pending external
   capability after 432 transitions; after dynamic-address and internal-indirect
   specialization plus the current scalar lowering, 326 instructions executed
   through 183 Wasm dispatches, and every unsupported prefix resumed at its
   exact interpreter RIP.
   An earlier deterministic `/c "echo hello"` prefix compiled 2,669 of 3,043
   transitions (about 88%) before reaching `msvcrt!_local_unwind`; every
   remaining denial in that prefix was an indirect call or jump and the
   semantic-denial inventory was empty. After adding the bounded guest-metadata
   unwind path and authenticating runtime-discovered instructions during Wasm
   journal commit, the same real command now halts with exit code zero after
   16,010 guest steps and 499 deterministic capability completions. It commits
   15,459 instructions through authenticated Wasm journals. Its denial
   inventory is now entirely intentional external/internal indirect control
   boundaries and loader/termination sentinel returns; no ordinary semantic
   denial remains on the proven route.
5. **Guest-created thread lifecycle.** `CreateThread` activates a parked core
   within fixed snapshot capacity, allocates a private stack/TEB/TLS clone,
   runs thread-attach callbacks, shares process memory deterministically, and
   parks with an exit code on return. Thread-handle polling/blocking waits,
   exit-code reads, close, and reversible thread-detach callbacks are
   implemented. Suspended creation, forced termination, additional
   synchronization objects beyond mutexes and semaphores, loader-lock
   contention, and capacity growth remain incomplete. Named and unnamed mutexes
   and semaphores now use shared reversible system-state keys; recursive mutex
   ownership, count consumption/release, previous-count writes, zero-time
   polling, and scheduler-yielding waits are exact tape effects. Unsupported
   security descriptors and invalid creation shapes remain fail-closed.
6. **Arbitrary host programs.** Host `exec`, ambient DLL loading, and ambient
   filesystem access remain deliberately prohibited. Broad compatibility must
   be gained through explicit virtual implementations, mapped guest modules, or
   registered bundle executors—not an unsafe fallback.

## Highest-value next work

1. Add suspended creation, forced-exit/process-detach behavior, events and
   wait sets, and native first-use delay-helper behavior.
2. Turn unsupported-import inventory into coherent capability families,
   prioritizing process lifecycle, console-screen operations, and the
   remaining exception/unwind APIs.
3. Extend specialized dynamic memory and indirect control beyond block-entry
   register-resolved cases, broaden calls/returns beyond block-entry boundaries,
   make compiled dispatch profitable for real `cmd.exe` workloads, then add an
   embedded native backend with the same
   state-and-journal ABI.

The active goal is therefore materially advanced but not complete. Completion
requires proving these broader compatibility and recompilation requirements,
not merely preserving the already successful `cmd.exe` route.
