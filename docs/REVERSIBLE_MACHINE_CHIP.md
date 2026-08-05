# Reversible machine chip layout

The virtual multicore uses a deliberately physical observation ABI. It is not
a claim about the host CPU's silicon; it is the memory map seen by the compiled
executor and its WebGPU display.

## Register banks

Each core owns one fixed 256-byte-aligned bank. The current AMD64 bank occupies
512 bytes and contains fifty-four contiguous 64-bit cells:

```text
RAX ... R15 RIP RFLAGS FS_BASE GS_BASE XMM0_LO XMM0_HI ... XMM15_LO XMM15_HI STEPS CALL_DEPTH
```

Each cell is exactly eight contiguous bytes, exposed to WebGPU as adjacent
little-endian low/high `u32` words. Core `n` begins at
`register_base + n * 512`. The remaining 80 bytes are bank padding reserved for
future architectural registers; no following core can overlap them.

## Reversible AMD64 state

The default machine program maps PE headers and sections at their preferred
image base, or at an explicitly selected runtime base after applying bounded
PE `DIR64`/`HIGHLOW` relocation records. Loader base, signed load bias, and
relocation count are exact system-state cells, so tapes restore the same address
namespace automatically and reject a conflicting resume request. Decoded
instruction addresses and relative branch targets move with the image; current
guest bytes remain authoritative for relocated absolute operands. Images with
no usable relocation directory fail closed when asked to move.

The loader then creates a zeroed 1 MiB stack and installs minimal PEB/TEB pages.
The stack begins with Windows x64 entry alignment and a sentinel return address.
General registers obey 8/16-bit preservation and 32-bit upper-half clearing;
memory is sparse, byte-addressed, little-endian, page-backed, and copy-on-write.
Unmapped access becomes a structured machine trap. FS and GS overrides add the
visible segment bases before memory access; in particular, byte `0x65` selects
the TEB-backed GS base.

Integer move, address, add/subtract, compare/test, logical, shift/rotate,
extend, stack, conditional, and atomic compare/exchange families currently
produce immutable successor states. Calls update both the architectural stack
and a reversible validation stack. Returning backward restores both without
executing a guessed inverse operation.

## Guest external calls

The PE loader catalogues import descriptors and every IAT slot by DLL and
symbol. It replaces file-time thunk RVAs with stable synthetic guest-external
addresses. An indirect call to one of those addresses pushes its real return
address and then pauses with `WAITING_EXTERNAL`; the request records the exact
reference, Windows x64 register arguments, stack pointer, and return address.

Dependency export directories are also represented structurally: module name,
ordinal base, address/name/ordinal tables, named and ordinal-only exports,
aliases, and forwarder strings are bounds-checked before entering the link
graph. Forwarder RVAs remain symbolic link edges and are never mistaken for
executable addresses. A real `kernel32.dll` parse catalogued 1,636 exports,
including 187 forwarders, without consulting the host loader.

The shell may explicitly supply dependency DLL byte images to `load_pe` through
`dependency_modules`, with optional deterministic `dependency_load_addresses`.
The linker decompiles each approved image, rejects address overlap and identity
ambiguity, resolves concrete exports and bounded forwarder chains, writes every
IAT slot, and exposes all mapped executable sections to the same reversible
dispatcher. A forwarder whose leaf module is not supplied remains a normal
capability request with its complete chain retained in the binding witness.

Alternatively, `dependency_provider` is an explicit bounded acquisition
capability. It is called only with DLL identities that the current import or
forwarder graph actually requires; returned bytes are parsed and replanned
until the graph converges, while refused identities remain external capability
ports. Module-count and aggregate-byte limits fail closed. A two-stage test
acquires `demo.dll`, follows its `KERNEL32.Sleep` forwarder, acquires an approved
`KERNEL32.dll`, and executes the final guest export.

PE delay-import descriptors are parsed separately from eager imports and retain
a delay flag in each binding witness. The current deterministic lowering writes
both the delayed IAT target and its module-handle cell before execution, which
supports direct delayed calls and exact replay. It does not yet reproduce every
observable side effect of the native first-use delay-load helper.

PE TLS directories now initialize the first virtual thread. The loader validates
the template, zero-fill bound, index cell, callback pointer table, and executable
callback RVAs. It allocates a TLS vector and module blocks from the reversible
system arena, writes the vector to `GS:[0x58]`, publishes each DWORD TLS index,
and schedules dependency callbacks before the subject callback and entry point.
Callbacks receive the Windows process-attach register tuple and return through
a reserved loader sentinel. Callback index, targets, module bases, TLS blocks,
and completion are ordinary exact state, so rewind and tape restart reproduce
the loader frontier. A synthetic callback sets `RAX=7`; three transitions run
the callback body, callback return, and subject return, and three reverse steps
recover the byte-for-byte initial state.

Mapped dependency DLL entry points share the startup queue. For each dependency,
its TLS callbacks run before `DllMain(module, DLL_PROCESS_ATTACH, NULL)`; only
after all approved modules succeed does control enter the subject. The queue
records call kind, module, target, index, and whether a nonzero return is
mandatory. A false `DllMain` result traps at its return instruction and retains
the failing module/index in exact state rather than entering the subject with a
partially initialized graph.

JSONL tape headers retain approved module bytes, bases, digests, and all IAT
bindings. Segmented tapes store each distinct module as
`modules/<sha256>.bin` and keep only its identity/address/digest in the
manifest. Resume reconstructs the link plan from those bytes and refuses to run
if any binding differs. A synthetic main image calls `demo.dll!Real` through a
real RIP-relative IAT instruction; the dependency executes AMD64, returns
`RAX=42`, and the same four-instruction route passes after JSONL and segmented
cold restart.

A completion can update `RAX` and apply bounded writes only to already mapped
guest memory. The completion itself is a journal edge, so reversing across it
restores the pending request and all pre-call state. The supplied deterministic
Windows bootstrap port is an exact capability allowlist, not Win32 passthrough.
Unknown functions remain pending for a shell policy to approve, capture, or
emulate.

`GetModuleHandleW(NULL)` reads the taped runtime loader base rather than a
host-side constant. A real relocated `cmd.exe /c "echo hello"` validation moved
the image from `0x140000000` to `0x150000000`, applied 372 relocations, executed
16,010 instructions, serviced 499 deterministic capability calls, emitted
`hello\r\n`, and halted with exit code zero.

## Program cache

Every decoded function receives a stable cache-line-aligned block. Its record
contains the subject virtual address, virtual-chip byte offset, capacity, and
occupied byte count. The shader colors one block per allocation by occupancy,
so gaps and dense code are visually distinct without inspecting source text.

## Shell-regulated reversible clock

The execution history versions PC, registers, flags, memory, and call stack.
Backward movement restores a prior version; forking preserves alternate
futures. The shell owns the clock: an external tick supplies elapsed time and
the mutable transitions-per-second setting converts it to an exact execution
budget. One tick publishes one completed observation generation. Presentation
may skip generations and never determines whether the machine is allowed to
advance. The same runner also has an optional maximum-speed worker mode.

## Triple-buffer observation ABI

`turing.machine-snapshot.v1` uses three preallocated slots. A single runner
writes a back slot, including every core's contiguous register bank and all
subject-output descriptors, then publishes `(generation, slot)` in one atomic
store. A single display reader leases the newest completed slot. The runner
never writes the published or leased slot, so neither side needs a shared data
lock and the shader cannot observe a partially written register file.

Every slot begins with the fixed `TMSNAP01` header. It identifies the complete
byte size, generation, direction, transition count, core/register dimensions,
register-bank offset and stride, per-core status offset, and subject-output
descriptor/payload regions. Output descriptors currently distinguish raw
bytes, UTF-8 terminals, RGBA8 framebuffers, and F32 audio. Reversible history
is not stored in these slots: observations may be discarded while the machine
journal retains the timeline.

## Resumable system tape

Immediate reverse stepping uses the in-memory graph. In parallel, the machine
maintains an append-only `turing-machine-system-tape` containing the original
subject binary and every forward/backward core transition. It records scalar
architecture state, pending external requests, virtual system/environment state, changed
copy-on-write memory pages, and virtual-filesystem contents. External
completions are explicit tape events. Periodic complete checkpoints bound
reconstruction cost; records between them contain page/VFS deltas. A JSONL
tape can reconstruct the latest state or a selected historical sequence after
the shell process exits.

The tape is also a validated dependency DAG. Every state record names its
previous state node on the same virtual core. Typed edges identify the request
consumed by an external completion, a request abandoned by an explicit branch,
and the historical state used as a branch source. Runtime-discovered code emits
a `runtime_dispatch` state node containing its validated PE targets and decoded
addresses. Resume follows parent lineage from the nearest checkpoint, rejects
missing, forward, or cross-core dependencies, restores the symbolic external
reference catalog, and reinstalls taped dispatch plans before execution.

### Bounded segments and branch heads

Long runs do not require their full state history to remain in memory.
`SegmentedMachineTapeStore` streams the chronological machine tape into
immutable gzip objects addressed by the SHA-256 digest of their canonical
payload. Its manifest retains checkpoint and per-core sequence indexes; cold
resume decodes the nearest checkpoint ancestry and caches only one segment.
The original subject binary is stored separately and verified by digest.

Trace SSA has a second, deliberately separate segment store. Exact machine
segments remain the replay authority, while `SegmentedMachineTraceSSAStore`
stores analysis operations in content-addressed chunks. A possible-world head
names its parent head and fork sequence, so it shares the parent's prefix and
stores only operations after the fork. Sibling paths can be streamed or
discarded from cache independently, and identical suffix chunks deduplicate.
Every SSA operation retains its source tape sequence, instruction bytes and
semantic identity; a reduced slice therefore remains auditable against exact
machine replay.

This is a distinction between persistence and residency: old paths remain
available on disk even though neither their machine states nor SSA operations
need to remain resident. Guest threads remain execution entities inside one
world; forked read heads are alternate worlds with private suffixes.

Possible-world execution now has its own exact-state DAG as well.
`SegmentedMachinePathStateStore` gives every head a full immutable anchor and a
parent/fork-position reference, then writes only states produced after that
fork. Parallel head workers append through a synchronized bounded tail; chunks
are independently decodable, gzip-compressed, SHA-256 addressed, and shared by
digest when sibling suffixes are identical. Reopening a head streams its parent
prefix and local suffix with one decoded chunk resident. Corrupt chunks fail
their digest check before state decoding. A dormant head can be explicitly
released: its partial tail is sealed, its append cursor and final full state
are removed from RAM, and only the small DAG descriptor remains. Appending to
that head later cold-loads its tip and begins a new independently decodable
segment. All heads and the one-object read cache can be released together, so
resident memory follows the chosen active frontier rather than the explored
tree's total size.

In other words, the branching tape is a segmented persistent DAG, not one
ever-growing in-memory list. A running head keeps its current checkpoint,
bounded mutable tail, and small dependency index hot. Sealed machine and SSA
suffixes remain independently addressable on disk and are paged back only for
the lineage being rewound, replayed, reduced, or extended.

Each retained forward state includes execution status, instruction address,
instruction bytes, decoded token, and semantic identity. A head can therefore
be cold-reversed directly from its exact suffix or lifted back into trace SSA
without consulting a guessed instruction stream. The direct
`segment_path_state_head_to_trace_ssa` route streams exact states into bounded
SSA chunks and updates final resource versions as the generator is exhausted;
neither the exact path nor its SSA projection is materialized wholesale.

Guest threads now share memory through an explicit core-index schedule. At one
virtual barrier, core 0 executes first, core 1 observes core 0's writes, and so
on; after the last turn every thread receives the same final immutable memory
object. Changed copy-on-write pages are reduced to compact byte ranges, and
overlapping ranges retain earlier/later core identities as race provenance.
The barrier commit records its cycle, core order, positions, range lengths, and
before/after digests on the exact tape. Reversing the barrier restores every
thread's pre-cycle state and removes the hot schedule record. Capability memory
writes occurring outside a barrier are journal-broadcast to sibling threads as
`shared_memory_sync` edges before another cycle may run. Possible-world read
heads do not use this merger: their memories remain isolated by definition.

`CreateThread` is admitted when the compiled shell reserves more than one core
slot. Idle slots are parked reversible heads, keeping register-bank and
snapshot/shader offsets fixed. Creation allocates a private stack and TEB from
the bounded system arena, clones the TLS vector/templates, runs TLS/DLL
thread-attach calls, and enters the requested routine with its parameter in
`RCX`. Memory remains shared under the barrier schedule. Returning from the
outer thread routine records its exit code and parks only that core. The spawn
tape node depends explicitly on its external completion; reversal restores the
parked slot and pending request. Thread handles support zero-time polling,
blocking `WaitForSingleObject`, `GetExitCodeThread`, and `CloseHandle`. A
blocking parent is reversibly parked while other cores advance, then its exact
external call completes when the child exit record becomes visible. A clean
thread return first executes TLS callbacks and dependency `DllMain` entries in
the exiting guest context with `DLL_THREAD_DETACH`; module groups unwind in
reverse initialization order while each PE TLS callback array retains its
declared order. The return code is preserved across cleanup calls, and the
thread handle becomes signalled only after the final callback returns. Security
attributes and suspended creation remain fail-closed or unsupported.

Mutex and semaphore handles use the same shared reversible system-state plane.
Named objects are case-insensitive mappings into that plane, so all guest
threads see one owner/count while alternate possible-world heads retain their
own suffix state. Recursive mutex acquisition and release, semaphore wait and
bounded release, previous-count memory writes, `WaitForSingleObject`, and
`WaitForSingleObjectEx` are journalled as ordinary exact completion effects.
An unavailable zero-time wait returns `WAIT_TIMEOUT`; a nonzero wait yields to
the virtual scheduler so another core can append the releasing edge. Dormant
branches retain only their content-addressed segment IDs and live tip metadata,
not an in-memory synchronization history.

Process-wide virtual filesystem and registry states cross that same journalled
core synchronization edge. Registry keys, typed binary values, handle access,
enumeration order, and last-write generations are immutable machine state—not
host observations. A completion on one guest thread becomes visible to its
siblings; reversing the synchronization edge restores each prior view. Forked
possible-world heads remain isolated and retain only their own segmented
suffix. Trace SSA records registry changes under a distinct `registry` effect
domain while exact tape state remains the replay authority.

The active executor now applies the same distinction to its immediate reverse
journal. Each core retains a configurable hot window (4,096 states by default).
Advancing at its tip evicts the oldest resident state only after the exact tape
observer has recorded it. Reversing across the window boundary decodes a
contiguous suffix from the dependency-validated JSONL or segmented tape,
discards the now-cold resident future, and continues backward. The logical
history position and tip remain absolute while the physical state list stays
bounded. A focused segmented test advances twenty transitions with four hot
states, reverses all twenty, and reconstructs the initial registers from the
reopened content-addressed tape store.

### Translated basic-block cache

Single-core free-spin execution now consumes immutable translated basic blocks
instead of redispatching every decoded instruction through the generic semantic
switch. Each block pre-binds ordinary effect handlers, stops at every control,
trap, or external-call boundary, and carries a SHA-256 fingerprint over guest
addresses, semantic identities, lengths, and instruction bytes. Runtime dispatch
decoding increments the translation generation and invalidates every cached
block before newly installed code can execute.

Executable guest pages also carry reversible version cells in machine system
state. Any guest semantic or admitted external completion that changes such a
page increments its version and clears the translated-block cache. The current
bytes are decoded at the next RIP; an unsupported mutation stops as
`BLOCKED_CONTROL` instead of falling through to stale host-side instructions.
Because bytes and version cells are ordinary exact state, rewinding restores
the earlier code/cache epoch and trace SSA retains the provenance of the write.

Translation does not coarsen reversal: block execution returns every
architectural successor, and the reversible executor commits one state, edge,
tape event, and observer callback per guest instruction. Multicore execution
continues to use its one-instruction barrier path, so cached blocks cannot make
one virtual core silently outrun another. A synthetic 200,000-instruction
straight-line-loop measurement showed roughly 1.10x throughput while retaining
all 200,000 reversible edges; heavier native handlers will have different
ratios.

The first backend recompilation tier now emits an actual WebAssembly module for
the safe register-only prefix of a translated block: `MOV`, `NOP`, and
`ADD`/`SUB`/`AND`/`OR`/`XOR` with register destinations and register or
immediate sources. Arithmetic covers 8/16/32/64-bit writes and computes the
reference `CF/PF/AF/ZF/SF/OF` word inside Wasm, including partial-register
preservation and 32-bit zero extension. Its linear-memory ABI is the contiguous 16-register bank
followed by RIP, flags, and step count. After every guest instruction the Wasm
module writes a complete state checkpoint together with guest address, semantic
ID, and instruction-byte digest. `commit_recompiled_journal` authenticates
those witnesses and appends the ordinary per-instruction reversible edges; a
test runs the emitted binary in Node, reconstructs the state, commits it, and
reverses exactly. The artifact reports its first unsupported instruction and
continuation RIP, while strict lowering fails closed instead of concealing an
interpreter fallback. Differential execution checks boundary arithmetic and
logical cases at every width against the interpreter, and a mixed three-op
block commits then reverses three distinct guest edges. Static displacement and
RIP-relative `MOV` loads/stores also execute against a bounded 64 KiB guest
mirror. Each memory instruction journals effect kind, address, width, observed
input, and output; replay verifies the input before changing paged memory.
Dynamic effective addresses, oversized mirrors, and compiled writes to any
translated executable page fail closed. Direct relative jumps and all canonical
AMD64 conditional predicates are now executable too: each control witness
declares its statically permitted successor set, the Wasm checkpoint records
the selected RIP, and journal reconstruction rejects any other target. Taken
and not-taken predicate cases match the interpreter before committing and
reversing. A direct CALL or ordinary RET at a compiled block entry can now be
specialized against an exact source state. Wasm updates RSP/RIP and the bounded
stack-memory mirror; the journal independently authenticates the memory effect,
shadow-stack push/pop value, and prior depth. Loader, termination, and outermost
returns deliberately stay in the interpreter lifecycle tier. At a block entry,
dynamic MOV addressing can be specialized across base/index/scale/displacement
and architectural FS/GS bases. The artifact guards every address-producing
register and segment base, cache identity contains the same inputs, and the
ordinary memory witness authenticates the selected location and value.
Register-resolved internal indirect CALL/JMP targets use the same scheme;
external targets remain capability requests and memory-resolved indirect
targets remain in the interpreter because they require another read witness.
Calls or dynamic accesses after a state-changing compiled prefix and embedded
native-code emission remain future tiers.

LEA is lowered as runtime Wasm address arithmetic rather than entry-state
substitution, so it observes base/index registers written earlier in the same
compiled block. Register/immediate CMP and TEST share the proven arithmetic
flag machinery but suppress the destination write. Exact-entry PUSH of a
register or immediate and POP to a register update RSP and the guest stack
mirror with the normal read/write witness while deliberately leaving the
validation call stack untouched. The dispatcher exposes bounded denial counts
by semantic token, allowing subsequent lowering work to follow measured real
subject frontiers rather than synthetic guesses.

CMP and TEST also accept one guest-memory operand. Static addresses remain
usable after earlier compiled instructions; a dynamic address is admitted only
at an exactly guarded block entry. The journal records a read effect with the
resolved address, data width, and observed value, while signed immediates are
extended to the operation width before flag calculation. No operand write is
performed. This removed the final comparison/test denial from the measured
startup frontier without adding another memory-effect channel.

Guest-window planning now grows the safe prefix greedily. If a later static or
specialized access would make the contiguous mirror exceed 64 KiB, the artifact
ends immediately before it, records that instruction as its shortfall, and
lets the runner resume there; the distant future access no longer rejects a
safe earlier MOV. Scalar ADD/SUB/AND/OR/XOR memory destinations reuse one
read-modify-write journal effect whose before and after values are independently
checked during reconstruction. MOVSX/MOVZX-style scalar extension accepts a
register or witnessed memory source; MOVSXD performs the 32→64 signed mapping
inside Wasm. Dispatcher diagnostics include bounded semantic, decoded-token,
and lowering-reason counts.

CDQE and CQO now perform the implicit accumulator sign extensions in Wasm.
BT/BTR/BTS/BTC accept register destinations and exact-entry witnessed memory;
only CF changes. A register index on a memory bit string is interpreted as a
signed quotient plus modulo bit index, so an index such as -1 selects bit 63 in
the preceding 64-bit word. The index register and resulting address are both
part of specialization provenance. Modifying forms journal the selected word
as one exact read-modify-write edge. Memory-width inference is token-boundary
aware, preventing the `IMM8` suffix in `BTR_RM32_IMM8` from turning a dword
operation into an 8-bit operation.

Exact-entry `REP STOSW` is represented as one bounded range-fill operation.
The Wasm loop writes at most 32,768 words (64 KiB), then records a compact fill
descriptor containing the original RDI, RCX count, AX word, direction, and
the exact final RCX/RDI checkpoint. Reconstruction validates that descriptor
against the parent state and rebuilds the changed immutable pages. The old
pages already live in the parent tape node, so reversal needs neither 8,192
micro-edges nor a second 16 KiB byte copy for the real command route.

Immediate SHL, NEG/NOT, SETcc, and INC/DEC now share that scalar register or
witnessed-memory path. Shift and unary flag results differentially match the
reference interpreter; INC/DEC preserve incoming CF. XCHG swaps two registers
or one 64-bit guest-memory cell and a register. The memory form is retained as
one indivisible guest instruction with one authenticated before/after effect,
so its implicit x86 atomicity is not split across virtual-core schedule edges.
CMOV unconditionally authenticates a memory source even when its predicate is
false, while preserving the untouched upper half of a false 32-bit destination.
SBB includes incoming CF and the effective-operand overflow boundary. SAR/ROL
retain their distinct flag contracts. Unsigned `MUL r/m64` computes both halves
of the 128-bit product into RDX:RAX with exact 32-bit limbs; a memory multiplier
is an authenticated read effect.
On a fresh deterministic `cmd.exe /c "echo hello"` run, 167 approved external
calls led to the unchanged fail-closed `msvcrt!_local_unwind` boundary after
3,043 transitions. This tier compiled 2,647 transitions (about 87%), up from
2,140 before the expanded scalar tier, while denied blocks fell from 236 to
104. Only one XMM move and one XMM XOR remain as non-control denials on this
route; vector lowering requires widening the compiled checkpoint ABI.

That widening is now implemented as `turing.machine-block-state.v2`. Each
checkpoint appends sixteen XMM values as 32 contiguous little-endian qwords;
the exact machine tape and the 54-cell `TMSNAP01` display bank therefore see
the same compiled vector values. Vector XOR, 128-bit loads, and 128-bit stores
execute through Wasm. Memory provenance contains low/high before and after
halves rather than truncating a 128-bit effect into the scalar witness. Because
the wider record is exactly 512 bytes, the Node host page-aligns the guest
mirror after the actual journal extent instead of assuming offset 4096; a
nine-instruction memory block verifies that the regions cannot overlap. The
same real run now compiles 2,669 of 3,043 transitions (about 88%). All 103
remaining denials are indirect calls or jumps retained at internal-target or
external-capability validation boundaries; no semantic denial remains on the
measured route.

The runner tier is now installed behind the explicit `node-wasm` machine
backend. Native Python owns one private persistent Node worker, sends it only
compiler-emitted modules plus bounded state/guest buffers, and caches both
artifacts and instantiated modules by digest. A compiled prefix returns its
normal instruction journal, so shell ticks, free spin, system-tape observers,
reverse execution, and the live HTML snapshot controller retain exactly the
same per-instruction behavior. A first unsupported instruction is cached as a
bounded fallback decision; host protocol corruption or Wasm execution failure
raises instead of silently switching engines. The module declares enough
64 KiB pages for state, journal, and the promised guest window rather than
assuming one page. Use `--machine-backend node-wasm` with either
`reversible_machine_viewer.py` or `reversible_machine_web_host.py`; the default
`translated` policy remains preferable until broader instruction coverage
makes cross-process dispatch faster on real subjects.

`python examples/reversible_cmd_probe.py --new` starts a real `cmd.exe /c
"echo hello"` run and writes `cmd-machine.tape.jsonl`. Running the same command
without `--new` reconstructs the pending machine state and continues from the
last unsupported boundary rather than replaying startup.

## Colored tape annotations

Tape annotations are append-only records targeting one sequence or a sequence
span. Each annotation has a machine-readable feature, message, named or RGBA
color, severity, and optional core, history position, RIP, external reference,
metadata, and superseded-annotation ID. This supports both human notes such as
“this might be wrong” and automatic execution diagnostics without rewriting
the recorded machine state.

A missing semantic handler automatically adds a red
`instruction_set_compatibility` annotation at the exact RIP with the decoded
instruction token, semantic token, encoded bytes, and AMD64 compatibility
status. Unsupported external calls and probe exceptions receive amber or
magenta annotations. The latest active per-core color is packed into the final
word of the `TMSNAP01` core-status record; the WebGPU compute shader blends it
into the RIP cell. Superseding annotations preserve the original note while
allowing a later review to mark it green/verified.

`BinaryMachineProgram.load_pe()` is the runtime boundary. It sends subject
bytes through the existing PE machine decompiler and constructs the executor,
clock, virtual cores, device buffers, and observation slots. Subject code does
not enter the emulator application's card or SSA representation.

The real `cmd.exe /c "echo hello"` probe now completes. A cold graph-based
resume reconstructs the segmented dependency chain, a halted AMD64 process at
instruction step 16,010, and exit code 0. Its authoritative reversible device state is
exactly `hello\r\n`, generation 1. Publishing that cold-resumed state produces
a `TERMINAL` / `UTF8` shader snapshot descriptor with the same generation and
the same bytes. The path includes CRT startup and exit callbacks, virtual
heap/environment/VFS/registry/virtual-memory state, locale and wide formatting,
console synchronization,
runtime-proven code dispatch, dynamic `GetProcAddress` links, and normal process
termination. Unknown semantics and capabilities still stop at annotated
boundaries; no arbitrary host-call fallback was added.

## Running the displays

Open the completed taped run in the native OpenGL viewer:

```powershell
$tape = Join-Path ([System.IO.Path]::GetTempPath()) 'turing-cmd-probe.tape.jsonl'
python examples\reversible_machine_viewer.py --tape $tape
```

The upper half is the contiguous AMD64 register bank and the lower half is the
guest terminal or framebuffer. Space pauses, `B` reverses, `F` advances, and
Escape closes the viewer. Add `--clocked --speed 1000` to replace free-spin
execution with a shell-regulated transition clock. This executes the emulated
guest and its capability ports; it does not launch or proxy a host `cmd.exe`.

To make a self-contained HTML view of the current segmented state:

```powershell
python -m src.compiler.dream_document examples\reversible_chip_simulator.dream `
  --emit-shell build\reversible_chip_cmd_card.html `
  --machine-tape build\cmd-interactive.segmented-tape
```

`--machine-tape` accepts either JSONL or a segmented directory. It reconstructs
the latest state through the dependency graph, publishes a fresh `TMSNAP01`
register/device buffer directly from retained state (without recompiling the
subject PE), and embeds it in the page.
Opening the HTML therefore displays the taped terminal and registers without a
Python process. That embedded artifact is a shader view of one authoritative
tape state, not a JavaScript emulator.

For a continuing run, start the loopback machine-program host instead:

```powershell
python examples\reversible_machine_web_host.py `
  --tape build\cmd-interactive.segmented-tape --open
```

The Python controller is the sole writer of guest state. It services the
capability-gated Windows port, advances a bounded batch, publishes one complete
`TMSNAP01` flip, and then repeats as fast as the current frontier permits. The
same-origin HTML shell polls only the newest immutable generation; skipped
display frames do not slow or alter execution. Browser request threads can
enqueue bounded terminal bytes but cannot touch the machine. At a receptive
prompt, type into the command field along the bottom of the display and press
Enter. The equivalent programmatic call is
`await TuringMachineSnapshots.sendTerminalInput("dir\r\n")`.

Use `--new` to compile the selected binary and create a fresh segmented tape,
or omit it to resume the existing directory. `--demo-card hello-card` installs
the demonstration bundle executor so a typed `hello-card alpha beta` command
travels through the same virtual `CreateProcessW` and child-tape path. The
server rejects non-loopback bind addresses by construction.

The HTML controller shows all 54 contiguous architectural/observation cells as
named 64-bit hexadecimal values over the shader's occupancy colors. The lower
half remains the program-owned terminal/framebuffer. A browser render of the
real card run is retained at `build/reversible_chip_cmd_card_final.png`.

The interactive probe can also resume a segmented directory tape. The current
long-lived interactive `cmd.exe` evidence contains 69,949 records in 278
content-addressed segments. At sequence 67,946, real `cmd.exe` resolved
`C:\work\hello-card.exe`, invoked `bundle:demo/hello-card@local` through
`card-set:hello-card:v1`, observed exit code zero, printed
`[card-set:hello-card] alpha beta`, and returned to a pending `ReadConsoleW` at
step 67,459. The deployment links to its own content-addressed child tape.

A post-cache live regression used a disposable copy of that receptive tape.
The controller executed `echo live-check`, published 578 complete snapshot
generations, appended 7,756 state events and 31 immutable segments, and returned
to `api-ms-win-core-console-l1-1-0.dll!ReadConsoleW` without a controller
failure. The hot history range was `[3661, 7757)` with exactly 4,096 resident
states while the reopened store contained 77,705 records in 309 segments. Its
retained terminal ends with `C:\work>live-check` followed by a new
`C:\work>` prompt.

A fresh compiled `cmd.exe /c "echo hello"` proof now traverses the x64 CRT
local-unwind route as well. The capability reads `.pdata`/`.xdata` from the
reversible guest image, identifies the one C termination scope crossed by the
target IP, and enters its guest `__finally` routine with `RCX=1` and the
establisher frame in `RDX`. Unfamiliar or unbounded unwind shapes remain
pending. The program then halts with exit code zero after 15,205 guest steps
and 589 deterministic external completions. Wasm journal validation re-decodes
runtime-discovered continuation instructions from current guest bytes, so the
cleanup path retains the same instruction witnesses and reverse edges as
statically catalogued code.

The supported probe can create that proof directly rather than materializing a
large JSONL history first:

```powershell
python examples/reversible_cmd_probe.py --new --segmented `
  --machine-backend node-wasm `
  --tape build/cmd-echo.segmented-tape /c "echo hello"
```

The current proof produced 16,510 records in 66 immutable segments (8,570,148
bytes), with 15,459 of 16,010 guest instructions committed from authenticated
Wasm journals and 499 deterministic capability completions. It halted with
exit code zero and retained `hello\r\n`. Every
remaining denial is an intentional indirect-control or lifecycle-return
boundary; the ordinary semantic denial inventory is empty.
Cold invocation of the same command without `--new` restored the halted tip at
absolute position 16,509 and printed `subject_output_tail=b'hello\r\n'` without
adding another state. Direct reverse decoded only the 125 records in the final
segment, and replay reproduced the halted tip byte-for-byte. JSONL and segmented
loaders now install the persisted absolute position as the executor's cold
history base; resumed reversal therefore enters the taped prefix instead of
mistaking the loaded tip for position zero.

A separate real command-pipeline probe exercised the reversible anonymous-pipe
and CRT-descriptor tier. `cmd.exe` created and closed `pipe.1`, crossed its
MSVCRT `longjmp` error-recovery path, and halted normally at guest step 23,232
with 539 deterministic completions and exit code 255 for an intentionally
unresolved pipeline command. The segmented tape contains 23,773 records. Cold
reopen restored its terminal and exit exactly; one backward step returned to
the non-halted predecessor and one forward step reproduced the complete tip.
The registered-card resolver still needs a pipeline-specific resolution proof,
but pipe transport, endpoint lifecycle, nonlocal control, and exact replay no
longer stop at an unsupported capability.

## Common root-site bundle

The Dream machine is published through the same immutable program-bundle
contract as compiled Python interiors, rather than a demo-only directory:

```powershell
python examples/reversible_machine_web_host.py `
  --publish-bundle C:\dev\Powershell --publish-only
```

`publish_prebuilt_program_bundle` owns content addressing, atomic version
creation, artifact SHA-256 inventory, the `turing-program-bundle-v1` manifest,
and refresh of the root shell's static gallery. The current public version
contains the Dream source, the PE-generator source, and a project-authored
2,048-byte PE32+ AMD64 subject whose entry bytes are `90 c3` (`NOP; RET`). No
Windows system executable is copied into the repository.

The static page embeds two full `TMSNAP01` generations produced by the real
executor. Its standard forward, backward, pause, single-step, and speed controls
therefore operate as a finite reversible replay on GitHub Pages. With the
Python owner running, the same shell uses `/snapshot`, `/control`, `/subject`,
and `/input` for unrestricted tape execution and admitted system activity.
Static arbitrary-binary loading remains intentionally unavailable until the
machine owner is lowered into a browser runtime.
