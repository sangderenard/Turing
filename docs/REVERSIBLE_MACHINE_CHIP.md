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
image base, creates a zeroed 1 MiB stack, and installs minimal PEB/TEB pages.
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

A completion can update `RAX` and apply bounded writes only to already mapped
guest memory. The completion itself is a journal edge, so reversing across it
restores the pending request and all pre-call state. The supplied deterministic
Windows bootstrap port is an exact capability allowlist, not Win32 passthrough.
Unknown functions remain pending for a shell policy to approve, capture, or
emulate.

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

`BinaryMachineProgram.load_pe()` is the runtime boundary. It sends subject
bytes through the existing PE machine decompiler and constructs the executor,
clock, virtual cores, device buffers, and observation slots. Subject code does
not enter the emulator application's card or SSA representation.

As of the `cmd.exe` execution probe, the actual system binary advances through
PE validation and security-cookie setup and captures imports including time,
process/thread identity, tick count, module-handle lookup, and CRT setup. The
frontier is intentionally moved only by implementing coherent semantic or
capability families; unsupported behavior remains visible rather than being
silently approximated.
