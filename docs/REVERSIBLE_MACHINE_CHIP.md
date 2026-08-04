# Reversible machine chip layout

The virtual multicore uses a deliberately physical observation ABI. It is not
a claim about the host CPU's silicon; it is the memory map seen by the compiled
executor and its WebGPU display.

## Register banks

Each core owns one fixed 256-byte-aligned bank. The bank begins with twenty
contiguous 64-bit cells:

```text
RAX RCX RDX RBX RSP RBP RSI RDI R8 ... R15 RIP RFLAGS STEPS CALL_DEPTH
```

Each cell is exactly eight contiguous bytes, exposed to WebGPU as adjacent
little-endian low/high `u32` words. Core `n` begins at
`register_base + n * 256`. The remaining 96 bytes are bank padding reserved for
future architectural registers; no following core can overlap them.

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
