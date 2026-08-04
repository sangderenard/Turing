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

## Reversible clock

The execution history versions PC, registers, flags, memory, and call stack.
Backward movement restores a prior version; forking preserves alternate
futures. The wall-clock governor caps both cycles per rendered frame and total
cycles. Excess elapsed time is dropped rather than accumulated as catch-up
debt, preventing a paused or self-hosted machine from producing a time-dilation
spiral when rendering resumes.
