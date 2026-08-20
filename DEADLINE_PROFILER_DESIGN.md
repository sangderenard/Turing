# The deadline profiler: fitting kernels to shader time, not just to speed

A design note, written 2026-08-19 as a side path from the identity/deployment
work. It concerns the GPU lane, which is far behind the CPU lane, and the one
thing that makes the GPU lane categorically different: **on Windows, a shader
does not merely need to be fast, it needs to finish.**

Nothing here is built yet. Everything cited as present was read this session.

---

## 1. The premise: two different objective types

The repository already has a calibrator, and it is good. `deployment_calibration.py`
refuses to believe a static table, probes the actual machine, requires a real
margin (1.15×) before promoting a pool, persists verdicts keyed by workload
signature *and* machine fingerprint, and degrades to serial whenever a verdict
is absent, stale or foreign. Its own docstring: *"test the water, then earn the
default."*

It optimizes a **ratio**. Is the pool 1.15× faster than serial? That is the
whole question, and it is the right question, because on the CPU **the deadline
is infinite**. A chain that takes 200 ms instead of 20 ms is a disappointment.

On the GPU under Windows, exceeding the deadline is not a disappointment. The
Timeout Detection and Recovery watchdog resets the device. The dispatch does not
return late — it does not return, the context is lost, and every other buffer on
that device dies with it. **It is a liveness property, not a performance
property**, and no ratio-based verdict can express it.

So the profiler this note proposes produces a different kind of verdict:

| | CPU today | GPU needed |
|---|---|---|
| objective | maximize throughput | **satisfy a deadline**, then maximize throughput |
| verdict | "pool of 4, 1.3× serial" | "≤ N elements per dispatch on this device" |
| failure mode | slower than hoped | device reset, process dies |
| deadline | ∞ | `TdrDelay`, default 2 s, user-modifiable |

**The unification, which is the reason to build one tool and not two:** the CPU
is the case where the budget is infinite. Set the deadline to ∞ and the
constraint goes vacuous, the chunking degenerates to a single chunk, and the
model reduces *exactly* to today's throughput calibration. One profiler, one
verdict type, with the CPU as the degenerate instance — not a GPU bolt-on beside
a CPU tool.

---

## 2. What already exists, and the precise shape of the gap

### 2.1 The yield mechanism exists and lowers to both sides

This is the good news and it is better than expected. `nodus/src/kernel_isa.h`
already carries a cooperative work budget in KernelIR:

```
// Cooperative yielding rides here: when KernelIR::budget_value_id names
// a buffer, the loop's condition also consults that budget so a kernel
// stops of its own accord before a watchdog (Windows TDR) stops it for
// us. Dressing the control primitive means every generated kernel is
// bounded by construction, with no caller obliged to remember.
```

And it is *implemented on both backends*, not just described:

| backend | site |
|---|---|
| Eigen (CPU) | `src/kernels/kernel_eigen.cpp:783` |
| SPIR-V (GPU) | `src/kernels/kernel_spirv.cpp:742` |

`kNoBudget` is the default, so every existing kernel is unbudgeted, and the host
is expected to re-dispatch until the kernel reports completion.

### 2.2 Nothing measures anything on the GPU

`include/kernel_vulkan_runner.h` contains **no timing code whatsoever** — no
`chrono`, no timestamp queries, no query pool.

Checked repo-wide, not just in nodus: a search of every sibling project under
`C:/dev/Powershell` for `vkCmdWriteTimestamp` / `QueryPool` / `timestampPeriod`
returns hits only inside `vcpkg_installed` third-party headers. The one apparent
exception, `amp/`, matches on a bare `timestamp: float` dataclass field — an
application event time, not a device query. **There is no GPU timing measurement
anywhere in the tree.**

**So the gap is exact, and it is not the mechanism.** The kernel can be told to
yield after N units of work. Nothing on earth currently knows what N should be,
because nothing measures how long a unit of work takes on the device in front of
you. `budget_value_id` is a steering wheel attached to nothing.

### 2.3 Nothing reads the deadline either

Searching every project under `C:/dev/Powershell` for `TdrDelay`, `TDR`,
`device_lost`, `deviceLost` or `DXGI_ERROR_DEVICE` returns **exactly one file**:
`nodus/src/kernel_isa.h`, and that is the single comment quoted above. The
budget is not derived from the actual watchdog setting because the actual
watchdog setting is never read, anywhere, by anything.

### 2.4 Operators arrive faster than benchmarks could be written

40 generated Eigen vocabulary tools already exist under
`module_library/test_build/vocab_tool_test/source/tools/vocab/eigen/`
(`abstract_tensor_abs.cpp`, `_acos`, `_acosh`, …), produced by the vocab→tool
pathway. Operators are being **generated**, continuously. Any profiler that
requires a human to hand-write a benchmark per operator is obsolete on arrival.

**Consequence for the design: the probe must be generated from the operator's
own signature, by the same mechanism that generates the tool.** If a new
operator can be lowered, it can be probed, with no additional authoring.

---

## 3. The tool

### 3.1 What it produces

Not a winner. A **cost model** per `(device, operator, dispatch shape, loop
order, dtype)`:

```
cost(n) ≈ fixed_overhead + per_unit * n
```

* `fixed_overhead` — submit, barrier, fence wait, and on GPU the descriptor and
  command-buffer cost. Measured by probing at several small `n` and taking the
  intercept, not by assuming zero.
* `per_unit` — the slope. This is the number that must be measured per device,
  because integrated and discrete parts differ by more than an order of
  magnitude on the same kernel.

From which the thing the caller actually wants falls out:

```
max_units_per_dispatch = (deadline - margin - fixed_overhead) / per_unit
```

`margin` is not decoration. The GPU is shared with the desktop compositor and
whatever else the user is running; a dispatch that measured 1.4 s alone can
exceed 2 s while a video plays. **Recommendation: budget to a fraction of the
watchdog (start at 0.25) and treat the remainder as contention headroom**,
because the cost of being wrong is a device reset, not a slow frame.

### 3.2 Reading the deadline, never writing it

`TdrDelay` lives at
`HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers\TdrDelay` (REG_DWORD,
seconds, default 2 when absent; `TdrDdiDelay` defaults to 5). The profiler
**reads** it, records it in the machine fingerprint, and derives the budget from
it.

**It must never write it, and the documentation must not suggest a user raise
it.** Raising the watchdog to make a kernel fit converts "one kernel gets
killed" into "the whole desktop hangs for the new timeout" — and it hides the
defect on the developer's machine while shipping it to everyone else. Chunking
is the fix. The registry value is an input, not a knob.

A machine with the watchdog disabled entirely (headless compute, TCC-mode parts,
some Linux configurations) is simply the `deadline = ∞` case again, and lands on
the same code path as the CPU.

### 3.3 Loop interchange is part of the key, not a separate concern

This is why the tool has to be aware of interchange rather than sit downstream
of it. On CPU, interchange changes cache locality — a ratio question. On GPU it
changes **which loop becomes the dispatch dimension and which stays serial
inside the kernel**, and that changes occupancy, memory coalescing and therefore
the slope `per_unit` by a large factor.

So `loop order` is part of the profile key, and the profiler's output feeds
interchange rather than merely consuming its result:

* an interchange that raises `per_unit` but shrinks the serial inner extent may
  still be the right choice, because it makes a chunk *fit*;
* an interchange that lowers `per_unit` but forces an indivisible inner loop can
  be **illegal** on GPU, because the resulting dispatch cannot be chunked below
  the deadline at any size.

Neither of those judgements is expressible without a deadline in the model.
That is the concrete sense in which this "works into loop interchange": it turns
interchange from a locality heuristic into a feasibility test.

### 3.4 What it does to deployment classification

`deployment_classification.py` picks graphics-output / shader-compute /
thread-workers / host-linear per region. Today that is a static classification
of what is *legal*.

With a deadline model it gains a rung: **shader-compute is admissible only if a
chunked dispatch fits the budget with margin.** A region whose smallest legal
chunk still exceeds the deadline is not a shader-compute region on this machine,
whatever its shape says — it degrades to thread-workers or host-linear, with the
measured numbers in the reason trail, exactly as `deployment_calibration`
already degrades a losing pool to serial with its ratio recorded.

The existing commitment ladder — *shadow* → *measured* → *committed* — carries
over unchanged, and its safety property carries over with it: an absent, stale
or foreign verdict means the conservative path. For the GPU, "conservative" is
"smaller chunks or don't dispatch at all", and being wrong in the safe direction
now means the device survives.

### 3.5 BLAS: from tuning knob to feasibility constraint

The BLAS lane (`AT.blas` — scal/axpy/dot/gemv/gemm compiling and matching the
oracle bit-exact; the kernel bank's specialized gemm at 1.57×) currently picks
tiles for efficiency, with auto-tiling named as the next step.

On the GPU, tile choice stops being only an efficiency decision:

* a `gemm` whose full dispatch exceeds the watchdog **must** be split, and the
  split factor comes from `max_units_per_dispatch`, not from a cache-size
  heuristic;
* the split has to respect the reduction structure — a `k`-split needs partial
  accumulation across dispatches, which is exactly what `budget_value_id`'s
  re-dispatch-until-complete protocol already provides;
* the largest tile that fits is usually the fastest tile that fits, so the
  deadline gives an *upper* bound and the efficiency model chooses beneath it.

That layering is the design: **the deadline bounds the search space, efficiency
searches inside it.** On CPU the bound is infinite and efficiency searches the
whole space, which is what happens today.

---

## 3.6 Reconciliation: what the CPU lane already built

**Written against `af00599`; the tree has since moved to `020ec07`.** Several
pieces this note would otherwise have proposed from scratch now exist on the CPU
side, and the design's job is to extend them rather than to duplicate them.

| landed | what it means here |
|---|---|
| `65ccad9` emit host target lines and storage-derived `noalias` | the "declare what you know" gap in §2 of the identity handoff is closed for LLVM. The GPU equivalent — declaring the deadline — is still open, and is this note. |
| `f18680b` deterministic SSA identity pass | operators reaching the GPU lane are now strength-reduced before they get there, so a probe measures the shipped kernel |
| `020ec07` deterministic loop interchange wired into SSA lowering | **§3.3 is no longer hypothetical.** Interchange is a real decision made in a real module; the deadline turns it from a locality choice into a feasibility test on the GPU |
| `26f652c` / `56806ef` chart-ranked tile choice with truthful reasons | `tiling_strategy.py` already keeps a candidate ladder |
| `tools/kernel_bank_probe.py`, `tools/blas_backend_bakeoff.py`, `tools/profile_eigh.py` | CPU-side measurement harnesses whose statistics protocol the GPU probe should share, not re-invent |

The most important of these is `tiling_strategy.py`, whose stated contract is
that *"every choice AND every veto carries a reason string, and the full
candidate set survives alongside the decision so a dispatcher with different
constraints can fall back."*

**That is precisely the seam a deadline needs, and it already exists.** A
deadline-bounded GPU dispatcher *is* a dispatcher with different constraints: it
walks the surviving candidate ladder and takes the highest-ranked tile whose
measured dispatch fits the budget, vetoing the rest with a reason that names the
measured time and the watchdog value. §3.5's layering — deadline bounds the
search space, efficiency searches inside it — needs no new decision structure,
only a new veto predicate and the measurement to evaluate it.

So the revised framing: **the CPU lane has the decision machinery and no
deadline; the GPU lane has the yield mechanism and no measurement.** The tool
supplies the measurement, and the deadline becomes one more constraint the
existing ladder already knows how to be vetoed by.

---

## 4. Build order

1. **Timestamps in the Vulkan runner.** Query pool around dispatch,
   `timestampPeriod` from device properties, elapsed reported to the host.
   Nothing downstream is possible without this and it is self-contained.
2. **Read `TdrDelay`** into the machine fingerprint. Small, and it makes the
   deadline a measured input rather than a constant.
3. **A signature-generated probe.** From an operator's signature, emit the
   probe the same way the vocab pathway emits the tool. Ladder over `n`, fit
   intercept and slope, best-of-k on a monotonic clock, warmup discarded — the
   same protocol `deployment_calibration` already uses, so the CPU and GPU
   probes share their statistics.
4. **The deadline verdict type**, persisted through `RepositoryArtifactCache`
   beside the existing verdicts, fingerprint including device id, driver
   version and `TdrDelay`. A verdict from another machine is not evidence; a
   verdict from the same machine with a different driver is not evidence either.
5. **Feed chunking**: `budget_value_id` gets its value from the verdict. This is
   the moment the steering wheel connects.
6. **Feed classification and interchange**, in that order — classification is a
   filter on an existing decision, interchange is a new search.
7. **BLAS tiling under the bound.**

Steps 1–2 are worth doing even if the rest is deferred: today nobody can answer
"how long did that shader take" or "how long is it allowed to take", and both
questions have single-file answers.

---

## 5. Risks worth stating up front

* **Probing costs GPU time on a device that punishes long work.** The probe
  ladder must start small and stop on the first sign of approaching the
  deadline. A profiler that trips the watchdog while measuring the watchdog is
  a plausible and embarrassing failure.
* **Contention makes the measurement non-reproducible.** Best-of-k helps;
  margin is what actually saves you. Do not tune the margin down to make a
  benchmark look good.
* **`timestampPeriod` and timestamp validity vary by queue family.** Check
  `timestampValidBits` rather than assuming, or the numbers are silently
  garbage — the exact failure class this repository's debugging doctrine calls
  a measurement you cannot take being mistaken for a zero.
* **The CPU path must not regress.** The unification is only worth it if
  `deadline = ∞` provably reproduces today's verdicts. That should be a test,
  not a hope.
