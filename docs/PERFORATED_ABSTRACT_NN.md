# Perforated AbstractNN comparison

`PerforatedLinear` is an AbstractTensor-native dendritic extension of an
ordinary linear layer.  It is inspired by the architecture described in
*Perforated Backpropagation: A Neuroscience Inspired Extension to Artificial
Neural Networks* (Brenner and Itti, 2025), but it does not copy or claim to
implement Perforated AI's proprietary perforated-backpropagation trainer.

For output neuron `j`, the forward equation is:

```text
y = x @ W + b + (gain * tanh(x @ D + db)) @ fixed_branch_route
```

The dendrites are neuron-specific: branch `k` of output `j` has its own input
weights, bias, and learned gain.  The layer exposes three parameter surfaces:

- `base_parameters()` for the ordinary point neuron;
- `dendrite_parameters()` for the nonlinear branches;
- `parameters()` for the complete model.

This supports a simple perforation cycle using the same AbstractTensor Adam
optimizer throughout:

1. train the ordinary base while dendrites are absent from the graph;
2. call `perforate(True)` and train only dendrite parameters on the remaining
   error;
3. keep the learned dendrites fixed and consolidate the base parameters.

The layer's numerical forward path contains only AbstractTensor operations:
matmul, multiply, add, and tanh. Consequently it can run on the
NumPy or Torch/CUDA eager backends, use the repository's own reverse-mode
autograd, and be captured as a `FusedProgram` for translation.

Run the reproducible held-out comparison with automatic CUDA selection:

```powershell
python -m src.common.tensors.abstract_nn.demo_perforated_regression
```

Or select a backend explicitly:

```powershell
python -m src.common.tensors.abstract_nn.demo_perforated_regression --backend numpy
python -m src.common.tensors.abstract_nn.demo_perforated_regression --backend torch-cuda
```

The comparison starts the ordinary and perforated contestants from identical,
independent base weights.  Both get the same total number of Adam steps.  The
synthetic target deliberately contains both a linear part and a held-out
dendritic nonlinear part, so this is a structural capability check rather than
a claim of general benchmark superiority.

## LLVM contract and tape-free reverse

`src.compiler.perforated_network_llvm.compile_perforated_network` authors the
same layer on the SSA AbstractTensor backend and emits two native libraries:

- an isolated forward entry;
- a combined forward/VJP entry whose backward comes from the ProcessGraph
  backward generator, not a captured or replayed gradient tape.

The versioned `contract.json` beside the libraries names every contiguous
row-major float64 input, parameter, prediction, upstream prediction-adjoint,
and gradient buffer by semantic name, value ID, role, and fixed shape.  This is
the fast training boundary: run the compiled VJP and feed its parameter
gradients to Adam without rebuilding or walking the tape.

```python
from src.compiler.perforated_network_llvm import compile_perforated_network

compiled = compile_perforated_network(
    ".turing-cache/perforated-llvm",
    batch=2,
    in_dim=4,
    out_dim=3,
    dendrites_per_neuron=2,
)
print(compiled.manifest_path)
```

The routing matrix is a `fixed-structure` input in the native contract.  Its
rows contain one `1` each, assigning every contiguous dendrite group to its
output neuron; it participates in reverse computation but is deliberately not
published as an Adam parameter or gradient.

### Native changing-minibatch Adam cycle

`compile_perforated_adam_chunk` emits one native entry containing the
perforated forward, weighted loss, ProcessGraph-generated VJP, and a stateful
Adam cycle. Its `x`, `target`, `sample_weight`, and `loss_scale` inputs carry a
leading fixed chunk axis; each loop iteration repoints the motion ABI at the
next slice before recomputing gradients. Parameters, first and second moments,
bias-correction powers, and the iteration counter are caller-owned in/out
buffers. There is no Python callback inside the cycle and no tape backward.

The same entry performs ordinary averaged gradient accumulation followed by
global L2 norm clipping before Adam. Both are compile options
(`gradient_accumulation_steps` and `max_global_gradient_norm`) recorded in
contract version 2. Gradient accumulators are caller-owned heap buffers rather
than native stack allocations, so union-sized engine models do not consume the
small Windows thread stack. The contract also publishes the final pre-clip and
post-clip norms.

The live and headless engine demo now uses this entry for actual training. It
banks one complete shuffled dataset pass, including sample weights,
information-dropout values, and per-motion dendrite masks. A runtime `steps`
scalar may exceed that fixed bank length; LLVM cycles the bank internally, so
epochs and passes add compute without multiplying host memory. The live shadow
replay reuses the same native entry with `steps=1`.

The demo defaults are intentionally solve-oriented: all catalogued engine
profiles, 256 named samples and 256 episodes per fuel/profile, 3,000 additional
random simulator transitions, 32 epochs with one complete dataset pass each,
batch 64, accumulation 4 (effective batch 256), global clip 1.0, and Adam
learning rate 0.002. Every setting remains a CLI option for smaller development
runs. `passes_per_epoch=1` keeps the ordinary meaning of an epoch: one full
visit to the captured dataset.

The explicit full-shaped `loss_scale` lets each minibatch retain exact
valid-row normalization without differentiating a dynamic divisor, which the
current repository LLVM call lowering cannot yet emit. The Adam arithmetic is
the LLVM backend realization of the same law as functional AbstractTensor
`adam_step`; joining that optimizer graph itself to the motion before LLVM is
still a later composition step, not something this smoke claims.

The focused three-minibatch regression uses accumulation 2 and clipping 0.15,
then matches an independent NumPy reference for final loss, every parameter,
every first/second moment, both beta powers, iteration count, and both gradient
norms. An actual headless LDT engine run (`423 -> 212`, batch 2) compiled an
18-batch bank and executed 72 motions in one native call in 0.999 seconds; its
training loss moved from 1.11127 to 0.02979. Its contract and DLL are cached
under the selected output directory.

Contract version 2 adds three runtime graph-control ports without changing the
compiled topology:

- `dendrite_mask` gates individual learned branches;
- `network_authority` selects learned outputs per batch row and output port;
- `simulator_delta` supplies the matching graph simulator outputs wherever
  network authority is zero.

Intermediate authority values crossfade the two sources. Because authority is
inside the AbstractTensor expression, the ProcessGraph-generated VJP also
gates parameter gradients for simulator-owned outputs. This provides a stable
scene-graph handoff ABI: synchronize state/history at the boundary, change the
authority tensor, and do not recompile LLVM. Dense masked branches still incur
dense compute; compute-eliding scene deployments should compile independently
dispatchable machine islands or bounded topology variants.

## Whole-engine multifuel comparison

`demo_perforated_multifuel_engine` uses the sibling `engine_toy` project's real
LDT-465 `EngineCycleSim` as a teacher. It captures independent trajectories for
starting, idling, idle recovery, compensated idle load, load recovery,
high-end operation, upshifts, and downshifts. Seeded low-frequency noise acts
only on genuine operator and electrical-load commands; engine state and every
target remain direct simulator observations. Component damage ledgers are
reduced to health statistics so hundreds of graph records do not become
thousands of trivial identity outputs.

The recent idle-load adjustment remains owned by
`ecu.EngineControlUnit.idle_load_feedforward_frac`. The benchmark supplies
`known_accessory_shaft_load_w` through `EngineCycleSim`, applies the matching
mechanical reaction as `shaft_power_w / transfer_omega`, records that ECU-owned
feedforward value as an input, and never copies its formula into Turing. Every
trajectory synchronizes the unloaded service shaft and ramps the existing
clutch before load is applied.

Unlike the staged structural demo above, this experiment enables dendrites at
construction and passes `parameters()` to Adam on every step. Thus every base
and dendritic trainable tensor is optimized jointly while the model learns the
complete moving next-state delta:

```powershell
python -m src.common.tensors.abstract_nn.demo_perforated_multifuel_engine --backend torch-cuda
```

The current contract has 416 inputs: the complete numeric engine/state/control
surface plus a compact one-transition history of important state and control
differences. It predicts all 212 numeric state deltas under a stable ABI rather
than removing outputs that happened to be constant in one capture. The
contract explicitly identifies crank, RMS, accessory, compression-brake, and
load-shaft torque/power outputs. Dataset defaults are 1,280 transitions from 40
independent episodes; train/test separation is by complete episode.

The live learner defaults to a union over all 29 catalogue profiles and adds
3,000 direct simulator transitions balanced across them. Each profile also
contributes named start/idle/recovery/high-end/shift trajectories for its
compatible fuels. Heterogeneous configuration and state fields are zero-padded
into one stable ABI, with an explicit `engine_profile.<identity>` selector.
Randomized trajectories cover correlated throttle, shaft load, direct brake,
electrical demand, gear, clutch, starter, ignition, and fuel commands, including
cold starts and genuinely preconditioned operating states.
`--engine-profiles <identity> ...` selects a smaller union and
`--random-coverage-samples 0` keeps only named scenarios.

Training applies 2% mean-value dropout to non-command input information and 2%
dendrite-branch dropout by default. Explicit controls, fuel, and engine-profile
selectors are protected. This teaches the union transform to tolerate absent
subgraph information without erasing command responsiveness.

## Compiled live engine learner

`demo_engine_dendrite_live` compiles the exact engine learner dimensions before
training. Every optimization step calls the standalone LLVM forward and the
ProcessGraph-generated LLVM VJP; Python retains only Adam's moment arrays and
the presentation loop. One epoch is a configurable `N` complete dataset
passes (`--passes-per-epoch N`), and every pass weights every training
transition exactly once. Fixed native batches are padded with zero-weight rows.

Interactive audio is driven solely by a second batch-1 LLVM instance carrying
the current weights. Its predicted deltas are integrated into its own engine
state. A shadow `EngineCycleSim` receives the same operator commands and emits
real `(state, controls, next-state)` examples into an 8,192-transition replay
buffer; it is a teacher, not the audio/state source. The learned state is
re-anchored to the teacher every `--shadow-correction-seconds` (2 s by default)
and immediately after numerical divergence. This makes the task explicitly a
state-transition problem rather than physical-parameter identification.

The interactive controls are W/S (throttle), E/D (brake), comma/period (gear),
C/V (clutch), Z/X (electrical load), G (starter signal), K (ignition toggle),
F (fuel), and brackets (engine profile). Starter and ignition are explicit transition inputs, not inferred
from resulting engine state. The keyboard-driven network path is timed
continuously: isolated compiled batch-1 latency and throughput, end-to-end
transition latency, 5 ms real-time factor, and deadline-miss rate remain visible
alongside teacher drift and correction count.

The 5 ms learned transition remains on the foreground real-time path. Physical
teacher stepping and both captured/live-replay Adam updates run through a
single background worker; completed weights are copied into the batch-1 rollout
artifact only between ticks. The window remains open after configured dataset
passes complete so live teacher replay can continue until the user exits.

```powershell
python -m src.common.tensors.abstract_nn.demo_engine_dendrite_live
```

The same runner has a non-windowed path. It executes the compiled learner,
sonifies a scripted closed-loop neural rollout, then writes a PNG, stereo WAV,
metrics JSON (including rollout runtime performance), weights plus normalization
metadata, and the native ABI contract:

```powershell
python -m src.common.tensors.abstract_nn.demo_engine_dendrite_live `
  --headless --output-dir .turing-cache/engine-dendrite-live
```

Use `--no-audio` only on a machine where neither playback nor a WAV is wanted.
The headless path never creates an OpenGL context, so it is suitable for CI and
remote hosts; its image is a software rendering of the same dot geometry and
metrics used by the OpenGL display.

### Startup caches

The live and headless runners cache both expensive preparation stages beneath
the selected `--output-dir`:

- `dataset-cache/` stores the captured real-simulator transition union as a
  pickle-free NPZ;
- `compiled/` and `compiled-rollout/` retain the native batch-training and
  batch-1 inference DLLs plus enough ABI metadata to reopen them directly.

The dataset key includes the selected engines, capture counts, episode layout,
seed, and all top-level `engine_toy` Python source timestamps. Native artifacts
also carry a fingerprint of the Turing tensor/compiler Python sources. A
matching rerun prints `[dataset cache hit]` and `[LLVM cache hit]`; configuration
or relevant source changes cause a cold rebuild. Use `--no-cache` to force both
capture and compilation. Artifacts created before this cache metadata was
added compile once and become reusable thereafter.

### Historical pre-transition-context measurements

A 3,000-step batch-16 run over the former 240-row engine dataset reduced sampled
minibatch loss from `1.07403` to `0.129688`. Held-out normalized MSE started at
`1.17674`, reached its best value of `0.311003` at step 1,000, and ended at
`0.322508`. The saved NPZ and final dendrite image therefore contain the
restored step-1,000 weights, not the more overfit step-3,000 state.

These figures predate the 416-to-212 complete machine-replacement contract and
must not be quoted as its performance. `src.compiler.benchmark_perforated_batches`
measures both isolated native inference and the compiled forward/VJP learner.
On the former 381-to-99 network,
two-branch network, batch 16 was the training-throughput knee at about 514
samples/s. Inference was already saturated at batch 8–16 (1,329–1,330
samples/s) and fell at batch 32. Compile times were about nine seconds for
batches 2 through 32 and 13.3 seconds for the singleton build.

Batch 1 originally exposed a reverse-ABI alias: whole-module storage analysis
published a coincident helper's `[381,198]` shape for the root `[1,99]`
base-bias gradient. Public statically shaped outputs now use their settled root
shape, just as public inputs already did. The exact large singleton contract
and native Adam step are covered by regression and batch 1 now measures about
1,058 inference samples/s and 168 learning samples/s.

```powershell
python -m src.compiler.benchmark_perforated_batches `
  --batches 1 2 4 8 16 32 --in-dim 381 --out-dim 99
```

`benchmark_engine_compiled_rate` compares equivalent 5 ms next-state motions
using the retained long-run weights. On the test host, `EngineCycleSim` ran at
21.48 transitions/s (46.56 ms/call), or 21.19/s with synchronous stereo audio.
Compiled batch-1 inference ran at 1,131.6 transitions/s (0.884 ms/call), 52.7x
the physics engine's transition rate. Batch 16 processed 1,339.3 samples/s,
62.4x the engine rate per sample, at 11.95 ms aggregate-call latency.

```powershell
python -m src.common.tensors.abstract_nn.benchmark_engine_compiled_rate `
  --weights .turing-cache/engine-dendrite-long-final/best-compiled-learner-weights.npz
```

The expanded version-2 `416 -> 212` graph-authority contract measured 2.659 ms
per compiled singleton call (376.1 transitions/s) and 35.037 ms per batch-16
call (456.7 transitions/s) on the same host. In that short run the physical
engine measured 25.24 transitions/s, making the compiled paths 14.9x and 18.1x
faster per sample respectively. These are short diagnostic timings rather than
a stabilized performance study, but batch-1 remains inside the 5 ms transition
budget despite the complete state output and authority gates.

## Test stand cooling and mechanical losses

The capture loader uses `engine_toy.engine_test_stand.EngineTestStand`, a
subclass of the real fixed-step engine solver. Missing liquid-cooling service
equipment belongs to this bench: a finite reservoir, electric circulation
pump and ambient heat rejector connect to an existing coolant circuit or,
through an isolated exchanger, an oil circuit. An engine with its own pump
and active heat exchanger receives no additional cooling equipment. A machine
without a liquid thermal circuit retains its existing air cooling. The engine
specification and drivetrain graph are not rewritten to install accessories.

Bench equipment is sized from the engine power and circuit heat share using
explicit design temperature differences. Its configuration, reservoir
temperature, flow and heat readings appear under `state.test_stand_cooling`
in the training ABI, including live shadow captures and rollout seeds.
Previously saved networks do not include these added inputs/outputs and
must be retrained for this ABI.

Neutral is a disconnected shaft, so episode setup preserves load speed when
the gear ratio is zero. It must not synchronize by dividing crank speed by
an epsilon. Automatic-transmission oxidation retains its temperature law but
compares damage to remaining fluid life in log space before exponentiating.

The drivetrain now solves declared bearing, clutch and rolling-contact
friction as inertia-limited impulses. Each contact records dissipated energy;
one unambiguous connected thermal-liquid circuit (or an explicit
`heat_sink_node`) receives it. Otherwise `unrejected_heat_j` retains the energy
on the contact ledger. This ledger is not a calibrated part-temperature or
wear model. Crank reaction is averaged across substeps so early friction work
is not lost. Bearing coefficients absent from the authored graph remain an
explicit modeling gap; the change does not invent a friction rating from a
bearing radius alone or replace the engine's existing aggregate FMEP model.
