# AbstractNN named state and real signal-graph experiment

Owner directed expansion of AbstractNN using AbstractTensor throughout, toward
lowering the existing Spectral signal graph. This is not a replacement solver.

Added `abstract_nn.module`: Module, Parameter, ModuleDict and ParameterDict.
Parameter is explicitly a factory returning the same AbstractTensor backend,
with independent payload and registration on the existing gradient tape. It is
not a new numeric backend or a claim of full torch.nn compatibility. Named
parameters/buffers/children, recursive training mode, state collection and
validated loading are provided. Existing Linear/Model classes remain unchanged.
Public `AbstractTensor.broadcast_shapes` reuses the existing batched-matmul
shape resolver rather than adding another broadcast implementation.

Verified CPU tests cover registry mutation, sharing, cycles, persistence,
state-load validation, backend identity, gradient propagation, complex arithmetic
and broadcasting. No shader links, GPU renders, installs or deployment occurred.

## Actual GraphSolver experiment

Original Spectral source at aca51a4 is preserved in untracked local scratch.
The experiment ports imports to AbstractTensor/AbstractNN, uses the existing
backend-neutral dtype strings (complex128, float64, etc.) and CPU device string.
The physical Optional[List[str]] spelling experiment is separately retained.
No equations, SCC scheduling or graph solver implementation were replaced.

With torch unavailable (`sys.modules['torch'] = None`), actual GraphSolver
construction and run_schedule succeeded for two-frame native complex128 data:

- Chain radio -> rack (0.5) -> light (0.5): input [0.4, 0.8], output [0.1, 0.2].
- Same chain plus light -> rack (0.2): output [1/9, 2/9].
- Assertions check error below 1e-12 and NumPyTensorOperations output identity.

This proves these real zero-delay graph paths, not all source modules, training,
nonlinear SCCs, delayed scheduling, deployment, or LLVM lowering. ComplexPrecision
is a separate existing precision-expansion type; the original graph explicitly
uses native complex128, and that numerical choice was preserved here.

## Compiler limit

The sanctioned source-divider LLVM attempt for network_materializer.compile_nodes
still failed in SSA lowering with KeyError 43 after 14.86 seconds. That compiler
checkout had concurrent dirty edits; no attribution to a clean release is made.
No LLVM product was emitted. The experiment and JSON receipt remain untracked
local developer scratch, not claimed to be SQL registered. Supported remote
filing was unavailable in this session. No live service state was edited.

Follow-up against the isolated clean compiler plus annotation ABI repair
9eb866d4 reproduced SSA KeyError 44 in 12.51 seconds. Repro and receipt were
sent to the compiler worker for investigation; this is a distinct run from the
dirty-checkout KeyError 43 above. Focused tests finish at 11 passed (0.55 s).

Remaining work: port and test the complete materializer path and delayed/nonlinear
graph paths, complete compiler lowering, and integrate an actual service worker
boundary before enabling radio routing on the Torch-free host.
