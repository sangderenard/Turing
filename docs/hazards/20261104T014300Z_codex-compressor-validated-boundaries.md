2026-11-04 Frozen tensor capacity and owned storage integration (MEASURED):

Validation checkout: C:/Users/alber/.codex/worktrees/compressor-validation/turing,
on source fixes equivalent to canonical 2389d289. Seven production files and
their focused tests/receipts were copied from the frozen repair checkout.
Canonical remains unchanged during full compressor RUN37.

Dynamic topk allocations consume their exact guarded capacity bound. The C
public wrapper exposes native shape/count through ordinary output ports;
runtime ABI v2 and its audit correlate capacity, logical count/shape, and the
result binding. DLL tests cover values and indices, rank one/two, k=0..3,
repeated executions, and exact int64 values greater than 2**53.

Computed arange/cast tensors keep logical element identities separate from
native pointer carriers. Their existing owned heap Allocas use exact SSA
counts and caller-owned private owner cells. The concordance records data,
count, owner, call binding, downstream nonescape decisions, and release.
Aggregate analysis exposes one physical carrier per logical result. C publishes
the arena pointer into its pointer cell, loads that pointer at the numeric span
call boundary, and uses the exact live count formal for both allocation and
kernel arguments. C emission units cite the binding/SSA causes. Runtime-sized
tensor storage without an allocation contract is refused instead of sized as
one scalar. Dynamic state follows common aggregate ABI projections in the
backend-neutral storage requirements.

Frozen combined gate:
`python -u -m pytest tests/test_dynamic_arange_metadata.py tests/test_topk_public_native.py tests/test_tensor_metadata_call_position.py tests/test_tensor_runtime_abi_audit.py tests/test_tensor_owned_heap_binding_audit.py tests/test_owned_tensor_c_publication.py tests/test_public_sequence_input_ports.py tests/test_public_owned_sequence_return.py -q --tb=short`
passed 22/22 in 104.93 seconds. This includes 24 repeated private tensor native
executions plus int32 overflow refusal, forged identity/capacity/alias/owner
negatives, existing bytes input ports, owned concat DLL/executable round trips,
and real compressor packet tails assembled at the final native concat stage.
The earlier root scratch also compared 24 calls directly with actual Python
arange/cast/sum results (0,1,7,16,127,-3,7,0; three calls each), all equal.
Those are bounded stages; the encoder stages of the packet-tail test remain
the Python reference. Full native compressor parity is still unverified.

Mapping audit after the combined gate: 28 rows, zero findings/unsourced,
CLOSED. Whitespace check clean. A subsequent review identified that release
auditing must enumerate every private caller owner rather than only existing
release rows. The writer already refuses unsupported lifetimes. That audit
coverage now requires an exact release receipt for each private owner binding.
The regression removes both the receipt and the live release Call and confirms
audit refusal. Copied audit/test follow-up plus the retained native scalar
regression: 3 passed in 14.30s. No runtime lifetime policy changed.

Whole compressor RUN37 on canonical 2389d289 passed the former unknown-copy
ABI withdrawal failure, then stopped after 430.165s at `_header` sequence15's
append/fixed helper metadata transport, proposing ptr->int64 with unchanged
causes. Its actual failing SSA work is retained in canonical
build/compressor_native_37_failed_ssa_1.pkl (129228461 bytes) and _2.pkl
(128921821 bytes). This full run predates the tensor ownership integration.
That exact writer is the next investigation; the compiler goal remains active.
