# 2026-10-29 Handler entry authority and next codec compile error

The reducer now selects a handler's entry binding only when every candidate
protected Call/raise site has the same exact value for that name. It records
the source sites and selected bindings before canonical relabel; the relabel
maps all identities or refuses. `SOURCE_EXCEPTION_HANDLER_ENTRY` posts the
handler owner, incoming `SOURCE_EXCEPTION_TRANSFER` cells, and unanimous
canonical value cells. The audit checks each selected value against every
incoming source state and the page's provenance. Native SSA dispatch also
requires the caught transfer to appear in the handler-entry row.

Measured on the codec graph: handler 55 of Try 56 has two incoming transfers,
calls 65 (`float(value)`) and 80 (`int(value)`). It selects `index=2` and
`value=4` among its common bindings; it does not select `numeric`, which
exists only at call 80. The direct exception-transfer/handler-entry audit
reported zero findings. A separate small source with a handler reading `x`
after one protected call before and another after `x` changes refused in the
reducer with `handler entry requires an exceptional value Phi`, naming both
incoming values. Such a Phi remains unimplemented.

The loop composer now retains the codec's authored Try with protected calls
65 and 80 in its body and one handler. Native SSA explicitly refuses each
protected call until it has a failure edge, so this path cannot silently
compile without exception dispatch. Public codec lowering currently stops
earlier: `region value 4 (input 'value' in function
'tensor_octets_to_bytes') has no settled dtype`. The ProgramABI declares
`octets` as an int64 rank-1 span; node 78 is its `tolist()` result, node 4
is the `enumerate(values)` loop target, and node 4 has no tensor/descriptor
dtype. The `tolist` node has only its tensor operand edge and no recorded
list element type. No dtype was guessed into the loop target.

Validation discovery had also refused guard 49 because its Raise node was
removed. The exact `SOURCE_CONTROL_SPECIALIZATION` fact for guard 49 selects
`orelse` and rejects its raising `body`; discovery now skips that proven
rejected arm. After this fix, public lowering reached the value-4 dtype
refusal above. The real native codec artifact and conversion failure edges
remain unfinished.

Verification: the native C corpus cases `c1166_try_raise_handler_loop_state`
and `c1167_try_raise_superclass_handler` both passed (2 passed, 651
deselected). All nine edited Python modules compiled with `py_compile`, and
`git diff --check` passed. The broad `view` concordance audit still exits 1
at `selected return has no exact source call reachability authority`, the
same finding recorded before these edits; the focused exception page audit
above had zero findings.
